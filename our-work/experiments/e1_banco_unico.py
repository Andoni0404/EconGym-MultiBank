"""E1 (nivel 1): un solo banco. ¿Aprende el RL algo mejor que las reglas fijas?

Entrena PPO o SAC (Stable-Baselines3) sobre BancoUnicoEnv y lo compara con las reglas
(prudente, clasica, agresiva) en las MISMAS semillas de evaluación.

Con 1 banco no hay competencia (es un monopolio): el objetivo es comprobar que el entorno
se puede aprender y fijar hiperparámetros de partida antes de pasar a N bancos.

Uso (desde our-work/):
    python -m experiments.e1_banco_unico --pasos 50000 --hogares 100          # prueba rápida
    python -m experiments.e1_banco_unico --algo ppo --pasos 300000 --entornos 8  # por la noche, 1000 hogares
    python -m experiments.e1_banco_unico --algo sac --pasos 100000
    python -m experiments.e1_banco_unico --solo-evaluar results/e1/ppo_h100_s1    # reevaluar un modelo guardado

Salida:
    results/e1/<algo>_h<hogares>_s<semilla>/   modelo.zip + vecnormalize.pkl
                                               fotos/: el banco al 0, 25, 50 y 75% del entrenamiento
                                               (para verlo aprender en el visor: python -m multibank.visual --fotos)
    results/e1/tb/                             TensorBoard:  tensorboard --logdir results/e1/tb
    results/e1/evaluacion.csv                  una fila por (política, semilla de evaluación)
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from multibank.env import BancoUnicoEnv
from multibank.reglas import REGLAS
from settings.config import Config

AQUI = Path(__file__).resolve().parent
RESULTADOS = AQUI.parent / "results" / "e1"
SEMILLAS_EVAL = [1001, 1002, 1003, 1004, 1005]   # distintas de las de entrenamiento y de la calibración
REGLAS_E1 = ["prudente", "clasica", "agresiva"]


EMPRESAS_MIN = 25   # con pocas empresas, que impague UNA es ~20% de la cartera: las quiebras serían pura suerte


def config(hogares, semilla):
    return Config(n_hogares=hogares, n_empresas=max(hogares // 20, EMPRESAS_MIN), n_bancos=1, seed=semilla)


def crear_entorno(hogares, semilla):
    def _f():
        # El Runner del original crea una carpeta runN nueva al arrancar (agents/log_path.py). Con
        # varios procesos a la vez, dos pueden intentar crear la misma: se reintenta tras una espera.
        for intento in range(20):
            try:
                return Monitor(BancoUnicoEnv(config(hogares, semilla)))
            except FileExistsError:
                time.sleep(0.2 + 0.3 * np.random.default_rng(semilla + intento).random())
        raise RuntimeError("No se pudo crear el entorno: choque repetido al crear la carpeta runN del original")
    return _f


# ---------------------------------------------------------------------------- evaluación
def jugar(env, politica, semilla):
    """Un episodio con una política obs -> acción. Devuelve métricas con la recompensa ORIGINAL."""
    obs, _ = env.reset(seed=semilla)
    total, impagos, anos = 0.0, [], 0
    quiebra = False
    while True:
        obs, r, terminado, truncado, info = env.step(politica(obs))
        total += r
        anos += 1
        impagos.append(float(np.mean(info["impaga"])))
        if terminado:
            quiebra = True
        if terminado or truncado:
            break
    s = env.multi.sistema
    return dict(
        recompensa_total=total,
        capital_final=float(s.K[0] / s.K0[0]),        # 1 = acaba con el capital inicial
        quiebra=quiebra,
        anos=anos,
        impago_medio=float(np.mean(impagos)),          # fracción media de clientes que impagan al año
    )


def politica_rl(modelo, vecnorm):
    def f(obs):
        o = vecnorm.normalize_obs(obs[None, :]) if vecnorm is not None else obs[None, :]
        accion, _ = modelo.predict(o, deterministic=True)
        return accion[0]
    return f


def evaluar(hogares, politicas):
    env = BancoUnicoEnv(config(hogares, SEMILLAS_EVAL[0]))
    filas = []
    for nombre, fabrica in politicas.items():
        pol = fabrica(env.multi)
        for s in SEMILLAS_EVAL:
            filas.append(dict(politica=nombre, semilla=s, **jugar(env, pol, s)))
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------- entrenamiento
FOTOS = (0.0, 0.25, 0.5, 0.75)        # momentos del entrenamiento en que se guarda el banco (el final aparte)


class GuardarFotos(BaseCallback):
    """Guarda el banco (y su normalización) en fotos/ al 0, 25, 50 y 75% del entrenamiento."""

    def __init__(self, carpeta, pasos_totales):
        super().__init__()
        self.carpeta = Path(carpeta) / "fotos"
        self.pendientes = [int(f * pasos_totales) for f in FOTOS]

    def _guardar(self):
        self.carpeta.mkdir(parents=True, exist_ok=True)
        n = self.num_timesteps
        self.model.save(self.carpeta / f"paso_{n}")
        self.model.get_vec_normalize_env().save(str(self.carpeta / f"vecnormalize_{n}.pkl"))

    def _on_training_start(self):
        self._guardar()
        self.pendientes.pop(0)

    def _on_step(self):
        if self.pendientes and self.num_timesteps >= self.pendientes[0]:
            self._guardar()
            self.pendientes.pop(0)
        return True


def entrenar(a):
    nombre = f"{a.algo}_h{a.hogares}_s{a.semilla}"
    carpeta = RESULTADOS / nombre
    carpeta.mkdir(parents=True, exist_ok=True)

    fabricas = [crear_entorno(a.hogares, a.semilla * 100 + i) for i in range(a.entornos)]
    venv = SubprocVecEnv(fabricas) if a.entornos > 1 else DummyVecEnv(fabricas)
    # Las magnitudes del banco tienen escalas muy distintas: se normalizan obs y recompensa
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=a.gamma)

    comunes = dict(gamma=a.gamma, verbose=1, seed=a.semilla, tensorboard_log=str(RESULTADOS / "tb"))
    if a.algo == "ppo":
        # Episodios de 30 años: n_steps múltiplo de 30 para que cada lote tenga episodios completos
        modelo = PPO("MlpPolicy", venv, n_steps=a.n_steps, batch_size=a.batch, learning_rate=a.lr,
                     ent_coef=a.ent_coef, gae_lambda=0.95, **comunes)
    else:
        modelo = SAC("MlpPolicy", venv, learning_rate=a.lr, batch_size=a.batch, learning_starts=1000, **comunes)

    t = time.time()
    modelo.learn(total_timesteps=a.pasos, tb_log_name=nombre, progress_bar=True,
                 callback=GuardarFotos(carpeta, a.pasos))
    print(f"\nEntrenado en {(time.time() - t) / 60:.1f} min")

    modelo.save(carpeta / "modelo")
    venv.save(str(carpeta / "vecnormalize.pkl"))
    venv.close()
    return carpeta


def cargar(carpeta):
    carpeta = Path(carpeta)
    algo = PPO if carpeta.name.startswith("ppo") else SAC
    modelo = algo.load(carpeta / "modelo")
    # Las estadísticas de normalización se cargan sobre un entorno cualquiera del mismo tamaño
    hogares = int(carpeta.name.split("_h")[1].split("_")[0])
    vecnorm = VecNormalize.load(str(carpeta / "vecnormalize.pkl"), DummyVecEnv([crear_entorno(hogares, 0)]))
    vecnorm.training = False
    return modelo, vecnorm, hogares


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p.add_argument("--pasos", type=int, default=50_000)
    p.add_argument("--hogares", type=int, default=100, help="100 = rápido; 1000 = economía completa")
    p.add_argument("--entornos", type=int, default=4, help="economías en paralelo (procesos)")
    p.add_argument("--semilla", type=int, default=1)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=600, help="pasos por entorno y actualización (PPO)")
    p.add_argument("--batch", type=int, default=120)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--solo-evaluar", type=str, default=None, help="carpeta de un modelo ya entrenado")
    a = p.parse_args()

    if a.solo_evaluar:
        carpeta = Path(a.solo_evaluar)
        modelo, vecnorm, hogares = cargar(carpeta)
    else:
        carpeta = entrenar(a)
        modelo, vecnorm, hogares = cargar(carpeta)

    politicas = {carpeta.name: lambda env: politica_rl(modelo, vecnorm)}
    for r in REGLAS_E1:
        politicas[r] = (lambda nombre: (lambda env: REGLAS[nombre](env)))(r)

    df = evaluar(hogares, politicas)
    salida = RESULTADOS / "evaluacion.csv"
    salida.parent.mkdir(parents=True, exist_ok=True)
    if salida.exists():
        df = pd.concat([pd.read_csv(salida), df]).drop_duplicates(["politica", "semilla"], keep="last")
    df.to_csv(salida, index=False)

    resumen = (df[df.politica.isin(politicas)]
               .groupby("politica")[["recompensa_total", "capital_final", "quiebra", "impago_medio"]]
               .agg(["mean", "std"]).round(3))
    print(f"\nEvaluación en {len(SEMILLAS_EVAL)} semillas ({hogares} hogares):\n")
    print(resumen.to_string())
    print(f"\nGuardado en {salida}")


if __name__ == "__main__":
    main()
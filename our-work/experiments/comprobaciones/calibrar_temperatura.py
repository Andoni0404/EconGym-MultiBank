"""Calibra `temperatura_softmax` con Optuna, usando solo bancos con reglas fijas (sin RL).

La temperatura decide cuánto se mueven los clientes hacia el banco más barato:
    - muy baja  -> el más barato se lo lleva casi todo (guerras de precios bruscas)
    - muy alta  -> a los clientes les da igual el precio (no hay competencia)

Cada trial mide dos cosas, con varias semillas:

    1. Sensibilidad al precio. Dos bancos con la misma regla fija, pero el banco 1 cobra
       `--diferencia` (por defecto 1 punto) más de margen que el banco 0. Se mide la cuota
       media de préstamos del banco caro (empresas y hogares) a partir del año `--calentamiento`.
       Objetivo: `--cuota-objetivo` (por defecto 0,35: quien cobra 1 punto más conserva un 35%).
       Este escenario se juega SIN impagos (pmax = 0): con impagos, los clientes del banco caro
       impagan más, pasan a la lista de morosos de ese banco y se van al barato, y eso movería la
       cuota aunque la temperatura fuera infinita. Así se mide solo el efecto del precio.

    2. Estabilidad. Dos bancos con la regla "competitiva" (bajan el margen si ganan y lo suben
       si pierden), uno más rápido que el otro. Se mide cuánto cambia la cuota de un año a otro.
       Si supera `--volatilidad-max` (por defecto 5 puntos al año), se penaliza.

    error = (cuota_cara - cuota_objetivo)^2 + peso * max(0, volatilidad - volatilidad_max)^2

Las dos referencias (0,35 y 5 puntos) son SUPUESTOS DE MODELADO: se pueden cambiar por
línea de comandos y hay que justificarlos en el informe.

Uso (desde our-work/):
    python -m experiments.comprobaciones.calibrar_temperatura                    # 40 trials, 3 semillas
    python -m experiments.comprobaciones.calibrar_temperatura --trials 60 --hogares 1000
    python -m experiments.comprobaciones.calibrar_temperatura --cuota-objetivo 0.4

Salida: results/calibracion_temperatura.csv (todos los trials) y el mejor valor por pantalla.
"""

import argparse
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import optuna

from multibank.env import MultiBankEnv
from multibank.reglas import competitiva
from settings.config import Config

warnings.filterwarnings("ignore")

AQUI = Path(__file__).resolve().parent
SALIDA = AQUI.parents[1] / "results" / "calibracion_temperatura.csv"

# Regla fija de referencia para el escenario asimétrico (como "clasica" pero sin reaccionar a pérdidas)
MARGEN_BASE = 0.03
DEPOSITO_BAJO_BASE = 0.005
PRIMA = 0.01
UMBRAL = 2.0


def jugar(env, politicas, semilla):
    """Juega un episodio y devuelve la serie de cuotas de préstamo del banco 1 (empresas, hogares)."""
    obs, _ = env.reset(seed=semilla)
    cuotas = []
    while True:
        acciones = np.stack([pol(obs[b]) for b, pol in enumerate(politicas)])
        obs, _, _, truncado, _ = env.step(acciones)
        ce, ch = env.sistema.cuota_prestamos()
        cuotas.append((ce[1], ch[1]))
        if truncado:
            return np.array(cuotas)


def sensibilidad(env, diferencia, semillas, calentamiento):
    """Cuota media del banco caro (cobra `diferencia` más) a partir del año de calentamiento."""
    barato = env.accion_desde(DEPOSITO_BAJO_BASE, MARGEN_BASE, MARGEN_BASE, PRIMA, UMBRAL)
    caro = env.accion_desde(DEPOSITO_BAJO_BASE, MARGEN_BASE + diferencia, MARGEN_BASE + diferencia, PRIMA, UMBRAL)
    medias = []
    for s in semillas:
        cuotas = jugar(env, [lambda o: barato, lambda o: caro], s)
        medias.append(cuotas[calentamiento:].mean())        # media de empresas y hogares
    return float(np.mean(medias))


def volatilidad(env, semillas, calentamiento):
    """Cambio medio de la cuota de un año a otro (en tanto por uno) con dos bancos 'competitiva'."""
    vols = []
    for s in semillas:
        politicas = [competitiva(env, paso=0.002), competitiva(env, paso=0.004)]
        cuotas = jugar(env, politicas, s)[calentamiento:]
        vols.append(np.abs(np.diff(cuotas, axis=0)).mean())
    return float(np.mean(vols))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--semillas", type=int, default=3, help="semillas por trial")
    p.add_argument("--hogares", type=int, default=100, help="100 = rápido; 1000 = economía completa")
    p.add_argument("--empresas", type=int, default=None, help="por defecto, hogares / 20")
    p.add_argument("--t-min", type=float, default=1e-3)
    p.add_argument("--t-max", type=float, default=1e-1)
    p.add_argument("--diferencia", type=float, default=0.01, help="margen extra del banco caro (0.01 = 1 punto)")
    p.add_argument("--cuota-objetivo", type=float, default=0.35, help="cuota que debería conservar el banco caro")
    p.add_argument("--volatilidad-max", type=float, default=0.05, help="cambio de cuota anual tolerable")
    p.add_argument("--peso", type=float, default=1.0, help="peso de la penalización por volatilidad")
    p.add_argument("--calentamiento", type=int, default=5, help="años iniciales que no se miden")
    p.add_argument("--seed", type=int, default=0, help="semilla de Optuna")
    a = p.parse_args()

    empresas = a.empresas or max(a.hogares // 20, 2)
    # Semillas de calibración distintas de las de los experimentos (1-5), para no sobreajustar
    semillas = list(range(100, 100 + a.semillas))

    def objetivo(trial):
        T = trial.suggest_float("temperatura_softmax", a.t_min, a.t_max, log=True)
        cfg = Config(n_hogares=a.hogares, n_empresas=empresas, n_bancos=2, temperatura_softmax=T)
        cfg_sin_impagos = replace(cfg, pmax_empresas=0.0, pmax_hogares=0.0)
        cuota_cara = sensibilidad(MultiBankEnv(cfg_sin_impagos), a.diferencia, semillas, a.calentamiento)
        vol = volatilidad(MultiBankEnv(cfg), semillas, a.calentamiento)
        error = (cuota_cara - a.cuota_objetivo) ** 2 + a.peso * max(0.0, vol - a.volatilidad_max) ** 2
        trial.set_user_attr("cuota_cara", cuota_cara)
        trial.set_user_attr("volatilidad", vol)
        return error

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=a.seed))
    estudio.optimize(objetivo, n_trials=a.trials, show_progress_bar=True)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    df = estudio.trials_dataframe(attrs=("number", "params", "user_attrs", "value"))
    df.sort_values("params_temperatura_softmax").to_csv(SALIDA, index=False)

    b = estudio.best_trial
    T = b.params["temperatura_softmax"]
    if T > 0.95 * a.t_max or T < 1.05 * a.t_min:
        print("\nAVISO: el mejor valor está en el borde del rango buscado; amplía --t-min / --t-max.")
    print(f"\nMejor temperatura_softmax = {b.params['temperatura_softmax']:.4f}")
    print(f"  cuota del banco caro    = {b.user_attrs['cuota_cara']:.3f}  (objetivo {a.cuota_objetivo})")
    print(f"  volatilidad anual       = {b.user_attrs['volatilidad']:.3f}  (máximo {a.volatilidad_max})")
    print(f"  error                   = {b.value:.5f}")
    print(f"Todos los trials en {SALIDA}")


if __name__ == "__main__":
    main()

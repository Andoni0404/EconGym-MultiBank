"""E2 (núcleo): N bancos que aprenden a la vez con IPPO o MAPPO, y evaluación justa.

Entrenamiento: cada banco con su propia red PPO, en varias economías en paralelo.
    --algo ippo   crítico de cada banco con SU observación      (algoritmos/ippo.py)
    --algo mappo  crítico de cada banco con la de TODOS los bancos (algoritmos/mappo.py)

Evaluación (en las MISMAS semillas para todo, distintas de las de entrenamiento):
    1. IPPO contra IPPO: los bancos entrenados compiten entre sí.
    2. IPPO contra cada regla, CON INTERCAMBIO DE POSICIONES: en cada economía el banco IPPO juega
       una vez en cada posición (como blancas y negras), y el resto de bancos son la regla. Así la
       suerte del reparto inicial de clientes no favorece a nadie.
    3. Referencia: cada regla contra sí misma.
    4. Con --contra <carpeta de otro modelo>: cara a cara contra ese modelo (p. ej. MAPPO contra IPPO),
       también intercambiando posiciones.
Además de las reglas de Igor se evalúa "accion_media": la acción 0 (valores intermedios de todo),
que es lo que hace una red sin entrenar. Sirve para ver cuánto aporta de verdad el aprendizaje.
Las reglas se crean de nuevo en cada episodio (la regla 'competitiva' guarda estado).

Uso (desde our-work/):
    python -m experiments.e2_ippo --pasos 100000 --hogares 100              # prueba
    python -m experiments.e2_ippo --pasos 500000 --entornos 8 --semilla 1   # de verdad (por la noche)
    python -m experiments.e2_ippo --modo ingenuo                            # E3: sin done por banco
    python -m experiments.e2_ippo --algo mappo --pasos 100000               # MAPPO
    python -m experiments.e2_ippo --solo-evaluar results/e2/mappo_n2_h100_p100k_correcto_s1 --contra results/e2/ippo_n2_h100_p100k_correcto_s1
    python -m experiments.e2_ippo --solo-evaluar results/e2/ippo_n2_h100_p100k_correcto_s1

Salida:
    results/e2/<nombre>/modelo.pt            redes y normalización
    results/e2/tb/<nombre>/                  TensorBoard
    results/e2/evaluacion.csv                una fila por (escenario, semilla, posición, banco)
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from torch.utils.tensorboard import SummaryWriter

from algoritmos.entornos_paralelos import EntornosParalelos, _crear
from algoritmos.ippo import IPPO, HiperIPPO
from algoritmos.mappo import MAPPO
from multibank.env import N_ACCIONES
from multibank.reglas import REGLAS
from settings.config import Config

AQUI = Path(__file__).resolve().parent
RESULTADOS = AQUI.parent / "results" / "e2"
SEMILLAS_EVAL = list(range(3001, 3011))      # 10 economías de evaluación
REGLAS_EVAL = ["prudente", "clasica", "agresiva", "competitiva", "accion_media"]
REGLAS = {**REGLAS, "accion_media": lambda env: (lambda obs: np.zeros(N_ACCIONES, np.float32))}
EMPRESAS_MIN = 25


def config(a):
    return Config(n_hogares=a.hogares, n_empresas=max(a.hogares // 20, EMPRESAS_MIN), n_bancos=a.bancos,
                  lam_quiebra=a.lam_quiebra)


def nombre(a):
    k = f"{a.pasos // 1000}k" if a.pasos >= 1000 else str(a.pasos)
    return f"{a.algo}_n{a.bancos}_h{a.hogares}_p{k}_{a.modo}_s{a.semilla}"


# ---------------------------------------------------------------------------- entrenamiento
def entrenar(a):
    carpeta = RESULTADOS / nombre(a)
    carpeta.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(str(RESULTADOS / "tb" / nombre(a)))

    envs = EntornosParalelos(config(a), a.entornos, a.semilla)
    E, N, n_obs = a.entornos, envs.N, envs.n_obs
    h = HiperIPPO(lr=a.lr, gamma=a.gamma, epochs=a.epochs, ent_coef=a.ent_coef, modo=a.modo)
    ippo = (MAPPO if a.algo == "mappo" else IPPO)(N, n_obs, N_ACCIONES, h, semilla=a.semilla)

    T = a.n_steps
    obs = envs.reset()
    vivo = np.ones((E, N), bool)
    nuevo_ep = np.ones(E, bool)
    ret_ep = np.zeros((E, N))
    episodios = []                     # (retorno por banco, quiebras por banco, margen medio)
    margenes_ep = [[] for _ in range(E)]
    pasos, it, t0 = 0, 0, time.time()

    while pasos < a.pasos:
        lote = {k: [] for k in ("obs", "accion", "logp", "valor", "recompensa", "terminado",
                                "fin_episodio", "vivo", "valor_final")}
        for _ in range(T):
            ippo.norm_obs.actualizar(obs.reshape(-1, n_obs)[vivo.reshape(-1)] if vivo.any() else obs.reshape(-1, n_obs))
            acciones, sal = ippo.actuar(obs)
            obs2, r, term, trunc, finales, infos = envs.step(acciones)

            v_final = np.zeros((E, N), np.float32)
            for e in range(E):
                if trunc[e]:
                    v_final[e] = ippo.valor(finales[e][None])[0]
            ret_ep += r
            for e in range(E):
                margenes_ep[e].append(infos[e]["margen_medio"])
            r_esc = ippo.escalar_recompensa(r.astype(np.float32), nuevo_ep)

            lote["obs"].append(obs); lote["accion"].append(sal["accion"]); lote["logp"].append(sal["logp"])
            lote["valor"].append(sal["valor"]); lote["recompensa"].append(r_esc)
            lote["terminado"].append(term & vivo); lote["fin_episodio"].append(trunc)
            lote["vivo"].append(vivo.copy()); lote["valor_final"].append(v_final)

            for e in range(E):
                if trunc[e]:
                    episodios.append((ret_ep[e].copy(), (~vivo[e] | term[e]).astype(float), np.mean(margenes_ep[e])))
                    ret_ep[e] = 0
                    margenes_ep[e] = []
            vivo = np.where(trunc[:, None], True, vivo & ~term)
            nuevo_ep = trunc.copy()
            obs = obs2
            pasos += E

        lote = {k: np.stack(v) for k, v in lote.items()}
        lote["valor_siguiente"] = ippo.valor(obs)
        stats = ippo.actualizar(lote)
        it += 1

        # --- registro
        recientes = episodios[-50:]
        if recientes:
            rets = np.stack([x[0] for x in recientes]); quieb = np.stack([x[1] for x in recientes])
            for b in range(N):
                tb.add_scalar(f"banco_{b}/retorno_episodio", rets[:, b].mean(), pasos)
                tb.add_scalar(f"banco_{b}/tasa_quiebra", quieb[:, b].mean(), pasos)
            tb.add_scalar("mercado/margen_medio_empresas", np.mean([x[2] for x in recientes]), pasos)
        for s in stats:
            for k in ("perdida_pi", "perdida_v", "explained_variance", "approx_kl", "std"):
                tb.add_scalar(f"banco_{s['banco']}/{k}", s[k], pasos)
        if it % 5 == 0 or pasos >= a.pasos:
            txt = " | ".join(f"b{b}: ret {rets[:, b].mean():5.2f} quiebra {quieb[:, b].mean():.2f}"
                             for b in range(N)) if recientes else "sin episodios aún"
            ev = np.mean([s["explained_variance"] for s in stats]) if stats else float("nan")
            print(f"pasos {pasos:8d} ({pasos / (time.time() - t0):4.0f}/s)  {txt}  "
                  f"margen {100 * np.mean([x[2] for x in recientes]) if recientes else 0:.2f} pts  ev {ev:.2f}", flush=True)

    envs.cerrar()
    ippo.guardar(carpeta / "modelo.pt")
    tb.close()
    print(f"\nEntrenado en {(time.time() - t0) / 60:.1f} min -> {carpeta}")
    return carpeta


# ---------------------------------------------------------------------------- evaluación
def jugar(env, politicas, semilla):
    """politicas: lista (una por banco) de funciones obs_banco -> acción. Devuelve métricas por banco."""
    obs, _ = env.reset(seed=semilla)
    margen, cuota, credito = [], [], []
    while True:
        acc = np.stack([p(obs[b]) for b, p in enumerate(politicas)])
        obs, r, term, trunc, info = env.step(acc)
        s = env.sistema
        margen.append(env.tipos.empresas - info["base"])
        cuota.append(np.mean(s.cuota_prestamos(), axis=0))
        credito.append(info["credito_empresas"])
        if trunc:
            break
    s = env.sistema
    margen, cuota = np.array(margen)[5:], np.array(cuota)[5:]
    return [dict(capital_final=float(s.K[b] / s.K0[b]), quiebra=bool(s.quebrado[b]),
                 margen_empresas=float(margen[:, b].mean()), cuota_prestamos=float(cuota[:, b].mean()))
            for b in range(env.N)], float(np.mean(credito))


def evaluar(ippo, nombre_modelo, a, rival=None, nombre_rival=None):
    env = _crear(config(a))
    N = env.N
    filas = []
    alg = nombre_modelo.split("_")[0]                      # "ippo" o "mappo"

    def red(k, modelo=ippo):
        return lambda o: modelo.actuar(o[None, None], determinista=True, banco=k)[0][0, 0]

    def anotar(escenario, semilla, posicion, politicas_nombres, res, credito):
        for b, (m, nom) in enumerate(zip(res, politicas_nombres)):
            filas.append(dict(modelo=nombre_modelo, escenario=escenario, semilla=semilla, posicion_rl=posicion,
                              banco=b, politica=nom, credito_empresas=credito, **m))

    for s in SEMILLAS_EVAL:
        # 1. IPPO contra IPPO
        res, cred = jugar(env, [red(b) for b in range(N)], s)
        anotar(f"{alg}_vs_{alg}", s, -1, [f"{alg}_{b}" for b in range(N)], res, cred)
        if rival is not None:
            alg_r = nombre_rival.split("_")[0]
            for p in range(N):                               # cara a cara, en cada posición
                pols = [red(b % rival.N, rival) for b in range(N)]
                pols[p] = red(0)
                res, cred = jugar(env, pols, s)
                anotar(f"{alg}_vs_{alg_r}_rival", s, p,
                       [f"{alg_r}_rival"] * p + [f"{alg}_0"] + [f"{alg_r}_rival"] * (N - p - 1), res, cred)
        for regla in REGLAS_EVAL:
            # 2. IPPO (red 0) contra la regla, en cada posición
            for p in range(N):
                pols = [REGLAS[regla](env) for _ in range(N)]          # reglas nuevas en cada episodio
                pols[p] = red(0)
                res, cred = jugar(env, pols, s)
                anotar(f"{alg}_vs_{regla}", s, p, [regla] * p + [f"{alg}_0"] + [regla] * (N - p - 1), res, cred)
            # 3. la regla contra sí misma
            res, cred = jugar(env, [REGLAS[regla](env) for _ in range(N)], s)
            anotar(f"{regla}_vs_{regla}", s, -1, [regla] * N, res, cred)
    return pd.DataFrame(filas)


def resumen(df):
    modelo = df.modelo.iloc[0].split("_")[0]
    df = df.assign(es_rl=df.politica.str.startswith(modelo + "_") & ~df.politica.str.endswith("_rival"))
    t = (df.groupby(["escenario", "es_rl"])[["capital_final", "quiebra", "margen_empresas", "cuota_prestamos", "credito_empresas"]]
         .mean().round(3))
    t["margen_empresas"] = (100 * t["margen_empresas"]).round(2)
    return t.rename(columns={"margen_empresas": "margen_pts"})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=["ippo", "mappo"], default="ippo")
    p.add_argument("--bancos", type=int, default=2)
    p.add_argument("--pasos", type=int, default=100_000, help="pasos de entorno en total (suma de todas las economías)")
    p.add_argument("--hogares", type=int, default=100)
    p.add_argument("--entornos", type=int, default=8)
    p.add_argument("--semilla", type=int, default=1)
    p.add_argument("--modo", choices=["correcto", "ingenuo"], default="correcto", help="tratamiento de la quiebra (E3)")
    p.add_argument("--lam-quiebra", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--n-steps", type=int, default=120, help="pasos por economía y actualización (múltiplo de 30)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--solo-evaluar", type=str, default=None)
    p.add_argument("--contra", type=str, default=None, help="carpeta de otro modelo para el cara a cara")
    a = p.parse_args()

    carpeta = Path(a.solo_evaluar) if a.solo_evaluar else entrenar(a)
    ippo = IPPO.cargar(carpeta / "modelo.pt")
    a.bancos = ippo.N

    print("\nEvaluando (10 economías, intercambiando posiciones)...")
    rival = IPPO.cargar(Path(a.contra) / "modelo.pt") if a.contra else None
    df = evaluar(ippo, carpeta.name, a, rival, Path(a.contra).name if a.contra else None)
    salida = RESULTADOS / "evaluacion.csv"
    salida.parent.mkdir(parents=True, exist_ok=True)
    if salida.exists():
        viejo = pd.read_csv(salida)
        df = pd.concat([viejo[viejo.modelo != carpeta.name], df])
    df.to_csv(salida, index=False)

    print(f"\nResumen de {carpeta.name} (es_rl=True: el banco del modelo; False: su rival):\n")
    print(resumen(df[df.modelo == carpeta.name]).to_string())
    print(f"\nGuardado en {salida}")


if __name__ == "__main__":
    main()

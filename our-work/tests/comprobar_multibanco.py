"""Comprueba que el entorno funciona bien con VARIOS bancos, usando solo reglas fijas (sin RL).

Cinco comprobaciones; cada una imprime sus números y OK / REVISAR:

    1. Precio -> cuota      el banco que cobra más se queda con menos clientes
    2. Balance              activo = depósitos + capital en cada paso, con acciones al azar (N = 2, 3, 5)
    3. Economía estable     con N = 2, 3, 5 bancos la simulación llega al final sin NaN
    4. Quiebra              un banco quebrado deja de captar depósitos y de prestar, y los demás siguen
    5. Competencia          (informativa) margen y capital con N = 1, 2, 3, 5 bancos. Con reglas fijas
                            no se espera que baje el margen: 'competitiva' no mira a los rivales.
                            El efecto de la competencia se estudia con RL (E6).

Uso (desde our-work/, con el .venv activado):
    python -m tests.comprobar_multibanco                 # 100 hogares, 25 empresas: unos minutos
    python -m tests.comprobar_multibanco --hogares 1000  # economía completa: más lento
"""

import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from multibank.env import N_ACCIONES, MultiBankEnv
from multibank.reglas import REGLAS
from settings.config import Config

SEMILLAS = [2001, 2002, 2003]
EMPRESAS_MIN = 25


def cfg_(a, n_bancos, **kw):
    return Config(n_hogares=a.hogares, n_empresas=max(a.hogares // 20, EMPRESAS_MIN), n_bancos=n_bancos, **kw)


def resultado(ok, texto):
    print(f"   -> {'OK' if ok else 'REVISAR'}: {texto}\n")
    return ok


def jugar(env, politicas, semilla, al_paso=None):
    """Un episodio; politicas = lista de funciones obs -> acción (una por banco)."""
    obs, _ = env.reset(seed=semilla)
    while True:
        acciones = np.stack([pol(obs[b]) for b, pol in enumerate(politicas)])
        obs, r, term, trunc, info = env.step(acciones)
        if al_paso is not None:
            al_paso(env, r, term, info)
        if trunc:
            return


# ---------------------------------------------------------------------------- 1
def precio_cuota(a):
    print("1. Precio -> cuota: dos bancos con la misma regla; el banco 1 cobra 1 punto más de margen")
    env = MultiBankEnv(cfg_(a, 2))
    barato = env.accion_desde(0.005, 0.03, 0.03, 0.01, 2.0)
    caro = env.accion_desde(0.005, 0.04, 0.04, 0.01, 2.0)
    cuotas = []
    for s in SEMILLAS:
        serie = []
        jugar(env, [lambda o: barato, lambda o: caro], s,
              lambda e, r, t, i: serie.append(np.mean(e.sistema.cuota_prestamos(), axis=0)[1]))
        cuotas.append(np.mean(serie[5:]))
    c = float(np.mean(cuotas))
    print(f"   cuota media de préstamos del banco caro: {c:.2f} (por semilla: {np.round(cuotas, 2)})")
    return resultado(0.05 < c < 0.48, "el caro tiene menos cuota, pero no cero" if 0.05 < c < 0.48 else
                     "la cuota no responde al precio como se espera")


# ---------------------------------------------------------------------------- 2
def balance(a):
    print("2. Balance con acciones al azar")
    rng = np.random.default_rng(0)
    for N in (2, 3, 5):
        env = MultiBankEnv(cfg_(a, N))
        pasos = 0
        try:
            for s in SEMILLAS[:2]:
                def comprobar(e, r, t, i):
                    nonlocal pasos
                    e.sistema.comprobar_balance()
                    pasos += 1
                jugar(env, [lambda o: rng.uniform(-1, 1, N_ACCIONES)] * N, s, comprobar)
        except Exception as ex:                     # comprobar_balance lanza un error si no cuadra
            return resultado(False, f"N={N}: el balance no cuadra ({ex})")
        print(f"   N={N}: {pasos} pasos, el balance cuadra en todos")
    return resultado(True, "activo = depósitos + capital en todos los pasos")


# ---------------------------------------------------------------------------- 3
def estable(a):
    print("3. Economía estable con regla 'competitiva' en todos los bancos")
    ok = True
    for N in (2, 3, 5):
        env = MultiBankEnv(cfg_(a, N))
        anos, nans = [], 0
        for s in SEMILLAS:
            obs_ok = []
            jugar(env, [REGLAS["competitiva"](env) for _ in range(N)], s,
                  lambda e, r, t, i: obs_ok.append(np.isfinite(r).all() and np.isfinite(e.observar()).all()))
            anos.append(env.t)
            nans += int(not all(obs_ok))
        print(f"   N={N}: años simulados {anos}, episodios con NaN: {nans}")
        ok &= nans == 0 and min(anos) >= 25
    return resultado(ok, "sin NaN y llegando a ~30 años" if ok else "episodios cortos o con NaN")


# ---------------------------------------------------------------------------- 4
def quiebra(a):
    print("4. Quiebra forzada del banco 0 en el año 5 (3 bancos con regla 'clasica')")
    env = MultiBankEnv(cfg_(a, 3))
    pols = [REGLAS["clasica"](env) for _ in range(3)]
    obs, _ = env.reset(seed=SEMILLAS[0])
    s = env.sistema
    registro = []
    for t in range(30):
        if t == 5:                                   # hundimos su capital
            perdida = 3 * s.K[0]
            s.K[0] -= perdida
            s.Liq[0] -= perdida
        obs, r, term, trunc, info = env.step(np.stack([p(obs[b]) for b, p in enumerate(pols)]))
        registro.append((env.t, bool(s.quebrado[0]), float(s.Dep[0]), float(s.L[0].sum()),
                         float(s.L[1:].sum()), int((~s.quebrado).sum()), float(r[0])))
        if trunc:
            break
    for fila in registro[3:10]:
        print(f"   año {fila[0]:2d}  quebrado={fila[1]!s:5}  depósitos b0={fila[2]:12.0f}  "
              f"préstamos b0={fila[3]:12.0f}  préstamos resto={fila[4]:12.0f}  vivos={fila[5]}  r0={fila[6]:+.2f}")
    tras = [f for f in registro if f[0] > 6]
    ok = (registro[5][1] and all(f[1] for f in tras) and all(f[2] == 0 for f in tras)
          and all(tras[i + 1][3] <= tras[i][3] + 1e-6 for i in range(len(tras) - 1))
          and all(f[5] == 2 for f in tras) and all(f[6] == 0 for f in tras))
    return resultado(ok, "quiebra en el año del golpe; después sin depósitos, sus préstamos solo bajan, "
                         "recompensa 0 y los otros 2 siguen" if ok else "el tratamiento de la quiebra no es el esperado")


# ---------------------------------------------------------------------------- 5
def competencia(a):
    print("5. Competencia: regla 'competitiva' con N = 1, 2, 3, 5 bancos")
    filas = []
    for N in (1, 2, 3, 5):
        env = MultiBankEnv(cfg_(a, N))
        margenes, capitales = [], []
        for s in SEMILLAS:
            serie = []
            jugar(env, [REGLAS["competitiva"](env) for _ in range(N)], s,
                  lambda e, r, t, i: serie.append(float(np.mean(e.tipos.empresas[~e.sistema.quebrado]) - i["base"])))
            margenes.append(np.mean(serie[5:]))
            capitales.append(float(np.mean(env.sistema.K / env.sistema.K0)))
        filas.append((N, np.mean(margenes), np.mean(capitales)))
        print(f"   N={N}: margen medio a empresas = {100 * filas[-1][1]:.2f} puntos  |  capital final medio por banco = {filas[-1][2]:.2f}")
    print("   (informativa) La regla 'competitiva' baja el margen mientras gana, mire o no a los rivales,\n"
          "   así que acaba en el mismo margen con 1 o con 5 bancos. Que la competencia baje márgenes depende\n"
          "   de que los bancos reaccionen entre sí: eso es lo que tiene que aprender (o no) el RL en E6.\n"
          "   Ojo: K0 es el 15% de lo que presta cada banco al empezar, así que K/K0 no se compara entre N.\n")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hogares", type=int, default=100)
    a = p.parse_args()
    print(f"Economía: {a.hogares} hogares, {max(a.hogares // 20, EMPRESAS_MIN)} empresas, semillas {SEMILLAS}\n")
    res = [f(a) for f in (precio_cuota, balance, estable, quiebra, competencia)]
    print(f"Resumen: {sum(res)}/{len(res)} comprobaciones OK")


if __name__ == "__main__":
    main()

"""Graba una simulación del entorno MultiBank y genera el visor: our-work/viz/simulacion.html

Uso (desde our-work):
    uv run python -m multibank.visual --abrir
    uv run python -m multibank.visual --politicas agresiva,clasica,prudente --hogares 1000 --abrir
    uv run python -m multibank.visual --politicas azar,azar --semilla 3

Cada banco juega con una política, separadas por comas:
    - una regla: prudente, agresiva, clasica, competitiva
    - azar
    - un banco ENTRENADO: la carpeta donde se guardó
        results/e1/ppo_h100_s1                 nivel 1 (SB3): ocupa 1 banco
        results/e2/ippo_n2_h100_p100k_..._s1   niveles 2 y 3 (IPPO/MAPPO): ocupa TODOS sus bancos
        results/e2/ippo_n2_..._s1#0            solo la red del banco 0 (para ponerla contra reglas)

    uv run python -m multibank.visual --politicas results/e1/ppo_h100_s1,prudente --abrir
    uv run python -m multibank.visual --politicas results/e2/mappo_n2_h100_p100k_correcto_s1 --abrir

Ver cómo aprende (--fotos): si el primer banco entrenado guardó fotos durante el entrenamiento
(carpeta fotos/), se graba la misma economía con cada foto y el visor deja elegir el momento:
    uv run python -m multibank.visual --politicas results/e1/ppo_h100_s1,prudente --fotos --abrir
"""

import argparse
import json
import pickle
import re
import webbrowser
from pathlib import Path

import numpy as np

from multibank.env import N_ACCIONES, MultiBankEnv
from multibank.reglas import REGLAS
from settings.config import Config

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent                                   # our-work/
EMPRESAS_MIN = 25                                    # igual que en los experimentos
SALIDA = AQUI.parent / "viz" / "simulacion.html"
PLANTILLA = AQUI / "visor.html"


def r(x, d=0):
    return [round(float(v), d) for v in np.ravel(x)]


def foto(env, info, recompensas):
    s, e, t = env.sistema, env.econ, env.tipos
    F = s.F
    Lb = s.L.sum(axis=1)
    deuda_cliente = s.L.sum(axis=0)
    principal = np.where(deuda_cliente > 0, s.L.argmax(axis=0), -1)
    moroso = s.es_moroso().any(axis=0)
    ce, ch = s.cuota_prestamos()
    # El tipo que paga cada cliente a su banco principal (sin deuda: el más barato que le ofrecen)
    tipos_c = s.tipo_por_cliente(t)
    tipo_cliente = np.where(principal >= 0, tipos_c[np.maximum(principal, 0), np.arange(s.C)], tipos_c.min(axis=0))
    return {
        "base": float(info["base"]),
        "pib": float(e.main_gov.GDP),
        "inflacion": float(getattr(e, "inflation_rate", 0.0)),
        "bancos": [{
            "dep": float(t.deposito[b]), "emp": float(t.empresas[b]), "hog": float(t.hogares[b]),
            "prima": float(t.prima[b]), "umbral": float(t.umbral[b]),
            "capital": float(s.K[b]), "capital0": float(s.K0[b]), "ratio": float(s.K[b] / max(Lb[b], 1e-9)),
            "p_emp": float(s.L[b, :F].sum()), "p_hog": float(s.L[b, F:].sum()), "bonos": float(s.G[b]),
            "depositos": float(s.Dep[b]), "liquidez": float(s.Liq[b]),
            "beneficio": float(info["beneficio"][b]), "perdida": float(info["perdida"][b]),
            "recompensa": float(recompensas[b]), "quebrado": bool(s.quebrado[b]),
            "cuota_emp": float(ce[b]), "cuota_hog": float(ch[b]), "cuota_dep": float(s.Dep[b] / max(s.Dep.sum(), 1e-9)),
            "morosos": int(s.es_moroso()[b].sum()),
        } for b in range(s.N)],
        "empresas": {
            "deuda": r(deuda_cliente[:F]), "banco": [int(x) for x in principal[:F]],
            "impaga": [int(x) for x in info["impaga"][:F]], "p": r(info["p_impago"][:F], 4),
            "moroso": [int(x) for x in moroso[:F]], "racionado": r(info["racionado"][:F]),
            "tipo": r(tipo_cliente[:F], 4),
        },
        "hogares": {
            "riqueza": r(e.households.at), "ingreso": r(e.households.income),
            "deuda": r(deuda_cliente[F:]), "banco": [int(x) for x in principal[F:]],
            "impaga": [int(x) for x in info["impaga"][F:]], "moroso": [int(x) for x in moroso[F:]],
            "tipo": r(tipo_cliente[F:], 4),
            "edad": [int(x) for x in np.ravel(e.households.age)],
        },
    }


# ---------------------------------------------------------------------------- bancos entrenados
def es_modelo(nombre):
    return nombre.split("#")[0] not in REGLAS and nombre != "azar"


def ruta(nombre):
    p = Path(nombre.split("#")[0])
    return p if p.is_absolute() or p.exists() else RAIZ / p


def cargar_sb3(archivo, normalizador):
    """Banco de nivel 1 (PPO o SAC de SB3) -> función obs -> acción."""
    from stable_baselines3 import PPO, SAC
    algo = SAC if "sac" in archivo.parent.name.lower() or "sac" in archivo.parents[1].name.lower() else PPO
    modelo = algo.load(archivo)
    vecnorm = None
    if normalizador.exists():
        with open(normalizador, "rb") as f:
            vecnorm = pickle.load(f)                 # solo hacen falta sus medias, no su entorno

    def f(obs):
        o = obs[None, :]
        if vecnorm is not None:
            o = np.clip((o - vecnorm.obs_rms.mean) / np.sqrt(vecnorm.obs_rms.var + vecnorm.epsilon),
                        -vecnorm.clip_obs, vecnorm.clip_obs).astype(np.float32)
        return modelo.predict(o, deterministic=True)[0][0]
    return f


def cargar_ippo(archivo):
    """IPPO / MAPPO -> lista de funciones obs -> acción, una por red (banco)."""
    from algoritmos.ippo import IPPO
    m = IPPO.cargar(archivo)
    return [(lambda k: (lambda obs: m.actuar(obs[None, None], determinista=True, banco=k)[0][0, 0]))(k)
            for k in range(m.N)], type(m).__name__


def fotos_de(carpeta):
    """Las fotos guardadas durante el entrenamiento + el modelo final: [(etiqueta, archivo, normalizador)]."""
    carpeta = ruta(str(carpeta))
    salida = []
    if (carpeta / "fotos").is_dir():
        archivos = sorted((carpeta / "fotos").glob("paso_*"), key=lambda a: int(re.findall(r"\d+", a.stem)[0]))
        for a in archivos:
            n = int(re.findall(r"\d+", a.stem)[0])
            salida.append((f"tras {n:,} pasos".replace(",", "."), a, carpeta / "fotos" / f"vecnormalize_{n}.pkl"))
    final = carpeta / "modelo.zip" if (carpeta / "modelo.zip").exists() else carpeta / "modelo.pt"
    salida.append(("final", final, carpeta / "vecnormalize.pkl"))
    return salida


def politicas_de(nombre, foto=None):
    """Una entrada de --politicas que es un modelo -> [(etiqueta, función)] (uno o varios bancos)."""
    carpeta = ruta(nombre)
    _, archivo, norm = foto or fotos_de(carpeta)[-1]
    if archivo.suffix == ".zip":
        return [(f"{carpeta.name} (RL)", cargar_sb3(archivo, norm))]
    redes, tipo = cargar_ippo(archivo)
    if "#" in nombre:
        k = int(nombre.split("#")[1])
        return [(f"{tipo} banco {k} ({carpeta.name})", redes[k])]
    return [(f"{tipo} banco {k} ({carpeta.name})", r) for k, r in enumerate(redes)]


def montar(nombres, semilla, foto=None):
    """--politicas -> (etiquetas, fábricas). Una fábrica recibe el entorno y devuelve obs -> acción."""
    rng = np.random.default_rng(semilla)
    etiquetas, fabricas, primero = [], [], True
    for n in nombres:
        if n == "azar":
            etiquetas.append("azar")
            fabricas.append(lambda env: (lambda obs: rng.uniform(-1, 1, N_ACCIONES)))
        elif not es_modelo(n):
            etiquetas.append(n)
            fabricas.append((lambda r: (lambda env: REGLAS[r](env)))(n))
        else:
            for etq, f in politicas_de(n, foto if primero else None):
                etiquetas.append(etq)
                fabricas.append((lambda g: (lambda env: g))(f))
            primero = False
    return etiquetas, fabricas


def config(hogares, n_bancos):
    return Config(n_hogares=hogares, n_empresas=max(hogares // 20, EMPRESAS_MIN), n_bancos=n_bancos)


# ---------------------------------------------------------------------------- grabar
def grabar(cfg, nombres, semilla, fabricas=None, env=None):
    env = env or MultiBankEnv(cfg)
    if fabricas is None:
        nombres, fabricas = montar(nombres, semilla)
    politicas = [f(env) for f in fabricas]
    obs, _ = env.reset(seed=semilla)
    historia, motivo = [], "se completó el episodio"
    while True:
        acciones = np.stack([p(obs[b]) for b, p in enumerate(politicas)])
        obs, recompensas, terminado, truncado, info = env.step(acciones)
        historia.append(foto(env, info, recompensas))
        if truncado:
            if terminado.all():
                motivo = "quebraron todos los bancos"
            elif info["economia_termina"] and env.t < cfg.episode_length:
                motivo = "la economía del original se rompió"
            break
    return {
        "politicas": nombres, "n_empresas": env.sistema.F, "n_hogares": env.sistema.H,
        "capital_minimo": cfg.capital_minimo, "semilla": semilla, "fin": motivo,
        "empresa_peso": r(env.sistema.peso_empresa, 5),
        "historia": historia,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--politicas", default="agresiva,clasica,prudente",
                   help="una por banco: " + ", ".join(list(REGLAS) + ["azar"]) + " o la carpeta de un banco entrenado")
    p.add_argument("--hogares", type=int, default=100, help="100 o 1000 (empresas: hogares/20, mínimo 25, como en los experimentos)")
    p.add_argument("--semilla", type=int, default=1)
    p.add_argument("--fotos", action="store_true", help="grabar una simulación por cada foto del entrenamiento")
    p.add_argument("--abrir", action="store_true", help="abrir el visor en el navegador al acabar")
    args = p.parse_args()

    nombres = [n.strip() for n in args.politicas.split(",")]
    etiquetas, _ = montar(nombres, args.semilla)
    cfg = config(args.hogares, len(etiquetas))
    env = MultiBankEnv(cfg)

    modelos = [n for n in nombres if es_modelo(n)]
    if args.fotos and modelos:
        grabaciones = []
        for foto in fotos_de(modelos[0]):
            etq, fabricas = montar(nombres, args.semilla, foto)
            g = grabar(cfg, etq, args.semilla, fabricas, env)    # misma economía y semilla en todas
            g["etiqueta"] = foto[0]
            grabaciones.append(g)
            print(f"  {foto[0]:>18}: {len(g['historia'])} años, capital final "
                  + ", ".join(f"{b['capital'] / b['capital0']:.1f}x" for b in g["historia"][-1]["bancos"]))
        datos = {"grabaciones": grabaciones}
        resumen = f"{len(grabaciones)} momentos del entrenamiento"
    else:
        etq, fabricas = montar(nombres, args.semilla)
        datos = grabar(cfg, etq, args.semilla, fabricas, env)
        resumen = f"{len(datos['historia'])} años grabados ({datos['fin']})"

    html = PLANTILLA.read_text(encoding="utf-8").replace("__DATOS__", json.dumps(datos, separators=(",", ":")))
    SALIDA.parent.mkdir(exist_ok=True)
    SALIDA.write_text(html, encoding="utf-8")
    print(f"{resumen} -> {SALIDA}")
    if args.abrir:
        webbrowser.open(SALIDA.as_uri())


if __name__ == "__main__":
    main()

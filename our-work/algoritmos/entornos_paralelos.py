"""Varias economías MultiBankEnv en paralelo, cada una en su propio proceso.

Cada proceso tiene su propia copia de EconGym (que usa el generador aleatorio global de numpy),
así que las economías no se pisan entre sí. Cuando un episodio termina (truncado), el entorno
se reinicia solo y devuelve también la observación final, para hacer bootstrap con ella.
"""

import multiprocessing as mp
import time

import numpy as np


def _crear(cfg):
    from multibank.env import MultiBankEnv
    # El Runner del original crea una carpeta runN al arrancar: con varios procesos a la vez,
    # dos pueden chocar al crear la misma. Se reintenta tras una espera.
    for intento in range(20):
        try:
            return MultiBankEnv(cfg)
        except FileExistsError:
            time.sleep(0.2 + 0.3 * np.random.default_rng(intento).random())
    raise RuntimeError("No se pudo crear el entorno (choque al crear la carpeta runN del original)")


def _trabajador(conexion, cfg, semilla_base):
    env = _crear(cfg)
    episodio = 0
    while True:
        orden, dato = conexion.recv()
        if orden == "reset":
            obs, _ = env.reset(seed=semilla_base + episodio)
            conexion.send(obs)
        elif orden == "step":
            obs, r, term, trunc, info = env.step(dato)
            final = None
            if trunc:
                final = obs
                episodio += 1
                obs, _ = env.reset(seed=semilla_base + episodio)
            margen = float(np.mean(env.tipos.empresas - info["base"]))
            conexion.send((obs, r, term, trunc, final, dict(
                nuevas_quiebras=info["nuevas_quiebras"], beneficio=info["beneficio"],
                margen_empresas=env.tipos.empresas - info["base"], margen_medio=margen)))
        elif orden == "info":
            conexion.send((env.N, env.n_obs))
        elif orden == "cerrar":
            conexion.close()
            return


class EntornosParalelos:
    def __init__(self, cfg, n, semilla):
        ctx = mp.get_context("spawn")
        self.conex, self.procs = [], []
        for i in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_trabajador, args=(b, cfg, semilla * 100_000 + i * 10_000), daemon=True)
            p.start()
            self.conex.append(a)
            self.procs.append(p)
        self.conex[0].send(("info", None))
        self.N, self.n_obs = self.conex[0].recv()

    def reset(self):
        for c in self.conex:
            c.send(("reset", None))
        return np.stack([c.recv() for c in self.conex])

    def step(self, acciones):
        for c, a in zip(self.conex, acciones):
            c.send(("step", a))
        res = [c.recv() for c in self.conex]
        obs = np.stack([x[0] for x in res])
        r = np.stack([x[1] for x in res])
        term = np.stack([x[2] for x in res])
        trunc = np.array([x[3] for x in res])
        finales = [x[4] for x in res]
        infos = [x[5] for x in res]
        return obs, r, term, trunc, finales, infos

    def cerrar(self):
        for c in self.conex:
            try:
                c.send(("cerrar", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)

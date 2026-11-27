"""IPPO: PPO independiente para N bancos. Cada banco tiene su propia red (actor y crítico) y su optimizador.

Lo que lo diferencia de un PPO de un solo agente (y lo que se estudia en E3):

    - `done` POR BANCO. La quiebra de un banco es `terminated` solo para él: su valor futuro es 0.
      El final del episodio (30 años) es `truncated`: el banco seguiría existiendo, así que se
      hace bootstrap con el valor del estado final.
    - Banco ZOMBI. Tras quebrar, el banco sigue en el entorno con recompensa 0, pero sus pasos
      no entran en la loss (máscara `vivo`). Así el número de agentes es fijo y no se aprende
      de transiciones que no dependen de lo que hace.

    modo="correcto"  -> lo anterior.
    modo="ingenuo"   -> como un PPO de un agente mal adaptado: el final de episodio se trata como
                        terminal (sin bootstrap) y los pasos del zombi SÍ entran en la loss (E3).

Uso:
    ippo = IPPO(n_bancos=2, n_obs=18, n_acc=5)
    acciones, info = ippo.actuar(obs)          # obs: (E, N, n_obs) -> acciones (E, N, n_acc)
    ippo.guardar(ruta) / IPPO.cargar(ruta)
El bucle de entrenamiento (recoger datos y llamar a `actualizar`) está en experiments/e2_ippo.py.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class HiperIPPO:
    lr: float = 3e-4
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 10
    minibatches: int = 4
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    oculta: int = 64
    modo: str = "correcto"           # "correcto" | "ingenuo"  (E3)


class Normalizador:
    """Media y varianza en línea (como VecNormalize de SB3)."""

    def __init__(self, forma, clip=10.0):
        self.media = np.zeros(forma, np.float64)
        self.var = np.ones(forma, np.float64)
        self.n = 1e-4
        self.clip = clip

    def actualizar(self, x):
        x = np.asarray(x, np.float64).reshape(-1, *self.media.shape)
        m, v, k = x.mean(0), x.var(0), x.shape[0]
        d = m - self.media
        tot = self.n + k
        self.media = self.media + d * k / tot
        self.var = (self.var * self.n + v * k + d ** 2 * self.n * k / tot) / tot
        self.n = tot

    def __call__(self, x):
        return np.clip((x - self.media) / np.sqrt(self.var + 1e-8), -self.clip, self.clip).astype(np.float32)


class ActorCritico(nn.Module):
    def __init__(self, n_obs, n_acc, oculta, n_obs_critico=None):
        super().__init__()

        def mlp(salida, escala, entrada=n_obs):
            capas = [nn.Linear(entrada, oculta), nn.Tanh(), nn.Linear(oculta, oculta), nn.Tanh(), nn.Linear(oculta, salida)]
            for c in capas:
                if isinstance(c, nn.Linear):
                    nn.init.orthogonal_(c.weight, np.sqrt(2))
                    nn.init.zeros_(c.bias)
            nn.init.orthogonal_(capas[-1].weight, escala)
            return nn.Sequential(*capas)

        self.actor = mlp(n_acc, 0.01)
        self.critico = mlp(1, 1.0, n_obs_critico or n_obs)
        self.log_std = nn.Parameter(torch.zeros(n_acc))

    def distribucion(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp())

    def valor(self, obs):
        return self.critico(obs).squeeze(-1)


class IPPO:
    def __init__(self, n_bancos, n_obs, n_acc, hiper: HiperIPPO | None = None, semilla=0):
        self.h = hiper or HiperIPPO()
        torch.manual_seed(semilla)
        self.N, self.n_obs, self.n_acc = n_bancos, n_obs, n_acc
        self.redes = [ActorCritico(n_obs, n_acc, self.h.oculta, self.n_obs_critico()) for _ in range(n_bancos)]
        self.opt = [torch.optim.Adam(r.parameters(), lr=self.h.lr, eps=1e-5) for r in self.redes]
        self.norm_obs = Normalizador((n_obs,))               # compartido: todos ven las mismas magnitudes
        self.escala_r = [Normalizador(()) for _ in range(n_bancos)]   # escala de la recompensa, por banco
        self.ret_acum = None

    # ------------------------------------------------------------------ crítico
    # IPPO: el crítico de cada banco solo ve SU observación. MAPPO (mappo.py) lo cambia por la de todos.
    def n_obs_critico(self):
        return self.n_obs

    def entrada_critico(self, o, b):
        """o: (..., N, n_obs) ya normalizada -> entrada del crítico del banco b: (..., n_obs_critico)."""
        return o[..., b, :]

    # ------------------------------------------------------------------ actuar
    @torch.no_grad()
    def actuar(self, obs, determinista=False, banco=None):
        """obs (E, N, n_obs) sin normalizar -> acciones (E, N, n_acc) en [-1, 1], log-probs y valores.
        Con `banco=k` usa la red k para todas las posiciones (para enfrentarla a reglas)."""
        o = torch.as_tensor(self.norm_obs(obs))
        acc, logp, val = [], [], []
        for b in range(o.shape[1]):
            red = self.redes[b if banco is None else banco]
            d = red.distribucion(o[:, b])
            a = d.mean if determinista else d.sample()
            acc.append(a)
            logp.append(d.log_prob(a).sum(-1))
            val.append(red.valor(self.entrada_critico(o, b)) if banco is None else torch.zeros(o.shape[0]))
        a = torch.stack(acc, 1)
        return a.clamp(-1, 1).numpy(), dict(accion=a.numpy(), logp=torch.stack(logp, 1).numpy(),
                                             valor=torch.stack(val, 1).numpy())

    @torch.no_grad()
    def valor(self, obs):
        o = torch.as_tensor(self.norm_obs(obs))
        return torch.stack([self.redes[b].valor(self.entrada_critico(o, b)) for b in range(self.N)], 1).numpy()

    def escalar_recompensa(self, r, nuevo_episodio):
        """Divide la recompensa por la desviación del retorno descontado (como VecNormalize). r: (E, N)."""
        if self.ret_acum is None:
            self.ret_acum = np.zeros_like(r, np.float64)
        self.ret_acum = self.ret_acum * self.h.gamma * (~nuevo_episodio)[:, None] + r
        out = np.empty_like(r, np.float32)
        for b in range(self.N):
            self.escala_r[b].actualizar(self.ret_acum[:, b])
            out[:, b] = np.clip(r[:, b] / np.sqrt(self.escala_r[b].var + 1e-8), -10, 10)
        return out

    # ------------------------------------------------------------------ aprender
    def actualizar(self, lote):
        """lote: dict de arrays (T, E, N, ...) con obs, accion, logp, valor, recompensa, terminado,
        fin_episodio, vivo y valor_siguiente (valor del estado tras el último paso)."""
        h = self.h
        T = lote["recompensa"].shape[0]
        r, v, vivo = lote["recompensa"], lote["valor"], lote["vivo"]
        term, fin = lote["terminado"], lote["fin_episodio"][..., None]          # term (T,E,N), fin (T,E,1)
        v_sig = np.concatenate([v[1:], lote["valor_siguiente"][None]], 0)

        if h.modo == "correcto":
            # terminal para el banco: su quiebra (su valor futuro es 0)
            # fin del episodio sin quiebra (truncado): bootstrap con el valor del estado final
            no_terminal = ~term
            v_sig = np.where(fin & ~term, lote["valor_final"], v_sig)
            corta_gae = fin | term
            mascara = vivo
        else:  # ingenuo: fin de episodio = terminal y el zombi cuenta en la loss
            no_terminal = ~(term | fin)
            corta_gae = fin | term
            mascara = np.ones_like(vivo)

        ventaja = np.zeros_like(r)
        ultimo = 0.0
        for t in reversed(range(T)):
            delta = r[t] + h.gamma * v_sig[t] * no_terminal[t] - v[t]
            ultimo = delta + h.gamma * h.gae_lambda * (~corta_gae[t]) * ultimo
            ventaja[t] = ultimo
        retorno = ventaja + v

        stats = []
        for b in range(self.N):
            m = mascara[..., b].reshape(-1).astype(bool)
            if m.sum() < 8:
                continue
            o_todos = torch.as_tensor(self.norm_obs(lote["obs"]))                       # (T, E, N, n_obs)
            obs = o_todos[:, :, b].reshape(-1, self.n_obs)[m]
            obs_c = self.entrada_critico(o_todos, b).reshape(-1, self.n_obs_critico())[m]
            acc = torch.as_tensor(lote["accion"][:, :, b].reshape(-1, self.n_acc)[m])
            logp0 = torch.as_tensor(lote["logp"][..., b].reshape(-1)[m])
            adv = torch.as_tensor(ventaja[..., b].reshape(-1)[m], dtype=torch.float32)
            ret = torch.as_tensor(retorno[..., b].reshape(-1)[m], dtype=torch.float32)
            red, opt = self.redes[b], self.opt[b]
            n = len(adv)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)          # normalizada sobre todo el lote del banco
            for _ in range(h.epochs):
                orden = torch.randperm(n)
                for idx in torch.tensor_split(orden, min(h.minibatches, n // 8)):   # minilotes de tamaño parecido
                    d = red.distribucion(obs[idx])
                    logp = d.log_prob(acc[idx]).sum(-1)
                    ratio = (logp - logp0[idx]).clamp(-20, 20).exp()
                    a = adv[idx]
                    perdida_pi = -torch.min(ratio * a, ratio.clamp(1 - h.clip, 1 + h.clip) * a).mean()
                    perdida_v = 0.5 * (red.valor(obs_c[idx]) - ret[idx]).pow(2).mean()
                    entropia = d.entropy().sum(-1).mean()
                    perdida = perdida_pi + h.vf_coef * perdida_v - h.ent_coef * entropia
                    opt.zero_grad()
                    perdida.backward()
                    nn.utils.clip_grad_norm_(red.parameters(), h.max_grad_norm)
                    opt.step()
            with torch.no_grad():
                v_pred = red.valor(obs_c)
                var_ret = ret.var()
                ev = float(1 - (ret - v_pred).var() / (var_ret + 1e-8))
                kl = float((logp0 - red.distribucion(obs).log_prob(acc).sum(-1)).mean())
            stats.append(dict(banco=b, perdida_pi=float(perdida_pi), perdida_v=float(perdida_v),
                              explained_variance=ev, approx_kl=kl, std=float(red.log_std.exp().mean())))
        return stats

    # ------------------------------------------------------------------ guardar
    def guardar(self, ruta):
        torch.save(dict(tipo=type(self).__name__, N=self.N, n_obs=self.n_obs, n_acc=self.n_acc, hiper=asdict(self.h),
                        redes=[r.state_dict() for r in self.redes],
                        norm=(self.norm_obs.media, self.norm_obs.var, self.norm_obs.n)), ruta)

    @classmethod
    def cargar(cls, ruta):
        """Carga un IPPO o un MAPPO (según lo que se guardó)."""
        d = torch.load(ruta, weights_only=False)
        if d.get("tipo") == "MAPPO" and cls is IPPO:
            from algoritmos.mappo import MAPPO
            cls = MAPPO
        ippo = cls(d["N"], d["n_obs"], d["n_acc"], HiperIPPO(**d["hiper"]))
        for red, sd in zip(ippo.redes, d["redes"]):
            red.load_state_dict(sd)
        ippo.norm_obs.media, ippo.norm_obs.var, ippo.norm_obs.n = d["norm"]
        return ippo

"""MAPPO: como IPPO (cada banco con su actor y su crítico), pero el CRÍTICO de cada banco ve las
observaciones de TODOS los bancos durante el entrenamiento ("entrenamiento centralizado,
ejecución descentralizada").

    - Actor: igual que en IPPO. Cada banco decide solo con SU observación, también al evaluar.
    - Crítico: recibe la observación de todos los bancos concatenada. Así puede distinguir
      "me ha ido mal porque el rival ha bajado tipos" de "me ha ido mal por mala suerte",
      y estima mejor el valor en un entorno que cambia porque los demás aprenden (no estacionariedad).

El crítico solo se usa para entrenar, así que al evaluar MAPPO no tiene ninguna información extra.
Todo lo demás (done por banco, banco zombi, modo correcto/ingenuo) es idéntico a IPPO.
"""

from algoritmos.ippo import IPPO


class MAPPO(IPPO):
    def n_obs_critico(self):
        return self.N * self.n_obs

    def entrada_critico(self, o, b):
        # (..., N, n_obs) -> (..., N * n_obs): el estado de todos los bancos. El banco b se
        # distingue porque cada banco tiene su propio crítico.
        return o.reshape(*o.shape[:-2], self.N * self.n_obs)

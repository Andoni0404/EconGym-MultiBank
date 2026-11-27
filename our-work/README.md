# Nuestro trabajo

Aquí va todo lo que hemos hecho nosotros (código, experimentos, resultados, informe).
El código original de EconGym está en `../EconGym-original/` y no se toca.

## Estructura

```
our-work/
├── settings/            config.py: todos los parámetros (Config() = experimentos, Config.pruebas() = economía pequeña)
├── multibank/           el entorno
│   ├── env.py           MultiBankEnv (N bancos) y BancoUnicoEnv (1 banco, API de Gymnasium para SB3)
│   ├── banking.py       los N bancos: préstamos, intereses, impagos, morosos, capital mínimo y quiebra
│   ├── reglas.py        reglas fijas para comparar con el RL: prudente, agresiva, clásica y competitiva
│   ├── core/            las 5 partes de la economía, heredando del original
│   ├── econgym.py       el puente que carga la economía del original
│   └── visual.py        graba una simulación y genera el visor (visor.html)
├── algoritmos/          algoritmos de RL reutilizables (IPPO; luego MAPPO y PPO-Lagrangiano). No se llama
│                        `agents/` porque chocaría con la carpeta `agents/` del EconGym original
├── experiments/         un script por experimento del plan
│   ├── e1_banco_unico.py      E1: PPO / SAC con 1 banco frente a las reglas
│   ├── e2_ippo.py             E2: N bancos con IPPO; evaluación con intercambio de posiciones
│   ├── cola.py + cola.txt     cola de experimentos para dejarlos corriendo (p. ej. por la noche)
│   └── comprobaciones/        calibrar_temperatura.py (Optuna) y comprobar_multibanco.py
├── analysis/            tablas y gráficas para el informe a partir de los CSV
├── results/             salidas (solo se suben los CSV finales; modelos, logs y TensorBoard no)
└── tests/               tests (`pytest` desde la raíz)
```

## Cómo se usa

Desde `our-work/`:

```bash
# Ver una simulación (abre el navegador)
uv run python -m multibank.visual --abrir
uv run python -m multibank.visual --politicas agresiva,clasica,prudente --hogares 1000 --abrir

# Tests (desde la raíz del repo)
uv run pytest

# Comprobaciones del entorno
python -m experiments.comprobaciones.comprobar_multibanco
python -m experiments.comprobaciones.calibrar_temperatura --trials 40 --hogares 1000

# Experimentos
python -m experiments.e1_banco_unico --pasos 100000 --entornos 8
python -m experiments.cola          # lanza en orden lo que haya en experiments/cola.txt
```

En código:

```python
from multibank.env import MultiBankEnv, BancoUnicoEnv
from settings.config import Config

env = MultiBankEnv(Config(n_bancos=3))          # 1000 hogares, 50 empresas
obs, info = env.reset(seed=1)                   # obs: (3, 18)
obs, recompensas, terminado, truncado, info = env.step(acciones)   # acciones: (3, 5) en [-1, 1]

# Un banco con SB3
from stable_baselines3 import PPO
PPO("MlpPolicy", BancoUnicoEnv(Config.pruebas())).learn(10_000)
```

## Qué decide cada banco (acción: 5 números en [-1, 1])

1. Tipo de depósito: entre el tipo base − 1% y el tipo base.
2. Margen a empresas: entre base + 1% y base + 8%.
3. Margen a hogares: entre base + 1% y base + 8%.
4. Extra por riesgo: de 0 a 5% por cada "vez" que el cliente va más endeudado que el típico de su clase.
5. Umbral: la deuda máxima, en años de ingreso (0 a 4), de un hogar al que acepta prestar.

Cada cliente paga su propio tipo: base + margen de su clase + extra × (riesgo − 1), como mucho base + 15%.

## Qué ve cada banco (observación: 18 números)

Tipo base, inflación, su capital frente a sus préstamos, su capital frente al inicial, sus cuotas
(empresas, hogares y depósitos), sus pérdidas del último año, sus morosos, sus últimas decisiones
(márgenes, extra por riesgo, umbral), los márgenes y el extra por riesgo de los rivales (si `ver_rivales`), el año del episodio y la rentabilidad de las empresas.

## Recompensa

Cambio de su capital en el año / capital inicial (con signo). Si quiebra: − λ (`lam_quiebra`) y el episodio
termina para ese banco (`terminated`). El episodio acaba a los 30 años (`truncated`).

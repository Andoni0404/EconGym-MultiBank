"""Cola de experimentos: lanza uno detrás de otro, sin que haya que estar delante.

Cada línea de `experiments/cola.txt` es un experimento (los argumentos de `python -m ...`).
Las líneas vacías y las que empiezan por # se ignoran. Ejemplo de cola.txt:

    # E1: PPO con 3 semillas y SAC
    experiments.e1_banco_unico --pasos 300000 --entornos 8 --semilla 2
    experiments.e1_banco_unico --pasos 300000 --entornos 8 --semilla 3
    experiments.e1_banco_unico --algo sac --pasos 100000 --entornos 8 --semilla 1

Qué hace:
    - Ejecuta las líneas en orden, una cada vez.
    - Guarda la salida de cada una en results/logs/<n>_<fecha>.log (la consola solo muestra el progreso).
    - Si una falla, lo apunta y sigue con la siguiente: un error no tira la noche entera.
    - Apunta las terminadas en results/cola_hecho.txt. Si se relanza la cola (por ejemplo tras
      un apagado), se salta las que ya terminaron bien.

Uso (desde our-work/, con el .venv activado):
    python -m experiments.cola                  # usa experiments/cola.txt
    python -m experiments.cola mi_cola.txt      # otra lista
    python -m experiments.cola --rehacer        # ignora lo ya hecho y lo repite todo
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent                       # our-work/
LOGS = RAIZ / "results" / "logs"
HECHO = RAIZ / "results" / "cola_hecho.txt"


def leer_cola(ruta):
    lineas = []
    for linea in Path(ruta).read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#"):
            lineas.append(linea)
    return lineas


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rehacer = "--rehacer" in sys.argv
    ruta = Path(args[0]) if args else AQUI / "cola.txt"
    cola = leer_cola(ruta)

    LOGS.mkdir(parents=True, exist_ok=True)
    hechos = set() if rehacer or not HECHO.exists() else set(HECHO.read_text(encoding="utf-8").splitlines())

    print(f"Cola: {len(cola)} experimentos ({ruta})")
    resumen = []
    for i, linea in enumerate(cola, 1):
        if linea in hechos:
            print(f"[{i}/{len(cola)}] ya hecho, se salta: {linea}")
            resumen.append((linea, "saltado", 0.0))
            continue

        log = LOGS / f"{i:02d}_{datetime.now():%Y%m%d_%H%M%S}.log"
        print(f"[{i}/{len(cola)}] {datetime.now():%H:%M}  {linea}\n          log: {log}")
        t = time.time()
        with open(log, "w", encoding="utf-8") as f:
            f.write(f"$ python -m {linea}\n\n")
            f.flush()
            proc = subprocess.run([sys.executable, "-m", *linea.split()], cwd=RAIZ,
                                  stdout=f, stderr=subprocess.STDOUT)
        minutos = (time.time() - t) / 60

        if proc.returncode == 0:
            with open(HECHO, "a", encoding="utf-8") as f:
                f.write(linea + "\n")
            print(f"          OK en {minutos:.1f} min")
            resumen.append((linea, "OK", minutos))
        else:
            print(f"          FALLÓ (código {proc.returncode}) en {minutos:.1f} min: mira el log")
            resumen.append((linea, "FALLÓ", minutos))

    print("\nResumen:")
    for linea, estado, minutos in resumen:
        print(f"  {estado:8s} {minutos:6.1f} min  {linea}")


if __name__ == "__main__":
    main()

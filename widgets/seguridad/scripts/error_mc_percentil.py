"""
Cuánto ruido Monte Carlo tiene el extremo del intervalo según cuántas réplicas.

POR QUÉ EXISTE. `N_REPLICAS = 10000` se justificaba con una medición de Codex
que no quedó guardada en ninguna parte: ni el experimento ni el estimador ni el
universo de perfiles. Al revisar la recalibración, Codex intentó reproducirla y
le dieron otros números — no porque los primeros fueran falsos, sino porque no
había con qué comprobarlos. Una constante que gobierna el tamaño de los
artefactos publicados no puede apoyarse en un número irreproducible.

QUÉ MIDE. Para cada uno de los 1.008 perfiles de la UI y cada extremo del
intervalo al nivel que la pregunta publica: el desvío estándar de ese extremo si
el bootstrap se hubiera corrido con B réplicas.

CÓMO, y esto cambió después de la tercera vuelta de Codex. La primera versión
remuestreaba SIN reposición y corregía por población finita, lo que la obligaba
a extrapolar por 1/raíz(B) para llegar a B=10.000 —a B=N el estimador da cero
por construcción, porque un subconjunto sin reposición del tamaño del total es
el total—. La extrapolación era el punto débil: la ley 1/raíz(B) es asintótica y
en la cola del 0,5%, con B=250, hay 1,25 observaciones esperadas.

Ahora se remuestrea CON REPOSICIÓN, que es el bootstrap estándar de la
variabilidad de un cuantil calculado sobre B sorteos independientes. No necesita
corrección por población finita, no necesita extrapolar, y sirve igual en B=N.
Codex hizo esa comprobación a mano antes que el script: para cadena perpetua dio
0,269 y 2,419 contra los 0,273 y 2,453 que daba la extrapolación.

QUÉ NO MIDE. La variabilidad de haber tomado otra muestra de la población: eso
es el bootstrap mismo, no su error de simulación. Tampoco el SESGO del cuantil
ni su error cuadrático total; sólo la dispersión.

Uso:
    python widgets/seguridad/scripts/error_mc_percentil.py
    python widgets/seguridad/scripts/error_mc_percentil.py --pregunta pena_muerte
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from widgets.seguridad import config
from widgets.seguridad.model import build_features, _interp

REPETICIONES = 200
SEMILLA = 42


def _perfiles():
    import itertools
    return [
        dict(tramo_edad=te, es_mujer=mu, nivel_educ=ed, ideologia=id_,
             victima=vi, es_montevideo=mv)
        for te, mu, ed, id_, vi, mv in itertools.product(
            sorted(set(config.EDAD_UI_TO_CODE.values())), (0, 1),
            sorted(set(config.EDUC_UI_TO_CODE.values())),
            sorted(set(config.IDEOLOGIA_UI_TO_CODE.values())),
            sorted(set(config.VICTIMA_UI_TO_CODE.values())),
            sorted(set(config.REGION_UI_TO_CODE.values())))
    ]


def probabilidades_por_replica(modelo, Xp):
    """Matriz réplicas x perfiles, ya con el mapa de calibración aplicado."""
    boot = modelo["bootstrap"]
    # `orden` es intercepto + los predictores en el orden serializado, que no
    # tiene por qué ser el de config.PREDICTORES.
    orden = boot["orden"][1:]
    Xp = Xp[:, [config.PREDICTORES.index(k) for k in orden]]
    coefs = np.array(boot["replicas"], dtype=float)
    Z = coefs[:, 0][:, None] + coefs[:, 1:] @ Xp.T
    P = 100.0 / (1.0 + np.exp(-Z))
    cal = modelo.get("calibracion")
    reps = (cal or {}).get("replicas") or []
    if cal:
        for i in range(P.shape[0]):
            xs, ys = (reps[i % len(reps)] if reps
                      else (cal["grilla"], cal["valores"]))
            P[i] = [_interp(xs, ys, v / 100.0) * 100 for v in P[i]]
    return P


def medir(slug, bes):
    ruta = config.ruta_modelo(slug)
    with open(ruta, encoding="utf-8") as f:
        modelo = json.load(f)
    perfiles = _perfiles()
    Xp = np.array([[build_features(**p)[k] for k in config.PREDICTORES]
                   for p in perfiles], dtype=float)
    P = probabilidades_por_replica(modelo, Xp)
    total = P.shape[0]
    nivel = modelo.get("nivel_calibrado", 95)
    cola = (100 - nivel) / 2

    rng = np.random.default_rng(SEMILLA)
    print(f"\n{slug}  (nivel {nivel}, {total} réplicas serializadas, "
          f"{len(perfiles)} perfiles, {REPETICIONES} repeticiones)")
    print(f"  {'B':>7} {'mediana':>9} {'p95':>7} {'máx':>7}   "
          f"(desvío estándar del extremo, en pp)")
    for B in bes:
        ext = np.empty((REPETICIONES, P.shape[1], 2))   # reps x perfiles x extremos
        for r in range(REPETICIONES):
            # CON REPOSICIÓN: ver el encabezado. Sin reposición hay que corregir
            # por población finita y aun así el estimador degenera en B=N.
            sub = P[rng.integers(0, total, size=B)]
            ext[r, :, 0] = np.percentile(sub, cola, axis=0)
            ext[r, :, 1] = np.percentile(sub, 100 - cola, axis=0)
        sd = ext.std(axis=0, ddof=1).ravel()
        marca = "   <- el que se publica" if B == total else ""
        print(f"  {B:>7} {np.median(sd):>8.3f} {np.percentile(sd, 95):>7.3f} "
              f"{sd.max():>7.3f}{marca}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pregunta", action="append", dest="preguntas")
    ap.add_argument("--bes", type=int, nargs="+",
                    default=[1000, 2500, 5000, 10000])
    args = ap.parse_args()
    for slug in (args.preguntas or config.SLUGS):
        if config.ruta_modelo(slug).exists():
            medir(slug, args.bes)


if __name__ == "__main__":
    main()

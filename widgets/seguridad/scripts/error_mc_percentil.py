"""
Cuánto ruido Monte Carlo tiene el extremo del intervalo según cuántas réplicas.

POR QUÉ EXISTE. `N_REPLICAS = 10000` se justificaba con una medición de Codex
que no quedó guardada en ninguna parte: ni el experimento ni el estimador ni el
universo de perfiles. Al revisar la recalibración, Codex intentó reproducirla y
le dieron otros números — no porque los primeros fueran falsos, sino porque no
había con qué comprobarlos. Una constante que gobierna el tamaño de los
artefactos publicados no puede apoyarse en un número irreproducible.

QUÉ MIDE. Para cada uno de los 1.008 perfiles de la UI y cada extremo del
intervalo al nivel que la pregunta publica: el desvío estándar del extremo
cuando se lo calcula con B réplicas en vez de con todas las que trae el
artefacto. Se estima remuestreando SIN reposición subconjuntos de tamaño B de
las réplicas serializadas, que es la variabilidad que uno se ahorra al subir B.

QUÉ NO MIDE. La variabilidad de haber tomado otra muestra de la población: eso
es el bootstrap mismo, no su error de simulación. Acá el objeto de estudio es el
ruido que agrega ESTIMAR el cuantil con pocas réplicas.

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
    medidos = {}
    for B in bes:
        # B = total daría cero por construcción: subconjuntos sin reposición del
        # tamaño del total son siempre el total. Para saber el ruido a B=total
        # harían falta bootstraps independientes; acá se extrapola.
        if B >= total:
            continue
        ext = np.empty((REPETICIONES, P.shape[1], 2))   # reps x perfiles x extremos
        for r in range(REPETICIONES):
            sub = P[rng.choice(total, size=B, replace=False)]
            ext[r, :, 0] = np.percentile(sub, cola, axis=0)
            ext[r, :, 1] = np.percentile(sub, 100 - cola, axis=0)
        # CORRECCIÓN POR POBLACIÓN FINITA. Los subconjuntos salen sin
        # reposición de las N réplicas serializadas, así que su dispersión está
        # achicada por raíz(1 - B/N): a B=5.000 sobre N=10.000 el desvío medido
        # es un 29% menor que el real. Sin esta corrección la ley 1/raíz(B) se
        # ve rota justo donde más se la necesita.
        sd = ext.std(axis=0, ddof=1).ravel() / np.sqrt(1.0 - B / total)
        medidos[B] = (float(np.median(sd)), float(np.percentile(sd, 95)),
                      float(sd.max()))
        print(f"  {B:>7} {medidos[B][0]:>8.3f} {medidos[B][1]:>7.3f} "
              f"{medidos[B][2]:>7.3f}")

    # ¿Vale la ley 1/raíz(B)? Se comprueba antes de usarla para extrapolar.
    if len(medidos) >= 2:
        bs = sorted(medidos)
        razones = [medidos[bs[0]][0] / medidos[b][0] * np.sqrt(bs[0] / b)
                   for b in bs[1:]]
        print(f"  ley 1/raíz(B): la mediana escala con factor "
              f"{np.mean(razones):.2f} (1,00 sería exacto)")
        base = bs[-1]
        f = np.sqrt(base / total)
        print(f"  {total:>7} {medidos[base][0]*f:>8.3f} "
              f"{medidos[base][1]*f:>7.3f} {medidos[base][2]*f:>7.3f}"
              f"   <- extrapolado desde B={base}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pregunta", action="append", dest="preguntas")
    ap.add_argument("--bes", type=int, nargs="+",
                    default=[250, 500, 1000, 2500, 5000])
    args = ap.parse_args()
    for slug in (args.preguntas or config.SLUGS):
        if config.ruta_modelo(slug).exists():
            medir(slug, args.bes)


if __name__ == "__main__":
    main()

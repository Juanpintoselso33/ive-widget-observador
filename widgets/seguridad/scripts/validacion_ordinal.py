"""
Validación interna (no producción): logit ordinal vs. el binario de producción.

El widget dicotomiza la escala Likert 1-5 —a favor si ≥4, en contra si ≤2— y
descarta los neutrales, que en pena de muerte son 671 casos, un 20% de la
muestra. Eso es una decisión fuerte: si al usar la escala completa la estructura
de asociaciones se diera vuelta, el widget estaría mostrando un ordenamiento que
el dato no sostiene.

Este script ajusta un logit ordinal (proportional odds) sobre los cinco puntos,
sin excluir a nadie, y compara los coeficientes con los del modelo binario. Es
el equivalente de scripts/validacion_ordinal.py del widget IVE, donde la misma
verificación mostró 17 de 19 signos coincidentes y los top-5 en idéntico orden.

Caveat heredado del IVE: statsmodels OrderedModel no acepta sample_weight, así
que la validación corre SIN ponderar. Sirve para chequear la estructura, no las
magnitudes.

Uso:
    python widgets/seguridad/scripts/validacion_ordinal.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from scipy.stats import spearmanr
from statsmodels.miscmodels.ordinal_model import OrderedModel

from widgets.seguridad.config import DATA_FILE, PREDICTORES, PREGUNTA
from widgets.seguridad import train_model as tm
from widgets.seguridad.model import load_model


def main():
    df = tm.preparar(pd.read_csv(DATA_FILE, encoding="utf-8-sig"))

    # --- Ordinal sobre la escala completa, sin excluir neutrales -------------
    d = df[df["likert"].notna()].copy()
    X = d[PREDICTORES]
    y = d["likert"].astype(int)
    print(f"Pregunta: {PREGUNTA['columna']}")
    print(f"Ordinal sobre la escala completa: n={len(d)} "
          f"(el binario usa {int(df['a_favor'].notna().sum())})")
    print(f"Distribución Likert: {dict(sorted(y.value_counts().items()))}\n")

    modelo = OrderedModel(y, X, distr="logit").fit(method="bfgs", disp=False)
    coef_ord = {p: modelo.params[p] for p in PREDICTORES}

    # --- Binario de producción ----------------------------------------------
    coef_bin = load_model()["coefficients"]

    # --- Comparación ---------------------------------------------------------
    # En el ordinal, la escala corre de "totalmente en desacuerdo" a
    # "totalmente de acuerdo", así que un coeficiente positivo empuja hacia el
    # acuerdo: misma dirección que el binario. Los signos son comparables
    # directamente.
    filas = []
    for p in PREDICTORES:
        b, o = coef_bin[p], coef_ord[p]
        filas.append((p, b, o, (b > 0) == (o > 0)))

    iguales = sum(1 for _, _, _, ok in filas if ok)
    print(f"{'predictor':24s} {'binario':>10s} {'ordinal':>10s}  signo")
    for p, b, o, ok in sorted(filas, key=lambda r: -abs(r[1])):
        print(f"{p:24s} {b:+10.3f} {o:+10.3f}  {'ok' if ok else 'DIFIERE'}")

    print(f"\nSignos coincidentes: {iguales}/{len(filas)}")

    orden_bin = [p for p, _, _, _ in sorted(filas, key=lambda r: -abs(r[1]))]
    orden_ord = [p for p, _, _, _ in sorted(filas, key=lambda r: -abs(r[2]))]
    rho, _ = spearmanr(
        [orden_bin.index(p) for p in PREDICTORES],
        [orden_ord.index(p) for p in PREDICTORES],
    )
    print(f"Spearman entre rankings de magnitud: {rho:.3f}")
    top5_bin, top5_ord = orden_bin[:5], orden_ord[:5]
    print(f"Top-5 binario: {top5_bin}")
    print(f"Top-5 ordinal: {top5_ord}")
    print(f"Coinciden en el top-5 (como conjunto): "
          f"{len(set(top5_bin) & set(top5_ord))}/5")

    if iguales == len(filas) and rho > 0.7:
        print("\nCONCLUSION: la dicotomización no altera la estructura. "
              "El binario es defendible para la pregunta del widget.")
    else:
        print("\nCONCLUSION: hay discrepancias. Revisar antes de publicar "
              "afirmaciones sobre el ordenamiento de los factores.")


if __name__ == "__main__":
    main()

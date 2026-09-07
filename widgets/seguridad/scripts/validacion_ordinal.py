"""
Validación interna: logit ordinal vs. el binario de producción.

El widget dicotomiza la escala Likert 1-5 —a favor si ≥4, en contra si ≤2— y
descarta los neutrales, que según la pregunta son entre el 12% y el 23% de la
muestra. Eso es una decisión fuerte: si al usar la escala completa la estructura
de asociaciones se diera vuelta, el widget estaría mostrando un ordenamiento que
el dato no sostiene.

Este script ajusta un logit ordinal (proportional odds) sobre los cinco puntos,
sin excluir a nadie, compara los coeficientes con los del modelo binario y
**escribe el resultado dentro del JSON de esa pregunta**, bajo la clave
`robustez`. La sección "Cómo se calcula" del widget redacta su párrafo de "qué
sostiene y qué no" desde ahí.

Por qué se guarda en el JSON y no en un informe aparte: ese párrafo antes tenía
las conclusiones escritas a mano en components.py, valían para una sola pregunta
y para un corte ideológico que después cambió, y nada las volvía a chequear.
Ahora, si no se corrió esta validación, el widget lo DICE en vez de afirmar algo
no verificado.

Correr esto DESPUÉS de train_model.py: re-entrenar pisa el JSON y se lleva la
clave `robustez` puesta, que es lo correcto —los coeficientes cambiaron, la
validación vieja ya no habla de ellos—.

Caveat heredado del IVE: statsmodels OrderedModel no acepta sample_weight, así
que la validación corre SIN ponderar. Sirve para chequear la estructura, no las
magnitudes.

Uso:
    python widgets/seguridad/scripts/validacion_ordinal.py
    python widgets/seguridad/scripts/validacion_ordinal.py --pregunta pena_muerte
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
from datetime import date

import pandas as pd
from scipy.stats import spearmanr
from statsmodels.miscmodels.ordinal_model import OrderedModel

from widgets.seguridad.config import (
    DATA_FILE, PREDICTORES, PREGUNTAS, SLUGS, ruta_modelo,
)
from widgets.seguridad import train_model as tm
from widgets.seguridad.model import load_model

# Cuántos de los efectos más grandes del binario se nombran como "los que
# aguantan". Cinco es lo que entra en una oración sin volverse un inventario.
TOP = 5


def validar(df_crudo, slug):
    """Corre la comparación de una pregunta y devuelve el bloque `robustez`."""
    pregunta = PREGUNTAS[slug]
    print(f"\n{'=' * 64}")
    print(f"Pregunta: {slug} → {pregunta['columna']}")
    print("=" * 64)

    df = tm.preparar(df_crudo, pregunta)

    # --- Ordinal sobre la escala completa, sin excluir neutrales -------------
    d = df[df["likert"].notna()].copy()
    X = d[PREDICTORES]
    y = d["likert"].astype(int)
    print(f"Ordinal sobre la escala completa: n={len(d)} "
          f"(el binario usa {int(df['a_favor'].notna().sum())})")
    print(f"Distribución Likert: {dict(sorted(y.value_counts().items()))}\n")

    modelo = OrderedModel(y, X, distr="logit").fit(method="bfgs", disp=False)
    coef_ord = {p: float(modelo.params[p]) for p in PREDICTORES}

    # --- Binario de producción ----------------------------------------------
    coef_bin = load_model(slug)["coefficients"]

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

    no_sostienen = [p for p, _, _, ok in filas if not ok]
    coincide = {p for p, _, _, ok in filas if ok}
    # Los más grandes del binario que además mantienen el signo: son los que la
    # metodología puede nombrar como "aguantan". Un efecto grande que cambia de
    # signo NO entra acá aunque sea de los mayores.
    sostienen_top = [p for p in orden_bin if p in coincide][:TOP]

    print(f"Aguantan (top {TOP} por magnitud): {sostienen_top}")
    print(f"Cambian de signo: {no_sostienen or 'ninguno'}")

    return {
        "n_ordinal": int(len(d)),
        "n_binario": int(df["a_favor"].notna().sum()),
        "n_predictores": len(filas),
        "signos_coincidentes": int(iguales),
        "spearman_magnitud": round(float(rho), 3),
        "sostienen_top": sostienen_top,
        "no_sostienen": no_sostienen,
        "ponderado": False,
        "validado": date.today().isoformat(),
    }


def guardar(slug, robustez):
    """Escribe el bloque `robustez` dentro del JSON de la pregunta."""
    ruta = ruta_modelo(slug)
    with open(ruta, encoding="utf-8") as f:
        modelo = json.load(f)

    # El JSON tiene que ser el de ESTA pregunta: si alguien corre la validación
    # contra un archivo movido, mejor abortar que pegarle un bloque de robustez
    # que habla de otros coeficientes.
    if modelo.get("pregunta_slug") != slug:
        raise SystemExit(
            f"{ruta} dice ser de «{modelo.get('pregunta_slug')}», no de «{slug}». "
            "No se escribe nada."
        )

    modelo["robustez"] = robustez
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(modelo, f, ensure_ascii=False, indent=2)
    print(f"Escrito en {ruta} (clave `robustez`)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pregunta", choices=SLUGS, action="append", dest="preguntas",
        help="validar sólo esta pregunta (se puede repetir). Por defecto, las cuatro.",
    )
    args = parser.parse_args()
    slugs = args.preguntas or SLUGS

    df_crudo = pd.read_csv(DATA_FILE, encoding="utf-8-sig")

    resumen = []
    for slug in slugs:
        robustez = validar(df_crudo, slug)
        guardar(slug, robustez)
        resumen.append((slug, robustez))

    print(f"\n{'=' * 64}")
    print("RESUMEN")
    print("=" * 64)
    for slug, r in resumen:
        veredicto = (
            "estructura estable"
            if not r["no_sostienen"] and r["spearman_magnitud"] > 0.7
            else "hay efectos que no aguantan"
        )
        print(f"  {slug:22s} {r['signos_coincidentes']}/{r['n_predictores']} signos, "
              f"rho={r['spearman_magnitud']:+.3f} — {veredicto}")


if __name__ == "__main__":
    main()

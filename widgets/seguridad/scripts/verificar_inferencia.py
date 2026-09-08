"""
Verifica que la inferencia de producción coincida con sklearn.

`model.py` no importa sklearn: reconstruye el vector de dummies a mano y evalúa
la logística con aritmética de Python. Eso es bueno para producción —la app pesa
menos y no depende de la versión de sklearn— pero abre un modo de falla que
ningún test unitario agarra: que `build_features()` arme el vector en un orden
distinto del que tenía la matriz de entrenamiento. Los coeficientes se aplicarían
a la dummy equivocada y el resultado sería un número plausible y falso, sin un
solo error en pantalla.

Este script cierra esa brecha: re-ajusta el modelo con sklearn sobre la misma
base y el mismo `C`, evalúa los 1.008 perfiles que el lector puede elegir por los
dos caminos, y compara.

Corre fuera de los tests porque necesita sklearn y la base del cliente, que no
está en el repo. Es una verificación de publicación, no de CI.

Última corrida: 7/9/2026, las cuatro preguntas, peor discrepancia 0,0000000000 pp.

Uso:
    python widgets/seguridad/scripts/verificar_inferencia.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import itertools
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from widgets.seguridad import config, train_model as tm
from widgets.seguridad.model import (
    build_features, predict_probability, _calibrar, _sigmoid_pct, _z,
)

# Tolerancia. No es "cero" a secas porque las dos vías hacen las mismas cuentas
# en distinto orden y el punto flotante no está obligado a coincidir bit a bit;
# 1e-9 pp es varios órdenes de magnitud más chico que cualquier diferencia que
# provenga de un vector mal armado, que se mide en puntos porcentuales enteros.
TOLERANCIA_PP = 1e-9


def perfiles_de_la_ui():
    """Las combinaciones que el selector puede producir."""
    return [
        dict(tramo_edad=te, es_mujer=mu, nivel_educ=ed, ideologia=id_,
             victima=vi, es_montevideo=mv)
        for te, mu, ed, id_, vi, mv in itertools.product(
            sorted(set(config.EDAD_UI_TO_CODE.values())),
            (0, 1),
            sorted(set(config.EDUC_UI_TO_CODE.values())),
            sorted(set(config.IDEOLOGIA_UI_TO_CODE.values())),
            sorted(set(config.VICTIMA_UI_TO_CODE.values())),
            sorted(set(config.REGION_UI_TO_CODE.values())),
        )
    ]


def main():
    df_crudo = pd.read_csv(config.DATA_FILE, encoding="utf-8-sig")
    perfiles = perfiles_de_la_ui()
    print(f"Perfiles elegibles en la UI: {len(perfiles)}\n")

    peor_global = 0.0
    fallo = False
    for slug in config.SLUGS:
        ruta = config.ruta_modelo(slug)
        if not ruta.exists():
            print(f"  {slug:22s} sin entrenar, se saltea")
            continue

        with open(ruta, encoding="utf-8") as f:
            modelo = json.load(f)

        df = tm.preparar(df_crudo, config.PREGUNTAS[slug])
        d = df[df["a_favor"].notna()]
        X = d[config.PREDICTORES].values
        y = d["a_favor"].values.astype(int)
        w = d[config.PONDERADOR].values

        sk = LogisticRegression(C=modelo["model_info"]["C"], max_iter=2000,
                                random_state=tm.RANDOM_STATE)
        sk.fit(X, y, sample_weight=w)

        # El MISMO vector por los dos caminos: build_features() es el de
        # producción, y el orden de columnas es el de PREDICTORES, que es el que
        # usó el entrenamiento.
        Xp = np.array([[build_features(**p)[k] for k in config.PREDICTORES]
                       for p in perfiles])
        # Se comparan LAS DOS ESCALAS, cruda y calibrada, y no sólo la
        # calibrada. Razón: el mapa tiene mesetas —85% y 89% crudos caen los dos
        # en 87,58%— así que comparar sólo después del mapa PIERDE información.
        # Codex lo midió intercambiando dos dummies de edad: el error global se
        # seguía detectando, pero 78 perfiles con discrepancia cruda quedaban
        # indistinguibles. La comparación cruda es la que de verdad chequea que
        # el vector de features y el orden de los coeficientes coincidan.
        crudo_sklearn = sk.predict_proba(Xp)[:, 1] * 100
        crudo_produccion = np.array(
            [_sigmoid_pct(_z(modelo["coefficients"], build_features(**p)))
             for p in perfiles])
        por_sklearn = np.array([_calibrar(modelo, v) for v in crudo_sklearn])
        por_produccion = np.array([predict_probability(modelo, **p) for p in perfiles])

        peor_crudo = float(np.abs(crudo_sklearn - crudo_produccion).max())
        peor_global = max(peor_global, peor_crudo)
        if peor_crudo > TOLERANCIA_PP:
            fallo = True
            print(f"  {slug:22s} EN ESCALA CRUDA discrepa {peor_crudo:.10f} pp")
        if modelo.get("calibracion"):
            print(f"  {slug:22s} recalibrada — se comparan las dos escalas; "
                  f"cruda: {peor_crudo:.10f} pp")

        peor = float(np.abs(por_sklearn - por_produccion).max())
        peor_global = max(peor_global, peor)
        estado = "ok" if peor <= TOLERANCIA_PP else "DISCREPA"
        if peor > TOLERANCIA_PP:
            fallo = True
            i = int(np.abs(por_sklearn - por_produccion).argmax())
            print(f"  {slug:22s} peor = {peor:.10f} pp  {estado}")
            print(f"      perfil: {perfiles[i]}")
            print(f"      sklearn={por_sklearn[i]:.6f}  producción={por_produccion[i]:.6f}")
        else:
            print(f"  {slug:22s} peor = {peor:.10f} pp  {estado}")

    print(f"\nPeor discrepancia global: {peor_global:.10f} pp")
    if fallo:
        raise SystemExit(
            "La inferencia de producción NO coincide con sklearn. Casi siempre "
            "es que build_features() y PREDICTORES quedaron desalineados."
        )
    print("La inferencia de producción coincide con sklearn.")


if __name__ == "__main__":
    main()
    print("\nCORRESPONDENCIA DE CATEGORÍAS (referencia independiente)")
    verificar_correspondencia_categorias()


# ============================================================
# REFERENCIA INDEPENDIENTE: ¿las categorías de la UI son las del entrenamiento?
# ============================================================
def verificar_correspondencia_categorias():
    """
    Comprueba que un perfil elegido en la UI produzca EL MISMO vector de dummies
    que produce el pipeline de entrenamiento para un encuestado con esos mismos
    atributos crudos.

    POR QUÉ HACE FALTA, y es el agujero que marcó Codex: `main()` compara dos
    caminos que AMBOS llaman a `build_features`. Le invirtió a mano las dummies
    de edad en los dos lados y el script terminó conforme. Detecta que los
    coeficientes estén desalineados; NO detecta que "30-44 años" en el selector
    se traduzca a la dummy de otro tramo.

    Acá la referencia es independiente: sale de `train_model.preparar()`, que es
    la que construyó la matriz con la que se estimaron los coeficientes. Si la UI
    y el entrenamiento discrepan en qué significa una categoría, salta.
    """
    df = tm.preparar(pd.read_csv(config.DATA_FILE, encoding="utf-8-sig"),
                     config.PREGUNTAS[config.PREGUNTA_DEFECTO])
    # Sólo encuestados que corresponden a un perfil REALMENTE elegible: los que
    # tienen alguna dummy oculta encendida no se pueden expresar desde la UI.
    d = df[(df["victima_sin_dato"] == 0) & (df["ideol_no_ubica"] == 0)
           & df["tramo_edad"].notna()]

    ideol_de = {}
    for i, (nombre, desde, hasta, _) in enumerate(config.ESPEC_CRUDA["ideol_tramos"], start=1):
        for v in range(desde, hasta + 1):
            ideol_de[v] = i
    educ = config.ESPEC_CRUDA["educ_colapso"]
    col_ideol = "var_242 | Autoubicacion izquierda-derecha (0-10)"

    fallas = 0
    for _, fila in d.iterrows():
        # Del dato CRUDO a los códigos que produciría la UI.
        entrada = dict(
            tramo_edad=int(fila["tramo_edad"]),
            es_mujer=int(fila["es_mujer"]),
            nivel_educ=educ[int(fila["nivel_educativo"])],
            ideologia=ideol_de[int(fila[col_ideol])],
            victima=(3 if fila["victima_con_violencia"] else
                     2 if fila["victima_sin_violencia"] else 1),
            es_montevideo=int(fila["es_montevideo"]),
        )
        desde_ui = build_features(**entrada)
        for nombre in config.PREDICTORES:
            if desde_ui[nombre] != int(fila[nombre]):
                fallas += 1
                if fallas <= 3:
                    print(f"  DISCREPA en '{nombre}': la UI dice {desde_ui[nombre]} "
                          f"y el entrenamiento {int(fila[nombre])} — {entrada}")
                break

    print(f"  encuestados comprobados: {len(d)}   con discrepancia: {fallas}")
    if fallas:
        raise SystemExit(
            "La traducción de categorías de la UI NO coincide con la del "
            "entrenamiento: el widget aplicaría coeficientes de otra categoría."
        )
    print("  la UI y el entrenamiento entienden lo mismo por cada categoría.")

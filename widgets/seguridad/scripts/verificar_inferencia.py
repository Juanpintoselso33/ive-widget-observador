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
from widgets.seguridad.model import build_features, predict_probability, _calibrar

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
        # Al crudo de sklearn se le aplica EL MISMO mapa de recalibración que
        # usa producción. Sin esto, una pregunta recalibrada da una discrepancia
        # enorme —9,6 pp cuando se implementó— y el script la reporta como
        # desalineación de dummies, que es justo el error que viene a detectar.
        # Lo que se compara sigue siendo lo que importa: que el vector de
        # features y el orden de los coeficientes coincidan.
        crudo = sk.predict_proba(Xp)[:, 1] * 100
        por_sklearn = np.array([_calibrar(modelo, v) for v in crudo])
        por_produccion = np.array([predict_probability(modelo, **p) for p in perfiles])
        if modelo.get("calibracion"):
            print(f"  {slug:22s} (recalibrada: se compara contra sklearn + el mapa)")

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

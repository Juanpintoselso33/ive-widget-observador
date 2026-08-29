"""
Lógica de predicción del widget de seguridad.

Python puro (sin Streamlit ni sklearn) para que sea testeable y liviano en
producción: lee el JSON de coeficientes y evalúa la logística.

A diferencia de widgets/ive/model.py, que suma los términos uno por uno, acá
el vector de features se arma en un dict y la suma se hace iterando sobre los
predictores declarados en config.PREDICTORES. Agregar una variable al modelo
es tocar build_features() y la lista, no la aritmética.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import math

from widgets.seguridad.config import MODEL_COEFFICIENTS_PATH, PREDICTORES


def load_model():
    """Carga los coeficientes del modelo desde JSON."""
    with open(MODEL_COEFFICIENTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_features(tramo_edad, es_mujer, nivel_educ, ideologia, victima, es_montevideo):
    """
    Traduce los inputs de la UI (códigos de config.py) al vector de dummies.

    Cada bloque omite su categoría de referencia: 18-29 en edad, hombre en
    sexo, primaria o menos en educación, centro en ideología, no víctima, e
    interior en región.
    """
    return {
        "edad_30_44": int(tramo_edad == 2),
        "edad_45_59": int(tramo_edad == 3),
        "edad_60_plus": int(tramo_edad == 4),
        "es_mujer": int(es_mujer),
        "educ_secundaria": int(nivel_educ == 2),
        "educ_ter_incomp": int(nivel_educ == 3),
        "educ_ter_comp": int(nivel_educ == 4),
        "ideol_izquierda": int(ideologia == 1),
        "ideol_derecha": int(ideologia == 3),
        "ideol_no_ubica": int(ideologia == 4),
        "victima_sin_violencia": int(victima == 2),
        "victima_con_violencia": int(victima == 3),
        # Siempre 0: la UI obliga a elegir una de las tres opciones reales.
        # El coeficiente existe para que los sin dato del entrenamiento no
        # contaminen la categoría de referencia (ver config.PREDICTORES).
        "victima_sin_dato": 0,
        "es_montevideo": int(es_montevideo),
    }


def _z(coef, features):
    """Suma el intercepto más los términos declarados en PREDICTORES."""
    z = coef["intercept"]
    for nombre in PREDICTORES:
        z += coef[nombre] * features[nombre]
    return z


def _sigmoid_pct(z):
    return (1 / (1 + math.exp(-z))) * 100


def predict_probability(model, tramo_edad, es_mujer, nivel_educ, ideologia,
                        victima, es_montevideo):
    """
    Probabilidad de estar a favor, condicional a tener postura definida.
    Returns: float en 0-100.
    """
    features = build_features(tramo_edad, es_mujer, nivel_educ, ideologia,
                              victima, es_montevideo)
    return _sigmoid_pct(_z(model["coefficients"], features))


def predict_probability_neutral(model, tramo_edad, es_mujer, nivel_educ, ideologia,
                                victima, es_montevideo):
    """
    Probabilidad de no fijar postura (Likert=3 o sin respuesta) según el perfil.
    Returns: float en 0-100.
    """
    features = build_features(tramo_edad, es_mujer, nivel_educ, ideologia,
                              victima, es_montevideo)
    return _sigmoid_pct(_z(model["coefficients_neutral"], features))

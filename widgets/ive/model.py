"""
Lógica de predicción del modelo IVE.
Carga de coeficientes y cálculo de probabilidad via regresión logística Ridge.

Este módulo es puro Python (sin dependencia de Streamlit) para facilitar testing.
El caching con @st.cache_data se aplica en app.py.

Modelo v2: dummies completas + interacciones (sin variables ordinales lineales).
Incluye un modelo secundario de neutralidad (P(NS-NC)) sobre los mismos predictores.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import math

from widgets.ive.config import MODEL_COEFFICIENTS_PATH


def load_model():
    """Carga los coeficientes del modelo desde JSON."""
    with open(MODEL_COEFFICIENTS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def _z_from_inputs(coef, tramo_edad, es_mujer, nivel_educ, religiosidad,
                   es_montevideo, tiene_hijos, hogar, balotaje):
    edad_25_34 = 1 if tramo_edad == 2 else 0
    edad_35_44 = 1 if tramo_edad == 3 else 0
    edad_45_54 = 1 if tramo_edad == 4 else 0
    edad_55_plus = 1 if tramo_edad == 5 else 0

    educ_secundaria = 1 if nivel_educ == 2 else 0
    educ_ter_incomp = 1 if nivel_educ == 3 else 0
    educ_ter_comp = 1 if nivel_educ == 4 else 0

    relig_poco = 1 if religiosidad == 2 else 0
    relig_bastante = 1 if religiosidad == 3 else 0
    relig_mucho = 1 if religiosidad == 4 else 0

    hogar_3_4 = 1 if hogar == 2 else 0
    hogar_5_plus = 1 if hogar == 3 else 0

    balotaje_martinez = 1 if balotaje == "martinez" else 0
    balotaje_lacalle = 1 if balotaje == "lacalle" else 0

    mujer_x_relig_mucho = es_mujer * relig_mucho
    mujer_x_tiene_hijos = es_mujer * tiene_hijos

    z = coef['intercept']
    z += coef['edad_25_34'] * edad_25_34
    z += coef['edad_35_44'] * edad_35_44
    z += coef['edad_45_54'] * edad_45_54
    z += coef['edad_55_plus'] * edad_55_plus
    z += coef['es_mujer'] * es_mujer
    z += coef['educ_secundaria'] * educ_secundaria
    z += coef['educ_ter_incomp'] * educ_ter_incomp
    z += coef['educ_ter_comp'] * educ_ter_comp
    z += coef['relig_poco'] * relig_poco
    z += coef['relig_bastante'] * relig_bastante
    z += coef['relig_mucho'] * relig_mucho
    z += coef['es_montevideo'] * es_montevideo
    z += coef['tiene_hijos'] * tiene_hijos
    z += coef['hogar_3_4'] * hogar_3_4
    z += coef['hogar_5_plus'] * hogar_5_plus
    z += coef['balotaje_martinez'] * balotaje_martinez
    z += coef['balotaje_lacalle'] * balotaje_lacalle
    z += coef['mujer_x_relig_mucho'] * mujer_x_relig_mucho
    z += coef['mujer_x_tiene_hijos'] * mujer_x_tiene_hijos

    return z


def predict_probability(model, tramo_edad, es_mujer, nivel_educ, religiosidad,
                        es_montevideo, tiene_hijos, hogar, balotaje):
    """
    Calcula la probabilidad de apoyar el IVE (condicional a tener postura definida).
    Returns: float en 0-100.
    """
    z = _z_from_inputs(
        model['coefficients'],
        tramo_edad, es_mujer, nivel_educ, religiosidad,
        es_montevideo, tiene_hijos, hogar, balotaje,
    )
    return (1 / (1 + math.exp(-z))) * 100


def predict_probability_neutral(model, tramo_edad, es_mujer, nivel_educ, religiosidad,
                                es_montevideo, tiene_hijos, hogar, balotaje):
    """
    Calcula la probabilidad de no fijar postura (Likert=3 o NS-NC) según el perfil.
    Returns: float en 0-100.
    """
    z = _z_from_inputs(
        model['coefficients_neutral'],
        tramo_edad, es_mujer, nivel_educ, religiosidad,
        es_montevideo, tiene_hijos, hogar, balotaje,
    )
    return (1 / (1 + math.exp(-z))) * 100

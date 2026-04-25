"""
Lógica de predicción del modelo IVE.
Carga de coeficientes y cálculo de probabilidad via regresión logística Ridge.

Este módulo es puro Python (sin dependencia de Streamlit) para facilitar testing.
El caching con @st.cache_data se aplica en app.py.

Modelo v2: dummies completas + interacciones (sin variables ordinales lineales).
Incluye un modelo secundario de neutralidad (P(NS-NC)) sobre los mismos predictores.
"""

import json
import math

from config import MODEL_COEFFICIENTS_PATH


def load_model():
    """Carga los coeficientes del modelo desde JSON."""
    with open(MODEL_COEFFICIENTS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def _z_from_inputs(coef, tramo_edad, es_mujer, nivel_educ, religiosidad,
                   es_montevideo, tiene_hijos, hogar, balotaje):
    """
    Construye el vector de dummies + interacciones desde los 8 inputs del usuario
    y calcula el logit z = intercept + sum(coef_i * x_i).

    `coef` es un dict con las mismas claves que PREDICTORS + 'intercept'.
    Se usa tanto para el modelo principal de apoyo al IVE como para el modelo
    secundario de neutralidad (mismas variables, distintos coeficientes).
    """
    edad_25_34 = 1 if tramo_edad == 2 else 0
    edad_35_44 = 1 if tramo_edad == 3 else 0
    edad_45_54 = 1 if tramo_edad == 4 else 0
    edad_55_plus = 1 if tramo_edad == 5 else 0

    educ_cb = 1 if nivel_educ == 2 else 0
    educ_bach_incomp = 1 if nivel_educ == 3 else 0
    educ_bach_comp = 1 if nivel_educ == 4 else 0
    educ_ter_incomp = 1 if nivel_educ == 5 else 0
    educ_ter_comp = 1 if nivel_educ == 6 else 0

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
    z += coef['educ_cb'] * educ_cb
    z += coef['educ_bach_incomp'] * educ_bach_incomp
    z += coef['educ_bach_comp'] * educ_bach_comp
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

    P(Y=1 | con_postura) = 1 / (1 + exp(-z))

    Modelo v2: variables ordinales como dummies, con interacciones.
    Referencias: edad=18-24, educación=primaria, religiosidad=nada, hogar=1-2, balotaje=otros.

    Returns:
        float: Probabilidad como porcentaje (0-100).
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

    Reportada en la UI como dato secundario para hacer explícita la condicionalidad
    del modelo principal: el % de IVE es entre los que tienen postura definida.

    Returns:
        float: Probabilidad como porcentaje (0-100).
    """
    z = _z_from_inputs(
        model['coefficients_neutral'],
        tramo_edad, es_mujer, nivel_educ, religiosidad,
        es_montevideo, tiene_hijos, hogar, balotaje,
    )
    return (1 / (1 + math.exp(-z))) * 100

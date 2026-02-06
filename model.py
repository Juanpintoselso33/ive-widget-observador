"""
Lógica de predicción del modelo IVE.
Carga de coeficientes y cálculo de probabilidad via regresión logística Ridge.

Este módulo es puro Python (sin dependencia de Streamlit) para facilitar testing.
El caching con @st.cache_data se aplica en app.py.

Modelo v2: dummies completas + interacciones (sin variables ordinales lineales).
"""

import json
import math

from config import MODEL_COEFFICIENTS_PATH


def load_model():
    """Carga los coeficientes del modelo desde JSON."""
    with open(MODEL_COEFFICIENTS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def predict_probability(model, tramo_edad, es_mujer, nivel_educ, religiosidad,
                        es_montevideo, tiene_hijos, hogar, balotaje):
    """
    Calcula la probabilidad de apoyar el IVE usando regresión logística Ridge.

    P(Y=1) = 1 / (1 + exp(-z))
    donde z = intercept + sum(coef_i * x_i)

    Modelo v2: variables ordinales como dummies (no lineales), con interacciones.
    Referencias: edad=18-24, educación=primaria, religiosidad=nada, hogar=1-2, balotaje=otros.

    Args:
        model: Dict con clave 'coefficients' conteniendo los coeficientes.
        tramo_edad: int (1-5) tramo de edad (1=18-24, 2=25-34, 3=35-44, 4=45-54, 5=55+).
        es_mujer: int (0/1).
        nivel_educ: int (1-5) nivel educativo (1=primaria, 2=ems_incomp, 3=ems_comp, 4=ter_incomp, 5=ter_comp).
        religiosidad: int (1-4) religiosidad (1=nada, 2=poco, 3=bastante, 4=mucho).
        es_montevideo: int (0/1).
        tiene_hijos: int (0/1).
        hogar: int (1-3) personas en hogar (1=1-2, 2=3-4, 3=5+).
        balotaje: str código de balotaje ("martinez", "lacalle", "otros").

    Returns:
        float: Probabilidad como porcentaje (0-100).
    """
    coef = model['coefficients']

    # Edad dummies (ref: 18-24 = tramo 1)
    edad_25_34 = 1 if tramo_edad == 2 else 0
    edad_35_44 = 1 if tramo_edad == 3 else 0
    edad_45_54 = 1 if tramo_edad == 4 else 0
    edad_55_plus = 1 if tramo_edad == 5 else 0

    # Educación dummies (ref: primaria = nivel 1)
    educ_ems_incomp = 1 if nivel_educ == 2 else 0
    educ_ems_comp = 1 if nivel_educ == 3 else 0
    educ_ter_incomp = 1 if nivel_educ == 4 else 0
    educ_ter_comp = 1 if nivel_educ == 5 else 0

    # Religiosidad dummies (ref: nada = 1)
    relig_poco = 1 if religiosidad == 2 else 0
    relig_bastante = 1 if religiosidad == 3 else 0
    relig_mucho = 1 if religiosidad == 4 else 0

    # Hogar dummies (ref: 1-2 = hogar 1)
    hogar_3_4 = 1 if hogar == 2 else 0
    hogar_5_plus = 1 if hogar == 3 else 0

    # Balotaje dummies (ref: otros/no votó/blanco)
    balotaje_martinez = 1 if balotaje == "martinez" else 0
    balotaje_lacalle = 1 if balotaje == "lacalle" else 0

    # Interacciones
    mujer_x_relig_mucho = es_mujer * relig_mucho
    mujer_x_tiene_hijos = es_mujer * tiene_hijos

    z = coef['intercept']
    z += coef['edad_25_34'] * edad_25_34
    z += coef['edad_35_44'] * edad_35_44
    z += coef['edad_45_54'] * edad_45_54
    z += coef['edad_55_plus'] * edad_55_plus
    z += coef['es_mujer'] * es_mujer
    z += coef['educ_ems_incomp'] * educ_ems_incomp
    z += coef['educ_ems_comp'] * educ_ems_comp
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

    probability = 1 / (1 + math.exp(-z))
    return probability * 100

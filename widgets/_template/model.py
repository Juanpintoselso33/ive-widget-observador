"""
Modelo del widget [NOMBRE].
Adaptar para el dataset específico.
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import math
from widgets._template.config import MODEL_COEFFICIENTS_PATH


def load_model():
    with open(MODEL_COEFFICIENTS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def predict_probability(model, *inputs):
    """
    Calcula la probabilidad del outcome para el perfil dado.
    Adaptar la construcción de dummies según los predictores del modelo.
    Returns: float en 0-100.
    """
    coef = model['coefficients']
    z = coef['intercept']
    # TODO: construir dummies y sumar coeficientes
    # z += coef['var_x'] * x_value
    return (1 / (1 + math.exp(-z))) * 100

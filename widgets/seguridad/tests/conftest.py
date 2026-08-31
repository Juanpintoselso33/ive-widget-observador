"""
Fixtures del widget de seguridad.

Los tests corren contra coeficientes SINTÉTICOS, no contra el JSON de
producción: así verifican la aritmética y el armado del vector de features sin
romperse cada vez que se re-entrena el modelo o se cambia de pregunta.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest

from widgets.seguridad.config import PREDICTORES


@pytest.fixture
def coef_neutros():
    """Todos los coeficientes en cero: la probabilidad debe dar 50%."""
    return {"intercept": 0.0, **{p: 0.0 for p in PREDICTORES}}


@pytest.fixture
def modelo_sintetico(coef_neutros):
    """
    Modelo con un solo efecto activo (es_mujer = +1 en la escala logit) para
    poder verificar que el término entra donde corresponde y no en otro lado.
    """
    coef = dict(coef_neutros)
    coef["es_mujer"] = 1.0
    return {
        "coefficients": coef,
        "coefficients_neutral": dict(coef_neutros),
        "prob_favor_nacional": 36.7,
        "stats_by_group": {"hombres": 40.0, "mujeres": 33.0, "chico": None},
        "model_info": {"n": 100, "ponderador": "w_norm"},
    }


@pytest.fixture
def perfil_base():
    """Perfil en todas las categorías de referencia."""
    return dict(
        tramo_edad=1,      # 18-29
        es_mujer=0,        # hombre
        nivel_educ=1,      # secundaria o menos
        ideologia=4,       # Centro (el 5 de la escala): la referencia
        victima=1,         # no fue víctima
        es_montevideo=0,   # interior
    )

"""
Fixtures compartidos para los tests del widget IVE.
Usa datos sintéticos -- NUNCA el model_coefficients.json de producción.

Modelo v2: dummies completas + interacciones.
"""

import sys
from pathlib import Path

import pytest

# Agregar ive_widget al path para que los imports funcionen como en Streamlit
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def synthetic_model():
    """Modelo sintético con coeficientes simplificados para testing."""
    return {
        "coefficients": {
            "intercept": 0.0,
            # Edad (ref: 18-24)
            "edad_25_34": 0.1,
            "edad_35_44": 0.2,
            "edad_45_54": -0.1,
            "edad_55_plus": 0.3,
            # Género
            "es_mujer": 1.0,
            # Educación (ref: primaria)
            "educ_ems_incomp": 0.1,
            "educ_ems_comp": 0.2,
            "educ_ter_incomp": 0.3,
            "educ_ter_comp": 0.4,
            # Religiosidad (ref: nada)
            "relig_poco": -0.3,
            "relig_bastante": -0.8,
            "relig_mucho": -1.5,
            # Región
            "es_montevideo": 0.5,
            # Hijos
            "tiene_hijos": -0.5,
            # Personas en hogar (ref: 1-2)
            "hogar_3_4": -0.4,
            "hogar_5_plus": -0.8,
            # Balotaje 2019 (ref: otros/no votó/blanco)
            "balotaje_martinez": 1.0,
            "balotaje_lacalle": -0.7,
            # Interacciones
            "mujer_x_relig_mucho": 0.5,
            "mujer_x_tiene_hijos": -0.3,
        },
        "odds_ratios": {
            "edad_25_34": 1.105, "edad_35_44": 1.221, "edad_45_54": 0.905,
            "edad_55_plus": 1.350, "es_mujer": 2.718,
            "educ_ems_incomp": 1.105, "educ_ems_comp": 1.221,
            "educ_ter_incomp": 1.350, "educ_ter_comp": 1.492,
            "relig_poco": 0.741, "relig_bastante": 0.449, "relig_mucho": 0.223,
            "es_montevideo": 1.649, "tiene_hijos": 0.607,
            "hogar_3_4": 0.670, "hogar_5_plus": 0.449,
            "balotaje_martinez": 2.718, "balotaje_lacalle": 0.497,
            "mujer_x_relig_mucho": 1.649, "mujer_x_tiene_hijos": 0.741,
        },
        "model_info": {
            "pseudo_r2": 0.35,
            "n_observations": 100,
            "n_predictors": 20,
            "regularization": "Ridge (L2)",
            "C": 1.0,
            "model_version": 2,
        },
        "variable_ranges": {
            "tramo_edad_num": {
                "options": [1, 2, 3, 4, 5],
                "labels": ["18-24 años", "25-34 años", "35-44 años", "45-54 años", "55+ años"],
                "default": 2,
            },
            "es_mujer": {"options": [0, 1], "labels": ["Hombre", "Mujer"], "default": 0},
            "nivel_educ_num": {
                "options": [1, 2, 3, 4, 5],
                "labels": ["Primaria", "EMS incompleta", "EMS completa", "Terciaria incompleta", "Terciaria completa"],
                "default": 3,
            },
            "religiosidad_num": {
                "options": [1, 2, 3, 4],
                "labels": ["Nada", "Poco", "Bastante", "Mucho"],
                "default": 2,
            },
            "es_montevideo": {"options": [0, 1], "labels": ["Interior", "Montevideo"], "default": 0},
            "tiene_hijos": {"options": [0, 1], "labels": ["No", "Sí"], "default": 0},
            "hogar_num": {
                "options": [1, 2, 3],
                "labels": ["1-2 personas", "3-4 personas", "5 o mas"],
                "default": 2,
            },
            "balotaje": {
                "options": ["otros", "martinez", "lacalle"],
                "labels": ["No votó/Blanco", "Martínez (FA)", "Lacalle (Coalición)"],
                "default": "otros",
            },
        },
        "stats_by_group": {
            "balotaje_martinez": 93.0,
            "balotaje_lacalle": 55.0,
            "religiosidad_nada": 92.8,
            "religiosidad_poco": 86.7,
            "religiosidad_bastante": 65.9,
            "religiosidad_mucho": 14.6,
            "hogar_1_2": 85.0,
            "hogar_3_4": 73.0,
            "hogar_5_plus": 50.0,
        },
        "predictor_names": [
            "edad_25_34", "edad_35_44", "edad_45_54", "edad_55_plus",
            "es_mujer",
            "educ_ems_incomp", "educ_ems_comp", "educ_ter_incomp", "educ_ter_comp",
            "relig_poco", "relig_bastante", "relig_mucho",
            "es_montevideo", "tiene_hijos",
            "hogar_3_4", "hogar_5_plus",
            "balotaje_martinez", "balotaje_lacalle",
            "mujer_x_relig_mucho", "mujer_x_tiene_hijos",
        ],
        "prob_nacional": 78.6,
    }

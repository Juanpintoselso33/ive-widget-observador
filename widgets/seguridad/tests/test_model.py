"""Tests de la lógica de predicción del widget de seguridad."""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import math

import pytest

from widgets.seguridad.config import PREDICTORES
from widgets.seguridad.model import (
    build_features, predict_probability, predict_probability_neutral,
)


def test_coeficientes_en_cero_dan_cincuenta(coef_neutros, perfil_base):
    modelo = {"coefficients": coef_neutros}
    assert predict_probability(modelo, **perfil_base) == pytest.approx(50.0)


def test_efecto_de_ser_mujer_entra_en_el_logit(modelo_sintetico, perfil_base):
    """Con es_mujer=+1 en logit, pasar de hombre a mujer da sigmoid(1)."""
    hombre = predict_probability(modelo_sintetico, **perfil_base)
    mujer = predict_probability(modelo_sintetico, **{**perfil_base, "es_mujer": 1})
    assert hombre == pytest.approx(50.0)
    assert mujer == pytest.approx(1 / (1 + math.exp(-1.0)) * 100)
    assert mujer > hombre


def test_el_modelo_de_neutralidad_usa_sus_propios_coeficientes(modelo_sintetico, perfil_base):
    """El efecto de es_mujer está sólo en el modelo principal, no en el neutral."""
    perfil_mujer = {**perfil_base, "es_mujer": 1}
    assert predict_probability_neutral(modelo_sintetico, **perfil_mujer) == pytest.approx(50.0)
    assert predict_probability(modelo_sintetico, **perfil_mujer) != pytest.approx(50.0)


def test_probabilidad_siempre_en_rango(modelo_sintetico):
    """Ningún perfil puede caer fuera de 0-100."""
    for tramo in (1, 2, 3, 4):
        for educ in (1, 2, 3, 4):
            for ideol in (1, 2, 3, 4):
                for vic in (1, 2, 3):
                    for mvd in (0, 1):
                        for mujer in (0, 1):
                            p = predict_probability(
                                modelo_sintetico, tramo_edad=tramo, es_mujer=mujer,
                                nivel_educ=educ, ideologia=ideol, victima=vic,
                                es_montevideo=mvd,
                            )
                            assert 0.0 <= p <= 100.0


class TestBuildFeatures:
    def test_devuelve_exactamente_los_predictores_declarados(self, perfil_base):
        assert set(build_features(**perfil_base)) == set(PREDICTORES)

    def test_perfil_de_referencia_es_todo_cero(self, perfil_base):
        assert set(build_features(**perfil_base).values()) == {0}

    def test_las_dummies_de_un_bloque_son_mutuamente_excluyentes(self, perfil_base):
        """Sólo una categoría de edad puede estar activa a la vez."""
        for tramo, esperada in [(2, "edad_30_44"), (3, "edad_45_59"), (4, "edad_60_plus")]:
            f = build_features(**{**perfil_base, "tramo_edad": tramo})
            activas = [k for k in ("edad_30_44", "edad_45_59", "edad_60_plus") if f[k]]
            assert activas == [esperada]

    def test_victima_con_y_sin_violencia_no_se_solapan(self, perfil_base):
        sin_v = build_features(**{**perfil_base, "victima": 2})
        con_v = build_features(**{**perfil_base, "victima": 3})
        assert (sin_v["victima_sin_violencia"], sin_v["victima_con_violencia"]) == (1, 0)
        assert (con_v["victima_sin_violencia"], con_v["victima_con_violencia"]) == (0, 1)

    def test_victima_sin_dato_nunca_se_activa_desde_la_ui(self, perfil_base):
        """
        La dummy existe sólo para que los sin dato del entrenamiento no
        contaminen la referencia. Ninguna opción de la UI debe encenderla:
        si alguna lo hiciera, se le estaría aplicando a un usuario real el
        coeficiente de los que no contestaron.
        """
        for victima in (1, 2, 3):
            f = build_features(**{**perfil_base, "victima": victima})
            assert f["victima_sin_dato"] == 0

    def test_centro_ideologico_es_la_referencia(self, perfil_base):
        f = build_features(**{**perfil_base, "ideologia": 2})
        assert f["ideol_izquierda"] == 0
        assert f["ideol_derecha"] == 0
        assert f["ideol_no_ubica"] == 0

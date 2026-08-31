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
    import itertools
    for tramo, educ, ideol, vic, mvd, mujer in itertools.product(
            (1, 2, 3, 4), (1, 2, 3), range(1, 8), (1, 2, 3), (0, 1), (0, 1)):
        p = predict_probability(
            modelo_sintetico, tramo_edad=tramo, es_mujer=mujer, nivel_educ=educ,
            ideologia=ideol, victima=vic, es_montevideo=mvd,
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

    def test_las_dummies_que_no_ofrece_la_ui_quedan_siempre_en_cero(self, perfil_base):
        """
        `victima_sin_dato` e `ideol_no_ubica` existen para que los casos sin
        respuesta del entrenamiento no contaminen las categorías de referencia,
        pero la UI no los ofrece: ninguna combinación puede encenderlos, o se le
        estaría aplicando a un lector el coeficiente de quien no contestó.
        """
        import itertools
        for vic, ideol in itertools.product((1, 2, 3), range(1, 8)):
            f = build_features(**{**perfil_base, "victima": vic,
                                  "ideologia": ideol})
            assert f["victima_sin_dato"] == 0
            assert f["ideol_no_ubica"] == 0

    def test_la_referencia_ideologica_deja_todas_las_dummies_en_cero(self, perfil_base):
        """
        La referencia es el Centro (el 5 de la escala), el cuarto tramo.
        Si alguien reordena IDEOLOGIA_UI_TO_CODE sin tocar ESPEC_CRUDA, este
        test se pone rojo — que es lo que tiene que pasar: los códigos de la UI
        y los tramos de la especificación son posicionales entre sí.
        """
        from widgets.seguridad.config import ESPEC_CRUDA, IDEOLOGIA_UI_TO_CODE
        nombres = [n for n, _, _, _ in ESPEC_CRUDA["ideol_tramos"]]
        ref = ESPEC_CRUDA["ideol_referencia"]
        codigo_ref = nombres.index(ref) + 1

        f = build_features(**{**perfil_base, "ideologia": codigo_ref})
        activas = [k for k in f if k.startswith("ideol_") and f[k]]
        assert activas == [], f"la referencia encendió {activas}"

        # Y cada tramo NO referencia enciende exactamente su propia dummy.
        for codigo, nombre in enumerate(nombres, start=1):
            if nombre == ref:
                continue
            f = build_features(**{**perfil_base, "ideologia": codigo})
            activas = [k for k in f if k.startswith("ideol_") and f[k]]
            assert activas == [f"ideol_{nombre}"], (
                f"código {codigo} ({list(IDEOLOGIA_UI_TO_CODE)[codigo-1]}) "
                f"encendió {activas}, se esperaba ideol_{nombre}"
            )

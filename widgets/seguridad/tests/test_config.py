"""
Tests de coherencia de la configuración.

Cubren sobre todo el mecanismo de pregunta parametrizada: la idea es que
cambiar PREGUNTA_ACTIVA no pueda dejar el widget en un estado inconsistente
sin que un test lo avise.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json

import pytest

from widgets.seguridad import config


def test_la_pregunta_activa_existe_en_el_catalogo():
    assert config.PREGUNTA_ACTIVA in config.PREGUNTAS


def test_toda_pregunta_declara_los_campos_que_usa_la_ui():
    for slug, pregunta in config.PREGUNTAS.items():
        for campo in ("columna", "titulo", "afirma", "verbo"):
            assert campo in pregunta, f"'{slug}' no declara '{campo}'"
            assert pregunta[campo].strip(), f"'{slug}' tiene '{campo}' vacío"


def test_no_hay_dos_preguntas_apuntando_a_la_misma_columna():
    columnas = [p["columna"] for p in config.PREGUNTAS.values()]
    assert len(columnas) == len(set(columnas))


def test_la_escala_likert_cubre_los_cinco_puntos():
    assert sorted(config.LIKERT_MAP.values()) == [1, 2, 3, 4, 5]


def test_favor_contra_y_neutral_no_se_solapan():
    favor, contra = set(config.LIKERT_FAVOR), set(config.LIKERT_CONTRA)
    assert not favor & contra
    assert config.LIKERT_NEUTRAL not in favor | contra
    assert favor | contra | {config.LIKERT_NEUTRAL} == set(config.LIKERT_MAP.values())


class TestMapeosUI:
    """Cada mapeo de la UI debe cubrir los códigos que espera build_features."""

    @pytest.mark.parametrize("mapeo,esperados", [
        ("EDAD_UI_TO_CODE", {1, 2, 3, 4}),
        ("EDUC_UI_TO_CODE", {1, 2, 3}),
        ("IDEOLOGIA_UI_TO_CODE", {1, 2, 3, 4}),
        ("VICTIMA_UI_TO_CODE", {1, 2, 3}),
        ("REGION_UI_TO_CODE", {0, 1}),
    ])
    def test_codigos_completos_y_sin_repetir(self, mapeo, esperados):
        valores = list(getattr(config, mapeo).values())
        assert len(valores) == len(set(valores)), f"{mapeo} tiene códigos repetidos"
        assert set(valores) == esperados


def test_predictores_sin_duplicados():
    assert len(config.PREDICTORES) == len(set(config.PREDICTORES))


@pytest.mark.skipif(
    not config.MODEL_COEFFICIENTS_PATH.exists(),
    reason="El modelo todavía no fue entrenado",
)
class TestModeloEntrenado:
    """Sólo corren si ya existe model_coefficients.json."""

    @pytest.fixture
    def modelo(self):
        with open(config.MODEL_COEFFICIENTS_PATH, encoding="utf-8") as f:
            return json.load(f)

    def test_estan_todos_los_coeficientes_que_espera_la_inferencia(self, modelo):
        faltan = set(config.PREDICTORES) - set(modelo["coefficients"])
        assert not faltan, f"faltan coeficientes: {faltan}"
        assert "intercept" in modelo["coefficients"]

    def test_el_modelo_de_neutralidad_tiene_los_mismos_predictores(self, modelo):
        assert set(modelo["coefficients"]) == set(modelo["coefficients_neutral"])

    def test_el_json_corresponde_a_la_pregunta_activa(self, modelo):
        assert modelo["pregunta_slug"] == config.PREGUNTA_ACTIVA, (
            "El JSON entrenado no corresponde a PREGUNTA_ACTIVA: "
            "hay que volver a correr train_model.py"
        )

    def test_las_tasas_publicadas_son_porcentajes(self, modelo):
        assert 0 <= modelo["prob_favor_nacional"] <= 100
        assert 0 <= modelo["prob_neutral_nacional"] <= 100

    def test_los_conteos_publicados_reconcilian_con_el_n(self, modelo):
        """
        La sección de metodología muestra estos números al lector: si no
        cierran contra el N de la encuesta, se publica una inconsistencia.
        """
        info = modelo["model_info"]
        assert info["n"] + info["n_excluidos"] == info["n_encuesta"]
        assert info["n_neutrales_explicitos"] + info["n_sin_respuesta"] == info["n_excluidos"]

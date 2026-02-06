"""
Tests para config.py -- verifica constantes, rutas y mapeos.
"""

from pathlib import Path

from config import (
    WIDGET_DIR,
    MODEL_COEFFICIENTS_PATH,
    COLORS,
    BALOTAJE_UI_TO_CODE,
    PROB_THRESHOLDS,
    get_interpretation,
)


class TestPaths:
    """Tests para las rutas configuradas."""

    def test_widget_dir_exists(self):
        assert WIDGET_DIR.exists()

    def test_model_coefficients_path_exists(self):
        assert MODEL_COEFFICIENTS_PATH.exists()

    def test_model_coefficients_is_json(self):
        assert MODEL_COEFFICIENTS_PATH.suffix == ".json"


class TestColors:
    """Tests para la paleta de colores."""

    def test_primary_color_is_hex(self):
        assert COLORS["primary"].startswith("#")
        assert len(COLORS["primary"]) == 7

    def test_required_colors_exist(self):
        required = ["primary", "text", "background", "success", "danger", "warning"]
        for key in required:
            assert key in COLORS, f"Missing color: {key}"


class TestBalotajeMapping:
    """Tests para el mapeo de balotaje."""

    def test_three_options(self):
        assert len(BALOTAJE_UI_TO_CODE) == 3

    def test_all_codes_are_strings(self):
        for code in BALOTAJE_UI_TO_CODE.values():
            assert isinstance(code, str)

    def test_no_duplicate_codes(self):
        codes = list(BALOTAJE_UI_TO_CODE.values())
        assert len(codes) == len(set(codes))


class TestProbThresholds:
    """Tests para los umbrales de probabilidad."""

    def test_thresholds_are_descending(self):
        thresholds = [t[0] for t in PROB_THRESHOLDS]
        assert thresholds == sorted(thresholds, reverse=True)

    def test_lowest_threshold_is_zero(self):
        assert PROB_THRESHOLDS[-1][0] == 0

    def test_get_interpretation_returns_tuple(self):
        result = get_interpretation(50)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_interpretation_covers_full_range(self):
        """Cada valor de 0 a 100 debe tener interpretación."""
        for p in range(0, 101):
            color, text = get_interpretation(p)
            assert color.startswith("#")
            assert len(text) > 0

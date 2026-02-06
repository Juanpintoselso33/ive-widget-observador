"""
Tests para la lógica de predicción del modelo IVE (v2: dummies + interacciones).
Usa datos sintéticos del conftest.py -- NO lee model_coefficients.json.
"""

import math

import pytest

from model import predict_probability
from config import get_interpretation, BALOTAJE_UI_TO_CODE

# Claves de coeficientes del modelo v2 (20 predictores + intercept = 21 keys)
_V2_COEF_KEYS = [
    "intercept",
    "edad_25_34", "edad_35_44", "edad_45_54", "edad_55_plus",
    "es_mujer",
    "educ_ems_incomp", "educ_ems_comp", "educ_ter_incomp", "educ_ter_comp",
    "relig_poco", "relig_bastante", "relig_mucho",
    "es_montevideo", "tiene_hijos",
    "hogar_3_4", "hogar_5_plus",
    "balotaje_martinez", "balotaje_lacalle",
    "mujer_x_relig_mucho", "mujer_x_tiene_hijos",
]


def _make_zero_coef():
    """Crea un dict de coeficientes con todos los valores en 0."""
    return {k: 0.0 for k in _V2_COEF_KEYS}


class TestSigmoid:
    """Tests para la función sigmoid / predict_probability."""

    def test_all_reference_levels_gives_fifty_percent(self, synthetic_model):
        """Con intercept=0 y todos los inputs en nivel de referencia, sigmoid(0)=50%."""
        result = predict_probability(
            synthetic_model, tramo_edad=1, es_mujer=0, nivel_educ=1,
            religiosidad=1, es_montevideo=0, tiene_hijos=0, hogar=1, balotaje="otros",
        )
        assert result == pytest.approx(50.0, abs=0.01)

    def test_large_positive_z_near_100(self, synthetic_model):
        """Un z muy positivo debe dar probabilidad cercana a 100%."""
        result = predict_probability(
            synthetic_model, tramo_edad=5, es_mujer=1, nivel_educ=5,
            religiosidad=1, es_montevideo=1, tiene_hijos=0, hogar=1, balotaje="martinez",
        )
        # z = 0.3 + 1.0 + 0.4 + 0.5 + 1.0 = 3.2 -> sigmoid(3.2) ≈ 96.1%
        assert result > 95.0

    def test_large_negative_z_near_zero(self, synthetic_model):
        """Un z muy negativo debe dar probabilidad cercana a 0%."""
        result = predict_probability(
            synthetic_model, tramo_edad=1, es_mujer=0, nivel_educ=1,
            religiosidad=4, es_montevideo=0, tiene_hijos=1, hogar=1, balotaje="lacalle",
        )
        # z = -1.5 + (-0.5) + (-0.5) = -2.5 -> sigmoid(-2.5) ≈ 7.6%
        assert result < 20.0

    def test_result_is_percentage(self, synthetic_model):
        """El resultado debe estar en rango 0-100, no 0-1."""
        result = predict_probability(
            synthetic_model, tramo_edad=3, es_mujer=1, nivel_educ=3,
            religiosidad=2, es_montevideo=1, tiene_hijos=0, hogar=1, balotaje="otros",
        )
        assert 0 <= result <= 100

    def test_symmetry_around_zero(self):
        """Sigmoid(z) + sigmoid(-z) = 100%."""
        coef_pos = _make_zero_coef()
        coef_pos["intercept"] = 2.0
        coef_neg = _make_zero_coef()
        coef_neg["intercept"] = -2.0

        model_pos = {"coefficients": coef_pos}
        model_neg = {"coefficients": coef_neg}

        # All at reference levels so z = intercept only
        pos = predict_probability(model_pos, 1, 0, 1, 1, 0, 0, 1, "otros")
        neg = predict_probability(model_neg, 1, 0, 1, 1, 0, 0, 1, "otros")
        assert pos + neg == pytest.approx(100.0, abs=0.01)


class TestBalotajeEncoding:
    """Tests para la codificación de balotaje en predict_probability."""

    def test_martinez_applies_positive_coefficient(self, synthetic_model):
        """Votar Martínez debe aplicar el coeficiente balotaje_martinez (positivo)."""
        prob_mart = predict_probability(
            synthetic_model, 1, 0, 1, 1, 0, 0, 1, "martinez",
        )
        prob_otros = predict_probability(
            synthetic_model, 1, 0, 1, 1, 0, 0, 1, "otros",
        )
        assert prob_mart > prob_otros

    def test_lacalle_applies_negative_coefficient(self, synthetic_model):
        """Votar Lacalle debe aplicar el coeficiente balotaje_lacalle (negativo)."""
        prob_lac = predict_probability(
            synthetic_model, 1, 0, 1, 1, 0, 0, 1, "lacalle",
        )
        prob_otros = predict_probability(
            synthetic_model, 1, 0, 1, 1, 0, 0, 1, "otros",
        )
        assert prob_lac < prob_otros

    def test_martinez_vs_lacalle_different(self, synthetic_model):
        """Martínez y Lacalle deben producir probabilidades distintas."""
        prob_mart = predict_probability(synthetic_model, 1, 0, 1, 1, 0, 0, 1, "martinez")
        prob_lac = predict_probability(synthetic_model, 1, 0, 1, 1, 0, 0, 1, "lacalle")
        assert prob_mart != pytest.approx(prob_lac, abs=0.01)

    def test_otros_activates_no_dummies(self):
        """'otros' no debe activar ningún dummy de balotaje."""
        coef = _make_zero_coef()
        coef["balotaje_martinez"] = 1.0
        coef["balotaje_lacalle"] = -1.0
        model = {"coefficients": coef}
        result = predict_probability(model, 1, 0, 1, 1, 0, 0, 1, "otros")
        assert result == pytest.approx(50.0, abs=0.01)


class TestDummyEncoding:
    """Tests para la codificación de variables ordinales como dummies."""

    def test_reference_age_no_effect(self, synthetic_model):
        """Tramo 1 (18-24) es referencia, tramo 2 activa edad_25_34 (coef > 0)."""
        prob_ref = predict_probability(synthetic_model, 1, 0, 1, 1, 0, 0, 1, "otros")
        prob_25_34 = predict_probability(synthetic_model, 2, 0, 1, 1, 0, 0, 1, "otros")
        assert prob_25_34 > prob_ref

    def test_reference_educ_no_effect(self, synthetic_model):
        """Nivel 1 (primaria) es referencia, nivel 5 activa educ_ter_comp (coef > 0)."""
        prob_ref = predict_probability(synthetic_model, 1, 0, 1, 1, 0, 0, 1, "otros")
        prob_ter = predict_probability(synthetic_model, 1, 0, 5, 1, 0, 0, 1, "otros")
        assert prob_ter > prob_ref

    def test_reference_relig_no_effect(self, synthetic_model):
        """Religiosidad 1 (nada) es referencia, relig 4 activa relig_mucho (coef < 0)."""
        prob_ref = predict_probability(synthetic_model, 1, 0, 1, 1, 0, 0, 1, "otros")
        prob_mucho = predict_probability(synthetic_model, 1, 0, 1, 4, 0, 0, 1, "otros")
        assert prob_mucho < prob_ref

    def test_each_age_tramo_different(self, synthetic_model):
        """Cada tramo de edad debe producir una probabilidad distinta."""
        probs = [
            predict_probability(synthetic_model, t, 0, 1, 1, 0, 0, 1, "otros")
            for t in range(1, 6)
        ]
        assert len(set(round(p, 4) for p in probs)) == 5


class TestInteractions:
    """Tests para los términos de interacción."""

    def test_mujer_relig_mucho_interaction(self, synthetic_model):
        """La interacción mujer*relig_mucho modifica el efecto de género para muy religiosas."""
        # Gender gap for very religious
        prob_mujer_relig = predict_probability(
            synthetic_model, 1, 1, 1, 4, 0, 0, 1, "otros"
        )
        prob_hombre_relig = predict_probability(
            synthetic_model, 1, 0, 1, 4, 0, 0, 1, "otros"
        )
        gap_religious = prob_mujer_relig - prob_hombre_relig

        # Gender gap for not religious (reference)
        prob_mujer_no_relig = predict_probability(
            synthetic_model, 1, 1, 1, 1, 0, 0, 1, "otros"
        )
        prob_hombre_no_relig = predict_probability(
            synthetic_model, 1, 0, 1, 1, 0, 0, 1, "otros"
        )
        gap_not_religious = prob_mujer_no_relig - prob_hombre_no_relig

        # Gaps differ due to interaction (mujer_x_relig_mucho = 0.5)
        assert abs(gap_religious - gap_not_religious) > 1.0

    def test_mujer_hijos_interaction(self, synthetic_model):
        """La interacción mujer*hijos modifica el efecto de hijos para mujeres."""
        # Effect of children on women
        prob_mujer_hijos = predict_probability(
            synthetic_model, 1, 1, 1, 1, 0, 1, 1, "otros"
        )
        prob_mujer_sin = predict_probability(
            synthetic_model, 1, 1, 1, 1, 0, 0, 1, "otros"
        )
        hijos_effect_mujer = prob_mujer_hijos - prob_mujer_sin

        # Effect of children on men
        prob_hombre_hijos = predict_probability(
            synthetic_model, 1, 0, 1, 1, 0, 1, 1, "otros"
        )
        prob_hombre_sin = predict_probability(
            synthetic_model, 1, 0, 1, 1, 0, 0, 1, "otros"
        )
        hijos_effect_hombre = prob_hombre_hijos - prob_hombre_sin

        # Effects differ due to interaction (mujer_x_tiene_hijos = -0.3)
        assert abs(hijos_effect_mujer - hijos_effect_hombre) > 1.0

    def test_interaction_only_activates_when_both_true(self):
        """La interacción solo se activa cuando ambas variables son 1."""
        coef = _make_zero_coef()
        coef["es_mujer"] = 0.0
        coef["relig_mucho"] = 0.0
        coef["mujer_x_relig_mucho"] = 2.0
        model = {"coefficients": coef}

        # Neither: mujer=0, relig=1 (ref) -> interaction=0
        p_neither = predict_probability(model, 1, 0, 1, 1, 0, 0, 1, "otros")
        assert p_neither == pytest.approx(50.0, abs=0.01)

        # Only mujer: mujer=1, relig=1 (ref) -> interaction=0
        p_only_mujer = predict_probability(model, 1, 1, 1, 1, 0, 0, 1, "otros")
        assert p_only_mujer == pytest.approx(50.0, abs=0.01)

        # Only relig: mujer=0, relig=4 (mucho) -> interaction=0
        p_only_relig = predict_probability(model, 1, 0, 1, 4, 0, 0, 1, "otros")
        assert p_only_relig == pytest.approx(50.0, abs=0.01)

        # Both: mujer=1, relig=4 -> interaction=1*1=1, z=2.0
        p_both = predict_probability(model, 1, 1, 1, 4, 0, 0, 1, "otros")
        assert p_both > 80.0  # sigmoid(2.0) ≈ 88%


class TestHogar:
    """Tests para la variable personas en el hogar."""

    def test_large_household_reduces_probability(self, synthetic_model):
        """Hogar grande (5+) debe reducir probabilidad vs hogar chico (1-2)."""
        prob_small = predict_probability(
            synthetic_model, 1, 0, 1, 1, 0, 0, 1, "otros"
        )
        prob_large = predict_probability(
            synthetic_model, 1, 0, 1, 1, 0, 0, 3, "otros"
        )
        assert prob_large < prob_small


class TestBalotajeMapping:
    """Tests para el mapeo UI -> código de modelo (balotaje)."""

    def test_all_options_mapped(self):
        """Todas las opciones de balotaje deben tener mapeo."""
        assert "Martínez (FA)" in BALOTAJE_UI_TO_CODE
        assert "Lacalle (Coalición)" in BALOTAJE_UI_TO_CODE
        assert "No votó/Blanco" in BALOTAJE_UI_TO_CODE

    def test_correct_codes(self):
        """Los códigos de balotaje deben ser los esperados."""
        assert BALOTAJE_UI_TO_CODE["Martínez (FA)"] == "martinez"
        assert BALOTAJE_UI_TO_CODE["Lacalle (Coalición)"] == "lacalle"
        assert BALOTAJE_UI_TO_CODE["No votó/Blanco"] == "otros"


class TestInterpretation:
    """Tests para los umbrales de interpretación."""

    def test_high_prob_is_very_likely(self):
        color, text = get_interpretation(80)
        assert "muy probable que apoyes" in text

    def test_mid_prob_is_divided(self):
        color, text = get_interpretation(47)
        assert "dividido" in text

    def test_low_prob_is_very_likely_against(self):
        color, text = get_interpretation(15)
        assert "muy probable que te opongas" in text

    def test_boundary_70(self):
        color, text = get_interpretation(70)
        assert "muy probable que apoyes" in text

    def test_boundary_55(self):
        color, text = get_interpretation(55)
        assert "probable que apoyes" in text


class TestDeterminism:
    """Tests de determinismo: mismos inputs deben dar mismos outputs."""

    def test_same_inputs_same_output(self, synthetic_model):
        """Llamadas repetidas con mismos inputs dan mismo resultado."""
        args = (synthetic_model, 3, 1, 3, 2, 1, 0, 1, "martinez")
        r1 = predict_probability(*args)
        r2 = predict_probability(*args)
        assert r1 == r2

    def test_output_is_float(self, synthetic_model):
        """El resultado debe ser un float."""
        result = predict_probability(
            synthetic_model, 3, 1, 3, 2, 1, 0, 1, "martinez",
        )
        assert isinstance(result, float)

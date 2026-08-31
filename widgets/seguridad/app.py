"""
Widget de seguridad pública — El Observador
Entry point standalone. También importable desde el root app.py.
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from shared.styles import get_custom_css
from shared.config import get_colors
from widgets.seguridad.model import (
    load_model as _load_model, predict_probability, intervalo_probabilidad,
    banda_decision,
)
from widgets.seguridad.components import (
    render_header, render_inputs, render_probability_bar,
    render_result_card, render_comparisons, render_methodology, render_footer,
)

from widgets.seguridad.config import (
    PREGUNTA, PREGUNTA_ACTIVA, PREDICTORES, huella_contrato,
)

# El título sale de la pregunta activa y no va hardcodeado: si se cambia la
# pregunta, la pestaña del navegador tiene que acompañar. Antes decía "pena de
# muerte" pasara lo que pasara.
st.set_page_config(
    page_title=f"{PREGUNTA['titulo_corto']} | El Observador",
    layout="centered",
    initial_sidebar_state="collapsed",
)

try:
    theme_mode = st.context.theme.type
except AttributeError:
    theme_mode = "light"

colors = get_colors(theme_mode)
st.markdown(get_custom_css(theme_mode), unsafe_allow_html=True)


@st.cache_data
def load_model():
    return _load_model()


try:
    MODEL = load_model()
except FileNotFoundError:
    st.error(
        "No se encontró el archivo de coeficientes. "
        "Ejecutá primero `widgets/seguridad/train_model.py`."
    )
    st.stop()

# El contrato entre la configuración y el modelo entrenado se verifica ACÁ, al
# arrancar, y no sólo en los tests: cambiar PREGUNTA_ACTIVA sin re-entrenar
# dejaría el título de una pregunta con los coeficientes de otra, y en
# producción nadie corre pytest antes de servir la página. Mejor una pantalla
# de error explícita que un widget que responde cualquier cosa con confianza.
_slug = MODEL.get("pregunta_slug")
if _slug != PREGUNTA_ACTIVA:
    st.error(
        f"El modelo entrenado corresponde a la pregunta «{_slug}» pero la "
        f"configuración pide «{PREGUNTA_ACTIVA}». Volvé a correr "
        "`widgets/seguridad/train_model.py` antes de publicar."
    )
    st.stop()

# La huella cubre los mapeos y las referencias, no sólo los nombres de las
# dummies: si una categoría cambia de significado conservando su nombre, el
# chequeo de "no falta ninguna" pasaría igual y la inferencia aplicaría
# coeficientes entrenados con otra codificación.
_contrato = MODEL.get("contrato")
if _contrato != huella_contrato():
    st.error(
        "El modelo entrenado no corresponde a la configuración actual "
        f"(contrato {_contrato} contra {huella_contrato()}). Cambió algún mapeo, "
        "predictor o categoría: volvé a correr `widgets/seguridad/train_model.py`."
    )
    st.stop()

_esperados = set(PREDICTORES)
_reales = set(MODEL.get("coefficients", {})) - {"intercept"}
if _esperados != _reales or "intercept" not in MODEL.get("coefficients", {}):
    st.error(
        "Los predictores del modelo no coinciden exactamente con los que espera "
        f"la aplicación (faltan: {sorted(_esperados - _reales)}; "
        f"sobran: {sorted(_reales - _esperados)}). Volvé a correr "
        "`widgets/seguridad/train_model.py`."
    )
    st.stop()

render_header(MODEL)
inputs = render_inputs()

# No se calcula la neutralidad por perfil: la UI muestra la tasa general porque
# ese modelo casi no discrimina (pseudo-R² 0,03). Calcularla igual sólo abriría
# la posibilidad de que un JSON neutral defectuoso rompa la página.
prob = predict_probability(MODEL, *inputs)
intervalo = intervalo_probabilidad(MODEL, *inputs)
# El que se muestra y el que decide sobre el 50% son distintos a propósito:
# ver el docstring de model.banda_decision().
banda = banda_decision(MODEL, *inputs)

render_probability_bar(prob)
render_result_card(MODEL, prob, colors, intervalo, banda)
render_comparisons(MODEL, prob, colors)
render_methodology(MODEL)
render_footer(MODEL)

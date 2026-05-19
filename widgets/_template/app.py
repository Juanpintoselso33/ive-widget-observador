"""
Widget [NOMBRE] — El Observador
Entry point. Reemplazá [NOMBRE] y adaptá las llamadas a render_*.
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from shared.styles import get_custom_css
from shared.config import get_colors
from widgets._template.model import load_model as _load_model, predict_probability
from widgets._template.components import render_header, render_inputs, render_probability_bar, render_result

st.set_page_config(
    page_title="[Título] | El Observador",
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
    st.error("No se encontró model_coefficients.json. Ejecutá train_model.py primero.")
    st.stop()

render_header()
inputs = render_inputs(MODEL)
prob = predict_probability(MODEL, *inputs)

render_probability_bar(prob, colors)
render_result(prob, colors, theme_mode)

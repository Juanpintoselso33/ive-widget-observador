"""
Widget IVE — El Observador
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
from widgets.ive.model import load_model as _load_model, predict_probability, predict_probability_neutral
from widgets.ive.components import (
    render_header, render_inputs, render_probability_bar,
    render_result_card, render_comparisons, render_methodology, render_footer,
)

st.set_page_config(
    page_title="¿Apoyás el IVE? | El Observador",
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
    st.error("Error: No se encontró el archivo de coeficientes. "
             "Ejecutá primero `widgets/ive/train_model.py`")
    st.stop()

render_header()
inputs = render_inputs(MODEL)
prob = predict_probability(MODEL, *inputs)
prob_nacional = MODEL.get('prob_nacional', 78.6)

prob_neutral = None
if 'coefficients_neutral' in MODEL:
    prob_neutral = predict_probability_neutral(MODEL, *inputs)

render_probability_bar(prob, colors)
render_result_card(prob, prob_nacional, colors, theme_mode, prob_neutral=prob_neutral)
render_comparisons(MODEL, prob, colors)
render_methodology(MODEL)
render_footer(MODEL)

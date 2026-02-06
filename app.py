"""
Widget Interactivo: \u00bfCu\u00e1l es tu probabilidad de apoyar el IVE?
El Observador - Encuesta Uruguay 2025/2026

Basado en el modelo "Build a Voter" de The Economist
Adaptado para medir apoyo a la interrupci\u00f3n voluntaria del embarazo

Autor: El Observador / Equipo de Datos
"""

import streamlit as st
from styles import get_custom_css
from config import get_colors
from model import load_model as _load_model, predict_probability
from components import (
    render_header,
    render_inputs,
    render_probability_bar,
    render_result_card,
    render_comparisons,
    render_methodology,
    render_footer,
)

# ============================================================
# CONFIGURACI\u00d3N DE P\u00c1GINA
# ============================================================
st.set_page_config(
    page_title="\u00bfApoyas el IVE? | El Observador",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# DETECCI\u00d3N DE TEMA + ESTILOS
# ============================================================
try:
    theme_mode = st.context.theme.type
except AttributeError:
    theme_mode = "light"

colors = get_colors(theme_mode)
st.markdown(get_custom_css(theme_mode), unsafe_allow_html=True)

# ============================================================
# CARGAR MODELO (con cache de Streamlit)
# ============================================================
@st.cache_data
def load_model():
    return _load_model()

try:
    MODEL = load_model()
except FileNotFoundError:
    st.error("Error: No se encontr\u00f3 el archivo de coeficientes. "
             "Ejecuta primero `train_model.py`")
    st.stop()

# ============================================================
# RENDERIZAR WIDGET
# ============================================================
render_header()

inputs = render_inputs(MODEL)
prob = predict_probability(MODEL, *inputs)
prob_nacional = MODEL.get('prob_nacional', 78.6)

render_probability_bar(prob, colors)
render_result_card(prob, prob_nacional, colors, theme_mode)
render_comparisons(MODEL, prob, colors)
render_methodology(MODEL)
render_footer(MODEL)

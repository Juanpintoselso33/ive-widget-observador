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
from shared.styles import get_observador_css
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

# El Figma "Producto UY" no tiene variante oscura, así que el widget no sigue el
# tema del sistema del lector y queda siempre en claro. Con la paleta oscura de
# Streamlit el verde profundo del titular y del número quedaba ilegible, y
# elegir otro verde para el modo oscuro sería inventar una decisión de diseño
# que la diseñadora no tomó. (Antes se leía `st.context.theme.type`; el tema ya
# venía fijo en claro desde `.streamlit/config.toml`, así que esa rama nunca
# daba "dark" en producción.)
st.markdown(get_observador_css(), unsafe_allow_html=True)


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

# LA BANDA GRIS DEL FIGMA. Desde el gradiente hasta el pie, el diseño va sobre
# #EDEDED y no sobre blanco: es un tercio del área. No se puede hacer sólo con
# CSS porque esto son bloques sueltos de Streamlit —markdown, `st.tabs`, más
# markdown— y no hay un ancestro común que envuelva justo a esos y a ninguno
# más. Un `st.container(key=...)` sí lo crea, y Streamlit le pone la clase
# `st-key-<key>`, que es la vía soportada para engancharle CSS.
with st.container(key="banda_resultado"):
    render_probability_bar(prob)
    render_result_card(prob, prob_nacional, prob_neutral=prob_neutral)
    render_comparisons(MODEL, prob_nacional)
    render_methodology(MODEL)
    render_footer(MODEL)

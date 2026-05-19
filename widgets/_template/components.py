"""
Componentes UI del widget [NOMBRE].
Adaptar render_inputs() para los selectboxes propios de este widget.
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from shared.config import get_colors, get_interpretation


def render_header():
    st.markdown('<h1 class="widget-title">[Título del widget]</h1>', unsafe_allow_html=True)
    st.markdown('<p class="widget-subtitle">[Subtítulo]</p>', unsafe_allow_html=True)


def render_inputs(model):
    """
    Retorna tuple de inputs para pasar a predict_probability().
    Adaptar los selectboxes según las variables del modelo.
    """
    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox("Variable 1", ["Opción A", "Opción B"])
    with col2:
        var2 = st.selectbox("Variable 2", ["Opción X", "Opción Y"])
    return (var1, var2)


def render_probability_bar(prob, colors):
    filled = int(prob)
    st.markdown(f"""
    <div class="prob-bar-container">
      <div class="prob-bar-fill" style="width:{filled}%"></div>
    </div>
    """, unsafe_allow_html=True)


def render_result(prob, colors, mode):
    color, text = get_interpretation(prob, mode)
    st.markdown(f"""
    <div class="result-card">
      <div class="prob-number" style="color:{color}">{prob:.0f}%</div>
      <div class="prob-label">Es <strong>{text}</strong></div>
    </div>
    """, unsafe_allow_html=True)

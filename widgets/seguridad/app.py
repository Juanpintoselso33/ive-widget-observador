"""
Widget de seguridad pública — El Observador
Entry point standalone. También importable desde el root app.py.

Publica las CUATRO preguntas punitivas que pidió Tomer y deja que el lector
elija cuál estimar: hay un modelo entrenado por pregunta y todos se cargan al
arrancar.
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from shared.styles import get_observador_css
from shared.config import OBSERVADOR_COLORS
from widgets.seguridad.model import (
    load_modelos as _load_modelos, predict_probability, intervalo_probabilidad,
    banda_decision,
)
from widgets.seguridad.components import (
    render_selector_pregunta, render_header, render_inputs,
    render_probability_bar, render_result_card, render_comparisons,
    render_methodology, render_footer, CLAVE_PREGUNTA,
)

from widgets.seguridad.config import (
    PREGUNTAS, SLUGS, PREGUNTA_DEFECTO, ETIQUETA_A_SLUG, PREDICTORES,
    huella_contrato,
)

# El título de la pestaña sigue a la pregunta elegida. Se lee de session_state
# ANTES de set_page_config porque ésa tiene que ser la primera orden de
# Streamlit de la corrida; leer el estado no dibuja nada, así que es válido.
# En la primera corrida todavía no hay nada guardado y sale la de por defecto.
_etiqueta_elegida = st.session_state.get(CLAVE_PREGUNTA)
_slug_inicial = ETIQUETA_A_SLUG.get(_etiqueta_elegida, PREGUNTA_DEFECTO)

st.set_page_config(
    page_title=f"{PREGUNTAS[_slug_inicial]['titulo_corto']} | El Observador",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# El Figma "Producto UY" no tiene variante oscura, así que el widget deja de
# seguir el tema del sistema del lector y queda siempre en claro. Con la paleta
# oscura de Streamlit el verde profundo del titular y del número quedaba
# ilegible, y elegir otro verde para el modo oscuro sería inventar una decisión
# de diseño que la diseñadora no tomó.
colors = OBSERVADOR_COLORS
st.markdown(get_observador_css(), unsafe_allow_html=True)


@st.cache_data
def load_modelos():
    return _load_modelos()


try:
    MODELOS = load_modelos()
except FileNotFoundError as e:
    st.error(
        f"Falta un archivo de coeficientes ({e.filename}). Ejecutá primero "
        "`python widgets/seguridad/train_model.py`."
    )
    st.stop()

# El contrato entre la configuración y cada modelo entrenado se verifica ACÁ, al
# arrancar, y no sólo en los tests: un JSON desalineado dejaría el título de una
# pregunta con los coeficientes de otra, y en producción nadie corre pytest
# antes de servir la página. Mejor una pantalla de error explícita que un widget
# que responde cualquier cosa con confianza.
#
# Se verifican los CUATRO aunque el lector vaya a mirar uno: si se validara sólo
# el elegido, un JSON roto quedaría escondido hasta que alguien seleccionara esa
# pregunta, que es justo el momento en que ya no hay nadie mirando la consola.
_problemas = []
for _slug in SLUGS:
    _modelo = MODELOS[_slug]

    if _modelo.get("pregunta_slug") != _slug:
        _problemas.append(
            f"«{_slug}»: el JSON dice ser de «{_modelo.get('pregunta_slug')}»"
        )
        continue

    # La huella cubre los mapeos y las referencias, no sólo los nombres de las
    # dummies: si una categoría cambia de significado conservando su nombre, el
    # chequeo de "no falta ninguna" pasaría igual y la inferencia aplicaría
    # coeficientes entrenados con otra codificación.
    _contrato = _modelo.get("contrato")
    if _contrato != huella_contrato(_slug):
        _problemas.append(
            f"«{_slug}»: contrato {_contrato} contra {huella_contrato(_slug)} "
            "— cambió algún mapeo, predictor o categoría"
        )
        continue

    _esperados = set(PREDICTORES)
    _reales = set(_modelo.get("coefficients", {})) - {"intercept"}
    if _esperados != _reales or "intercept" not in _modelo.get("coefficients", {}):
        _problemas.append(
            f"«{_slug}»: predictores distintos "
            f"(faltan: {sorted(_esperados - _reales)}; "
            f"sobran: {sorted(_reales - _esperados)})"
        )

if _problemas:
    st.error(
        "Los modelos entrenados no corresponden a la configuración actual:\n\n- "
        + "\n- ".join(_problemas)
        + "\n\nVolvé a correr `python widgets/seguridad/train_model.py`."
    )
    st.stop()

# ============================================================
# RENDER
# ============================================================
slug = render_selector_pregunta()
MODEL = MODELOS[slug]

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
render_comparisons(MODEL)
render_methodology(MODEL)
render_footer(MODEL)

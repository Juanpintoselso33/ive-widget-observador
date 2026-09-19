"""
Widget IVE — El Observador
Entry point standalone. También importable desde el root app.py.

El cuerpo vive en `main()` A PROPÓSITO, y no suelto a nivel de módulo: el
`app.py` de la raíz —que es el *Main file path* del deploy— lo importa y lo
llama, así que este archivo queda registrado en `sys.modules` y el watcher de
Streamlit lo vigila. Antes la raíz lo ejecutaba con `runpy.run_path()`, que lo
corre en un `__main__` temporal y lo saca de `sys.modules`: funcionaba, pero
editar ESTE archivo no recargaba la app al correr `streamlit run app.py` (los
módulos importados, como `components.py`, sí se recargaban, que es lo que hacía
difícil de notar el agujero). Lo marcó Codex.
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from shared.styles import get_observador_css
# Sin `predict_probability_neutral`: la tasa de "no toma posición" salió de la
# tarjeta por pedido editorial (ver render_result_card). El modelo auxiliar se
# sigue entrenando y viajando en el JSON, así que volver a mostrarla es sumar la
# llamada de nuevo, no reentrenar nada.
from widgets.ive.model import load_model as _load_model, predict_probability
from widgets.ive.components import (
    render_header, render_inputs, render_probability_bar,
    render_result_card, render_comparisons, render_methodology, render_footer,
)


@st.cache_data
def load_model():
    return _load_model()


# La versión de caja se pide por query param: `?resumen=1`. Va por URL y no por
# un control en pantalla porque quien la elige es quien arma el embebido en la
# nota, no el lector. Tomer lo pidió el 18/9/2026: "ver si puede haber una
# versión resumida para que entre en una caja".
PARAM_RESUMEN = "resumen"
_VERDADEROS = {"1", "true", "si", "sí"}


def modo_resumen():
    """
    Si el embebido pidió la versión de caja.

    CON EL PARÁMETRO REPETIDO GANA EL ÚLTIMO, que es lo que hace `st.query_params`
    y por eso no se reimplementa: `?resumen=0&resumen=1` es caja y
    `?resumen=1&resumen=0` no lo es. La versión anterior se quedaba con el
    PRIMERO y decía en el docstring que eso era lo que hacía Streamlit — era
    falso, y como el test le pasaba un diccionario con una lista en vez del
    proxy real, verificaba justamente la semántica equivocada. Lo marcó Codex.

    `get_all()` es la API para ver todos los valores; se usa sólo para quedarse
    con el último de forma explícita, y si no existe —versiones viejas— se cae
    a `get()`, que ya devuelve ese mismo último valor.
    """
    params = st.query_params
    obtener_todos = getattr(params, "get_all", None)
    if callable(obtener_todos):
        valores = obtener_todos(PARAM_RESUMEN)
        valor = valores[-1] if valores else None
    else:
        valor = params.get(PARAM_RESUMEN)

    return str(valor).strip().lower() in _VERDADEROS


def main():
    st.set_page_config(
        page_title="¿Apoyás el IVE? | El Observador",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    # El Figma "Producto UY" no tiene variante oscura, así que el widget no
    # sigue el tema del sistema del lector y queda siempre en claro. Con la
    # paleta oscura de Streamlit el verde profundo del titular y del número
    # quedaba ilegible, y elegir otro verde para el modo oscuro sería inventar
    # una decisión de diseño que la diseñadora no tomó. (Antes se leía
    # `st.context.theme.type`; el tema ya venía fijo en claro desde
    # `.streamlit/config.toml`, así que esa rama nunca daba "dark" en prod.)
    st.markdown(get_observador_css(), unsafe_allow_html=True)

    try:
        MODEL = load_model()
    except FileNotFoundError:
        st.error("Error: No se encontró el archivo de coeficientes. "
                 "Ejecutá primero `widgets/ive/train_model.py`")
        st.stop()

    # SIN FALLBACK. Había un `MODEL.get('prob_nacional', 78.6)` y el modelo
    # entrenado trae 76,5: si el artefacto perdiera la clave, el widget
    # publicaría un promedio nacional que no es el de este modelo, sin avisar.
    # Desde que la comparación por grupo se mide contra este número, un
    # fallback equivocado ya no desplaza sólo la tarjeta — corre los quince
    # deltas de la grilla. Mejor una pantalla de error que quince cifras
    # mansamente equivocadas. Lo marcó Codex.
    prob_nacional = MODEL.get("prob_nacional")
    if prob_nacional is None:
        st.error(
            "El archivo de coeficientes no trae `prob_nacional`, que es el "
            "promedio nacional contra el que se compara todo el widget. "
            "Volvé a correr `widgets/ive/train_model.py`."
        )
        st.stop()

    resumen = modo_resumen()

    render_header()
    inputs = render_inputs(MODEL)
    prob = predict_probability(MODEL, *inputs)

    # LA BANDA GRIS DEL FIGMA. Desde el gradiente hasta el pie, el diseño va
    # sobre #EDEDED y no sobre blanco: es un tercio del área. No se puede hacer
    # sólo con CSS porque esto son bloques sueltos de Streamlit —markdown,
    # `st.tabs`, más markdown— y no hay un ancestro común que envuelva justo a
    # esos y a ninguno más. Un `st.container(key=...)` sí lo crea, y Streamlit
    # le pone la clase `st-key-<key>`, que es la vía soportada para engancharle
    # CSS.
    with st.container(key="banda_resultado"):
        render_probability_bar(prob)
        render_result_card(prob)
        # En la caja entran el formulario y el resultado. Lo que se saca es lo
        # que hace scroll: la comparación por grupos y el desplegable del
        # modelo. El promedio nacional queda dicho en la comparación, así que
        # en la caja no aparece en ningún lado — es el costo de la versión
        # corta, y por eso la caja tiene que linkear al widget completo.
        if not resumen:
            render_comparisons(MODEL, prob_nacional)
            render_methodology(MODEL)
        render_footer(MODEL, resumido=resumen)


if __name__ == "__main__":
    main()

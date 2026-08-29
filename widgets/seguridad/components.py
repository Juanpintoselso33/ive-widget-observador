"""
Componentes UI del widget de seguridad.

Reusa las clases CSS de shared/styles.py (mismas que el widget IVE), pero NO
reusa la semántica de color de shared.config.get_interpretation: ahí el apoyo
se pinta de verde y la oposición de rojo, lo cual es razonable para el IVE y
sería editorializar acá — pintar de verde "apoya la pena de muerte" es tomar
partido. Este widget usa una escala de intensidad de un solo tono.
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from widgets.seguridad.config import (
    PREGUNTA, EDAD_UI_TO_CODE, EDUC_UI_TO_CODE, IDEOLOGIA_UI_TO_CODE,
    VICTIMA_UI_TO_CODE, REGION_UI_TO_CODE,
)

# Escala neutra: la intensidad del color acompaña la magnitud, sin valorarla.
# Cada plantilla trae su oración completa —no se le antepone "Es"— porque la
# franja del medio no admite esa construcción ("Es estás dividido/a").
INTENSIDAD = [
    (70, "Es muy probable que {verbo}"),
    (55, "Es probable que {verbo}"),
    (45, "Estás dividido/a"),
    (30, "Es poco probable que {verbo}"),
    (0,  "Es muy poco probable que {verbo}"),
]


def interpretar(prob, colors):
    """Devuelve (color, texto) sin cargar valoración moral en el color."""
    for umbral, plantilla in INTENSIDAD:
        if prob >= umbral:
            return colors["primary"], plantilla.format(verbo=PREGUNTA["verbo"])
    return colors["primary"], INTENSIDAD[-1][1].format(verbo=PREGUNTA["verbo"])


def render_header(model):
    st.markdown(
        f'<h1 class="main-title">{model.get("pregunta_titulo", PREGUNTA["titulo"])}</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle">Basado en la encuesta de El Observador sobre seguridad '
        'pública, entre uruguayos <em>con opinión formada</em> sobre el tema. '
        'Seleccioná tus características:</p>',
        unsafe_allow_html=True,
    )


def render_inputs():
    """
    Renderiza los selectores y devuelve los valores codificados, en el orden
    que espera model.predict_probability().

    Returns:
        tuple: (tramo_edad, es_mujer, nivel_educ, ideologia, victima, es_montevideo)
    """
    col1, col2 = st.columns(2)

    with col1:
        edad_sel = st.selectbox(
            "Edad", options=list(EDAD_UI_TO_CODE), index=1,
            help="Tu tramo de edad",
        )
        sexo = st.selectbox("Sexo", options=["Hombre", "Mujer"], index=0)
        educ_sel = st.selectbox(
            "Nivel educativo", options=list(EDUC_UI_TO_CODE), index=1,
            help="El máximo nivel que alcanzaste",
        )

    with col2:
        ideol_sel = st.selectbox(
            "Ideología", options=list(IDEOLOGIA_UI_TO_CODE), index=1,
            help="En política se habla de izquierda y derecha. ¿Dónde te ubicás?",
        )
        victima_sel = st.selectbox(
            "¿Fuiste víctima de un delito en los últimos 12 meses?",
            options=list(VICTIMA_UI_TO_CODE), index=0,
        )
        region_sel = st.selectbox(
            "Región", options=list(REGION_UI_TO_CODE), index=1,
        )

    return (
        EDAD_UI_TO_CODE[edad_sel],
        1 if sexo == "Mujer" else 0,
        EDUC_UI_TO_CODE[educ_sel],
        IDEOLOGIA_UI_TO_CODE[ideol_sel],
        VICTIMA_UI_TO_CODE[victima_sel],
        REGION_UI_TO_CODE[region_sel],
    )


def render_probability_bar(prob):
    st.markdown('<hr class="editorial-divider">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="prob-bar-wrapper">
        <div class="prob-endpoints">
            <span class="prob-endpoint prob-endpoint--contra">EN CONTRA</span>
            <span class="prob-endpoint prob-endpoint--favor">A FAVOR</span>
        </div>
        <div class="prob-container">
            <div class="prob-indicator" style="left: {prob}%;">
                <div class="prob-label">{prob:.0f}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_result_card(model, prob, colors, prob_neutral=None):
    color, texto = interpretar(prob, colors)

    # La diferencia se calcula sobre los valores YA redondeados que ve el
    # lector: si en pantalla dicen 53% y 37%, la brecha tiene que decir 16pp.
    # Restar primero y redondear después da 17pp y la cuenta no cierra a la
    # vista, que en una pieza periodística se lee como un error.
    prob_r = round(prob)
    nacional_r = round(model["prob_favor_nacional"])
    diff = prob_r - nacional_r
    arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="

    neutral_html = ""
    if prob_neutral is not None:
        neutral_html = (
            f'<div style="margin-top:8px;font-size:0.85em;color:{colors["text_muted"]};">'
            f'Además, <strong>{prob_neutral:.0f}%</strong> de las personas con tu perfil '
            f'no toma posición clara sobre el tema.</div>'
        )

    posicion = "por encima" if diff > 0 else "por debajo" if diff < 0 else "igual"
    brecha = (
        f"{arrow} tu perfil está {abs(diff)}pp {posicion}"
        if diff else "= tu perfil coincide con el promedio"
    )

    st.markdown(f"""
    <div class="result-card">
        <div class="result-number" style="color: {color};">{prob_r}%</div>
        <div class="result-text">
            Probabilidad de {model.get("pregunta_afirma", PREGUNTA["afirma"])},
            <em>entre quienes tienen postura definida</em>.<br>
            <strong style="color: {color};">{texto}</strong> según tus características.
        </div>
        <div class="result-nacional">
            Promedio nacional:
            <span class="result-nacional-value">{nacional_r}%</span>
            <span class="result-nacional-diff" style="color: {colors["text_muted"]};">
                {brecha}
            </span>
        </div>
        {neutral_html}
    </div>
    """, unsafe_allow_html=True)


# Etiqueta legible para cada grupo del bloque comparativo.
GRUPOS_LABEL = {
    "izquierda": "Se ubica a la izquierda",
    "derecha": "Se ubica a la derecha",
    "victima": "Fue víctima de un delito",
    "no_victima": "No fue víctima",
    "hombres": "Hombres",
    "mujeres": "Mujeres",
    "montevideo": "Montevideo",
    "interior": "Interior",
    "edad_18_29": "18 a 29 años",
    "edad_60_plus": "60 años o más",
}


def render_comparisons(model, prob, colors):
    st.markdown('<hr class="editorial-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Cómo se compara con otros grupos</div>',
                unsafe_allow_html=True)

    stats = model.get("stats_by_group", {})
    # Se omiten los grupos que el entrenamiento marcó como None (n < 30).
    visibles = [(k, v) for k, v in stats.items() if v is not None and k in GRUPOS_LABEL]
    visibles.sort(key=lambda kv: kv[1], reverse=True)

    prob_r = round(prob)
    for i in range(0, len(visibles), 2):
        cols = st.columns(2)
        for col, (clave, valor) in zip(cols, visibles[i:i + 2]):
            # Misma regla que en la tarjeta de resultado: la brecha se calcula
            # sobre los porcentajes redondeados que se muestran.
            valor_r = round(valor)
            delta = valor_r - prob_r
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{GRUPOS_LABEL[clave]}</div>
                    <div class="metric-value">{valor_r}%</div>
                    <div class="metric-delta" style="color:{colors['text_muted']};">
                        {'+' if delta > 0 else ''}{delta}pp vs. tu perfil
                    </div>
                </div>
                """, unsafe_allow_html=True)


def render_methodology(model):
    info = model.get("model_info", {})
    with st.expander("Cómo se calcula"):
        st.markdown(f"""
El porcentaje sale de una **regresión logística ponderada** ajustada sobre la
encuesta de El Observador de seguridad pública (mayo de 2026), con el ponderador
de diseño muestral `{info.get('ponderador', 'w_norm')}`.

- **Encuestados:** {info.get('n_encuesta', '—')}
- **Casos con postura definida:** {info.get('n', '—')}
- **Excluidos:** {info.get('n_excluidos', '—')}
  ({info.get('n_neutrales_explicitos', '—')} contestaron "ni de acuerdo ni en desacuerdo"
  y {info.get('n_sin_respuesta', '—')} no contestaron)
- **Pseudo-R² de McFadden:** {info.get('mcfadden_r2', '—')}
- **Categorías de referencia:** {', '.join(f'{k}: {v}' for k, v in model.get('referencias', {}).items())}

El resultado es **condicional a tener postura definida**: quien contesta "ni de
acuerdo ni en desacuerdo" queda fuera del cálculo principal y se estima aparte.

Son **probabilidades basadas en correlaciones estadísticas de la encuesta, no
predicciones sobre una persona concreta**. Dos personas con el mismo perfil
pueden opinar distinto: el modelo describe tendencias de grupo.
        """)


def render_footer(model):
    st.markdown(f"""
    <p class="footer-text">
        Fuente: {model.get('fuente', 'Encuesta El Observador')} ·
        Modelo actualizado el {model.get('entrenado', '—')}
    </p>
    """, unsafe_allow_html=True)

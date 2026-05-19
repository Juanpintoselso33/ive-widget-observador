"""
Componentes UI del widget IVE.
Cada función renderiza una sección del widget usando Streamlit.
Diseño editorial inspirado en The Economist + El Observador.
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from widgets.ive.config import BALOTAJE_UI_TO_CODE, get_interpretation, get_colors


def render_header():
    """Renderiza título y subtítulo."""
    st.markdown(
        '<h1 class="main-title">¿Cuál es tu probabilidad de apoyar '
        'el derecho a decidir sobre el embarazo?</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle">Basado en la encuesta de El Observador a uruguayos '
        '<em>con opinión formada</em> sobre el tema. '
        'Seleccioná tus características:</p>',
        unsafe_allow_html=True,
    )


def render_inputs(model):
    """
    Renderiza los selectboxes de entrada y devuelve los valores codificados.

    Returns:
        tuple: (tramo_edad, es_mujer, nivel_educ, religiosidad_num,
                es_montevideo, tiene_hijos, hogar, balotaje)
    """
    ranges = model['variable_ranges']

    col1, col2 = st.columns(2)

    with col1:
        edad_options = ranges['tramo_edad_num']['labels']
        edad_sel = st.selectbox(
            "Edad", options=edad_options, index=1,
            help="Selecciona tu tramo de edad",
        )
        tramo_edad = edad_options.index(edad_sel) + 1

        sexo_options = ranges['es_mujer']['labels']
        sexo = st.selectbox(
            "Sexo", options=sexo_options, index=0,
            help="Selecciona tu sexo",
        )
        es_mujer = 1 if sexo == "Mujer" else 0

        educ_options = ranges['nivel_educ_num']['labels']
        educ_default = ranges['nivel_educ_num'].get('default', 4) - 1
        educacion = st.selectbox(
            "Nivel educativo", options=educ_options, index=educ_default,
            help="Selecciona tu nivel educativo más alto",
        )
        nivel_educ = educ_options.index(educacion) + 1

        balotaje_options = ranges['balotaje']['labels']
        balotaje_sel = st.selectbox(
            "Balotaje 2019", options=balotaje_options, index=0,
            help="¿A quién votaste en el balotaje de 2019?",
        )
        balotaje = BALOTAJE_UI_TO_CODE[balotaje_sel]

    with col2:
        relig_options = ranges['religiosidad_num']['labels']
        religiosidad = st.selectbox(
            "Religiosidad", options=relig_options, index=1,
            help="¿Cuán religioso/a te consideras?",
        )
        religiosidad_num = relig_options.index(religiosidad) + 1

        region_options = ranges['es_montevideo']['labels']
        region = st.selectbox(
            "Región", options=region_options, index=0,
            help="¿Dónde vives?",
        )
        es_montevideo = 1 if region == "Montevideo" else 0

        hijos_options = ranges['tiene_hijos']['labels']
        hijos = st.selectbox(
            "¿Tienes hijos?", options=hijos_options, index=0,
            help="¿Tienes hijos/as?",
        )
        tiene_hijos = 1 if hijos == "Sí" else 0

        hogar_options = ranges['hogar_num']['labels']
        hogar_sel = st.selectbox(
            "Personas en el hogar", options=hogar_options, index=1,
            help="Cantidad de personas que viven en tu hogar",
        )
        hogar = hogar_options.index(hogar_sel) + 1

    return tramo_edad, es_mujer, nivel_educ, religiosidad_num, es_montevideo, tiene_hijos, hogar, balotaje


def render_probability_bar(prob, colors=None):
    """Renderiza la barra visual de probabilidad con gradiente editorial."""
    if colors is None:
        colors = get_colors()

    st.markdown('<hr class="editorial-divider">', unsafe_allow_html=True)
    bar_html = f"""
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
    """
    st.markdown(bar_html, unsafe_allow_html=True)


def render_result_card(prob, prob_nacional, colors=None, mode="light", prob_neutral=None):
    """Renderiza la tarjeta de resultado con promedio nacional y % de neutrales."""
    if colors is None:
        colors = get_colors(mode)

    color, texto = get_interpretation(prob, mode)

    diff = prob - prob_nacional
    arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
    diff_color = colors["success"] if diff > 0 else colors["danger"] if diff < 0 else colors["text_muted"]

    neutral_html = ""
    if prob_neutral is not None:
        neutral_html = (
            f'<div class="result-neutral" '
            f'style="margin-top:8px;font-size:0.85em;color:{colors["text_muted"]};">'
            f'Además, <strong>{prob_neutral:.0f}%</strong> de personas con tu perfil '
            f'no toma posición clara sobre el tema.'
            f'</div>'
        )

    st.markdown(f"""
    <div class="result-card">
        <div class="result-number" style="color: {color};">{prob:.0f}%</div>
        <div class="result-text">
            Probabilidad de apoyar el derecho a decidir sobre el embarazo
            <em>entre quienes tienen postura definida</em>.<br>
            <strong style="color: {color};">Es {texto}</strong> al IVE según tus características.
        </div>
        <div class="result-nacional">
            Promedio nacional:
            <span class="result-nacional-value">{prob_nacional:.0f}%</span>
            <span class="result-nacional-diff" style="color: {diff_color};">
                {arrow} {abs(diff):.0f}pp vs. tu
            </span>
        </div>
        {neutral_html}
    </div>
    """, unsafe_allow_html=True)


def render_comparisons(model, prob, colors=None):
    """Renderiza las comparaciones por grupo en tabs."""
    if colors is None:
        colors = get_colors()

    st.markdown('<hr class="editorial-divider">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header">Comparación con otros grupos</div>',
        unsafe_allow_html=True,
    )

    stats = model['stats_by_group']

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Por religiosidad", "Por balotaje 2019", "Por educación", "Por edad"]
    )

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        _group_metric(col1, stats, 'religiosidad_nada', "Nada religioso", prob, colors)
        _group_metric(col2, stats, 'religiosidad_poco', "Poco religioso", prob, colors)
        _group_metric(col3, stats, 'religiosidad_bastante', "Bastante religioso", prob, colors)
        _group_metric(col4, stats, 'religiosidad_mucho', "Muy religioso", prob, colors)

    with tab2:
        col1, col2 = st.columns(2)
        _group_metric(col1, stats, 'balotaje_martinez', "Martínez (FA)", prob, colors)
        _group_metric(col2, stats, 'balotaje_lacalle', "Lacalle (Coalición)", prob, colors)

    with tab3:
        col1, col2, col3, col4 = st.columns(4)
        _group_metric(col1, stats, 'educacion_primaria', "Primaria o menos", prob, colors)
        _group_metric(col2, stats, 'educacion_secundaria', "Secundaria", prob, colors)
        _group_metric(col3, stats, 'educacion_ter_incomp', "Terciaria incompleta", prob, colors)
        _group_metric(col4, stats, 'educacion_ter_comp', "Terciaria completa+", prob, colors)

    with tab4:
        col1, col2, col3, col4, col5 = st.columns(5)
        _group_metric(col1, stats, 'edad_18-24', "18-24", prob, colors)
        _group_metric(col2, stats, 'edad_25-34', "25-34", prob, colors)
        _group_metric(col3, stats, 'edad_35-44', "35-44", prob, colors)
        _group_metric(col4, stats, 'edad_45-54', "45-54", prob, colors)
        _group_metric(col5, stats, 'edad_55+', "55+", prob, colors)


def _group_metric(col, stats, key, label, prob, colors=None, show_delta=True):
    """Helper: renderiza una metric card HTML para una estadística de grupo."""
    if colors is None:
        colors = get_colors()

    val = stats.get(key, 0)
    with col:
        delta_html = ""
        if show_delta and val != prob:
            diff = val - prob
            diff_color = colors["success"] if diff > 0 else colors["danger"]
            delta_html = f'<div class="metric-delta" style="color: {diff_color};">{diff:+.0f}pp</div>'

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}%</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)


def render_methodology(model):
    """Renderiza el expander con la explicación metodológica."""
    with st.expander("¿Cómo funciona este modelo?"):
        st.markdown("""
        Este widget utiliza un **modelo de regresión logística** entrenado con datos de la encuesta de
        El Observador realizada en Uruguay.

        **Variables más influyentes:**

        1. **Religiosidad**: Es el factor más importante. Las personas más religiosas tienen
           significativamente menor probabilidad de apoyar el IVE.

        2. **Voto político**: Votantes de Martínez en el balotaje tienen mayor probabilidad de apoyo
           que votantes de Lacalle.

        3. **Sexo**: Las mujeres tienden a apoyar más el IVE que los hombres.

        4. **Educación**: Mayor nivel educativo se asocia con mayor apoyo al derecho a decidir.

        5. **Región**: Pueden existir diferencias entre Montevideo y el Interior.

        **Nota metodológica:**
        - El modelo excluye a quienes respondieron "Ni de acuerdo ni en desacuerdo"
        - Los resultados son probabilidades basadas en correlaciones estadísticas, no predicciones individuales
        - Pseudo R² del modelo: {:.1%}
        """.format(model['model_info']['pseudo_r2']))


def render_footer(model):
    """Renderiza el pie de página."""
    st.markdown("""
    <div class="footer-text">
        <strong>El Observador</strong> | Encuesta realizada en Uruguay 2025/2026<br>
        Basado en {} respuestas ponderadas | Modelo de regresión logística<br>
        <em>Las probabilidades son estimaciones estadísticas basadas en grupos,
        no predicciones individuales</em>
    </div>
    """.format(model['model_info']['n_observations']), unsafe_allow_html=True)

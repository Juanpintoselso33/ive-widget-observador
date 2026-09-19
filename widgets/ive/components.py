"""
Componentes UI del widget IVE.
Cada función renderiza una sección del widget usando Streamlit.

Sigue el Figma "Producto UY", página **Widget IVE** — que es literalmente el
mock de este widget: el frame tiene "¿Tienes hijos?", "Balotaje 2019" y
"Religiosidad". Los valores están en `docs/diseno/figma-producto-uy.md` y la
hoja de estilos es `shared.styles.get_observador_css()`, la misma que ya usaba
el widget de seguridad.
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
# Sin la paleta: desde que la hoja del Figma pinta el número, el énfasis y las
# diferencias, ningún componente de acá elige un color. El único que queda es el
# `style="left: …%"` del indicador de la barra, que es posición y no color.
from shared.config import PROB_THRESHOLDS
from widgets.ive.config import BALOTAJE_UI_TO_CODE


def texto_interpretacion(prob):
    """
    El texto de interpretación, SIN color.

    `shared.config.get_interpretation()` devuelve además una clave de color
    semántica —verde, ámbar, rojo— que la paleta del Figma no tiene: la
    diseñadora resolvió el widget con dos verdes, un naranja y un azul, y
    ninguno codifica "bueno" o "malo". Acá se recorren los mismos umbrales y se
    descarta esa clave a propósito; el color del resultado lo pone la hoja.
    """
    for umbral, _clave_color, texto in PROB_THRESHOLDS:
        if prob >= umbral:
            return texto
    return PROB_THRESHOLDS[-1][2]


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
            "Balotaje 2024", options=balotaje_options, index=0,
            help="¿A quién votaste en el balotaje de 2024?",
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


def render_probability_bar(prob):
    """
    La barra de probabilidad con el gradiente naranja → azul del Figma.

    SIN LÍNEA DIVISORIA ACÁ. La había, y ahora el corte entre el formulario y el
    resultado lo hace el borde de la banda gris, que empieza justo en este
    punto: dejar las dos era una raya suelta sobre el gris.

    Los extremos van en capitalización normal, no en versalitas: el Figma los
    tiene como cuerpo —"A favor" / "En contra"— y la hoja no les pone
    `text-transform`, así que en mayúsculas quedaban gritando.
    """
    bar_html = f"""
    <div class="prob-bar-wrapper">
        <div class="prob-endpoints">
            <span class="prob-endpoint prob-endpoint--contra">En contra</span>
            <span class="prob-endpoint prob-endpoint--favor">A favor</span>
        </div>
        <div class="prob-container">
            <div class="prob-indicator" style="left: {prob}%;">
                <div class="prob-label">{prob:.0f}%</div>
            </div>
        </div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)


def render_result_card(prob, prob_nacional, prob_neutral=None):
    """
    La tarjeta de resultado: número grande, interpretación y promedio nacional.

    SIN COLORES EN LÍNEA. Los ponía todos el llamador con la paleta semántica
    vieja; ahora el número y el énfasis los pinta la hoja con el verde sólido
    del Figma, y el único color que varía es el de la diferencia contra el
    promedio, que sigue al SIGNO y no a una valoración: azul si el perfil queda
    por encima del promedio, naranja si queda por debajo. Son los mismos dos
    colores que los extremos del gradiente, y se aplican con las mismas clases
    que las diferencias por grupo.
    """
    texto = texto_interpretacion(prob)

    # La diferencia se calcula sobre los valores YA REDONDEADOS que ve el
    # lector: si en pantalla dicen 81% y 77%, la brecha tiene que decir 4pp.
    # Restar primero y redondear después da 5pp y la cuenta no cierra a la
    # vista, que en una pieza periodística se lee como un error. Es el mismo
    # criterio que en el widget de seguridad.
    prob_r = round(prob)
    nacional_r = round(prob_nacional)
    diff = prob_r - nacional_r

    # El formato sale del Figma, que lo muestra como "↓2pp por debajo". Dice
    # DÓNDE CAE TU PERFIL respecto del promedio: la versión anterior decía
    # "↑ 4pp vs. tu" colgando de la línea del promedio nacional, que además de
    # quedar cortada se leía al revés —como si el que estuviera por encima fuera
    # el promedio— porque la resta es perfil menos nacional.
    if diff > 0:
        brecha = f"↑{abs(diff)}pp por encima"
        clase_brecha = "grupo-celda-delta--sube"
    elif diff < 0:
        brecha = f"↓{abs(diff)}pp por debajo"
        clase_brecha = "grupo-celda-delta--baja"
    else:
        brecha = "igual al promedio"
        clase_brecha = ""

    neutral_html = ""
    if prob_neutral is not None:
        neutral_html = (
            f'<div class="result-neutral">'
            f'Además, <strong>{prob_neutral:.0f}%</strong> de las personas con tu perfil '
            f'no toma posición clara sobre el tema y queda fuera de este cálculo.'
            f'</div>'
        )

    st.markdown(f"""
    <div class="result-card">
        <div class="result-number">{prob_r}%</div>
        <div class="result-text">
            Probabilidad de apoyar el derecho a decidir sobre el embarazo
            <em>entre quienes tienen postura definida</em>.<br>
            <strong>Es {texto}</strong> al IVE según tus características.
        </div>
        <div class="result-nacional">
            Promedio nacional:
            <span class="result-nacional-value">{nacional_r}%</span>
            <span class="result-nacional-diff {clase_brecha}">
                {brecha}
            </span>
        </div>
        {neutral_html}
    </div>
    """, unsafe_allow_html=True)


# Las dimensiones del bloque comparativo y la etiqueta de cada grupo. Declarado
# acá y no cableado en el render para que agregar o sacar un grupo no obligue a
# tocar también el reparto de columnas — que es justo lo que había antes, con un
# `st.columns(n)` distinto por solapa.
GRUPOS_ORDEN = [
    ("Por religiosidad", ["religiosidad_nada", "religiosidad_poco",
                          "religiosidad_bastante", "religiosidad_mucho"]),
    ("Por balotaje 2024", ["balotaje_martinez", "balotaje_lacalle"]),
    ("Por educación", ["educacion_primaria", "educacion_secundaria",
                       "educacion_ter_incomp", "educacion_ter_comp"]),
    ("Por edad", ["edad_18-24", "edad_25-34", "edad_35-44",
                  "edad_45-54", "edad_55+"]),
]

GRUPOS_LABEL = {
    "religiosidad_nada": "Nada religioso",
    "religiosidad_poco": "Poco religioso",
    "religiosidad_bastante": "Bastante religioso",
    "religiosidad_mucho": "Muy religioso",
    "balotaje_martinez": "Orsi (FA)",
    "balotaje_lacalle": "Delgado (Coalición)",
    "educacion_primaria": "Primaria o menos",
    "educacion_secundaria": "Secundaria",
    "educacion_ter_incomp": "Terciaria incompleta",
    "educacion_ter_comp": "Terciaria completa+",
    "edad_18-24": "18-24",
    "edad_25-34": "25-34",
    "edad_35-44": "35-44",
    "edad_45-54": "45-54",
    "edad_55+": "55+",
}


def render_comparisons(model, prob_nacional):
    """
    Comparación por grupos con el formato del Figma: una solapa por dimensión y
    adentro una fila de números grandes con su diferencia contra el promedio.

    Antes era una `metric-card` por columna de `st.columns`. El Figma lo resuelve
    con una grilla de cuatro columnas fijas —dos en móvil— y celdas separadas por
    un filete, que es lo que estila `.grupo-cifras` / `.grupo-celda`. La grilla
    también arregla lo que el reparto por columnas hacía mal con la solapa de
    edad: cinco `st.columns` en una fila dejaban las celdas más angostas que las
    de las otras solapas; en la grilla, la quinta baja a una segunda fila
    alineada con la primera.

    LA DIFERENCIA ES CONTRA EL PROMEDIO NACIONAL, no contra la predicción de tu
    perfil, y es un cambio de qué se está midiendo. Restarla contra el perfil
    mezclaba dos cosas distintas: estas son tasas OBSERVADAS por grupo en la
    encuesta, sin ajustar, y la del perfil es una predicción que controla por
    todo lo demás. El widget de seguridad ya lo había corregido por ese motivo.
    """
    st.markdown('<hr class="editorial-divider">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header">Comparación con otros grupos</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle">Porcentaje que se declaró a favor en cada grupo de '
        'la encuesta, sin ajustar por las demás características.</p>',
        unsafe_allow_html=True,
    )

    stats = model['stats_by_group']

    # Se arma primero y se dibuja después: una dimensión sin ningún grupo con
    # dato no debe generar una solapa vacía.
    dimensiones = []
    for titulo, claves in GRUPOS_ORDEN:
        celdas = [(k, stats[k]) for k in claves if stats.get(k) is not None]
        if celdas:
            dimensiones.append((titulo, celdas))

    if not dimensiones:
        return

    nacional_r = round(prob_nacional)

    for solapa, (_, celdas) in zip(st.tabs([t for t, _ in dimensiones]), dimensiones):
        with solapa:
            html = []
            for clave, valor in celdas:
                # Sobre los valores YA redondeados, que son los que se ven:
                # restar antes y redondear después deja cuentas que no cierran
                # a la vista.
                d = round(valor) - nacional_r
                if d:
                    signo = "+" if d > 0 else "−"
                    clase = "sube" if d > 0 else "baja"
                    delta_html = (
                        f'<div class="grupo-celda-delta '
                        f'grupo-celda-delta--{clase}">{signo}{abs(d)}pp</div>'
                    )
                else:
                    delta_html = ('<div class="grupo-celda-delta">'
                                  'igual al promedio</div>')
                html.append(
                    f'<div class="grupo-celda">'
                    f'<div class="grupo-celda-label">{GRUPOS_LABEL[clave]}</div>'
                    f'<div class="grupo-celda-valor">{round(valor)}%</div>'
                    f'{delta_html}</div>'
                )
            st.markdown(f'<div class="grupo-cifras">{"".join(html)}</div>',
                        unsafe_allow_html=True)

    st.markdown(
        f'<div class="grupo-nota-ref">La diferencia es contra el promedio '
        f'nacional, {nacional_r}%.</div>',
        unsafe_allow_html=True,
    )


def render_methodology(model):
    """Renderiza el expander con la explicación metodológica."""
    with st.expander("¿Cómo funciona este modelo?"):
        st.markdown("""
        Este widget utiliza un **modelo de regresión logística** entrenado con datos de la encuesta de
        El Observador realizada en Uruguay.

        **Variables más influyentes:**

        1. **Religiosidad**: Es el factor más importante. Las personas más religiosas tienen
           significativamente menor probabilidad de apoyar el IVE.

        2. **Voto político**: Votantes de Orsi (FA) en el balotaje 2024 tienen mayor probabilidad de apoyo
           que votantes de Delgado (Coalición).

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

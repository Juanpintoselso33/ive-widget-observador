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
    VICTIMA_UI_TO_CODE, REGION_UI_TO_CODE, BALOTAJE_UI_TO_CODE,
)

# Escala neutra: la intensidad del color acompaña la magnitud, sin valorarla.
#
# El texto describe AL GRUPO, no a quien está mirando. La versión anterior decía
# "Es probable que apoyes", que es una predicción individual — y la propia
# sección de metodología aclara que el modelo no hace eso. Con un AUC de 0,74 el
# modelo separa grupos de manera moderada; tutear al lector con un pronóstico
# sobre él afirma bastante más de lo que el dato aguanta.
INTENSIDAD = [
    (70, "La mayoría de las personas con este perfil está a favor"),
    (55, "Más de la mitad de las personas con este perfil está a favor"),
    (45, "Las personas con este perfil se dividen casi por la mitad"),
    (30, "La mayoría de las personas con este perfil está en contra"),
    (0,  "La amplia mayoría de las personas con este perfil está en contra"),
]


def interpretar(prob, colors):
    """Devuelve (color, texto) sin cargar valoración moral en el color."""
    for umbral, texto in INTENSIDAD:
        if prob >= umbral:
            return colors["primary"], texto
    return colors["primary"], INTENSIDAD[-1][1]


def render_header(model):
    st.markdown(
        f'<h1 class="main-title">{model.get("pregunta_titulo", PREGUNTA["titulo"])}</h1>',
        unsafe_allow_html=True,
    )
    # El enunciado textual del cuestionario, entre comillas. El widget mide el
    # acuerdo con ESA frase; si arriba se muestra una paráfrasis y el modelo
    # estima otra cosa, el número dice algo distinto de lo que el lector cree.
    enunciado = model.get("pregunta_enunciado", PREGUNTA.get("enunciado", ""))
    if enunciado:
        st.markdown(
            f'<p class="subtitle">A los encuestados se les leyó esta frase: '
            f'<em>«{enunciado}»</em>.</p>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<p class="subtitle">Basado en la encuesta de El Observador sobre seguridad '
        'pública de mayo de 2026, entre uruguayos <em>con opinión formada</em> sobre '
        'el tema. Elegí un perfil:</p>',
        unsafe_allow_html=True,
    )


def render_inputs():
    """
    Renderiza los selectores y devuelve los valores codificados, en el orden
    que espera model.predict_probability().

    Returns:
        tuple: (tramo_edad, es_mujer, nivel_educ, ideologia, victima,
                es_montevideo, balotaje)
    """
    # Cuatro a la izquierda y tres a la derecha, agrupadas por tipo: quién sos
    # de un lado, qué pensás y qué te pasó del otro. Repartirlas 3/4 dejaba un
    # hueco visible al pie de la primera columna.
    col1, col2 = st.columns(2)

    with col1:
        edad_sel = st.selectbox(
            "Edad", options=list(EDAD_UI_TO_CODE), index=1,
            help="Tu tramo de edad",
        )
        sexo = st.selectbox(
            "Sexo", options=["Hombre", "Mujer"], index=0,
            help="La encuesta relevó esta variable de forma binaria, así que el "
                 "modelo sólo puede estimar sobre esas dos categorías.",
        )
        educ_sel = st.selectbox(
            "Nivel educativo", options=list(EDUC_UI_TO_CODE), index=1,
            help="El máximo nivel que alcanzaste",
        )
        region_sel = st.selectbox(
            "Región", options=list(REGION_UI_TO_CODE), index=1,
        )

    with col2:
        ideol_sel = st.selectbox(
            "Ideología", options=list(IDEOLOGIA_UI_TO_CODE), index=1,
            help="En política se habla normalmente de izquierda y derecha. "
                 "En una escala de 0 a 10, ¿dónde te ubicarías?",
        )
        balotaje_sel = st.selectbox(
            "¿A quién votaste en el balotaje de 2024?",
            options=list(BALOTAJE_UI_TO_CODE), index=2,
        )
        victima_sel = st.selectbox(
            "¿Fuiste víctima de un delito en los últimos 12 meses?",
            options=list(VICTIMA_UI_TO_CODE), index=0,
        )

    return (
        EDAD_UI_TO_CODE[edad_sel],
        1 if sexo == "Mujer" else 0,
        EDUC_UI_TO_CODE[educ_sel],
        IDEOLOGIA_UI_TO_CODE[ideol_sel],
        VICTIMA_UI_TO_CODE[victima_sel],
        REGION_UI_TO_CODE[region_sel],
        BALOTAJE_UI_TO_CODE[balotaje_sel],
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


def render_result_card(model, prob, colors):
    color, texto = interpretar(prob, colors)

    # La diferencia se calcula sobre los valores YA redondeados que ve el
    # lector: si en pantalla dicen 53% y 37%, la brecha tiene que decir 16pp.
    # Restar primero y redondear después da 17pp y la cuenta no cierra a la
    # vista, que en una pieza periodística se lee como un error.
    prob_r = round(prob)
    nacional_r = round(model["prob_favor_nacional"])
    diff = prob_r - nacional_r
    arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="

    # La tasa de "no toma posición" se muestra GENERAL, no por perfil. El modelo
    # de neutralidad tiene un pseudo-R² de 0,03: prácticamente no distingue
    # perfiles, pero al personalizarlo mostraba diferencias de más de veinte
    # puntos entre uno y otro. Eso es ruido presentado como dato.
    neutral_html = ""
    tasa_neutral = model.get("prob_neutral_nacional")
    if tasa_neutral is not None:
        neutral_html = (
            f'<div style="margin-top:8px;font-size:0.85em;color:{colors["text_muted"]};">'
            f'Aparte, <strong>{tasa_neutral:.0f}%</strong> de los uruguayos no toman '
            f'posición clara sobre el tema y quedan fuera de este cálculo.</div>'
        )

    posicion = "por encima" if diff > 0 else "por debajo" if diff < 0 else "igual"
    brecha = (
        f"{arrow} este perfil está {abs(diff)}pp {posicion}"
        if diff else "= este perfil coincide con el promedio"
    )

    st.markdown(f"""
    <div class="result-card">
        <div class="result-number" style="color: {color};">{prob_r}%</div>
        <div class="result-text">
            El modelo estima que, entre quienes tienen estas características y
            <em>postura definida</em>, ese es el porcentaje que declara
            {model.get("pregunta_afirma", PREGUNTA["afirma"])}.<br>
            <strong style="color: {color};">{texto}</strong>
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
    "educ_secundaria": "Secundaria o menos",
    "educ_ter_incompleta": "Terciaria incompleta",
    "educ_ter_completa": "Terciaria completa",
    "voto_orsi": "Votó a Orsi",
    "voto_delgado": "Votó a Delgado",
    "voto_blanco_no_voto": "Blanco, anulado o no votó",
}


def render_comparisons(model, prob, colors):
    st.markdown('<hr class="editorial-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Qué declaró cada grupo</div>',
                unsafe_allow_html=True)
    # Son tasas OBSERVADAS por grupo, no predicciones ajustadas. Antes se
    # restaban contra la estimación del perfil elegido y ese delta mezclaba dos
    # cosas distintas: un promedio descriptivo contra una predicción que
    # controla por todo lo demás.
    st.markdown(
        f'<p class="subtitle">Porcentaje que se declaró a favor en cada grupo de '
        f'la encuesta, sin ajustar por las demás características.</p>',
        unsafe_allow_html=True,
    )

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
                        de la encuesta
                    </div>
                </div>
                """, unsafe_allow_html=True)


def render_methodology(model):
    info = model.get("model_info", {})
    cob = model.get("cobertura_perfiles", {})
    with st.expander("Cómo se calcula"):
        st.markdown(f"""
El porcentaje sale de una **regresión logística ponderada** ajustada sobre la
encuesta de El Observador de seguridad pública (mayo de 2026), con el ponderador
de diseño muestral `{info.get('ponderador', 'w_norm')}`.

- **Encuestados:** {info.get('n_encuesta', '—')}
- **Casos con postura definida:** {info.get('n', '—')}
- **Tamaño efectivo (Kish):** {info.get('n_efectivo_kish', '—')} — por la dispersión
  de los ponderadores, esas respuestas rinden como esa cantidad a efectos de
  precisión, bastante menos que el total nominal
- **Excluidos:** {info.get('n_excluidos', '—')}
  ({info.get('n_neutrales_explicitos', '—')} contestaron "ni de acuerdo ni en desacuerdo"
  y {info.get('n_sin_respuesta', '—')} no contestaron)
- **Pseudo-R² de McFadden:** {info.get('mcfadden_r2', '—')}
- **Categorías de referencia:** {', '.join(f'{k}: {v}' for k, v in model.get('referencias', {}).items())}

El resultado es **condicional a tener postura definida**: quien contesta "ni de
acuerdo ni en desacuerdo" queda fuera del cálculo principal.

Son **correlaciones estadísticas de una encuesta, no predicciones sobre una
persona concreta ni relaciones de causa y efecto**. Dos personas con el mismo
perfil pueden opinar distinto: el modelo describe tendencias de grupo, y las
separa de manera moderada.

**Cuidado con los perfiles poco frecuentes.** {cob.get('posibles', '—')}
combinaciones se pueden elegir acá, pero sólo {cob.get('observados', '—')}
aparecen en la encuesta, y apenas {cob.get('con_30_o_mas', '—')} tienen 30
casos o más. El modelo es aditivo y estima las que faltan combinando
información de perfiles parecidos, no observándolas: cuanto más inusual sea la
combinación elegida, más extrapolación hay detrás del número, y no se muestran
intervalos de confianza.

**Qué sostiene y qué no.** Los factores que más pesan —nivel educativo, edad y
autoubicación ideológica— se mantienen estables cuando se estima el modelo de
otra forma. En cambio **el efecto del sexo, el de la región y el contraste entre 30-44 y
18-29 años son chicos y no son robustos**: cambian de signo según la
especificación, así que este widget no permite afirmar que las mujeres apoyen
más que los varones, ni Montevideo más que el interior, ni que los de 30 a 44
apoyen más que los más jóvenes. Lo que sí se sostiene en edad es el contraste
de los mayores de 60, que apoyan claramente menos. Las tasas por grupo que se muestran abajo son descriptivas
de la muestra, no efectos ajustados.
        """)


def render_footer(model):
    st.markdown(f"""
    <p class="footer-text">
        Fuente: {model.get('fuente', 'Encuesta El Observador')} ·
        Modelo actualizado el {model.get('entrenado', '—')}
    </p>
    """, unsafe_allow_html=True)

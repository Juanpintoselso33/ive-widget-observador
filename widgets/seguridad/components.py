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
    VICTIMA_UI_TO_CODE, REGION_UI_TO_CODE, ESPEC_CRUDA,
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


def interpretar(prob, colors, intervalo=None, banda=None):
    """
    Devuelve (color, texto) sin cargar valoración moral en el color.

    Si el intervalo de confianza cruza el 50%, no se afirma de qué lado está la
    mayoría: con estos intervalos —que rondan los 25 puntos— una estimación de
    43% puede corresponder tanto a una mayoría en contra como a favor, y decir
    "la mayoría está en contra" sería afirmar más de lo que el dato aguanta.

    `intervalo` es el que se MUESTRA; `banda` es el que DECIDE. Son distintos a
    propósito: el extremo del intervalo está simulado con 1.000 réplicas y tiene
    su propio error, que no importa para mostrar un rango pero sí para una regla
    binaria contra el 50%. `banda` viene de model.banda_decision() y es ese
    mismo intervalo corrido hacia afuera lo que la simulación no puede resolver.
    Si no se pasa, se cae al intervalo mostrado — que es el comportamiento
    anterior, menos prudente.
    """
    decisorio = banda or intervalo
    # La comparación va sobre los extremos REDONDEADOS, que son los que ve el
    # lector, y es inclusiva: si en pantalla dice "entre 25% y 50%", afirmar que
    # la mayoría está en contra contradice lo que el propio intervalo muestra.
    if decisorio and round(decisorio[0]) <= 50 <= round(decisorio[1]):
        return colors["primary"], (
            "El margen de error no permite afirmar de qué lado está la mayoría "
            "en este perfil"
        )
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
                es_montevideo)
    """
    # Tres y tres, agrupadas por tipo: quién sos de un lado, qué pensás y qué
    # te pasó del otro. Al sacar el selector de balotaje quedaron seis, así que
    # las columnas cierran parejas.
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

    with col2:
        # index=2 es "Centroizquierda (5)", la categoría modal y la referencia
        # del modelo: el widget abre en el perfil más común, no en un extremo.
        ideol_sel = st.selectbox(
            "Identificación ideológica", options=list(IDEOLOGIA_UI_TO_CODE), index=2,
            help="En política se habla normalmente de izquierda y derecha. En "
                 "una escala de 0 a 10, ¿dónde te ubicarías? Los tramos entre "
                 "paréntesis son los valores de esa escala.",
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


def render_result_card(model, prob, colors, intervalo=None, banda=None):
    color, texto = interpretar(prob, colors, intervalo, banda)

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
            f'<div class="result-neutral">'
            f'Aparte, <strong>{tasa_neutral:.0f}%</strong> de los uruguayos no toman '
            f'posición clara sobre el tema y quedan fuera de este cálculo.</div>'
        )

    # El intervalo va junto al número, no escondido en la metodología: con 571
    # casos efectivos y perfiles que muchas veces no existen en la muestra, la
    # amplitud es parte del dato.
    intervalo_html = ""
    if intervalo:
        bajo, alto = intervalo
        intervalo_html = (
            f'<div class="result-intervalo">Intervalo de confianza del 95%: '
            f'entre <strong>{bajo:.0f}%</strong> y <strong>{alto:.0f}%</strong></div>'
        )

    posicion = "por encima" if diff > 0 else "por debajo" if diff < 0 else "igual"
    brecha = (
        f"{arrow} este perfil está {abs(diff)}pp {posicion}"
        if diff else "= este perfil coincide con el promedio"
    )

    st.markdown(f"""
    <div class="result-card">
        <div class="result-number" style="color: {color};">{prob_r}%</div>
        {intervalo_html}
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
    # Las seis etiquetas ideológicas salen de IDEOLOGIA_UI_TO_CODE, que es la
    # misma tabla que ve el lector en el selector: si se renombra un tramo, se
    # renombra en los dos lados o en ninguno.
    **{f"ideol_{nombre}": etiqueta
       for (nombre, _, _), etiqueta in zip(ESPEC_CRUDA["ideol_tramos"],
                                           IDEOLOGIA_UI_TO_CODE)},
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
}


# Las dimensiones, y dentro de cada una el orden que tiene sentido leer
# (izquierda a derecha, educación y edad de menor a mayor), no el orden por
# valor: ordenar todos por magnitud daba un ranking sin estructura, en el que
# "mujeres" quedaba pegado a "interior" por casualidad aritmética. Con los seis
# tramos ideológicos importa todavía más: puestos en orden se lee de un vistazo
# si el apoyo crece de izquierda a derecha o no.
GRUPOS_ORDEN = [
    ("Identificación ideológica",
     [f"ideol_{nombre}" for nombre, _, _ in ESPEC_CRUDA["ideol_tramos"]]),
    ("Victimización", ["victima", "no_victima"]),
    ("Nivel educativo", ["educ_secundaria", "educ_ter_incompleta", "educ_ter_completa"]),
    ("Edad", ["edad_18_29", "edad_60_plus"]),
    ("Sexo", ["hombres", "mujeres"]),
    ("Región", ["montevideo", "interior"]),
]


def render_comparisons(model):
    # Sin `prob` ni `colors`: las tasas por grupo son descriptivas y ya no se
    # restan contra la estimación del perfil, y el color no codifica nada acá.
    st.markdown('<hr class="editorial-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Qué declaró cada grupo</div>',
                unsafe_allow_html=True)
    # Son tasas OBSERVADAS por grupo, no predicciones ajustadas. Antes se
    # restaban contra la estimación del perfil elegido y ese delta mezclaba dos
    # cosas distintas: un promedio descriptivo contra una predicción que
    # controla por todo lo demás.
    st.markdown(
        '<p class="subtitle">Porcentaje que se declaró a favor en cada grupo de '
        'la encuesta, sin ajustar por las demás características.</p>',
        unsafe_allow_html=True,
    )

    stats = model.get("stats_by_group", {})
    nacional = model.get("prob_favor_nacional")

    # Una sola llamada a st.markdown en vez de dieciséis: Streamlit envuelve
    # cada una en su propio contenedor con margen, y eso era buena parte de la
    # sensación de "bloques sueltos apilados".
    bloques = []
    for titulo, claves in GRUPOS_ORDEN:
        # Se omiten los grupos que el entrenamiento marcó como None (n < 30).
        filas = [(k, stats[k]) for k in claves
                 if stats.get(k) is not None and k in GRUPOS_LABEL]
        if not filas:
            continue
        html_filas = "".join(
            f'<div class="grupo-fila">'
            f'<div class="grupo-label">{GRUPOS_LABEL[clave]}</div>'
            f'<div class="grupo-barra">'
            f'<div class="grupo-barra-fill" style="width:{max(0, min(100, valor)):.1f}%"></div>'
            + (f'<div class="grupo-barra-ref" style="left:{nacional:.1f}%"></div>'
               if nacional is not None else "")
            + f'</div>'
            f'<div class="grupo-valor">{round(valor)}%</div>'
            f'</div>'
            for clave, valor in filas
        )
        bloques.append(
            f'<div class="grupo-bloque">'
            f'<div class="grupo-titulo">{titulo}</div>{html_filas}</div>'
        )

    if nacional is not None:
        bloques.append(
            f'<div class="grupo-nota-ref"><span class="grupo-nota-marca"></span>'
            f'La línea marca el promedio nacional ({round(nacional)}%).</div>'
        )

    st.markdown("".join(bloques), unsafe_allow_html=True)


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
combinación elegida, más extrapolación hay detrás del número y más ancho es
su intervalo.

**Por qué a veces el intervalo no llega al 50% y aun así no se afirma de qué
lado está la mayoría.** El intervalo no se calcula con una fórmula cerrada: se
simula, remuestreando la encuesta mil veces. Un extremo que cae en 49% podría
haber caído en 51% con otra simulación, así que para afirmar que la mayoría está
de un lado se exige un margen un poco más ancho que el que se muestra. Cuando el
extremo queda pegado al 50%, el widget prefiere no afirmar.

**Sobre la escala ideológica.** La pregunta fue: *"en una escala donde cero es
la extrema izquierda y 10 es la extrema derecha, ¿dónde se ubicaría usted?"*.
Va de 0 a 10, así que **el 5 es el punto medio exacto**, y es la respuesta más
elegida: un tercio de los encuestados se ubica ahí. Los siete tramos son
simétricos alrededor de ese centro, para que "extrema izquierda" y "extrema
derecha" abarquen lo mismo y se puedan comparar. El más chico es *Izquierda
extrema (0-1)*, con 80 casos: es el número más frágil de esta página.

**Qué sostiene y qué no.** Estimado de otra manera —sobre la escala completa de
acuerdo, sin excluir a los neutrales— el modelo mantiene los efectos grandes:
**quienes se ubican en el extremo derecho de la escala apoyan mucho más**, y
apoyan claramente menos quienes se ubican a la izquierda, quienes tienen
estudios terciarios y los mayores de 60. Haber sido víctima de un delito con
violencia también aguanta.

En cambio **no son robustos el efecto del sexo, el de la región, el contraste
entre 30-44 y 18-29 años, ni los dos tramos de la derecha moderada
(centroderecha y derecha)**: cambian de signo según cómo se estime. O sea que
este widget no permite afirmar que las mujeres apoyen más que los varones, ni
Montevideo más que el interior, ni ordenar con confianza esa zona de la escala.
Lo que sí queda firme son los extremos y la izquierda.

Las tasas por grupo que se muestran abajo son descriptivas de la muestra, no
efectos ajustados: mezclan el efecto propio del grupo con el de todo lo demás
que lo acompaña.
        """)


def render_footer(model):
    st.markdown(f"""
    <p class="footer-text">
        Fuente: {model.get('fuente', 'Encuesta El Observador')} ·
        Modelo actualizado el {model.get('entrenado', '—')}
    </p>
    """, unsafe_allow_html=True)

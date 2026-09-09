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
    PREGUNTAS, SLUGS, PREGUNTA_DEFECTO, ETIQUETA_A_SLUG,
    EDAD_UI_TO_CODE, EDUC_UI_TO_CODE, IDEOLOGIA_UI_TO_CODE,
    VICTIMA_UI_TO_CODE, REGION_UI_TO_CODE, ESPEC_CRUDA,
    IDEOLOGIA_INDICE_DEFECTO, CREDITO, PREDICTORES_OCULTOS,
)

# Clave del selector de pregunta en st.session_state. Vive acá y la importa
# app.py, que la necesita para saber qué título de pestaña poner ANTES de
# dibujar el selector. Escrita a mano en los dos lados, un renombre en uno
# dejaría al otro leyendo una clave que no existe y la pestaña se quedaría
# siempre en la pregunta por defecto, en silencio.
CLAVE_PREGUNTA = "seguridad_pregunta"

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


def render_selector_pregunta():
    """
    Selector de la medida punitiva. Devuelve el slug elegido.

    Va ARRIBA del título y no entre los selectores de perfil porque no es una
    característica del lector: es qué se está midiendo. Mezclarlo con edad y
    región lo haría leer como un atributo más del perfil.

    El valor se guarda en session_state bajo CLAVE_PREGUNTA, que es lo que
    app.py lee para titular la pestaña.
    """
    etiquetas = [PREGUNTAS[s]["etiqueta"] for s in SLUGS]
    etiqueta = st.radio(
        "Medida",
        options=etiquetas,
        index=SLUGS.index(PREGUNTA_DEFECTO),
        key=CLAVE_PREGUNTA,
        horizontal=True,
        help="Las cuatro se preguntaron en la misma encuesta, con la misma "
             "escala de acuerdo. Cada una tiene su propio modelo.",
    )
    return ETIQUETA_A_SLUG[etiqueta]


def render_header(model):
    st.markdown(
        f'<h1 class="main-title">{model["pregunta_titulo"]}</h1>',
        unsafe_allow_html=True,
    )
    # El enunciado textual del cuestionario, entre comillas. El widget mide el
    # acuerdo con ESA frase; si arriba se muestra una paráfrasis y el modelo
    # estima otra cosa, el número dice algo distinto de lo que el lector cree.
    st.markdown(
        f'<p class="subtitle">A los encuestados se les leyó esta frase: '
        f'<em>«{model["pregunta_enunciado"]}»</em>.</p>',
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
        # El índice sale de config y no va a mano: apuntaba a la categoría
        # equivocada desde que se pasó de seis tramos a siete —el widget abría
        # en "Centroizquierda (4)" mientras el comentario decía "Centro (5)"—.
        ideol_sel = st.selectbox(
            "Identificación ideológica", options=list(IDEOLOGIA_UI_TO_CODE),
            index=IDEOLOGIA_INDICE_DEFECTO,
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
    # SIN LÍNEA DIVISORIA ACÁ. La había, y ahora el corte entre el formulario y
    # el resultado lo hace el borde de la banda gris del Figma, que empieza
    # justo en este punto. Dejar las dos era una raya suelta sobre el gris.
    st.markdown(f"""
    <div class="prob-bar-wrapper">
        <div class="prob-endpoints">
            <span class="prob-endpoint prob-endpoint--contra">En contra</span>
            <span class="prob-endpoint prob-endpoint--favor">A favor</span>
        </div>
        <div class="prob-container">
            <div class="prob-indicator" style="left: {prob}%;">
                <div class="prob-label">{formato_pct(prob)}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def brecha_nacional(prob_r, nacional_r, intervalo, brecha_iv=None):
    """
    La línea que compara el perfil contra el promedio nacional.

    POR QUÉ ES UNA FUNCIÓN APARTE Y NO TRES LÍNEAS ADENTRO DE LA TARJETA.
    Vivía adentro de `render_result_card()`, que dibuja, así que ningún test la
    podía alcanzar — y por eso se publicó mal. Codex barrió los 4.032 resultados
    (4 preguntas x 1.008 perfiles) el 7/9/2026 y encontró **1.883** en los que el
    widget afirmaba "este perfil está X pp por encima/debajo" con un intervalo
    que CONTIENE el promedio. Su ejemplo: mano dura, hombre de 18-29, secundaria
    o menos, extrema izquierda, no víctima, del interior — mostraba 55%,
    intervalo 27%-77%, promedio 67%, y afirmaba "12pp por debajo". El punto está
    12 pp abajo; ese intervalo no sostiene que el perfil lo esté.

    Es exactamente la prudencia que el widget ya tenía para el 50% —ver
    `interpretar()` y `model.banda_decision()`— y que a esta comparación le
    faltaba.

    QUÉ SE CORRIGE. La AFIRMACIÓN, no el número: la brecha se sigue mostrando,
    porque es información, pero atribuida a la estimación puntual y no al
    perfil. Se compara contra el intervalo REDONDEADO, que es el que ve el
    lector, y de forma inclusiva.

    YA NO SE COMPARA UN INTERVALO CONTRA UN PUNTO. Durante un tiempo esta
    función decidía preguntando si el intervalo DEL PERFIL contenía al promedio
    nacional, tratando al promedio como si fuera exacto. Pero el promedio se
    estima con la misma muestra y tiene su propia incertidumbre. En el docstring
    anterior yo había escrito que ignorarla era "conservador de un solo lado":
    NO ESTABA DEMOSTRADO. La varianza de la resta es Var(perfil) + Var(promedio)
    − 2·Cov, y el signo del efecto depende de esa covarianza, que nadie había
    calculado. Tampoco es siempre positiva —eso también lo escribí y era falso:
    medida perfil por perfil, hay covarianzas negativas de hasta −1,45 pp² en
    tres de las cuatro preguntas—. Si es chica o negativa, el chequeo viejo
    afirma DE MÁS, que es justo la clase de error del que ya se sacaron 1.883
    casos.

    Ahora `train_model` guarda, junto a cada réplica de coeficientes, la tasa
    nacional de ESE mismo remuestreo, y `model.intervalo_brecha()` bootstrapea
    la diferencia directamente: la covarianza entra sola, sin estimarla ni
    suponerle signo. `brecha_iv` es ese intervalo.

    Si no viene —artefacto viejo, sin la tasa por réplica— se cae al chequeo
    anterior, que es lo que había. Peor, pero no roto.
    """
    diff = prob_r - nacional_r
    if not diff:
        return "= este perfil coincide con el promedio"

    arrow = "↑" if diff > 0 else "↓"
    posicion = "por encima" if diff > 0 else "por debajo"

    if brecha_iv is not None:
        # Redondeo inclusivo, pero NO por el motivo que decía este comentario:
        # el intervalo de la resta no se muestra, así que "es lo que ve el
        # lector" era falso —la tarjeta muestra el intervalo del perfil—. El
        # motivo es el otro: la brecha que se publica va redondeada a enteros, y
        # una regla binaria que resuelve más fino que el número que acompaña
        # afirma con una precisión que el texto no tiene. Lo marcó Codex.
        promedio_dentro = round(brecha_iv[0]) <= 0 <= round(brecha_iv[1])
    else:
        promedio_dentro = (
            intervalo is not None
            and round(intervalo[0]) <= nacional_r <= round(intervalo[1])
        )
    if promedio_dentro:
        return (f"{arrow} la estimación puntual queda {abs(diff)}pp {posicion}, "
                "pero el margen de error no permite afirmar la diferencia")
    return f"{arrow} este perfil está {abs(diff)}pp {posicion}"


def render_result_card(model, prob, colors, intervalo=None, banda=None,
                       brecha_iv=None):
    color, texto = interpretar(prob, colors, intervalo, banda)

    # La diferencia se calcula sobre los valores YA redondeados que ve el
    # lector: si en pantalla dicen 53% y 37%, la brecha tiene que decir 16pp.
    # Restar primero y redondear después da 17pp y la cuenta no cierra a la
    # vista, que en una pieza periodística se lee como un error.
    prob_r = round(prob)
    nacional_r = round(model["prob_favor_nacional"])

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
            f'<div class="result-intervalo">Intervalo estimado: '
            f'entre <strong>{formato_pct(bajo)}</strong> y '
            f'<strong>{formato_pct(alto)}</strong></div>'
        )

    brecha = brecha_nacional(prob_r, nacional_r, intervalo, brecha_iv)
    # EL COLOR SIGUE AL SIGNO, como en las diferencias por grupo: azul para el
    # lado "a favor" y naranja para el "en contra", los mismos dos colores que
    # los extremos del gradiente. Estaba cableado en naranja pasara lo que
    # pasara —con un `!important` en la hoja que además pisaba el color en
    # línea que se le pasaba acá—, así que una diferencia positiva salía del
    # color de las negativas. Lo marcó Codex.
    _d = prob_r - nacional_r
    clase_brecha = ("grupo-celda-delta--sube" if _d > 0
                    else "grupo-celda-delta--baja" if _d < 0 else "")

    st.markdown(f"""
    <div class="result-card">
        <div class="result-number" style="color: {color};">{formato_pct(prob)}</div>
        {intervalo_html}
        <div class="result-text">
            El modelo estima que, entre quienes tienen estas características y
            <em>postura definida</em>, ese es el porcentaje que declara
            {model["pregunta_afirma"]}.<br>
            <strong style="color: {color};">{texto}</strong>
        </div>
        <div class="result-nacional">
            Promedio nacional:
            <span class="result-nacional-value">{formato_pct(model["prob_favor_nacional"])}</span>
            <span class="result-nacional-diff {clase_brecha}">
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
    # De la misma lista que el selector, no de un zip entre dos estructuras
    # paralelas: el zip le ponía etiquetas cambiadas a las tasas publicadas si
    # alguien reordenaba el diccionario de la UI.
    **{f"ideol_{nombre}": etiqueta
       for nombre, _, _, etiqueta in ESPEC_CRUDA["ideol_tramos"]},
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
     [f"ideol_{nombre}" for nombre, _, _, _ in ESPEC_CRUDA["ideol_tramos"]]),
    ("Victimización", ["victima", "no_victima"]),
    ("Nivel educativo", ["educ_secundaria", "educ_ter_incompleta", "educ_ter_completa"]),
    ("Edad", ["edad_18_29", "edad_60_plus"]),
    ("Sexo", ["hombres", "mujeres"]),
    ("Región", ["montevideo", "interior"]),
]


def render_comparisons(model):
    """
    Comparación por grupos, con el formato del Figma: una solapa por dimensión
    y adentro una fila de números grandes con su diferencia contra el promedio.

    Reemplaza la lista de dieciséis barras apiladas. El Figma resuelve esto con
    chips por dimensión —"Por religiosidad", "Por educación"…— y muestra sólo la
    dimensión elegida, que es lo que evita el muro. Streamlit no tiene chips,
    pero `st.tabs` con las solapas estilizadas como píldoras da la misma pieza.

    La diferencia contra el promedio nacional se muestra debajo de cada número,
    como en el Figma, y va en color: azul si el grupo está por encima, naranja
    si está por debajo. Es la ÚNICA codificación de color del widget, y no
    valora la medida: dice de qué lado del promedio cae el grupo.
    """
    st.markdown('<hr class="editorial-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Comparación con otros grupos</div>',
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

    # Se arma primero y se dibuja después: una dimensión cuyos grupos estén
    # todos por debajo del mínimo de casos no debe generar una solapa vacía.
    dimensiones = []
    for titulo, claves in GRUPOS_ORDEN:
        celdas = [(k, stats[k]) for k in claves
                  if stats.get(k) is not None and k in GRUPOS_LABEL]
        if celdas:
            dimensiones.append((titulo, celdas))

    if not dimensiones:
        return

    for solapa, (_, celdas) in zip(st.tabs([t for t, _ in dimensiones]), dimensiones):
        with solapa:
            html = []
            for clave, valor in celdas:
                delta_html = ""
                if nacional is not None:
                    # Sobre los valores YA redondeados, que son los que se ven:
                    # restar antes y redondear después deja cuentas que no
                    # cierran a la vista.
                    d = round(valor) - round(nacional)
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

    if nacional is not None:
        st.markdown(
            f'<div class="grupo-nota-ref">La diferencia es contra el promedio '
            f'nacional de esta pregunta ({round(nacional)}%).</div>',
            unsafe_allow_html=True,
        )


# Nombre legible de cada predictor, para poder redactar en castellano qué
# efectos aguantan y cuáles no sin escribir la lista a mano por pregunta.
PREDICTOR_LABEL = {
    "edad_30_44": "tener entre 30 y 44 años",
    "edad_45_59": "tener entre 45 y 59",
    "edad_60_plus": "tener 60 o más",
    "es_mujer": "el sexo",
    "educ_ter_incomp": "la terciaria incompleta",
    "educ_ter_comp": "la terciaria completa",
    "victima_sin_violencia": "haber sido víctima sin violencia",
    "victima_con_violencia": "haber sido víctima con violencia",
    "victima_sin_dato": "no haber contestado sobre victimización",
    "es_montevideo": "vivir en Montevideo",
    "ideol_no_ubica": "no ubicarse en la escala ideológica",
    **{f"ideol_{nombre}": f"ubicarse en {etiqueta.lower()}"
       for nombre, _, _, etiqueta in ESPEC_CRUDA["ideol_tramos"]},
}


def _robustez_md(model):
    """
    El párrafo de "qué sostiene y qué no", redactado desde los datos de la
    validación ordinal que trae el JSON.

    Si el modelo NO trae `robustez`, devuelve una advertencia en vez de
    inventar la lista. La versión anterior tenía las conclusiones escritas a
    mano —"no son robustos el sexo, la región…"— y valían para una sola
    pregunta y un corte ideológico que ya no es el que usa el widget: al pasar
    a cuatro modelos, ese párrafo habría seguido en pantalla afirmando de las
    otras tres algo que nadie verificó.
    """
    rob = model.get("robustez")
    if not rob:
        return (
            "**Qué sostiene y qué no.** Para esta pregunta todavía no se corrió "
            "la validación sobre la escala completa de acuerdo, así que no hay "
            "nada verificado acá sobre qué efectos aguantan al estimarlos de "
            "otra manera. Leé el ordenamiento con esa reserva."
        )

    # Se descartan las dummies que la UI nunca enciende: decirle al lector que
    # no se puede afirmar nada sobre "no ubicarse en la escala" es ruido, porque
    # no es una opción que él pueda elegir. Siguen contadas en el total, que es
    # una propiedad del modelo, no de lo que se ofrece en pantalla.
    frágiles = [PREDICTOR_LABEL.get(p, p) for p in rob.get("no_sostienen", [])
                if p not in PREDICTORES_OCULTOS]
    firmes = [PREDICTOR_LABEL.get(p, p) for p in rob.get("sostienen_top", [])
              if p not in PREDICTORES_OCULTOS]

    partes = [
        "**Qué sostiene y qué no.** Estimado de otra manera —sobre la escala "
        "completa de acuerdo, sin excluir a los neutrales— "
        f"{rob['signos_coincidentes']} de los {rob['n_predictores']} efectos "
        "mantienen el signo."
    ]
    if firmes:
        partes.append(f"Los más grandes aguantan: {_enumerar(firmes)}.")
    if frágiles:
        cuales = _enumerar(frágiles)
        verbo = "Da vuelta el signo" if len(frágiles) == 1 else "Dan vuelta el signo"
        partes.append(
            f"**{verbo} según cómo se estime**: {cuales}. O sea que este widget "
            "no permite afirmar nada sobre eso, aunque el modelo le asigne un "
            "valor."
        )
    else:
        partes.append(
            "Ningún efecto cambia de signo entre las dos formas de estimar."
        )
    return " ".join(partes)


def formato_pct(valor):
    """
    Porcentaje redondeado a entero, salvo cuando redondear diría algo falso.

    `{:.0f}` convierte 0,18% en «0%», y «0%» no es un redondeo: es la
    afirmación de que NADIE con ese perfil está a favor, que el dato no dice.
    Codex encontró 65 perfiles así el 7/9/2026, todos en humillación a los
    presos —60 años o más, ubicados de centroizquierda hacia la izquierda—, con
    estimaciones reales entre 0,176% y 0,499% e intervalos que llegaban al 1-3%.
    El peor mostraba «0%» con intervalo «0% a 1%» sobre un valor de 0,176%.

    Se corrige en los dos extremos por simetría: el mismo redondeo produciría
    «100%» a partir de 99,7%, y afirmar unanimidad es el mismo error dado
    vuelta. Con estas cuatro preguntas no se da hoy, pero el widget se
    re-entrena con otras.
    """
    if 0 < valor < 0.5:
        return "<1%"
    if 99.5 < valor < 100:
        return ">99%"
    return f"{round(valor)}%"


def _enumerar(items):
    """«a, b y c». Vacío si no hay nada."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " y " + items[-1]


def render_methodology(model):
    info = model.get("model_info", {})
    cob = model.get("cobertura_perfiles", {})
    robustez_md = _robustez_md(model)
    # Los dos extremos ideológicos son un solo valor de la escala cada uno, así
    # que son los tramos más chicos; el número sale del JSON y no va escrito a
    # mano, que es como quedó publicando "80 casos" después de mover los bordes.
    tam = model.get("tamanio_tramos_ideologicos", {})
    extremos = _enumerar([
        f"*{etiqueta}* ({tam[f'ideol_{nombre}']} casos)"
        for nombre, _, _, etiqueta in ESPEC_CRUDA["ideol_tramos"]
        if f"ideol_{nombre}" in tam and nombre in ("izq_extrema", "der_extrema")
    ])
    frágiles_md = (
        f"Los tramos más chicos son {extremos}: son los números más frágiles "
        "de esta página."
        if extremos else ""
    )
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

**Qué es el intervalo, y por qué ya no dice "de confianza del 95%".** Se midió
cuánto cubre de verdad: se tomó el modelo como si fuera el mundo, se simularon
resultados desde él y se rehízo todo el procedimiento doscientas veces por
pregunta. Pidiendo el 95% clásico, el intervalo contenía el valor verdadero
entre el 90% y el 93% de las veces, no el 95%. Ahora se pide un percentil más
ancho, calibrado por esa simulación, y el promedio llega al 95%.

Pero el promedio es sobre todos los perfiles, y **el lector recibe el de su
perfil**: hay combinaciones poco frecuentes donde la cobertura sigue siendo
bastante menor. Por eso el rótulo dice "intervalo estimado" a secas y no promete
un 95% que no se puede sostener perfil por perfil. Sigue siendo la mejor medida
disponible de cuánta incertidumbre hay detrás del número, y sigue siendo ancha a
propósito.

**El intervalo también cubre el desacuerdo entre modelos, no sólo el de la
muestra.** Esa simulación mide cuánto se movería el número con otra muestra,
suponiendo que la forma del modelo es la correcta. Pero la forma no está dada:
se probaron nueve maneras razonables de escribirlo —agregando interacciones
entre ideología y educación, entre educación y edad, y así— y la encuesta no
alcanza para decidir cuál es mejor. Distintas maneras dan números distintos para
un mismo perfil. El intervalo que se muestra se estira hasta contener lo que
dicen todas ellas, así que si dos modelos igual de defendibles discrepan, esa
discrepancia está adentro. En tres de las cuatro preguntas casi no cambia nada;
en pena de muerte movió 113 de los 1.008 perfiles.

**Por qué a veces el intervalo no llega al 50% y aun así no se afirma de qué
lado está la mayoría.** El intervalo no se calcula con una fórmula cerrada: se
simula, remuestreando la encuesta mil veces. Un extremo que cae en 49% podría
haber caído en 51% con otra simulación, así que para afirmar que la mayoría está
de un lado se exige un margen un poco más ancho que el que se muestra. Cuando el
extremo queda pegado al 50%, el widget prefiere no afirmar.

**Sobre la escala ideológica.** La pregunta fue: *"en una escala donde cero es
la extrema izquierda y 10 es la extrema derecha, ¿dónde se ubicaría usted?"*.
Va de 0 a 10, así que **el 5 es el punto medio exacto**, y es la respuesta más
elegida: un tercio de los encuestados se ubica ahí. Los siete tramos son los
que trae la base etiquetada de la encuesta, simétricos alrededor de ese centro,
para que "extrema izquierda" y "extrema derecha" abarquen lo mismo y se puedan
comparar. {frágiles_md}

{robustez_md}

Las tasas por grupo que se muestran abajo son descriptivas de la muestra, no
efectos ajustados: mezclan el efecto propio del grupo con el de todo lo demás
que lo acompaña.
        """)


def render_footer(model):
    # La fuente y el crédito salen del JSON, no de una constante importada acá:
    # así el pie dice de qué encuesta salieron ESTOS coeficientes y no de cuál
    # salen los que se entrenarían hoy. El fallback a config es para un JSON
    # viejo, anterior a que el crédito se serializara.
    st.markdown(f"""
    <p class="footer-text">
        Fuente: {model.get('fuente', 'Encuesta El Observador')}<br>
        {model.get('credito', CREDITO)} ·
        Modelo actualizado el {model.get('entrenado', '—')}
    </p>
    """, unsafe_allow_html=True)

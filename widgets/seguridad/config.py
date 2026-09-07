"""
Configuración del widget de seguridad pública.

La pregunta que modela el widget está PARAMETRIZADA: se elige cambiando
`PREGUNTA_ACTIVA` y re-entrenando. Todas las candidatas comparten la misma
escala Likert 1-5, así que el pipeline no cambia al cambiar de pregunta.

Colores y umbrales vienen de shared.config.
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.config import (  # noqa: E402
    LIGHT_COLORS, DARK_COLORS, COLORS,
    get_colors, PROB_THRESHOLDS, get_interpretation,
)

# ============================================================
# RUTAS
# ============================================================
WIDGET_DIR = Path(__file__).parent

# Un JSON por pregunta. Antes había uno solo (`model_coefficients.json`) porque
# el widget publicaba una pregunta por vez; ahora el lector elige entre las
# cuatro y hacen falta los cuatro juegos de coeficientes al mismo tiempo.
MODELOS_DIR = WIDGET_DIR / "modelos"


def ruta_modelo(slug):
    """Ruta del JSON de coeficientes de una pregunta."""
    return MODELOS_DIR / f"model_{slug}.json"

# La base vive en el repo de encuestas, no en éste (son datos del cliente y
# el .gitignore excluye *.csv). Se puede pisar con la variable de entorno
# SEGURIDAD_DATA_FILE para correr desde otra máquina.
_DEFAULT_DATA = (
    Path.home() / "dev/trabajo/Observador/Observador/Observador-encuesta"
    / "encuestas/observador_2026_05_seguridad/output/base_etiquetada.csv"
)
DATA_FILE = Path(os.environ.get("SEGURIDAD_DATA_FILE", _DEFAULT_DATA))

# ============================================================
# PREGUNTAS CANDIDATAS
# ============================================================
# Las CUATRO que pidió Tomer (WhatsApp, 7/9/2026), en su orden y con las
# columnas que él mismo señaló en la base etiquetada: AC, Z, Y y AA.
#
# El widget publica las cuatro a la vez y el lector elige cuál estimar, así que
# hay un modelo entrenado por pregunta. Antes se publicaba una sola y cambiar
# de pregunta era editar una constante y re-entrenar.
#
# Se cayó `aumentar_penas` (var_228, "Se deberían aumentar las penas para todos
# los delitos"): Tomer no la incluyó en su lista y la columna tampoco viene en
# la base recortada que mandó. Está en la base completa, así que volver a
# sumarla es agregar la entrada acá y re-entrenar.
#
#   columna    : nombre exacto en base_etiquetada.csv
#   enunciado  : el texto TEXTUAL del cuestionario, que se muestra al lector.
#                No es decorativo: el widget mide acuerdo con esa frase exacta,
#                y parafrasearla cambia lo que el numero significa.
#   etiqueta   : el nombre corto con el que la pregunta aparece en el selector
#   titulo     : encabezado del widget
#   titulo_corto: para la pestaña del navegador
#   afirma     : que significa estar de acuerdo, para redactar el resultado
PREGUNTAS = {
    "politico_mano_dura": {
        "columna": "var_233 | Votaria politico de mano dura",
        "enunciado": "Votaría a un político que promoviera castigos más duros para los delincuentes",
        "etiqueta": "Votar a un político de mano dura",
        "titulo": "¿Quiénes votarían a un político de mano dura?",
        "titulo_corto": "¿Votarías mano dura?",
        "afirma": "votar a un político que promueva castigos más duros para los delincuentes",
    },
    "cadena_perpetua": {
        "columna": "var_230 | Cadena perpetua tres delitos",
        "enunciado": ("Una persona que ha sido condenada por tres delitos graves debería "
                      "recibir cadena perpetua sin posibilidad de libertad condicional"),
        "etiqueta": "Cadena perpetua por tres delitos graves",
        "titulo": "¿Quiénes apoyan la cadena perpetua en Uruguay?",
        "titulo_corto": "¿Apoyás la cadena perpetua?",
        "afirma": ("estar de acuerdo con la cadena perpetua sin libertad condicional "
                   "para quien fue condenado por tres delitos graves"),
    },
    "pena_muerte": {
        "columna": "var_229 | Pena de muerte por homicidio",
        "enunciado": "Una persona condenada por homicidio debería recibir la pena de muerte",
        "etiqueta": "Pena de muerte por homicidio",
        "titulo": "¿Quiénes apoyan la pena de muerte en Uruguay?",
        "titulo_corto": "¿Apoyás la pena de muerte?",
        "afirma": "estar de acuerdo con que una persona condenada por homicidio reciba la pena de muerte",
    },
    "humillacion_presos": {
        "columna": "var_231 | Presos merecen humillacion",
        "enunciado": ("Quienes están presos merecen la humillación, intimidación y "
                      "degradación que allí puedan recibir"),
        "etiqueta": "Humillación a los presos",
        "titulo": "¿Quiénes creen que los presos merecen humillación?",
        "titulo_corto": "¿Los presos merecen humillación?",
        "afirma": ("estar de acuerdo con que quienes están presos merecen la humillación, "
                   "intimidación y degradación que allí puedan recibir"),
    },
}

# La que aparece seleccionada al abrir: la primera de la lista de Tomer.
#
# Las cuatro NO son intercambiables como puerta de entrada, y conviene tenerlo
# a la vista. Apoyo ponderado entre quienes tienen postura definida:
# cadena perpetua 78,6% · mano dura 67,0% · pena de muerte 36,7% ·
# humillación a los presos 11,0%. Las dos de los extremos dan perfiles casi
# planos —con 11% de apoyo casi ningún perfil se despega— así que abrir en
# ellas haría parecer que el widget no discrimina. Mano dura reparte mejor y
# además es la que él puso primero.
PREGUNTA_DEFECTO = "politico_mano_dura"

# El orden del selector es el del dict, que es el de la lista de Tomer.
SLUGS = list(PREGUNTAS)

# Etiqueta visible -> slug, para el selector.
ETIQUETA_A_SLUG = {PREGUNTAS[s]["etiqueta"]: s for s in SLUGS}

# ============================================================
# ESCALA LIKERT
# ============================================================
# Mismo criterio que el widget IVE: ≥4 a favor, ≤2 en contra, 3 excluido
# (se modela aparte como "no toma posición").
LIKERT_MAP = {
    "Totalmente en desacuerdo": 1,
    "En desacuerdo": 2,
    "Ni de acuerdo ni en desacuerdo": 3,
    "De acuerdo": 4,
    "Totalmente de acuerdo": 5,
}
LIKERT_FAVOR = (4, 5)
LIKERT_CONTRA = (1, 2)
LIKERT_NEUTRAL = 3

PONDERADOR = "w_norm"

# ============================================================
# CRÉDITOS
# ============================================================
# Tomer, 7/9/2026: "en la fuente, siempre es la encuesta de El
# Observador-UMAD-Ferreira y el crédito tuyo".
#
# Van en una constante y se serializan dentro de cada JSON: si vivieran sólo en
# el componente del pie, un modelo publicado quedaría sin decir de qué encuesta
# salió en cuanto alguien reordenara la UI.
FUENTE = "Encuesta El Observador-UMAD-Ferreira sobre seguridad pública, mayo de 2026"
CREDITO = "Análisis y desarrollo: Juan Ignacio Pintos Elso"

# ============================================================
# MAPEOS UI -> CÓDIGO DEL MODELO
# ============================================================
EDAD_UI_TO_CODE = {
    "18-29 años": 1,   # referencia
    "30-44 años": 2,
    "45-59 años": 3,
    "60 años o más": 4,
}

# Tres categorías: "Primaria o menos" se unió con "Secundaria" porque sola
# tenía 28 casos y era la referencia del modelo. Ver EDUC_COLAPSO en
# train_model.py.
EDUC_UI_TO_CODE = {
    "Secundaria o menos": 1,   # referencia
    "Terciaria incompleta": 2,
    "Terciaria completa o más": 3,
}

# Autoubicación en la escala de 0 a 10, en siete tramos simétricos.
#
# EL REPARTO NO SE ELIGE ACÁ: lo fijó Tomer al mandar la base ya etiquetada
# (`base_etiquetada_SEGURIDAD_para probabilidad.xlsx`, 7/9/2026), donde
# `var_242` viene con la etiqueta en vez del número. Estos siete tramos son la
# transcripción exacta de esa columna — se verificó cruzándola contra la
# columna numérica de la base completa, valor por valor, en los 3.377 casos.
# El test `test_tramos_ideologicos_reproducen_la_base_etiquetada` deja fijos los
# conteos resultantes para que correr un borde no pase inadvertido.
#
# Es un reparto simétrico alrededor del 5 (anchos 1, 2, 2, 1, 2, 2, 1), que es
# lo que hace comparables los dos extremos. La versión anterior también era
# simétrica pero con otros anchos (2, 2, 1, 1, 1, 2, 2), así que los
# coeficientes ideológicos de este modelo NO son comparables con los de aquélla
# aunque las dummies se sigan llamando igual.
#
# El 5 sigue siendo la referencia: en una escala de 0 a 10 es el punto medio
# exacto y es la respuesta modal (1.092 de 3.377 casos, el 32%). Una referencia
# chica hace que todos los coeficientes se estimen contra pocos casos, que es el
# problema que ya hubo con "Primaria o menos" y sus 28 casos.
#
# CAVEAT DE TAMAÑO: con este corte "Extrema izquierda" es sólo el 0 y queda en
# 42 casos —contra los 80 que tenía cuando abarcaba 0-1—, y "Extrema derecha"
# es sólo el 10, con 131. Entran al modelo, pero son los dos números más
# frágiles del gráfico comparativo y no conviene titular con ellos.
#
# La referencia es el Centro por ser el tramo modal: una referencia chica hace
# que todos los coeficientes se estimen contra pocos casos, que es el problema
# que ya hubo con "Primaria o menos" y sus 28 casos.
#
# "No se ubica" NO se ofrece en la UI. Esa dummy agrupa a quienes no
# contestaron la escala: 80 en la encuesta completa, de los cuales 62 entran al
# modelo principal de apoyo (los otros no tienen postura definida y quedan en el
# modelo secundario de neutralidad). Y no contestar no es lo
# mismo que ubicarse en el centro: el cuestionario NO ofrece "no sabe" entre
# las opciones —son exactamente los once valores— así que un nulo es una
# pregunta salteada. Sigue existiendo como predictor, para que esos casos no
# contaminen la referencia, pero queda siempre en cero desde la interfaz.
# Especificación de la transformación dato crudo -> dummy. Vive acá, y no
# repartida entre config y train_model, porque TODA ella tiene que entrar en la
# huella: cambiar cualquiera de estos valores cambia el significado de las
# dummies aunque los nombres queden iguales.
ESPEC_CRUDA = {
    # Escala nivel_educativo (1-10) del proveedor -> las 3 categorías del modelo.
    "educ_colapso": {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 3, 9: 3, 10: 3},
    # Cortes de edad (bordes de pd.cut) y sus códigos.
    "edad_cortes": [17, 29, 44, 59, 120],
    # Tramos de la autoubicación 0-10. Cada uno es [desde, hasta] inclusive y
    # el nombre es el sufijo de la dummy. El tramo marcado como referencia no
    # genera dummy. Van explícitos y no como dos umbrales sueltos porque ahora
    # son seis cortes y un umbral no alcanza para describirlos.
    # [sufijo de la dummy, desde, hasta, etiqueta] — inclusive en los dos
    # bordes. Es la FUENTE ÚNICA: de acá salen las dummies del entrenamiento,
    # las de la inferencia, las opciones del selector y las etiquetas del
    # bloque comparativo. Antes las etiquetas vivían en un diccionario aparte
    # y la correspondencia era posicional: reordenarlo dejaba los 71 tests en
    # verde y le ponía nombres cambiados a las tasas publicadas.
    "ideol_tramos": [
        ["izq_extrema", 0, 0, "Extrema izquierda"],
        ["izquierda", 1, 2, "Izquierda"],
        ["centroizq", 3, 4, "Centroizquierda"],
        ["centro", 5, 5, "Centro"],
        ["centroderecha", 6, 7, "Centroderecha"],
        ["derecha", 8, 9, "Derecha"],
        ["der_extrema", 10, 10, "Extrema derecha"],
    ],
    "ideol_referencia": "centro",
    # Códigos crudos de las demás variables.
    "sexo_valores": {"mujer": "Mujer", "hombre": "Hombre"},
    "dpto_montevideo": 1,
    # Etiquetas crudas de victimización. Viven acá y no en train_model.py para
    # que entren en la huella: son parte de la definición de las dummies.
    "victima_etiquetas": {
        "no": ["no"],
        "sin_violencia": ["sí  sin violencia", "si  sin violencia", "sí sin violencia"],
        "con_violencia": ["sí  con violencia", "si  con violencia", "sí con violencia"],
    },
}

def _etiqueta_tramo(desde, hasta, etiqueta):
    """«Centro (5)» si el tramo es un valor solo, «Izquierda (2-3)» si son dos."""
    rango = f"{desde}" if desde == hasta else f"{desde}-{hasta}"
    return f"{etiqueta} ({rango})"


# Derivado de ESPEC_CRUDA["ideol_tramos"], no escrito a mano: el orden, los
# valores de la escala y el nombre de la dummy salen todos de la misma lista,
# así que no pueden desalinearse entre sí.
IDEOLOGIA_UI_TO_CODE = {
    _etiqueta_tramo(desde, hasta, etiqueta): i
    for i, (_, desde, hasta, etiqueta) in enumerate(ESPEC_CRUDA["ideol_tramos"], start=1)
}

# Índice del selector: el tramo de referencia, que es el modal. Iba a mano y
# quedó apuntando a la categoría equivocada al pasar de seis tramos a siete —
# el widget abría en "Centroizquierda (4)" mientras el comentario decía
# "Centro (5)".
IDEOLOGIA_INDICE_DEFECTO = next(
    i for i, (nombre, _, _, _) in enumerate(ESPEC_CRUDA["ideol_tramos"])
    if nombre == ESPEC_CRUDA["ideol_referencia"]
)


VICTIMA_UI_TO_CODE = {
    "No": 1,                       # referencia
    "Sí, sin violencia": 2,
    "Sí, con violencia": 3,
}

REGION_UI_TO_CODE = {
    "Montevideo": 1,
    "Interior": 0,
}

# ============================================================
# PREDICTORES DEL MODELO
# ============================================================
# El orden importa sólo para la legibilidad de los reportes; el vector se
# arma por nombre, no por posición.
# Sin voto de balotaje: Tomer pidió expresamente "poner identificación
# ideológica y sacar partidos políticos" (31/8/2026). Es una decisión
# editorial suya, no un problema del modelo — el balotaje discriminaba bien.
# Consecuencia estadística a tener presente: parte de lo que antes explicaba
# el voto ahora lo absorbe la ideología declarada, así que los coeficientes
# ideológicos de este modelo NO son comparables con los de la versión anterior.
PREDICTORES = [
    "edad_30_44", "edad_45_59", "edad_60_plus",
    "es_mujer",
    "educ_ter_incomp", "educ_ter_comp",
    "ideol_izq_extrema", "ideol_izquierda", "ideol_centroizq",
    "ideol_centroderecha", "ideol_derecha", "ideol_der_extrema",
    "ideol_no_ubica",
    "victima_sin_violencia", "victima_con_violencia", "victima_sin_dato",
    "es_montevideo",
]

# Los dos predictores que la UI NUNCA enciende: agrupan a quienes no
# contestaron esas preguntas. Existen para que esos casos no contaminen las
# categorías de referencia, pero no son opciones que el lector pueda elegir, así
# que no tienen por qué aparecer en el texto que él lee. La lista vive acá para
# que la metodología los pueda descartar sin repetir los nombres a mano.
PREDICTORES_OCULTOS = ("ideol_no_ubica", "victima_sin_dato")

# `victima_sin_dato` existe para que los 53 casos sin respuesta no se mezclen
# con quienes contestaron "No" —eso contaminaba la referencia y la tasa que se
# publica del grupo "No fue víctima"—, pero en la UI queda siempre en cero: el
# widget obliga a elegir una de las tres opciones reales. Es un predictor de
# entrenamiento, no de interacción.



def huella_contrato(slug):
    """
    Huella de TODO el contrato entre la configuración y el modelo entrenado de
    la pregunta `slug`.

    Va por pregunta y no una sola para todas: los cuatro modelos comparten los
    mapeos y la codificación, pero cada JSON tiene que poder decir de qué
    pregunta es. Si la huella fuera común, mover un JSON encima de otro pasaría
    el chequeo y el widget serviría los coeficientes de la pregunta equivocada
    con el título correcto.

    Verificar sólo que no falten predictores es un chequeo de subconjunto y deja
    pasar el caso peligroso: que un nombre de dummy siga existiendo pero
    signifique otra cosa. Pasó de verdad al colapsar educación de cuatro
    categorías a tres — `educ_ter_incomp` sobrevivió con el mismo nombre y otro
    código detrás, así que un JSON viejo habría cargado sin protestar y la
    inferencia habría aplicado coeficientes de otra codificación.

    Incluye TODO lo que define el significado de una dummy: los mapeos de la UI,
    las referencias, la escala Likert y la especificación de transformación
    cruda (cortes de edad, colapso educativo, tramos ideológicos). Una versión
    anterior de esta función decía en su docstring que cubría las referencias y
    no las cubría, y sobre todo dejaba afuera `EDUC_COLAPSO` — con lo cual
    cambiar el colapso educativo dejaba exactamente la misma huella, que es
    justo el agujero que esto viene a tapar.

    Los mapeos se ordenan antes de hashear: reordenar `PREDICTORES` no cambia el
    modelo y no debería invalidar el JSON.
    """
    import hashlib
    import json as _json
    if slug not in PREGUNTAS:
        raise KeyError(f"pregunta desconocida: {slug!r}. Son {SLUGS}.")
    material = _json.dumps({
        "pregunta": slug,
        "columna": PREGUNTAS[slug]["columna"],
        # ordenados: el orden de la lista no tiene significado semántico
        "predictores": sorted(PREDICTORES),
        # los mapeos SÍ van como listas ordenadas de pares: cambiar qué opción
        # ofrece la UI, o qué código le corresponde, cambia el contrato
        "edad": sorted(EDAD_UI_TO_CODE.items()),
        "educacion": sorted(EDUC_UI_TO_CODE.items()),
        "ideologia": sorted(IDEOLOGIA_UI_TO_CODE.items()),
        "victima": sorted(VICTIMA_UI_TO_CODE.items()),
        "region": sorted(REGION_UI_TO_CODE.items()),
        "referencias": sorted(REFERENCIAS.items()),
        "likert": sorted(LIKERT_MAP.items()),
        "favor": sorted(LIKERT_FAVOR),
        "contra": sorted(LIKERT_CONTRA),
        "neutral": LIKERT_NEUTRAL,
        "ponderador": PONDERADOR,
        "espec_cruda": _json.dumps(ESPEC_CRUDA, sort_keys=True),
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


REFERENCIAS = {
    "edad": "18-29 años",
    "sexo": "Hombre",
    "educacion": "Secundaria o menos",
    "ideologia": _etiqueta_tramo(*[t[1:] for t in ESPEC_CRUDA["ideol_tramos"]
                                  if t[0] == ESPEC_CRUDA["ideol_referencia"]][0]),
    "victima": "No fue víctima",
    "region": "Interior",
}

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
MODEL_COEFFICIENTS_PATH = WIDGET_DIR / "model_coefficients.json"

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
# Tomer pidió explorar "qué pregunta rinde más" con Nicolás Trajtenberg.
# Todas son Likert 1-5 con el mismo formato de respuesta, así que cambiar de
# pregunta es cambiar PREGUNTA_ACTIVA y correr train_model.py de nuevo.
#
#   columna    : nombre exacto en base_etiquetada.csv
#   enunciado  : el texto TEXTUAL del cuestionario, que se muestra al lector.
#                No es decorativo: el widget mide acuerdo con esa frase exacta,
#                y parafrasearla cambia lo que el numero significa.
#   titulo     : encabezado del widget
#   titulo_corto: para la pestaña del navegador
#   afirma     : que significa estar de acuerdo, para redactar el resultado
PREGUNTAS = {
    "pena_muerte": {
        "columna": "var_229 | Pena de muerte por homicidio",
        "enunciado": "Una persona condenada por homicidio debería recibir la pena de muerte",
        "titulo": "¿Quiénes apoyan la pena de muerte en Uruguay?",
        "titulo_corto": "¿Apoyás la pena de muerte?",
        "afirma": "estar de acuerdo con que una persona condenada por homicidio reciba la pena de muerte",
    },
    "cadena_perpetua": {
        "columna": "var_230 | Cadena perpetua tres delitos",
        "enunciado": ("Una persona que ha sido condenada por tres delitos graves debería "
                      "recibir cadena perpetua sin posibilidad de libertad condicional"),
        "titulo": "¿Quiénes apoyan la cadena perpetua en Uruguay?",
        "titulo_corto": "¿Apoyás la cadena perpetua?",
        "afirma": ("estar de acuerdo con la cadena perpetua sin libertad condicional "
                   "para quien fue condenado por tres delitos graves"),
    },
    "aumentar_penas": {
        "columna": "var_228 | Aumentar penas todos los delitos",
        "enunciado": "Se deberían aumentar las penas para todos los delitos",
        "titulo": "¿Quiénes apoyan aumentar las penas en Uruguay?",
        "titulo_corto": "¿Apoyás aumentar las penas?",
        "afirma": "estar de acuerdo con que se aumenten las penas para todos los delitos",
    },
    "politico_mano_dura": {
        "columna": "var_233 | Votaria politico de mano dura",
        "enunciado": "Votaría a un político que promoviera castigos más duros para los delincuentes",
        "titulo": "¿Quiénes votarían a un político de mano dura?",
        "titulo_corto": "¿Votarías mano dura?",
        "afirma": "votar a un político que promueva castigos más duros para los delincuentes",
    },
    "humillacion_presos": {
        "columna": "var_231 | Presos merecen humillacion",
        "enunciado": ("Quienes están presos merecen la humillación, intimidación y "
                      "degradación que allí puedan recibir"),
        "titulo": "¿Quiénes creen que los presos merecen humillación?",
        "titulo_corto": "¿Los presos merecen humillación?",
        "afirma": ("estar de acuerdo con que quienes están presos merecen la humillación, "
                   "intimidación y degradación que allí puedan recibir"),
    },
}

PREGUNTA_ACTIVA = "pena_muerte"
PREGUNTA = PREGUNTAS[PREGUNTA_ACTIVA]

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
# Parte del esquema que pidió Tomer (WhatsApp, 31/8/2026) —extrema izquierda,
# izquierda, centroizquierda, centroderecha, derecha, extrema derecha— y le
# agrega "Centro", que es lo que falta al pasarlo a la escala real.
#
# POR QUÉ NO SE USA SU REPARTO TAL CUAL. Él lo armó sobre una escala de 1 a 10.
# La del cuestionario va de 0 a 10, y el enunciado es explícito: "en una escala
# donde CERO es la extrema izquierda y 10 es la extrema derecha". O sea que el
# 0 no es un valor residual: es, por definición de la pregunta, el extremo. Su
# tramo más a la izquierda arranca en 1 y deja afuera justo ese valor.
#
# Y en una escala de 1 a 10 el punto medio es 5,5, así que llamar
# "centroizquierda" al 5 cierra; en una de 0 a 10 el 5 es el medio EXACTO y es
# la respuesta modal (842 de 2.672 casos, el 32%).
#
# POR QUÉ SIETE Y NO SEIS. Con once valores y el 5 solo en el centro, cualquier
# reparto simétrico tiene que tener una cantidad IMPAR de categorías. Con seis
# no hay forma: o el extremo izquierdo abarca tres valores contra dos del
# derecho, o queda uno contra tres. Esa asimetría rompe justamente la
# comparación en la que se apoya el hallazgo principal —que el extremo derecho
# se despega— porque compararía una red ancha contra una angosta.
#
# Los siete tramos son simétricos alrededor del 5: anchos 2, 2, 1, 1, 1, 2, 2.
#
# CAVEAT DE TAMAÑO: "Izquierda extrema" (0-1) tiene 80 casos, el tramo más
# chico. Es suficiente para entrar en el modelo pero es el número más frágil
# del gráfico comparativo; conviene no titular con él.
#
# La referencia es el Centro por ser el tramo modal: una referencia chica hace
# que todos los coeficientes se estimen contra pocos casos, que es el problema
# que ya hubo con "Primaria o menos" y sus 28 casos.
#
# "No se ubica" NO se ofrece en la UI. En el entrenamiento esa dummy agrupa a
# los 80 encuestados que no contestaron la escala. Y no contestar no es lo
# mismo que ubicarse en el centro: el cuestionario NO ofrece "no sabe" entre
# las opciones —son exactamente los once valores— así que un nulo es una
# pregunta salteada. Sigue existiendo como predictor, para que esos casos no
# contaminen la referencia, pero queda siempre en cero desde la interfaz.
IDEOLOGIA_UI_TO_CODE = {
    "Izquierda extrema (0-1)": 1,
    "Izquierda (2-3)": 2,
    "Centroizquierda (4)": 3,
    "Centro (5)": 4,               # referencia
    "Centroderecha (6)": 5,
    "Derecha (7-8)": 6,
    "Derecha extrema (9-10)": 7,
}

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

# `victima_sin_dato` existe para que los 53 casos sin respuesta no se mezclen
# con quienes contestaron "No" —eso contaminaba la referencia y la tasa que se
# publica del grupo "No fue víctima"—, pero en la UI queda siempre en cero: el
# widget obliga a elegir una de las tres opciones reales. Es un predictor de
# entrenamiento, no de interacción.

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
    "ideol_tramos": [
        ["izq_extrema", 0, 1],
        ["izquierda", 2, 3],
        ["centroizq", 4, 4],
        ["centro", 5, 5],
        ["centroderecha", 6, 6],
        ["derecha", 7, 8],
        ["der_extrema", 9, 10],
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


def huella_contrato():
    """
    Huella de TODO el contrato entre la configuración y el modelo entrenado.

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
    material = _json.dumps({
        "pregunta": PREGUNTA_ACTIVA,
        "columna": PREGUNTA["columna"],
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
    "ideologia": "Centro (5)",
    "victima": "No fue víctima",
    "region": "Interior",
}

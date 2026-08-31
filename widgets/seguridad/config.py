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

# Autoubicación 0-10 agrupada. El centro es la referencia porque es la
# categoría modal (32% de la muestra se ubica en el 5 exacto).
#
# "No se ubica" NO se ofrece en la UI. En el entrenamiento esa dummy agrupa a
# los 80 encuestados que no contestaron la escala, y no contestar una encuesta
# no es lo mismo que no ubicarse políticamente: ofrecérsela al lector le
# aplicaría el coeficiente de un grupo definido por otra cosa. Sigue existiendo
# como predictor —para que esos casos no contaminen el centro, que es la
# referencia— pero queda siempre en cero desde la interfaz, igual que
# victima_sin_dato.
IDEOLOGIA_UI_TO_CODE = {
    "Izquierda (0-3)": 1,
    "Centro (4-6)": 2,        # referencia
    "Derecha (7-10)": 3,
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

# Voto en el balotaje 2024. Es el control que el widget IVE ya tenía y a éste
# le faltaba: mejora la discriminación (AUC 0,748 -> 0,752) y sobre todo evita
# atribuirle a la ideología declarada un efecto que en parte es del voto.
#
# La referencia es "blanco, anulado o no votó" (códigos 3 y 4), y no es una
# categoría de descarte: es justamente el grupo MÁS punitivo de la muestra.
# Tanto los votantes de Orsi como los de Delgado apoyan menos la pena de muerte
# que quienes no eligieron ninguno.
#
# Los 153 que NO RECUERDAN a quién votaron (código 5) llevan dummy propia y no
# entran en esa referencia: no acordarse no es lo mismo que haber votado en
# blanco, y meterlos juntos hacía que la etiqueta publicada dijera una cosa y
# el grupo fuera otra. Como ideol_no_ubica y victima_sin_dato, la UI no la
# ofrece y queda siempre en cero.
BALOTAJE_UI_TO_CODE = {
    "Orsi": 1,
    "Delgado": 2,
    "Blanco, anulado o no votó": 0,   # referencia
}

# ============================================================
# PREDICTORES DEL MODELO
# ============================================================
# El orden importa sólo para la legibilidad de los reportes; el vector se
# arma por nombre, no por posición.
PREDICTORES = [
    "edad_30_44", "edad_45_59", "edad_60_plus",
    "es_mujer",
    "educ_ter_incomp", "educ_ter_comp",
    "ideol_izquierda", "ideol_derecha", "ideol_no_ubica",
    "victima_sin_violencia", "victima_con_violencia", "victima_sin_dato",
    "es_montevideo",
    "bal_orsi", "bal_delgado", "bal_no_recuerda",
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
    # Corte de la autoubicación 0-10 en izquierda / centro / derecha.
    "ideol_izquierda_hasta": 3,
    "ideol_derecha_desde": 7,
    # Códigos crudos de las demás variables.
    "sexo_mujer": "Mujer",
    "dpto_montevideo": 1,
    "balotaje_codigos": {"orsi": 1, "delgado": 2, "referencia": [3, 4], "no_recuerda": 5},
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
    cruda (cortes de edad, colapso educativo, códigos de balotaje). Una versión
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
        "balotaje": sorted(BALOTAJE_UI_TO_CODE.items()),
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
    "ideologia": "Centro (4-6)",
    "victima": "No fue víctima",
    "region": "Interior",
    "balotaje": "Blanco, anulado o no votó",
}

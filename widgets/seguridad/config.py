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
#   columna  : nombre exacto en base_etiquetada.csv
#   titulo   : el que se muestra como pregunta del widget
#   afirma   : qué significa "estar de acuerdo" (para redactar el resultado)
#   verbo    : cómo se lee la probabilidad en la UI
PREGUNTAS = {
    "pena_muerte": {
        "columna": "var_229 | Pena de muerte por homicidio",
        "titulo": "¿Cuál es tu probabilidad de apoyar la pena de muerte para homicidas?",
        "titulo_corto": "¿Apoyás la pena de muerte?",
        "afirma": "apoyar la pena de muerte para quien comete un homicidio",
        "verbo": "apoyes",
    },
    "cadena_perpetua": {
        "columna": "var_230 | Cadena perpetua tres delitos",
        "titulo": "¿Cuál es tu probabilidad de apoyar la cadena perpetua por tres delitos?",
        "titulo_corto": "¿Apoyás la cadena perpetua?",
        "afirma": "apoyar la cadena perpetua para quien comete tres delitos graves",
        "verbo": "apoyes",
    },
    "aumentar_penas": {
        "columna": "var_228 | Aumentar penas todos los delitos",
        "titulo": "¿Cuál es tu probabilidad de apoyar el aumento de penas para todos los delitos?",
        "titulo_corto": "¿Apoyás aumentar las penas?",
        "afirma": "apoyar el aumento de penas para todos los delitos",
        "verbo": "apoyes",
    },
    "politico_mano_dura": {
        "columna": "var_233 | Votaria politico de mano dura",
        "titulo": "¿Cuál es tu probabilidad de votar a un político de mano dura?",
        "titulo_corto": "¿Votarías mano dura?",
        "afirma": "votar a un político que prometa mano dura contra el delito",
        "verbo": "lo votes",
    },
    "humillacion_presos": {
        "columna": "var_231 | Presos merecen humillacion",
        "titulo": "¿Cuál es tu probabilidad de creer que los presos merecen ser humillados?",
        "titulo_corto": "¿Los presos merecen humillación?",
        "afirma": "creer que quien está preso merece pasar por situaciones humillantes",
        "verbo": "lo creas",
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
IDEOLOGIA_UI_TO_CODE = {
    "Izquierda (0-3)": 1,
    "Centro (4-6)": 2,        # referencia
    "Derecha (7-10)": 3,
    "No se ubica": 4,
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
PREDICTORES = [
    "edad_30_44", "edad_45_59", "edad_60_plus",
    "es_mujer",
    "educ_ter_incomp", "educ_ter_comp",
    "ideol_izquierda", "ideol_derecha", "ideol_no_ubica",
    "victima_sin_violencia", "victima_con_violencia", "victima_sin_dato",
    "es_montevideo",
]

# `victima_sin_dato` existe para que los 53 casos sin respuesta no se mezclen
# con quienes contestaron "No" —eso contaminaba la referencia y la tasa que se
# publica del grupo "No fue víctima"—, pero en la UI queda siempre en cero: el
# widget obliga a elegir una de las tres opciones reales. Es un predictor de
# entrenamiento, no de interacción.

REFERENCIAS = {
    "edad": "18-29 años",
    "sexo": "Hombre",
    "educacion": "Secundaria o menos",
    "ideologia": "Centro (4-6)",
    "victima": "No fue víctima",
    "region": "Interior",
}

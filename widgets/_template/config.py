"""
Config del widget [NOMBRE].
Reemplazá WIDGET_NAME, WIDGET_SLUG, y definí los mapeos propios.
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.config import LIGHT_COLORS, DARK_COLORS, COLORS, get_colors, get_interpretation

WIDGET_NAME = "Widget [NOMBRE]"
WIDGET_SLUG = "template"

WIDGET_DIR = Path(__file__).parent
MODEL_COEFFICIENTS_PATH = WIDGET_DIR / "model_coefficients.json"
DATA_FILE = _ROOT.parent / "base_limpia.csv"

# Definí aquí los mapeos de inputs a código del modelo
# Ejemplo:
# CATEGORY_UI_TO_CODE = {
#     "Opción A": "a",
#     "Opción B": "b",
# }

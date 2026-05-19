"""
Configuración específica del widget IVE.
Rutas, mapeos y constantes propias de este widget.
Colores y umbrales vienen de shared.config.
"""

import sys
from pathlib import Path

# Asegurar que la raíz del proyecto esté en sys.path
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
# base_limpia.csv está en el directorio padre del repo
DATA_FILE = _ROOT.parent / "base_limpia.csv"

# ============================================================
# MAPEO BALOTAJE (UI label -> código del modelo)
# ============================================================
BALOTAJE_UI_TO_CODE = {
    "No votó/Blanco": "otros",
    "Martínez (FA)": "martinez",
    "Lacalle (Coalición)": "lacalle",
}

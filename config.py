"""
Configuraci\u00f3n central del widget IVE.
Rutas, constantes, paleta de colores y mapeos.
"""

from pathlib import Path

# ============================================================
# RUTAS
# ============================================================
WIDGET_DIR = Path(__file__).parent
MODEL_COEFFICIENTS_PATH = WIDGET_DIR / "model_coefficients.json"
BASE_DIR = WIDGET_DIR.parent
DATA_FILE = BASE_DIR / "base_limpia.csv"

# ============================================================
# PALETA DE COLORES (Economist + El Observador editorial)
# ============================================================
LIGHT_COLORS = {
    "primary": "#2E45B8",
    "accent": "#E3120B",
    "success": "#1DC9A4",
    "danger": "#E3120B",
    "warning": "#D4A017",
    "background": "#FFFFFF",
    "secondary_bg": "#F2F2F2",
    "text": "#121212",
    "text_muted": "#6B6B6B",
    "border": "#D9D9D9",
    "card_bg": "#FFFFFF",
    "card_shadow": "rgba(0,0,0,0.06)",
}

DARK_COLORS = {
    "primary": "#475ED1",
    "accent": "#F6423C",
    "success": "#36E2BD",
    "danger": "#F6423C",
    "warning": "#E2B93B",
    "background": "#0E1117",
    "secondary_bg": "#1A1C2E",
    "text": "#E8E8E8",
    "text_muted": "#9B9B9B",
    "border": "#2A2A3A",
    "card_bg": "#1A1C2E",
    "card_shadow": "rgba(0,0,0,0.3)",
}

# Backwards-compatible alias (used by tests and default mode)
COLORS = LIGHT_COLORS


def get_colors(mode="light"):
    """Retorna la paleta de colores seg\u00fan el modo del tema."""
    return DARK_COLORS if mode == "dark" else LIGHT_COLORS


# ============================================================
# MAPEO BALOTAJE (UI label -> codigo del modelo)
# ============================================================
BALOTAJE_UI_TO_CODE = {
    "No vot\u00f3/Blanco": "otros",
    "Mart\u00ednez (FA)": "martinez",
    "Lacalle (Coalici\u00f3n)": "lacalle",
}

# ============================================================
# UMBRALES DE INTERPRETACION
# ============================================================
# (umbral_minimo, clave_color_semantica, texto_interpretacion)
PROB_THRESHOLDS = [
    (70, "success", "muy probable que apoyes"),
    (55, "success", "probable que apoyes"),
    (45, "warning", "dividido/a"),
    (30, "danger", "probable que te opongas"),
    (0, "danger", "muy probable que te opongas"),
]


def get_interpretation(prob, mode="light"):
    """Devuelve (color_hex, texto) seg\u00fan la probabilidad y el modo de tema."""
    colors = get_colors(mode)
    for threshold, color_key, text in PROB_THRESHOLDS:
        if prob >= threshold:
            return colors[color_key], text
    return colors["danger"], "muy probable que te opongas"

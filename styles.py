"""
Estilos CSS del widget IVE.
Dise\u00f1o editorial inspirado en The Economist + El Observador (rebrand 2024).
"""

from config import get_colors


def get_custom_css(mode="light"):
    """Genera CSS editorial adaptado al modo de tema (light/dark)."""
    c = get_colors(mode)

    return f"""
<style>
    /* ============================================================
       TIPOGRAFIA
       ============================================================ */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@400;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* ============================================================
       LAYOUT: OCULTAR CHROME DE STREAMLIT + ANCHO EDITORIAL
       ============================================================ */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    .main > div {{
        max-width: 720px;
        margin: 0 auto;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}

    /* ============================================================
       BARRA ROJA SUPERIOR (SIGNATURE ECONOMIST)
       ============================================================ */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: {c['accent']};
        z-index: 999;
    }}

    /* ============================================================
       TITULO PRINCIPAL
       ============================================================ */
    .main-title {{
        font-family: 'IBM Plex Serif', Georgia, serif;
        font-size: 2rem;
        font-weight: 700;
        color: {c['text']};
        margin-bottom: 0.25rem;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }}

    .subtitle {{
        font-size: 1rem;
        color: {c['text_muted']};
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }}

    /* ============================================================
       SECTION HEADERS CON LINEA DE ACENTO
       ============================================================ */
    .section-header {{
        font-family: 'IBM Plex Serif', Georgia, serif;
        font-size: 1.25rem;
        font-weight: 600;
        color: {c['text']};
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid {c['accent']};
    }}

    /* ============================================================
       BARRA DE PROBABILIDAD (GRADIENTE EDITORIAL)
       ============================================================ */
    .prob-bar-wrapper {{
        margin: 2rem 0 1.5rem 0;
    }}

    .prob-endpoints {{
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }}

    .prob-endpoint {{
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }}

    .prob-endpoint--contra {{
        color: {c['danger']};
    }}

    .prob-endpoint--favor {{
        color: {c['primary']};
    }}

    .prob-container {{
        background: linear-gradient(90deg, {c['danger']} 0%, {c['text_muted']} 50%, {c['primary']} 100%);
        border-radius: 25px;
        height: 44px;
        position: relative;
        box-shadow: 0 2px 8px {c['card_shadow']};
    }}

    .prob-indicator {{
        position: absolute;
        top: -8px;
        transform: translateX(-50%);
        width: 3px;
        height: 60px;
        background: {c['text']};
        border-radius: 2px;
    }}

    .prob-label {{
        position: absolute;
        top: -38px;
        transform: translateX(-50%);
        background: {c['text']};
        color: {c['background']};
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        font-family: 'IBM Plex Serif', Georgia, serif;
        white-space: nowrap;
    }}

    /* ============================================================
       TARJETA DE RESULTADO
       ============================================================ */
    .result-card {{
        background: {c['card_bg']};
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-top: 3px solid {c['accent']};
        box-shadow: 0 2px 12px {c['card_shadow']};
    }}

    .result-number {{
        font-family: 'IBM Plex Serif', Georgia, serif;
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1;
        letter-spacing: -0.02em;
    }}

    .result-text {{
        font-size: 0.95rem;
        color: {c['text_muted']};
        margin-top: 0.75rem;
        line-height: 1.5;
    }}

    .result-nacional {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 1rem;
        padding: 0.5rem 1rem;
        background: {c['secondary_bg']};
        border-radius: 6px;
        font-size: 0.85rem;
        color: {c['text_muted']};
    }}

    .result-nacional-value {{
        font-family: 'IBM Plex Serif', Georgia, serif;
        font-weight: 700;
        font-size: 1.1rem;
        color: {c['text']};
    }}

    .result-nacional-diff {{
        font-weight: 600;
        font-size: 0.85rem;
    }}

    /* ============================================================
       METRIC CARDS (COMPARACIONES)
       ============================================================ */
    .metric-card {{
        background: {c['secondary_bg']};
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }}

    .metric-label {{
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: {c['text_muted']};
        margin-bottom: 0.4rem;
    }}

    .metric-value {{
        font-family: 'IBM Plex Serif', Georgia, serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: {c['text']};
    }}

    .metric-delta {{
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }}

    /* ============================================================
       TABS ESTILO UNDERLINE
       ============================================================ */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        border-bottom: 1px solid {c['border']};
    }}

    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        padding: 0.5rem 1rem;
        font-size: 0.85rem;
        font-weight: 500;
        color: {c['text_muted']};
    }}

    .stTabs [aria-selected="true"] {{
        border-bottom: 2px solid {c['accent']} !important;
        color: {c['text']} !important;
        font-weight: 600;
    }}

    .stTabs [data-baseweb="tab-highlight"] {{
        background: {c['accent']} !important;
    }}

    .stTabs [data-baseweb="tab-border"] {{
        display: none;
    }}

    /* ============================================================
       SELECTORES
       ============================================================ */
    .stSelectbox > div > div {{
        border-radius: 6px;
    }}

    .stSelectbox label {{
        font-size: 0.85rem;
        font-weight: 600;
        color: {c['text']};
    }}

    /* ============================================================
       EXPANDER (METODOLOGIA)
       ============================================================ */
    .streamlit-expanderHeader {{
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.9rem;
        font-weight: 600;
        color: {c['text']};
    }}

    /* ============================================================
       DIVIDER
       ============================================================ */
    .editorial-divider {{
        border: none;
        border-top: 1px solid {c['border']};
        margin: 1.5rem 0;
    }}

    /* ============================================================
       FOOTER
       ============================================================ */
    .footer-text {{
        font-size: 0.75rem;
        color: {c['text_muted']};
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid {c['border']};
        line-height: 1.6;
    }}

    .footer-text strong {{
        color: {c['text']};
    }}
</style>
"""


# Backwards-compatible constant (default light mode)
CUSTOM_CSS = get_custom_css("light")

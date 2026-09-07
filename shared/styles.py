"""
Estilos CSS del widget IVE.
Diseño editorial inspirado en The Economist + El Observador (rebrand 2024).
"""

from shared.config import get_colors, OBSERVADOR_COLORS


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
        margin: 1.5rem 0 3.5rem 0;
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
        top: -6px;
        transform: translateX(-50%);
        width: 3px;
        height: 56px;
        background: {c['text']};
        border-radius: 2px;
    }}

    .prob-label {{
        position: absolute;
        top: 58px;
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

    /* Van pegados al número grande, no sueltos con estilos inline. */
    .result-intervalo {{
        font-size: 0.9rem;
        color: {c['text_muted']};
        margin-top: 0.3rem;
        margin-bottom: 0.9rem;
    }}

    .result-neutral {{
        margin-top: 0.75rem;
        font-size: 0.85rem;
        color: {c['text_muted']};
        line-height: 1.5;
    }}

    /* ============================================================
       BARRAS COMPARATIVAS AGRUPADAS (widget de seguridad)

       El widget IVE compara cuatro o cinco grupos por dimensión y le alcanza
       con tabs de metric cards. El de seguridad tiene dieciséis grupos en siete
       dimensiones de dos o tres cada una: en tabs quedaban casi vacías, y en
       una grilla plana quedaban dieciséis cajas grises apiladas sin jerarquía.
       Con barras el lector compara mirando, no leyendo dieciséis números.
       ============================================================ */
    .grupo-bloque {{
        margin-bottom: 1.4rem;
    }}

    .grupo-titulo {{
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {c['text_muted']};
        margin-bottom: 0.45rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid {c['border']};
    }}

    .grupo-fila {{
        display: grid;
        grid-template-columns: 12rem 1fr 2.8rem;
        align-items: center;
        gap: 0.85rem;
        padding: 0.32rem 0;
    }}

    .grupo-label {{
        font-size: 0.85rem;
        color: {c['text']};
        line-height: 1.3;
    }}

    .grupo-barra {{
        position: relative;
        height: 10px;
        border-radius: 5px;
        background: {c['secondary_bg']};
        overflow: visible;
    }}

    .grupo-barra-fill {{
        height: 100%;
        border-radius: 5px;
        background: {c['primary']};
    }}

    /* Marca del promedio nacional: la misma tasa observada, así que la
       comparación es contra algo del mismo tipo y no contra la estimación
       ajustada del perfil elegido. */
    .grupo-barra-ref {{
        position: absolute;
        top: -4px;
        bottom: -4px;
        width: 1px;
        background: {c['text']};
        opacity: 0.55;
    }}

    .grupo-valor {{
        font-family: 'IBM Plex Serif', Georgia, serif;
        font-weight: 700;
        font-size: 1rem;
        text-align: right;
        color: {c['text']};
        font-variant-numeric: tabular-nums;
    }}

    .grupo-nota-ref {{
        font-size: 0.75rem;
        color: {c['text_muted']};
        margin-top: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }}

    .grupo-nota-marca {{
        display: inline-block;
        width: 1px;
        height: 12px;
        background: {c['text']};
        opacity: 0.55;
    }}

    @media (max-width: 640px) {{
        .grupo-fila {{
            grid-template-columns: 1fr 2.8rem;
            grid-template-areas: "label valor" "barra barra";
            gap: 0.3rem 0.6rem;
            padding: 0.45rem 0;
        }}
        .grupo-label {{ grid-area: label; }}
        .grupo-valor {{ grid-area: valor; }}
        .grupo-barra {{ grid-area: barra; }}
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
    [data-testid="stExpander"] {{
        margin-top: 1.5rem;
    }}

    .streamlit-expanderHeader {{
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.9rem;
        font-weight: 600;
        color: {c['text']};
    }}

    /* ============================================================
       MOBILE: forzar apilado de columnas
       ============================================================ */
    @media (max-width: 640px) {{
        [data-testid="stHorizontalBlock"] {{
            flex-direction: column !important;
        }}
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
            width: 100% !important;
            flex: 1 1 100% !important;
        }}
        .main-title {{
            font-size: 1.6rem;
        }}
        .result-number {{
            font-size: 2.75rem;
        }}
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


def get_observador_css():
    """
    Hoja de estilos del Figma "Producto UY".

    SEPARADA de get_custom_css() a propósito. Esa la comparten el widget IVE
    —que está publicado y sirviendo— y el de seguridad; cambiarla habría
    re-diseñado de rebote una app en producción que nadie pidió tocar. Esta es
    opt-in: la usa el widget que la importe.

    Diferencias con la hoja editorial anterior, todas del Figma:
      · Una sola variante, CLARA. El Figma no trae modo oscuro, así que el
        widget deja de seguir el tema del sistema del lector.
      · El gradiente va de naranja a azul lavanda, no de rojo a azul.
      · El valor sobre el gradiente va en una pastilla negra.
      · El titular y el número grande van en verde profundo, no en negro/azul.
      · Los grupos se comparan con solapas por dimensión y una fila de números
        grandes, en vez de una lista larga de barras.

    Las tipografías son las que ya estaban (IBM Plex Serif/Sans): el Figma usa
    otras, pero sin acceso de inspección no se puede saber cuáles, y adivinar
    una familia de marca es peor que usar uno cercano y decirlo.
    """
    c = OBSERVADOR_COLORS

    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@400;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* El Figma es claro. Se fija el fondo en vez de heredar el tema del
       lector, porque con la paleta oscura de Streamlit el verde profundo del
       titular queda ilegible. */
    .stApp, .main, body {{
        background: {c['background']} !important;
        color: {c['text']} !important;
    }}

    #MainMenu, footer, header {{visibility: hidden;}}

    .main > div {{
        max-width: 720px;
        margin: 0 auto;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }}

    /* ---------- Titulares ---------- */
    .main-title {{
        font-family: 'IBM Plex Serif', Georgia, serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: {c['primary']} !important;
        margin: 0.5rem 0 0.6rem 0;
        line-height: 1.15;
        letter-spacing: -0.015em;
    }}

    .subtitle {{
        font-size: 0.95rem;
        color: {c['text_muted']} !important;
        margin-bottom: 1rem;
        line-height: 1.55;
    }}

    .section-header {{
        font-family: 'IBM Plex Serif', Georgia, serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: {c['text']} !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 0 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid {c['border']};
    }}

    /* ---------- Controles ---------- */
    /* El relleno de los desplegables. El COLOR del texto no se toca acá: lo
       gobierna el tema base de Streamlit, que este widget fija en claro desde
       `.streamlit/config.toml`. Intentar forzarlo con CSS contra los nodos
       internos de baseweb fue un callejón: los data-testid cambian entre
       versiones y quedaban radios negros y valores ilegibles. */
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {{
        background: {c['input_bg']} !important;
        border: none !important;
        border-radius: 8px !important;
    }}

    .stSelectbox label, .stRadio label {{
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: {c['text']} !important;
    }}

    /* El punto del radio elegido toma `primaryColor` del config compartido,
       que es el azul del widget IVE — el único que lo usa. Acá se pisa con el
       verde del Figma, sin tocar el config y sin afectar al otro widget. */
    /* El punto va por estructura y no por atributo porque Streamlit lo dibuja
       en un div anidado SIN testid, role ni aria-checked: no hay nada estable
       a lo que agarrarse. Lo que sí es estable es `:has(input:checked)` sobre
       la opción, que acota la regla a la elegida — sin eso, se pintarían de
       verde también los puntos vacíos. Verificado contra el DOM real, no
       supuesto: los selectores por `data-baseweb` no matcheaban nada. */
    [data-testid="stRadioOption"]:has(input:checked) div div div {{
        background-color: {c['primary']} !important;
        border-color: {c['primary']} !important;
    }}

    /* ---------- Barra de probabilidad ---------- */
    .prob-bar-wrapper {{ margin: 1.25rem 0 3.25rem 0; }}

    .prob-endpoints {{
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }}

    .prob-endpoint {{
        font-size: 0.85rem;
        font-weight: 500;
        color: {c['text']};
        text-transform: none;
        letter-spacing: 0;
    }}

    .prob-container {{
        background: linear-gradient(90deg, {c['accent']} 0%, #C9C6C0 50%, #A8B4E0 100%);
        border-radius: 6px;
        height: 30px;
        position: relative;
    }}

    .prob-indicator {{
        position: absolute;
        top: -6px;
        bottom: -6px;
        width: 2px;
        background: {c['text']};
        transform: translateX(-1px);
    }}

    /* La pastilla negra del Figma, debajo de la marca. */
    .prob-label {{
        position: absolute;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        margin-top: 6px;
        background: {c['text']};
        color: #FFFFFF;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 999px;
        white-space: nowrap;
    }}

    /* ---------- Tarjeta de resultado ---------- */
    .result-card {{
        background: {c['card_bg']};
        border: 1px solid {c['border']};
        border-radius: 12px;
        padding: 1.5rem 1.5rem 1.25rem 1.5rem;
        box-shadow: 0 2px 10px {c['card_shadow']};
        margin-bottom: 1rem;
    }}

    .result-number {{
        font-family: 'IBM Plex Serif', Georgia, serif;
        font-size: 3.25rem;
        font-weight: 700;
        line-height: 1;
        color: {c['primary']} !important;
        margin-bottom: 0.35rem;
    }}

    .result-intervalo {{
        font-size: 0.85rem;
        color: {c['text_muted']};
        margin-bottom: 0.75rem;
    }}

    .result-text {{
        font-size: 0.95rem;
        color: {c['text']};
        line-height: 1.6;
    }}

    .result-text strong {{ color: {c['primary']} !important; }}

    .result-nacional {{
        margin-top: 1rem;
        padding-top: 0.85rem;
        border-top: 1px solid {c['border']};
        font-size: 0.9rem;
        color: {c['text']};
    }}

    .result-nacional-value {{ font-weight: 700; }}

    /* El Figma pinta la diferencia contra el promedio en naranja. */
    .result-nacional-diff {{
        color: {c['accent']} !important;
        font-weight: 500;
    }}

    .result-neutral {{
        margin-top: 0.75rem;
        font-size: 0.85rem;
        color: {c['text_muted']};
    }}

    /* ---------- Comparación por grupos: solapas + números grandes ---------- */
    /* Las píldoras del Figma. Van contra `data-testid="stTab"`, que es lo que
       expone esta versión de Streamlit; los selectores `data-baseweb` no
       matcheaban y las solapas salían como rectángulos apretados con el texto
       desbordado. */
    [data-testid="stTabs"] [role="tablist"] {{
        gap: 0.4rem;
        border-bottom: none !important;
        margin-bottom: 1.1rem;
        flex-wrap: wrap;
    }}

    [data-testid="stTab"] {{
        background: {c['background']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 0.4rem 0.85rem !important;
        font-size: 0.85rem;
        color: {c['text']} !important;
        height: auto !important;
        white-space: nowrap;
    }}

    [data-testid="stTab"][aria-selected="true"] {{
        background: {c['primary']} !important;
        border-color: {c['primary']} !important;
        color: #FFFFFF !important;
    }}

    [data-testid="stTab"][aria-selected="true"] p {{ color: #FFFFFF !important; }}

    [data-testid="stTabs"] [data-baseweb="tab-highlight"],
    [data-testid="stTabs"] [data-baseweb="tab-border"] {{
        display: none !important;
    }}

    /* Grid y no flex: con flex, los grupos que pasan a una segunda fila se
       estiran para llenarla y quedan desalineados respecto de la primera. Con
       siete tramos ideológicos eso pasa siempre. */
    .grupo-cifras {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(128px, 1fr));
        gap: 1rem 0;
    }}

    .grupo-celda {{
        padding: 0.25rem 0.75rem;
        border-left: 1px solid {c['border']};
        text-align: left;
    }}

    /* El borde separador se saca en la primera columna de CADA fila, no sólo
       en la primera celda: si no, las filas de abajo arrancan con una línea
       suelta a la izquierda. */
    .grupo-celda:first-child {{ border-left: none; padding-left: 0; }}

    .grupo-celda-label {{
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: {c['text_muted']};
        margin-bottom: 0.4rem;
        line-height: 1.35;
        min-height: 2.7em;
        /* Sin esto, "Centroizquierda" se cortaba como "CENTROIZQUIER / DA". */
        overflow-wrap: normal;
        word-break: keep-all;
        hyphens: none;
    }}

    .grupo-celda-valor {{
        font-family: 'IBM Plex Serif', Georgia, serif;
        font-size: 1.7rem;
        font-weight: 700;
        color: {c['text']} !important;
        line-height: 1;
    }}

    .grupo-celda-delta {{
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.2rem;
    }}

    .grupo-celda-delta--sube {{ color: {c['azul']}; }}
    .grupo-celda-delta--baja {{ color: {c['accent']}; }}

    .grupo-nota-ref {{
        margin-top: 1rem;
        font-size: 0.8rem;
        color: {c['text_muted']};
    }}

    /* ---------- Varios ---------- */
    .editorial-divider {{
        border: none;
        border-top: 1px solid {c['border']};
        margin: 1.5rem 0;
    }}

    [data-testid="stExpander"] {{
        border: 1px solid {c['border']} !important;
        border-radius: 8px !important;
        background: {c['background']} !important;
    }}

    .footer-text {{
        font-size: 0.8rem;
        color: {c['text_muted']};
        line-height: 1.6;
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 1px solid {c['border']};
    }}

    /* ---------- Móvil ---------- */
    @media (max-width: 640px) {{
        [data-testid="stHorizontalBlock"] {{ flex-direction: column !important; }}
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
            width: 100% !important;
            flex: 1 1 100% !important;
        }}
        .main-title {{ font-size: 1.6rem; }}
        .result-number {{ font-size: 2.6rem; }}
        .grupo-cifras {{ grid-template-columns: repeat(2, 1fr); gap: 0.75rem 1rem; }}
        .grupo-celda {{ border-left: none; padding: 0 0 0.25rem 0; }}
        .grupo-celda-label {{ min-height: 0; }}
    }}
</style>
"""

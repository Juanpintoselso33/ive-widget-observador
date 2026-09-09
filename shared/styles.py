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
    Hoja de estilos del Figma "Producto UY", página **Widget IVE**.

    SEPARADA de get_custom_css() a propósito. Esa la comparten el widget IVE
    —que está publicado y sirviendo— y el de seguridad; cambiarla habría
    re-diseñado de rebote una app en producción que nadie pidió tocar. Esta es
    opt-in: la usa el widget que la importe.

    LOS VALORES SALEN DEL PANEL DE INSPECCIÓN, uno por uno, no de muestrear una
    captura. Están escritos en `docs/diseno/figma-producto-uy.md` junto a los
    dos frames exportados a 2x, que son la evidencia.

    TODOS LOS SELECTORES DE ACÁ SE VERIFICARON CONTRA EL DOM REAL de Streamlit
    1.63.0, con el widget corriendo. Hacía falta: la versión anterior tenía
    CINCO reglas que no matcheaban nada y que por eso nunca se habían notado.
    Quedan anotadas una por una donde corresponde, porque el patrón se repite y
    conviene no volver a caer:

      · `.main > div` no existe; el contenedor es `stMainBlockContainer`.
      · `.main` tampoco existe, así que sobraba en la regla de fondo.
      · `[data-baseweb="select"]` murió cuando el Selectbox pasó a React Aria;
        el control es `[data-testid="stSelectbox"] [role="group"]`.
      · `[data-baseweb="tab-highlight"]` y `"tab-border"` murieron por lo mismo;
        la línea de selección la dibuja `.react-aria-SelectionIndicator`.
      · `stHeaderActionElement` va en PLURAL, `stHeaderActionElements`.

    Y dos trampas de anidamiento, que no son selectores muertos sino reglas que
    aciertan en el nodo equivocado:

      · El texto del titular no vive en el `h1` sino en un `span` que Streamlit
        mete adentro con su propia clase. Hay que estilar los dos, en escritorio
        y también en la consulta de móvil.
      · Las etiquetas y las solapas envuelven su texto en un contenedor de
        markdown con `font-size` propio (0,875rem). Fijar el tamaño en el
        `label` o en el `stTab` no llega al texto: hay que ir al `p`.

    SOBRE LOS INTERLINEADOS DEL FIGMA. Se aplican donde el texto del diseño
    ocupa varias líneas —titular 30/42 y bajada 23/23— y no donde es de una sola
    línea, aunque el panel diga 35px o 68px. En Figma esos textos están con
    "vertical trim: cap height", que recorta el interlineado: la caja de la
    etiqueta "Religiosidad" mide 14px de alto, no 35. Copiar el número a CSS no
    reproduciría el diseño, lo rompería.
    """
    c = OBSERVADOR_COLORS

    return f"""
<style>
    /* Instrument Sans es variable y trae eje de ANCHO: la bajada del Figma usa
       la variante Condensed, que sale de la misma familia con `wdth: 75`. Por
       eso el rango 75..100 en la URL. */
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wdth,wght@75..100,400..700&family=Libre+Baskerville:wght@400;700&display=swap');

    /* Sin `[class*="css"]`: las clases de Streamlit son `st-emotion-cache-…` y
       ese selector sólo acertaría por casualidad si el hash llevara "css". */
    html, body {{
        font-family: 'Instrument Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 19px;
    }}

    /* FIJAR LA FAMILIA EN body NO ALCANZA. Streamlit pone su propia Source Sans
       en los contenedores de markdown y de los widgets, así que todo lo que
       escribe el widget la hereda de ahí y no de body: medido, la mitad de los
       nodos con texto seguía en Source Sans después de cambiar body; ahora son
       cero de 301. Va ANTES que las de Libre Baskerville, y eso importa:
       `.main-title span` gana por especificidad, pero `.section-header` EMPATA
       con `[data-testid="stMarkdownContainer"] *` y gana sólo por venir
       después. Mover este bloque más abajo le cambiaría la tipografía al título
       de sección. (Decía que las dos ganaban por especificidad; era falso y lo
       marcó Codex.) */
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
    [data-testid="stSelectbox"], [data-testid="stSelectbox"] *,
    [data-testid="stTab"], [data-testid="stTab"] * {{
        font-family: 'Instrument Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}

    /* El Figma es claro. Se fija el fondo en vez de heredar el tema del lector,
       porque con la paleta oscura de Streamlit el verde del titular queda
       ilegible. (Iba también contra `.main`, que en esta versión no existe.) */
    .stApp, body {{
        background: {c['background']} !important;
        color: {c['text']} !important;
    }}

    /* Sin `footer`: esta app no tiene ese elemento. El pie es un `p`. */
    #MainMenu, header {{visibility: hidden;}}

    /* El marco: 698px de ancho, filete superior de 2px y sombra. */
    [data-testid="stMainBlockContainer"] {{
        max-width: 698px !important;
        margin: 0 auto;
        border-top: 2px solid {c['primary']};
        box-shadow: 0 5px 6px {c['card_shadow']};
        padding: 1.5rem 0.8rem 2rem 0.8rem !important;
    }}

    /* ---------- Titulares ---------- */
    .main-title, .main-title span {{
        font-family: 'Libre Baskerville', Georgia, serif !important;
        font-size: 30px !important;
        font-weight: 700 !important;
        color: {c['primary']} !important;
        line-height: 42px !important;
        letter-spacing: 0 !important;
    }}

    .main-title {{ margin: 0.5rem 0 0.6rem 0; }}

    /* El botoncito de ancla que Streamlit cuelga del encabezado no tiene que
       heredar los 30px. El testid va en PLURAL. */
    .main-title [data-testid="stHeaderActionElements"] {{
        font-size: 14px !important;
    }}

    .subtitle {{
        font-family: 'Instrument Sans', sans-serif !important;
        font-size: 23px !important;
        font-stretch: 75% !important;
        color: {c['text']} !important;
        margin-bottom: 1rem;
        line-height: 23px !important;
    }}

    .section-header, .section-header span {{
        font-family: 'Libre Baskerville', Georgia, serif !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        color: {c['text']} !important;
        text-transform: uppercase;
        letter-spacing: 0 !important;
    }}

    .section-header {{
        margin: 0 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid {c['text']};
    }}

    /* ---------- Controles ---------- */
    /* El control es un `div[role=group]` dentro del stSelectbox. El selector de
       baseweb que había acá no matcheaba nada desde que Streamlit pasó el
       Selectbox a React Aria, así que el borde, el radio y el alto del diseño
       no llegaban nunca; el relleno coincidía de casualidad porque el tema
       compartido ya usaba el mismo gris. */
    /* El alto va con `height`, no sólo con `min-height`: el control trae
       `height: 2.5rem` y con el mínimo solo quedaba en 47,5px en vez de los 37
       del diseño. Lo marcó Codex. */
    [data-testid="stSelectbox"] [role="group"] {{
        background: {c['input_bg']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: 9px !important;
        height: 37px !important;
        min-height: 37px !important;
    }}

    /* El valor elegido va en gris, no en el negro de la etiqueta, y a 19px: el
       input conserva `font-size: 0.875rem` y quedaba en 16,6. */
    [data-testid="stSelectbox"] [role="group"] * {{
        color: {c['input_text']} !important;
        font-size: 19px !important;
    }}

    /* Y HAY QUE DEVOLVERLE LA SEÑAL DE FOCO. Streamlit la da cambiando el color
       del borde con `[data-focus-within]`, y el borde de arriba, con su
       `!important`, dejaba el mismo color enfocado y sin enfocar: al llegar con
       Tab al campo cerrado no pasaba nada visible. Lo marcó Codex. */
    [data-testid="stSelectbox"] [role="group"][data-focus-within] {{
        border-color: {c['primary']} !important;
    }}

    /* EL TAMAÑO VA EN EL `p`, no en el `label`: Streamlit mete el texto de la
       etiqueta en un contenedor de markdown con font-size propio. */
    [data-testid="stWidgetLabel"] p {{
        font-size: 19px !important;
        font-weight: 400 !important;
        color: {c['text']} !important;
    }}

    /* EL PUNTO DEL RADIO QUEDA AZUL, no verde, y es a propósito.
       Streamlit lo dibuja en un div anidado sin testid, sin role y sin
       aria-checked, y a la misma profundidad que la caja del texto de la
       etiqueta. Toda regla estructural que agarraba el punto agarraba también
       esa caja y pintaba un rectángulo verde detrás del texto — que es mucho
       peor que un punto del color equivocado. Se probaron cinco selectores
       contra el DOM real; ninguno separa los dos.
       Si alguna vez hace falta, la vía limpia es un componente propio, no CSS.
       Toma `primaryColor` de .streamlit/config.toml. */

    /* ---------- La banda gris ---------- */
    /* Del gradiente al pie, el Figma va sobre gris y no sobre blanco: un tercio
       del área. `app.py` envuelve esa parte en un `st.container(key=...)` y
       Streamlit le pone `st-key-<key>`, que es la vía soportada para
       engancharle CSS. Los márgenes negativos la sacan a sangre hasta los
       bordes del marco, que es como está en el diseño. */
    .st-key-banda_resultado {{
        background: {c['secondary_bg']};
        /* El ancho VA EXPLÍCITO. Con sólo los márgenes negativos, la banda se
           corría a la izquierda pero no se ensanchaba: medido, quedaba en 668px
           dentro de un marco de 698. Es un item de un contenedor flex en
           columna y el margen negativo derecho no le agrega ancho.
           Y va con `!important` porque la clase de emotion que Streamlit le
           pone al mismo nodo declara `width: 100%`, con la misma
           especificidad. */
        margin-left: -0.8rem !important;
        margin-right: -0.8rem !important;
        margin-bottom: -2rem !important;
        width: calc(100% + 1.6rem) !important;
        /* Y hay que soltar el `max-width: 100%` que Streamlit le pone al mismo
           nodo: con él puesto, el ancho pedido se recorta al del envoltorio y
           la banda se quedaba 30px corta. Medido: 667,6 contra 698. */
        max-width: none !important;
        padding: 1.25rem 0.8rem 2rem 0.8rem;
    }}

    /* ---------- Barra de probabilidad ---------- */
    .prob-bar-wrapper {{ margin: 0.5rem 0 1.25rem 0; }}

    .prob-endpoints {{
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }}

    .prob-endpoint {{
        font-size: 19px;
        font-weight: 400;
        color: {c['text']};
    }}

    /* Gradiente de DOS paradas, naranja a azul. La versión anterior metía un
       gris cálido en el medio que no existe en el diseño. */
    .prob-container {{
        background: linear-gradient(90deg, {c['accent']} 0%, {c['azul']} 100%);
        border-radius: 6px;
        height: 31px;
        position: relative;
        margin-top: 2.2rem;
    }}

    .prob-indicator {{
        position: absolute;
        top: -6px;
        bottom: -6px;
        width: 2px;
        background: {c['text']};
        transform: translateX(-1px);
    }}

    /* LA PASTILLA VA ARRIBA DE LA BARRA. Estaba abajo, y en los dos frames del
       Figma está arriba: medido sobre los PNG exportados, 65 filas de pastilla
       por encima de la barra y ninguna por debajo. */
    .prob-label {{
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        margin-bottom: 6px;
        background: {c['text']};
        color: #FFFFFF;
        font-size: 19px;
        font-weight: 400;
        padding: 3px 10px;
        border-radius: 7px;
        white-space: nowrap;
    }}

    /* ---------- Tarjeta de resultado ---------- */
    /* Sin borde: en el Figma se separa sólo por la sombra. */
    .result-card {{
        background: {c['card_bg']};
        border: none;
        border-radius: 10px;
        padding: 1.5rem 1.5rem 1.25rem 1.5rem;
        box-shadow: 0 5px 6px {c['card_shadow']};
        margin-bottom: 1rem;
    }}

    /* El número grande NO va en el verde del titular: va en el sólido. */
    .result-number {{
        font-family: 'Instrument Sans', sans-serif !important;
        font-size: 50px !important;
        font-weight: 600 !important;
        line-height: 1 !important;
        color: {c['solid']} !important;
        margin-bottom: 0.35rem;
    }}

    .result-text {{
        font-size: 19px;
        color: {c['text']};
        line-height: 1.6;
    }}

    .result-text strong {{ color: {c['solid']} !important; }}

    .result-nacional {{
        margin-top: 1rem;
        padding-top: 0.85rem;
        border-top: 1px solid {c['border']};
        font-size: 17px;
        color: {c['text']};
    }}

    .result-nacional-value {{ font-weight: 700; }}

    /* SIN COLOR ACÁ. El color lo pone la clase de signo que agrega
       `components.py` —`--sube` o `--baja`—, igual que en las diferencias por
       grupo. Esta regla lo forzaba en naranja con `!important`, así que una
       diferencia positiva salía del color de las negativas. */
    .result-nacional-diff {{
        font-weight: 400;
    }}

    .result-neutral {{
        margin-top: 0.75rem;
        font-size: 17px;
        color: {c['text_muted']};
    }}

    /* ---------- Comparación por grupos: solapas + números grandes ---------- */
    [data-testid="stTabs"] [role="tablist"] {{
        gap: 10px;
        border-bottom: none !important;
        margin-bottom: 1.1rem;
        flex-wrap: wrap;
    }}

    [data-testid="stTab"] {{
        background: {c['background']};
        border: 1px solid {c['border']};
        border-radius: 7px;
        padding: 9px !important;
        color: {c['text']} !important;
        height: auto !important;
        white-space: nowrap;
    }}

    /* El tamaño va en el `p`, por lo mismo que las etiquetas. Y el
       INTERLINEADO también, porque es lo que decide el alto de la pastilla: el
       Figma la tiene en 30px con 9px de padding arriba y abajo, o sea 12px de
       caja de texto. Con el interlineado que trae el contenedor de markdown la
       pastilla salía bastante más alta y el padding no alcanzaba para
       corregirlo. Quedan en 32px medidos, no 30: son los 30 del diseño más el
       borde de 1px de cada lado, que el Figma no tiene. */
    [data-testid="stTab"] p {{
        font-size: 16px !important;
        line-height: 12px !important;
    }}

    [data-testid="stTab"][aria-selected="true"] {{
        background: {c['solid']} !important;
        border-color: {c['solid']} !important;
    }}

    [data-testid="stTab"][aria-selected="true"] p {{ color: #FFFFFF !important; }}

    /* Dos cosas distintas y hay que apagar las dos: el indicador de selección,
       que dibuja React Aria en su propio nodo, y la línea de base del tablist,
       que NO es un `border-bottom` sino un `::after` absoluto. Apagar sólo el
       primero dejaba la raya, y los dos selectores `data-baseweb` que había
       antes no matcheaban ninguna de las dos. Lo marcó Codex. */
    [data-testid="stTabs"] .react-aria-SelectionIndicator,
    [data-testid="stTabs"] [role="tablist"]::after {{
        display: none !important;
    }}

    /* EL ANILLO DE FOCO SE RECOLOREA, NO SE SACA. La versión anterior ponía
       `box-shadow: none`, que es lo que Streamlit usa para dibujarlo: al
       recorrer las solapas con el teclado desaparecía la señal de foco. */
    /* El anillo nativo es un `box-shadow` con el color primario del tema, así
       que hay que apagarlo Y poner el verde; si no, quedan los dos. Y el
       `outline-offset` va NEGATIVO porque el tablist tiene `overflow-x: auto` y
       recortaba el anillo que sobresalía. Lo marcó Codex. */
    [data-testid="stTab"]:focus-visible,
    [data-testid="stTab"][data-focus-visible] {{
        box-shadow: none !important;
        outline: 2px solid {c['primary']} !important;
        outline-offset: -2px !important;
    }}

    /* Grid y no flex: con flex, los grupos que pasan a una segunda fila se
       estiran para llenarla y quedan desalineados respecto de la primera. Con
       siete tramos ideológicos eso pasa siempre.
       CUATRO COLUMNAS FIJAS, no `auto-fit`, y por dos motivos. Es lo que hace
       el Figma —cuatro en escritorio, dos en móvil— y además es lo único que
       permite sacarle el borde izquierdo a la primera celda DE CADA FILA: con
       `auto-fit` el CSS no sabe cuántas columnas entraron, así que `nth-child`
       no puede apuntarlas y las filas de abajo arrancaban con una raya suelta.
       Con cuatro columnas de 698px de ancho entra "CENTROIZQUIERDA" a 13px sin
       partirse, que es la etiqueta más larga que puede tocar. */
    .grupo-cifras {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem 0;
    }}

    .grupo-celda {{
        padding: 0.25rem 0.75rem;
        border-left: 1px solid {c['border']};
        text-align: left;
    }}

    /* La primera celda DE CADA FILA, ahora sí: con la grilla en cuatro columnas
       fijas, son la 1, la 5, la 9... */
    .grupo-celda:nth-child(4n + 1) {{ border-left: none; padding-left: 0; }}

    /* Del panel: 13px, peso 400. Con 14px y 600 las etiquetas de siete tramos
       ideológicos se pisaban entre columnas. */
    .grupo-celda-label {{
        font-size: 13px;
        font-weight: 400;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: {c['text']};
        margin-bottom: 0.4rem;
        line-height: 1.35;
        min-height: 2.7em;
        /* Sin esto, "Centroizquierda" se cortaba como "CENTROIZQUIER / DA". */
        overflow-wrap: normal;
        word-break: keep-all;
        hyphens: none;
    }}

    .grupo-celda-valor {{
        font-family: 'Instrument Sans', sans-serif !important;
        font-size: 25px !important;
        font-weight: 500 !important;
        color: {c['text']} !important;
        line-height: 1 !important;
    }}

    .grupo-celda-delta {{
        font-size: 17px;
        font-weight: 700;
        margin-top: 0.2rem;
    }}

    .grupo-celda-delta--sube {{ color: {c['azul']}; }}
    .grupo-celda-delta--baja {{ color: {c['accent']}; }}

    .grupo-nota-ref {{
        margin-top: 1rem;
        font-size: 16px;
        color: {c['text_muted']};
    }}

    /* ---------- Varios ---------- */
    .editorial-divider {{
        border: none;
        border-top: 1px solid {c['border']};
        margin: 1.5rem 0;
    }}

    /* El borde nativo vive en el `details`, no en el div con el testid: la
       regla anterior agregaba un SEGUNDO borde alrededor en vez de cambiar el
       que ya estaba. Lo marcó Codex. */
    [data-testid="stExpander"] details {{
        border: 1px solid {c['border']} !important;
        border-radius: 8px !important;
        background: transparent !important;
    }}

    .footer-text {{
        font-size: 16px;
        color: {c['text_muted']};
        line-height: 1.6;
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 1px solid {c['border']};
    }}

    /* ---------- Móvil ---------- */
    /* EL MÓVIL DEL FIGMA MANTIENE DOS COLUMNAS DE CAMPOS. Sacar la regla propia
       de apilado no alcanzaba: por debajo de 640px Streamlit le pone a cada
       columna un `min-width` del 100% y las apila igual. Hay que anularlo.
       El titular se achica en las DOS reglas —`.main-title` y su `span`—,
       porque el span trae tamaño propio con `!important` y no hereda. */
    @media (max-width: 640px) {{
        [data-testid="stColumn"] {{
            min-width: 0 !important;
            flex: 1 1 calc(50% - 0.5rem) !important;
        }}
        html, body, [class*="css"] {{ font-size: 16px; }}
        [data-testid="stWidgetLabel"] p {{ font-size: 16px !important; }}
        .main-title, .main-title span {{
            font-size: 24px !important;
            line-height: 32px !important;
        }}
        .subtitle {{ font-size: 19px !important; line-height: 20px !important; }}
        .result-number {{ font-size: 40px !important; }}
        /* En el frame de 331px la barra mide 18px, no 31. Medido sobre el PNG. */
        .prob-container {{ height: 18px; }}
        .prob-endpoint, .prob-label {{ font-size: 16px; }}
        /* Dos columnas en móvil, como el frame de 331px. El borde separador
           se saca entero: a ese ancho no hay lugar para el padding que pide. */
        .grupo-cifras {{ grid-template-columns: repeat(2, 1fr); gap: 0.75rem 1rem; }}
        .grupo-celda, .grupo-celda:nth-child(4n + 1) {{
            border-left: none;
            padding: 0 0 0.25rem 0;
        }}
        .grupo-celda-label {{ min-height: 0; }}
    }}
</style>
"""

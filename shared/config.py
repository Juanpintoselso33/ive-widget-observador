"""
Configuración editorial compartida — paleta y umbrales de El Observador.
Usada por todos los widgets de la plataforma.
"""

# ============================================================
# PALETA DE COLORES (Economist + El Observador editorial)
# ============================================================
LIGHT_COLORS = {
    "primary": "#2E45B8",
    "accent": "#E3120B",
    # Verde, ámbar y rojo OSCURECIDOS por contraste (7/9/2026).
    #
    # Los valores anteriores —#1DC9A4 y #D4A017— daban 2,11:1 y 2,38:1 sobre
    # blanco, y 1,89:1 y 2,12:1 sobre las tarjetas. WCAG AA pide 4,5 para texto
    # normal y 3 para texto grande: no llegaban ni al mínimo del número grande
    # del resultado, que es justo donde se usan (get_interpretation, más el
    # delta chico del widget IVE).
    #
    # Se corrigen ahora porque el tema pasó a estar FIJO EN CLARO para las dos
    # apps —ver .streamlit/config.toml—, así que quien antes recibía la paleta
    # oscura, que sí tenía contraste suficiente, pasaría a ésta. Lo encontró
    # Codex revisando ese cambio.
    #
    # Nuevos ratios, sobre blanco y sobre #F2F2F2: success 5,14 y 4,60;
    # warning 5,54 y 4,95; danger 6,09 y 5,44. Los tres pasan AA en los dos
    # fondos. `accent` no se toca: es la barra roja superior, no texto.
    "success": "#0E7C63",
    "danger": "#C50F09",
    "warning": "#8A6100",
    "background": "#FFFFFF",
    "secondary_bg": "#F2F2F2",
    "text": "#121212",
    "text_muted": "#6B6B6B",
    "border": "#D9D9D9",
    "card_bg": "#FFFFFF",
    "card_shadow": "rgba(0,0,0,0.06)",
}

DARK_COLORS = {
    # Aclarado desde #475ED1 por accesibilidad: contra el fondo oscuro daba un
    # contraste de 3,4 y contra las tarjetas 3,0, cuando WCAG AA pide 4,5 para
    # texto normal. Este tono llega a 6,2 y 5,6 respectivamente, y es el mismo
    # azul. Afecta a los dos widgets, sólo en modo oscuro: se usa para el
    # número grande del resultado y para el extremo "A FAVOR" del gradiente.
    "primary": "#7B8FE8",
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

COLORS = LIGHT_COLORS


def get_colors(mode="light"):
    return DARK_COLORS if mode == "dark" else LIGHT_COLORS


# ============================================================
# UMBRALES DE INTERPRETACION
# (umbral_minimo, clave_color_semantica, texto_interpretacion)
# ============================================================
PROB_THRESHOLDS = [
    (70, "success", "muy probable que apoyes"),
    (55, "success", "probable que apoyes"),
    (45, "warning", "dividido/a"),
    (30, "danger", "probable que te opongas"),
    (0,  "danger", "muy probable que te opongas"),
]


def get_interpretation(prob, mode="light"):
    colors = get_colors(mode)
    for threshold, color_key, text in PROB_THRESHOLDS:
        if prob >= threshold:
            return colors[color_key], text
    return colors["danger"], "muy probable que te opongas"


# ============================================================
# PALETA "PRODUCTO UY" — el Figma de la diseñadora
# ============================================================
# Sale del archivo de Figma "Producto UY" (node 1-2) que pasó Tomer el 7/9/2026,
# página "Widget IVE". Los valores están LEÍDOS DEL PANEL DE INSPECCIÓN, uno por
# uno, desde el 8/9/2026. Este encabezado decía lo contrario —que estaban
# muestreados de una imagen y pendientes de inspección— y se contradecía con el
# bloque de abajo; lo marcó Codex.
#
# Es una paleta CLARA y sin variante oscura, porque el Figma no la tiene. El
# widget que la use queda en claro pase lo que pase con el tema del lector.
OBSERVADOR_COLORS = {
    # LEÍDOS DEL PANEL DE INSPECCIÓN DE FIGMA el 8/9/2026, uno por uno, no
    # muestreados de una captura. Ver docs/diseno/figma-producto-uy.md, que trae
    # además los dos frames exportados a 2x como evidencia.
    #
    # VAN TAL CUAL, sin retoques de legibilidad. Antes había variantes
    # oscurecidas del naranja y del azul para que la diferencia contra el
    # promedio pasara WCAG AA; Juan pidió el diseño exacto y se sacaron. Queda
    # dicho lo que eso cuesta, porque es medible y no es chico:
    #
    #   sobre blanco    sobre #EDEDED   (AA para texto normal pide 4,5:1)
    #   #F57F00  2,65:1     2,27:1
    #   #93B6EE  2,07:1     1,76:1
    #   #999998  2,85:1     2,44:1
    #
    # Los tres se usan como TEXTO —la diferencia contra el promedio, las
    # diferencias por grupo y las notas al pie—, así que son tres textos que no
    # cumplen AA. Si alguna vez se quiere volver atrás, las variantes que
    # conservaban el tono y pasaban SOBRE BLANCO eran #B55E00 (4,60), #2F73DE
    # (4,54) y #767675 (4,55). Ojo, porque decir "pasaban" a secas era engañoso
    # y lo marcó Codex: sobre la banda gris esas mismas dan 3,93 / 3,88 / 3,88,
    # o sea que tampoco alcanzan ahí. Para pasar AA sobre el gris habría que
    # oscurecerlas más.

    # SON DOS VERDES, no uno. Muestrear una captura los promediaba y daba un
    # tercer verde que no existe en el diseño.
    "primary": "#006B36",   # titular y filete superior de 2px
    "solid": "#0D443B",     # botón, pastilla activa y número grande

    # El naranja y el azul son los MISMOS del gradiente y de las diferencias.
    "accent": "#F57F00",
    "azul": "#93B6EE",

    "background": "#FFFFFF",
    # Banda de la zona de resultado y comparación: un tercio del área del diseño.
    "secondary_bg": "#EDEDED",
    "text": "#1B1B19",
    "text_muted": "#999998",
    "border": "#D0CFCF",
    "card_bg": "#FFFFFF",
    "card_shadow": "rgba(0,0,0,0.15)",
    "input_bg": "#F2F2F2",
    "input_text": "#515151",
}

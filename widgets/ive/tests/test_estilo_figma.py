"""
El HTML que emite el widget IVE tiene que ser el que estila la hoja del Figma.

Por qué existe este archivo: el widget se pasó a `get_observador_css()` el
19/9/2026 y esa hoja NO estila las clases que emitía antes —`metric-card`,
`metric-label`, `metric-value`, `metric-delta`—. Un widget que siga emitiendo
las viejas no rompe, no tira excepción y pasa cualquier test de modelo: se ve
mal, y sólo se nota abriéndolo. Eso es exactamente lo que había pasado, y por
eso acá se asertan las dos direcciones: que están las clases nuevas y que NO
están las viejas.

Las clases esperadas están escritas a mano, literales. No salen de leer la
función que se verifica ni de la hoja: si el esperado lo generara el código bajo
prueba, el test pasaría igual con el widget roto.
"""

import ast
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from shared.styles import get_observador_css  # noqa: E402
from widgets.ive import components  # noqa: E402


# Las que tiene que emitir, escritas a mano.
CLASES_FIGMA = [
    "grupo-cifras", "grupo-celda", "grupo-celda-label", "grupo-celda-valor",
    "grupo-celda-delta", "result-card", "result-number", "result-text",
    "result-nacional", "prob-container", "prob-label", "section-header",
]

# Las de la hoja vieja, que ninguna regla de la nueva toca.
CLASES_MUERTAS = ["metric-card", "metric-label", "metric-value", "metric-delta"]

# Modificadores que se emiten a propósito SIN regla propia. En el Figma los dos
# extremos del gradiente son cuerpo negro —el color lo lleva la barra, no el
# texto—, así que `.prob-endpoint--contra` y `--favor` no tienen que pintar
# nada; quedan como gancho por si alguna vez hay que distinguirlos, y el widget
# de seguridad emite exactamente los mismos.
#
# La lista va nombrada una por una y no como patrón "cualquier cosa con --":
# `grupo-celda-delta--sube` SÍ tiene regla y tiene que seguir teniéndola, y una
# exención genérica se la llevaría puesta junto con cualquier modificador mal
# escrito.
SIN_REGLA_A_PROPOSITO = {"prob-endpoint--contra", "prob-endpoint--favor"}


class _Solapa:
    """Sustituto de lo que devuelve `st.tabs`: sólo tiene que ser un `with`."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def render(monkeypatch):
    """
    Captura lo que el widget le manda a Streamlit y devuelve el HTML junto.

    Devuelve también los títulos de solapa pedidos, que es lo único que no viaja
    por `st.markdown`.
    """
    trozos = []
    solapas = []

    def fake_markdown(html, **kwargs):
        trozos.append(str(html))

    def fake_tabs(titulos):
        solapas.extend(titulos)
        return [_Solapa() for _ in titulos]

    monkeypatch.setattr(components.st, "markdown", fake_markdown)
    monkeypatch.setattr(components.st, "tabs", fake_tabs)

    class Resultado:
        @property
        def html(self):
            return "\n".join(trozos)

        @property
        def solapas(self):
            return list(solapas)

    return Resultado()


def _clases_usadas(html):
    """Las clases que aparecen en atributos `class="..."` del HTML."""
    encontradas = set()
    for valor in re.findall(r'class="([^"]+)"', html):
        encontradas.update(valor.split())
    return encontradas


# ----------------------------------------------------------------------
# Las clases del Figma, y las viejas ausentes
# ----------------------------------------------------------------------

def test_emite_las_clases_del_figma(render, synthetic_model):
    components.render_comparisons(synthetic_model, synthetic_model["prob_nacional"])
    components.render_result_card(81.0, 76.5, prob_neutral=19.0)
    components.render_probability_bar(81.0)

    usadas = _clases_usadas(render.html)
    faltan = [c for c in CLASES_FIGMA if c not in usadas]
    assert not faltan, f"el widget no emite estas clases del Figma: {faltan}"


def test_no_quedan_clases_de_la_hoja_vieja(render, synthetic_model):
    components.render_comparisons(synthetic_model, synthetic_model["prob_nacional"])
    components.render_result_card(81.0, 76.5, prob_neutral=19.0)

    usadas = _clases_usadas(render.html)
    # SIN ESTO EL TEST ES DECORATIVO: si la captura fallara y `usadas` quedara
    # vacío, "ninguna clase vieja está presente" sería verdad y el test pasaría
    # con el widget sin renderizar nada.
    assert "result-card" in usadas

    vivas = [c for c in CLASES_MUERTAS if c in usadas]
    assert not vivas, (
        f"siguen emitiéndose clases que la hoja del Figma no estila: {vivas}"
    )


def test_toda_clase_propia_que_emite_existe_en_la_hoja(render, synthetic_model):
    """
    Control contra el agujero de la clase huérfana: emitir `grupo-celda-valo`
    en vez de `grupo-celda-valor` no rompe nada, no lo agarra ningún otro test
    y se ve como texto pelado.

    Sólo mira las clases propias del proyecto: las de Streamlit y las de emotion
    las pone el framework y no tienen por qué estar en esta hoja.
    """
    components.render_comparisons(synthetic_model, synthetic_model["prob_nacional"])
    components.render_result_card(81.0, 76.5, prob_neutral=19.0)
    components.render_probability_bar(81.0)
    components.render_header()
    components.render_footer(synthetic_model)

    hoja = get_observador_css()
    propias = [c for c in _clases_usadas(render.html)
               if not c.startswith(("st-", "css-", "row-widget"))
               and c not in SIN_REGLA_A_PROPOSITO]
    # Que el filtro no se haya comido todo: si `propias` quedara vacío, el
    # bucle de abajo no probaría nada y el test pasaría igual.
    assert len(propias) >= len(CLASES_FIGMA)

    # CON LÍMITE, no por subcadena: `.grupo-celda-delta--sub` es prefijo de
    # `.grupo-celda-delta--sube`, así que un typo que corta la clase por la
    # mitad se daba por estilado. Detrás del nombre tiene que venir algo que
    # CIERRE el selector, no otro carácter de clase. Lo marcó Codex.
    def estilada(clase):
        return re.search(rf"\.{re.escape(clase)}(?![-\w])", hoja) is not None

    huerfanas = [c for c in propias if not estilada(c)]
    assert not huerfanas, (
        f"el widget emite clases que la hoja no estila: {sorted(huerfanas)}"
    )


# ----------------------------------------------------------------------
# Los entry points
# ----------------------------------------------------------------------

def _arbol(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def _llamadas(arbol):
    """Los nombres de función efectivamente LLAMADOS en el archivo."""
    nombres = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Call):
            f = nodo.func
            if isinstance(f, ast.Name):
                nombres.add(f.id)
            elif isinstance(f, ast.Attribute):
                nombres.add(f.attr)
    return nombres


def test_el_entry_carga_la_hoja_del_figma_y_arma_la_banda():
    """
    El fallo que este archivo viene a prevenir NO estaba en los componentes:
    la hoja del Figma existía hacía días y el widget IVE simplemente no la
    cargaba. Todo el resto de la suite pasaba igual.

    Y la banda gris tampoco sale de la hoja: `get_observador_css()` engancha
    `.st-key-banda_resultado`, que sólo existe si el entry envuelve esa parte en
    `st.container(key="banda_resultado")`. Sin el contenedor la hoja entra igual
    y la zona del resultado queda blanca, que es un defecto mudo.

    Va por AST y no por substring: buscar el texto `get_observador_css` daba por
    buena una importación sin uso —o una mención en un comentario— con la hoja
    vieja cargándose por otra vía. Lo marcó Codex.
    """
    arbol = _arbol(Path(__file__).parent.parent / "app.py")
    llamadas = _llamadas(arbol)

    assert "get_observador_css" in llamadas, "la hoja del Figma no se LLAMA"
    assert "get_custom_css" not in llamadas

    # El contenedor con la clave exacta, como llamada con su keyword.
    claves = [
        kw.value.value
        for nodo in ast.walk(arbol) if isinstance(nodo, ast.Call)
        for kw in nodo.keywords
        if isinstance(nodo.func, ast.Attribute) and nodo.func.attr == "container"
        and kw.arg == "key" and isinstance(kw.value, ast.Constant)
    ]
    assert "banda_resultado" in claves


def test_el_entry_del_deploy_no_duplica_el_widget():
    """
    `app.py` de la raíz era una copia casi literal del entry del IVE, y esa
    copia es la razón de que el estilo viviera en dos lados y sólo uno se
    actualizara. Tiene que delegar, no repetir.

    Por AST, otra vez: la versión anterior buscaba las palabras "widgets", "ive"
    y "app.py" en el archivo, y las tres estaban en el docstring — así que el
    test pasaba igual con el cuerpo entero borrado. Lo marcó Codex.
    """
    arbol = _arbol(Path(__file__).parent.parent.parent.parent / "app.py")

    # Importa el entry real y lo ejecuta.
    importa_el_entry = any(
        isinstance(n, ast.ImportFrom) and n.module == "widgets.ive.app"
        and any(a.name == "main" for a in n.names)
        for n in ast.walk(arbol)
    )
    assert importa_el_entry, "la raíz no importa el entry del IVE"
    assert "main" in _llamadas(arbol), "lo importa pero no lo llama"

    # Y no renderiza por su cuenta: si vuelve a hacerlo, volvió la duplicación.
    llamadas = _llamadas(arbol)
    assert not llamadas & {"render_result_card", "render_comparisons",
                           "render_header", "get_custom_css"}


# ----------------------------------------------------------------------
# La diferencia se mide contra el promedio nacional
# ----------------------------------------------------------------------

def test_el_delta_por_grupo_es_contra_el_promedio_nacional(render, synthetic_model):
    """
    `religiosidad_nada` vale 92,8 en el sintético. Contra un nacional de 80 la
    diferencia que se publica tiene que ser +13pp — redondeando 92,8 a 93 ANTES
    de restar, que es como se ve en pantalla.
    """
    components.render_comparisons(synthetic_model, 80.0)
    assert "+13pp" in render.html


def test_el_delta_sigue_a_la_base_nacional_que_se_le_pasa(monkeypatch, synthetic_model):
    """
    Control de que la base se USA y no está cableada: con el test anterior solo,
    una implementación con `nacional_r = 80` adentro pasaba igual, porque 80 es
    el único valor que se probaba. Acá se renderiza dos veces con bases
    distintas y se exige que cada delta se mueva EXACTAMENTE lo que se movió la
    base. Lo marcó Codex.
    """
    def render_con(nacional):
        trozos = []
        monkeypatch.setattr(components.st, "markdown",
                            lambda html, **k: trozos.append(str(html)))
        monkeypatch.setattr(components.st, "tabs",
                            lambda titulos: [_Solapa() for _ in titulos])
        components.render_comparisons(synthetic_model, nacional)
        return "\n".join(trozos)

    def deltas(html):
        # El signo del widget es el menos tipográfico (−), no el guion.
        return [int(v.replace("−", "-").replace("+", ""))
                for v in re.findall(r'grupo-celda-delta--\w+">([+−]\d+)pp', html)]

    con_80 = deltas(render_con(80.0))
    con_70 = deltas(render_con(70.0))

    assert con_80, "no se emitió ningún delta: el render falló"
    assert len(con_80) == len(con_70)
    # Bajar la base diez puntos tiene que subir cada delta diez puntos.
    assert all(b - a == 10 for a, b in zip(con_80, con_70)), (
        f"los deltas no siguen a la base: {con_80} contra {con_70}"
    )


def test_el_delta_no_depende_del_perfil_del_lector(render, synthetic_model):
    """
    Control negativo del cambio de significado: antes el delta se calculaba
    contra la predicción del perfil, así que movía al mover los selectores. Hoy
    no puede: `render_comparisons` ya ni recibe la probabilidad del perfil.
    """
    import inspect

    firma = inspect.signature(components.render_comparisons)
    assert list(firma.parameters) == ["model", "prob_nacional"]


def test_toda_clave_del_modelo_esta_publicada_o_declarada_afuera():
    """
    Contrato contra el artefacto REAL, no contra el sintético.

    `render_comparisons` omite en silencio cualquier clave que no reconozca, así
    que una dimensión nueva entrenada en el modelo quedaría invisible sin que
    nada fallara — y una dejada afuera a propósito (hogar) no se distingue de un
    olvido. Este test obliga a decidir: se publica o se declara.
    """
    modelo = json.loads(
        (Path(__file__).parent.parent / "model_coefficients.json").read_text(encoding="utf-8")
    )
    del_modelo = set(modelo["stats_by_group"])
    publicadas = {k for _, claves in components.GRUPOS_ORDEN for k in claves}

    assert del_modelo, "el artefacto no trae stats_by_group"
    sin_decidir = del_modelo - publicadas - components.GRUPOS_NO_PUBLICADOS
    assert not sin_decidir, (
        f"claves del modelo que no se publican ni están declaradas afuera: "
        f"{sorted(sin_decidir)}"
    )

    # Y al revés: una etiqueta que apunte a una clave que el modelo ya no trae
    # deja una celda muda en la grilla.
    fantasma = publicadas - del_modelo
    assert not fantasma, f"la grilla pide claves que el modelo no trae: {sorted(fantasma)}"


def test_una_dimension_sin_datos_no_genera_solapa(render, synthetic_model):
    """
    El sintético trae religiosidad y balotaje, pero no educación ni edad. Una
    solapa vacía es peor que una solapa de menos: el lector la abre y no hay
    nada.
    """
    components.render_comparisons(synthetic_model, synthetic_model["prob_nacional"])

    assert render.solapas == ["Por religiosidad", "Por balotaje 2024"]


# ----------------------------------------------------------------------
# El color sigue al signo, y no lo pone el componente
# ----------------------------------------------------------------------

@pytest.mark.parametrize("prob, nacional, clase_esperada, clase_prohibida", [
    (81.0, 76.5, "grupo-celda-delta--sube", "grupo-celda-delta--baja"),
    (60.0, 76.5, "grupo-celda-delta--baja", "grupo-celda-delta--sube"),
])
def test_la_brecha_contra_el_nacional_lleva_la_clase_del_signo(
    render, prob, nacional, clase_esperada, clase_prohibida
):
    components.render_result_card(prob, nacional)

    assert clase_esperada in render.html
    assert clase_prohibida not in render.html


def test_un_perfil_igual_al_promedio_no_lleva_clase_de_signo(render):
    components.render_result_card(76.5, 76.5)

    assert "igual al promedio" in render.html
    assert "grupo-celda-delta--sube" not in render.html
    assert "grupo-celda-delta--baja" not in render.html


def test_la_brecha_dice_de_que_lado_cae_el_perfil(render):
    """
    La resta es perfil menos nacional, pero la frase cuelga de la línea del
    promedio nacional: decir "↑4pp" a secas ahí se leía como si el que estuviera
    por encima fuera el promedio. Tiene que decir dónde cae el perfil.
    """
    components.render_result_card(81.0, 76.5)
    assert "↑5pp por encima" in render.html

    # 60 contra 76,5 da 16pp y no 17: los dos valores se redondean ANTES de
    # restar, y `round(76.5)` es 76 —Python redondea al par, no hacia arriba—,
    # que es también lo que se muestra como promedio nacional. La cuenta cierra
    # con lo que se ve, que es de lo que se trata.
    components.render_result_card(60.0, 76.5)
    assert "↓16pp por debajo" in render.html


def test_el_componente_no_elige_colores(render, synthetic_model):
    """
    Los colores los pone la hoja. Un `style="color: …"` en línea es la paleta
    semántica vieja —verde, ámbar, rojo— que el Figma no tiene, y además pisa a
    la hoja.
    """
    components.render_comparisons(synthetic_model, synthetic_model["prob_nacional"])
    components.render_result_card(81.0, 76.5, prob_neutral=19.0)
    components.render_probability_bar(81.0)

    estilos = re.findall(r'style="([^"]*)"', render.html)
    # El control de que esto no pasa por vacío: el indicador de la barra SÍ
    # lleva un `style` —la posición, `left: …%`—, así que si no aparece ningún
    # atributo de estilo es que no se renderizó nada y el test no probó nada.
    assert any("left:" in e for e in estilos), (
        "no se emitió ni el `style` de posición del indicador: la captura falló"
    )

    con_color = [e for e in estilos if "color" in e.lower()]
    assert not con_color, f"colores elegidos en línea: {con_color}"

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


# Las que tiene que emitir, escritas a mano. Sin `result-nacional`: el promedio
# nacional salió de la tarjeta por pedido editorial del 19/9/2026 y ahora vive
# sólo en el bloque de comparación. La regla sigue en la hoja porque la usa el
# widget de seguridad.
CLASES_FIGMA = [
    "grupo-cifras", "grupo-celda", "grupo-celda-label", "grupo-celda-valor",
    "grupo-celda-delta", "result-card", "result-number", "result-text",
    "prob-container", "prob-label", "section-header",
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
    components.render_result_card(81.0)
    components.render_probability_bar(81.0)

    usadas = _clases_usadas(render.html)
    faltan = [c for c in CLASES_FIGMA if c not in usadas]
    assert not faltan, f"el widget no emite estas clases del Figma: {faltan}"


def test_no_quedan_clases_de_la_hoja_vieja(render, synthetic_model):
    components.render_comparisons(synthetic_model, synthetic_model["prob_nacional"])
    components.render_result_card(81.0)

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
    components.render_result_card(81.0)
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


@pytest.mark.parametrize("valor, esperado", [
    ("1", True), ("true", True), ("si", True), ("sí", True), ("SÍ", True),
    ("0", False), ("false", False), ("", False), (None, False),
    ("cualquiera", False),
])
def test_el_modo_caja_se_pide_por_query_param(monkeypatch, valor, esperado):
    """
    `?resumen=1` es lo que elige la versión de caja. Se prueban también los
    valores que NO la activan: sin el control negativo, una implementación que
    devolviera siempre True pasaría igual.
    """
    from widgets.ive import app as entry

    params = {} if valor is None else {entry.PARAM_RESUMEN: valor}
    monkeypatch.setattr(entry.st, "query_params", params)

    assert entry.modo_resumen() is esperado


def test_el_modo_caja_tolera_el_parametro_repetido(monkeypatch):
    """`?resumen=1&resumen=0` llega como lista; no puede explotar."""
    from widgets.ive import app as entry

    monkeypatch.setattr(entry.st, "query_params", {entry.PARAM_RESUMEN: ["1", "0"]})
    assert entry.modo_resumen() is True


def _correr_entry(monkeypatch, synthetic_model, query_params):
    """
    Ejecuta `main()` con Streamlit doblado y devuelve qué secciones se dibujaron.

    EJECUTA DE VERDAD en vez de leer el AST. La primera versión de este test
    miraba si las llamadas estaban dentro de algún `ast.If`, y eso no
    discriminaba: cambiar `if not resumen:` por `if True:` seguía siendo un
    `If`, el widget quedaba mostrando todo siempre y el test pasaba igual.
    """
    from widgets.ive import app as entry

    dibujadas = []

    class _Ctx:
        def __enter__(self): return self
        def __exit__(self, *e): return False

    monkeypatch.setattr(entry.st, "query_params", query_params)
    monkeypatch.setattr(entry.st, "set_page_config", lambda **k: None)
    monkeypatch.setattr(entry.st, "markdown", lambda *a, **k: None)
    monkeypatch.setattr(entry.st, "container", lambda **k: _Ctx())
    monkeypatch.setattr(entry, "load_model", lambda: synthetic_model)
    monkeypatch.setattr(entry, "predict_probability", lambda *a, **k: 75.0)
    monkeypatch.setattr(entry, "render_inputs", lambda m: (2, 0, 2, 2, 0, 0, 2, "otros"))

    for nombre in ("render_header", "render_probability_bar", "render_result_card",
                   "render_comparisons", "render_methodology", "render_footer"):
        monkeypatch.setattr(entry, nombre,
                            lambda *a, _n=nombre, **k: dibujadas.append(_n))

    entry.main()
    return dibujadas


def test_la_caja_saca_la_comparacion_y_la_metodologia(monkeypatch, synthetic_model):
    """Lo que define la versión de caja es qué NO dibuja."""
    from widgets.ive import app as entry

    dibujadas = _correr_entry(monkeypatch, synthetic_model,
                              {entry.PARAM_RESUMEN: "1"})

    assert "render_comparisons" not in dibujadas
    assert "render_methodology" not in dibujadas
    # Y lo que SÍ es la caja, para que el test no pase por no dibujar nada.
    assert "render_result_card" in dibujadas
    assert "render_probability_bar" in dibujadas
    assert "render_header" in dibujadas
    assert "render_footer" in dibujadas


def test_la_version_completa_dibuja_todo(monkeypatch, synthetic_model):
    """El otro lado del control: sin el parámetro no se saca nada."""
    dibujadas = _correr_entry(monkeypatch, synthetic_model, {})

    assert "render_comparisons" in dibujadas
    assert "render_methodology" in dibujadas
    assert "render_result_card" in dibujadas


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
# Los textos que pidió Tomer (19/9/2026)
# ----------------------------------------------------------------------

def test_la_tarjeta_quedo_sin_el_bloque_tecnico(render):
    """
    Pedido editorial de Tomer: el bloque de resultado "es muy técnico" y queda
    sólo con el número y la frase. Salen la línea "entre quienes tienen postura
    definida", el promedio nacional con su brecha, y el % de neutrales.

    Se asertan las AUSENCIAS y también la presencia, porque un render que
    fallara dejaría pasar las tres ausencias sin probar nada.
    """
    components.render_result_card(81.0)

    assert "81%" in render.html
    assert "ejercicio de probabilidades y no una confirmación" in render.html

    assert "postura definida" not in render.html
    assert "Promedio nacional" not in render.html
    assert "result-nacional" not in render.html
    assert "no toma posición clara" not in render.html


def test_la_tarjeta_no_recibe_el_promedio_nacional():
    """
    El promedio salió de la tarjeta, así que tampoco tiene que seguir
    entrando por la firma: un parámetro que nadie usa vuelve a aparecer en
    pantalla solo.
    """
    import inspect

    assert list(inspect.signature(components.render_result_card).parameters) == ["prob"]


def test_los_textos_del_titulo_y_la_bajada(render):
    components.render_header()

    assert "interrupción voluntaria del embarazo" in render.html
    assert "Seleccioná tus características:" in render.html
    # La bajada quedó en una línea: la aclaración metodológica se fue al
    # desplegable del modelo.
    assert "con opinión formada" not in render.html


def test_los_creditos_de_la_encuesta(render, synthetic_model, monkeypatch):
    """
    Los créditos que pidió Tomer, con las tres instituciones y la persona.
    Van juntos y en el mismo lugar, así que se verifican los cuatro.
    """
    class _Exp:
        def __enter__(self): return self
        def __exit__(self, *e): return False

    monkeypatch.setattr(components.st, "expander", lambda *a, **k: _Exp())
    components.render_methodology(synthetic_model)

    for credito in ("El Observador", "UMAD", "Juan Pablo Ferreira",
                    "Juan Ignacio Pintos"):
        assert credito in render.html, f"falta el crédito: {credito}"
    # El typo del mensaje original ("por de El Observador") no se copia.
    assert "por de El Observador" not in render.html


def test_la_bajada_de_la_comparacion(render, synthetic_model):
    components.render_comparisons(synthetic_model, synthetic_model["prob_nacional"])

    assert "A continuación puedes compararte con otras características" in render.html


def test_el_componente_no_elige_colores(render, synthetic_model):
    """
    Los colores los pone la hoja. Un `style="color: …"` en línea es la paleta
    semántica vieja —verde, ámbar, rojo— que el Figma no tiene, y además pisa a
    la hoja.
    """
    components.render_comparisons(synthetic_model, synthetic_model["prob_nacional"])
    components.render_result_card(81.0)
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

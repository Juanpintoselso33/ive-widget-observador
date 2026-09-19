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

    huerfanas = [c for c in propias if f".{c}" not in hoja]
    assert not huerfanas, (
        f"el widget emite clases que la hoja no estila: {sorted(huerfanas)}"
    )


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


def test_el_delta_no_depende_del_perfil_del_lector(render, synthetic_model):
    """
    Control negativo del cambio de significado: antes el delta se calculaba
    contra la predicción del perfil, así que movía al mover los selectores. Hoy
    no puede: `render_comparisons` ya ni recibe la probabilidad del perfil.
    """
    import inspect

    firma = inspect.signature(components.render_comparisons)
    assert list(firma.parameters) == ["model", "prob_nacional"]


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

    assert "color:" not in render.html.replace("color: ", "color:").lower()

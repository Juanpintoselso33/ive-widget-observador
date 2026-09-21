"""La versión resumida, en un navegador de verdad, cruzando el corte de ancho.

Los tests de `test_paridad.py` sólo cargan `modelo.js` en Node: verifican que
`disposicion(1100)` diga "columnas", pero seguirían en verde si el widget no
reacomodara nada, midiera el nodo equivocado, devolviera la tarjeta a otro
lugar o dejara de avisar su altura. Lo marcó Codex. Éste abre la página en
Chrome, la achica y la agranda, y mira el DOM.

Necesita Playwright con Chrome instalado. En la máquina de uno, si no está,
se saltea; en la CI de Pages corre con `IVE_EXIGIR_NAVEGADOR=1`, y ahí no
poder correr es un error, no un salteo — si no, el deploy queda verde sin
haber probado nada (lo marcó Codex).
"""

import functools
import http.server
import os
import threading
from pathlib import Path

import pytest

EXIGIR = bool(os.environ.get("IVE_EXIGIR_NAVEGADOR"))

if EXIGIR:
    import playwright.sync_api as playwright_api
else:
    playwright_api = pytest.importorskip("playwright.sync_api")

WEB = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def url():
    class Callado(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

    manejador = functools.partial(Callado, directory=str(WEB))
    servidor = http.server.ThreadingHTTPServer(("127.0.0.1", 0), manejador)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{servidor.server_address[1]}/ive/"
    servidor.shutdown()


@pytest.fixture(scope="module")
def navegador():
    with playwright_api.sync_playwright() as p:
        try:
            b = p.chromium.launch(headless=True, channel="chrome")
        except Exception as e:  # sin Chrome instalado
            if EXIGIR:
                raise
            pytest.skip(f"no hay Chrome para Playwright: {e}")
        yield b
        b.close()


ESTADO = """() => {
  const w = document.getElementById('widget');
  const tarjeta = w.querySelector('.result-card');
  const barra = w.querySelector('.prob-bar-wrapper');
  const campos = document.getElementById('campos');
  const fila = w.querySelector('.fila-apaisada');
  const selects = [...campos.querySelectorAll('select')];
  const c = document.createElement('canvas').getContext('2d');
  const recortados = selects.filter(s => {
    const cs = getComputedStyle(s);
    c.font = cs.fontSize + ' ' + cs.fontFamily;
    const largo = Math.max(...[...s.options].map(o => c.measureText(o.text).width));
    return largo > s.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
  }).map(s => s.id);
  return {
    apaisado: w.classList.contains('apaisado'),
    tarjeta_en_fila: tarjeta.parentNode === fila,
    tarjeta_tras_barra: barra.nextElementSibling === tarjeta,
    campos_en_fila: campos.parentNode === fila,
    campos_antes_de_fila: campos.nextElementSibling === fila,
    desborda: fila.scrollWidth > fila.clientWidth || campos.scrollWidth > campos.clientWidth,
    recortados,
    foco: document.activeElement && document.activeElement.id,
    aria: tarjeta.getAttribute('aria-live'),
  };
}"""


def _abrir(navegador, url, ancho):
    pagina = navegador.new_page(viewport={"width": ancho, "height": 900})
    pagina.goto(url + "?resumen=1")
    pagina.locator(".result-number").wait_for()
    pagina.evaluate("document.fonts.ready")
    return pagina


def _estado(pagina):
    return pagina.evaluate(ESTADO)


@pytest.mark.parametrize("ancho", [1100, 1280, 1440])
def test_en_columnas_no_se_corta_ninguna_opcion(navegador, url, ancho):
    pagina = _abrir(navegador, url, ancho)
    e = _estado(pagina)
    pagina.close()
    assert e["apaisado"] and e["tarjeta_en_fila"] and e["campos_en_fila"]
    assert not e["desborda"]
    assert e["recortados"] == []


@pytest.mark.parametrize("ancho", [360, 700, 1099])
def test_en_angosto_es_la_caja_de_siempre(navegador, url, ancho):
    pagina = _abrir(navegador, url, ancho)
    e = _estado(pagina)
    pagina.close()
    assert not e["apaisado"]
    assert e["tarjeta_tras_barra"] and e["campos_antes_de_fila"]
    assert e["aria"] == "polite"


def test_cruzar_el_corte_ida_y_vuelta_conserva_el_foco(navegador, url):
    pagina = _abrir(navegador, url, 1099)
    pagina.focus("#campo-nivelEduc")

    pagina.set_viewport_size({"width": 1100, "height": 900})
    pagina.wait_for_function("document.getElementById('widget').classList.contains('apaisado')")
    e = _estado(pagina)
    assert e["tarjeta_en_fila"] and e["foco"] == "campo-nivelEduc"

    pagina.set_viewport_size({"width": 1099, "height": 900})
    pagina.wait_for_function("!document.getElementById('widget').classList.contains('apaisado')")
    e = _estado(pagina)
    assert e["tarjeta_tras_barra"] and e["foco"] == "campo-nivelEduc"
    pagina.close()


def test_la_version_completa_no_se_entera(navegador, url):
    pagina = navegador.new_page(viewport={"width": 1440, "height": 900})
    pagina.goto(url)
    pagina.locator(".result-number").wait_for()
    assert pagina.evaluate("document.querySelector('.fila-apaisada, .apaisado')") is None
    pagina.close()


def test_avisa_la_altura_nueva_al_cruzar_el_corte(navegador, url):
    """Dentro de un iframe, el alto que manda tiene que ser el del layout nuevo."""
    pagina = navegador.new_page(viewport={"width": 1280, "height": 900})
    pagina.goto(url + "tests/")  # cualquier página del mismo origen sirve de anfitriona
    pagina.set_content(
        '<div style="width:1200px" id="caja">'
        f'<iframe id="f" src="{url}?resumen=1" style="width:100%;border:0" scrolling="no"></iframe></div>'
        "<script>window.altos=[];addEventListener('message',e=>{"
        "if(e.data&&e.data.tipo==='ive-widget:alto')altos.push(e.data.alto)});</script>"
    )
    pagina.wait_for_function("window.altos.length > 0")
    pagina.wait_for_timeout(800)
    ancho_alto = pagina.evaluate("window.altos[window.altos.length-1]")

    pagina.evaluate("window.altos=[]; document.getElementById('caja').style.width='600px'")
    pagina.wait_for_function("window.altos.length > 0")
    pagina.wait_for_timeout(800)
    angosto_alto = pagina.evaluate("window.altos[window.altos.length-1]")
    pagina.close()

    # En dos columnas mide ~545px; como caja a 600px de ancho, ~780. Se compara
    # una contra otra y no contra un número fijo, que se rompía con cada ajuste
    # de espacios.
    assert ancho_alto + 150 < angosto_alto


@pytest.mark.parametrize("version", ["", "?resumen=1"])
@pytest.mark.parametrize("ancho", [320, 360, 390, 480, 540, 600])
def test_en_celular_no_se_corta_ninguna_opcion(navegador, url, version, ancho):
    """Con dos columnas parejas, abajo de 600px se cortaban las opciones largas."""
    pagina = navegador.new_page(viewport={"width": ancho, "height": 900})
    pagina.goto(url + version)
    pagina.locator(".result-number").wait_for()
    pagina.evaluate("document.fonts.ready")
    recortados = pagina.evaluate(ESTADO.replace(
        "const fila = w.querySelector('.fila-apaisada');",
        "const fila = w.querySelector('.fila-apaisada') || campos;"))["recortados"]
    pagina.close()
    assert recortados == []


@pytest.mark.parametrize("version", ["", "?resumen=1"])
@pytest.mark.parametrize("ancho", [360, 540, 640, 1280])
def test_el_orden_de_lectura_es_el_que_se_ve(navegador, url, version, ancho):
    """El tabulador recorre los campos en el mismo orden en que se ven."""
    pagina = navegador.new_page(viewport={"width": ancho, "height": 900})
    pagina.goto(url + version)
    pagina.locator(".result-number").wait_for()
    en_dom, en_pantalla = pagina.evaluate("""() => {
      const s = [...document.querySelectorAll('#campos select')];
      const pos = s.map(e => [e.id, Math.round(e.getBoundingClientRect().top), e.getBoundingClientRect().left]);
      const vis = [...pos].sort((a, b) => a[1] - b[1] || a[2] - b[2]).map(p => p[0]);
      return [s.map(e => e.id), vis];
    }""")
    pagina.close()
    assert en_dom == en_pantalla


def test_en_celular_hogar_va_al_lado_de_hijos_y_vuelve(navegador, url):
    pagina = _abrir(navegador, url, 360)
    orden = "() => [...document.querySelectorAll('#campos select')].map(e => e.id)"
    assert pagina.evaluate(orden)[-2:] == ["campo-hogar", "campo-balotaje"]
    pagina.set_viewport_size({"width": 700, "height": 900})
    pagina.wait_for_function("document.querySelectorAll('#campos select')[7].id === 'campo-hogar'")
    assert pagina.evaluate(orden)[-2:] == ["campo-balotaje", "campo-hogar"]
    pagina.close()

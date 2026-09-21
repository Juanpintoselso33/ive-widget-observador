"""
El widget estático tiene que dar EXACTAMENTE los mismos números que el de
Streamlit, en todos los perfiles posibles.

Por qué existe: la versión estática reimplementa la inferencia en JavaScript, y
un coeficiente mal portado, una dummy corrida o un redondeo distinto no rompen
nada — publican otro número. Nadie lo nota mirando la pantalla, porque el
número equivocado se ve igual de creíble que el correcto.

El recorrido es la grilla COMPLETA: 5 tramos de edad × 2 sexos × 4 niveles
educativos × 4 de religiosidad × 2 regiones × 2 (hijos) × 3 tamaños de hogar ×
3 opciones de balotaje = 5.760 perfiles. No es una muestra: es todo lo que el
lector puede armar.

Necesita `node` en el PATH. Si no está, los tests se saltean en vez de fallar:
que no haya Node no dice nada sobre si el porte está bien.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_WEB = Path(__file__).resolve().parent.parent
_ROOT = _WEB.parent.parent
sys.path.insert(0, str(_ROOT))

from widgets.ive.model import predict_probability  # noqa: E402

# El skip va POR TEST y no al módulo entero: puesto arriba, una máquina sin
# Node se salteaba también la sincronía del modelo y el redondeo, que son
# Python puro. Una corrida así daba "todo verde" sin haber validado nada.
# Lo marcó Codex.
necesita_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="hace falta node para correr el JS"
)

# Los mismos rangos que el lector puede elegir en pantalla.
TRAMOS_EDAD = (1, 2, 3, 4, 5)
SEXOS = (0, 1)
NIVELES_EDUC = (1, 2, 3, 4)
RELIGIOSIDAD = (1, 2, 3, 4)
REGIONES = (0, 1)
HIJOS = (0, 1)
HOGARES = (1, 2, 3)
BALOTAJES = ("otros", "martinez", "lacalle")


def _perfiles():
    import itertools
    for c in itertools.product(TRAMOS_EDAD, SEXOS, NIVELES_EDUC, RELIGIOSIDAD,
                               REGIONES, HIJOS, HOGARES, BALOTAJES):
        yield dict(zip(("tramoEdad", "esMujer", "nivelEduc", "religiosidad",
                        "esMontevideo", "tieneHijos", "hogar", "balotaje"), c))


@pytest.fixture(scope="module")
def modelo_web():
    return json.loads((_WEB / "modelo.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def modelo_produccion():
    ruta = _ROOT / "widgets" / "ive" / "model_coefficients.json"
    return json.loads(ruta.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def probabilidades_js():
    """Corre el JS real en Node y devuelve su resultado para cada perfil."""
    # Con `node -e` no hay nombre de script, así que los argumentos arrancan en
    # argv[1] y no en argv[2] como en `node archivo.js`.
    script = """
      const path = require('path');
      const M = require(path.join(process.argv[1], 'modelo.js')).ModeloIVE;
      const modelo = require(path.join(process.argv[1], 'modelo.json'));
      const perfiles = JSON.parse(require('fs').readFileSync(process.argv[2], 'utf8'));
      process.stdout.write(JSON.stringify(
        perfiles.map(p => M.predecir(modelo, p))
      ));
    """
    perfiles = list(_perfiles())
    tmp = _WEB / "tests" / "_perfiles.json"
    tmp.write_text(json.dumps(perfiles), encoding="utf-8")
    try:
        salida = subprocess.run(
            ["node", "-e", script, str(_WEB), str(tmp)],
            capture_output=True, text=True, check=True, timeout=120,
        ).stdout
    finally:
        tmp.unlink(missing_ok=True)
    return perfiles, json.loads(salida)


@necesita_node
def test_la_grilla_completa_da_lo_mismo_en_python_y_en_js(modelo_produccion, probabilidades_js):
    """
    Cero diferencia visible. Se compara con tolerancia de 1e-9 porque los dos
    lenguajes hacen la misma cuenta en coma flotante de 64 bits, pero pueden
    diferir en el último bit; cualquier error de porte es de otra magnitud.
    """
    perfiles, js = probabilidades_js

    assert len(perfiles) == 5760, f"la grilla no está completa: {len(perfiles)}"
    assert len(js) == len(perfiles)

    peor = 0.0
    fallos = []
    for perfil, valor_js in zip(perfiles, js):
        valor_py = predict_probability(
            modelo_produccion,
            perfil["tramoEdad"], perfil["esMujer"], perfil["nivelEduc"],
            perfil["religiosidad"], perfil["esMontevideo"], perfil["tieneHijos"],
            perfil["hogar"], perfil["balotaje"],
        )
        d = abs(valor_py - valor_js)
        peor = max(peor, d)
        if d > 1e-9 and len(fallos) < 5:
            fallos.append((perfil, valor_py, valor_js))

    assert not fallos, f"perfiles con distinto resultado (peor Δ={peor}): {fallos}"
    # Control de que la comparación no pasó por vacío: con 5.760 perfiles y un
    # modelo real, la diferencia máxima tiene que existir aunque sea ínfima.
    assert peor < 1e-9


@necesita_node
def test_los_dos_extremos_del_modelo_no_se_pisan(probabilidades_js):
    """
    Que la grilla produzca RANGO, y no el mismo número siempre.

    Sin esto, un `predecir()` que devolviera una constante pasaría el test de
    paridad anterior sólo con que el Python también estuviera roto igual — o,
    más realista, si alguien rompiera el porte de los coeficientes dejando el
    intercepto solo.
    """
    _, js = probabilidades_js
    assert min(js) < 20, f"el perfil más opuesto da {min(js)}"
    assert max(js) > 95, f"el perfil más favorable da {max(js)}"


def test_el_modelo_del_web_no_se_desincroniza_del_de_produccion(modelo_web, modelo_produccion):
    """
    `web/ive/modelo.json` es una COPIA recortada del artefacto entrenado, y dos
    copias se separan solas: un reentrenamiento actualiza una y deja la otra
    publicando los coeficientes viejos, sin que nada falle.
    """
    assert modelo_web["coefficients"] == modelo_produccion["coefficients"]
    assert modelo_web["prob_nacional"] == modelo_produccion["prob_nacional"]
    assert modelo_web["stats_by_group"] == modelo_produccion["stats_by_group"]
    assert modelo_web["variable_ranges"] == modelo_produccion["variable_ranges"]
    assert (modelo_web["model_info"]["n_observations"]
            == modelo_produccion["model_info"]["n_observations"])
    assert (modelo_web["model_info"]["pseudo_r2"]
            == modelo_produccion["model_info"]["pseudo_r2"])


def test_el_web_no_publica_el_modelo_de_neutralidad(modelo_web):
    """
    Lo que no se muestra, no viaja al navegador. El modelo de neutralidad salió
    de pantalla cuando se adelgazó la tarjeta; mandarlo igual sería publicar
    coeficientes que nadie usa.
    """
    for clave in ("coefficients_neutral", "odds_ratios_neutral",
                  "model_info_neutral", "prob_neutral_nacional"):
        assert clave not in modelo_web


@necesita_node
@pytest.mark.parametrize("valor, esperado", [
    (76.5, 76),   # el promedio nacional: Python da 76, Math.round daría 77
    (77.5, 78),
    (78.8, 79),
    (14.6, 15),
    (0.5, 0),
    # Casi-empates: con la tolerancia que tenía, estos dos daban distinto que
    # Python. El modelo actual no los produce, pero un reentrenamiento sí.
    (76.5000000001, 77),
    (77.4999999999, 77),
    (-0.5, 0),
    (2.5, 2),
])
def test_el_redondeo_del_js_es_el_de_python(valor, esperado):
    """
    JavaScript redondea 76,5 a 77 y Python a 76 — redondeo al par. El promedio
    nacional es exactamente 76,5 y TODAS las diferencias por grupo se calculan
    contra él ya redondeado, así que con el redondeo de JS las quince cifras de
    la grilla saldrían corridas un punto.
    """
    salida = subprocess.run(
        ["node", "-e",
         f"const M=require('{_WEB}/modelo.js').ModeloIVE;process.stdout.write(String(M.redondear({valor})))"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert int(salida) == esperado == round(valor)


# ----------------------------------------------------------------------
# El mapeo de la interfaz al modelo
# ----------------------------------------------------------------------

def _perfil_esperado(rangos, i):
    """
    La conversión índice→perfil, escrita A MANO y de forma independiente.

    Es el control del mapeo: el test anterior comparaba Python contra JS con
    perfiles YA codificados, así que nunca ejercía esta conversión. Cambiar un
    `idx + 1` por `idx`, dar vuelta sexo y región o alterar el orden de las
    opciones pasaba en verde mientras el lector recibía otro perfil.
    """
    balotaje = {"No votó/Blanco": "otros", "Orsi (FA)": "martinez",
                "Delgado (Coalición)": "lacalle"}
    return {
        "tramoEdad": i["tramoEdad"] + 1,
        "esMujer": i["esMujer"],
        "nivelEduc": i["nivelEduc"] + 1,
        "religiosidad": i["religiosidad"] + 1,
        "esMontevideo": i["esMontevideo"],
        "tieneHijos": i["tieneHijos"],
        "hogar": i["hogar"] + 1,
        "balotaje": balotaje[rangos["balotaje"]["labels"][i["balotaje"]]],
    }


@necesita_node
def test_el_mapeo_de_indices_al_modelo(modelo_web):
    """Toda combinación de posiciones de los ocho desplegables, no una muestra."""
    import itertools

    rangos = modelo_web["variable_ranges"]
    combos = [
        dict(zip(("tramoEdad", "esMujer", "nivelEduc", "religiosidad",
                  "esMontevideo", "tieneHijos", "hogar", "balotaje"), c))
        for c in itertools.product(
            range(len(rangos["tramo_edad_num"]["labels"])),
            range(len(rangos["es_mujer"]["labels"])),
            range(len(rangos["nivel_educ_num"]["labels"])),
            range(len(rangos["religiosidad_num"]["labels"])),
            range(len(rangos["es_montevideo"]["labels"])),
            range(len(rangos["tiene_hijos"]["labels"])),
            range(len(rangos["hogar_num"]["labels"])),
            range(len(rangos["balotaje"]["labels"])),
        )
    ]
    assert len(combos) == 5760

    tmp = _WEB / "tests" / "_indices.json"
    tmp.write_text(json.dumps(combos), encoding="utf-8")
    try:
        salida = subprocess.run(
            ["node", "-e", """
               const path = require('path');
               const M = require(path.join(process.argv[1], 'modelo.js')).ModeloIVE;
               const rangos = require(path.join(process.argv[1], 'modelo.json')).variable_ranges;
               const combos = JSON.parse(require('fs').readFileSync(process.argv[2], 'utf8'));
               process.stdout.write(JSON.stringify(combos.map(i => M.perfilDesdeIndices(rangos, i))));
             """, str(_WEB), str(tmp)],
            capture_output=True, text=True, check=True, timeout=120,
        ).stdout
    finally:
        tmp.unlink(missing_ok=True)

    perfiles_js = json.loads(salida)
    for indices, perfil in zip(combos, perfiles_js):
        assert perfil == _perfil_esperado(rangos, indices), f"con los índices {indices}"


@necesita_node
def test_el_widget_arranca_en_el_mismo_perfil_que_la_version_publicada(modelo_web):
    """
    Los valores por defecto de los ocho desplegables. Si uno cambia, el lector
    ve otro número apenas abre la nota, antes de tocar nada.
    """
    esperado = {"tramoEdad": 2, "esMujer": 0, "nivelEduc": 2, "religiosidad": 2,
                "esMontevideo": 0, "tieneHijos": 0, "hogar": 2, "balotaje": "otros"}
    salida = subprocess.run(
        ["node", "-e", """
           const path = require('path');
           const M = require(path.join(process.argv[1], 'modelo.js')).ModeloIVE;
           const r = require(path.join(process.argv[1], 'modelo.json')).variable_ranges;
           const i = {
             tramoEdad: M.indicePorDefecto(r.tramo_edad_num),
             esMujer: M.indicePorDefecto(r.es_mujer),
             nivelEduc: M.indicePorDefecto(r.nivel_educ_num),
             religiosidad: M.indicePorDefecto(r.religiosidad_num),
             esMontevideo: M.indicePorDefecto(r.es_montevideo),
             tieneHijos: M.indicePorDefecto(r.tiene_hijos),
             hogar: M.indicePorDefecto(r.hogar_num),
             balotaje: M.indicePorDefecto(r.balotaje)
           };
           process.stdout.write(JSON.stringify(M.perfilDesdeIndices(r, i)));
         """, str(_WEB)],
        capture_output=True, text=True, check=True,
    ).stdout
    assert json.loads(salida) == esperado


# ----------------------------------------------------------------------
# Qué versión se dibuja y cómo se acomoda
# ----------------------------------------------------------------------

def _node(expr, *args):
    return json.loads(subprocess.run(
        ["node", "-e",
         "const M=require(process.argv[1]+'/modelo.js').ModeloIVE;"
         f"process.stdout.write(JSON.stringify({expr}))",
         str(_WEB), *args],
        capture_output=True, text=True, check=True,
    ).stdout)


@necesita_node
@pytest.mark.parametrize("resumen, apaisado, esperado", [
    ([], [], False),            # la nota
    (["1"], [], True),          # la home
    # `?apaisado=1` quedó como sinónimo: existió un rato como versión aparte y
    # un código ya pegado con ese parámetro tiene que seguir andando.
    ([], ["1"], True),
    (["0"], ["1"], True),
    # Repetido, gana el último — la misma regla que la versión Streamlit.
    (["1", "0"], [], False),
    (["0", "1"], [], True),
    # Valores que NO activan: sin este control, un `parametroActivo` que
    # devolviera siempre True pasaría casi todos los casos de arriba.
    (["cualquiera"], ["no"], False),
    ([""], [""], False),
])
def test_la_version_segun_la_url(resumen, apaisado, esperado):
    r = _node("M.version(JSON.parse(process.argv[2]),JSON.parse(process.argv[3]))",
              json.dumps(resumen), json.dumps(apaisado))
    assert r == {"resumen": esperado}


@necesita_node
@pytest.mark.parametrize("ancho, esperado", [
    (320, "caja"), (400, "caja"), (700, "caja"),
    (899, "caja"),       # el borde, de los dos lados
    (900, "columnas"),
    (1100, "columnas"), (1280, "columnas"),
])
def test_la_disposicion_la_decide_el_ancho(ancho, esperado):
    """
    La versión resumida se acomoda por el ANCHO DISPONIBLE, no por la URL: así
    la home usa un solo embed para escritorio y móvil. Se prueban los dos lados
    del corte porque un `>` en vez de `>=` corre el borde un pixel y no se ve.
    """
    assert _node(f"M.disposicion({ancho})") == esperado

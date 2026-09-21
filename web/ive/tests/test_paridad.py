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

pytestmark = pytest.mark.skipif(
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


@pytest.mark.parametrize("valor, esperado", [
    (76.5, 76),   # el promedio nacional: Python da 76, Math.round daría 77
    (77.5, 78),
    (78.8, 79),
    (14.6, 15),
    (0.5, 0),
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

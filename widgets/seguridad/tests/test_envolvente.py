"""
La envolvente de especificación: que ensanche, que no angoste, y que el widget
no afirme nada que una especificación admisible contradiga.

POR QUÉ CONTRA EL ARTEFACTO DE PRODUCCIÓN Y NO CONTRA COEFICIENTES SINTÉTICOS.
El resto de los tests del widget usan coeficientes sintéticos a propósito, para
verificar aritmética sin depender del entrenamiento. Acá la propiedad que
interesa es justamente una relación ENTRE DOS ARTEFACTOS REALES —los modelos y
la envolvente—, que es lo que se puede desincronizar al reentrenar. Un test
sintético no la tocaría.

EL COSTO SE CONTROLA MUESTREANDO PERFILES. Barrer los 1.008 perfiles por las
cuatro preguntas evaluando 10.000 réplicas en cada uno tarda minutos, que es
demasiado para la suite. Se toma una muestra fija por semilla, más los perfiles
que el estudio marcó como los que se salían del intervalo, que son justo los
casos donde la envolvente hace algo.
"""

import itertools
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest

from widgets.seguridad import config
from widgets.seguridad import model as m

CAMPOS = ("tramo_edad", "es_mujer", "nivel_educ", "ideologia", "victima",
          "es_montevideo")
MUESTRA = 25


def _perfiles_ui():
    return [
        dict(zip(CAMPOS, v))
        for v in itertools.product(
            sorted(set(config.EDAD_UI_TO_CODE.values())), (0, 1),
            sorted(set(config.EDUC_UI_TO_CODE.values())),
            sorted(set(config.IDEOLOGIA_UI_TO_CODE.values())),
            sorted(set(config.VICTIMA_UI_TO_CODE.values())),
            sorted(set(config.REGION_UI_TO_CODE.values())))
    ]


def _desde_clave(clave):
    return dict(zip(CAMPOS, (int(x) for x in clave.split("-"))))


@pytest.fixture(scope="module")
def envolvente():
    env = m.load_envolvente()
    if env is None:
        pytest.skip("no hay envolvente_espec.json en este árbol")
    return env


@pytest.fixture(scope="module")
def modelos():
    try:
        return m.load_modelos()
    except FileNotFoundError:
        pytest.skip("no hay modelos entrenados en este árbol")


def _casos(env, slug):
    """Perfiles a probar: los que la envolvente mueve, más una muestra fija."""
    tabla = env["preguntas"][slug]["perfiles"]
    todos = _perfiles_ui()
    rnd = random.Random(20260909)
    muestra = rnd.sample(todos, MUESTRA)
    # Los que más importan: aquellos donde la envolvente sobresale del rango de
    # las réplicas, o sea donde ensanchar cambia algo.
    anchos = sorted(tabla.items(), key=lambda kv: kv[1][0] - kv[1][1])[:MUESTRA]
    return muestra + [_desde_clave(k) for k, _ in anchos]


@pytest.mark.parametrize("slug", config.SLUGS)
def test_el_intervalo_publicado_contiene_a_todas_las_especificaciones(
        envolvente, modelos, slug):
    """Es la propiedad por la que existe todo esto."""
    tabla = envolvente["preguntas"][slug]["perfiles"]
    for perfil in _casos(envolvente, slug):
        lo, hi = m.intervalo_probabilidad(modelos[slug], **perfil)
        e_lo, e_hi = tabla[config.clave_perfil(**perfil)]
        assert lo <= e_lo + 1e-9, f"{slug} {perfil}: {lo} > {e_lo}"
        assert hi >= e_hi - 1e-9, f"{slug} {perfil}: {hi} < {e_hi}"


@pytest.mark.parametrize("slug", config.SLUGS)
def test_ensanchar_nunca_angosta(envolvente, modelos, slug):
    """El intervalo con envolvente contiene al de sin envolvente, siempre."""
    for perfil in _casos(envolvente, slug):
        for fn in (m.intervalo_probabilidad, m.banda_decision,
                   m.intervalo_brecha):
            crudo = fn(modelos[slug], **perfil, aplicar_envolvente=False)
            ancho = fn(modelos[slug], **perfil)
            if crudo is None:
                continue
            assert ancho[0] <= crudo[0] + 1e-9
            assert ancho[1] >= crudo[1] - 1e-9


@pytest.mark.parametrize("slug", config.SLUGS)
def test_ninguna_afirmacion_queda_contradicha(envolvente, modelos, slug):
    """
    Si el widget afirma de qué lado está la mayoría, ninguna especificación
    admisible puede poner al perfil del otro lado. Y lo mismo para la brecha
    contra el promedio nacional.

    Antes de la envolvente esto NO se cumplía: en pena de muerte, 2 afirmaciones
    de mayoría y 4 de brecha quedaban contradichas por alguna especificación.
    """
    modelo = modelos[slug]
    tabla = envolvente["preguntas"][slug]["perfiles"]
    nacional = modelo["prob_favor_nacional"]
    for perfil in _casos(envolvente, slug):
        e_lo, e_hi = tabla[config.clave_perfil(**perfil)]

        banda = m.banda_decision(modelo, **perfil)
        if banda and not (round(banda[0]) <= 50 <= round(banda[1])):
            # afirma un lado -> ninguna especificación en el otro
            assert not (e_lo < 50 < e_hi), (
                f"{slug} {perfil}: afirma lado pero las especificaciones "
                f"van de {e_lo} a {e_hi}")

        brecha = m.intervalo_brecha(modelo, **perfil)
        if brecha and not (round(brecha[0]) <= 0 <= round(brecha[1])):
            assert not (e_lo < nacional < e_hi), (
                f"{slug} {perfil}: afirma brecha pero las especificaciones "
                f"cruzan el promedio {nacional}")


def test_sin_archivo_la_aritmetica_no_rompe(modelos, monkeypatch):
    """
    Que falte la envolvente no puede tirar una excepción: las funciones
    devuelven el intervalo sin ensanchar, que es lo que se publicaba antes.
    """
    monkeypatch.setattr(m, "_ENVOLVENTE", None)
    monkeypatch.setattr(m, "_ENVOLVENTE_CARGADA", True)
    perfil = dict(zip(CAMPOS, (2, 1, 1, 7, 1, 1)))
    slug = config.SLUGS[0]
    assert m.envolvente_perfil(slug, **perfil) is None
    con = m.intervalo_probabilidad(modelos[slug], **perfil)
    sin = m.intervalo_probabilidad(modelos[slug], **perfil,
                                   aplicar_envolvente=False)
    assert con == sin


def test_sin_archivo_el_arranque_lo_grita(modelos, monkeypatch):
    """
    Y sin embargo TIENE que avisar. Degradar en silencio significaría publicar
    intervalos más angostos con la pantalla idéntica, que es la clase de error
    que nadie nota. La aritmética aguanta; el arranque protesta.
    """
    monkeypatch.setattr(m, "_ENVOLVENTE", None)
    monkeypatch.setattr(m, "_ENVOLVENTE_CARGADA", True)
    slug = config.SLUGS[0]
    problemas = m.problemas_de_envolvente(slug, modelos[slug])
    assert problemas, "que falte el archivo tiene que ser un problema"
    assert "envolvente_espec.json" in problemas[0]


def test_una_envolvente_de_otro_modelo_se_detecta(modelos, monkeypatch):
    """
    El control que ata el artefacto a estos coeficientes. Se corre la envolvente
    lejos del punto publicado y `problemas_de_envolvente` tiene que decirlo: si
    no lo dijera, un reentrenamiento sin regenerar dejaría al widget ensanchando
    con números de otros datos, que es peor que no ensanchar.
    """
    slug = "pena_muerte"
    env = m.load_envolvente()
    if env is None:
        pytest.skip("no hay envolvente_espec.json en este árbol")
    corrida = json.loads(json.dumps(env))
    tabla = corrida["preguntas"][slug]["perfiles"]
    for k in tabla:
        tabla[k] = [tabla[k][0] + 40.0, tabla[k][1] + 40.0]
    monkeypatch.setattr(m, "_ENVOLVENTE", corrida)
    monkeypatch.setattr(m, "_ENVOLVENTE_CARGADA", True)

    problemas = m.problemas_de_envolvente(slug, modelos[slug], _perfiles_ui())
    assert problemas, "una envolvente corrida 40 pp tiene que dar problema"
    assert any("FUERA de la envolvente" in p for p in problemas), problemas


def test_contrato_distinto_se_detecta(modelos, monkeypatch):
    """Si cambió la configuración, la envolvente vieja no se puede usar."""
    slug = config.SLUGS[0]
    env = m.load_envolvente()
    if env is None:
        pytest.skip("no hay envolvente_espec.json en este árbol")
    corrida = json.loads(json.dumps(env))
    corrida["preguntas"][slug]["contrato"] = "0000000000000000"
    monkeypatch.setattr(m, "_ENVOLVENTE", corrida)
    monkeypatch.setattr(m, "_ENVOLVENTE_CARGADA", True)
    problemas = m.problemas_de_envolvente(slug, modelos[slug])
    assert any("contrato" in p for p in problemas), problemas


def test_ensanchar_es_una_union_no_una_suma():
    """
    La distinción no es cosmética: sumar el rango entre especificaciones al
    ancho del bootstrap contaría dos veces el mismo ruido de estimación. Si la
    envolvente cae adentro, el intervalo no se toca.
    """
    assert m._ensanchar((10.0, 30.0), (15.0, 25.0)) == (10.0, 30.0)
    assert m._ensanchar((10.0, 30.0), (5.0, 25.0)) == (5.0, 30.0)
    assert m._ensanchar((10.0, 30.0), (15.0, 44.0)) == (10.0, 44.0)
    assert m._ensanchar((10.0, 30.0), None) == (10.0, 30.0)
    assert m._ensanchar(None, (1.0, 2.0)) is None
    # Con desplazamiento: la envolvente se lleva a la escala de la resta.
    assert m._ensanchar((-5.0, 5.0), (80.0, 90.0), desplazamiento=70.0) == (
        -5.0, 20.0)

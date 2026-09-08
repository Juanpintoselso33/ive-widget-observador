"""
Tests del GENERADOR de las réplicas bootstrap.

Existen porque hasta acá ningún test ejecutaba `bootstrap_coeficientes`: todo
lo que había miraba el JSON ya producido. Codex lo marcó con un ejemplo
concreto — volver a fijar C en vez de re-elegirlo por réplica dejaba los 101
tests en verde y achicaba todos los intervalos publicados en varios puntos.

Un test sobre el artefacto comprueba que el archivo tiene la forma esperada;
sólo un test sobre el generador comprueba que el procedimiento que lo produjo
sigue siendo el declarado.

Corre sobre datos sintéticos chicos: no toca la base real ni el JSON de
producción.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import pytest

from widgets.seguridad import train_model as tm
from widgets.seguridad.config import PREDICTORES


@pytest.fixture(scope="module")
def sintetico():
    """
    Muestra chica con estructura suficiente para que la CV tenga algo que
    elegir: señal débil en tres predictores y ruido en el resto, que es la
    situación en la que C importa.
    """
    rng = np.random.default_rng(7)
    n = 400
    X = rng.integers(0, 2, size=(n, len(PREDICTORES))).astype(float)
    beta = np.zeros(len(PREDICTORES))
    beta[:3] = [1.2, -0.9, 0.6]
    z = -0.2 + X @ beta
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    w = rng.uniform(0.5, 1.5, size=n)
    d = pd.DataFrame({"estrato": rng.choice(["a", "b", "c", "d"], size=n)})
    return d, X, y, w


def test_el_bootstrap_reelige_c_en_cada_replica(sintetico):
    """
    El corazón del asunto. Si C se vuelve a fijar, `c_por_replica` colapsa a un
    único valor y este test falla — que es exactamente lo que no pasaba antes.
    """
    d, X, y, w = sintetico
    coefs, meta, _ = tm.bootstrap_coeficientes(d, X, y, w, n_replicas=15)

    assert len(meta["c_por_replica"]) > 1, (
        "todas las réplicas eligieron el mismo C: o la re-selección se rompió, "
        f"o dejó de hacerse. Distribución: {meta['c_por_replica']}"
    )
    assert set(meta["c_por_replica"]) <= {str(c) for c in tm.C_GRID}
    assert sum(meta["c_por_replica"].values()) == len(coefs)


def test_la_metadata_reconcilia_con_las_replicas(sintetico):
    d, X, y, w = sintetico
    coefs, meta, _ = tm.bootstrap_coeficientes(d, X, y, w, n_replicas=15)

    assert meta["solicitadas"] == 15
    assert meta["utiles"] == len(coefs)
    assert meta["utiles"] <= meta["solicitadas"]
    assert meta["semilla"] == tm.RANDOM_STATE


def test_cada_replica_trae_intercepto_mas_un_coeficiente_por_predictor(sintetico):
    d, X, y, w = sintetico
    coefs, _, _ = tm.bootstrap_coeficientes(d, X, y, w, n_replicas=5)
    assert coefs, "no sobrevivió ninguna réplica"
    for fila in coefs:
        assert len(fila) == len(PREDICTORES) + 1
        assert all(np.isfinite(v) for v in fila)


def test_es_reproducible(sintetico):
    """Misma semilla, mismos coeficientes: si no, el JSON no es auditable."""
    d, X, y, w = sintetico
    a, _, _ = tm.bootstrap_coeficientes(d, X, y, w, n_replicas=5)
    b, _, _ = tm.bootstrap_coeficientes(d, X, y, w, n_replicas=5)
    assert a == b


def test_el_remuestreo_es_estratificado(sintetico, monkeypatch):
    """
    Cada réplica conserva el tamaño de cada estrato. Se comprueba mirando los
    índices que el generador elige, no el resultado: un remuestreo simple daría
    estratos de tamaño variable y los intervalos serían otros.
    """
    d, X, y, w = sintetico
    tamanos = d["estrato"].value_counts().to_dict()

    vistos = []

    class RngEspia:
        """Delega todo en el generador real y anota los llamados a choice."""

        def __init__(self, rng):
            self._rng = rng

        def choice(self, a, size=None, replace=True, **kw):
            vistos.append((tuple(a), size))
            return self._rng.choice(a, size=size, replace=replace, **kw)

        def __getattr__(self, nombre):
            return getattr(self._rng, nombre)

    real = np.random.default_rng
    monkeypatch.setattr(tm.np.random, "default_rng",
                        lambda *a, **kw: RngEspia(real(*a, **kw)))
    tm.bootstrap_coeficientes(d, X, y, w, n_replicas=2)

    assert vistos, "el generador no llamó a choice: cambió el remuestreo"
    for indices, size in vistos:
        # Cada llamada remuestrea UN estrato completo, con su tamaño original.
        estrato = d["estrato"].iloc[list(indices)].unique()
        assert len(estrato) == 1, "una llamada mezcló estratos"
        assert size == len(indices) == tamanos[estrato[0]]


def test_los_mapas_quedan_apareados_con_los_coeficientes_aunque_se_descarte(
        sintetico, monkeypatch):
    """
    Si una réplica de coeficientes se descarta, el mapa de ESE sorteo también
    tiene que caer.

    Es el defecto latente que marcó Codex el 8/9/2026: `bootstrap_coeficientes`
    descarta réplicas —sin variación en la dependiente, o con un fold degenerado
    en la CV— y compacta su lista, mientras que las réplicas del mapa se
    generaban todas. A partir del primer descarte, `model.py` pega la réplica i
    de coeficientes con el mapa del sorteo i+1, y el intervalo publicado sale de
    dos remuestreos distintos.

    No estaba pasando —las cuatro preguntas tienen 10.000 de 10.000— y por eso
    hace falta forzarlo: se descarta a mano la tercera réplica.
    """
    d, X, y, w = sintetico
    d = d.assign(a_favor=y)

    real = tm.elegir_c
    llamadas = {"n": 0}

    def elegir_c_que_falla_en_la_tercera(*a, **k):
        llamadas["n"] += 1
        if llamadas["n"] == 4:      # 1 es el ajuste principal si lo hubiera
            return None, None
        return real(*a, **k)

    monkeypatch.setattr(tm, "elegir_c", elegir_c_que_falla_en_la_tercera)
    coefs, meta, validos = tm.bootstrap_coeficientes(d, X, y, w, n_replicas=8)
    monkeypatch.setattr(tm, "elegir_c", real)

    assert meta["utiles"] < meta["solicitadas"], (
        "no se descartó ninguna réplica: el test no probó nada"
    )
    assert len(validos) == len(coefs)
    assert sorted(set(validos)) == validos, "los sorteos válidos vienen desordenados"

    todos = tm.ajustar_calibracion(d, X, y, w, n_replicas=8)
    filtrados = tm.ajustar_calibracion(d, X, y, w, n_replicas=8,
                                       sorteos_validos=validos)
    assert todos is not None and filtrados is not None

    assert len(filtrados["replicas"]) == len(coefs), (
        "quedan más mapas que coeficientes: el apareamiento se corre"
    )
    esperado = [todos["replicas"][k] for k in validos]
    assert filtrados["replicas"] == esperado, (
        "los mapas que sobrevivieron no son los de los sorteos que sobrevivieron"
    )

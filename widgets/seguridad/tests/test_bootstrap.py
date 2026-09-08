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


def _reconstruir_sorteos(d, n):
    """
    Los n remuestreos estratificados que produce `default_rng(RANDOM_STATE)`.

    Se rehace acá a propósito, sin llamar a ninguna función de train_model: un
    test que verifica el apareamiento no puede pedirle los índices al código que
    está verificando.
    """
    estratos = d["estrato"].values
    indices = [np.where(estratos == e)[0] for e in np.unique(estratos)]
    rng = np.random.default_rng(tm.RANDOM_STATE)
    return [np.concatenate([rng.choice(ix, size=len(ix), replace=True)
                            for ix in indices]) for _ in range(n)]


def _nodos_a_mano(oof, y, w, k):
    """
    Los nodos (x, y) del mapa, reimplementados a mano para el test.

    NO llama a `tm._nodos_calibracion`: si el esperado sale de la misma función
    que el observado, los dos se equivocan juntos. Codex lo mostró metiendo
    `w = np.ones_like(w)` adentro de esa función — los mapas bootstrap perdían
    la ponderación y los 148 tests seguían en verde.

    Los grupos son de igual MASA PONDERADA, no de igual cantidad de casos, y a
    los extremos se les pega un 0 y un 1 con la monotonía forzada.
    """
    orden = sorted(range(len(oof)), key=lambda i: oof[i])
    total = sum(w)
    acum, corriente = {}, 0.0
    for i in orden:
        corriente += w[i]
        acum[i] = min(int(corriente / total * k), k - 1)
    xs, ys = [], []
    for grupo in sorted(set(acum.values())):
        ix = [i for i in range(len(oof)) if acum[i] == grupo]
        peso = sum(w[i] for i in ix)
        xs.append(sum(oof[i] * w[i] for i in ix) / peso)
        ys.append(sum(y[i] * w[i] for i in ix) / peso)
    xs = [0.0] + xs + [1.0]
    ys = [min(ys[0], float(min(oof)))] + ys + [max(ys[-1], float(max(oof)))]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1])
    return xs, ys


def _oof_como_produccion(X, y, w):
    """Las predicciones fuera de fold sobre las que se ajusta el mapa."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    cv = StratifiedKFold(5, shuffle=True, random_state=tm.RANDOM_STATE)
    oof = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        c_fold, _ = tm.elegir_c(X[tr], y[tr], w[tr])
        m = LogisticRegression(C=c_fold, max_iter=2000,
                               random_state=tm.RANDOM_STATE)
        m.fit(X[tr], y[tr], sample_weight=w[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def test_los_mapas_quedan_apareados_con_los_coeficientes_aunque_se_descarte(
        sintetico, monkeypatch):
    """
    La réplica j de coeficientes y la j del mapa tienen que salir del MISMO
    remuestreo, incluso cuando alguna réplica se descarta.

    Es el defecto latente que marcó Codex el 8/9/2026: `bootstrap_coeficientes`
    descarta réplicas —sin variación en la dependiente, o con un fold degenerado
    en la CV— y compacta su lista, mientras que las réplicas del mapa se
    generaban todas. A partir del primer descarte, `model.py` pega la réplica j
    de coeficientes con el mapa del sorteo j+1, y el intervalo publicado sale de
    dos remuestreos distintos.

    LA PRIMERA VERSIÓN DE ESTE TEST NO PROBABA ESO. Comparaba el mapa filtrado
    contra `[mapa_completo[k] for k in validos]`, o sea que usaba como esperado
    la misma lista de índices que quería verificar: pasaba igual si
    `bootstrap_coeficientes` devolvía `range(len(coefs))` —perdiendo el hueco—
    y también si el calibrador cambiaba de semilla. Lo mostró Codex con los dos
    controles negativos.

    Ahora los sorteos se reconstruyen acá, con `default_rng(RANDOM_STATE)` y los
    mismos estratos, sin preguntarle nada a train_model; y se comprueba que el
    coeficiente j y el mapa j salen del sorteo reconstruido que les toca.

    No estaba pasando en producción —las cuatro preguntas tienen 10.000 de
    10.000— así que hay que forzar el descarte a mano.
    """
    from sklearn.linear_model import LogisticRegression

    d, X, y, w = sintetico
    d = d.assign(a_favor=y)
    n = 8

    real = tm.elegir_c
    llamadas = {"n": 0}

    def elegir_c_que_falla_en_una(*a, **k):
        llamadas["n"] += 1
        if llamadas["n"] == 4:
            return None, None
        return real(*a, **k)

    monkeypatch.setattr(tm, "elegir_c", elegir_c_que_falla_en_una)
    coefs, meta, validos = tm.bootstrap_coeficientes(d, X, y, w, n_replicas=n)
    monkeypatch.setattr(tm, "elegir_c", real)

    assert meta["utiles"] < meta["solicitadas"], (
        "no se descartó ninguna réplica: el test no probó nada"
    )
    assert len(validos) == len(coefs)
    assert sorted(set(validos)) == validos, "los sorteos válidos vienen desordenados"
    assert set(range(len(coefs))) != set(validos), (
        "los índices válidos son 0..k sin hueco: el descarte no quedó registrado"
    )

    calibracion = tm.ajustar_calibracion(d, X, y, w, n, validos)
    assert calibracion is not None
    assert len(calibracion["replicas"]) == len(coefs), (
        "quedan más mapas que coeficientes: el apareamiento se corre"
    )

    sorteos = _reconstruir_sorteos(d, n)
    oof = _oof_como_produccion(X, y, w)

    for j, k in enumerate(validos):
        idx = sorteos[k]

        # El coeficiente j tiene que ser el del sorteo k.
        c_k, _ = tm.elegir_c(X[idx], y[idx], w[idx])
        m = LogisticRegression(C=c_k, max_iter=2000, random_state=tm.RANDOM_STATE)
        m.fit(X[idx], y[idx], sample_weight=w[idx])
        esperado = [float(m.intercept_[0])] + [float(v) for v in m.coef_[0]]
        assert np.allclose(coefs[j], esperado, atol=1e-8), (
            f"la réplica {j} de coeficientes no sale del sorteo {k}"
        )

        # Y el mapa j, del MISMO sorteo k, con los nodos calculados a mano.
        xs_k, ys_k = _nodos_a_mano(list(oof[idx]), list(y[idx]), list(w[idx]),
                                   tm.NODOS_CALIBRACION)
        xs_j, ys_j = calibracion["replicas"][j]
        assert np.allclose(xs_j, [round(float(v), 6) for v in xs_k], atol=1e-6), (
            f"el mapa {j} no sale del sorteo {k}"
        )
        assert np.allclose(ys_j, [round(float(v), 6) for v in ys_k], atol=1e-6), (
            f"el mapa {j} no sale del sorteo {k}"
        )

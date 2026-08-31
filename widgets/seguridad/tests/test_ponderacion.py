"""
Tests de la PONDERACIÓN por diseño muestral.

Existen porque Codex probó tres mutaciones sobre el corazón estadístico del
modelo y las tres dejaban los 71 tests en verde:

  - sacar `sample_weight` de todos los `fit`
  - sacar `sample_weight` del `log_loss`
  - promediar los folds sin ponderar por su masa de pesos

Las tres producen un modelo distinto y publicable, sin un solo error en
pantalla. Es el mismo agujero que ya había aparecido dos veces en esta base:
los tests miraban el artefacto final y no el procedimiento que lo produce.

Sobre datos sintéticos: no tocan la base real ni el JSON de producción.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from widgets.seguridad import train_model as tm


@pytest.fixture
def datos_donde_el_peso_da_vuelta_el_signo():
    """
    Un predictor binario cuya relación con la respuesta se INVIERTE al ponderar.

    Sin ponderar: entre los x=1 mandan los muchos y=0, así que el coeficiente
    sale negativo. Ponderando: los pocos y=1 pesan 60 veces más y lo dan vuelta.
    Es el caso que distingue "usa los pesos" de "los ignora" sin ambigüedad.
    """
    n_ceros, n_unos = 120, 12
    x = np.concatenate([np.ones(n_ceros + n_unos), np.zeros(120)])
    y = np.concatenate([np.zeros(n_ceros), np.ones(n_unos), np.zeros(60), np.ones(60)])
    w = np.concatenate([np.ones(n_ceros), np.full(n_unos, 60.0), np.ones(120)])
    X = x.reshape(-1, 1)
    return X, y.astype(int), w


def test_el_ajuste_final_usa_los_pesos(datos_donde_el_peso_da_vuelta_el_signo):
    """
    Si `sample_weight` desaparece del fit, el coeficiente cambia de signo.
    """
    X, y, w = datos_donde_el_peso_da_vuelta_el_signo

    sin_pesos = LogisticRegression(C=1e6, max_iter=2000, random_state=tm.RANDOM_STATE)
    sin_pesos.fit(X, y)
    assert sin_pesos.coef_[0][0] < 0, "la fixture no discrimina: revisar los datos"

    modelo, _, _ = tm._ajustar(X, y, w, "test")
    assert modelo.coef_[0][0] > 0, (
        "el ajuste final ignoró los ponderadores: con pesos el efecto es "
        f"positivo y salió {modelo.coef_[0][0]:.3f}"
    )


def _score_ponderado_de_referencia(X, y, w, c):
    """
    Reimplementación explícita e independiente del score de `elegir_c` para un
    C dado: mismos folds, pesos en el fit Y en la pérdida, y promedio de folds
    ponderado por su masa.

    Es un "golden": si la implementación deja de ponderar en cualquiera de los
    tres lugares, los números se separan.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=tm.RANDOM_STATE)
    scores, masas = [], []
    for tr, te in cv.split(X, y):
        m = LogisticRegression(C=c, max_iter=2000, random_state=tm.RANDOM_STATE)
        m.fit(X[tr], y[tr], sample_weight=w[tr])
        p = m.predict_proba(X[te])[:, 1]
        scores.append(-log_loss(y[te], p, sample_weight=w[te], labels=[0, 1]))
        masas.append(w[te].sum())
    return float(np.average(scores, weights=masas)), scores, masas


def test_la_cv_pondera_en_el_fit_en_la_perdida_y_entre_folds(
        datos_donde_el_peso_da_vuelta_el_signo, monkeypatch):
    """
    El score que devuelve `elegir_c` tiene que coincidir con el ponderado en
    los tres lugares. Se fija la grilla a un solo C para poder comparar contra
    un número concreto.
    """
    X, y, w = datos_donde_el_peso_da_vuelta_el_signo
    monkeypatch.setattr(tm, "C_GRID", [1.0])

    esperado, scores, masas = _score_ponderado_de_referencia(X, y, w, 1.0)
    obtenido_c, obtenido = tm.elegir_c(X, y, w)

    assert obtenido_c == 1.0
    assert obtenido == pytest.approx(esperado, rel=1e-9), (
        "el score de la CV no coincide con el ponderado en fit + pérdida + "
        "promedio por masa"
    )

    # Y que el promedio por masa NO sea lo mismo que el promedio simple, para
    # que la comparación de arriba distinga de verdad las dos cosas.
    assert float(np.mean(scores)) != pytest.approx(esperado, rel=1e-6), (
        "en esta fixture las masas de los folds son casi iguales, así que el "
        "test no distingue el promedio ponderado del simple: cambiar los pesos"
    )


def test_la_perdida_pondera_el_fold_de_validacion(
        datos_donde_el_peso_da_vuelta_el_signo, monkeypatch):
    """
    Aísla la pérdida: se compara contra la misma CV pero con log_loss SIN
    pesos. Si la implementación dejara de ponderar la pérdida, devolvería este
    otro número.
    """
    X, y, w = datos_donde_el_peso_da_vuelta_el_signo
    monkeypatch.setattr(tm, "C_GRID", [1.0])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=tm.RANDOM_STATE)
    scores, masas = [], []
    for tr, te in cv.split(X, y):
        m = LogisticRegression(C=1.0, max_iter=2000, random_state=tm.RANDOM_STATE)
        m.fit(X[tr], y[tr], sample_weight=w[tr])
        p = m.predict_proba(X[te])[:, 1]
        scores.append(-log_loss(y[te], p, labels=[0, 1]))   # <- sin pesos
        masas.append(w[te].sum())
    sin_pesos_en_la_perdida = float(np.average(scores, weights=masas))

    _, obtenido = tm.elegir_c(X, y, w)
    assert obtenido != pytest.approx(sin_pesos_en_la_perdida, rel=1e-6), (
        "el score coincide con el de una pérdida SIN ponderar: la CV no está "
        "usando los pesos del fold de validación"
    )


def test_el_bootstrap_hereda_la_ponderacion(monkeypatch):
    """
    El bootstrap re-elige C con `elegir_c`, así que hereda las tres
    ponderaciones. Se comprueba que efectivamente la llama y no tenga una copia
    propia: la duplicación de esa lógica fue un hallazgo previo de Codex.
    """
    import pandas as pd
    rng = np.random.default_rng(3)
    n = 200
    X = rng.integers(0, 2, size=(n, 3)).astype(float)
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ [1.0, -0.8, 0.5])))).astype(int)
    w = rng.uniform(0.5, 1.5, size=n)
    d = pd.DataFrame({"estrato": rng.choice(list("abc"), size=n)})

    llamadas = []
    original = tm.elegir_c
    monkeypatch.setattr(tm, "elegir_c",
                        lambda *a, **k: (llamadas.append(1), original(*a, **k))[1])
    tm.bootstrap_coeficientes(d, X, y, w, n_replicas=3)
    assert len(llamadas) == 3, (
        f"el bootstrap llamó a elegir_c {len(llamadas)} veces sobre 3 réplicas: "
        "o tiene una copia propia de la CV, o dejó de re-elegir C"
    )

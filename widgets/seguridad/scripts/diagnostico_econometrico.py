"""
Diagnóstico econométrico de los cuatro modelos.

Responde tres preguntas que el pipeline de entrenamiento no responde: si el
modelo discrimina y acierta **fuera de muestra**, si las probabilidades están
**calibradas**, y si alguna otra especificación anda mejor.

QUÉ SE CORRIGIÓ EL 7/9/2026, DESPUÉS DE UNA AUDITORÍA DE CODEX
La primera versión de este script tenía tres defectos de método, y dos me
hicieron publicar conclusiones falsas:

1. FUGA DEL HIPERPARÁMETRO. Cargaba el `C` guardado en cada JSON —elegido por CV
   sobre TODA la muestra— y lo fijaba en cada fold, así que el hiperparámetro
   había visto los folds de validación. Ahora `C` se elige DENTRO de cada fold
   externo (CV anidada). El efecto es chico, unos +0,002 de AUC, pero el número
   que se publica tiene que salir de un método correcto.

2. EL CONTRASTE DE CALIBRACIÓN ERA EL EQUIVOCADO, DOS VECES. Primero reporté el
   peor desvío por decil como hallazgo, sin nula. Después construí la nula, vi
   que 11,7 pp caía adentro y concluí que las cuatro estaban calibradas — y eso
   también estaba mal: un MÁXIMO sobre diez bins tiene poca potencia y es ciego a
   que varios bins se desvíen de forma coordinada. Encima los bins salían de
   cuantiles NO ponderados, con masa entre 157 y 548: no eran décimos comparables
   de la población.
   Ahora: bins de igual MASA PONDERADA y Hosmer-Lemeshow ponderado con varianza
   w², calibrado por Monte Carlo. Con eso, **mano dura sí está descalibrada**.

3. NO ERA REPRODUCIBLE. Las simulaciones y la pendiente estaban narradas en el
   README y no implementadas acá. Ahora corren.

Y una corrección de criterio, la más de fondo: **el AUC no es la métrica de este
producto**. Mide ordenamiento individual, y el widget publica una TASA POR
PERFIL. Dos modelos con el mismo AUC pueden imprimir porcentajes muy distintos.
Se reporta log-loss y Brier primero, y el AUC como diagnóstico de cuánta
heterogeneidad hay entre perfiles.

Todo ponderado por `w_norm`. Corre fuera de los tests porque necesita sklearn y
la base del cliente, que no está en el repo.

Uso:
    python widgets/seguridad/scripts/diagnostico_econometrico.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import warnings

import numpy as np
import numpy.linalg as la
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from widgets.seguridad import config, train_model as tm

warnings.filterwarnings("ignore")

FOLDS = 5
BINS = 10
REPLICAS_MC = 1000
VIF_ALTO = 5.0


def oof_anidado(d, cols, semilla=42):
    """
    Predicciones out-of-fold con `C` elegido DENTRO de cada fold externo.

    Es lo que separa una estimación honesta de una optimista: si `C` se elige
    mirando toda la muestra, el fold que evalúa el modelo ya influyó en él.
    """
    X = d[cols].values.astype(float)
    y = d["a_favor"].values.astype(int)
    w = d[config.PONDERADOR].values
    oof = np.zeros(len(y))

    externo = StratifiedKFold(FOLDS, shuffle=True, random_state=semilla)
    for tr, te in externo.split(X, y):
        mejor_c, mejor_score = None, -np.inf
        interno = StratifiedKFold(FOLDS, shuffle=True, random_state=semilla)
        for C in tm.C_GRID:
            scores, masas = [], []
            for i2, t2 in interno.split(X[tr], y[tr]):
                m = LogisticRegression(C=C, max_iter=3000, random_state=tm.RANDOM_STATE)
                m.fit(X[tr][i2], y[tr][i2], sample_weight=w[tr][i2])
                scores.append(-log_loss(y[tr][t2], m.predict_proba(X[tr][t2])[:, 1],
                                        sample_weight=w[tr][t2], labels=[0, 1]))
                masas.append(w[tr][t2].sum())
            # Promedio PONDERADO POR LA MASA de cada fold, igual que
            # train_model.elegir_c(). Acá se promediaban por igual, así que el
            # diagnóstico podía elegir un C distinto del que usa producción y
            # medir otro modelo. Lo marcó Codex: con el criterio de producción,
            # cadena perpetua da AUC 0,655 y no 0,665.
            s = float(np.average(scores, weights=masas))
            if s > mejor_score:
                mejor_c, mejor_score = C, s
        m = LogisticRegression(C=mejor_c, max_iter=3000, random_state=tm.RANDOM_STATE)
        m.fit(X[tr], y[tr], sample_weight=w[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return y, oof, w


def bins_de_igual_masa(p, w, k=BINS):
    """
    Bins con la misma masa PONDERADA, no la misma cantidad de casos.

    Con cuantiles sin ponderar, los bins tenían entre 157 y 548 de masa en mano
    dura: comparar un desvío ponderado sobre bins definidos sin ponderar mezcla
    dos poblaciones distintas.
    """
    orden = np.argsort(p)
    acum = np.cumsum(w[orden]) / w.sum()
    b = np.zeros(len(p), dtype=int)
    b[orden] = np.minimum((acum * k).astype(int), k - 1)
    return b


def hosmer_lemeshow(y, p, w, b):
    """
    HL ponderado con varianza w², no el ingenuo que trata los pesos como
    observaciones repetidas — con un deff de 4,7 eso exageraría muchísimo.

    A diferencia del máximo por decil, suma los desvíos de TODOS los bins, así
    que ve descalibración coordinada aunque ningún bin sea extremo por separado.
    Esa ceguera es la que me hizo dar por calibrado un modelo que no lo está.
    """
    t = 0.0
    for k in sorted(set(b)):
        m = b == k
        obs = np.sum(w[m] * y[m])
        esp = np.sum(w[m] * p[m])
        var = np.sum(w[m] ** 2 * p[m] * (1 - p[m]))
        if var > 0:
            t += (obs - esp) ** 2 / var
    return t


def _p_valor_mc(y, p, w, b, rng):
    """p por Monte Carlo: la nula del HL generada por el propio modelo."""
    obs = hosmer_lemeshow(y, p, w, b)
    nulos = np.array([hosmer_lemeshow((rng.random(len(p)) < p).astype(int), p, w, b)
                      for _ in range(REPLICAS_MC)])
    return obs, float((nulos >= obs).mean())


def _metricas(y, p, w):
    return {
        "auc": roc_auc_score(y, p, sample_weight=w),
        "brier": float(np.average((p - y) ** 2, weights=w)),
        "logloss": log_loss(y, p, sample_weight=w, labels=[0, 1]),
    }


def main():
    rng = np.random.default_rng(tm.RANDOM_STATE)
    df0 = pd.read_csv(config.DATA_FILE, encoding="utf-8-sig")

    print("=" * 74)
    print("DESEMPEÑO FUERA DE MUESTRA (CV anidada: C se elige dentro de cada fold)")
    print("=" * 74)
    print("log-loss y Brier son el criterio; el AUC es diagnóstico de heterogeneidad,")
    print("no la métrica del producto — el widget publica una tasa, no un ranking.\n")
    print(f"{'pregunta':22s} {'logloss':>9s} {'Brier':>9s} {'AUC':>7s}")
    guardado = {}
    for slug in config.SLUGS:
        if not config.ruta_modelo(slug).exists():
            continue
        df = tm.preparar(df0, config.PREGUNTAS[slug])
        d = df[df["a_favor"].notna()]
        y, p, w = oof_anidado(d, list(config.PREDICTORES))
        guardado[slug] = (y, p, w, d)
        m = _metricas(y, p, w)
        print(f"{slug:22s} {m['logloss']:9.4f} {m['brier']:9.5f} {m['auc']:7.3f}")

    print("\n" + "=" * 74)
    print("CALIBRACIÓN — Hosmer-Lemeshow ponderado, bins de igual masa, p por MC")
    print("=" * 74)
    for slug, (y, p, w, _) in guardado.items():
        b = bins_de_igual_masa(p, w)
        hl, pv = _p_valor_mc(y, p, w, b, rng)
        sesgo = (np.average(p, weights=w) - np.average(y, weights=w)) * 100
        clip = np.clip(p, 1e-6, 1 - 1e-6)
        lp = np.log(clip / (1 - clip))
        pend = LogisticRegression(C=1e6, max_iter=3000).fit(
            lp.reshape(-1, 1), y, sample_weight=w).coef_[0][0]
        veredicto = "DESCALIBRADO" if pv < 0.05 else "sin descalibración detectable"
        print(f"  {slug:22s} HL={hl:6.2f} p={pv:.3f}  sesgo {sesgo:+5.2f}pp  "
              f"pendiente {pend:5.2f}  {veredicto}")
    print("\n  El sesgo agregado y la pendiente pueden verse bien y el modelo estar")
    print("  descalibrado igual: si los desvíos cambian de signo, se cancelan.")

    print("\n" + "=" * 74)
    print("COLINEALIDAD, CELDAS CHICAS Y EFECTO DE DISEÑO")
    print("=" * 74)
    _, _, _, d = next(iter(guardado.values()))
    Xc = np.column_stack([np.ones(len(d)),
                          d[config.PREDICTORES].astype(float).values])
    vifs = []
    for i, nombre in enumerate(config.PREDICTORES, start=1):
        otros = np.delete(Xc, i, axis=1)
        yv = Xc[:, i]
        beta, *_ = la.lstsq(otros, yv, rcond=None)
        r2 = 1 - ((yv - otros @ beta) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
        vifs.append((nombre, 1 / max(1e-12, 1 - r2)))
    for nombre, v in sorted(vifs, key=lambda t: -t[1])[:3]:
        print(f"  {nombre:24s} VIF={v:6.2f}" + ("   ALTO" if v > VIF_ALTO else ""))
    for nombre in config.PREDICTORES:
        n1 = int(d[nombre].sum())
        if n1 < 50:
            tasa = d.loc[d[nombre] == 1, "a_favor"].mean() * 100
            aviso = "   SEPARACIÓN" if tasa in (0.0, 100.0) else ""
            print(f"  celda chica: {nombre:22s} n={n1:4d}  a favor {tasa:5.1f}%{aviso}")
    w = d[config.PONDERADOR].values
    kish = w.sum() ** 2 / (w ** 2).sum()
    print(f"  n nominal {len(w)}   n efectivo (Kish) {kish:.0f}   deff {len(w) / kish:.2f}")


def buscar_especificacion(semillas=(1, 2, 3, 4, 5)):
    """
    ¿Alguna otra especificación anda mejor? Con CV anidada y varias particiones.

    Se repite sobre varias semillas porque las diferencias son de milésimas: con
    una sola partición, el ganador lo elige el azar. Se reporta en cuántas gana,
    que dice más que la media.

    Lo que ya se descartó en el barrido del 7/9/2026, sin efecto en ninguna de
    las cuatro: mover `C`, interacción educación x ideología, tamaño del hogar,
    situación laboral, splines y cuadrática en edad, L1, elastic net, Firth, y
    modelar el Likert completo —que además PIERDE contra el binario fuera de
    muestra, algo que la validación ordinal no había contestado porque ajusta
    dentro de muestra y sin pesos—. Abrir la ideología a escala lineal sube
    cadena perpetua y hunde pena de muerte: no es un reemplazo.

    CAVEAT que hay que decir: comparar muchas especificaciones sobre los mismos
    folds y quedarse con la mejor es, en sí, una forma de sobreajuste. Estos
    números sirven para DESCARTAR palancas, no para demostrar un techo.
    """
    df0 = pd.read_csv(config.DATA_FILE, encoding="utf-8-sig")
    print("BALOTAJE — la única palanca que quedó viva, medida bien")
    print("(negativo en logloss y Brier = mejora)\n")
    print(f"{'pregunta':22s} {'ΔAUC':>8s} {'ΔBrier':>9s} {'Δlogloss':>10s}  gana")
    for slug in config.SLUGS:
        if not config.ruta_modelo(slug).exists():
            continue
        df = tm.preparar(df0, config.PREGUNTAS[slug]).copy()
        b = df["Voto balotaje"]
        df["bal_orsi"] = (b == "Orsi").astype(int)
        df["bal_delgado"] = (b == "Delgado").astype(int)
        d = df[df["a_favor"].notna()]
        base = list(config.PREDICTORES)
        con = base + ["bal_orsi", "bal_delgado"]
        da, db, dl, gana = [], [], [], 0
        for s in semillas:
            y, p1, w = oof_anidado(d, base, s)
            _, p2, _ = oof_anidado(d, con, s)
            m1, m2 = _metricas(y, p1, w), _metricas(y, p2, w)
            da.append(m2["auc"] - m1["auc"])
            db.append(m2["brier"] - m1["brier"])
            dl.append(m2["logloss"] - m1["logloss"])
            gana += m2["logloss"] < m1["logloss"]
        print(f"{slug:22s} {np.mean(da):+8.4f} {np.mean(db):+9.5f} "
              f"{np.mean(dl):+10.5f}  {gana}/{len(semillas)}")


def probar_recalibracion(rng=None):
    """
    ¿Arregla la descalibración de mano dura una recalibración? Medido bien.

    Platt (logística sobre el logit) e isotónica, las dos ajustadas EN UN SEGUNDO
    NIVEL DE FOLDS sobre las predicciones out-of-fold. Ajustarlas sobre las
    mismas predicciones que después evalúan daría una mejora inventada: una
    isotónica con suficientes nodos calza cualquier cosa dentro de muestra.
    """
    rng = rng or np.random.default_rng(11)
    df0 = pd.read_csv(config.DATA_FILE, encoding="utf-8-sig")

    def pval(y, p, w):
        b = bins_de_igual_masa(p, w)
        obs = hosmer_lemeshow(y, p, w, b)
        nul = np.array([hosmer_lemeshow((rng.random(len(p)) < p).astype(int), p, w, b)
                        for _ in range(600)])
        return obs, float((nul >= obs).mean())

    print(f"{'pregunta':22s} {'variante':>10s} {'HL':>7s} {'p':>6s} "
          f"{'logloss':>9s} {'Brier':>9s}")
    for slug in config.SLUGS:
        if not config.ruta_modelo(slug).exists():
            continue
        df = tm.preparar(df0, config.PREGUNTAS[slug])
        d = df[df["a_favor"].notna()]
        y, p, w = oof_anidado(d, list(config.PREDICTORES))
        variantes = {"cruda": p}
        for nombre in ("platt", "isotonica"):
            q = np.zeros(len(y))
            for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=99).split(
                    p.reshape(-1, 1), y):
                if nombre == "platt":
                    clip = np.clip(p, 1e-6, 1 - 1e-6)
                    lp = np.log(clip / (1 - clip))
                    m = LogisticRegression(C=1e6, max_iter=3000).fit(
                        lp[tr].reshape(-1, 1), y[tr], sample_weight=w[tr])
                    q[te] = m.predict_proba(lp[te].reshape(-1, 1))[:, 1]
                else:
                    m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
                        p[tr], y[tr], sample_weight=w[tr])
                    q[te] = np.clip(m.predict(p[te]), 1e-6, 1 - 1e-6)
            variantes[nombre] = q
        for nombre, q in variantes.items():
            hl, pv = pval(y, q, w)
            m = _metricas(y, q, w)
            print(f"{slug:22s} {nombre:>10s} {hl:7.2f} {pv:6.3f} "
                  f"{m['logloss']:9.4f} {m['brier']:9.5f}")


if __name__ == "__main__":
    main()
    print()
    buscar_especificacion()
    print()
    print("=" * 74)
    print("¿ARREGLA UNA RECALIBRACIÓN? (ajustada fuera de muestra)")
    print("=" * 74)
    probar_recalibracion()

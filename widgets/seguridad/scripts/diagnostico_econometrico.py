"""
Diagnóstico econométrico de los cuatro modelos.

Lo que el pipeline YA hacía —CV para elegir `C`, bootstrap estratificado,
validación ordinal— responde "¿el modelo está bien estimado?" y "¿la estructura
aguanta si lo estimo de otra forma?". No responde tres preguntas distintas, que
son las que cubre este script:

  1. ¿DISCRIMINA FUERA DE MUESTRA? El AUC dentro de muestra siempre halaga.
     Acá se calcula también por validación cruzada de 5 folds, y la caída entre
     los dos es la medida del sobreajuste.
  2. ¿ESTÁN CALIBRADAS LAS PROBABILIDADES? El widget no publica un ranking:
     publica un NÚMERO que dice "el 65% de la gente con este perfil". Si el
     modelo discrimina bien pero está descalibrado, el orden de los perfiles es
     correcto y el número que se imprime, no. Se compara predicho contra
     observado por deciles de riesgo, fuera de muestra y ponderado.
  3. ¿HAY COLINEALIDAD O CELDAS DEGENERADAS? VIF sobre los 17 predictores y
     tamaño de las categorías chicas, para detectar separación.

Más el efecto de diseño (deff) del ponderador, que dice cuánto de la muestra
nominal sobrevive a la ponderación.

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

import json
import warnings

import numpy as np
import numpy.linalg as la
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from widgets.seguridad import config, train_model as tm

warnings.filterwarnings("ignore")

FOLDS = 5
DECILES = 10
# Por encima de esto, un VIF indica que el coeficiente de esa variable está
# estimado sobre muy poca variación propia. 5 es el corte habitual.
VIF_ALTO = 5.0


def _fuera_de_muestra(X, y, w, C):
    """Predicciones out-of-fold: cada caso predicho por un modelo que no lo vio."""
    cv = StratifiedKFold(FOLDS, shuffle=True, random_state=tm.RANDOM_STATE)
    oof = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        m = LogisticRegression(C=C, max_iter=2000, random_state=tm.RANDOM_STATE)
        m.fit(X[tr], y[tr], sample_weight=w[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def main():
    df0 = pd.read_csv(config.DATA_FILE, encoding="utf-8-sig")

    print("=" * 72)
    print("DISCRIMINACIÓN — dentro y fuera de muestra")
    print("=" * 72)
    print(f"{'pregunta':22s} {'AUC in':>7s} {f'AUC {FOLDS}-fold':>11s} "
          f"{'caída':>7s} {'Brier':>7s}")
    guardado = {}
    for slug in config.SLUGS:
        ruta = config.ruta_modelo(slug)
        if not ruta.exists():
            continue
        df = tm.preparar(df0, config.PREGUNTAS[slug])
        d = df[df["a_favor"].notna()]
        X = d[config.PREDICTORES].values
        y = d["a_favor"].values.astype(int)
        w = d[config.PONDERADOR].values
        with open(ruta, encoding="utf-8") as f:
            C = json.load(f)["model_info"]["C"]

        m = LogisticRegression(C=C, max_iter=2000, random_state=tm.RANDOM_STATE)
        m.fit(X, y, sample_weight=w)
        auc_in = roc_auc_score(y, m.predict_proba(X)[:, 1], sample_weight=w)
        oof = _fuera_de_muestra(X, y, w, C)
        auc_oof = roc_auc_score(y, oof, sample_weight=w)
        brier = float(np.average((oof - y) ** 2, weights=w))
        guardado[slug] = (y, oof, w, d)
        print(f"{slug:22s} {auc_in:7.3f} {auc_oof:11.3f} "
              f"{auc_in - auc_oof:+7.3f} {brier:7.3f}")

    print("\n" + "=" * 72)
    print("CALIBRACIÓN — predicho contra observado, por decil, fuera de muestra")
    print("=" * 72)
    print("Es lo que decide si el NÚMERO que se publica es correcto, no sólo el")
    print("orden de los perfiles.\n")
    for slug, (y, oof, w, _) in guardado.items():
        q = pd.qcut(oof, DECILES, labels=False, duplicates="drop")
        filas = []
        for k in sorted(set(q)):
            msk = q == k
            filas.append((np.average(oof[msk], weights=w[msk]) * 100,
                          np.average(y[msk], weights=w[msk]) * 100))
        peor = max(abs(a - b) for a, b in filas)
        print(f"  {slug:22s} peor desvío {peor:5.1f} pp   "
              f"D1 {filas[0][0]:.0f}%/{filas[0][1]:.0f}%   "
              f"D10 {filas[-1][0]:.0f}%/{filas[-1][1]:.0f}%")

    print("\n" + "=" * 72)
    print("COLINEALIDAD Y CELDAS CHICAS")
    print("=" * 72)
    _, _, _, d = next(iter(guardado.values()))
    Xc = np.column_stack([np.ones(len(d)), d[config.PREDICTORES].astype(float).values])
    vifs = []
    for i, nombre in enumerate(config.PREDICTORES, start=1):
        otros = np.delete(Xc, i, axis=1)
        yv = Xc[:, i]
        beta, *_ = la.lstsq(otros, yv, rcond=None)
        r2 = 1 - ((yv - otros @ beta) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
        vifs.append((nombre, 1 / max(1e-12, 1 - r2)))
    for nombre, v in sorted(vifs, key=lambda t: -t[1])[:5]:
        print(f"  {nombre:24s} VIF={v:6.2f}" + ("   ALTO" if v > VIF_ALTO else ""))

    print()
    for nombre in config.PREDICTORES:
        n1 = int(d[nombre].sum())
        if n1 < 50:
            tasa = d.loc[d[nombre] == 1, "a_favor"].mean() * 100
            aviso = "   SEPARACIÓN" if tasa in (0.0, 100.0) else ""
            print(f"  celda chica: {nombre:24s} n={n1:4d}  a favor {tasa:5.1f}%{aviso}")

    print("\n" + "=" * 72)
    print("EFECTO DE DISEÑO DEL PONDERADOR")
    print("=" * 72)
    w = d[config.PONDERADOR].values
    kish = w.sum() ** 2 / (w ** 2).sum()
    print(f"  n nominal {len(w)}   n efectivo (Kish) {kish:.0f}   deff {len(w) / kish:.2f}")
    print(f"  ponderador: min {w.min():.3f}  max {w.max():.2f}  "
          f"CV {w.std() / w.mean():.2f}")
    print("  Los intervalos del widget salen de bootstrap estratificado, así que")
    print("  ya lo incorporan; este número dice cuánto pesa.")


if __name__ == "__main__":
    main()


# ============================================================
# ¿SE PUEDE MEJORAR? — búsqueda de especificación
# ============================================================
# Se corre aparte de main(): responde otra pregunta. main() dice cómo anda el
# modelo que se publica; esto dice si alguna otra especificación anda mejor,
# medido SIEMPRE fuera de muestra y ponderado. Dentro de muestra cualquier
# variable extra "mejora": por eso no se mira ese número.
#
# Resultado del barrido del 7/9/2026, en AUC fuera de muestra:
#
#   especificación          mano dura  cad.perp.  pena mte.  humillación
#   base (la publicada)         0,776     0,665      0,782       0,814
#   C=1 / C=10                  0,773     0,659      0,781       0,815
#   ideología lineal 0-10       0,775     0,682      0,747       0,815
#   + educación x ideología     0,777     0,666      0,784       0,813
#   + tamaño del hogar          0,782     0,665      0,779       0,815
#   + situación laboral         0,775     0,664      0,776       0,816
#   + VOTO DE BALOTAJE          0,795     0,681      0,784       0,811
#
# CONCLUSIÓN. El modelo está cerca del techo de lo que dan estas variables.
# El regularizador no es palanca, las interacciones tampoco, y el hogar y la
# situación laboral mueven menos que el ruido.
#
# Hay UNA palanca real y una falsa:
#
#   · REAL: reponer el voto de balotaje. +0,019 en mano dura y +0,016 en cadena
#     perpetua, sin costo en las otras dos. Está afuera por decisión editorial
#     de Tomer (31/8/2026: "poner identificación ideológica y sacar partidos
#     políticos"), no por un problema del modelo. Es el precio de esa decisión,
#     medido.
#
#   · FALSA: pasar la ideología a escala lineal 0-10. Sube cadena perpetua
#     (+0,017) pero HUNDE pena de muerte (-0,035): ahí la relación no es
#     monótona y los siete tramos capturan algo que una recta borra. Mejorar la
#     pregunta más débil rompiendo otra no es mejorar.

def buscar_especificacion():
    """Compara especificaciones alternativas por AUC fuera de muestra."""
    df0 = pd.read_csv(config.DATA_FILE, encoding="utf-8-sig")
    ide_dummies = [f"ideol_{n}" for n, _, _, _ in config.ESPEC_CRUDA["ideol_tramos"]
                   if n != config.ESPEC_CRUDA["ideol_referencia"]]

    def auc(d, cols, C):
        X = d[cols].values.astype(float)
        y = d["a_favor"].values.astype(int)
        w = d[config.PONDERADOR].values
        return roc_auc_score(y, _fuera_de_muestra(X, y, w, C), sample_weight=w)

    print(f"{'pregunta':20s} {'base':>6s} {'ideol lin':>10s} "
          f"{'+hogar':>8s} {'+balotaje':>10s}")
    for slug in config.SLUGS:
        ruta = config.ruta_modelo(slug)
        if not ruta.exists():
            continue
        df = tm.preparar(df0, config.PREGUNTAS[slug]).copy()
        df["ideol_lineal"] = (
            df["var_242 | Autoubicacion izquierda-derecha (0-10)"].fillna(5.0).astype(float))
        df["hogar_3_4"] = df["cant_personas"].between(3, 4).astype(int)
        df["hogar_5_plus"] = (df["cant_personas"] >= 5).astype(int)
        b = df["Voto balotaje"]
        df["bal_orsi"] = (b == "Orsi").astype(int)
        df["bal_delgado"] = (b == "Delgado").astype(int)
        d = df[df["a_favor"].notna()]

        base = list(config.PREDICTORES)
        sin_ide = [p for p in base if p not in ide_dummies]
        with open(ruta, encoding="utf-8") as f:
            C = json.load(f)["model_info"]["C"]

        r = [auc(d, base, C),
             auc(d, sin_ide + ["ideol_lineal"], C),
             auc(d, base + ["hogar_3_4", "hogar_5_plus"], C),
             auc(d, base + ["bal_orsi", "bal_delgado"], C)]
        marca = lambda v: f"{v:.3f}" + ("*" if v > r[0] + 0.005 else " ")
        print(f"{slug:20s} {r[0]:.3f}  " + " ".join(marca(v) for v in r[1:]))
    print("\n* = mejora de más de 0,005 sobre la base, fuera de muestra y ponderado.")

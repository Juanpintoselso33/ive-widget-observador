"""
Entrenamiento del modelo del widget de seguridad.

Replica el enfoque del widget IVE: regresión logística binaria con penalización
L2, ponderada por diseño muestral (w_norm), C elegido por validación cruzada.
Exporta model_coefficients.json, que es lo único que consume la app.

La pregunta modelada sale de config.PREGUNTA_ACTIVA. Cambiarla y volver a
correr este script alcanza para cambiar de pregunta: no hay nada específico de
"pena de muerte" en el código.

Uso:
    SEGURIDAD_DATA_FILE=/ruta/base.csv python widgets/seguridad/train_model.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
from datetime import date

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from widgets.seguridad.config import (
    DATA_FILE, MODEL_COEFFICIENTS_PATH, PREGUNTA, PREGUNTA_ACTIVA,
    LIKERT_MAP, LIKERT_FAVOR, LIKERT_CONTRA, LIKERT_NEUTRAL,
    PONDERADOR, PREDICTORES, REFERENCIAS, huella_contrato, ESPEC_CRUDA,
    EDAD_UI_TO_CODE, EDUC_UI_TO_CODE, IDEOLOGIA_UI_TO_CODE, VICTIMA_UI_TO_CODE,
    REGION_UI_TO_CODE,
)

RANDOM_STATE = 42
C_GRID = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

# Escala nivel_educativo (1-10) del proveedor, colapsada.
#
# El widget IVE advierte que este mapeo es "inferido, sin codebook". Para ESTA
# base eso no corresponde: el colapso coincide exactamente con la columna
# `nivel_educ` etiquetada que viene en la encuesta, así que las categorías están
# verificadas y no inferidas. El caveat estaba copiado del otro widget.
#
# Tres categorías, no cuatro. El widget IVE separa "Primaria o menos" de
# "Secundaria", pero en esta encuesta esa categoría tiene 28 casos de 2.672
# (1,0%) — y era la REFERENCIA, o sea que los tres coeficientes de educación,
# que son los más grandes del modelo, se estimaban contra 28 personas. Al
# colapsarla con Secundaria la referencia pasa a tener 641 casos.
#
# Se pierde el contraste más extremo (los de primaria declaraban 58,6% de
# apoyo, el valor más alto de toda la muestra), pero ese número no es
# publicable con esa base. La muestra sobre-representa fuerte a los más
# educados: 53% tiene terciaria completa.
# Vive en config.ESPEC_CRUDA para que entre en la huella del contrato: cuando
# estaba acá, cambiar el colapso educativo dejaba la misma huella y un JSON
# viejo cargaba sin protestar con las dummies significando otra cosa.
EDUC_COLAPSO = ESPEC_CRUDA["educ_colapso"]

# Columnas que la base tiene que traer sí o sí para poder entrenar.
COLUMNAS_REQUERIDAS = [
    "edad", "sexo", "nivel_educativo", "dpto_ech", "estrato", PONDERADOR,
    "var_241 | Victima de delito ultimos 12 meses",
    "var_242 | Autoubicacion izquierda-derecha (0-10)",
]


def cargar():
    if not DATA_FILE.exists():
        raise SystemExit(
            f"No se encontró la base en {DATA_FILE}.\n"
            "Pasá la ruta con la variable de entorno SEGURIDAD_DATA_FILE."
        )
    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
    print(f"Base: {DATA_FILE.name} — {len(df)} filas")
    print(f"Pregunta activa: {PREGUNTA_ACTIVA} → {PREGUNTA['columna']}")
    return df


def preparar(df):
    """Construye la variable dependiente y las dummies de los predictores."""
    col = PREGUNTA["columna"]
    if col not in df.columns:
        raise SystemExit(f"La columna '{col}' no está en la base.")

    df = df.copy()

    # Validación de dominios, antes de construir nada.
    #
    # La etiqueta Likert ya se valida abajo, pero los predictores no se
    # validaban: una edad fuera de rango caía en 18-29, un sexo desconocido en
    # "hombre", una educación inesperada en "primaria", un departamento raro en
    # "interior". Todas conversiones silenciosas a la categoría de referencia.
    # Con la base actual no pasa, pero el widget está pensado para re-entrenarse
    # con otras columnas y una base con otro formato produciría un modelo
    # plausible y equivocado, sin un solo error en pantalla.
    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        raise SystemExit(f"La base no tiene estas columnas: {faltantes}")

    problemas = []

    # Los NULOS se cuentan como problema, no sólo los valores raros: un nulo
    # cae silenciosamente en la categoría de referencia, que es exactamente la
    # conversión invisible que esta validación quiere evitar.
    def revisar_nulos(col, etiqueta):
        n = int(df[col].isna().sum())
        if n:
            problemas.append(f"'{etiqueta}' tiene {n} valor(es) nulo(s), "
                             "que caerían en la categoría de referencia")

    sexos = set(df["sexo"].dropna().unique()) - set(ESPEC_CRUDA["sexo_valores"].values())
    if sexos:
        problemas.append(f"valores de 'sexo' no esperados: {sorted(sexos)}")
    revisar_nulos("sexo", "sexo")

    educ = set(df["nivel_educativo"].dropna().unique()) - set(EDUC_COLAPSO)
    if educ:
        problemas.append(f"códigos de 'nivel_educativo' fuera de 1-10: {sorted(educ)}")
    revisar_nulos("nivel_educativo", "nivel_educativo")

    # dpto_ech no se validaba: cualquier código raro o nulo se volvía "Interior".
    dptos = set(df["dpto_ech"].dropna().unique()) - set(range(1, 20))
    if dptos:
        problemas.append(f"códigos de 'dpto_ech' fuera de 1-19: {sorted(dptos)}")
    revisar_nulos("dpto_ech", "dpto_ech")

    # La escala es discreta 0-10: un 3,5 pasaba como válido y caía en "Centro".
    ideol = df["var_242 | Autoubicacion izquierda-derecha (0-10)"].dropna()
    fuera = ideol[(ideol < 0) | (ideol > 10)]
    if len(fuera):
        problemas.append(f"autoubicación fuera de 0-10: {len(fuera)} casos")
    no_enteras = ideol[ideol != ideol.round()]
    if len(no_enteras):
        problemas.append(
            f"{len(no_enteras)} autoubicación(es) no entera(s) "
            f"(ejemplos: {sorted(no_enteras.unique())[:3]}): la escala es discreta"
        )

    # Dominio CERRADO, no "cualquier texto que contenga violencia": algo como
    # "Sí, violencia desconocida" pasaba la validación y después se codificaba
    # como "No".
    _ve = ESPEC_CRUDA["victima_etiquetas"]
    vic_validos = {e for grupo in _ve.values() for e in grupo}
    vic = {str(v).strip().lower()
           for v in df["var_241 | Victima de delito ultimos 12 meses"].dropna().unique()}
    vic_raros = vic - vic_validos
    if vic_raros:
        problemas.append(f"respuestas de victimización no esperadas: {sorted(vic_raros)}")

    # Una edad fuera de rango (la encuesta trae años de nacimiento cargados como
    # edad, tipo 1985) terminaba en NaN y de ahí caía en el tramo de referencia
    # 18-29: un dato inválido convertido en la categoría base. Ahora aborta.
    edades = df["edad"].dropna()
    fuera_edad = edades[(edades < 18) | (edades > 110)]
    if len(fuera_edad):
        problemas.append(
            f"{len(fuera_edad)} edad(es) fuera de 18-110 "
            f"(ejemplos: {sorted(fuera_edad.unique())[:3]}): caerían en el tramo 18-29"
        )
    revisar_nulos("edad", "edad")

    w = df[PONDERADOR]
    if w.isna().any() or (w <= 0).any() or not np.isfinite(w.dropna()).all():
        problemas.append("hay ponderadores nulos, no positivos o no finitos")

    if problemas:
        raise SystemExit(
            "La base no pasa la validación de dominios:\n  - "
            + "\n  - ".join(problemas)
            + "\nRevisá la base o actualizá los mapeos en config.py antes de entrenar."
        )

    # Una etiqueta que no esté en LIKERT_MAP se mapearía a NaN y terminaría
    # contada como "no toma posición", produciendo coeficientes y tasas
    # plausibles pero equivocados, sin ningún error visible. Como la pregunta
    # es parametrizable y se re-entrena sobre otras columnas, un cambio mínimo
    # de formato o de etiqueta alcanzaría para eso: se corta acá.
    desconocidas = set(df[col].dropna().unique()) - set(LIKERT_MAP)
    if desconocidas:
        raise SystemExit(
            f"La columna '{col}' trae etiquetas que no están en LIKERT_MAP: "
            f"{sorted(desconocidas)}.\n"
            "Actualizá LIKERT_MAP en config.py o revisá la base antes de entrenar."
        )

    df["likert"] = df[col].map(LIKERT_MAP)

    # Dependiente principal: a favor vs en contra, excluyendo neutrales.
    df["a_favor"] = np.nan
    df.loc[df["likert"].isin(LIKERT_FAVOR), "a_favor"] = 1
    df.loc[df["likert"].isin(LIKERT_CONTRA), "a_favor"] = 0

    # Dependiente secundaria: no toma posición. Junta dos cosas distintas
    # —neutral explícito y falta de respuesta— y por eso se contabilizan por
    # separado más abajo: son 671 y 34, y publicar sólo los primeros haría que
    # los totales no cierren contra el N de la encuesta.
    df["neutral_explicito"] = (df["likert"] == LIKERT_NEUTRAL).astype(int)
    df["sin_respuesta"] = df["likert"].isna().astype(int)
    df["es_neutral"] = ((df["likert"] == LIKERT_NEUTRAL) | df["likert"].isna()).astype(int)

    # --- Edad: outliers fuera antes de tramificar (mismo criterio que el IVE)
    df.loc[(df["edad"] < 18) | (df["edad"] > 110), "edad"] = np.nan
    tramo = pd.cut(df["edad"], ESPEC_CRUDA["edad_cortes"], labels=[1, 2, 3, 4])
    df["tramo_edad"] = tramo.astype("float")
    df["edad_30_44"] = (df["tramo_edad"] == 2).astype(int)
    df["edad_45_59"] = (df["tramo_edad"] == 3).astype(int)
    df["edad_60_plus"] = (df["tramo_edad"] == 4).astype(int)

    df["es_mujer"] = (df["sexo"] == ESPEC_CRUDA["sexo_valores"]["mujer"]).astype(int)

    educ = df["nivel_educativo"].map(EDUC_COLAPSO)
    df["educ_ter_incomp"] = (educ == 2).astype(int)
    df["educ_ter_comp"] = (educ == 3).astype(int)

    # --- Ideología: los seis tramos de ESPEC_CRUDA["ideol_tramos"], con el
    # marcado como referencia sin dummy. Los tramos se leen de la
    # especificación y no se escriben a mano acá: si se escribieran a mano, la
    # huella del contrato cambiaría al editar la espec pero las dummies no,
    # que es exactamente el agujero que Codex encontró con los códigos de
    # balotaje.
    ideol = df["var_242 | Autoubicacion izquierda-derecha (0-10)"]
    referencia = ESPEC_CRUDA["ideol_referencia"]
    cubierto = ideol.isna()
    for nombre, desde, hasta in ESPEC_CRUDA["ideol_tramos"]:
        en_tramo = ideol.between(desde, hasta)
        cubierto |= en_tramo
        if nombre != referencia:
            df[f"ideol_{nombre}"] = en_tramo.astype(int)
    # Un valor de la escala que no cae en ningún tramo quedaría con TODAS las
    # dummies en cero, o sea silenciosamente dentro de la referencia. Con la
    # base actual no pasa, pero el widget está pensado para re-entrenarse.
    if (~cubierto).any():
        sueltos = sorted(ideol[~cubierto].dropna().unique())
        raise SystemExit(
            f"estos valores de autoubicación no caen en ningún tramo de "
            f"ideol_tramos y caerían en la referencia: {sueltos}"
        )
    df["ideol_no_ubica"] = ideol.isna().astype(int)

    # --- Víctima de delito en los últimos 12 meses. La base distingue con y
    # sin violencia; se mantiene la distinción porque pesan muy distinto.
    #
    # Los sin dato llevan dummy propia (igual que ideol_no_ubica). Antes caían
    # en la referencia con las dos dummies en cero, o sea que se mezclaban con
    # quienes contestaron "No": eso contaminaba la categoría de referencia del
    # modelo y también la tasa del grupo "No fue víctima" que se publica en el
    # bloque comparativo.
    vic = df["var_241 | Victima de delito ultimos 12 meses"]
    vic_norm = vic.astype(str).str.strip().str.lower()
    _ve = ESPEC_CRUDA["victima_etiquetas"]
    df["victima_sin_dato"] = vic.isna().astype(int)
    df["victima_sin_violencia"] = vic_norm.isin(_ve["sin_violencia"]).astype(int)
    df["victima_con_violencia"] = vic_norm.isin(_ve["con_violencia"]).astype(int)
    # "No" real: contestó y no fue víctima. Es lo que alimenta stats_by_group.
    df["victima_no_real"] = vic_norm.isin(_ve["no"]).astype(int)

    n_vic_sd = int(df["victima_sin_dato"].sum())
    if n_vic_sd:
        print(f"  víctima sin dato: {n_vic_sd} casos ({n_vic_sd / len(df) * 100:.1f}%) "
              f"— con dummy propia, no mezclados con los 'No'")

    df["es_montevideo"] = (df["dpto_ech"] == ESPEC_CRUDA["dpto_montevideo"]).astype(int)

    return df


def elegir_c(X, y, w):
    """
    Elige C por validación cruzada ponderada. ÚNICA implementación.

    La usan el ajuste principal y cada réplica bootstrap. Antes estaba escrita
    dos veces y el requisito de que fueran la misma CV vivía en un comentario;
    con eso, cualquier cambio en una de las dos copias desalineaba en silencio
    los intervalos respecto del punto estimado, y ningún test lo notaba.

    5 folds estratificados, la grilla de C_GRID, pesos muestrales tanto en el
    ajuste como en la pérdida, y promedio de los folds ponderado por su masa de
    pesos: cada log_loss ya está ponderado adentro, pero promediarlos por igual
    le daría el mismo peso a folds con distinta masa muestral.

    Devuelve (mejor_c, mejor_score), o (None, None) si algún fold queda sin las
    dos clases en ENTRENAMIENTO. En ese caso la CV no es comparable entre
    valores de C y el llamador decide qué hacer — la réplica se descarta, no se
    saltea el fold. Saltearlo es peor de lo que parece: con una única
    observación de la clase minoritaria, el único fold que la tiene en
    validación es justamente el que se saltearía, así que C terminaría elegido
    evaluando sólo folds de validación sin minoría.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    particion = list(cv.split(X, y))
    if any(len(np.unique(y[idx_train])) < 2 for idx_train, _ in particion):
        return None, None

    mejor_c, mejor_score = None, -np.inf
    for c in C_GRID:
        scores, masas = [], []
        for idx_train, idx_test in particion:
            m = LogisticRegression(C=c, max_iter=2000, random_state=RANDOM_STATE)
            m.fit(X[idx_train], y[idx_train], sample_weight=w[idx_train])
            p = m.predict_proba(X[idx_test])[:, 1]
            scores.append(-log_loss(
                y[idx_test], p, sample_weight=w[idx_test], labels=[0, 1],
            ))
            masas.append(w[idx_test].sum())
        score = float(np.average(scores, weights=masas))
        if score > mejor_score:
            mejor_c, mejor_score = c, score

    return mejor_c, mejor_score
def _ajustar(X, y, w, etiqueta):
    """
    Elige C por CV ponderada y devuelve el modelo ajustado.

    Los folds se recorren a mano en vez de usar cross_val_score porque el
    scorer NO recibe los pesos: `sample_weight` llega al fit del estimador,
    pero con el metadata routing deshabilitado (que es el default) la métrica
    evalúa el fold de validación sin ponderar. Entrenaba ponderado y evaluaba
    sin ponderar, lo cual puede elegir otra regularización sin avisar.

    Hacerlo explícito además saca la dependencia de `params=`, que existe
    recién desde scikit-learn 1.4, y deja el script compatible con el piso
    declarado en requirements.txt.
    """
    # La CV vive en elegir_c() y no acá: ver el docstring de esa función.
    mejor_c, mejor_score = elegir_c(X, y, w)
    if mejor_c is None:
        raise SystemExit(
            f"[{etiqueta}] la validación cruzada no pudo correr: algún fold "
            "quedó sin las dos clases en entrenamiento."
        )

    modelo = LogisticRegression(C=mejor_c, max_iter=2000, random_state=RANDOM_STATE)
    modelo.fit(X, y, sample_weight=w)
    print(f"  [{etiqueta}] C={mejor_c}  CV ponderada neg-log-loss={mejor_score:.4f}  n={len(y)}")
    return modelo, mejor_c, mejor_score


def bootstrap_coeficientes(d, X, y, w, n_replicas=1000):
    """
    Coeficientes de B réplicas bootstrap, para poder mostrar intervalos.

    El remuestreo es ESTRATIFICADO: con reemplazo dentro de cada uno de los 28
    estratos de la encuesta (`estrato`: departamentos del interior y tramos de
    ranking en Montevideo), conservando el tamaño de cada uno. Es lo apropiado
    porque respeta el diseño disponible — no porque ensanche: medido sobre esta
    base, el bootstrap sin estratificar da intervalos incluso un poco más
    anchos. Sigue siendo una aproximación incompleta: la base no trae
    información de conglomerados.

    C SE VUELVE A ELEGIR EN CADA RÉPLICA, con la misma CV ponderada que el
    modelo principal. Fijarlo en el C del ajuste original trata la selección del
    hiperparámetro como si fuera un dato y achica los intervalos: reestimando,
    la amplitud mediana pasa de 25,8 a 29,1 puntos y unos 50 perfiles cambian
    la conclusión sobre si el intervalo cruza el 50%. Es lo que hace que esto
    tarde unos minutos, y vale la pena.

    Se guardan los coeficientes y no los intervalos por perfil: así la app puede
    calcular el de cualquier combinación sin arrastrar 1.296 pares de números, y
    `model.py` sigue sin depender de sklearn.

    Devuelve (coeficientes, meta). `meta` va al JSON para que el artefacto sea
    auditable sin re-correr esto: cuántas réplicas se pidieron, cuántas
    sobrevivieron, con qué semilla y cómo se repartió el C elegido. Si alguna
    vez se vuelve a fijar C, la distribución colapsa a un solo valor y se ve.
    """
    estratos = d["estrato"].values
    indices_por_estrato = [np.where(estratos == e)[0] for e in np.unique(estratos)]
    rng = np.random.default_rng(RANDOM_STATE)
    coeficientes = []
    c_elegidos = []

    for i in range(n_replicas):
        idx = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in indices_por_estrato
        ])
        Xb, yb, wb = X[idx], y[idx], w[idx]
        # Una réplica puede quedar sin variación en la dependiente; se descarta.
        if len(np.unique(yb)) < 2:
            continue

        # La MISMA CV que el ajuste principal, porque es literalmente la misma
        # función. Si alguna réplica tiene un fold degenerado se descarta
        # entera: ver el docstring de elegir_c().
        mejor_c, _ = elegir_c(Xb, yb, wb)
        if mejor_c is None:
            continue

        m = LogisticRegression(C=mejor_c, max_iter=2000, random_state=RANDOM_STATE)
        m.fit(Xb, yb, sample_weight=wb)
        coeficientes.append([float(m.intercept_[0])] + [float(v) for v in m.coef_[0]])
        c_elegidos.append(float(mejor_c))

        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{n_replicas} réplicas")

    meta = {
        "solicitadas": int(n_replicas),
        "utiles": len(coeficientes),
        "semilla": int(RANDOM_STATE),
        "c_por_replica": {str(c): c_elegidos.count(c) for c in sorted(set(c_elegidos))},
    }
    return coeficientes, meta


def _mcfadden(modelo, X, y, w):
    """Pseudo-R² con log-likelihood nulo calculado sobre la media ponderada."""
    p = modelo.predict_proba(X)[:, 1]
    ll = np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p)))
    p0 = np.average(y, weights=w)
    ll0 = np.sum(w * (y * np.log(p0) + (1 - y) * np.log(1 - p0)))
    return 1 - ll / ll0


def main():
    df = preparar(cargar())
    w_col = PONDERADOR

    # --- Modelo principal ---------------------------------------------------
    d = df[df["a_favor"].notna()].copy()
    X = d[PREDICTORES].values
    y = d["a_favor"].values.astype(int)
    w = d[w_col].values

    n_neutral = int(df["neutral_explicito"].sum())
    n_sin_resp = int(df["sin_respuesta"].sum())
    n_excluidos = n_neutral + n_sin_resp
    prop_cruda = y.mean() * 100
    prop_pond = np.average(y, weights=w) * 100

    # Los totales tienen que cerrar contra el N de la encuesta: si no, la
    # sección de metodología del widget publica cifras que no reconcilian.
    assert len(d) + n_excluidos == len(df), (
        f"los conteos no cierran: {len(d)} + {n_excluidos} != {len(df)}"
    )

    print(f"\nCon posición definida: {len(d)}")
    print(f"  excluidos: {n_excluidos}  ({n_neutral} neutrales + {n_sin_resp} sin respuesta)")
    print(f"A favor — crudo: {prop_cruda:.1f}%  ponderado: {prop_pond:.1f}%")

    print("\nAjustando modelos:")
    modelo, c_ppal, cv_ppal = _ajustar(X, y, w, "principal")
    r2 = _mcfadden(modelo, X, y, w)
    print(f"  [principal] McFadden pseudo-R² = {r2:.4f}")

    print("\nBootstrap estratificado para los intervalos (re-elige C en cada")
    print("réplica, así que tarda unos minutos)...")
    N_REPLICAS = 1000
    boot, boot_meta = bootstrap_coeficientes(d, X, y, w, N_REPLICAS)
    print(f"  {boot_meta['utiles']} réplicas útiles sobre {N_REPLICAS}")
    print(f"  C elegido por réplica: {boot_meta['c_por_replica']}")

    coeficientes = {"intercept": float(modelo.intercept_[0])}
    for nombre, valor in zip(PREDICTORES, modelo.coef_[0]):
        coeficientes[nombre] = float(valor)
    odds = {k: float(np.exp(v)) for k, v in coeficientes.items() if k != "intercept"}

    # --- Modelo secundario: no toma posición --------------------------------
    dn = df[df[w_col].notna()].copy()
    Xn = dn[PREDICTORES].values
    yn = dn["es_neutral"].values.astype(int)
    wn = dn[w_col].values
    modelo_n, c_neu, cv_neu = _ajustar(Xn, yn, wn, "neutralidad")
    r2_n = _mcfadden(modelo_n, Xn, yn, wn)
    prop_neutral = np.average(yn, weights=wn) * 100
    print(f"  [neutralidad] McFadden pseudo-R² = {r2_n:.4f} — tasa ponderada {prop_neutral:.1f}%")

    coef_neutral = {"intercept": float(modelo_n.intercept_[0])}
    for nombre, valor in zip(PREDICTORES, modelo_n.coef_[0]):
        coef_neutral[nombre] = float(valor)

    # --- Cobertura de los perfiles que ofrece la UI --------------------------
    # Cuántas de las combinaciones que el lector puede elegir existen de verdad
    # en la muestra. El modelo es aditivo y puede estimar las que faltan, pero
    # conviene decir cuántas salen de una extrapolación y no de casos reales.
    # Sólo cuentan los casos que corresponden a un perfil REALMENTE elegible en
    # la UI. Los que tienen alguna de las dummies ocultas activas quedan fuera:
    # si no, la UI terminaría afirmando que un perfil aparece en la encuesta
    # cuando en realidad no hay ningún caso exacto.
    #
    # El número de perfiles posibles se DERIVA de los mapeos de la UI en vez de
    # escribirse a mano. Antes era la constante 4*2*3*3*3*2*3; al sacar el
    # balotaje y abrir la ideología a seis tramos habría quedado publicando
    # 1.296 combinaciones sobre un espacio de 864, sin que nada lo detectara.
    elegibles = d[(d["victima_sin_dato"] == 0) & (d["ideol_no_ubica"] == 0)]
    ideol_codigo = sum(
        elegibles[f"ideol_{nombre}"] * i
        for i, (nombre, _, _) in enumerate(ESPEC_CRUDA["ideol_tramos"], start=1)
        if nombre != ESPEC_CRUDA["ideol_referencia"]
    )
    perfiles = list(zip(
        elegibles["tramo_edad"], elegibles["es_mujer"],
        elegibles["educ_ter_incomp"] * 1 + elegibles["educ_ter_comp"] * 2,
        ideol_codigo,
        elegibles["victima_sin_violencia"] * 1 + elegibles["victima_con_violencia"] * 2,
        elegibles["es_montevideo"],
    ))
    conteo = pd.Series(perfiles).value_counts()
    posibles = (len(EDAD_UI_TO_CODE) * 2 * len(EDUC_UI_TO_CODE)
                * len(IDEOLOGIA_UI_TO_CODE) * len(VICTIMA_UI_TO_CODE)
                * len(REGION_UI_TO_CODE))
    cobertura = {
        "posibles": int(posibles),
        "observados": int(len(conteo)),
        "con_30_o_mas": int((conteo >= 30).sum()),
    }

    # --- Tasas por grupo (para el bloque comparativo de la UI) --------------
    stats = {}
    grupos = {
        "hombres": df["es_mujer"] == 0,
        "mujeres": df["es_mujer"] == 1,
        "montevideo": df["es_montevideo"] == 1,
        "interior": df["es_montevideo"] == 0,
        # Un grupo por tramo ideológico, incluida la referencia (que no tiene
        # dummy y se define por tener todas las demás en cero y haberse ubicado).
        **{
            f"ideol_{nombre}": (
                df[f"ideol_{nombre}"] == 1 if nombre != ESPEC_CRUDA["ideol_referencia"]
                else (df["ideol_no_ubica"] == 0) & (sum(
                    df[f"ideol_{n}"] for n, _, _ in ESPEC_CRUDA["ideol_tramos"]
                    if n != ESPEC_CRUDA["ideol_referencia"]) == 0)
            )
            for nombre, _, _ in ESPEC_CRUDA["ideol_tramos"]
        },
        "victima": (df["victima_sin_violencia"] == 1) | (df["victima_con_violencia"] == 1),
        # Sólo quienes contestaron "No": si se define por las dummies en cero
        # se cuelan los sin dato y la tasa publicada sale corrida.
        "no_victima": df["victima_no_real"] == 1,
        "edad_18_29": df["tramo_edad"] == 1,
        "edad_60_plus": df["tramo_edad"] == 4,
        # Educación es el predictor más fuerte del modelo y faltaba en el
        # bloque comparativo: se mostraban sexo, edad, región, ideología y
        # victimización, todos más débiles, y no el que más pesa.
        "educ_secundaria": (df["educ_ter_incomp"] == 0) & (df["educ_ter_comp"] == 0),
        "educ_ter_incompleta": df["educ_ter_incomp"] == 1,
        "educ_ter_completa": df["educ_ter_comp"] == 1,
    }
    for nombre, mascara in grupos.items():
        sub = df[mascara & df["a_favor"].notna()]
        if len(sub) >= 30:
            stats[nombre] = round(np.average(sub["a_favor"], weights=sub[w_col]) * 100, 1)
        else:
            stats[nombre] = None  # n insuficiente para publicar

    salida = {
        "pregunta_slug": PREGUNTA_ACTIVA,
        "contrato": huella_contrato(),
        "predictores": list(PREDICTORES),
        "pregunta_columna": PREGUNTA["columna"],
        "pregunta_titulo": PREGUNTA["titulo"],
        "pregunta_afirma": PREGUNTA["afirma"],
        "pregunta_enunciado": PREGUNTA["enunciado"],
        "pregunta_titulo_corto": PREGUNTA["titulo_corto"],
        "coefficients": coeficientes,
        "odds_ratios": odds,
        "coefficients_neutral": coef_neutral,
        # Réplicas bootstrap: [intercept, *coeficientes en el orden de
        # PREDICTORES]. La app calcula el intervalo percentil con esto.
        "bootstrap": {
            "orden": ["intercept"] + list(PREDICTORES),
            "replicas": [[round(v, 5) for v in fila] for fila in boot],
            **boot_meta,
        },
        "prob_favor_nacional": round(prop_pond, 2),
        "prob_neutral_nacional": round(prop_neutral, 2),
        "stats_by_group": stats,
        "referencias": REFERENCIAS,
        "cobertura_perfiles": cobertura,
        "model_info": {
            "n": int(len(d)),
            # N efectivo de Kish: la dispersión de los ponderadores hace que
            # 2.672 respuestas "pesen" como unas 571 a efectos de precisión.
            # Publicar sólo el nominal exagera bastante la solidez.
            "n_efectivo_kish": int(round(w.sum() ** 2 / (w ** 2).sum())),
            "n_excluidos": n_excluidos,
            "n_neutrales_explicitos": n_neutral,
            "n_sin_respuesta": n_sin_resp,
            "n_encuesta": int(len(df)),
            "C": c_ppal,
            "cv_neg_log_loss": round(float(cv_ppal), 4),
            "mcfadden_r2": round(float(r2), 4),
            "ponderador": w_col,
        },
        "model_info_neutral": {
            "n": int(len(dn)),
            "C": c_neu,
            "cv_neg_log_loss": round(float(cv_neu), 4),
            "mcfadden_r2": round(float(r2_n), 4),
        },
        "fuente": "Encuesta El Observador — Seguridad pública, mayo 2026",
        "entrenado": date.today().isoformat(),
    }

    with open(MODEL_COEFFICIENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)
    print(f"\nEscrito: {MODEL_COEFFICIENTS_PATH}")

    print("\nOdds ratios (orden por magnitud del efecto):")
    for nombre, valor in sorted(odds.items(), key=lambda kv: abs(np.log(kv[1])), reverse=True):
        print(f"  {nombre:24s} OR={valor:6.3f}")


if __name__ == "__main__":
    main()

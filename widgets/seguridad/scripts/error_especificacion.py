"""
Cuánto se mueve el número publicado si la especificación fuera otra.

QUÉ PREGUNTA CONTESTA, y por qué no la contestaba nada de lo que había. El
estudio de cobertura (`cobertura_simulada.py`) mide si el intervalo cubre la
verdad SUPONIENDO QUE LA ESPECIFICACIÓN ES CORRECTA: la verdad que simula es el
propio modelo publicado. Por eso el comentario de `config.NIVEL_CALIBRADO` dice
que el error de especificación "se suma encima y no está medido". Esto lo mide.

Y NO ES LO MISMO QUE `diagnostico_econometrico.buscar_especificacion()`. Ese
barrido pregunta *¿alguna otra especificación PREDICE mejor?* y contesta que no:
mover C, splines en edad, L1, elastic net, Firth y el Likert completo no le ganan
al modelo base fuera de muestra. Pero "ninguna predice mejor" NO quiere decir
"todas dicen lo mismo". Dos modelos pueden empatar en log-loss sobre 2.700 casos
y discrepar varios puntos en un perfil concreto de los 1.008 que el widget
publica — sobre todo en los perfiles con pocos casos detrás, que son justo los
que el lector puede armar sin darse cuenta.

Entonces la pregunta es otra: entre las especificaciones que son ESTADÍSTICAMENTE
INDISTINGUIBLES de la publicada, ¿cuánto se mueve el porcentaje de cada perfil?
Ese movimiento es incertidumbre real que el intervalo bootstrap NO incluye,
porque el bootstrap remuestrea casos con la forma funcional fija.

QUÉ SE COMPARA. Sólo formas funcionales sobre LAS MISMAS SEIS VARIABLES que pide
la UI. Agregar una variable que el widget no pregunta —balotaje, situación
laboral— es otra pregunta (variable omitida) y además obligaría a marginalizar
sobre su distribución para poder predecir un perfil; queda afuera y dicho.

CÓMO SE DECIDE SI SON INDISTINGUIBLES, y acá me equivoqué en la primera versión.
Log-loss fuera de muestra con CV anidada —el C se elige dentro de cada fold,
nunca sobre los datos con los que se evalúa—, repetido sobre varias particiones
porque las diferencias son de milésimas y con una sola el ganador lo elige el
azar.

La comparación es APAREADA: cada especificación se evalúa sobre LAS MISMAS
particiones que la base, así que lo que hay que mirar es la diferencia dentro de
cada partición, no la dispersión de cada una por separado. La primera versión
dividía la diferencia por el desvío ENTRE particiones de la base, que es mucho
más grande, y por eso declaraba "indistinguible" casi todo. Con el error apareado
—media de las diferencias sobre su propio error estándar— la cosa se separa: hay
especificaciones que son claramente peores y salen, y hay al menos una que es
claramente MEJOR que la base, que es un hallazgo aparte y no un empate.

DOS CAVEATS QUE HAY QUE DECIR IGUAL.

1. Esto acota el error de especificación DENTRO de la familia que se probó. Si
   la verdad tiene una forma que no está en la lista, el número la subestima. No
   hay forma de medir eso sin conocer el mundo.

2. EL TEST DE "MEJOR QUE LA BASE" NO ES DE FIAR AL PIE DE LA LETRA, y conviene
   no titular con él. El error estándar apareado se calcula sobre 5 particiones
   de LOS MISMOS datos, y esas diferencias están correlacionadas entre sí: es
   sabido que no existe un estimador insesgado de la varianza de la validación
   cruzada de K folds (Bengio y Grandvalet, 2004), y este error estándar la
   subestima. Sumado a que acá se comparan 8 especificaciones por 4 preguntas
   —32 comparaciones—, un t de -4 no vale lo que valdría en un test único.
   El patrón lo confirma: gana una interacción distinta en cada pregunta
   —ideología x educación en dos, ideología x región en otra, ninguna en la
   cuarta—, que es justo lo que se ve cuando se está minando ruido.

   LO QUE SÍ ES SÓLIDO no depende de ese test: son especificaciones defendibles,
   la muestra no las ordena con claridad, y DISCREPAN. Esa discrepancia es el
   número que interesa.

Uso:
    python widgets/seguridad/scripts/error_especificacion.py
    python widgets/seguridad/scripts/error_especificacion.py --pregunta pena_muerte
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from widgets.seguridad import config, train_model as tm
from widgets.seguridad.model import (build_features, intervalo_probabilidad,
                                     banda_decision, _interp)

SEMILLAS = (1, 2, 3, 4, 5)

# Las seis variables de la UI, agrupadas por el bloque de dummies que las
# representa. Sirve para armar interacciones sin escribirlas a mano.
BLOQUES = {
    "edad": ["edad_30_44", "edad_45_59", "edad_60_plus"],
    "sexo": ["es_mujer"],
    "educ": ["educ_ter_incomp", "educ_ter_comp"],
    "ideol": ["ideol_izq_extrema", "ideol_izquierda", "ideol_centroizq",
              "ideol_centroderecha", "ideol_derecha", "ideol_der_extrema",
              "ideol_no_ubica"],
    "victima": ["victima_sin_violencia", "victima_con_violencia",
                "victima_sin_dato"],
    "region": ["es_montevideo"],
}


def perfiles_ui():
    """Los 1.008 perfiles que el lector puede armar."""
    return [
        dict(tramo_edad=te, es_mujer=mu, nivel_educ=ed, ideologia=id_,
             victima=vi, es_montevideo=mv)
        for te, mu, ed, id_, vi, mv in itertools.product(
            sorted(set(config.EDAD_UI_TO_CODE.values())), (0, 1),
            sorted(set(config.EDUC_UI_TO_CODE.values())),
            sorted(set(config.IDEOLOGIA_UI_TO_CODE.values())),
            sorted(set(config.VICTIMA_UI_TO_CODE.values())),
            sorted(set(config.REGION_UI_TO_CODE.values())))
    ]


def _interacciones(a, b):
    """Nombres de las columnas producto entre dos bloques."""
    return [(x, y) for x in BLOQUES[a] for y in BLOQUES[b]]


def especificaciones():
    """
    Las formas funcionales que se comparan, todas sobre las mismas seis
    variables. Cada una devuelve la lista de columnas base más los pares a
    multiplicar.
    """
    base = list(config.PREDICTORES)
    pares = [("ideol", "victima"), ("ideol", "educ"), ("ideol", "edad"),
             ("ideol", "sexo"), ("ideol", "region"), ("educ", "edad"),
             ("victima", "region")]
    specs = {"base": (base, [])}
    for a, b in pares:
        specs[f"{a}x{b}"] = (base, _interacciones(a, b))
    # Todas las de segundo orden a la vez: la más flexible de la familia lineal.
    todas = []
    for a, b in itertools.combinations(BLOQUES, 2):
        todas += _interacciones(a, b)
    specs["todas_2do_orden"] = (base, todas)
    return specs


def matriz(d, cols, pares):
    X = d[cols].values.astype(float)
    if pares:
        extra = np.column_stack([d[x].values * d[y].values for x, y in pares])
        X = np.hstack([X, extra])
    return X


def matriz_perfiles(Xp_dict, cols, pares):
    X = np.column_stack([Xp_dict[c] for c in cols])
    if pares:
        extra = np.column_stack([Xp_dict[x] * Xp_dict[y] for x, y in pares])
        X = np.hstack([X, extra])
    return X


def _mapa(d, X, y, w):
    """
    El mapa de recalibración de producción, sin réplicas (acá sólo se necesita
    la curva central). `ajustar_calibracion` hace su propia CV interna, así que
    darle sólo el fold de entrenamiento es lo correcto y no filtra nada.
    """
    return tm.ajustar_calibracion(d.assign(a_favor=y), X, y, w, 0, None)


def oof_logloss(d, cols, pares, w, y, semilla, recalibra):
    """
    Log-loss fuera de muestra con CV anidada: el C se elige DENTRO de cada fold
    de entrenamiento, nunca sobre los casos con los que después se evalúa.

    SI LA PREGUNTA SE RECALIBRA, el mapa también se ajusta dentro del fold y se
    aplica al fold de prueba. Sin eso se estaría comparando un procedimiento que
    no es el que se publica — que es justo el defecto que tenía la primera
    versión de este script y que marcó Codex.
    """
    X = matriz(d, cols, pares)
    cv = StratifiedKFold(5, shuffle=True, random_state=semilla)
    p = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        c, _ = tm.elegir_c(X[tr], y[tr], w[tr])
        if c is None:
            return None
        m = LogisticRegression(C=c, max_iter=2000, random_state=tm.RANDOM_STATE)
        m.fit(X[tr], y[tr], sample_weight=w[tr])
        pte = m.predict_proba(X[te])[:, 1]
        if recalibra:
            cal = _mapa(d.iloc[tr], X[tr], y[tr], w[tr])
            if cal is None:
                return None
            pte = np.array([_interp(cal["grilla"], cal["valores"], v) for v in pte])
        p[te] = pte
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.average(y * np.log(p) + (1 - y) * np.log(1 - p), weights=w))


def probabilidades_perfiles(d, cols, pares, w, y, Xp_dict, recalibra):
    """
    Ajusta sobre TODOS los datos y predice los 1.008 perfiles, en 0-100,
    PASANDO POR EL PROCEDIMIENTO COMPLETO: si la pregunta se recalibra, el mapa
    se ajusta con la misma receta de producción y se aplica.

    La primera versión devolvía la logística cruda y por eso, en mano dura, la
    "base" del estudio no era el número publicado: diferían una mediana de 3,52
    pp y hasta 9,32. Lo encontró Codex.
    """
    X = matriz(d, cols, pares)
    c, _ = tm.elegir_c(X, y, w)
    m = LogisticRegression(C=c, max_iter=2000, random_state=tm.RANDOM_STATE)
    m.fit(X, y, sample_weight=w)
    Xp = matriz_perfiles(Xp_dict, cols, pares)
    p = m.predict_proba(Xp)[:, 1]
    if recalibra:
        cal = _mapa(d, X, y, w)
        if cal is not None:
            p = np.array([_interp(cal["grilla"], cal["valores"], v) for v in p])
    return p * 100, c


def analizar(slug, df0, verbose=True):
    df = tm.preparar(df0, config.PREGUNTAS[slug])
    d = df[df["a_favor"].notna()].copy()
    y = d["a_favor"].values.astype(int)
    w = d[config.PONDERADOR].values

    perfiles = perfiles_ui()
    feats = [build_features(**p) for p in perfiles]
    Xp_dict = {c: np.array([f[c] for f in feats], dtype=float)
               for c in config.PREDICTORES}

    with open(config.ruta_modelo(slug), encoding="utf-8") as f:
        modelo = json.load(f)
    recalibra = slug in config.PREGUNTAS_A_RECALIBRAR

    specs = especificaciones()
    ll = {}
    for nombre, (cols, pares) in specs.items():
        vals = [oof_logloss(d, cols, pares, w, y, s, recalibra)
                for s in SEMILLAS]
        if any(v is None for v in vals):
            continue
        ll[nombre] = np.array(vals)

    base_media = float(np.mean(ll["base"]))
    n = len(SEMILLAS)
    # t de dos colas al 95% con n-1 grados de libertad, para 5 particiones.
    T = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 10: 2.262}.get(n, 2.0)

    if verbose:
        print(f"\n{'='*78}\n{slug}\n{'='*78}")
        print(f"log-loss fuera de muestra, {n} particiones, comparación APAREADA "
              f"(base {base_media:.5f})\n")
        print(f"  {'especificación':<20} {'logloss':>9} {'Δ vs base':>11} "
              f"{'EE(Δ)':>8} {'t':>7}  {'veredicto':>14}")

    admitidas, mejores = [], []
    for nombre, v in sorted(ll.items(), key=lambda kv: np.mean(kv[1])):
        m = float(np.mean(v))
        if nombre == "base":
            admitidas.append(nombre)
            if verbose:
                print(f"  {nombre:<20} {m:>9.5f} {0.0:>+11.5f} {'—':>8} {'—':>7}  "
                      f"{'(referencia)':>14}")
            continue
        dif = v - ll["base"]                       # apareada, misma partición
        md = float(np.mean(dif))
        ee = float(np.std(dif, ddof=1) / np.sqrt(n))
        tt = md / ee if ee > 0 else float("inf")
        if tt < -T:
            veredicto, ok = "MEJOR", True
            mejores.append(nombre)
        elif tt > T:
            veredicto, ok = "peor", False
        else:
            veredicto, ok = "indistinguible", True
        if ok:
            admitidas.append(nombre)
        if verbose:
            print(f"  {nombre:<20} {m:>9.5f} {md:>+11.5f} {ee:>8.5f} {tt:>+7.2f}  "
                  f"{veredicto:>14}")

    # Probabilidades de los 1.008 perfiles bajo cada especificación admitida.
    P = {}
    for nombre in admitidas:
        cols, pares = specs[nombre]
        P[nombre], _ = probabilidades_perfiles(d, cols, pares, w, y,
                                               Xp_dict, recalibra)

    M = np.vstack([P[nm] for nm in admitidas])        # specs x perfiles
    base_p = P["base"]
    rango = M.max(axis=0) - M.min(axis=0)
    desvio_vs_base = np.abs(M - base_p).max(axis=0)

    # ¿Todo el rango lo produce la especificación más flexible? Se mide sacándola.
    sin_flex = [nm for nm in admitidas if nm != "todas_2do_orden"]
    rango_sin_flex = (np.vstack([P[nm] for nm in sin_flex]).max(axis=0)
                      - np.vstack([P[nm] for nm in sin_flex]).min(axis=0)
                      ) if len(sin_flex) > 1 else np.zeros(len(base_p))

    # El ancho del intervalo publicado, para poder comparar magnitudes.
    anchos = np.array([
        (lambda iv: iv[1] - iv[0])(intervalo_probabilidad(modelo, **p))
        for p in perfiles
    ])

    # SOPORTE DE CADA PERFIL EN LA MUESTRA. Sin esto, el titular lo dominan los
    # perfiles que casi no existen —donde toda especificación extrapola y es
    # esperable que discrepen—. Se cuenta cuántos encuestados comparten las seis
    # características exactas del perfil, con el ponderador de diseño.
    claves_d = list(zip(*[d[c].values for c in config.PREDICTORES]))
    from collections import defaultdict
    peso = defaultdict(float)
    for k, ww in zip(claves_d, w):
        peso[k] += float(ww)
    claves_p = list(zip(*[Xp_dict[c] for c in config.PREDICTORES]))
    soporte = np.array([peso.get(tuple(int(v) for v in k), 0.0) for k in claves_p])
    con_soporte = soporte > 0

    def bloque(mascara, etiqueta):
        if not mascara.any():
            return None
        r, dv = rango[mascara], desvio_vs_base[mascara]
        return {"etiqueta": etiqueta, "n": int(mascara.sum()),
                "rango_mediana": float(np.median(r)),
                "rango_p95": float(np.percentile(r, 95)),
                "rango_max": float(r.max()),
                "desvio_mediana": float(np.median(dv))}

    cortes = [bloque(np.ones(len(rango), bool), "todos"),
              bloque(con_soporte, "con al menos un caso"),
              bloque(soporte >= 10, "con 10+ de peso"),
              bloque(~con_soporte, "sin ningún caso")]
    cortes = [c for c in cortes if c]

    # ¿CUÁNTAS AFIRMACIONES QUE EL WIDGET HACE SE DARÍAN VUELTA?
    #
    # La primera versión contaba cruces de estimaciones puntuales —perfiles con
    # alguna especificación de cada lado del 50%— y presentaba eso como
    # "cambian de qué lado está la mayoría". No es lo mismo, y Codex mostró por
    # qué: de esos cruces, el intervalo publicado YA contenía el 50% en 136 de
    # 136, 116 de 116, 170 de 172 y 47 de 47. O sea que el widget ya se estaba
    # absteniendo en casi todos, y no había ninguna afirmación que dar vuelta.
    #
    # Lo que hay que contar es sobre las afirmaciones que el widget SÍ hace, y
    # con sus reglas:
    #   · mayoría: `components.interpretar()` decide con la BANDA de decisión
    #     —no con el intervalo mostrado—, redondeada y de forma inclusiva.
    #   · brecha: `components.brecha_nacional()` decide con el intervalo
    #     MOSTRADO, redondeado y de forma inclusiva.
    #
    # QUÉ NO SE PUEDE HACER CON ESTO, y hay que decirlo: no se bootstrapea cada
    # especificación alternativa, así que no se sabe qué afirmaría ELLA. Lo que
    # se mide es si su estimación puntual contradice la afirmación publicada,
    # que es una cota inferior de la sensibilidad y no la cifra exacta.
    bandas = [banda_decision(modelo, **pf) for pf in perfiles]
    ivs = [intervalo_probabilidad(modelo, **pf) for pf in perfiles]
    nacional_r = round(modelo["prob_favor_nacional"])

    afirma_mayoria = np.array([
        not (round(b[0]) <= 50 <= round(b[1])) for b in bandas])
    contradice_mayoria = ((M > 50).any(axis=0) & (M < 50).any(axis=0))
    vuelta_mayoria = int((afirma_mayoria & contradice_mayoria).sum())

    afirma_brecha = np.array([
        not (round(iv[0]) <= nacional_r <= round(iv[1])) for iv in ivs])
    Mr = np.round(M)
    contradice_brecha = ((Mr > nacional_r).any(axis=0)
                         & (Mr < nacional_r).any(axis=0))
    vuelta_brecha = int((afirma_brecha & contradice_brecha).sum())

    # ¿El intervalo publicado YA contiene lo que dicen las otras
    # especificaciones? Es la pregunta que decide si haría falta ensancharlo.
    lo = np.array([iv[0] for iv in ivs]); hi = np.array([iv[1] for iv in ivs])
    exceso = np.maximum(np.maximum(lo - M.min(axis=0), 0),
                        np.maximum(M.max(axis=0) - hi, 0))
    fuera = exceso > 1e-9

    resumen = {
        "slug": slug,
        "admitidas": admitidas,
        "mejores_que_base": mejores,
        "descartadas": [nm for nm in ll if nm not in admitidas],
        "rango_mediana_sin_la_flexible": float(np.median(rango_sin_flex)),
        "logloss_base": base_media,
        "rango_mediana": float(np.median(rango)),
        "rango_p95": float(np.percentile(rango, 95)),
        "rango_max": float(rango.max()),
        "desvio_vs_base_mediana": float(np.median(desvio_vs_base)),
        "desvio_vs_base_max": float(desvio_vs_base.max()),
        "ancho_intervalo_mediano": float(np.median(anchos)),
        "razon_mediana": float(np.median(rango) / np.median(anchos)),
        "afirma_mayoria": int(afirma_mayoria.sum()),
        "afirmaciones_de_mayoria_que_se_dan_vuelta": vuelta_mayoria,
        "afirma_brecha": int(afirma_brecha.sum()),
        "afirmaciones_de_brecha_que_se_dan_vuelta": vuelta_brecha,
        "perfiles_con_alguna_espec_fuera_del_intervalo": int(fuera.sum()),
        "exceso_mediano_de_los_que_se_salen": float(
            np.median(exceso[fuera])) if fuera.any() else 0.0,
        "exceso_maximo": float(exceso.max()),
        "perfiles": len(perfiles),
        "perfiles_sin_soporte": int((~con_soporte).sum()),
        "cortes_por_soporte": cortes,
    }

    if verbose:
        print(f"\n  Especificaciones admitidas: {len(admitidas)} de {len(ll)}")
        print(f"  ({', '.join(admitidas)})\n")
        print(f"  MOVIMIENTO DEL NÚMERO PUBLICADO, en puntos porcentuales:")
        print(f"    rango entre especificaciones   mediana {resumen['rango_mediana']:5.2f}  "
              f"p95 {resumen['rango_p95']:5.2f}  máx {resumen['rango_max']:5.2f}")
        print(f"    desvío contra la base          mediana {resumen['desvio_vs_base_mediana']:5.2f}  "
              f"máx {resumen['desvio_vs_base_max']:5.2f}")
        print(f"    ancho del intervalo publicado  mediana {resumen['ancho_intervalo_mediano']:5.2f}")
        print(f"    rango / ancho                  {resumen['razon_mediana']:.1%} del intervalo")
        print(f"    rango sin la más flexible      mediana "
              f"{resumen['rango_mediana_sin_la_flexible']:5.2f}")
        if mejores:
            print(f"    OJO: le ganan a la base -> {', '.join(mejores)}")
        print(f"\n  POR SOPORTE EN LA MUESTRA (rango entre especificaciones):")
        print(f"    {'perfiles':<24} {'n':>5} {'mediana':>8} {'p95':>7} {'máx':>7}")
        for cc in cortes:
            print(f"    {cc['etiqueta']:<24} {cc['n']:>5} {cc['rango_mediana']:>8.2f} "
                  f"{cc['rango_p95']:>7.2f} {cc['rango_max']:>7.2f}")
        print(f"\n  AFIRMACIONES DEL WIDGET QUE SE DARÍAN VUELTA:")
        print(f"    «la mayoría está...»       {vuelta_mayoria:4d} de "
              f"{int(afirma_mayoria.sum()):4d} que se afirman")
        print(f"    «X pp por encima/debajo»   {vuelta_brecha:4d} de "
              f"{int(afirma_brecha.sum()):4d} que se afirman")
        print(f"\n  ¿EL INTERVALO PUBLICADO YA CONTIENE A LAS OTRAS ESPECIFICACIONES?")
        print(f"    perfiles con alguna afuera  {int(fuera.sum()):4d} de "
              f"{len(perfiles)} ({fuera.mean():.1%})")
        print(f"    exceso de los que se salen  mediana "
              f"{(np.median(exceso[fuera]) if fuera.any() else 0):.2f} pp, "
              f"máx {exceso.max():.2f} pp")
    return resumen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pregunta", action="append", dest="preguntas")
    ap.add_argument("--salida", default=None, help="archivo JSON con el resumen")
    args = ap.parse_args()

    df0 = pd.read_csv(config.DATA_FILE, encoding="utf-8-sig")
    salida = []
    for slug in (args.preguntas or config.SLUGS):
        if config.ruta_modelo(slug).exists():
            salida.append(analizar(slug, df0))

    if args.salida:
        Path(args.salida).write_text(json.dumps(salida, indent=1,
                                                ensure_ascii=False))
        print(f"\nresumen en {args.salida}")


if __name__ == "__main__":
    main()

"""
Cobertura REAL de los intervalos, por simulación con verdad conocida.

QUÉ RESPONDE, Y POR QUÉ NINGÚN CHEQUEO ANTERIOR LO RESPONDÍA.
El widget publica "70%, entre 51% y 86%". Que ese intervalo sea de 95% es una
afirmación sobre repeticiones: si se repitiera el estudio muchas veces, el
intervalo contendría el valor verdadero en el 95% de ellas.

Nada de lo que se hizo antes mide eso:

  · Comparar la tasa observada de una celda contra el intervalo mezcla el error
    del modelo con el ruido binomial de la celda, y no distingue cuál manda.
  · Ensanchar el intervalo por el error de la celda y contar coincidencias
    tampoco: no "descuenta" el ruido, y con celdas de cero apoyos la corrección
    normal se degenera —le asigna error estándar cero, o sea certeza absoluta a
    no haber observado ningún éxito—.
  · El bootstrap GENERA los intervalos; no puede validarse a sí mismo.

Lo único que lo responde es esto: fijar una verdad, generar datos desde ella,
correr el pipeline COMPLETO, y contar.

CÓMO
1. La verdad es el modelo publicado: para cada encuestado, su probabilidad
   ajustada y calibrada. Se la toma como P(Y=1) real.
2. En cada simulación se sortea y_i ~ Bernoulli(p_i) para toda la muestra,
   conservando X y los ponderadores.
3. Se corre el pipeline entero sobre esos datos: elección de C por CV ponderada,
   ajuste, bootstrap estratificado con re-elección de C en cada réplica, mapa de
   calibración fuera de muestra, y percentiles.
4. Para cada perfil de la UI se pregunta si el intervalo contiene la
   probabilidad VERDADERA de ese perfil, que se conoce por construcción.

QUÉ NO MIDE, y hay que decirlo: la verdad es el propio modelo, así que esto mide
la cobertura del procedimiento SUPONIENDO QUE LA ESPECIFICACIÓN ES CORRECTA. No
mide el error de especificación —que el mundo no sea aditivo en estas variables—,
que es la otra mitad del problema y no se puede medir sin conocer el mundo.
Una cobertura baja acá es una mala noticia inequívoca; una alta no absuelve.

Uso:
    python widgets/seguridad/scripts/cobertura_simulada.py --cronometrar
    python widgets/seguridad/scripts/cobertura_simulada.py --sims 200 --replicas 300
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import itertools
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from widgets.seguridad import config, train_model as tm
from widgets.seguridad.model import build_features, _interp, _sigmoid_pct, _z

warnings.filterwarnings("ignore")

NIVEL = 95


def huella_estudio(slug, modelo):
    """
    Sello que ata una salida del estudio al procedimiento Y a la verdad que usó.

    `config.huella_contrato` NO alcanza para esto y usarla fue un error: Codex
    verificó que sobrevive intacta a cambiar `C_GRID`, `NODOS_CALIBRACION`,
    `RANDOM_STATE` y `N_REPLICAS`, y que tampoco mira el artefacto que el
    simulador usa como verdad. Con ese sello, una medición vieja seguía
    respaldando un nivel después de cambiar la receta de entrenamiento.

    Y USARLA ENTERA TAMPOCO SIRVE, por el motivo opuesto: `huella_contrato`
    incluye `NIVEL_CALIBRADO`, que es justamente lo que este estudio decide. Con
    eso adentro, subir el nivel de una pregunta invalidaba las ocho corridas que
    lo habían elegido — la conclusión anulaba a sus propios insumos. Lo marcó
    Codex en la tercera vuelta. Acá se toma sólo la parte del contrato que
    cambia el SIGNIFICADO de los datos, no la que sale del estudio.

    Qué entra:
      · el contrato de codificación menos el nivel: predictores, mapeos de la
        UI, referencias, escala Likert, ponderador, especificación cruda y la
        columna de la pregunta;
      · las perillas numéricas del procedimiento: la grilla de C EN SU ORDEN
        —`elegir_c` se queda con el primer empatado, así que invertirla cambia
        qué C sale—, los nodos del mapa, la semilla, la cantidad de réplicas de
        producción, el nivel base y los niveles y factores evaluados;
      · el ARTEFACTO usado como verdad: coeficientes y mapa central, porque la
        cobertura se mide contra las probabilidades que salen de él;
      · el CÓDIGO de las funciones que definen el procedimiento, vía el AST
        reimpreso con `ast.unparse` y sin docstrings. Un cambio de lógica
        invalida el estudio; uno de comentario, docstring o formato, no.

    QUÉ SIGUE SIN CUBRIR, y conviene tenerlo escrito:
      · la base de datos de entrada;
      · la lógica de cualquier función que no esté en la lista de abajo —entre
        ellas `entrenar`, que es la que cablea el apareamiento;
      · el reimpreso del AST puede variar entre versiones de Python. Codex
        verificó que `ast.dump` da hashes distintos en 3.11, 3.12 y 3.13 por
        campos nuevos como `type_params`; `ast.unparse` es bastante más estable
        porque no serializa nombres de campos, pero no está garantizado. Por eso
        la salida guarda aparte la versión de Python con la que se selló, para
        poder distinguir "cambió el procedimiento" de "cambió el intérprete".
    """
    import ast
    import hashlib
    import inspect

    def estructura(fn):
        arbol = ast.parse(inspect.getsource(fn).lstrip())
        for nodo in ast.walk(arbol):
            if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef, ast.Module)):
                cuerpo = nodo.body
                if (cuerpo and isinstance(cuerpo[0], ast.Expr)
                        and isinstance(cuerpo[0].value, ast.Constant)
                        and isinstance(cuerpo[0].value.value, str)):
                    nodo.body = cuerpo[1:] or [ast.Pass()]
        return ast.unparse(arbol)

    contrato = json.dumps({
        "pregunta": slug,
        "columna": config.PREGUNTAS[slug]["columna"],
        "predictores": sorted(config.PREDICTORES),
        "edad": sorted(config.EDAD_UI_TO_CODE.items()),
        "educacion": sorted(config.EDUC_UI_TO_CODE.items()),
        "ideologia": sorted(config.IDEOLOGIA_UI_TO_CODE.items()),
        "victima": sorted(config.VICTIMA_UI_TO_CODE.items()),
        "region": sorted(config.REGION_UI_TO_CODE.items()),
        "referencias": sorted(config.REFERENCIAS.items()),
        "likert": sorted(config.LIKERT_MAP.items()),
        "favor": sorted(config.LIKERT_FAVOR),
        "contra": sorted(config.LIKERT_CONTRA),
        "neutral": config.LIKERT_NEUTRAL,
        "ponderador": config.PONDERADOR,
        "recalibradas": sorted(config.PREGUNTAS_A_RECALIBRAR),
        "espec_cruda": json.dumps(config.ESPEC_CRUDA, sort_keys=True),
    }, sort_keys=True, ensure_ascii=False)

    piezas = [
        contrato,
        repr(list(tm.C_GRID)),          # EN SU ORDEN, no ordenada
        repr(tm.NODOS_CALIBRACION),
        repr(tm.RANDOM_STATE),
        repr(tm.N_REPLICAS),
        repr(NIVEL), repr(NIVELES), repr(FACTORES),
        json.dumps(modelo.get("coefficients"), sort_keys=True),
        json.dumps((modelo.get("calibracion") or {}).get("grilla")),
        json.dumps((modelo.get("calibracion") or {}).get("valores")),
    ] + [estructura(f) for f in (tm.elegir_c, tm.ajustar_calibracion,
                                 tm._nodos_calibracion, tm.bootstrap_coeficientes,
                                 una_simulacion)]
    return hashlib.sha256("|".join(piezas).encode()).hexdigest()[:16]

# Dos familias de corrección, evaluadas en la MISMA corrida porque lo caro es
# ajustar, no medir:
#   · subir el nivel nominal del percentil (95 -> 96, 97...);
#   · ensanchar el intervalo del 95% por un factor, alrededor de su centro.
# No son equivalentes: la primera sigue la forma de la distribución bootstrap y
# la segunda la estira. Se prueban las dos para ver cuál llega a 95% real sin
# ensanchar de más.
NIVELES = (95, 96, 97, 98, 99)
FACTORES = (1.00, 1.05, 1.10, 1.15, 1.20, 1.30)


def perfiles_ui():
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


def _percentil(ordenados, q):
    if not ordenados:
        return None
    if len(ordenados) == 1:
        return ordenados[0]
    pos = q * (len(ordenados) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordenados) - 1)
    t = pos - lo
    return ordenados[lo] * (1 - t) + ordenados[hi] * t


def una_simulacion(d, X, w, p_true, Xp, estratos, n_replicas, rng, recalibra):
    """
    Una repetición completa: sortea resultados, corre TODO y devuelve los
    intervalos de cada perfil.
    """
    y = (rng.random(len(p_true)) < p_true).astype(int)
    if len(np.unique(y)) < 2:
        return None

    c_ppal, _ = tm.elegir_c(X, y, w)
    if c_ppal is None:
        return None
    modelo = LogisticRegression(C=c_ppal, max_iter=2000,
                                random_state=tm.RANDOM_STATE)
    modelo.fit(X, y, sample_weight=w)

    # EL BOOTSTRAP DE COEFICIENTES VA PRIMERO, igual que en producción
    # (`entrenar` llama a bootstrap_coeficientes y después a
    # ajustar_calibracion): hace falta saber QUÉ SORTEOS SOBREVIVIERON para
    # pedirle al mapa exactamente esos y no perder el apareamiento. Acá el mapa
    # se ajustaba antes; el orden no cambia ningún resultado porque los dos
    # remuestreos se siembran por separado con `default_rng(RANDOM_STATE)` y el
    # sorteo de respuestas usa otro generador, pero conviene que el simulador
    # tenga el mismo orden que lo que simula. (Escribí que producción lo hacía
    # al revés; era falso y lo corrigió Codex.)

    # Bootstrap estratificado, re-eligiendo C en cada réplica, igual que
    # producción. Es lo que hace caro esto y también lo que hay que medir: con C
    # fijo los intervalos se achican y la cobertura medida no sería la del
    # procedimiento que se publica.
    # EL REMUESTREO DE COEFICIENTES USA SU PROPIO GENERADOR, sembrado igual que
    # producción, y NO el `rng` de la simulación.
    #
    # Por qué importa: en producción, `bootstrap_coeficientes` y
    # `ajustar_calibracion` arrancan las dos con `default_rng(RANDOM_STATE)` y
    # recorren los mismos estratos en el mismo orden, así que la réplica i de
    # coeficientes y la i del mapa salen del MISMO remuestreo. Acá se usaba el
    # generador de la simulación —ya avanzado por el sorteo de resultados— y esa
    # correspondencia se rompía: el simulador medía un procedimiento distinto
    # del publicado. Codex lo midió sobre los mismos resultados simulados: la
    # cobertura de mano dura al nivel 98 pasaba de 97,00% a 98,01%.
    rng_boot = np.random.default_rng(tm.RANDOM_STATE)
    indices = [np.where(estratos == e)[0] for e in np.unique(estratos)]
    coefs = []
    sorteos_validos = []
    for k in range(n_replicas):
        idx = np.concatenate([rng_boot.choice(ix, size=len(ix), replace=True)
                              for ix in indices])
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        cb, _ = tm.elegir_c(X[idx], yb, w[idx])
        if cb is None:
            continue
        mb = LogisticRegression(C=cb, max_iter=2000, random_state=tm.RANDOM_STATE)
        mb.fit(X[idx], yb, sample_weight=w[idx])
        coefs.append(np.r_[mb.intercept_[0], mb.coef_[0]])
        sorteos_validos.append(k)
    if len(coefs) < 30:
        return None
    coefs = np.array(coefs)

    # Mapa de calibración, con la misma receta que producción y sobre los mismos
    # sorteos que sobrevivieron arriba.
    mapa = None
    if recalibra:
        mapa = tm.ajustar_calibracion(d.assign(a_favor=y), X, y, w,
                                      n_replicas, sorteos_validos)
        if mapa is None:
            return None

    # Probabilidades de cada perfil por réplica, ya calibradas.
    Z = coefs[:, 0][:, None] + coefs[:, 1:] @ Xp.T          # réplicas x perfiles
    P = 1.0 / (1.0 + np.exp(-Z)) * 100
    if mapa:
        reps = mapa.get("replicas") or []
        for i in range(P.shape[0]):
            xs, ys = (reps[i % len(reps)] if reps
                      else (mapa["grilla"], mapa["valores"]))
            P[i] = [_interp(xs, ys, v / 100.0) * 100 for v in P[i]]

    P.sort(axis=0)
    cols = [list(P[:, j]) for j in range(P.shape[1])]

    por_nivel = {}
    for niv in NIVELES:
        cola = (100 - niv) / 2 / 100
        por_nivel[niv] = (np.array([_percentil(cc, cola) for cc in cols]),
                          np.array([_percentil(cc, 1 - cola) for cc in cols]))

    lo95, hi95 = por_nivel[NIVEL]
    centro = (lo95 + hi95) / 2
    por_factor = {}
    for fa in FACTORES:
        mitad = (hi95 - lo95) / 2 * fa
        # Se recorta a 0-100: una probabilidad ensanchada no puede salirse.
        por_factor[fa] = (np.clip(centro - mitad, 0, 100),
                          np.clip(centro + mitad, 0, 100))
    return por_nivel, por_factor


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--replicas", type=int, default=tm.N_REPLICAS,
                    help="por defecto, las MISMAS que producción")
    ap.add_argument("--pregunta", action="append", dest="preguntas")
    ap.add_argument("--semilla", type=int, default=20260908,
                    help="para repartir las simulaciones entre procesos")
    ap.add_argument("--cronometrar", action="store_true",
                    help="corre UNA simulación y reporta cuánto tarda")
    # POR QUÉ SE PUEDE ELEGIR LA CARPETA. `agregar_calibracion.py` exige que
    # todas las corridas de una pregunta compartan B, y con razón: promediar
    # coberturas medidas con distinto número de réplicas mezcla dos
    # procedimientos. Una corrida con otro B tiene que ir a otro lado, no al
    # lado de las que ya están.
    ap.add_argument("--cada", type=int, default=5,
                    help="guardar un parcial cada N simulaciones (0 = nunca)")
    ap.add_argument("--salidas", default=None,
                    help="carpeta donde escribir (por defecto scripts/salidas)")
    args = ap.parse_args()
    slugs = args.preguntas or config.SLUGS

    df0 = pd.read_csv(config.DATA_FILE, encoding="utf-8-sig")
    perfiles = perfiles_ui()
    Xp = np.array([[build_features(**p)[k] for k in config.PREDICTORES]
                   for p in perfiles], dtype=float)

    for slug in slugs:
        ruta = config.ruta_modelo(slug)
        if not ruta.exists():
            continue
        with open(ruta, encoding="utf-8") as f:
            publicado = json.load(f)

        df = tm.preparar(df0, config.PREGUNTAS[slug])
        d = df[df["a_favor"].notna()].copy()
        X = d[config.PREDICTORES].values.astype(float)
        w = d[config.PONDERADOR].values
        estratos = d["estrato"].values
        recalibra = slug in config.PREGUNTAS_A_RECALIBRAR
        arranque = time.time()

        # LA VERDAD: la probabilidad que el modelo publicado le asigna a cada
        # encuestado, ya calibrada. Es el mundo que se simula.
        coef = publicado["coefficients"]
        cal = publicado.get("calibracion")
        p_true = np.array([
            _interp(cal["grilla"], cal["valores"],
                    _sigmoid_pct(_z(coef, dict(zip(config.PREDICTORES, fila)))) / 100)
            if cal else
            _sigmoid_pct(_z(coef, dict(zip(config.PREDICTORES, fila)))) / 100
            for fila in X
        ])
        # Y la verdad de cada PERFIL, que es contra lo que se mide la cobertura.
        verdad_perfil = np.array([
            (_interp(cal["grilla"], cal["valores"],
                     _sigmoid_pct(_z(coef, dict(zip(config.PREDICTORES, fila)))) / 100)
             if cal else
             _sigmoid_pct(_z(coef, dict(zip(config.PREDICTORES, fila)))) / 100) * 100
            for fila in Xp
        ])

        rng = np.random.default_rng(args.semilla)
        if args.cronometrar:
            t0 = time.time()
            r = una_simulacion(d, X, w, p_true, Xp, estratos,
                               args.replicas, rng, recalibra)
            dt = time.time() - t0
            print(f"{slug}: una simulación con {args.replicas} réplicas = "
                  f"{dt:.1f}s  ->  {args.sims} sims serían {dt*args.sims/60:.0f} min")
            continue

        dentro_niv = {n: np.zeros(len(perfiles)) for n in NIVELES}
        dentro_fac = {f: np.zeros(len(perfiles)) for f in FACTORES}
        anchos_fac = {f: [] for f in FACTORES}
        validas = 0
        desde = 0

        # A un directorio del repo, no a la carpeta temporal de una sesión.
        destino = (Path(args.salidas) if args.salidas
                   else Path(__file__).parent / "salidas")
        destino.mkdir(exist_ok=True)
        salida = destino / f"cal-{slug}-{args.semilla}.json"
        huella = huella_estudio(slug, publicado)

        # REANUDAR. El 9/9/2026 se perdieron seis horas de ocho procesos porque
        # el resultado se escribía recién al final y la máquina se reinició.
        # Ahora se guarda un parcial cada `--cada` simulaciones, con el estado
        # del generador, y si al arrancar hay uno compatible se sigue desde ahí.
        # "Compatible" = misma huella del estudio, mismas réplicas, mismo
        # objetivo de sims. Si algo de eso cambió, el parcial no sirve y se
        # arranca de cero, diciéndolo.
        if salida.exists():
            previo = json.loads(salida.read_text(encoding="utf-8"))
            if not previo.get("parcial"):
                print(f"{slug}: ya hay una corrida COMPLETA en {salida.name}; "
                      "borrala si querés repetirla", flush=True)
                continue
            compatible = (previo.get("huella") == huella
                          and previo.get("replicas") == args.replicas
                          and previo.get("sims_objetivo") == args.sims)
            if compatible:
                desde = int(previo["sims_hechas"])
                validas = int(previo["sims_validas"])
                for n in NIVELES:
                    dentro_niv[n] = np.array(previo["niveles"][str(n)], float)
                for f in FACTORES:
                    dentro_fac[f] = np.array(previo["factores"][str(f)], float)
                    anchos_fac[f] = list(previo["anchos_lista"][str(f)])
                rng.bit_generator.state = previo["rng_state"]
                print(f"{slug}: reanudando desde la simulación {desde} de "
                      f"{args.sims} ({salida.name})", flush=True)
            else:
                print(f"{slug}: hay un parcial en {salida.name} pero NO es "
                      "compatible (cambió la huella, las réplicas o el objetivo); "
                      "se arranca de cero", flush=True)

        def guardar(parcial, sims_hechas):
            salida.write_text(json.dumps({
                "slug": slug, "sims_validas": validas, "replicas": args.replicas,
                "semilla": args.semilla,
                "parcial": parcial,
                "sims_hechas": sims_hechas, "sims_objetivo": args.sims,
                # Ata la medición al modelo que se usó como verdad. Sin esto,
                # una salida vieja sigue "respaldando" un nivel después de que
                # cambió la especificación, y el test que compara ambos pasa
                # igual. Lo marcó Codex el 8/9/2026.
                "huella": huella,
                "python": "%d.%d" % sys.version_info[:2],
                "segundos": round(time.time() - arranque, 1),
                # OJO: son ACIERTOS por perfil, no porcentajes. Quien los lea
                # tiene que dividir por "sims_validas".
                "niveles": {str(n): dentro_niv[n].tolist() for n in NIVELES},
                "factores": {str(f): dentro_fac[f].tolist() for f in FACTORES},
                "anchos": {str(f): (float(np.median(anchos_fac[f]))
                                    if anchos_fac[f] else None) for f in FACTORES},
                # La lista entera hace falta para reanudar; la mediana de
                # arriba es lo que consume el agregador.
                "anchos_lista": {str(f): anchos_fac[f] for f in FACTORES},
                # El estado del generador, para que reanudar dé exactamente la
                # misma secuencia que una corrida sin cortes.
                "rng_state": rng.bit_generator.state if parcial else None,
            }))

        t0 = time.time()
        for s in range(desde, args.sims):
            r = una_simulacion(d, X, w, p_true, Xp, estratos,
                               args.replicas, rng, recalibra)
            if r is None:
                continue
            por_nivel, por_factor = r
            for n, (lo, hi) in por_nivel.items():
                dentro_niv[n] += (lo <= verdad_perfil) & (verdad_perfil <= hi)
            for f, (lo, hi) in por_factor.items():
                dentro_fac[f] += (lo <= verdad_perfil) & (verdad_perfil <= hi)
                anchos_fac[f].append(float(np.median(hi - lo)))
            validas += 1
            if args.cada and (s + 1) % args.cada == 0 and (s + 1) < args.sims:
                guardar(parcial=True, sims_hechas=s + 1)
            if (s + 1) % 20 == 0:
                cob = dentro_niv[NIVEL].sum() / (validas * len(perfiles))
                print(f"  [{slug}] {s+1}/{args.sims} sims  cobertura al 95% "
                      f"{cob*100:.1f}%  ({time.time()-t0:.0f}s)", flush=True)

        print(f"\n{slug}: {validas} simulaciones válidas, {args.replicas} réplicas")
        print("  subir el NIVEL nominal:")
        for n in NIVELES:
            c_ = dentro_niv[n] / max(validas, 1)
            print(f"    {n}% -> real {c_.mean()*100:5.1f}%   "
                  f"bajo 90%: {(c_ < 0.90).sum():4d}   peor {c_.min()*100:5.1f}%")
        print("  ENSANCHAR el intervalo del 95% por un factor:")
        for f in FACTORES:
            c_ = dentro_fac[f] / max(validas, 1)
            print(f"    x{f:.2f} -> real {c_.mean()*100:5.1f}%   "
                  f"bajo 90%: {(c_ < 0.90).sum():4d}   peor {c_.min()*100:5.1f}%   "
                  f"ancho {np.median(anchos_fac[f]):5.1f}pp")

        guardar(parcial=False, sims_hechas=args.sims)
        print(f"  guardado en {salida.name}")


if __name__ == "__main__":
    main()

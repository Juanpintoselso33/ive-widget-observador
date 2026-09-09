"""
Ancho del intervalo por cuánto respaldo tiene el perfil en la muestra.

POR QUÉ EXISTE. El 9/9/2026 El Observador pidió sacar el intervalo de la
pantalla porque es demasiado ancho para servirle a un lector. Antes de aceptar
que el problema no tenía arreglo hubo que contestar una pregunta concreta: ¿el
ancho lo empujan los perfiles que casi no existen en la muestra, o también son
anchos los que sí tienen casos detrás? Si fuera lo primero, la salida era
restringir la grilla y no esconder el intervalo.

La respuesta —que también son anchos— se midió con un script descartable y el
número entró en la documentación sin quedar guardado en ningún lado. Lo marcó
Codex al revisar ese commit: una afirmación que gobierna una decisión de
producto no puede apoyarse sólo en la palabra de quien la corrió. Este script
existe para que se pueda volver a correr y dé lo mismo.

QUÉ MIDE. Para cada pregunta y cada uno de los 1.008 perfiles de la UI: el ancho
del intervalo que el widget calcula —con envolvente de especificación incluida,
o sea el que de verdad gobierna— cortado por el peso muestral de los
encuestados que comparten las seis características exactas del perfil.

QUÉ NO MIDE, y conviene no leerlo de más:

  · No demuestra que ninguna especificación pueda dar intervalos más angostos.
    El modelo comparte coeficientes entre perfiles y no estima 1.008
    proporciones independientes; que restringir por soporte no achique el ancho
    dice que el problema no es la extrapolación, no que el problema sea
    insalvable. Lo marcó Codex y es una distinción real.
  · El "soporte" es el peso de los encuestados con las seis características
    IDÉNTICAS. Es un piso, no la información que el modelo usa para ese perfil:
    al ser aditivo, también aprende de perfiles parecidos.

Uso:
    python widgets/seguridad/scripts/anchos_por_soporte.py
    python widgets/seguridad/scripts/anchos_por_soporte.py --salida ruta.json
"""

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from widgets.seguridad import config
from widgets.seguridad import train_model as tm
from widgets.seguridad.model import (build_features, intervalo_probabilidad,
                                     load_modelos)

CAMPOS = ("tramo_edad", "es_mujer", "nivel_educ", "ideologia", "victima",
          "es_montevideo")

# Los cortes son de PESO ponderado, no de casos crudos: el ponderador de diseño
# es lo que define cuánto pesa cada respuesta en el ajuste.
CORTES = (
    ("todos", lambda s: np.ones(len(s), bool)),
    ("sin ningún caso", lambda s: s == 0),
    ("con al menos uno", lambda s: s > 0),
    ("con 10+ de peso", lambda s: s >= 10),
)


def perfiles_ui():
    """Los 1.008 perfiles que el lector puede armar."""
    return [
        dict(zip(CAMPOS, v))
        for v in itertools.product(
            sorted(set(config.EDAD_UI_TO_CODE.values())), (0, 1),
            sorted(set(config.EDUC_UI_TO_CODE.values())),
            sorted(set(config.IDEOLOGIA_UI_TO_CODE.values())),
            sorted(set(config.VICTIMA_UI_TO_CODE.values())),
            sorted(set(config.REGION_UI_TO_CODE.values())))
    ]


def soporte_de_cada_perfil(d, w, perfiles):
    """Peso muestral de los encuestados con las seis características exactas."""
    peso = defaultdict(float)
    for clave, ww in zip(zip(*[d[c].values for c in config.PREDICTORES]), w):
        peso[clave] += float(ww)
    salida = []
    for p in perfiles:
        f = build_features(**p)
        salida.append(peso.get(tuple(int(f[c]) for c in config.PREDICTORES), 0.0))
    return np.array(salida)


def analizar(slug, df0, modelo, perfiles):
    d = tm.preparar(df0, config.PREGUNTAS[slug])
    d = d[d["a_favor"].notna()]
    w = d[config.PONDERADOR].values

    anchos = np.array([
        (lambda iv: iv[1] - iv[0])(intervalo_probabilidad(modelo, **p))
        for p in perfiles
    ])
    soporte = soporte_de_cada_perfil(d, w, perfiles)

    info = modelo.get("model_info", {})
    return {
        "slug": slug,
        "n": info.get("n"),
        "n_efectivo_kish": info.get("n_efectivo_kish"),
        "perfiles": len(perfiles),
        "cortes": [
            {
                "etiqueta": etiqueta,
                "n": int(m.sum()),
                "ancho_mediano": float(np.median(anchos[m])),
                "ancho_p90": float(np.percentile(anchos[m], 90)),
                "ancho_max": float(anchos[m].max()),
            }
            for etiqueta, fn in CORTES
            if (m := fn(soporte)).any()
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--salida", default=None,
                    help="por defecto scripts/salidas/anchos-por-soporte.json")
    args = ap.parse_args()

    df0 = pd.read_csv(config.DATA_FILE, encoding="utf-8-sig")
    modelos = load_modelos()
    perfiles = perfiles_ui()

    salida = [analizar(s, df0, modelos[s], perfiles) for s in config.SLUGS]

    print(f"\n{'pregunta':<20} {'grupo':<20} {'n':>5} {'mediana':>9} {'p90':>7}")
    for bloque in salida:
        for c in bloque["cortes"]:
            print(f"{bloque['slug']:<20} {c['etiqueta']:<20} {c['n']:>5} "
                  f"{c['ancho_mediano']:>9.1f} {c['ancho_p90']:>7.1f}")
        print()

    mejores = [c["ancho_mediano"] for b in salida for c in b["cortes"]
               if c["etiqueta"] == "con 10+ de peso"]
    print(f"En los perfiles MEJOR SOSTENIDOS el ancho mediano va de "
          f"{min(mejores):.1f} a {max(mejores):.1f} pp.")
    print("Ese es el número que decide: si acá fuera angosto, la salida sería "
          "restringir la grilla en vez de esconder el intervalo.")

    destino = (Path(args.salida) if args.salida
               else Path(__file__).parent / "salidas" / "anchos-por-soporte.json")
    destino.parent.mkdir(exist_ok=True)
    destino.write_text(json.dumps(salida, indent=1, ensure_ascii=False),
                       encoding="utf-8")
    print(f"\nsalida en {destino}")


if __name__ == "__main__":
    main()

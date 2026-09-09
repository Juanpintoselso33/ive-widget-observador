"""
Convierte la salida de `error_especificacion.py --detalle` en el artefacto que
lee el widget: `modelos/envolvente_espec.json`.

QUÉ ES LA ENVOLVENTE. Para cada uno de los 1.008 perfiles de la UI, el mínimo y
el máximo que le asigna cualquiera de las especificaciones que la muestra NO
logra distinguir de la publicada. El widget estira su intervalo hasta contenerla.

POR QUÉ EXISTE ESTE PASO Y NO SE HACE EN `train_model`. Calcular la envolvente
exige ajustar nueve especificaciones con validación cruzada anidada sobre cinco
particiones, o sea unas cuarenta y cinco corridas completas por pregunta. Meterlo
adentro del entrenamiento lo volvería impracticable de correr a mano. Y la
envolvente no es una propiedad de los coeficientes publicados sino de los datos y
del conjunto de formas que se probaron, así que separarla es también lo correcto
conceptualmente.

EL PRECIO DE SEPARARLO es que la envolvente puede quedar vieja. Por eso el
artefacto guarda la huella del contrato de cada pregunta, y
`model.problemas_de_envolvente()` verifica además que el punto publicado caiga
dentro de la envolvente de su perfil — un control que la huella no da, porque la
huella cubre la configuración y no los coeficientes.

QUÉ NO ES. No es una cota del error de especificación. Es la dispersión DENTRO de
la familia de nueve formas que se probó, bajo un criterio de admisión que el
propio estudio documenta como poco confiable al pie de la letra. Si la verdad
tiene una forma que no está en la lista, esto no la cubre. Ver el docstring de
`error_especificacion.py`.

Uso:
    python widgets/seguridad/scripts/error_especificacion.py --detalle \
        --salida /tmp/espec.json
    python widgets/seguridad/scripts/agregar_envolvente.py /tmp/espec.json
"""

import argparse
import datetime
import itertools
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from widgets.seguridad import config
from widgets.seguridad.model import load_model, predict_probability

CAMPOS = ("tramo_edad", "es_mujer", "nivel_educ", "ideologia", "victima",
          "es_montevideo")


def perfiles_ui():
    """Los 1.008 perfiles que el lector puede armar."""
    return [
        dict(zip(CAMPOS, vals))
        for vals in itertools.product(
            sorted(set(config.EDAD_UI_TO_CODE.values())), (0, 1),
            sorted(set(config.EDUC_UI_TO_CODE.values())),
            sorted(set(config.IDEOLOGIA_UI_TO_CODE.values())),
            sorted(set(config.VICTIMA_UI_TO_CODE.values())),
            sorted(set(config.REGION_UI_TO_CODE.values())))
    ]


def construir(estudio):
    """Arma el artefacto y verifica lo que tiene que cumplirse."""
    esperados = perfiles_ui()
    claves_esperadas = {config.clave_perfil(**p) for p in esperados}
    preguntas, avisos = {}, []

    for bloque in estudio:
        slug = bloque["slug"]
        detalle = bloque.get("detalle")
        if not detalle:
            avisos.append(f"{slug}: el estudio no trae --detalle, se saltea")
            continue

        modelo = load_model(slug)
        tabla, fuera = {}, 0
        for fila in detalle:
            perfil = {k: int(fila[k]) for k in CAMPOS}
            lo, hi = float(fila["spec_min"]), float(fila["spec_max"])
            if lo > hi:                       # no debería pasar nunca
                raise SystemExit(f"{slug}: envolvente al revés en {perfil}")
            # El control que de verdad ata el artefacto a estos coeficientes:
            # la base está entre las admitidas, así que el punto publicado tiene
            # que caer adentro. Si esto falla, el estudio se corrió contra otro
            # modelo y guardarlo sería peor que no tener envolvente.
            punto = predict_probability(modelo, **perfil)
            if not (lo - 1e-6 <= punto <= hi + 1e-6):
                fuera += 1
            tabla[config.clave_perfil(**perfil)] = [round(lo, 6), round(hi, 6)]

        if fuera:
            raise SystemExit(
                f"{slug}: el punto publicado cae fuera de la envolvente en "
                f"{fuera} perfiles. El estudio no corresponde a estos "
                f"coeficientes; reentrenar o rehacer el estudio.")
        if set(tabla) != claves_esperadas:
            faltan = claves_esperadas - set(tabla)
            sobran = set(tabla) - claves_esperadas
            raise SystemExit(
                f"{slug}: la grilla no coincide con la UI "
                f"(faltan {len(faltan)}, sobran {len(sobran)})")

        preguntas[slug] = {
            "contrato": config.huella_contrato(slug),
            "admitidas": bloque.get("admitidas", []),
            "descartadas": bloque.get("descartadas", []),
            "perfiles_fuera_del_intervalo":
                bloque.get("perfiles_con_alguna_espec_fuera_del_intervalo"),
            "exceso_maximo": bloque.get("exceso_maximo"),
            "perfiles": tabla,
        }

    faltantes = [s for s in config.SLUGS if s not in preguntas]
    if faltantes:
        raise SystemExit(
            "faltan preguntas en el estudio: " + ", ".join(faltantes) +
            ". El artefacto tiene que cubrir las cuatro o el widget ensancharía "
            "unas sí y otras no sin que se note.")

    return {
        "generado": datetime.date.today().isoformat(),
        "criterio": "min y max sobre las especificaciones que el log-loss fuera "
                    "de muestra no distingue de la publicada",
        "origen": "widgets/seguridad/scripts/error_especificacion.py --detalle",
        "preguntas": preguntas,
    }, avisos


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("estudio", help="JSON de error_especificacion.py --detalle")
    ap.add_argument("--salida", default=None,
                    help=f"por defecto {config.RUTA_ENVOLVENTE}")
    args = ap.parse_args()

    estudio = json.loads(Path(args.estudio).read_text(encoding="utf-8"))
    artefacto, avisos = construir(estudio)
    for a in avisos:
        print(f"AVISO: {a}")

    destino = Path(args.salida) if args.salida else config.RUTA_ENVOLVENTE
    destino.write_text(json.dumps(artefacto, ensure_ascii=False, indent=1),
                       encoding="utf-8")

    print(f"\nenvolvente en {destino}")
    for slug, p in artefacto["preguntas"].items():
        print(f"  {slug:<20} {len(p['perfiles']):>5} perfiles | "
              f"{len(p['admitidas'])} especificaciones admitidas | "
              f"{p['perfiles_fuera_del_intervalo']} se salían del intervalo "
              f"(máx {p['exceso_maximo']:.2f} pp)")


if __name__ == "__main__":
    main()

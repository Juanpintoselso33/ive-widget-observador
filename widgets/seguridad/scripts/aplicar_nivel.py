"""
Aplica los niveles que eligió `agregar_calibracion.py` sin reentrenar.

POR QUÉ NO ALCANZA CON EDITAR `config.NIVEL_CALIBRADO`. El nivel entra en
`huella_contrato()`, así que cambiarlo invalida la huella guardada en los cuatro
JSON de modelo y en `envolvente_espec.json`, y la app se niega a arrancar. La
salida obvia —reentrenar— son otras veinte horas de bootstrap para producir
exactamente los mismos coeficientes: el nivel no participa del ajuste, se aplica
al leer, en `intervalo_probabilidad()`.

POR QUÉ RE-SELLAR NO ES HACER TRAMPA, Y CÓMO SE GARANTIZA. Un script que
re-sella la huella cada vez que cambia la configuración vuelve inútil el chequeo
que la huella existe para hacer: pasaría siempre. Acá el re-sellado está
condicionado a una prueba.

Para cada pregunta se recalcula la huella SUSTITUYENDO el nivel que el JSON trae
guardado. Si esa huella reproduce exactamente la que el JSON tiene, entonces lo
único que cambió entre el modelo entrenado y la configuración actual es el
nivel, y re-sellar es legítimo. Si no la reproduce, cambió algo más —un mapeo,
una categoría, el colapso educativo— y el script ABORTA pidiendo un
reentrenamiento, que es lo correcto.

QUÉ TOCA:
  · `config.NIVEL_CALIBRADO`
  · `nivel_calibrado` y `contrato` de cada `modelos/model_<slug>.json`
  · `contrato` de cada pregunta en `modelos/envolvente_espec.json`

La envolvente en sí no cambia: son los mínimos y máximos de las estimaciones
puntuales entre especificaciones, que no dependen del nivel.

Uso:
    python widgets/seguridad/scripts/aplicar_nivel.py \
        --salidas widgets/seguridad/scripts/salidas-b10000
    python widgets/seguridad/scripts/aplicar_nivel.py --nivel pena_muerte=97 ...
"""

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from widgets.seguridad import config
from widgets.seguridad.config import (NIVEL_CALIBRADO, RUTA_ENVOLVENTE, SLUGS,
                                      huella_contrato, ruta_modelo)

CONFIG_PY = Path(config.__file__)


def huella_con(slug, nivel):
    """La huella del contrato como si `slug` publicara en `nivel`."""
    guardado = NIVEL_CALIBRADO[slug]
    NIVEL_CALIBRADO[slug] = nivel
    try:
        return huella_contrato(slug)
    finally:
        NIVEL_CALIBRADO[slug] = guardado


def niveles_desde_estudio(salidas):
    """Los niveles que elige el criterio sobre las corridas de `salidas`."""
    sys.path.insert(0, str(Path(__file__).parent))
    from agregar_calibracion import _cargar, elegir
    return {slug: int(elegir(corridas)[0])
            for slug, corridas in _cargar(salidas).items()}


def verificar_que_solo_cambia_el_nivel(modelos):
    """
    El control que hace legítimo el re-sellado. Devuelve la lista de problemas.
    """
    problemas = []
    for slug, m in modelos.items():
        guardado = m.get("nivel_calibrado")
        if guardado is None:
            problemas.append(f"«{slug}»: el JSON no trae nivel_calibrado")
            continue
        esperada = huella_con(slug, guardado)
        if m.get("contrato") != esperada:
            problemas.append(
                f"«{slug}»: con su propio nivel ({guardado}) la huella da "
                f"{esperada} y el JSON dice {m.get('contrato')} — cambió algo "
                "MÁS que el nivel, hay que reentrenar")
    return problemas


def escribir_config(nuevos):
    """Reescribe el bloque NIVEL_CALIBRADO conservando los comentarios."""
    texto = CONFIG_PY.read_text(encoding="utf-8")
    for slug, nivel in nuevos.items():
        patron = rf'(^    "{re.escape(slug)}":\s*)(\d+)(\s*,)'
        nuevo, n = re.subn(patron, rf"\g<1>{nivel}\g<3>", texto,
                           count=1, flags=re.MULTILINE)
        if n != 1:
            raise SystemExit(f"no encontré la línea de «{slug}» en config.py")
        texto = nuevo
    CONFIG_PY.write_text(texto, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--salidas", default=None,
                    help="carpeta de corridas; se toman los niveles del criterio")
    ap.add_argument("--nivel", action="append", default=[],
                    metavar="slug=N", help="fijar un nivel a mano")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    nuevos = niveles_desde_estudio(args.salidas) if args.salidas else {}
    for par in args.nivel:
        slug, _, n = par.partition("=")
        nuevos[slug] = int(n)
    if not nuevos:
        raise SystemExit("pasá --salidas o --nivel slug=N")
    if set(nuevos) - set(SLUGS):
        raise SystemExit(f"preguntas desconocidas: {sorted(set(nuevos) - set(SLUGS))}")

    modelos = {s: json.loads(ruta_modelo(s).read_text(encoding="utf-8"))
               for s in SLUGS}

    problemas = verificar_que_solo_cambia_el_nivel(modelos)
    if problemas:
        print("NO se puede re-sellar:")
        for p in problemas:
            print(f"  · {p}")
        raise SystemExit(1)
    print("verificado: entre los JSON y la configuración sólo difiere el nivel\n")

    print(f"{'pregunta':<20} {'antes':>6} {'ahora':>6}")
    for slug in SLUGS:
        n = nuevos.get(slug, NIVEL_CALIBRADO[slug])
        marca = "" if n == NIVEL_CALIBRADO[slug] else "   <-- cambia"
        print(f"{slug:<20} {NIVEL_CALIBRADO[slug]:>6} {n:>6}{marca}")
    if args.dry_run:
        print("\n--dry-run: no se escribió nada")
        return

    completos = {s: nuevos.get(s, NIVEL_CALIBRADO[s]) for s in SLUGS}
    escribir_config(completos)

    # Las huellas nuevas se calculan con los niveles nuevos ya puestos.
    for slug, nivel in completos.items():
        NIVEL_CALIBRADO[slug] = nivel
    huellas = {s: huella_contrato(s) for s in SLUGS}

    for slug in SLUGS:
        m = modelos[slug]
        m["nivel_calibrado"] = completos[slug]
        m["contrato"] = huellas[slug]
        ruta_modelo(slug).write_text(
            json.dumps(m, ensure_ascii=False), encoding="utf-8")

    if RUTA_ENVOLVENTE.exists():
        env = json.loads(RUTA_ENVOLVENTE.read_text(encoding="utf-8"))
        for slug, bloque in env.get("preguntas", {}).items():
            bloque["contrato"] = huellas[slug]
        RUTA_ENVOLVENTE.write_text(
            json.dumps(env, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nre-sellados: 4 modelos + {RUTA_ENVOLVENTE.name}")
    else:
        print("\nre-sellados: 4 modelos (no hay envolvente)")
    print("Verificá con los chequeos de arranque antes de commitear.")


if __name__ == "__main__":
    main()

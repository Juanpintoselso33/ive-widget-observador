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

Se recalculan las huellas SUSTITUYENDO EL DICCIONARIO ENTERO de niveles por el
que traen los propios JSON. Si esas huellas reproducen exactamente las que los
JSON tienen, entonces lo único que cambió entre los modelos entrenados y la
configuración actual es el nivel, y re-sellar es legítimo. Si no, cambió algo
más —un mapeo, una categoría, el colapso educativo— y el script ABORTA pidiendo
un reentrenamiento, que es lo correcto.

La envolvente pasa por la MISMA prueba antes de que se le toque el contrato:
que los modelos estén al día no dice nada sobre su procedencia, y re-sellarla a
ciegas borraría el chequeo de arranque que la habría rechazado.

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


def huellas_historicas(modelos):
    """
    Las huellas que corresponderían a los niveles que traen los propios JSON.

    SE SUSTITUYE EL DICCIONARIO ENTERO, no la entrada de una pregunta.
    `huella_contrato` hashea TODO `NIVEL_CALIBRADO`, así que reponer sólo el
    nivel de la pregunta que se está mirando deja las otras tres con el valor
    nuevo y la huella no reproduce la histórica. La versión anterior hacía eso:
    funcionaba de casualidad mientras la configuración todavía coincidía con los
    JSON, y rechazaba las cuatro preguntas en cuanto alguien editaba config.py
    antes de correr el script — mandándolo a un reentrenamiento innecesario.
    Lo marcó Codex.
    """
    guardado = dict(NIVEL_CALIBRADO)
    NIVEL_CALIBRADO.update({s: m["nivel_calibrado"] for s, m in modelos.items()
                            if m.get("nivel_calibrado") is not None})
    try:
        return {s: huella_contrato(s) for s in modelos}
    finally:
        NIVEL_CALIBRADO.clear()
        NIVEL_CALIBRADO.update(guardado)


def niveles_desde_estudio(salidas):
    """Los niveles que elige el criterio sobre las corridas de `salidas`."""
    sys.path.insert(0, str(Path(__file__).parent))
    from agregar_calibracion import _cargar, elegir
    return {slug: int(elegir(corridas)[0])
            for slug, corridas in _cargar(salidas).items()}


def verificar_que_solo_cambia_el_nivel(modelos, envolvente=None):
    """
    El control que hace legítimo el re-sellado. Devuelve (problemas, huellas).

    Verifica los cuatro JSON y, si hay envolvente, TAMBIÉN su contrato. Que los
    modelos estén al día no dice nada sobre la procedencia de la envolvente: si
    quedó de una codificación anterior, re-sellarla a ciegas borraría el chequeo
    de arranque que la habría rechazado. Lo marcó Codex.
    """
    problemas = []
    faltan = [s for s, m in modelos.items() if m.get("nivel_calibrado") is None]
    for s in faltan:
        problemas.append(f"«{s}»: el JSON no trae nivel_calibrado")
    if faltan:
        return problemas, {}

    huellas = huellas_historicas(modelos)
    for slug, m in modelos.items():
        if m.get("contrato") != huellas[slug]:
            problemas.append(
                f"«{slug}»: con los niveles que traen los JSON la huella da "
                f"{huellas[slug]} y el JSON dice {m.get('contrato')} — cambió "
                "algo MÁS que el nivel, hay que reentrenar")

    for slug, bloque in (envolvente or {}).get("preguntas", {}).items():
        if slug not in huellas:
            problemas.append(f"«{slug}»: la envolvente trae una pregunta que no existe")
        elif bloque.get("contrato") != huellas[slug]:
            problemas.append(
                f"«{slug}»: la envolvente dice {bloque.get('contrato')} y la "
                f"histórica es {huellas[slug]} — la envolvente NO es de esta "
                "codificación, regenerala con agregar_envolvente.py")
    return problemas, huellas


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

    env = (json.loads(RUTA_ENVOLVENTE.read_text(encoding="utf-8"))
           if RUTA_ENVOLVENTE.exists() else None)
    problemas, _ = verificar_que_solo_cambia_el_nivel(modelos, env)
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

    if env is not None:
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

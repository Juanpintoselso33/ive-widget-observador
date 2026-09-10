"""
Progreso de una corrida de `cobertura_simulada.py`, leído de sus parciales.

No toca los procesos: lee los `cal-*.json` que el simulador va guardando con
`--cada N` y muestra cuánto lleva cada uno y cuánto le falta. Para verlo en
vivo:

    watch -n 30 .venv/bin/python widgets/seguridad/scripts/progreso_cobertura.py

El tiempo por simulación sale de `segundos / sims_hechas` del propio parcial.
Ojo: `segundos` se cuenta desde el último ARRANQUE, así que después de reanudar
la estimación es la de ese tramo, no la de toda la corrida. Es la más honesta que
hay sin agregar contabilidad al simulador.
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from widgets.seguridad import config

BARRA = 24


def _fmt(seg):
    if seg is None:
        return "   —  "
    seg = int(seg)
    if seg >= 3600:
        return f"{seg // 3600}h{(seg % 3600) // 60:02d}m"
    return f"{seg // 60}m{seg % 60:02d}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--salidas", default=str(
        Path(__file__).parent / "salidas-b10000"))
    ap.add_argument("--semillas", default="601,602")
    args = ap.parse_args()

    semillas = [int(x) for x in args.semillas.split(",")]
    archivos = {}
    for ruta in glob.glob(os.path.join(args.salidas, "cal-*.json")):
        try:
            j = json.loads(Path(ruta).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # lo estaban escribiendo justo ahora
        archivos[(j["slug"], j["semilla"])] = (j, os.path.getmtime(ruta))

    ahora = time.time()
    print(f"{'pregunta':<20} {'sem':>4}  {'progreso':<{BARRA + 9}} "
          f"{'por sim':>8} {'falta':>7}  {'último parcial':>15}")
    total_hechas = total_obj = 0
    for slug in config.SLUGS:
        for sem in semillas:
            reg = archivos.get((slug, sem))
            if reg is None:
                print(f"{slug:<20} {sem:>4}  {'·' * BARRA}   —          —       —    pendiente")
                total_obj += 100
                continue
            j, mtime = reg
            hechas = j.get("sims_hechas", j.get("sims_validas", 0))
            obj = j.get("sims_objetivo", 100)
            total_hechas += hechas
            total_obj += obj
            llenas = int(BARRA * hechas / max(obj, 1))
            barra = "█" * llenas + "·" * (BARRA - llenas)
            por_sim = (j["segundos"] / hechas) if hechas else None
            falta = (por_sim * (obj - hechas)) if por_sim else None
            estado = ("COMPLETA" if not j.get("parcial")
                      else f"hace {_fmt(ahora - mtime)}")
            print(f"{slug:<20} {sem:>4}  {barra} {hechas:>3}/{obj:<3} "
                  f"{_fmt(por_sim):>8} {_fmt(falta):>7}  {estado:>15}")

    pct = 100 * total_hechas / max(total_obj, 1)
    print(f"\ntotal: {total_hechas}/{total_obj} simulaciones ({pct:.0f}%)")


if __name__ == "__main__":
    main()

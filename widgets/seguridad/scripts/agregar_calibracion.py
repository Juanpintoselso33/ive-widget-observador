"""
Lee las salidas de `cobertura_simulada.py` y elige el nivel de cada pregunta.

Existe para que el número que va a `config.NIVEL_CALIBRADO` salga de un
procedimiento escrito y no de mirar los logs. El criterio es explícito y
deliberadamente más duro que el promedio:

    el nivel más chico cuya cobertura llega al 95% en TODAS las corridas por
    separado, no sólo en el promedio de todas.

Un promedio de dos corridas que se apoya en una corrida buena no es una
garantía: en mano dura el nivel 96 promedia 95,10% con una corrida en 94,53%, y
en cadena perpetua el 98 promedia 95,19% con una en 94,86%. Los dos pasarían el
criterio del promedio y ninguno de los dos debería.

Uso:
    python widgets/seguridad/scripts/agregar_calibracion.py
"""

import collections
import glob
import json
import os
import statistics

OBJETIVO = 95.0
AQUI = os.path.dirname(os.path.abspath(__file__))
SALIDAS = os.path.join(AQUI, "salidas")


def _cargar():
    por_slug = collections.defaultdict(list)
    for ruta in sorted(glob.glob(os.path.join(SALIDAS, "cal-*.json"))):
        with open(ruta, encoding="utf-8") as f:
            j = json.load(f)
        por_slug[j["slug"]].append(j)
    return por_slug


def _cobertura_por_perfil(corridas, nivel):
    """Cobertura de cada perfil agregando corridas, ponderada por sims."""
    sims = sum(c["sims_validas"] for c in corridas)
    n = len(corridas[0]["niveles"][nivel])
    return [
        100.0 * sum(c["niveles"][nivel][i] / 100.0 * c["sims_validas"]
                    for c in corridas) / sims
        for i in range(n)
    ]


def elegir(corridas):
    """Devuelve (nivel, detalle) aplicando el criterio de todas-las-corridas."""
    detalle = []
    elegido = None
    for nivel in sorted(corridas[0]["niveles"], key=float):
        por_semilla = [statistics.mean(c["niveles"][nivel]) for c in corridas]
        comb = _cobertura_por_perfil(corridas, nivel)
        fila = {
            "nivel": nivel,
            "media": statistics.mean(comb),
            "por_semilla": por_semilla,
            "peor_perfil": min(comb),
            "bajo_90": sum(1 for c in comb if c < 90),
            "bajo_95": sum(1 for c in comb if c < OBJETIVO),
        }
        if elegido is None and all(m >= OBJETIVO for m in por_semilla):
            elegido = nivel
            fila["elegido"] = True
        detalle.append(fila)
    return elegido, detalle


def main():
    por_slug = _cargar()
    if not por_slug:
        print(f"No hay salidas en {SALIDAS}. Corré cobertura_simulada.py primero.")
        return
    for slug, corridas in sorted(por_slug.items()):
        sims = sum(c["sims_validas"] for c in corridas)
        reps = corridas[0]["replicas"]
        print(f"\n{slug}  ({sims} sims en {len(corridas)} corridas, B={reps})")
        print(f"  {'niv':>4} {'media':>8} {'por semilla':>22} "
              f"{'peor':>6} {'<90%':>5} {'<95%':>5}")
        elegido, detalle = elegir(corridas)
        for f in detalle:
            semillas = "  ".join(f"{m:.2f}" for m in f["por_semilla"])
            marca = "  <-- elegido" if f.get("elegido") else ""
            print(f"  {f['nivel']:>4} {f['media']:>7.2f}% {semillas:>22} "
                  f"{f['peor_perfil']:>5.1f} {f['bajo_90']:>5} "
                  f"{f['bajo_95']:>5}{marca}")
        print(f"  criterio (todas las corridas >= {OBJETIVO}%): "
              f"nivel {elegido or 'ninguno de los probados'}")


if __name__ == "__main__":
    main()

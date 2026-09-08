"""
Lee las salidas de `cobertura_simulada.py` y elige el nivel de cada pregunta.

Existe para que el número que va a `config.NIVEL_CALIBRADO` salga de un
procedimiento escrito y no de mirar los logs. El criterio:

    el nivel más chico cuya cobertura llega al 95% en TODAS las corridas por
    separado, no sólo en el promedio de todas.

QUÉ ES Y QUÉ NO ES ESE CRITERIO. Es un desempate conservador, no un test. Codex
lo midió al revisar esto (8/9/2026): con la cobertura verdadera justo en 95%,
dos corridas independientes caen las dos por encima cerca del 25% de las veces,
así que pasar el criterio no es una garantía al 95%. Y el error Monte Carlo de
estas corridas es del orden de un punto — los cuatro cortes elegidos tienen al
95% dentro de su margen. El criterio sirve para no elegir el nivel mirando un
promedio que se apoya en una sola corrida buena (mano dura promedia 95,10% en
el nivel 96 con una corrida en 94,53%; cadena perpetua 95,19% en el 98 con una
en 94,86%), no para afirmar que el nivel elegido cubre.

EL FORMATO DE LAS SALIDAS TIENE UNA TRAMPA. `cobertura_simulada.py` guarda en
`niveles` la CANTIDAD DE ACIERTOS de cada perfil, no un porcentaje. Coinciden
numéricamente cuando la corrida tiene exactamente 100 simulaciones, que es el
caso de las ocho salidas actuales, y por eso la primera versión de este script
—que los trataba como porcentajes— daba los números correctos por casualidad.
Con `--sims 200` habría elegido el nivel 95 para las cuatro preguntas. Lo
encontró Codex; acá se normaliza explícitamente por `sims_validas`.

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
    """
    Cobertura % de cada perfil sumando aciertos y simulaciones de las corridas.

    `niveles` trae ACIERTOS, no porcentajes: la división va acá, una sola vez.
    """
    sims = sum(c["sims_validas"] for c in corridas)
    n = len(corridas[0]["niveles"][nivel])
    return [
        100.0 * sum(c["niveles"][nivel][i] for c in corridas) / sims
        for i in range(n)
    ]


def _cobertura_de_una_corrida(corrida, nivel):
    """Cobertura % promedio sobre los perfiles, dentro de una sola corrida."""
    aciertos = corrida["niveles"][nivel]
    return 100.0 * statistics.mean(aciertos) / corrida["sims_validas"]


def elegir(corridas):
    """Devuelve (nivel, detalle) aplicando el criterio de todas-las-corridas."""
    detalle = []
    elegido = None
    for nivel in sorted(corridas[0]["niveles"], key=float):
        por_semilla = [_cobertura_de_una_corrida(c, nivel) for c in corridas]
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

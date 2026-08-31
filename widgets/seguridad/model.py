"""
Lógica de predicción del widget de seguridad.

Python puro (sin Streamlit ni sklearn) para que sea testeable y liviano en
producción: lee el JSON de coeficientes y evalúa la logística.

A diferencia de widgets/ive/model.py, que suma los términos uno por uno, acá
el vector de features se arma en un dict y la suma se hace iterando sobre los
predictores declarados en config.PREDICTORES. Agregar una variable al modelo
es tocar build_features() y la lista, no la aritmética.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import math

from widgets.seguridad.config import (
    MODEL_COEFFICIENTS_PATH, PREDICTORES, ESPEC_CRUDA, IDEOLOGIA_UI_TO_CODE,
)


def load_model():
    """Carga los coeficientes del modelo desde JSON."""
    with open(MODEL_COEFFICIENTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_features(tramo_edad, es_mujer, nivel_educ, ideologia, victima,
                   es_montevideo):
    """
    Traduce los inputs de la UI (códigos de config.py) al vector de dummies.

    Cada bloque omite su categoría de referencia: 18-29 en edad, hombre en
    sexo, secundaria o menos en educación, centroizquierda (el 5 de la escala)
    en ideología, no víctima e interior en región.

    Las dummies ideológicas se arman recorriendo IDEOLOGIA_UI_TO_CODE y no
    escribiendo `ideologia == 1` a mano para cada tramo: con seis tramos, un
    índice mal escrito daría un vector válido con la categoría equivocada, que
    es el error que produce un número plausible y falso.
    """
    features = {
        "edad_30_44": int(tramo_edad == 2),
        "edad_45_59": int(tramo_edad == 3),
        "edad_60_plus": int(tramo_edad == 4),
        "es_mujer": int(es_mujer),
        "educ_ter_incomp": int(nivel_educ == 2),
        "educ_ter_comp": int(nivel_educ == 3),
        "victima_sin_violencia": int(victima == 2),
        "victima_con_violencia": int(victima == 3),
        # Siempre 0: la UI obliga a elegir una de las tres opciones reales.
        # El coeficiente existe para que los sin dato del entrenamiento no
        # contaminen la categoría de referencia (ver config.PREDICTORES).
        "victima_sin_dato": 0,
        "es_montevideo": int(es_montevideo),
        # Siempre 0 desde la UI, igual que victima_sin_dato: "no se ubica"
        # agrupa a quienes no contestaron la escala, que no es una posición
        # política que el lector pueda elegir.
        "ideol_no_ubica": 0,
    }
    # El código que manda la UI es el índice del tramo dentro de
    # ESPEC_CRUDA["ideol_tramos"] (1-based), porque IDEOLOGIA_UI_TO_CODE se
    # deriva de esa misma lista. No hay dos estructuras que puedan
    # desalinearse.
    referencia = ESPEC_CRUDA["ideol_referencia"]
    for codigo, (nombre, _, _, _) in enumerate(ESPEC_CRUDA["ideol_tramos"], start=1):
        if nombre != referencia:
            features[f"ideol_{nombre}"] = int(ideologia == codigo)
    return features


def _z(coef, features):
    """Suma el intercepto más los términos declarados en PREDICTORES."""
    z = coef["intercept"]
    for nombre in PREDICTORES:
        z += coef[nombre] * features[nombre]
    return z


def _sigmoid_pct(z):
    return (1 / (1 + math.exp(-z))) * 100


def predict_probability(model, tramo_edad, es_mujer, nivel_educ, ideologia,
                        victima, es_montevideo):
    """
    Probabilidad de estar a favor, condicional a tener postura definida.
    Returns: float en 0-100.
    """
    features = build_features(tramo_edad, es_mujer, nivel_educ, ideologia,
                              victima, es_montevideo)
    return _sigmoid_pct(_z(model["coefficients"], features))


def predict_probability_neutral(model, tramo_edad, es_mujer, nivel_educ, ideologia,
                                victima, es_montevideo):
    """
    Probabilidad de no fijar postura (Likert=3 o sin respuesta) según el perfil.
    Returns: float en 0-100.
    """
    features = build_features(tramo_edad, es_mujer, nivel_educ, ideologia,
                              victima, es_montevideo)
    return _sigmoid_pct(_z(model["coefficients_neutral"], features))


def _probabilidades_bootstrap(model, tramo_edad, es_mujer, nivel_educ, ideologia,
                              victima, es_montevideo):
    """
    Las B probabilidades del perfil, una por réplica bootstrap, ordenadas.

    Devuelve None si el modelo no trae bootstrap.
    """
    boot = model.get("bootstrap")
    if not boot or not boot.get("replicas"):
        return None

    features = build_features(tramo_edad, es_mujer, nivel_educ, ideologia,
                              victima, es_montevideo)
    orden = boot["orden"]

    probabilidades = []
    for fila in boot["replicas"]:
        z = fila[0]  # intercept
        for nombre, coef in zip(orden[1:], fila[1:]):
            z += coef * features[nombre]
        probabilidades.append(_sigmoid_pct(z))

    probabilidades.sort()
    return probabilidades


def intervalo_probabilidad(model, tramo_edad, es_mujer, nivel_educ, ideologia,
                           victima, es_montevideo, nivel=95):
    """
    Intervalo de confianza percentil para la probabilidad estimada. ES EL QUE SE
    MUESTRA: la decisión editorial sobre el 50% no se toma con éste, sino con
    banda_decision() — ver el docstring de esa función.

    Se calcula sobre las réplicas bootstrap guardadas en el JSON: para cada una
    se evalúa la logística con el mismo vector de features y se toman los
    percentiles. Sin sklearn ni numpy — es aritmética sobre una lista.

    Devuelve (bajo, alto) en 0-100, o None si el modelo no trae bootstrap.
    """
    probabilidades = _probabilidades_bootstrap(
        model, tramo_edad, es_mujer, nivel_educ, ideologia, victima,
        es_montevideo)
    if probabilidades is None:
        return None
    cola = (100 - nivel) / 2 / 100
    return _percentil(probabilidades, cola), _percentil(probabilidades, 1 - cola)


# 1,96: el z de una banda del 95% para la posición del percentil.
_Z_MC = 1.959964


def banda_decision(model, tramo_edad, es_mujer, nivel_educ, ideologia, victima,
                   es_montevideo, nivel=95):
    """
    Extremos CONSERVADORES del intervalo, para decidir si se afirma de qué lado
    está la mayoría. No se muestran: sólo gobiernan esa decisión.

    POR QUÉ EXISTE. El extremo del intervalo no se conoce, se SIMULA con B
    réplicas, y esa simulación tiene su propio error. Eso es irrelevante para
    mostrar "15% a 49%", pero es decisivo para una regla binaria que compara ese
    extremo contra 50: un perfil cuyo extremo verdadero está en 49,5 cae de un
    lado o del otro según la semilla.

    Codex lo midió sobre el artefacto anterior remuestreando las réplicas
    guardadas: 54 de los perfiles cambiaban de conclusión en al menos 10% de las
    corridas simuladas. Subir B no lo arregla, sólo lo achica.

    CÓMO. La POSICIÓN del cuantil q dentro de B réplicas ordenadas sigue una
    binomial B(B, q), cuyo desvío expresado en escala de cuantil es
    sqrt(q(1-q)/B). Se corren los dos extremos hacia afuera 1,96 de esos desvíos
    y se toma el percentil resultante. Ojo con la distinción: eso mide la
    incertidumbre de QUÉ POSICIÓN del orden estadístico corresponde al cuantil,
    no la del VALOR de ese cuantil, que depende además de cuán apretadas estén
    las réplicas ahí. Con q=0,975 y B=1.000 el cuantil exterior es 0,98468, y en
    esa cola hay unas 25 observaciones esperadas: la normal está en zona
    marginal, pero no se rompe.

    QUÉ TAN BIEN FUNCIONA, medido por Codex sobre este modelo remuestreando
    1.000 veces las réplicas guardadas:

      - cobertura conjunta de los dos extremos: 95,54% (el objetivo es 95%),
        con un rango entre perfiles de 93,4% a 97,4%. O sea LIGERAMENTE
        CONSERVADORA, que es el lado correcto para errar.
      - afirmaciones del lado equivocado: 135 sobre 1.008.000 decisiones, o
        0,0134%.
      - costo: 3,77% de afirmaciones correctas que se suprimen y salen como
        "no se puede afirmar".

    QUÉ NO ARREGLA, y conviene tenerlo presente antes de titular: la banda
    controla muy bien el riesgo de afirmar del lado equivocado, pero NO vuelve
    la conclusión independiente del error Monte Carlo — mueve la frontera
    aleatoria del cuantil 97,5 a su límite superior, y siempre queda algún
    perfil cerca de la frontera nueva. Sobre los 1.008 perfiles actuales, 124
    cambian de conclusión alguna vez entre pseudo-corridas y 26 lo hacen en al
    menos el 25%. El más inestable —hombre de 30-44, terciaria completa,
    centroizquierda, víctima con violencia, de Montevideo— sale 498 a 502 entre
    "la mayoría está en contra" y "no se puede afirmar", y el artefacto publica
    la prudente.

    Devuelve (bajo, alto) en 0-100, o None si el modelo no trae bootstrap.
    """
    probabilidades = _probabilidades_bootstrap(
        model, tramo_edad, es_mujer, nivel_educ, ideologia, victima,
        es_montevideo)
    if probabilidades is None:
        return None

    b = len(probabilidades)
    cola = (100 - nivel) / 2 / 100
    q_bajo, q_alto = cola, 1 - cola
    holgura_bajo = _Z_MC * math.sqrt(q_bajo * (1 - q_bajo) / b)
    holgura_alto = _Z_MC * math.sqrt(q_alto * (1 - q_alto) / b)
    return (_percentil(probabilidades, max(0.0, q_bajo - holgura_bajo)),
            _percentil(probabilidades, min(1.0, q_alto + holgura_alto)))


def _percentil(ordenados, q):
    """
    Percentil con interpolación lineal (el "tipo 7", que es el que usan numpy y
    R por defecto).

    La versión anterior hacía `ordenados[int(q * n)]`, que con 400 réplicas y
    q=0,025 devuelve la posición 11 en vez de interpolar alrededor de la 10,975:
    corre los dos extremos hacia arriba y deja las colas asimétricas. Medido
    cuando se corrigió —sobre el modelo de entonces, que tenía 1.296 perfiles—
    movía algún extremo redondeado en 358 de ellos, hasta 2,5 puntos, y
    cambiaba la decisión sobre el 50% en 8.
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"q tiene que estar entre 0 y 1, llegó {q}")
    if not ordenados:
        return None
    if len(ordenados) == 1:
        return ordenados[0]
    pos = q * (len(ordenados) - 1)
    bajo = int(pos)
    alto = min(bajo + 1, len(ordenados) - 1)
    peso = pos - bajo
    return ordenados[bajo] * (1 - peso) + ordenados[alto] * peso

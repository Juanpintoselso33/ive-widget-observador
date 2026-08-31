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

from widgets.seguridad.config import MODEL_COEFFICIENTS_PATH, PREDICTORES


def load_model():
    """Carga los coeficientes del modelo desde JSON."""
    with open(MODEL_COEFFICIENTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_features(tramo_edad, es_mujer, nivel_educ, ideologia, victima,
                   es_montevideo, balotaje):
    """
    Traduce los inputs de la UI (códigos de config.py) al vector de dummies.

    Cada bloque omite su categoría de referencia: 18-29 en edad, hombre en
    sexo, secundaria o menos en educación, centro en ideología, no víctima, e
    interior en región, y blanco/no votó en balotaje.
    """
    return {
        "edad_30_44": int(tramo_edad == 2),
        "edad_45_59": int(tramo_edad == 3),
        "edad_60_plus": int(tramo_edad == 4),
        "es_mujer": int(es_mujer),
        "educ_ter_incomp": int(nivel_educ == 2),
        "educ_ter_comp": int(nivel_educ == 3),
        "ideol_izquierda": int(ideologia == 1),
        "ideol_derecha": int(ideologia == 3),
        "ideol_no_ubica": int(ideologia == 4),
        "victima_sin_violencia": int(victima == 2),
        "victima_con_violencia": int(victima == 3),
        # Siempre 0: la UI obliga a elegir una de las tres opciones reales.
        # El coeficiente existe para que los sin dato del entrenamiento no
        # contaminen la categoría de referencia (ver config.PREDICTORES).
        "victima_sin_dato": 0,
        "es_montevideo": int(es_montevideo),
        "bal_orsi": int(balotaje == 1),
        "bal_delgado": int(balotaje == 2),
        # Siempre 0 desde la UI, igual que las otras dummies de "sin dato".
        "bal_no_recuerda": 0,
    }


def _z(coef, features):
    """Suma el intercepto más los términos declarados en PREDICTORES."""
    z = coef["intercept"]
    for nombre in PREDICTORES:
        z += coef[nombre] * features[nombre]
    return z


def _sigmoid_pct(z):
    return (1 / (1 + math.exp(-z))) * 100


def predict_probability(model, tramo_edad, es_mujer, nivel_educ, ideologia,
                        victima, es_montevideo, balotaje):
    """
    Probabilidad de estar a favor, condicional a tener postura definida.
    Returns: float en 0-100.
    """
    features = build_features(tramo_edad, es_mujer, nivel_educ, ideologia,
                              victima, es_montevideo, balotaje)
    return _sigmoid_pct(_z(model["coefficients"], features))


def predict_probability_neutral(model, tramo_edad, es_mujer, nivel_educ, ideologia,
                                victima, es_montevideo, balotaje):
    """
    Probabilidad de no fijar postura (Likert=3 o sin respuesta) según el perfil.
    Returns: float en 0-100.
    """
    features = build_features(tramo_edad, es_mujer, nivel_educ, ideologia,
                              victima, es_montevideo, balotaje)
    return _sigmoid_pct(_z(model["coefficients_neutral"], features))


def _probabilidades_bootstrap(model, tramo_edad, es_mujer, nivel_educ, ideologia,
                              victima, es_montevideo, balotaje):
    """
    Las B probabilidades del perfil, una por réplica bootstrap, ordenadas.

    Devuelve None si el modelo no trae bootstrap.
    """
    boot = model.get("bootstrap")
    if not boot or not boot.get("replicas"):
        return None

    features = build_features(tramo_edad, es_mujer, nivel_educ, ideologia,
                              victima, es_montevideo, balotaje)
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
                           victima, es_montevideo, balotaje, nivel=95):
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
        es_montevideo, balotaje)
    if probabilidades is None:
        return None
    cola = (100 - nivel) / 2 / 100
    return _percentil(probabilidades, cola), _percentil(probabilidades, 1 - cola)


# 1,96: el z de una banda del 95% para la posición del percentil.
_Z_MC = 1.959964


def banda_decision(model, tramo_edad, es_mujer, nivel_educ, ideologia, victima,
                   es_montevideo, balotaje, nivel=95):
    """
    Extremos CONSERVADORES del intervalo, para decidir si se afirma de qué lado
    está la mayoría. No se muestran: sólo gobiernan esa decisión.

    Por qué existe. El extremo del intervalo no se conoce, se SIMULA con B
    réplicas, y esa simulación tiene su propio error. Con B=1.000 el percentil
    97,5 de un perfil cualquiera se mueve alrededor de un punto y medio de una
    corrida a otra. Eso es irrelevante para mostrar "15% a 49%", pero es
    decisivo para una regla binaria que compara ese extremo contra 50: un perfil
    cuyo extremo verdadero está en 49,5 cae de un lado o del otro según la
    semilla.

    Codex lo midió sobre este mismo artefacto, remuestreando las réplicas
    guardadas: 54 de los 1.296 perfiles cambian de conclusión en al menos 10% de
    las corridas simuladas, y 30 en al menos 25%. El caso testigo —mujer de
    18-29, terciaria incompleta, de izquierda, víctima sin violencia, del
    interior, voto en blanco— se muestra hoy como 15%-49% y el widget afirma
    "la amplia mayoría está en contra"; en 456 de 1.000 corridas ese extremo
    llega a 50 y el texto correcto habría sido el prudente.

    Subir B no arregla esto, sólo lo achica: siempre hay perfiles cuyo extremo
    verdadero está lo bastante cerca de 50 como para que la simulación no pueda
    resolver el lado. Lo que corresponde es no exigirle a la simulación una
    precisión que no tiene.

    Cómo. La posición del percentil q dentro de B réplicas sigue una binomial
    B(B, q), así que su desvío en escala de cuantil es sqrt(q(1-q)/B). Se corren
    los dos extremos hacia afuera ese desvío por 1,96 y se toma el percentil
    resultante. Afirmar mayoría requiere entonces que TODA la banda quede de un
    lado, no sólo el percentil puntual.

    Devuelve (bajo, alto) en 0-100, o None si el modelo no trae bootstrap.
    """
    probabilidades = _probabilidades_bootstrap(
        model, tramo_edad, es_mujer, nivel_educ, ideologia, victima,
        es_montevideo, balotaje)
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
    corre los dos extremos hacia arriba y deja las colas asimétricas. Sobre los
    1.296 perfiles eso movía algún extremo redondeado en 358 de ellos, hasta 2,5
    puntos, y cambiaba la decisión sobre el 50% en 8.
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

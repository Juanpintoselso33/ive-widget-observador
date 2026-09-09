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
    PREDICTORES, ESPEC_CRUDA, SLUGS, ruta_modelo, PREGUNTAS_A_RECALIBRAR,
)


def load_model(slug):
    """Carga los coeficientes de una pregunta desde su JSON."""
    with open(ruta_modelo(slug), "r", encoding="utf-8") as f:
        return json.load(f)


def load_modelos():
    """
    Los cuatro modelos, indexados por slug.

    Se cargan los cuatro al arrancar y no bajo demanda. Pesan unos 330 KB cada
    uno —casi todo son las 1.000 réplicas bootstrap—, o sea 1,3 MB que quedan
    cacheados una sola vez por sesión; a cambio, si falta un JSON o está
    desalineado, la página lo dice al abrir y no cuando el lector elige esa
    pregunta y ya está leyendo un número.
    """
    return {slug: load_model(slug) for slug in SLUGS}


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


def problemas_de_calibracion(slug, model):
    """
    Verifica que el mapa de recalibración sea el que corresponde y esté sano.

    POR QUÉ NO ALCANZA LA HUELLA DEL CONTRATO. La huella cubre qué preguntas se
    declaran recalibradas, pero no el CONTENIDO del JSON: Codex sacó el mapa de
    una copia en memoria conservando la huella, y pasaba todos los controles de
    arranque — `_calibrar` devolvía la probabilidad cruda en silencio, o sea el
    widget publicando sin recalibrar una pregunta que se declaró que lo necesita.
    Al revés también: un mapa pegado a otra pregunta se habría aplicado igual.

    Devuelve una lista de problemas, vacía si está todo bien.
    """
    problemas = []

    # EL APAREAMIENTO DE LA TASA NACIONAL, chequeado al arrancar y no sólo en
    # los tests. `intervalo_brecha()` devuelve None si los largos no coinciden
    # —se cae al chequeo viejo en vez de restar contra la réplica equivocada—,
    # pero eso pasa en silencio: sin este aviso, el widget publicaría con la
    # regla vieja durante meses sin que nadie se entere. Lo marcó Codex.
    boot = model.get("bootstrap") or {}
    nac = boot.get("nacional")
    if nac is not None and len(nac) != len(boot.get("replicas") or []):
        problemas.append(
            f"«{slug}»: la tasa nacional por réplica tiene {len(nac)} valores "
            f"y hay {len(boot.get('replicas') or [])} réplicas de coeficientes "
            "— no están apareadas y la brecha se calcularía contra la réplica "
            "equivocada"
        )

    cal = model.get("calibracion")
    debe_tener = slug in PREGUNTAS_A_RECALIBRAR

    if debe_tener and not cal:
        return problemas + [
            f"«{slug}» está declarada en PREGUNTAS_A_RECALIBRAR y su JSON no "
            "trae mapa: se publicaría sin recalibrar"]
    if cal and not debe_tener:
        return problemas + [
            f"«{slug}» trae un mapa de recalibración y no está declarada: se "
            "aplicaría una corrección que nadie pidió"]
    if not cal:
        return problemas

    fallas = list(problemas)
    xs, ys = cal.get("grilla"), cal.get("valores")
    if not xs or not ys or len(xs) != len(ys):
        return fallas + [
            f"«{slug}»: la grilla y los valores no tienen el mismo largo"]
    if len(xs) < 2:
        fallas.append(f"«{slug}»: la grilla tiene menos de dos puntos")
    if any(not math.isfinite(v) for v in xs + ys):
        fallas.append(f"«{slug}»: hay valores no finitos en el mapa")
    if any(b <= a for a, b in zip(xs, xs[1:])):
        fallas.append(f"«{slug}»: las abscisas del mapa no son estrictamente crecientes")
    if any(b < a for a, b in zip(ys, ys[1:])):
        fallas.append(f"«{slug}»: el mapa NO es monótono, puede dar vuelta el orden "
                      "de dos perfiles")
    if any(not (0.0 <= v <= 1.0) for v in ys):
        fallas.append(f"«{slug}»: el mapa devuelve valores fuera de 0-1")
    return fallas


def _calibrar(model, pct):
    """
    Aplica el mapa de recalibración del modelo, si lo trae. Identidad si no.

    El mapa se serializa como una GRILLA de puntos y acá se interpola linealmente
    — no se evalúa la spline. Dos razones: producción no importa scipy, y la
    interpolación lineal de una grilla monótona es monótona, así que no puede
    introducir inversiones que el ajuste no tenía.
    """
    cal = model.get("calibracion")
    if not cal:
        return pct
    xs, ys = cal["grilla"], cal["valores"]
    x = pct / 100.0
    if x <= xs[0]:
        return ys[0] * 100
    if x >= xs[-1]:
        return ys[-1] * 100
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    t = (x - xs[lo]) / (xs[hi] - xs[lo])
    return (ys[lo] + t * (ys[hi] - ys[lo])) * 100


def predict_probability(model, tramo_edad, es_mujer, nivel_educ, ideologia,
                        victima, es_montevideo):
    """
    Probabilidad de estar a favor, condicional a tener postura definida.
    Returns: float en 0-100.
    """
    features = build_features(tramo_edad, es_mujer, nivel_educ, ideologia,
                              victima, es_montevideo)
    return _calibrar(model, _sigmoid_pct(_z(model["coefficients"], features)))


def predict_probability_neutral(model, tramo_edad, es_mujer, nivel_educ, ideologia,
                                victima, es_montevideo):
    """
    Probabilidad de no fijar postura (Likert=3 o sin respuesta) según el perfil.
    Returns: float en 0-100.
    """
    features = build_features(tramo_edad, es_mujer, nivel_educ, ideologia,
                              victima, es_montevideo)
    return _sigmoid_pct(_z(model["coefficients_neutral"], features))


def _interp(xs, ys, x):
    """Interpolación lineal en una grilla o lista de nodos monótona."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    t = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + t * (ys[hi] - ys[lo])


def _probabilidades_bootstrap(model, tramo_edad, es_mujer, nivel_educ, ideologia,
                              victima, es_montevideo, ordenar=True):
    """
    Las B probabilidades del perfil, una por réplica bootstrap.

    `ordenar=False` las devuelve EN EL ORDEN DE LAS RÉPLICAS, que es lo que
    necesita `intervalo_brecha()`: para restarle a cada una la tasa nacional de
    su misma réplica, la posición i tiene que seguir siendo la réplica i.

    Devuelve None si el modelo no trae bootstrap.
    """
    boot = model.get("bootstrap")
    if not boot or not boot.get("replicas"):
        return None

    features = build_features(tramo_edad, es_mujer, nivel_educ, ideologia,
                              victima, es_montevideo)
    orden = boot["orden"]

    cal = model.get("calibracion") or {}
    reps_mapa = cal.get("replicas") or []

    probabilidades = []
    for i, fila in enumerate(boot["replicas"]):
        z = fila[0]  # intercept
        for nombre, coef in zip(orden[1:], fila[1:]):
            z += coef * features[nombre]
        # Cada réplica de coeficientes lleva SU PROPIA réplica del mapa, no el
        # mapa central. Así el intervalo incluye también la incertidumbre de
        # haber estimado la calibración; con el mapa fijo salía 2,5 pp más
        # angosto de lo que corresponde. Si no hay réplicas del mapa —modelo sin
        # recalibrar, o un JSON viejo— se cae al mapa central, que es el
        # comportamiento anterior.
        pct = _sigmoid_pct(z)
        if reps_mapa:
            xs_b, ys_b = reps_mapa[i % len(reps_mapa)]
            probabilidades.append(_interp(xs_b, ys_b, pct / 100.0) * 100)
        else:
            probabilidades.append(_calibrar(model, pct))

    if ordenar:
        probabilidades.sort()
    return probabilidades


def intervalo_probabilidad(model, tramo_edad, es_mujer, nivel_educ, ideologia,
                           victima, es_montevideo, nivel=None):
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
    # El percentil que se pide NO es 95: es el CALIBRADO, el que hace que el
    # intervalo cubra de verdad el 95%. El bootstrap percentil sub-cubre, y
    # cuánto está medido por simulación —ver config.NIVEL_CALIBRADO—. Pedir 95
    # a secas entregaba entre 90,4% y 93,2% según la pregunta.
    # El fallback a 95 es para un JSON viejo, sin el nivel serializado.
    if nivel is None:
        nivel = model.get("nivel_calibrado", 95)
    cola = (100 - nivel) / 2 / 100
    return _percentil(probabilidades, cola), _percentil(probabilidades, 1 - cola)


def intervalo_brecha(model, tramo_edad, es_mujer, nivel_educ, ideologia,
                     victima, es_montevideo, nivel=None):
    """
    Intervalo de la DIFERENCIA entre el perfil y el promedio nacional.

    POR QUÉ EXISTE. El widget afirmaba "este perfil está X pp por encima del
    promedio" comparando el INTERVALO del perfil contra el promedio nacional
    tratado como un PUNTO. Pero el promedio también se estimó con esta misma
    muestra y tiene su propia incertidumbre, así que ese chequeo comparaba una
    cosa con error contra otra cosa con error ignorando el error de la segunda.

    Yo había escrito en el código que eso era "conservador de un solo lado". NO
    ESTABA DEMOSTRADO, y el signo no es obvio: la varianza de la resta es
    Var(perfil) + Var(promedio) − 2·Cov, y de esa covarianza dependía todo. Si
    es grande, el chequeo viejo ensancha de más y es conservador; si es chica o
    negativa, ensancha de menos y AFIRMA DE MÁS — que es exactamente la clase de
    error del que ya se sacaron 1.883 casos. Nadie la había calculado.

    Y NO SIEMPRE ES POSITIVA, aunque suene razonable que lo sea por venir de la
    misma muestra: escribí eso acá y era falso. Codex la calculó perfil por
    perfil y encontró covarianzas NEGATIVAS en cadena perpetua, pena de muerte y
    humillación, de hasta −1,45 pp². Que el signo no se pueda anticipar es
    precisamente el motivo por el que hay que calcular la resta réplica a
    réplica en vez de suponerle una dirección.

    CÓMO SE ARREGLA. `train_model` serializa, junto a cada réplica de
    coeficientes, la tasa nacional DE ESA MISMA RÉPLICA — mismo remuestreo, misma
    posición—. Entonces la diferencia se puede calcular dentro de cada réplica y
    la covarianza entra sola, sin estimarla ni suponerle signo.

    Devuelve (bajo, alto) en puntos porcentuales, o None si el JSON no trae la
    tasa nacional por réplica (artefacto viejo): en ese caso el llamador se cae
    al chequeo anterior, que es lo que había.
    """
    boot = model.get("bootstrap") or {}
    nacional = boot.get("nacional")
    if not nacional:
        return None
    probabilidades = _probabilidades_bootstrap(
        model, tramo_edad, es_mujer, nivel_educ, ideologia, victima,
        es_montevideo, ordenar=False)
    if probabilidades is None:
        return None
    # LARGOS DISTINTOS = ARTEFACTO ROTO, y hay que rechazarlo en vez de
    # truncar. Con `min()` se descartaba el sobrante en silencio y la resta
    # quedaba contra réplicas que no eran las suyas: Codex lo mostró con un
    # ejemplo mínimo donde truncar convierte una abstención en una afirmación.
    # Devolver None hace que el llamador se caiga al chequeo anterior, que es
    # peor pero honesto; y `problemas_de_calibracion` lo reporta al arrancar.
    if len(probabilidades) != len(nacional):
        return None
    diferencias = sorted(pi - ni for pi, ni in zip(probabilidades, nacional))
    if nivel is None:
        nivel = model.get("nivel_calibrado", 95)
    cola = (100 - nivel) / 2 / 100
    return _percentil(diferencias, cola), _percentil(diferencias, 1 - cola)


# 1,96: el z de una banda del 95% para la posición del percentil.
_Z_MC = 1.959964


def banda_decision(model, tramo_edad, es_mujer, nivel_educ, ideologia, victima,
                   es_montevideo, nivel=None):
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

    # Parte del MISMO nivel calibrado que el intervalo mostrado y no del 95
    # nominal: si el mostrado se ensanchó por sub-cobertura, la banda que decide
    # sobre el 50% tiene que partir de ahí, o volvería a ser la más angosta de
    # las dos justo en la comparación que más importa.
    if nivel is None:
        nivel = model.get("nivel_calibrado", 95)
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

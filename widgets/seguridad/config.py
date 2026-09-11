"""
Configuración del widget de seguridad pública.

La pregunta que modela el widget está PARAMETRIZADA: se elige cambiando
`PREGUNTA_ACTIVA` y re-entrenando. Todas las candidatas comparten la misma
escala Likert 1-5, así que el pipeline no cambia al cambiar de pregunta.

Colores y umbrales vienen de shared.config.
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.config import (  # noqa: E402
    LIGHT_COLORS, DARK_COLORS, COLORS,
    get_colors, PROB_THRESHOLDS, get_interpretation,
)

# ============================================================
# RUTAS
# ============================================================
WIDGET_DIR = Path(__file__).parent

# Un JSON por pregunta. Antes había uno solo (`model_coefficients.json`) porque
# el widget publicaba una pregunta por vez; ahora el lector elige entre las
# cuatro y hacen falta los cuatro juegos de coeficientes al mismo tiempo.
MODELOS_DIR = WIDGET_DIR / "modelos"


def ruta_modelo(slug):
    """Ruta del JSON de coeficientes de una pregunta."""
    return MODELOS_DIR / f"model_{slug}.json"


# La envolvente de especificación NO va adentro de los JSON de modelo. Es una
# propiedad de los DATOS y del conjunto de formas funcionales que se probaron,
# no de los coeficientes publicados, y calcularla exige refitear nueve
# especificaciones con validación cruzada anidada: meterla en `train_model`
# multiplicaría por diez lo que tarda entrenar. Va en su propio archivo, lo
# genera `scripts/agregar_envolvente.py`, y `model.problemas_de_envolvente()`
# verifica al arrancar que corresponda a los modelos que se están sirviendo.
RUTA_ENVOLVENTE = MODELOS_DIR / "envolvente_espec.json"

# Que el archivo FALTE tiene que ser un error ruidoso, no una degradación
# silenciosa. Sin él la aritmética sigue funcionando y devuelve el intervalo sin
# ensanchar —que es lo que se publicaba antes—, pero eso es publicar intervalos
# más angostos sin que nada lo diga, que es la clase de error más difícil de
# notar: la pantalla se ve idéntica y los números están mal.
#
# NO se agregó a `huella_contrato` a propósito: la huella está guardada dentro de
# los cuatro JSON entrenados, así que sumarle un campo los invalidaría a todos y
# obligaría a reentrenar con 10.000 réplicas para un cambio que no toca ningún
# coeficiente. El chequeo va aparte, en `model.problemas_de_envolvente()`.
ENVOLVENTE_REQUERIDA = True

# Los seis códigos del perfil, en el orden de PREDICTORES de la UI. Es la clave
# del diccionario de la envolvente. Vive acá y no en cada lado porque si el
# generador y el lector la arman distinto, el lookup falla en silencio y el
# widget publicaría el intervalo sin ensanchar sin que nada avise.
def clave_perfil(tramo_edad, es_mujer, nivel_educ, ideologia, victima,
                 es_montevideo):
    """Clave canónica de un perfil de la UI, para tablas por perfil."""
    return "-".join(str(int(v)) for v in
                    (tramo_edad, es_mujer, nivel_educ, ideologia, victima,
                     es_montevideo))

# La base vive en el repo de encuestas, no en éste (son datos del cliente y
# el .gitignore excluye *.csv). Se puede pisar con la variable de entorno
# SEGURIDAD_DATA_FILE para correr desde otra máquina.
_DEFAULT_DATA = (
    Path.home() / "dev/trabajo/Observador/Observador/Observador-encuesta"
    / "encuestas/observador_2026_05_seguridad/output/base_etiquetada.csv"
)
DATA_FILE = Path(os.environ.get("SEGURIDAD_DATA_FILE", _DEFAULT_DATA))

# ============================================================
# PREGUNTAS CANDIDATAS
# ============================================================
# Las CUATRO que pidió Tomer (WhatsApp, 7/9/2026), en su orden y con las
# columnas que él mismo señaló en la base etiquetada: AC, Z, Y y AA.
#
# El widget publica las cuatro a la vez y el lector elige cuál estimar, así que
# hay un modelo entrenado por pregunta. Antes se publicaba una sola y cambiar
# de pregunta era editar una constante y re-entrenar.
#
# Se cayó `aumentar_penas` (var_228, "Se deberían aumentar las penas para todos
# los delitos"): Tomer no la incluyó en su lista y la columna tampoco viene en
# la base recortada que mandó. Está en la base completa, así que volver a
# sumarla es agregar la entrada acá y re-entrenar.
#
#   columna    : nombre exacto en base_etiquetada.csv
#   enunciado  : el texto TEXTUAL del cuestionario, que se muestra al lector.
#                No es decorativo: el widget mide acuerdo con esa frase exacta,
#                y parafrasearla cambia lo que el numero significa.
#   etiqueta   : el nombre corto con el que la pregunta aparece en el selector
#   titulo     : encabezado del widget
#   titulo_corto: para la pestaña del navegador
#   afirma     : que significa estar de acuerdo, para redactar el resultado
PREGUNTAS = {
    "politico_mano_dura": {
        "columna": "var_233 | Votaria politico de mano dura",
        "enunciado": "Votaría a un político que promoviera castigos más duros para los delincuentes",
        "etiqueta": "Votar a un político de mano dura",
        "titulo": "¿Quiénes votarían a un político de mano dura?",
        "titulo_corto": "¿Votarías mano dura?",
        "afirma": "votar a un político que promueva castigos más duros para los delincuentes",
    },
    "cadena_perpetua": {
        "columna": "var_230 | Cadena perpetua tres delitos",
        "enunciado": ("Una persona que ha sido condenada por tres delitos graves debería "
                      "recibir cadena perpetua sin posibilidad de libertad condicional"),
        "etiqueta": "Cadena perpetua por tres delitos graves",
        "titulo": "¿Quiénes apoyan la cadena perpetua en Uruguay?",
        "titulo_corto": "¿Apoyás la cadena perpetua?",
        "afirma": ("estar de acuerdo con la cadena perpetua sin libertad condicional "
                   "para quien fue condenado por tres delitos graves"),
    },
    "pena_muerte": {
        "columna": "var_229 | Pena de muerte por homicidio",
        "enunciado": "Una persona condenada por homicidio debería recibir la pena de muerte",
        "etiqueta": "Pena de muerte por homicidio",
        "titulo": "¿Quiénes apoyan la pena de muerte en Uruguay?",
        "titulo_corto": "¿Apoyás la pena de muerte?",
        "afirma": "estar de acuerdo con que una persona condenada por homicidio reciba la pena de muerte",
    },
    "humillacion_presos": {
        "columna": "var_231 | Presos merecen humillacion",
        "enunciado": ("Quienes están presos merecen la humillación, intimidación y "
                      "degradación que allí puedan recibir"),
        "etiqueta": "Humillación a los presos",
        "titulo": "¿Quiénes creen que los presos merecen humillación?",
        "titulo_corto": "¿Los presos merecen humillación?",
        "afirma": ("estar de acuerdo con que quienes están presos merecen la humillación, "
                   "intimidación y degradación que allí puedan recibir"),
    },
}

# La que aparece seleccionada al abrir: la primera de la lista de Tomer.
#
# Las cuatro NO son intercambiables como puerta de entrada, y conviene tenerlo
# a la vista. Apoyo ponderado entre quienes tienen postura definida:
# cadena perpetua 78,6% · mano dura 67,0% · pena de muerte 36,7% ·
# humillación a los presos 11,0%. Las dos de los extremos dan perfiles casi
# planos —con 11% de apoyo casi ningún perfil se despega— así que abrir en
# ellas haría parecer que el widget no discrimina. Mano dura reparte mejor y
# además es la que él puso primero.
PREGUNTA_DEFECTO = "politico_mano_dura"

# El orden del selector es el del dict, que es el de la lista de Tomer.
SLUGS = list(PREGUNTAS)

# Etiqueta visible -> slug, para el selector.
ETIQUETA_A_SLUG = {PREGUNTAS[s]["etiqueta"]: s for s in SLUGS}

# ============================================================
# ESCALA LIKERT
# ============================================================
# Mismo criterio que el widget IVE: ≥4 a favor, ≤2 en contra, 3 excluido
# (se modela aparte como "no toma posición").
LIKERT_MAP = {
    "Totalmente en desacuerdo": 1,
    "En desacuerdo": 2,
    "Ni de acuerdo ni en desacuerdo": 3,
    "De acuerdo": 4,
    "Totalmente de acuerdo": 5,
}
LIKERT_FAVOR = (4, 5)
LIKERT_CONTRA = (1, 2)
LIKERT_NEUTRAL = 3

PONDERADOR = "w_norm"

# ============================================================
# RECALIBRACIÓN
# ============================================================
# Preguntas cuyo modelo se sirve RECALIBRADO. Va acá, pre-especificado y dentro
# de la huella del contrato, y no se decide mirando resultados: elegir a
# posteriori qué recalibrar sobre los mismos datos con los que se evalúa es otra
# forma de sobreajuste.
#
# Hoy sólo `politico_mano_dura`, y por evidencia: es la única de las cuatro que
# rechaza el contraste de Hosmer-Lemeshow ponderado (p≈0,002 contra 0,30-0,94 de
# las otras tres). El mapa es una spline monótona sobre cinco nodos de igual masa
# ponderada, ajustada FUERA DE MUESTRA.
#
# Por qué cinco nodos y no una isotónica libre: la libre "arreglaba" el HL
# (p=0,50) pero un contraste sin bins la seguía rechazando con p=0,001 — estaba
# calzando los bins con los que se la evaluaba— y empeoraba el log-loss de 0,524
# a 0,566. La regularizada mejora las tres métricas a la vez. Lo encontró Codex
# el 7/9/2026, revisando mi conclusión de que no correspondía recalibrar.
#
# Por qué no Platt: el defecto no es una pendiente. La curva cambia de signo y
# los dos tramos superiores hay que agruparlos; una recta en escala logit no
# puede con esa forma.
PREGUNTAS_A_RECALIBRAR = ("politico_mano_dura",)


# ============================================================
# NIVEL CALIBRADO DEL INTERVALO
# ============================================================
# El percentil del bootstrap que hay que pedir para que el intervalo cubra de
# verdad el 95%. NO es un capricho: el bootstrap percentil SUB-CUBRE, y acá está
# medido cuánto.
#
# CÓMO SE MIDIÓ (scripts/cobertura_simulada.py, recalibrado el 8/9/2026). Se
# tomó el modelo publicado como verdad, se sortearon resultados desde él, y se
# corrió el PIPELINE COMPLETO —elección de C, ajuste, bootstrap con re-elección
# de C, mapa de calibración— 200 veces por pregunta (dos corridas de 100 con
# semillas 401 y 402). Después se contó, para cada uno de los 1.008 perfiles,
# cuántas veces el intervalo contenía la probabilidad verdadera, que se conoce
# por construcción. Son 800 pipelines completos —4 preguntas x 2 corridas x 100
# simulaciones—, unas dos horas con las ocho corridas en paralelo. (Decía
# "~3.200"; el número estaba mal y lo corrigió Codex.)
#
# Pidiendo el 95% nominal, la cobertura REAL es:
#   mano dura 94,1% · cadena perpetua 91,3% · pena de muerte 93,4% ·
#   humillación 91,9%
#
# La mejor es mano dura y la peor cadena perpetua. Con el simulador viejo el
# orden era otro y el comentario decía que la mejor era pena de muerte.
#
# O sea que el widget diría "95%" y entregaría entre 91 y 94.
#
# POR QUÉ SUBIR EL NIVEL Y NO ENSANCHAR POR UN FACTOR. Se probaron las dos.
# Subir el nivel deja menos perfiles malos, porque sigue la forma de la
# distribución bootstrap en vez de estirarla simétricamente — y cerca de 0 y de
# 100 esa distribución es muy asimétrica. En cadena perpetua, con esta medición:
# el nivel 99 da 96,75% de cobertura media y deja 11 perfiles por debajo del 90%
# (el peor, 88,0%); el factor x1,30 da 95,84% y deja 67 (el peor, 86,0%). No es
# a igualdad de cobertura —el nivel cubre casi un punto más— así que la
# comparación favorece al nivel por dos motivos a la vez y no aísla la forma.
# Con el simulador viejo este mismo ejemplo decía "1 contra 35"; eran otros
# números y no había con qué reproducirlos.
#
# CADA PREGUNTA NECESITA LO SUYO, así que no hay un número global. El criterio
# es el nivel más chico cuya cobertura llega al 95% en LAS DOS semillas por
# separado, no sólo en el promedio de las dos: en mano dura el nivel 96 promedia
# 95,10% pero una de las dos corridas da 94,53%, y en cadena perpetua el 98
# promedia 95,19% con una corrida en 94,86%.
#
# ES UN DESEMPATE CONSERVADOR, NO UN TEST, y conviene no venderlo como más de lo
# que es. Con la cobertura verdadera justo en 95%, cada corrida tiene alrededor
# de un 50% de chance de quedar por encima, así que las dos quedan por encima
# alrededor de una de cada cuatro veces — es una cuenta de servilleta, no un
# número que salga de estas salidas. Y el error Monte Carlo de estas mediciones es del orden del punto —los
# 1.008 perfiles comparten cada muestra simulada, así que no son 1.008
# experimentos independientes y las salidas no guardan la covarianza que haría
# falta para calcularlo exacto—, de modo que el 95% cae dentro del margen de los
# cuatro cortes elegidos. Sirve para no elegir el nivel mirando un promedio que
# se apoya en una sola corrida buena; no para afirmar que el nivel elegido
# cubre.
#
# MEDIDO A B=10.000, QUE ES CON LO QUE SE PUBLICA (11/9/2026). Ésta era la
# deuda vieja: el nivel se había elegido midiendo con 1.000 réplicas y se
# publica con 10.000, y el signo de esa diferencia no se conocía. Ya se conoce,
# y fue para el lado favorable: el intervalo cubre un poco MÁS de lo que decía
# la medición barata, así que tres de los cuatro niveles bajan un punto.
#
#   pregunta            antes   ahora   semilla 601   semilla 602
#   mano dura              98      97        95,88%        95,38%
#   cadena perpetua        99      98        95,84%        95,88%
#   pena de muerte         97      97        96,58%        95,52%
#   humillación            98      97        95,09%        96,22%
#
# 800 simulaciones, 100 por pregunta y semilla, unas 40 horas de máquina. Las
# salidas están en `scripts/salidas/` y REEMPLAZARON a las de B=1.000: el test
# compara contra lo que hay ahí, y mezclar dos estudios con distinto B daría un
# promedio que no es de ningún procedimiento.
#
# MANO DURA VOLVIÓ AL CRITERIO, y conviene decir por qué estaba afuera. Entre el
# 8/9 y el 11/9 publicó 98 cuando el criterio decía 97, y este comentario lo
# llamaba "decisión editorial declarada". NO LO ERA: la tomé yo, por la cola de
# esa pregunta, sin consultarlo con nadie, y el rótulo "editorial" le dio un
# peso que no tenía. Al preguntarlo, la respuesta fue que vale lo que dicen las
# simulaciones. Hoy las cuatro publican lo que dice el criterio.
#
# EL ARGUMENTO DE LA COLA SIGUE SIENDO CIERTO, y por eso queda escrito acá en
# vez de borrado: mano dura es la pregunta con la peor cola por lejos, y bajar
# de 98 a 97 duplica los perfiles mal cubiertos.
#
# LA COLA EN EL NIVEL PUBLICADO, que es lo que recibe el lector de SU perfil:
#   mano dura 97 → media 95,6%, peor perfil 74,0%, 68 perfiles bajo 90%
#   cadena perpetua 98 → media 95,9%, peor perfil 90,0%, ninguno bajo 90%
#   pena de muerte 97 → media 96,1%, peor perfil 92,0%, ninguno bajo 90%
#   humillación 97 → media 95,7%, peor perfil 87,0%, 5 bajo 90%
# El promedio tapa la cola, y por eso la UI nunca prometió un 95% que no se
# sostiene perfil por perfil. En mano dura la distancia entre el promedio y el
# peor perfil es de más de veinte puntos: si alguna vez se revisa un nivel, es
# ése, y con el dato a la vista en vez de con un rótulo inventado.
#
# APLICARLO NO EXIGE REENTRENAR. El nivel entra en `huella_contrato`, así que
# cambiarlo invalida la huella guardada en los cuatro JSON y en la envolvente;
# pero no participa del ajuste, se aplica al leer. `scripts/aplicar_nivel.py`
# re-sella los artefactos, y sólo lo hace tras probar que entre los JSON y la
# configuración no difiere nada más que el nivel.
#
# DESDE EL 9/9/2026 LA UI TAMPOCO MUESTRA EL INTERVALO. Decisión editorial de
# Tomer: "a la gente no le sirve de nada y es difícil de entender". El ancho es
# irreducible —aun en los perfiles con 10 o más casos ponderados detrás la
# mediana es de 26,8 / 21,0 / 28,0 / 13,1 pp según la pregunta— así que no era
# un problema de presentación. Todo lo que sigue en este bloque vale igual: el
# intervalo se calcula, se calibra y GOBIERNA lo que el widget afirma; lo que
# se sacó es su exhibición. Ver el comentario en `components.render_result_card`.
#
# DOS COSAS QUE ESTOS NÚMEROS NO RESUELVEN:
#
# 1. EL NIVEL SE ELIGIÓ CON 1.000 RÉPLICAS Y SE PUBLICA CON 10.000, y el signo
#    de esa diferencia NO se conoce. El simulador corre el bootstrap interno en
#    1.000 porque a 10.000 la medición llevaría unas veinte horas por corrida
#    —o sea otras veinte para el par, porque las dos semillas van en paralelo—.
#    Ese "veinte" es una extrapolación lineal de lo que tardó cada una de estas
#    corridas: entre 6.815 y 7.049 segundos según sus logs, o sea 18,9 a 19,6
#    horas si se multiplica por diez. Las ocho salidas de esa fecha no guardan
#    la duración; el simulador la guarda de acá en más, en la clave "segundos". Subir las réplicas achica el error Monte Carlo del cuantil
#    extremo —los números están en `train_model.py`, junto a N_REPLICAS— pero
#    menos ruido no es más cobertura: si el extremo ruidoso incluía la verdad
#    por accidente, achicar el ruido la deja afuera. Escribí que producción
#    "debería portarse igual o mejor" y no está justificado; lo marcó Codex, que
#    además comparó las primeras 1.000 réplicas de mano dura contra las 10.000 y
#    encontró 263 perfiles donde el intervalo se ANGOSTA, con hasta 4,98 pp de
#    movimiento en un extremo. Saber el signo exige medir a B=10.000.
#
# 2. LA VERDAD SIMULADA ES EL PROPIO MODELO, así que todo esto corrige la
#    sub-cobertura del PROCEDIMIENTO suponiendo que la forma funcional es la
#    correcta. El error de especificación se suma encima y NO entra en el
#    intervalo, porque el bootstrap remuestrea casos con la forma fija.
#
#    YA NO ES UNA ADVERTENCIA SIN NÚMERO. `scripts/error_especificacion.py` lo
#    mide: compara el procedimiento publicado —incluida la recalibración, para
#    la pregunta que la lleva— contra otras siete formas funcionales sobre LAS
#    MISMAS seis variables, se queda con las que la muestra no logra ordenar por
#    log-loss fuera de muestra, y mira cuánto se mueve el número de cada uno de
#    los 1.008 perfiles.
#
#    EL NÚMERO SE MUEVE. Entre especificaciones que la muestra no ordena, la
#    mediana del rango va de 3,7 pp (humillación) a 12,3 (pena de muerte), y el
#    p95 llega a 28,6. Sacando la más flexible por si fuera ella sola la que
#    empuja, la mediana queda entre 2,5 y 9,6. Los perfiles sin ningún caso en
#    la muestra discrepan más (4,1 a 14,4) que los que tienen al menos uno (3,5
#    a 10,7), que es lo esperable porque ahí toda especificación extrapola.
#
#    LAS AFIRMACIONES, EN CAMBIO, NO SE MUEVEN. De las 2.562 veces que el widget
#    afirma de qué lado está la mayoría, cambiarían DOS. De las 1.742 veces que
#    afirma una diferencia contra el promedio nacional, CUATRO. Seis de 4.304.
#    (Eran 1.736 y 4.298 antes de bootstrapear la diferencia; el estudio se
#    quedó midiendo la regla vieja cuando el widget cambió y lo marcó Codex.)
#
#    Y EL INTERVALO YA ABSORBÍA CASI TODO: contenía lo que dicen todas las
#    especificaciones admitidas en el 100% de los perfiles de mano dura, el
#    99,3% de cadena perpetua y el 98,1% de humillación. La excepción era PENA
#    DE MUERTE, con 113 perfiles (11,2%) donde alguna caía afuera y un exceso
#    máximo de 12,2 pp.
#
#    ESO SE CERRÓ EL 9/9/2026 ENSANCHANDO HASTA LA ENVOLVENTE. El intervalo que
#    se publica es ahora el más chico que contiene al bootstrap de la forma
#    publicada Y a lo que dicen las demás especificaciones admitidas. La tabla
#    por perfil vive en `modelos/envolvente_espec.json`, la genera
#    `scripts/agregar_envolvente.py` y `model.load_envolvente()` la aplica.
#
#    ES UNA UNIÓN, NO UNA SUMA, y por eso no contradice lo que dice más abajo
#    sobre no poder sumar este rango al ancho del bootstrap: sumarlos contaría
#    dos veces el ruido de estimación que los dos comparten; tomar el máximo no
#    supone independencia de nada.
#
#    POR QUÉ VALÍA LA PENA, más allá de los 113. Las dos especificaciones que
#    empujaban afuera en pena de muerte son las que PREDICEN MEJOR que la
#    publicada: `todas_2do_orden` con log-loss 0,532 contra 0,542 (t = −2,69,
#    que no llega al umbral de 2,776 y por eso entra como "indistinguible" en
#    vez de como mejor) e `ideolxeduc`, la única declarada mejor. O sea que el
#    intervalo dejaba afuera justamente a los modelos que la muestra prefiere.
#    Y no eran perfiles marginales: sólo el 31% de los 113 no tiene ningún caso
#    detrás, contra el 47% del total, y 79 de ellos llevan una afirmación de
#    mayoría y 68 una de brecha.
#
#    LO QUE COSTÓ: se retiran 2 afirmaciones de mayoría sobre 2.562 y 5 de
#    brecha sobre 1.742, todas de pena de muerte. El ancho mediano del intervalo
#    mostrado no se mueve en ninguna pregunta (mano dura 34,34; cadena 32,82;
#    humillación 15,00) y en pena de muerte pasa de 32,38 a 32,43 pp. Cambian
#    139 perfiles de los 4.032: 0, 7, 113 y 19.
#
#    Y ADEMÁS DESAPARECEN LAS CONTRADICCIONES: las 2 afirmaciones de mayoría y
#    las 4 de brecha que alguna especificación daba vuelta ahora no se afirman.
#    Es por construcción, no por suerte: si una especificación admisible cae del
#    otro lado del 50, el intervalo ensanchado contiene al 50 y el widget se
#    abstiene.
#
#    LO QUE NO ARREGLA. La envolvente cubre la dispersión DENTRO de las nueve
#    formas que se probaron, bajo un criterio de admisión que este mismo bloque
#    describe como poco confiable al pie de la letra. Si la verdad tiene una
#    forma que no está en la lista, esto no la alcanza.
#
#    DOS DEFECTOS QUE TUVO ESTE ESTUDIO Y QUE ENCONTRÓ CODEX, porque los números
#    de arriba son los de después de arreglarlos y los de antes estaban inflados:
#      · la "base" no incluía la recalibración, así que en mano dura se comparaba
#        contra algo que no era el número publicado —mediana 3,52 pp de
#        diferencia, hasta 9,32—. Arreglado, el rango de mano dura bajó de 10,4 a
#        5,8 pp y sus perfiles fuera del intervalo pasaron de 73 a CERO.
#      · se contaban cruces de estimaciones puntuales y se presentaban como
#        afirmaciones dadas vuelta. No lo eran: de esos cruces, el intervalo
#        publicado YA contenía el 50% en 136 de 136, 116 de 116, 170 de 172 y 47
#        de 47. El widget ya se abstenía en casi todos. Ahora se cuenta sobre las
#        afirmaciones que el widget hace, con la regla de `interpretar()` y la de
#        `brecha_nacional()`.
#
#    QUÉ NO DICE ESTE ESTUDIO. Cuál especificación es la correcta: aparece que
#    ideología x educación le gana a la base en dos preguntas e ideología x
#    región en una tercera, pero gana una distinta en cada una y ninguna en la
#    cuarta, ninguna pasa Bonferroni sobre 32 comparaciones, y el error estándar
#    de la validación cruzada está subestimado por construcción. Tampoco es una
#    COTA del error de especificación: si la verdad está fuera de la familia
#    probada, el rango puede quedar corto o largo. Y el rango mezcla forma
#    funcional con ruido de estimación, así que no es una incertidumbre
#    independiente que se pueda sumar al intervalo. Ver el docstring del script.
#
# Y una afirmación mía que quedó sobredicha: dije que subir el nivel gana sobre
# ensanchar por un factor "a igualdad de cobertura". No era a igualdad, ni antes
# ni ahora: en esta medición el nivel 99 da 96,75% y el factor x1,30 da 95,84%,
# casi un punto de diferencia. La ventaja del nivel sobre el factor sigue sin
# demostrarse limpiamente.
# Los cuatro salen del criterio sobre el estudio a B=10.000; el comentario de
# arriba tiene la tabla. Al lado va la cobertura que da el 95% nominal, que es
# lo que justifica pedir un percentil más ancho.
NIVEL_CALIBRADO = {
    "politico_mano_dura": 97,   # el 95% nominal da 93,4%
    "cadena_perpetua": 98,      # da 92,0%, la peor de las cuatro
    "pena_muerte": 97,          # da 93,7%
    "humillacion_presos": 97,   # da 93,3%
}

# ============================================================
# CRÉDITOS
# ============================================================
# Tomer, 7/9/2026: "en la fuente, siempre es la encuesta de El
# Observador-UMAD-Ferreira y el crédito tuyo".
#
# Van en una constante y se serializan dentro de cada JSON: si vivieran sólo en
# el componente del pie, un modelo publicado quedaría sin decir de qué encuesta
# salió en cuanto alguien reordenara la UI.
FUENTE = "Encuesta El Observador-UMAD-Ferreira sobre seguridad pública, mayo de 2026"
CREDITO = "Análisis y desarrollo: Juan Ignacio Pintos Elso"

# ============================================================
# MAPEOS UI -> CÓDIGO DEL MODELO
# ============================================================
EDAD_UI_TO_CODE = {
    "18-29 años": 1,   # referencia
    "30-44 años": 2,
    "45-59 años": 3,
    "60 años o más": 4,
}

# Tres categorías: "Primaria o menos" se unió con "Secundaria" porque sola
# tenía 28 casos y era la referencia del modelo. Ver EDUC_COLAPSO en
# train_model.py.
EDUC_UI_TO_CODE = {
    "Secundaria o menos": 1,   # referencia
    "Terciaria incompleta": 2,
    "Terciaria completa o más": 3,
}

# Autoubicación en la escala de 0 a 10, en siete tramos simétricos.
#
# EL REPARTO NO SE ELIGE ACÁ: lo fijó Tomer al mandar la base ya etiquetada
# (`base_etiquetada_SEGURIDAD_para probabilidad.xlsx`, 7/9/2026), donde
# `var_242` viene con la etiqueta en vez del número. Estos siete tramos son la
# transcripción exacta de esa columna — se verificó cruzándola contra la
# columna numérica de la base completa, valor por valor, en los 3.377 casos.
# El test `test_tramos_ideologicos_reproducen_la_base_etiquetada` deja fijos los
# conteos resultantes para que correr un borde no pase inadvertido.
#
# Es un reparto simétrico alrededor del 5 (anchos 1, 2, 2, 1, 2, 2, 1), que es
# lo que hace comparables los dos extremos. La versión anterior también era
# simétrica pero con otros anchos (2, 2, 1, 1, 1, 2, 2), así que los
# coeficientes ideológicos de este modelo NO son comparables con los de aquélla
# aunque las dummies se sigan llamando igual.
#
# El 5 sigue siendo la referencia: en una escala de 0 a 10 es el punto medio
# exacto y es la respuesta modal (1.092 de 3.377 casos, el 32%). Una referencia
# chica hace que todos los coeficientes se estimen contra pocos casos, que es el
# problema que ya hubo con "Primaria o menos" y sus 28 casos.
#
# CAVEAT DE TAMAÑO: con este corte "Extrema izquierda" es sólo el 0 y queda en
# 42 casos —contra los 80 que tenía cuando abarcaba 0-1—, y "Extrema derecha"
# es sólo el 10, con 131. Entran al modelo, pero son los dos números más
# frágiles del gráfico comparativo y no conviene titular con ellos.
#
# La referencia es el Centro por ser el tramo modal: una referencia chica hace
# que todos los coeficientes se estimen contra pocos casos, que es el problema
# que ya hubo con "Primaria o menos" y sus 28 casos.
#
# "No se ubica" NO se ofrece en la UI. Esa dummy agrupa a quienes no
# contestaron la escala: 80 en la encuesta completa, de los cuales 62 entran al
# modelo principal de apoyo (los otros no tienen postura definida y quedan en el
# modelo secundario de neutralidad). Y no contestar no es lo
# mismo que ubicarse en el centro: el cuestionario NO ofrece "no sabe" entre
# las opciones —son exactamente los once valores— así que un nulo es una
# pregunta salteada. Sigue existiendo como predictor, para que esos casos no
# contaminen la referencia, pero queda siempre en cero desde la interfaz.
# Especificación de la transformación dato crudo -> dummy. Vive acá, y no
# repartida entre config y train_model, porque TODA ella tiene que entrar en la
# huella: cambiar cualquiera de estos valores cambia el significado de las
# dummies aunque los nombres queden iguales.
ESPEC_CRUDA = {
    # Escala nivel_educativo (1-10) del proveedor -> las 3 categorías del modelo.
    "educ_colapso": {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 3, 9: 3, 10: 3},
    # Cortes de edad (bordes de pd.cut) y sus códigos.
    "edad_cortes": [17, 29, 44, 59, 120],
    # Tramos de la autoubicación 0-10. Cada uno es [desde, hasta] inclusive y
    # el nombre es el sufijo de la dummy. El tramo marcado como referencia no
    # genera dummy. Van explícitos y no como dos umbrales sueltos porque ahora
    # son seis cortes y un umbral no alcanza para describirlos.
    # [sufijo de la dummy, desde, hasta, etiqueta] — inclusive en los dos
    # bordes. Es la FUENTE ÚNICA: de acá salen las dummies del entrenamiento,
    # las de la inferencia, las opciones del selector y las etiquetas del
    # bloque comparativo. Antes las etiquetas vivían en un diccionario aparte
    # y la correspondencia era posicional: reordenarlo dejaba los 71 tests en
    # verde y le ponía nombres cambiados a las tasas publicadas.
    "ideol_tramos": [
        ["izq_extrema", 0, 0, "Extrema izquierda"],
        ["izquierda", 1, 2, "Izquierda"],
        ["centroizq", 3, 4, "Centroizquierda"],
        ["centro", 5, 5, "Centro"],
        ["centroderecha", 6, 7, "Centroderecha"],
        ["derecha", 8, 9, "Derecha"],
        ["der_extrema", 10, 10, "Extrema derecha"],
    ],
    "ideol_referencia": "centro",
    # Códigos crudos de las demás variables.
    "sexo_valores": {"mujer": "Mujer", "hombre": "Hombre"},
    "dpto_montevideo": 1,
    # Etiquetas crudas de victimización. Viven acá y no en train_model.py para
    # que entren en la huella: son parte de la definición de las dummies.
    "victima_etiquetas": {
        "no": ["no"],
        "sin_violencia": ["sí  sin violencia", "si  sin violencia", "sí sin violencia"],
        "con_violencia": ["sí  con violencia", "si  con violencia", "sí con violencia"],
    },
}

def _etiqueta_tramo(desde, hasta, etiqueta):
    """«Centro (5)» si el tramo es un valor solo, «Izquierda (2-3)» si son dos."""
    rango = f"{desde}" if desde == hasta else f"{desde}-{hasta}"
    return f"{etiqueta} ({rango})"


# Derivado de ESPEC_CRUDA["ideol_tramos"], no escrito a mano: el orden, los
# valores de la escala y el nombre de la dummy salen todos de la misma lista,
# así que no pueden desalinearse entre sí.
IDEOLOGIA_UI_TO_CODE = {
    _etiqueta_tramo(desde, hasta, etiqueta): i
    for i, (_, desde, hasta, etiqueta) in enumerate(ESPEC_CRUDA["ideol_tramos"], start=1)
}

# Índice del selector: el tramo de referencia, que es el modal. Iba a mano y
# quedó apuntando a la categoría equivocada al pasar de seis tramos a siete —
# el widget abría en "Centroizquierda (4)" mientras el comentario decía
# "Centro (5)".
IDEOLOGIA_INDICE_DEFECTO = next(
    i for i, (nombre, _, _, _) in enumerate(ESPEC_CRUDA["ideol_tramos"])
    if nombre == ESPEC_CRUDA["ideol_referencia"]
)


VICTIMA_UI_TO_CODE = {
    "No": 1,                       # referencia
    "Sí, sin violencia": 2,
    "Sí, con violencia": 3,
}

REGION_UI_TO_CODE = {
    "Montevideo": 1,
    "Interior": 0,
}

# ============================================================
# PREDICTORES DEL MODELO
# ============================================================
# El orden importa sólo para la legibilidad de los reportes; el vector se
# arma por nombre, no por posición.
# Sin voto de balotaje: Tomer pidió expresamente "poner identificación
# ideológica y sacar partidos políticos" (31/8/2026). Es una decisión
# editorial suya, no un problema del modelo — el balotaje discriminaba bien.
# Consecuencia estadística a tener presente: parte de lo que antes explicaba
# el voto ahora lo absorbe la ideología declarada, así que los coeficientes
# ideológicos de este modelo NO son comparables con los de la versión anterior.
PREDICTORES = [
    "edad_30_44", "edad_45_59", "edad_60_plus",
    "es_mujer",
    "educ_ter_incomp", "educ_ter_comp",
    "ideol_izq_extrema", "ideol_izquierda", "ideol_centroizq",
    "ideol_centroderecha", "ideol_derecha", "ideol_der_extrema",
    "ideol_no_ubica",
    "victima_sin_violencia", "victima_con_violencia", "victima_sin_dato",
    "es_montevideo",
]

# Los dos predictores que la UI NUNCA enciende: agrupan a quienes no
# contestaron esas preguntas. Existen para que esos casos no contaminen las
# categorías de referencia, pero no son opciones que el lector pueda elegir, así
# que no tienen por qué aparecer en el texto que él lee. La lista vive acá para
# que la metodología los pueda descartar sin repetir los nombres a mano.
PREDICTORES_OCULTOS = ("ideol_no_ubica", "victima_sin_dato")

# `victima_sin_dato` existe para que los 53 casos sin respuesta no se mezclen
# con quienes contestaron "No" —eso contaminaba la referencia y la tasa que se
# publica del grupo "No fue víctima"—, pero en la UI queda siempre en cero: el
# widget obliga a elegir una de las tres opciones reales. Es un predictor de
# entrenamiento, no de interacción.



def huella_contrato(slug):
    """
    Huella de TODO el contrato entre la configuración y el modelo entrenado de
    la pregunta `slug`.

    Va por pregunta y no una sola para todas: los cuatro modelos comparten los
    mapeos y la codificación, pero cada JSON tiene que poder decir de qué
    pregunta es. Si la huella fuera común, mover un JSON encima de otro pasaría
    el chequeo y el widget serviría los coeficientes de la pregunta equivocada
    con el título correcto.

    Verificar sólo que no falten predictores es un chequeo de subconjunto y deja
    pasar el caso peligroso: que un nombre de dummy siga existiendo pero
    signifique otra cosa. Pasó de verdad al colapsar educación de cuatro
    categorías a tres — `educ_ter_incomp` sobrevivió con el mismo nombre y otro
    código detrás, así que un JSON viejo habría cargado sin protestar y la
    inferencia habría aplicado coeficientes de otra codificación.

    Incluye TODO lo que define el significado de una dummy: los mapeos de la UI,
    las referencias, la escala Likert y la especificación de transformación
    cruda (cortes de edad, colapso educativo, tramos ideológicos). Una versión
    anterior de esta función decía en su docstring que cubría las referencias y
    no las cubría, y sobre todo dejaba afuera `EDUC_COLAPSO` — con lo cual
    cambiar el colapso educativo dejaba exactamente la misma huella, que es
    justo el agujero que esto viene a tapar.

    Los mapeos se ordenan antes de hashear: reordenar `PREDICTORES` no cambia el
    modelo y no debería invalidar el JSON.
    """
    import hashlib
    import json as _json
    if slug not in PREGUNTAS:
        raise KeyError(f"pregunta desconocida: {slug!r}. Son {SLUGS}.")
    material = _json.dumps({
        "pregunta": slug,
        "columna": PREGUNTAS[slug]["columna"],
        # ordenados: el orden de la lista no tiene significado semántico
        "predictores": sorted(PREDICTORES),
        # los mapeos SÍ van como listas ordenadas de pares: cambiar qué opción
        # ofrece la UI, o qué código le corresponde, cambia el contrato
        "edad": sorted(EDAD_UI_TO_CODE.items()),
        "educacion": sorted(EDUC_UI_TO_CODE.items()),
        "ideologia": sorted(IDEOLOGIA_UI_TO_CODE.items()),
        "victima": sorted(VICTIMA_UI_TO_CODE.items()),
        "region": sorted(REGION_UI_TO_CODE.items()),
        "referencias": sorted(REFERENCIAS.items()),
        "likert": sorted(LIKERT_MAP.items()),
        "favor": sorted(LIKERT_FAVOR),
        "contra": sorted(LIKERT_CONTRA),
        "neutral": LIKERT_NEUTRAL,
        "ponderador": PONDERADOR,
        "recalibradas": sorted(PREGUNTAS_A_RECALIBRAR),
        "nivel_calibrado": sorted(NIVEL_CALIBRADO.items()),
        "espec_cruda": _json.dumps(ESPEC_CRUDA, sort_keys=True),
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


REFERENCIAS = {
    "edad": "18-29 años",
    "sexo": "Hombre",
    "educacion": "Secundaria o menos",
    "ideologia": _etiqueta_tramo(*[t[1:] for t in ESPEC_CRUDA["ideol_tramos"]
                                  if t[0] == ESPEC_CRUDA["ideol_referencia"]][0]),
    "victima": "No fue víctima",
    "region": "Interior",
}

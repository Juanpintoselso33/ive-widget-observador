# Widget de seguridad pública — El Observador

Estima qué proporción de las personas con determinadas características apoya una
medida punitiva. Mismo molde que el widget IVE: regresión logística ponderada,
coeficientes serializados a JSON, inferencia en Python puro.

Pedido por Tomer Urwicz (19/8/2026) como contraparte del widget del IVE, para
publicar con los resultados de la encuesta de seguridad de mayo de 2026.

**El lector elige entre cuatro medidas.** Cada una tiene su propio modelo
entrenado y su propio JSON; el widget carga los cuatro al arrancar y sirve el que
corresponda al selector. Hasta el 7/9/2026 se publicaba una sola pregunta por
vez, elegida con una constante.

## Las cuatro preguntas

Son las que Tomer señaló por columna en la base etiquetada (7/9/2026):

| Slug | Var | Pregunta | Apoyo ponderado |
|---|---|---|---|
| `politico_mano_dura` *(la que abre)* | `var_233` | Votaría a un político que promoviera castigos más duros | **67,0%** |
| `cadena_perpetua` | `var_230` | Cadena perpetua sin condicional por tres delitos graves | **78,6%** |
| `pena_muerte` | `var_229` | Pena de muerte por homicidio | **36,7%** |
| `humillacion_presos` | `var_231` | Los presos merecen humillación, intimidación y degradación | **11,0%** |

Los porcentajes son entre quienes tienen **postura definida**, ponderados por
diseño muestral.

Cada entrada de `PREGUNTAS` lleva el **enunciado textual del cuestionario**, que
es el que se muestra al lector: el widget mide acuerdo con esa frase exacta y
parafrasearla cambia lo que el número significa.

Se cayó `aumentar_penas` (`var_228`), que existía como candidata: Tomer no la
incluyó y su columna tampoco viene en el recorte que mandó. Volver a sumarla es
agregar la entrada en `config.PREGUNTAS` y re-entrenar.

### Agregar o cambiar una pregunta

```bash
python widgets/seguridad/train_model.py                        # las cuatro
python widgets/seguridad/train_model.py --pregunta pena_muerte # una sola
python widgets/seguridad/scripts/validacion_ordinal.py         # después: robustez
```

`train_model.py` escribe `modelos/model_<slug>.json`. **La validación ordinal va
después y es parte del entregable**, no un extra: re-entrenar borra la clave
`robustez` del JSON, y sin ella la sección "Cómo se calcula" dice explícitamente
que no hay nada verificado sobre qué efectos aguantan. Es a propósito: los
coeficientes cambiaron, la validación vieja ya no habla de ellos.

`test_config.py` falla si un JSON no corresponde a su pregunta o a la
configuración actual, para que no se publique un widget con el título de una
pregunta y los coeficientes de otra.

## Datos

`base_etiquetada.csv` del repo de encuestas, que **no vive en este repo** (son
datos del cliente y el `.gitignore` excluye `*.csv`):

```
Observador-encuesta/encuestas/observador_2026_05_seguridad/output/base_etiquetada.csv
```

Se puede pisar la ruta con la variable de entorno `SEGURIDAD_DATA_FILE`.

El 7/9/2026 Tomer mandó además `base_etiquetada_SEGURIDAD_para probabilidad.xlsx`.
**No trae datos nuevos**: son los mismos 3.377 casos con 49 de las 65 columnas, y
47 de esas 49 son idénticas al CSV (`w_norm` coincide hasta el ruido de punto
flotante). Lo único distinto es que trae `var_242` ya etiquetada en tramos
ideológicos — que es lo que fijó el corte que usa el widget, ver abajo.

## Modelo

Regresión logística binaria con penalización L2, `sample_weight=w_norm`, `C`
elegido por validación cruzada estratificada de 5 folds sobre `neg_log_loss`.

**El ponderador no es opcional.** La muestra cruda está sesgada a montevideanos
(67% contra 44,5% ponderado), hombres (63,5% contra 49,2%) y mayores de 60 (39%
contra 28,9%). Sin ponderar, el apoyo a la pena de muerte da 26,9%; ponderado,
36,7%. Diez puntos de diferencia en el número que se publica.

Dependiente: a favor si Likert ≥ 4, en contra si ≤ 2. Los neutrales (Likert = 3)
se excluyen del modelo principal y se modelan aparte, igual que en el IVE, así
que el porcentaje es **condicional a tener postura definida**.

Predictores, con su categoría de referencia entre paréntesis: edad en cuatro
tramos (18-29), sexo (hombre), educación en **tres** niveles (secundaria o
menos), autoubicación ideológica en **siete tramos** (centro), víctima de delito
en 12 meses (no fue víctima) y región (interior).

**Sin voto de balotaje.** Tomer pidió expresamente "poner identificación
ideológica y sacar partidos políticos" (31/8/2026). Es una decisión editorial
suya, no un problema del modelo — el balotaje discriminaba bien. Consecuencia a
tener presente: parte de lo que antes explicaba el voto ahora lo absorbe la
ideología declarada.

Educación quedó en tres categorías y no cuatro: "Primaria o menos" tenía 28 casos
de 2.672 y era la referencia, así que los coeficientes más grandes del modelo se
estimaban contra 28 personas.

Dos dummies existen sólo para el entrenamiento y la UI nunca las enciende:
`victima_sin_dato` e `ideol_no_ubica` (ver `config.PREDICTORES_OCULTOS`). Agrupan
a quienes no contestaron esas preguntas, y no contestar una encuesta no es lo
mismo que no ubicarse políticamente ni que no haber sido víctima: ofrecérselas al
lector le aplicaría el coeficiente de un grupo definido por otra cosa. Están para
que esos casos no contaminen las categorías de referencia. En el caso de víctima,
esa mezcla corría tanto la referencia como la tasa publicada del grupo "No fue
víctima": 34,2% contra el 34,6% real.

### Métricas por pregunta

| Pregunta | n con postura | Excluidos | N efectivo (Kish) | McFadden | C |
|---|---|---|---|---|---|
| Mano dura | 2.710 | 667 | 576 | 0,242 | 0,1 |
| Cadena perpetua | 2.950 | 427 | 619 | 0,104 | 0,1 |
| Pena de muerte | 2.672 | 705 | 571 | 0,217 | 0,1 |
| Humillación | 2.969 | 408 | 632 | 0,252 | 0,5 |

Cadena perpetua es la que peor se explica (McFadden 0,10): con 78,6% de apoyo hay
poco que discriminar, casi todo el mundo está de acuerdo.

## Los siete tramos ideológicos NO los elige este repo

Los fijó la columna etiquetada de la base que mandó Tomer. El corte es:

| Escala 0-10 | Tramo | Casos |
|---|---|---|
| 0 | Extrema izquierda | 42 |
| 1-2 | Izquierda | 162 |
| 3-4 | Centroizquierda | 573 |
| 5 | Centro *(referencia)* | 1.092 |
| 6-7 | Centroderecha | 884 |
| 8-9 | Derecha | 413 |
| 10 | Extrema derecha | 131 |

Se verificó **caso por caso en los 3.377 registros** que estos bordes reproducen
exactamente las etiquetas de su archivo, y `test_config.py` deja los conteos
clavados: mover un borde no rompe nada más —entrena igual, sirve igual— y publica
otra cosa.

Anchos 1-2-2-1-2-2-1: simétricos alrededor del 5, que es lo que hace comparables
los dos extremos. Comparar una red ancha contra una angosta produciría por
construcción la diferencia que el widget deja ver.

**La versión anterior usaba otro corte** (0-1, 2-3, 4, 5, 6, 7-8, 9-10), también
simétrico pero con otros anchos. Los coeficientes ideológicos de este modelo **no
son comparables** con los de aquélla aunque las dummies se llamen igual.

**Caveat de tamaño:** con este corte cada extremo es un solo valor de la escala.
"Extrema izquierda" queda en 42 casos —contra los 80 que tenía— y "Extrema
derecha" en 131. Entran al modelo, pero son los dos números más frágiles del
gráfico comparativo y **no conviene titular con ellos**.

## Qué aguanta y qué no, por pregunta

La validación ordinal (`scripts/validacion_ordinal.py`) reajusta el modelo sobre
la escala Likert completa, sin excluir neutrales, y compara signos. El resultado
se guarda en cada JSON bajo `robustez` y la UI redacta su párrafo desde ahí.

| Pregunta | Signos que coinciden | Spearman | Efectos que dan vuelta |
|---|---|---|---|
| Mano dura | 15/17 | 0,90 | sexo |
| Cadena perpetua | 12/17 | 0,20 | edad 30-44, edad 45-59, sexo |
| Pena de muerte | 12/17 | 0,70 | edad 30-44, sexo, extrema izquierda, centroderecha, Montevideo |
| Humillación | 13/17 | 0,52 | centroderecha, víctima (con y sin violencia), Montevideo |

(Las dummies ocultas que también dan vuelta se cuentan en el total pero no se le
muestran al lector: no son opciones que pueda elegir.)

**El sexo no aguanta en tres de las cuatro.** El widget no permite afirmar que
las mujeres apoyen más o menos que los varones. **Cadena perpetua es la más
frágil** en conjunto: Spearman 0,20 entre los rankings de magnitud, o sea que el
ordenamiento de factores casi no se sostiene al cambiar la forma de estimar. Es
coherente con su McFadden de 0,10.

Lo que sí aguanta en las cuatro es la estructura ideológica gruesa.

**Caveat técnico:** `statsmodels.OrderedModel` no acepta `sample_weight`, así que
la validación corre sin ponderar. Sirve para chequear la estructura, no las
magnitudes.

## Caveats

- **La muestra sobre-representa fuerte a los más educados**: 53% tiene terciaria
  completa y sólo el 1% primaria o menos. El ponderador corrige la estimación
  poblacional, pero no crea casos donde no los hay.
- **Los intervalos son amplios: rondan los 25-30 puntos.** Salen de un bootstrap
  estratificado de 1.000 réplicas por pregunta, con remuestreo dentro de cada uno
  de los 28 estratos de la encuesta y **re-selección de `C` en cada réplica**.
  Fijar `C` achica los intervalos artificialmente, porque trata la elección del
  hiperparámetro como si fuera un dato conocido. Cuando el intervalo cruza el
  50%, el widget deja de afirmar de qué lado está la mayoría.
- **El estratificado se usa por respetar el diseño, no porque ensanche.** Medido
  sobre esta base, el bootstrap sin estratificar da intervalos incluso un poco
  más anchos (mediana 26,6 contra 25,8 puntos).
- **No es un bootstrap de diseño completo**: la base no trae información de
  conglomerados, así que respeta los estratos pero no el efecto de conglomeración.
- **Sin p-valores**: Ridge no provee errores estándar analíticos; los intervalos
  vienen del bootstrap, no de la teoría del estimador.
- **Muchos perfiles no existen en la muestra.** De las **1.008** combinaciones que
  el lector puede elegir, según la pregunta aparecen entre **534 y 571** en la
  encuesta, y sólo entre **7 y 12** tienen 30 casos o más. El resto se estima por
  extrapolación aditiva. Está advertido en la UI.
- **El N efectivo ronda 600, no 2.700.** La dispersión de los ponderadores hace
  que las respuestas rindan como esa cantidad a efectos de precisión (N de Kish).

## Auditoría exhaustiva de lo publicado (7/9/2026)

Codex barrió los **4.032 resultados** que el widget puede mostrar —4 preguntas x
1.008 perfiles— con sus 1.000 réplicas bootstrap cada uno, buscando
inconsistencias. Dos hallazgos reales, los dos ya corregidos:

1. **1.883 de 4.032 resultados afirmaban una diferencia contra el promedio
   nacional que su propio intervalo no sostenía.** El widget ya tenía esa
   prudencia para el 50% (`interpretar()` + `banda_decision()`) y le faltaba
   acá. Ahora la decide `components.brecha_nacional()`, que es pura y está
   barrida por `tests/test_texto_publicado.py`. Vivía adentro del render, que es
   por lo que ningún test la alcanzaba.
2. **65 perfiles se publicaban como «0%»** sobre estimaciones de 0,176% a
   0,499%, todos en humillación a los presos. «0%» no es un redondeo: afirma que
   nadie con ese perfil está a favor. Lo resuelve `components.formato_pct()`,
   que devuelve `<1%` y `>99%` en los dos extremos.

**Qué NO encontró:** ninguna inconsistencia de cableado. Verificó que los cuatro
contratos coinciden, que los cuatro JSON traen los 17 predictores, que
`bootstrap.orden` es exactamente `["intercept", *PREDICTORES]` y que la
inferencia arma las dummies por nombre y no por posición.

### Las inversiones en variables ordenadas son del dato, no del código

Encontró 11 inversiones adyacentes. De las 9 contrastables contra
`stats_by_group`, **las 9 están también en el dato crudo**. Ninguna aparece sólo
después de ajustar.

La que motivó la auditoría —en mano dura, **terciaria completa sale más punitiva
que terciaria incompleta**— es real en crudo (57,0% → 59,8%) pero **sólo
conserva el signo en el 88,6% de las réplicas**, y el intervalo de la diferencia
cruza el cero: los dos niveles están empatados y el widget muestra un orden que
el dato no sostiene. No es publicable como hallazgo.

Firmes (≥95% de las réplicas): extrema izquierda → izquierda en mano dura
(99,1%) y en pena de muerte (96,0%); centro → centroderecha en pena de muerte
(100%); 18-29 → 30-44 en cadena perpetua (98,5%). Ruido claro: extrema izquierda
→ izquierda en humillación, con 53,6%.

### Pendiente

- **El promedio nacional no trae su propia incertidumbre.** `brecha_nacional()`
  compara el intervalo del perfil contra un promedio tratado como exacto, así
  que es conservador de un solo lado. Lo limpio es bootstrapear la diferencia
  perfil−promedio, que necesita serializar la tasa nacional por réplica en
  `train_model.py`. No está hecho.
- **`stats_by_group` sólo guarda los tramos de edad extremos** (18-29 y 60+),
  así que las dos inversiones internas de edad no se pueden clasificar como "del
  dato" o "del ajuste" sin volver a la base.

## Diagnóstico econométrico (7/9/2026)

Lo corre `scripts/diagnostico_econometrico.py`. **Reescrito tras una auditoría
adversarial de Codex que volteó dos conclusiones que yo había publicado acá.** Lo
que sigue es la versión corregida; los errores quedan dichos porque el método por
el que se llegó importa tanto como el número.

### El criterio estaba mal elegido

El AUC mide **ordenamiento individual**. Este widget no clasifica personas:
publica **una tasa por perfil**. Dos modelos con el mismo AUC pueden imprimir
porcentajes muy distintos, y una recalibración monótona puede mejorar mucho el
producto sin mover el AUC ni un punto. El criterio ahora es **log-loss y Brier**;
el AUC queda como diagnóstico de cuánta heterogeneidad hay entre perfiles.

| Pregunta | log-loss | Brier | AUC *(diagnóstico)* |
|---|---:|---:|---:|
| Mano dura | 0,5244 | 0,16340 | 0,775 |
| Cadena perpetua | 0,4896 | 0,15586 | 0,665 |
| Pena de muerte | 0,5373 | 0,17994 | 0,782 |
| Humillación | 0,2862 | 0,08558 | 0,813 |

Todo con **CV anidada**: `C` se elige dentro de cada fold. La versión anterior
usaba el `C` guardado en el JSON, elegido por CV sobre toda la muestra, así que
el hiperparámetro había visto los folds de validación. La inflación medida era
chica —unos +0,002 de AUC— pero el método estaba mal.

### Mano dura SÍ está descalibrada

Es la corrección que más importa. Yo había afirmado que **las cuatro estaban
calibradas**, y era falso.

La secuencia de mis dos errores: primero reporté el peor desvío por decil
(11,7 pp) como un hallazgo, sin nula. Después construí la nula, vi que caía
adentro, y concluí que estaba calibrado. Ese segundo paso también estaba mal, por
dos razones: un **máximo sobre diez bins** tiene poca potencia y es ciego a que
varios bins se desvíen de forma coordinada; y los bins salían de cuantiles **sin
ponderar**, con masa entre 157 y 548, o sea que no eran décimos comparables de la
población.

Con bins de igual masa ponderada y Hosmer-Lemeshow ponderado con varianza w²,
calibrado por Monte Carlo:

| Pregunta | HL | p | Sesgo agregado | Pendiente | Veredicto |
|---|---:|---:|---:|---:|---|
| **Mano dura** | 29,95 | **0,002** | +0,08 pp | 0,94 | **descalibrado** |
| Cadena perpetua | 11,49 | 0,319 | +0,02 pp | 0,86 | sin descalibración detectable |
| Pena de muerte | 4,30 | 0,936 | +0,02 pp | 1,12 | sin descalibración detectable |
| Humillación | 8,31 | 0,527 | −0,15 pp | 0,96 | sin descalibración detectable |

Fijarse en que **el sesgo agregado y la pendiente de mano dura se ven bien**
(+0,08 pp y 0,94) y el modelo está descalibrado igual: los desvíos cambian de
signo a lo largo de la curva y se cancelan en el promedio. Es exactamente el
patrón que un estadístico agregado no puede ver.

**Qué significa para la publicación:** en mano dura, los porcentajes por perfil
tienen un error sistemático que el intervalo no describe, además del error
aleatorio que sí describe. Es la pregunta que abre el widget.

### Lo que salió limpio

Sobreajuste leve. Sin colinealidad: VIF máximo 3,49 en los tramos de edad, contra
un corte de 5. Sin separación: las dos celdas por debajo de 50 casos
(`ideol_izq_extrema` y `victima_sin_dato`, 37 cada una) tienen tasas interiores.
Efecto de diseño 4,70 — los 2.710 casos rinden como 576.

## ¿Se puede mejorar? Una palanca, y más chica de lo que dije

Se descartaron, sin efecto en ninguna de las cuatro: mover `C`, interacción
educación × ideología, tamaño del hogar, situación laboral, splines y cuadrática
en edad, L1, elastic net y Firth. **Modelar el Likert completo PIERDE** contra el
binario fuera de muestra (entre −0,003 y −0,019 de AUC, y peor Brier) — algo que
la validación ordinal no había contestado, porque ajusta dentro de muestra y sin
pesos. Abrir la ideología a escala lineal sube cadena perpetua y **hunde pena de
muerte**: no es un reemplazo.

Queda el **voto de balotaje**, medido con CV anidada sobre cinco particiones:

| Pregunta | ΔAUC | ΔBrier | Δlog-loss | Gana en |
|---|---:|---:|---:|---:|
| Mano dura | +0,0230 | −0,01028 | **−0,02045** | **5/5** |
| Pena de muerte | +0,0033 | −0,00133 | **−0,00458** | **5/5** |
| Cadena perpetua | +0,0083 | −0,00079 | −0,00038 | 3/5 |
| Humillación | −0,0077 | +0,00075 | +0,00284 | 1/5 |

**Mi afirmación anterior —"mejora mano dura y cadena perpetua, sin costo en las
otras dos"— era incorrecta en las dos mitades.** Con el método corregido: mejora
claramente **mano dura**, mejora poco pero consistentemente **pena de muerte**,
es indistinguible de cero en **cadena perpetua**, y **empeora humillación**.

El balotaje está afuera por decisión editorial de Tomer (31/8/2026: *"poner
identificación ideológica y sacar partidos políticos"*). Sigue siendo la única
palanca viva, pero alcanza a dos preguntas y no a las cuatro.

**"Cerca del techo" queda como resumen empírico, no como conclusión demostrada.**
Comparar muchas especificaciones sobre los mismos folds y quedarse con la mejor
es en sí una forma de sobreajuste: estos números sirven para descartar palancas,
no para probar que no hay ninguna.

### Cadena perpetua: lo que se puede decir y lo que era una excusa

Yo había escrito que su AUC bajo *"no tiene arreglo estadístico porque con 78,6%
de apoyo queda poco que explicar"*. **Es una racionalización y Codex la
desarmó:** el AUC es invariante al balance de clases —cambiar la proporción
manteniendo las distribuciones de puntaje conserva el ordenamiento— y la propia
base da el contraejemplo: **humillación tiene 11,0% de apoyo y AUC 0,813**.

Lo medible: con la selección de `C` de producción, cadena perpetua da AUC 0,655,
Brier 0,15790 contra 0,16874 del predictor constante (**6,4% mejor**) y log-loss
0,49540 contra 0,52068 (**4,9% mejor**). Y hay 712 respuestas contrarias sobre
2.950, o sea que casos negativos hay de sobra.

La frase defendible es: **estos predictores y esta especificación aportan una
mejora predictiva modesta para cadena perpetua.** No se midió un límite
irreparable ni se demostró que lo cause el apoyo mayoritario.

### Lo que sigue sin hacerse

- **Errores estándar design-aware por linealización de Taylor.** El bootstrap
  estratificado respeta los estratos pero la base no trae conglomerados.
- ~~Recalibrar mano dura.~~ **HECHO** — ver la sección de abajo.
- **Cobertura real de los intervalos.** El bootstrap los genera pero nadie
  demostró que cubran el 95% frente a error de especificación. Son anchos
  —mediana de 27,5, 24,5, 29,3 y 12,5 pp según la pregunta, y percentil 90 de
  hasta 51 pp—, así que el caveat no es teórico.
- **Validar tasas por celda agrupada.** Las 1.008 combinaciones no se pueden
  validar una por una: la mediana de casos efectivos por perfil observado es
  ~2, y sólo entre 45 y 54 celdas llegan a 10. Habría que preagrupar.
- **Un conjunto de test separado de verdad.** La validación es cruzada, no
  out-of-sample sobre datos reservados; con n efectivo 576 reservar un test
  costaría más de lo que informa.
- **Bootstrap de la diferencia perfil−promedio nacional**, ya anotado arriba.

## Mano dura se sirve RECALIBRADA

De las cuatro, es la única. Rechazaba el contraste de Hosmer-Lemeshow ponderado
(p≈0,002): entre las personas a las que el modelo asignaba ~65%, la frecuencia
real no rondaba 65%. Para un widget cuya frase es literalmente *"el 65% de la
gente con este perfil"*, eso no es un detalle técnico — es que el número no
significaba lo que dice.

**El mapa:** siete nodos de igual masa ponderada, ajustados **fuera de muestra**,
con interpolación lineal monótona. Se declara en
`config.PREGUNTAS_A_RECALIBRAR`, que entra en la huella del contrato.

*Una versión anterior ajustaba una spline PCHIP para el centro y rectas entre
nodos para las réplicas del bootstrap. Codex midió el costo de esa
inconsistencia: hasta **4,00 pp** de diferencia en un extremo del intervalo entre
los 1.008 perfiles, y en el perfil por defecto llegaba a cambiar si el intervalo
cruzaba el 50% — que es la regla con la que el widget decide si afirma de qué
lado está la mayoría. Un número y su intervalo no pueden salir de dos curvas
distintas, así que ahora es una sola.*

| Métrica | Sin mapa | Con mapa |
|---|---:|---:|
| Hosmer-Lemeshow | 29,95 (p=0,002) | 12,02 (**p=0,280**) |
| Contraste sin bins | p=0,024 | **p=0,844** |
| log-loss | 0,52439 | **0,50415** |
| Brier | 0,16340 | **0,15982** |

**Qué cambió en pantalla:** el perfil que abre pasó de 65% (IC 53-76) a **70%
(IC 51-86)**. El intervalo se ensancha por dos motivos distintos: el mapa sube la
cola superior, y ahora incluye la incertidumbre de haber estimado el mapa. Los coeficientes y las réplicas son idénticos; el mapa baja un poco
la cola inferior y sube bastante la superior.

### Dos caminos que se descartaron, y por qué

- **Isotónica libre:** "arreglaba" el HL (p=0,50) pero un contraste **sin bins**
  la seguía rechazando con p=0,001 — estaba calzando los bins con los que se la
  evaluaba— y empeoraba el log-loss de 0,524 a 0,566.
- **Platt:** no alcanza. El defecto no es una pendiente: la curva cambia de signo
  y los dos tramos superiores hay que agruparlos. Una recta en escala logit no
  puede con esa forma.

### Qué se puede afirmar de la calibración, y qué no

**No se puede decir "quedó calibrada".** Se puede decir que **dos contrastes con
puntos ciegos distintos no la rechazan**:

| | Mano dura cruda | Mano dura publicada |
|---|---:|---:|
| Hosmer-Lemeshow ponderado | p=0,002 · **rechaza** | p=0,280 |
| Contraste sin bins (kernel sobre el logit) | p=0,024 · **rechaza** | p=0,844 |

El segundo se construyó acá y **su primera versión no servía**: usaba el máximo
del desvío suavizado y el control negativo mostró que era ciega a la forma en S,
que es justo el defecto de mano dura. Con la integral del desvío al cuadrado, el
control da tamaño 7% bajo la nula y potencia 62% contra un cambio de pendiente,
67% contra un corrimiento y 78% contra una S marcada.

**Pero 30% contra una S suave.** O sea que deja pasar siete de cada diez
descalibraciones de ese tipo. "No rechazó" no es "está calibrada", y la
diferencia importa porque el widget publica el número, no un ranking.

### Lo que NO hace este mapa

- **No se aplica a las otras tres.** No mostraron la misma necesidad, y aplicarlo
  a ciegas descalibraba pena de muerte (p 0,92 → 0,02).
- **El intervalo YA incluye la incertidumbre de estimar el mapa.** Cada réplica
  de coeficientes lleva su propia réplica del mapa, del mismo remuestreo. Sin
  eso el intervalo salía 2,5 pp más angosto de lo que corresponde (hasta 6,9 en
  el peor perfil).
  Lo que **sigue faltando**: las predicciones fuera de muestra sobre las que se
  ajusta el mapa están congeladas —vienen del modelo estimado sobre la muestra
  original—, así que no se captura cómo cambiarían al reestimar el pipeline
  entero. El efecto neto sobre el intervalo no tiene signo garantizado.

## Decisión de diseño: se actualiza en vivo, sin botón de confirmar

El Figma "Producto UY" dibuja un flujo con botones **"Confirmar"** y **"Volver a
empezar"**: el lector elige sus características y recién entonces pide el
resultado. **El widget NO lo implementa, y es a propósito.**

Tomer lo decidió el 7/9/2026: *"dejalo actualizando en vivo, sin botón de
confirmar"*. Cada cambio en un selector recalcula el porcentaje al instante.

No es un pendiente ni un olvido. Si alguien compara la pantalla contra el Figma y
ve que faltan los botones, la respuesta es ésta; **no hay que agregarlos** sin
que el cliente cambie de opinión.

Es la única divergencia deliberada respecto del Figma en el comportamiento. Las
divergencias visuales que quedan están anotadas en `shared/config.py`
(`OBSERVADOR_COLORS`: colores muestreados de la imagen, no inspeccionados) y en
`shared/styles.py` (el punto del radio queda azul).

## Decisión de diseño: el color no valora

Este widget **no** usa `shared.config.get_interpretation`, que pinta el apoyo de
verde y la oposición de rojo. Para el IVE es razonable; acá pintar de verde
"apoya la pena de muerte" sería tomar partido. `components.INTENSIDAD` usa una
escala de un solo tono, donde el color acompaña la magnitud sin calificarla.

## Publicación

Fuente y crédito se serializan dentro de cada JSON (`config.FUENTE`,
`config.CREDITO`) y salen en el pie del widget:

> Encuesta El Observador-UMAD-Ferreira sobre seguridad pública, mayo de 2026.
> Análisis y desarrollo: Juan Ignacio Pintos Elso.

Desplegado en **https://observador-seguridad.streamlit.app/**, una app de Streamlit Cloud aparte de la del IVE pero
sobre el mismo repo, con *Main file path* `widgets/seguridad/app.py`. Sigue la
rama `master`: lo que se ve ahí es lo que esté mergeado, no lo que haya en una
rama de trabajo.

El código de embed, ya con esa URL puesta y con las alturas medidas para
escritorio y celular, está en
[`docs/embed/seguridad-widget-embed.html`](../../docs/embed/seguridad-widget-embed.html).

## Correr

```bash
streamlit run widgets/seguridad/app.py
pytest widgets/seguridad/tests -q
```

## Revisión

Revisado adversarialmente por Codex el 2026-08-29 (cuatro hallazgos) y en rondas
posteriores hasta el 2026-08-31. Los detalles están en el historial de git; lo
que se corrigió entonces sigue vigente: la CV elegía `C` con log-loss sin
ponderar, una etiqueta Likert inesperada se volvía "neutral" en silencio, los sin
dato de víctima contaminaban la referencia, y los excluidos publicados no
cerraban contra el N.

Verificación complementaria, re-corrida el 7/9/2026 sobre los cuatro modelos
nuevos con `scripts/verificar_inferencia.py`: se comparan las predicciones de
sklearn re-entrenado contra la inferencia en Python puro sobre los 1.008 perfiles
elegibles de la UI. **Peor discrepancia: 0,0000000000 pp** en las cuatro
preguntas. Es el chequeo que agarra el modo de falla propio de no usar sklearn en
producción: que `build_features()` arme el vector en un orden distinto del que
tenía la matriz de entrenamiento, y los coeficientes se apliquen a la dummy
equivocada sin que nada lo delate.

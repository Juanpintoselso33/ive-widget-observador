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

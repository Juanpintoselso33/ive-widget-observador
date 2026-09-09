# Crear un widget nuevo — Guía rápida

## 1. Copiar el template

```bash
cp -r widgets/_template widgets/<nombre_widget>
```

## 2. Adaptar los 4 archivos

### `config.py`
- Cambiar `WIDGET_NAME` y `WIDGET_SLUG`
- Agregar los mapeos UI→código del modelo
- Verificar que `DATA_FILE` apunta al CSV correcto

### `model.py`
- Adaptar `predict_probability()` con las variables del nuevo modelo
- Si hay modelo secundario (neutralidad, etc.), agregar `predict_probability_neutral()`

### `components.py`
- Adaptar `render_inputs()` con los selectboxes del nuevo widget
- Adaptar `render_result()` con el label y UI apropiados

### `app.py`
- Actualizar `page_title`
- Actualizar los imports para apuntar a `widgets.<nombre_widget>.*`

## 3. Entrenar el modelo

Crear `widgets/<nombre>/train_model.py` adaptado a los nuevos datos y ejecutarlo.

## 4. Testear

```bash
streamlit run widgets/<nombre>/app.py
```

## 5. Deploy

En Streamlit Cloud, configurar el entry point como `widgets/<nombre>/app.py`.

## 6. Figma (opcional)

Usar el skill `figma:figma-generate-design` para generar mockups de la UI antes de codificar.
Usar `figma:figma-code-connect` para vincular componentes Figma con los componentes Python.

---

# Lo que aprendimos con el widget de seguridad

Tres cosas que costaron plata averiguar y que aplican a cualquier widget de
"armá tu perfil" hecho sobre una encuesta. Están acá y no en el README de
seguridad porque son de la familia entera, no de esa pregunta.

## 1. Antes de diseñar nada, dividí el N efectivo por la cantidad de celdas

Es la cuenta que decide si el widget puede mostrar un número por perfil, y hay
que hacerla **antes** de maquetar, no después.

En seguridad, según la pregunta: entre **2.672 y 2.969** respuestas con postura
definida, que el efecto de diseño de los ponderadores deja en entre **571 y 632
efectivas** (Kish). Repartidas entre **1.008** combinaciones que el lector puede
armar, da entre 0,57 y 0,63 casos efectivos por celda.

El resultado medido: el intervalo mide **entre 13 y 28 puntos porcentuales de
mediana incluso en los perfiles con 10 o más de peso muestral detrás**, y entre
el 43% y el 47% de las combinaciones no tienen un solo encuestado.

```bash
python widgets/seguridad/scripts/anchos_por_soporte.py
```

Los números salen de ahí y quedan en `scripts/salidas/anchos-por-soporte.json`.
No se citan de memoria: la primera versión de esta guía decía "21 a 28" y estaba
mal —se había perdido de vista humillación, que da 13,1— porque la medición se
había corrido con un script descartable. Lo marcó Codex.

**Lo que ese número prueba y lo que no.** Prueba que el ancho NO lo empujan los
perfiles que casi no existen: restringir a los mejor sostenidos casi no lo
achica, así que el problema no es la extrapolación y restringir la grilla no lo
resolvería. **No prueba que ninguna especificación pueda dar intervalos más
angostos**: el modelo comparte coeficientes entre perfiles y no estima 1.008
proporciones independientes. Tomalo como diagnóstico de arranque, no como
imposibilidad demostrada.

Si la cuenta da mal, el margen de maniobra es menos celdas o más muestra, y las
dos son decisiones del principio. Y si no se puede cambiar ninguna, el widget
igual sirve —como orientación cualitativa, no como medición—, pero conviene
saberlo antes de prometerle precisión a nadie.

Regla práctica: cada variable que se le agrega al formulario **multiplica** las
celdas. Seis variables con 4, 2, 3, 7, 3 y 2 categorías ya dan mil.

## 2. El intervalo puede gobernar sin mostrarse

El 9/9/2026 El Observador pidió sacar el intervalo de la pantalla: *"a la gente
no le sirve de nada y es difícil de entender"*. El diagnóstico era correcto —ver
el punto 1—, así que no era discutible achicándolo.

El patrón que quedó, y que conviene repetir:

- el intervalo **se calcula igual**, se calibra igual y se guarda igual;
- **no se muestra**;
- **decide qué afirma el widget**: si cruza el umbral editorial (el 50%, o el
  cero en una diferencia), la página dice que no se puede afirmar en vez de
  afirmar algo que el dato no sostiene.

Así el lector no ve aritmética, y lo que lee ya está filtrado por la
incertidumbre. La contra hay que decirla igual: se publica un número preciso sin
señal visible de cuánto se puede mover, y eso es una pérdida real, no un detalle.

## 3. Si escondés una cantidad, sacá TODAS las referencias a ella

El error que casi se publica: sacamos la línea del intervalo y la tarjeta siguió
diciendo dos veces *"el margen de error no permite afirmar..."*. Mandaba al
lector a mirar algo que ya no estaba en la pantalla, y era **peor** que antes de
esconderlo.

Las frases se reescribieron para decir **qué pasa**, no **con qué se calculó**:

Las dos frases completas, tal como salen de `components.py` (verificadas
ejecutando las funciones, no transcritas):

| | antes | ahora |
|---|---|---|
| mayoría | El margen de error no permite afirmar de qué lado está la mayoría en este perfil | Con estos datos no se puede afirmar de qué lado está la mayoría en este perfil |
| brecha | ↑ la estimación puntual queda 3pp por encima, pero el margen de error no permite afirmar la diferencia | ↑ la estimación da 3pp por encima, pero con estos datos no se puede afirmar la diferencia |

Y una consecuencia para los tests: si el texto publicado se va a poder retocar,
**los tests no pueden asertar la frase literal**. En seguridad las dos frases de
abstención comparten un arranque que vive en la constante `MARCA_ABSTENCION`, y
los tests verifican contra ella. Un test pegado a la redacción se pone rojo por
un cambio de estilo, y el arreglo tentador —copiar la frase nueva al test— lo
deja sin probar nada, en silencio.

**Pero una constante compartida abre su propio agujero, y hay que taparlo.** Con
`MARCA_ABSTENCION = ""`, la comparación `MARCA_ABSTENCION in texto` es verdadera
para CUALQUIER texto y los doce tests pasan sin probar nada. Es la misma familia
de error que se quería evitar, movida de lugar. Lo marcó Codex. Hace falta:

- una aserción de que la marca no está vacía y de que tiene forma de frase;
- un **control negativo**: un texto donde el widget SÍ afirma no puede contener
  la marca. Sin eso, "el texto contiene la marca" se cumple igual con la marca
  vacía.

# Diseño del widget — archivo Figma "Producto UY"

Valores leídos del panel de inspección de Figma el **8/9/2026**, no muestreados
de una imagen. Antes de esta fecha la hoja de estilos usaba una aproximación
sacada a ojo de una captura, y **las dos familias tipográficas estaban mal**.

- Archivo: `Producto UY`, página **Widget IVE**
  <https://www.figma.com/design/tHTq5gDqpw6jKbWt0sIt2l/Producto-UY?node-id=1-2>
- Frames: `Frame 427321461` (escritorio, **698 × 1292**) y `Frame 427321466`
  (mobile, **331 × 987**). Hay un tercero, `Nota Explainer`, que muestra el
  widget metido dentro de una nota.
- Exportados a 2x en esta misma carpeta:
  `figma-widget-escritorio-698px@2x.png` y `figma-widget-mobile-331px@2x.png`.

> **El frame es el widget del IVE**, no uno de seguridad: tiene "¿Tienes hijos?",
> "Balotaje 2019" y "Religiosidad". O sea que "respetar el Figma" quiere decir
> que el de seguridad se vea como éste, con sus propios campos.

## Tipografía

| Rol | Familia | Peso | Tamaño / interlineado | Color |
|---|---|---|---|---|
| Titular | **Libre Baskerville** | 700 Bold | 30 / 42 px | `#006B36` |
| Título de sección ("COMPARACIÓN CON OTROS GRUPOS") | **Libre Baskerville** | 700 Bold | 18 / 68 px, versalitas | `#1B1B19` |
| Bajada | **Instrument Sans Condensed** | 400 | 23 / 23 px | `#1B1B19` |
| Cuerpo, etiquetas de campo, "A favor" / "En contra", pastilla del gradiente | **Instrument Sans** | 400 | 19 / 35 px | `#1B1B19` |
| Valor elegido dentro del selector | Instrument Sans | 400 | 19 / 35 px | `#515151` |
| Número grande del resultado | Instrument Sans | **600 SemiBold** | 50 px | `#0D443B` |
| Botón y pastillas de comparación | Instrument Sans | 400 | 16,26 px, centrado | `#FFFFFF` |
| Diferencia contra el promedio ("↓2pp por debajo") | Instrument Sans | 400 | 17 / 36 px | según signo |
| Etiqueta de grupo ("NADA religioso") | Instrument Sans | 400 | 13 px | `#1B1B19` |
| Número de grupo ("65.9%") | Instrument Sans | **500 Medium** | 25 px | `#1B1B19` |
| Diferencia por grupo ("-9pp") | Instrument Sans | **700 Bold** | 17 px | según signo |

Las dos están en Google Fonts. `Instrument Sans` es variable y trae eje de
ancho, así que la Condensed sale con `wdth` 75 de la misma familia.

## Colores

| Token | Hex | Dónde |
|---|---|---|
| Fondo del formulario y de la tarjeta | `#FFFFFF` | 45% del área |
| Fondo de la zona de resultado y comparación | `#EDEDED` | 33% del área |
| Relleno de los selectores | `#F2F2F2` | |
| Borde de los selectores | `#D0CFCF` | 1 px |
| Verde de acento | `#006B36` | titular y filete superior |
| Verde sólido | `#0D443B` | botón, pastilla activa, número grande |
| Naranja | `#F57F00` | extremo "en contra" del gradiente **y** diferencias negativas |
| Azul | `#93B6EE` | extremo "a favor" del gradiente **y** diferencias positivas |
| Texto | `#1B1B19` | |
| Texto tenue | `#999998` | "¿Cómo funciona este modelo?" |

**Son dos verdes, no uno.** El `#006B36` es el de titular y filete; el `#0D443B`,
más oscuro y apagado, es el de las superficies sólidas y el número. Muestrear una
captura los promediaba y daba un tercer verde que no existe.

**Los colores del gradiente son los mismos que los de las diferencias**: naranja
para el lado "en contra" y azul para el "a favor". El diseño usa los dos para las
diferencias por grupo, pero **no demuestra qué signo lleva cada uno**: los cuatro
valores del mock dicen `-9pp` y aparecen dos en azul y dos en naranja. Que el azul
sea el positivo es una inferencia por analogía con los extremos del gradiente, no
algo que el archivo diga. Marcado porque lo señaló Codex y es una lectura mía.

## Geometría

| | Valor |
|---|---|
| Ancho escritorio / mobile | 698 px / 331 px |
| Filete superior del marco | 2 px `#006B36` |
| Sombra del marco y de la tarjeta | `0 5px 6px rgba(0,0,0,0.15)` |
| Selector | alto 37 px, radio 9 px, borde 1 px |
| Botón y pastillas | alto 30 px, radio 7 px, padding 9 px, gap 10 px |
| Tarjeta del resultado | radio 10 px, sin borde |
| Barra de gradiente | alto **31 px** en escritorio y **18 px** en móvil |

## Detalles de layout que no son obvios

- **El mobile mantiene DOS columnas de campos.** No apila a una sola.
- La grilla de comparación pasa de 4 columnas en escritorio a **2 × 2** en mobile.
- Sólo la zona de resultado y comparación va sobre el gris; el formulario queda
  sobre blanco. El corte es una banda de ancho completo, no una tarjeta.
- El pie del diseño dice "¿Cómo funciona este modelo?" con una flecha `→` y en
  gris tenue. **El widget no lo reproduce**: usa un `st.expander("Cómo se
  calcula")` con el chevron nativo de Streamlit y sin el gris. Es una diferencia
  conocida, no un descuido — el expander despliega la metodología y el enlace
  del Figma no lleva a ningún lado.
- **La pastilla del valor va ARRIBA de la barra**, no debajo: medido sobre los
  PNG, 65 filas de pastilla por encima y ninguna por debajo en el frame de
  escritorio.

## Lo que este archivo no define

No trae estados (hover, foco, error), ni el ancho intermedio entre 331 y 698, ni
qué pasa arriba de 698. Todo eso queda a criterio y está resuelto en la hoja de
estilos, no acá.

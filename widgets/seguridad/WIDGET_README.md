# Widget de seguridad pública — El Observador

Estima la probabilidad de apoyar una medida punitiva según el perfil de quien lo
usa. Mismo molde que el widget IVE: regresión logística ponderada, coeficientes
serializados a JSON, inferencia en Python puro.

Pedido por Tomer Urwicz (19/8/2026) como contraparte del widget del IVE, para
publicar con los resultados de la encuesta de seguridad de mayo de 2026.
Nicolás Trajtenberg iba a opinar sobre qué pregunta rinde más — por eso la
pregunta está parametrizada y no hardcodeada.

## Cambiar la pregunta

Editar `PREGUNTA_ACTIVA` en `config.py` y volver a entrenar:

```bash
python widgets/seguridad/train_model.py
```

Candidatas ya cargadas en `PREGUNTAS`:

| Slug | Variable | Pregunta |
|---|---|---|
| `pena_muerte` *(activa)* | `var_229` | Pena de muerte por homicidio |
| `cadena_perpetua` | `var_230` | Cadena perpetua por tres delitos |
| `aumentar_penas` | `var_228` | Aumentar penas para todos los delitos |
| `politico_mano_dura` | `var_233` | Votaría a un político de mano dura |
| `humillacion_presos` | `var_231` | Los presos merecen ser humillados |

Todas comparten la escala Likert 1-5, así que el pipeline no cambia. Agregar una
pregunta nueva es sumar una entrada al dict con `columna`, `titulo`, `afirma` y
`verbo`; hay un test que verifica que ninguna quede incompleta.

`test_config.py::test_el_json_corresponde_a_la_pregunta_activa` falla a propósito
si el JSON entrenado no corresponde a `PREGUNTA_ACTIVA`, para que no se publique
un widget con el título de una pregunta y los coeficientes de otra.

## Datos

`base_etiquetada.csv` del repo de encuestas, que **no vive en este repo** (son
datos del cliente y el `.gitignore` excluye `*.csv`):

```
Observador-encuesta/encuestas/observador_2026_05_seguridad/output/base_etiquetada.csv
```

Se puede pisar la ruta con la variable de entorno `SEGURIDAD_DATA_FILE`.

## Modelo

Regresión logística binaria con penalización L2, `sample_weight=w_norm`, C
elegido por CV estratificada de 5 folds sobre `neg_log_loss`.

**El ponderador no es opcional.** La muestra cruda está sesgada a montevideanos
(67% contra 44,5% ponderado), hombres (63,5% contra 49,2%) y mayores de 60 (39%
contra 28,9%). Sin ponderar, el apoyo a la pena de muerte da 26,9%; ponderado,
36,7%. Diez puntos de diferencia en el número que se publica.

Dependiente: a favor si Likert ≥ 4, en contra si ≤ 2. Los neutrales (Likert = 3)
se excluyen del modelo principal y se modelan aparte, igual que en el IVE, así
que el porcentaje es **condicional a tener postura definida**.

Predictores, con su categoría de referencia entre paréntesis: edad en cuatro
tramos (18-29), sexo (hombre), educación en cuatro niveles (primaria o menos),
autoubicación ideológica agrupada (centro), víctima de delito en 12 meses (no
fue víctima) y región (interior).

Los sin dato llevan dummy propia (`ideol_no_ubica`, `victima_sin_dato`) en vez
de caer en la referencia. `victima_sin_dato` queda siempre en cero desde la UI
—el widget obliga a elegir una opción real— y existe sólo para que los 53 casos
sin respuesta no se mezclen con quienes contestaron "No", cosa que corría tanto
la categoría de referencia como la tasa publicada de ese grupo (34,2% contra el
34,6% real).

## Revisión

Revisado adversarialmente por Codex el 2026-08-29, que encontró cuatro
problemas, todos corregidos:

1. **La CV elegía `C` con log-loss sin ponderar.** `sample_weight` llegaba al
   `fit` de cada fold, pero el scorer lo ignoraba porque el metadata routing
   está deshabilitado por defecto: entrenaba ponderado y evaluaba sin ponderar.
   Ahora los folds se recorren a mano con `log_loss(..., sample_weight=w[test])`.
   En este dataset ambos caminos eligen `C=0.1`, así que los coeficientes no
   cambiaron — el riesgo era para los re-entrenamientos con otras preguntas.
   De paso saca la dependencia de `params=`, que exige scikit-learn ≥ 1.4
   mientras `requirements.txt` declara 1.3 como piso.
2. **Una etiqueta Likert inesperada se volvía "neutral" en silencio.** Ahora el
   entrenamiento aborta si aparece un valor fuera de `LIKERT_MAP`.
3. **Los sin dato de víctima contaminaban la referencia y el grupo publicado**
   (ver arriba).
4. **Los excluidos publicados eran 671 y son 705** (671 neutrales + 34 sin
   respuesta). Se exportan por separado y hay un assert más un test que
   verifican que los totales cierren contra el N de la encuesta.

Verificación propia complementaria: se comparan las predicciones de sklearn
re-entrenado contra la inferencia en Python puro sobre los **384 perfiles
posibles de la UI**; la peor discrepancia es 0,000000 pp.

## Caveats

- **El mapeo de `nivel_educativo`** (escala 1-10 → 4 categorías) se hereda del
  widget IVE y fue **inferido**: el proveedor no entregó codebook. La jerarquía
  se sostiene, las etiquetas podrían no ser exactas.
- **Pseudo-R² de McFadden ≈ 0,16**, bastante menor que el 0,37 del IVE. Es
  esperable: allá la religiosidad era un predictor potentísimo y acá no hay nada
  equivalente. Conviene decirlo, no maquillarlo.
- **Sin errores estándar ni p-valores**: Ridge no los provee analíticamente, y
  los pesos se tratan como frecuencias.

## Decisión de diseño: el color no valora

Este widget **no** usa `shared.config.get_interpretation`, que pinta el apoyo de
verde y la oposición de rojo. Para el IVE es razonable; acá pintar de verde
"apoya la pena de muerte" sería tomar partido. `components.INTENSIDAD` usa una
escala de un solo tono, donde el color acompaña la magnitud sin calificarla.

## Correr

```bash
streamlit run widgets/seguridad/app.py
pytest widgets/seguridad/tests -q
```

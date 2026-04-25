# CLAUDE.md — Widget IVE El Observador

## Descripción del proyecto

Widget interactivo "Construí tu votante" para **El Observador** (Uruguay), inspirado en el "Build a Voter" de The Economist para las elecciones estadounidenses de 2024.

El widget estima la **probabilidad de que una persona apoye el derecho de la mujer a decidir sobre su embarazo** (IVE — Interrupción Voluntaria del Embarazo, legalizada en Uruguay en 2012) en función de sus características sociodemográficas y actitudinales. El usuario selecciona 8 atributos propios y recibe un porcentaje personalizado de apoyo, comparado con el promedio nacional y con subgrupos.

## Stack tecnológico

- **Lenguaje:** Python 3.11
- **UI:** Streamlit (embebido vía `<iframe>` en artículos de El Observador)
- **Dependencias ML:** scikit-learn, pandas, numpy
- **Inferencia en prod:** JSON de coeficientes + lógica Python pura (sin sklearn en runtime)
- **Tests:** pytest con coeficientes sintéticos

## Estructura del proyecto

```
ive_widget/
├── app.py                    # Punto de entrada Streamlit
├── model.py                  # Lógica de predicción (sin sklearn, solo JSON + numpy)
├── train_model.py            # Pipeline de entrenamiento v2 (modelo actual)
├── model_coefficients.json   # Coeficientes serializados + metadatos
├── components.py             # Renderizado UI (barra gradiente, cards, tabs)
├── config.py                 # Paleta, rutas, umbrales de interpretación
├── styles.py                 # CSS customizado (estética El Observador / The Economist)
├── scripts/
│   ├── screening_variables.py   # Screening univariado e incremental de variables
│   ├── validacion_ordinal.py    # Validación logit ordinal (Likert 1-5) vs binario
│   ├── train_model_backup.py    # Versión v1 (legacy, solo referencia)
│   ├── validar_mapeo.py         # Validación del mapeo de voto 2019
│   ├── verificar_voto.py        # Verificación cruzada con datos históricos Factum
│   └── buscar_elecciones.py     # Exploración datos electorales
└── tests/
    ├── conftest.py              # Fixtures con coeficientes sintéticos
    ├── test_model.py            # Tests de predicción
    └── test_config.py           # Tests de configuración
```

## Modelo estadístico

### Fuente de datos

- **Dataset:** `base_limpia.csv` (directorio padre), encuesta El Observador Uruguay 2025–2026, ~3.300 casos; n efectivo = 2.802 tras exclusiones.
- **Archivo fuente original:** `muestra con weights 8- ENE 2026.xlsx`
- **Ponderación:** Variable `w_norm` (peso muestral normalizado, post-estratificación/calibración).

### Variable dependiente

Item `P174_Decidir_embarazo` (Likert 1–5):
- `favor_ive = 1` si Likert ≥ 4
- `favor_ive = 0` si Likert ≤ 2
- Excluidos (Likert = 3, "ni de acuerdo ni en desacuerdo"): ~22% de la muestra

Proporción ponderada de apoyo (entre quienes tienen posición definida): **76.55%**

### Modelo

**Regresión logística binaria con penalización L2 (Ridge), ponderada por diseño muestral.**

- **Solver:** L-BFGS, `max_iter=2000`, `random_state=42`
- **Regularización:** `C=0.5` (seleccionado por CV)
- **Pesos:** `sample_weight=w_norm`
- **Métricas:** McFadden pseudo-R² = 0.3685, CV neg-log-loss = −0.3543 (±0.0237)

### Predictores (20 variables, referencia entre paréntesis)

| Grupo | Variables | Referencia |
|-------|-----------|------------|
| Edad | edad_25_34, edad_35_44, edad_45_54, edad_55_plus | 18–24 |
| Sexo | es_mujer | Hombre |
| Educación | educ_bach_incomp, educ_bach_comp, educ_ter_incomp, educ_ter_comp | Primaria o menos |
| Religiosidad | relig_poco, relig_bastante, relig_mucho | Nada religioso |
| Región | es_montevideo | Interior |
| Hijos | tiene_hijos | Sin hijos |
| Hogar | hogar_3_4, hogar_5_plus | 1–2 personas |
| Balotaje 2019 | balotaje_martinez, balotaje_lacalle | Otros/blanco/no votó |
| Interacciones | mujer_x_relig_mucho, mujer_x_tiene_hijos | — |

### Efectos más relevantes (odds ratios)

- **Mayor:** `relig_mucho` OR=0.056 (muy religioso vs nada: 18× menos chances de apoyar)
- **Gender gap:** `es_mujer` OR=2.110 (las mujeres casi duplican las chances de apoyo)
- **Voto 2019:** `balotaje_martinez` OR=2.762, `balotaje_lacalle` OR=0.482
- **Educación:** `educ_ter_comp` (terciaria completa o más vs primaria o menos) — ver `model_coefficients.json` para OR actual tras recodificación a 5 categorías
- **Región:** `es_montevideo` OR=2.014

## Selección de hiperparámetros

5-fold stratified CV (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`), grilla `C ∈ {0.01, 0.1, 0.5, 1.0, 5.0, 10.0}`, scoring = `neg_log_loss` con pesos muestrales. `best_C = 0.5`.

## Diferencias v1 → v2

1. Variables ordinales como dummies (no escala 1–5 lineal)
2. Voto de balotaje 2019 (en lugar de partido primera vuelta)
3. Dos interacciones: `mujer×relig_mucho` y `mujer×tiene_hijos`
4. McFadden corregido: null LL con media ponderada (no simple)
5. C seleccionado por CV (no fijo en 1.0)

## Inferencia (app)

`model.py` lee `model_coefficients.json` vía `@st.cache_data`, construye el vector de 20 dummies/interacciones a partir de los 8 inputs del usuario, y evalúa:

```python
z = intercept + sum(coef[k] * x[k] for k in predictors)
prob = 1 / (1 + exp(-z))
```

Sin dependencia de sklearn en runtime.

## Umbrales de interpretación (config.py)

| Probabilidad | Etiqueta |
|---|---|
| ≥ 70% | muy probable que apoyes |
| ≥ 55% | probable que apoyes |
| ≥ 45% | dividido/a |
| ≥ 30% | probable que te opongas |
| < 30% | muy probable que te opongas |

## Modelo secundario: probabilidad de neutralidad

Junto al modelo principal de apoyo se entrena un **logit auxiliar** que predice
P(NS-NC) sobre los mismos 20 predictores y la misma penalización Ridge (C=0.5,
sample_weight=w_norm). La variable dependiente es:

- `es_neutral = 1` si Likert == 3 ("ni de acuerdo ni en desacuerdo") o falta respuesta
- `es_neutral = 0` si Likert ∈ {1, 2, 4, 5}

Se exporta en `model_coefficients.json` bajo claves separadas:
`coefficients_neutral`, `odds_ratios_neutral`, `model_info_neutral`,
`prob_neutral_nacional`. La UI lo muestra como dato secundario discreto bajo
el resultado principal: *"X% de personas con tu perfil no toma posición clara
sobre el tema."* Hace explícito que el % de IVE es **condicional a tener
postura definida**, no marginal sobre la población total.

Métricas: pseudo-R² ≈ 0.10 (esperable: la neutralidad es más ruidosa),
tasa nacional ponderada ≈ 19%.

## Validación interna (no-producción): logit ordinal

`scripts/validacion_ordinal.py` ajusta un modelo logit ordinal (proportional
odds) sobre la escala completa Likert 1–5 (sin excluir neutrales) y compara
sus coeficientes con los del modelo binario de producción.

**Resultado actual (con 20 predictores, post colapso CB → Bach incompleto):**
- 18/20 variables con mismo signo entre binario y ordinal
- **Top-5 predictores idénticos en ranking exacto:** relig_mucho, relig_bastante,
  balotaje_martinez, hogar_5_plus, educ_ter_comp
- Spearman top-5 = 1.0; Spearman global = 0.50
- Discrepancias en variables de baja magnitud (edad_35_44, edad_55_plus) y
  atenuación esperable en `balotaje_lacalle` (la asociación Lacalle→contra IVE
  se diluye al usar toda la escala porque muchos votantes Lacalle responden
  "de acuerdo" pero no "totalmente de acuerdo")

**Conclusión:** la dicotomización Likert ≥4 / ≤2 + exclusión de neutrales **no
invierte la estructura principal de asociaciones**. El binario es defendible
para la pregunta de interés (probabilidad de apoyar) y la pérdida de
información ocurre sobre todo en el gradiente "de acuerdo" vs "totalmente de
acuerdo", que no es relevante para el widget.

**Caveat técnico:** statsmodels OrderedModel no soporta `sample_weight`; la
validación se ejecuta sin pesos. La consistencia de signos y rangos top
sugiere que esto no afecta la conclusión cualitativa.

## Caveats metodológicos

- Neutrales excluidos: probabilidades condicionales a tener posición definida
- Sin p-valores ni errores estándar (Ridge no provee SEs analíticos)
- Sin conjunto de test separado (validación solo por CV + métricas in-sample)
- Pesos tratados como frecuencias (`sample_weight`); SEs design-aware requieren Taylor lineal o bootstrap de diseño (no implementado)
- El widget advierte explícitamente: "probabilidades basadas en correlaciones estadísticas, no predicciones individuales"

## Recodificación de educación (v2.1, 2026-04-24; iterada 2026-04-25 a 7 cat. y revertida a 5 cat.)

Se exploró pasar de la `nivel_educ` original (5 cats colapsadas por el proveedor) a una versión más fina sobre `nivel_educativo` (escala 1–10), abriendo Ciclo Básico (CB) y Bachillerato incompleto. Diagnóstico (`scripts/diagnostico_educ.py`, `scripts/test_sin_ridge.py`):

- Tasas crudas ponderadas de apoyo IVE eran monotónicas pero CB-Bach formaban una **meseta** (~72–75%, gap < 3pp).
- Coeficientes ajustados (controlando religiosidad, balotaje, edad) **invierten** levemente la jerarquía (CB > Bach inc, ~0.03–0.04 logit) tanto con Ridge como con MLE puro: la inversión es señal real, no artefacto, pero <1pp en probabilidad.
- Confounder principal: CB tiene mayor proporción de muy religiosos (12.7%) que Bach inc (9.5%); al controlar por religiosidad, CB se infla.

**Decisión final (5 categorías):** se vuelve a colapsar CB inc + CB comp + Bach inc en una única **"Bachillerato incompleto"**. La distinción ofrecía mejora marginal, ruido en la presentación y una inversión contraintuitiva que costaba más explicar que el valor que aportaba.

| Cat. UI | Codes `nivel_educativo` |
|---|---|
| Primaria o menos *(ref)* | 1, 2 |
| Bachillerato incompleto | 3, 4, 5 |
| Bachillerato completo | 6 |
| Terciaria incompleta | 7 |
| Terciaria completa o más | 8, 9, 10 |

**Etiquetas inferidas, no documentadas en el dataset.** El proveedor no entregó codebook para `nivel_educativo` (escala 1–10). Las etiquetas se infirieron a partir de:
1. Escala estándar del INE/Mineduc Uruguay (Sin instrucción / Primaria / CB inc / CB comp / Bach inc / Bach comp / Ter inc / Ter comp / Posg inc / Posg comp).
2. Cross-tabulación 1:1 con la `nivel_educ` colapsada del proveedor: `{1,2}↔1-PRIMARIA`, `{3,4,5}↔2-EMS INCOMP`, `6↔3-EMS COMP`, `7↔4-TER INCOMP`, `{8,9,10}↔5-TER COMP`.
3. Monotonía del apoyo al IVE en `model_coefficients.json` (ver `stats_by_group`).

**Riesgo asumido:** si el codebook real definiera otras etiquetas para los códigos 1–10, los nombres mostrados al usuario podrían ser incorrectos, aunque la jerarquía y la lógica de la regresión se mantendrían. Antes de publicación, conviene confirmar con el proveedor.

## Convenciones de código

- Training script es `train_model.py`; `scripts/train_model_backup.py` es legacy, no modificar
- Variable `dpto == 19` es Montevideo (v2); v1 usaba `dpto == 1` (bug corregido)
- Tests usan coeficientes sintéticos (no el JSON de producción) vía `conftest.py`
- No hay sklearn en producción (`model.py` es Python puro + JSON)

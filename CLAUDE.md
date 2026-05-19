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
ive_widget/                         # Raíz del repo (plataforma multi-widget)
├── shared/                         # Código editorial compartido
│   ├── styles.py                   # CSS (IBM Plex, Economist-style)
│   └── config.py                   # Paleta de colores + umbrales
├── widgets/
│   ├── ive/                        # Widget IVE — el original
│   │   ├── app.py                  # Entry standalone: streamlit run widgets/ive/app.py
│   │   ├── model.py                # Predicción (Python puro + JSON)
│   │   ├── components.py           # UI Streamlit
│   │   ├── config.py               # Config IVE (rutas, balotaje)
│   │   ├── train_model.py          # Pipeline de entrenamiento
│   │   ├── model_coefficients.json # Coeficientes serializados
│   │   └── tests/                  # Tests con coeficientes sintéticos
│   └── _template/                  # Scaffold para nuevos widgets
│       ├── app.py / model.py / components.py / config.py
│       └── WIDGET_README.md        # Guía para crear widget nuevo
├── app.py                          # Entry del deploy actual (→ IVE via shared/ + widgets/ive/)
├── scripts/                        # Scripts de análisis (sin cambios)
├── docs/
│   ├── widget-catalog.md           # Registro de widgets
│   └── bmad-output/                # Artefactos BMAD
└── requirements.txt
```

## Workflow para crear un widget nuevo

1. `cp -r widgets/_template widgets/<nombre>`
2. Adaptar `config.py`, `model.py`, `components.py`, `app.py`
3. Crear `widgets/<nombre>/train_model.py` y entrenarlo
4. Testear: `streamlit run widgets/<nombre>/app.py`
5. Registrar en `docs/widget-catalog.md`

### Entry points

| Quiero... | Comando |
|-----------|---------|
| Correr el IVE widget (actual deploy) | `streamlit run app.py` |
| Correr el IVE widget standalone | `streamlit run widgets/ive/app.py` |
| Correr widget nuevo | `streamlit run widgets/<nombre>/app.py` |
| Correr tests IVE | `pytest widgets/ive/tests/ -v` |

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
- **Métricas:** McFadden pseudo-R² = 0.3687, CV neg-log-loss = −0.3539 (±0.0240)

### Predictores (19 variables, referencia entre paréntesis)

| Grupo | Variables | Referencia |
|-------|-----------|------------|
| Edad | edad_25_34, edad_35_44, edad_45_54, edad_55_plus | 18–24 |
| Sexo | es_mujer | Hombre |
| Educación | educ_secundaria, educ_ter_incomp, educ_ter_comp | Primaria o menos |
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
- **Educación:** `educ_ter_comp` OR≈2.21 (terciaria completa o más vs primaria o menos); `educ_secundaria` OR≈1.73; `educ_ter_incomp` OR≈1.98
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

`model.py` lee `model_coefficients.json` vía `@st.cache_data`, construye el vector de 19 dummies/interacciones a partir de los 8 inputs del usuario, y evalúa:

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
P(NS-NC) sobre los mismos 19 predictores y la misma penalización Ridge (C=0.5,
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

**Resultado actual (19 predictores, post colapso Bach inc + Bach comp → Secundaria):**
- 17/19 variables con mismo signo entre binario y ordinal
- **Top-5 predictores idénticos en ranking exacto:** relig_mucho, relig_bastante,
  balotaje_martinez, hogar_5_plus, educ_ter_comp
- Spearman top-5 = 1.0; Spearman global ≈ 0.52
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

## Recodificación de educación (iterada 2026-04-24 → 2026-04-25, final 4 cat.)

Se exploró abrir la `nivel_educ` original del proveedor (5 cats colapsadas) sobre la escala fina `nivel_educativo` (1–10). Tras varias iteraciones (7 cats, 5 cats) se llegó a 4 categorías. Diagnósticos: `scripts/diagnostico_educ.py`, `scripts/diagnostico_bach.py`, `scripts/test_sin_ridge.py`.

**Iteración 1 (7→5 cat):** colapsar CB inc + CB comp + Bach inc en "Bachillerato incompleto". CB-Bach formaban una meseta (~72–75%) y los coeficientes ajustados invertían CB > Bach inc por confounder de religiosidad.

**Iteración 2 (5→4 cat, final):** colapsar Bach inc + Bach comp en una sola **"Secundaria"**. Las tasas crudas eran monotónicas (73.3% vs 76.7%) pero el efecto neto controlando por sexo, religión, balotaje y región era ~0 con leve inversión (gap ≈ 0.024 logit, <0.5pp probabilidad). Confounder principal: bach_comp tiene mayor proporción de mujeres (62.9% vs 48.3% en bach_inc), urbanos (48.3% vs 42.3%) y nada religiosos (31.8% vs 26.4%); su "ventaja cruda" es composición, no efecto educativo neto. Sin controles la jerarquía es la esperada (bach_comp > bach_inc, gap ≈ 0.18 logit en MLE puro), confirmando que el flip es mediación de confounders.

| Cat. UI | Codes `nivel_educativo` |
|---|---|
| Primaria o menos *(ref)* | 1, 2 |
| Secundaria | 3, 4, 5, 6 |
| Terciaria incompleta | 7 |
| Terciaria completa o más | 8, 9, 10 |

Coeficientes resultantes monotónicos: `educ_secundaria` 0.545 → `educ_ter_incomp` 0.681 → `educ_ter_comp` 0.795.

**Etiquetas inferidas, no documentadas en el dataset.** El proveedor no entregó codebook para `nivel_educativo` (escala 1–10). El mapeo se infirió de la escala estándar INE/Mineduc Uruguay y de la cross-tabulación 1:1 con la `nivel_educ` colapsada del proveedor.

**Riesgo asumido:** si el codebook real definiera otras etiquetas para los códigos 1–10, los nombres podrían ser incorrectos, aunque la jerarquía y la lógica de la regresión se mantendrían. Antes de publicación, conviene confirmar con el proveedor.

## Limpieza de outliers en edad

`train_model.py` setea a NaN los valores de `edad < 18` o `edad > 110` antes de cualquier feature. La encuesta tenía 2 casos con valores tipo año-de-nacimiento o fecha codificada como número (e.g. 1966, 26091962). No afectaba al modelo principal (los tramos de edad son categóricos y caían en `55+` o NaN), pero sí contaminaba `stats_by_group` y promedios crudos.

## Convenciones de código

- Training script es `train_model.py`; `scripts/train_model_backup.py` es legacy, no modificar
- Variable `dpto == 19` es Montevideo (v2); v1 usaba `dpto == 1` (bug corregido)
- Tests usan coeficientes sintéticos (no el JSON de producción) vía `conftest.py`
- No hay sklearn en producción (`model.py` es Python puro + JSON)

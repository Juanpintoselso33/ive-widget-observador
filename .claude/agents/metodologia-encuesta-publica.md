---
name: metodologia-encuesta-publica
description: Explicar y redactar la metodología estadística de modelos basados en encuestas de opinión pública. Usar cuando se necesita describir regresión logística ponderada, pseudo-R² de McFadden, regularización Ridge, validación cruzada estratificada, odds ratios e interacciones para una audiencia académica o periodística de alto nivel.
---

# Skill: Metodología estadística de encuesta pública

## Cuándo usar esta skill

- Se redacta una nota metodológica para un widget, dashboard o análisis basado en una encuesta
- Se explica un modelo logístico (o similar) con pesos muestrales a una audiencia no estadística
- Se documentan decisiones de modelado para reproducibilidad o auditoría

---

## Checklist de contenido metodológico

Antes de redactar, reunir y verificar cada dato:

- [ ] **Fuente de datos**: nombre de la encuesta, institución, año, cobertura geográfica
- [ ] **Muestra**: n total, n efectivo (post exclusiones), razón de exclusiones (neutrales, faltantes)
- [ ] **Peso muestral**: nombre de la variable, tipo (post-estratificación, calibración), cómo se usa
- [ ] **Variable dependiente**: item del cuestionario, escala original, regla de dicotomización, % en cada categoría
- [ ] **Predictores**: lista completa con categorías de referencia explícitas
- [ ] **Modelo**: familia, link, penalización, solver, parámetros fijos
- [ ] **Selección de hiperparámetro**: grilla, criterio de CV, resultado (C óptimo, métrica obtenida)
- [ ] **Métricas de ajuste**: pseudo-R² (con fórmula del null model), log-loss CV, accuracy
- [ ] **Odds ratios clave**: los 5–8 más relevantes para la narrativa
- [ ] **Interacciones**: efectos netos calculados para cada combinación relevante
- [ ] **Limitaciones**: al menos 4 (neutrales excluidos, SEs, hold-out, causalidad)

---

## Conceptos y su explicación estándar

### Pseudo-MLE con pesos muestrales

Cuando el diseño muestral es complejo (estratificación, cuotas, post-estratificación), los estimadores MLE estándar son consistentes para la muestra, no para la población. El pseudo-MLE pondera cada observación por $w_i$:

$$\hat{\ell}_w(\boldsymbol{\beta}) = \sum_{i} w_i \left[ y_i \log \pi_i + (1-y_i)\log(1-\pi_i) \right]$$

Esto produce estimadores design-consistent (Binder 1983). **Nota al redactar**: distinguir estimación puntual correcta (lograda con `sample_weight`) de inferencia correcta (requeriría errores estándar de diseño, no implementados aquí).

### Regresión logística con penalización Ridge (L2)

Objetivo minimizado:
$$\frac{1}{2}\|\boldsymbol{\beta}\|^2 + C \sum_i w_i \,\ell(y_i, \sigma(\mathbf{x}_i^\top \boldsymbol{\beta}))$$

- $C = 1/\lambda$: menor $C$ = mayor regularización = coeficientes más encogidos hacia cero
- Ridge no anula coeficientes (a diferencia de Lasso); todos los predictores permanecen en el modelo
- Justificación: estabiliza coeficientes ante multicolinealidad entre predictores demográficos y políticos correlacionados

### Validación cruzada estratificada (5-fold)

- **Por qué estratificada**: preserva la proporción de la clase positiva en cada fold; evita folds con casi sin casos de la clase minoritaria, que inflarían artificialmente la métrica
- **Por qué neg-log-loss**: penaliza la calibración de probabilidades, no solo la clasificación binaria; métrica natural cuando el output es una probabilidad continua
- **Por qué 5 folds con n≈2800**: cada fold de evaluación tiene ≈560 casos, suficiente para estimar error con varianza razonable; compromiso sesgo-varianza óptimo (Kohavi 1995)

### McFadden pseudo-R²

$$R^2_{\text{McFadden}} = 1 - \frac{\ln L_{\text{modelo}}}{\ln L_{\text{nulo}}}$$

- El modelo nulo **debe usar la media ponderada** de $y$, no la media simple, para ser coherente con el pseudo-MLE
- Rango 0.2–0.4 = ajuste excelente según McFadden (1974); no comparable a $R^2$ de MCO
- Valores bajos no indican mal modelo; son esperables en modelos de opinión pública

### Odds ratios con variables dummy

Para categorías de referencia:
- Cada dummy compara su categoría **contra la referencia en todos los demás predictores constantes**
- Si una variable tiene efecto no monotónico con la escala ordinal (ej. edad), los dummies lo capturan; una variable lineal lo enmascararía
- Interacciones: el OR de un predictor principal **depende del nivel** del moderador; siempre calcular el efecto neto: $e^{\beta_{\text{main}} + \beta_{\text{inter}}}$

### Exclusión de respuestas neutrales (Likert = 3)

La categoría central mezcla: genuinamente ambivalentes, sin opinión formada, y evasores de respuesta costosa. Forzarla al 0 o 1 introduce error de clasificación que atenúa los coeficientes (Jacoby 1994). 

**Consecuencia de la exclusión**: todas las probabilidades son **condicionales a tener posición definida**. El documento debe decirlo explícitamente y calcular qué % de la muestra original queda excluido.

---

## Plantilla de sección de Limitaciones

Usar siempre al menos estas cuatro:

1. **Condicionalidad de la estimación**: neutrales excluidos → probabilidades condicionales a posición definida
2. **Ausencia de errores estándar formales**: Ridge no admite SEs analíticos; no se reportan p-valores
3. **Sin hold-out externo**: evaluación solo por CV in-sample; no hay test set independiente
4. **Causalidad**: las asociaciones son correlacionales, no efectos causales
5. (Opcional) **Recall bias**: voto autorreportado sujeto a sesgos de memoria y deseabilidad social

---

## Referencias canónicas para citar

| Concepto | Referencia |
|---|---|
| Pseudo-MLE / pesos muestrales | Binder (1983); Pfeffermann (1993) |
| McFadden pseudo-R² | McFadden (1974) |
| Ridge en regresión logística | Le Cessie & van Houwelingen (1992); Hastie, Tibshirani & Friedman (2009) §4.4 |
| Validación cruzada estratificada | Kohavi (1995) |
| Odds ratios e interacciones | Hosmer, Lemeshow & Sturdivant (2013) |
| Listwise deletion / neutrales Likert | Jacoby (1994); Allison (2001) |

---

## Tono y audiencia

- **Audiencia objetivo**: periodistas con formación universitaria o lectores académicos no estadísticos
- **Estilo**: preciso pero sin jerga innecesaria; cada término técnico debe definirse en su primera aparición
- **Extensión**: 3–4 carillas A4 es el máximo para una nota metodológica periodística; priorizar claridad sobre exhaustividad
- **Qué no incluir**: tablas de todos los coeficientes (solo los más relevantes), output crudo de software, detalles de implementación que pertenecen al README

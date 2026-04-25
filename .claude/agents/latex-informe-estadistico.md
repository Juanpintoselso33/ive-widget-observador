---
name: latex-informe-estadistico
description: Redactar informes estadísticos académicos en LaTeX. Usar cuando se pide escribir una nota metodológica, apéndice técnico o documento de análisis con fórmulas, tablas y referencias bibliográficas en formato LaTeX.
---

# Skill: Redacción de informe estadístico en LaTeX

## Cuándo usar esta skill

- El usuario pide un documento LaTeX con contenido estadístico (metodología, resultados, apéndice técnico)
- Se necesita escribir fórmulas, tablas de coeficientes, o citas bibliográficas en formato formal
- El output final es un `.tex` compilable con `pdflatex` o `xelatex`

---

## Checklist obligatorio antes de escribir

- [ ] Confirmar el **directorio de destino** (siempre guardar en `doc/`, no en la raíz del proyecto)
- [ ] Confirmar el **idioma** (español: usar `babel[spanish,es-tabla]`)
- [ ] Confirmar si se requiere **`\maketitle`** o un encabezado personalizado
- [ ] Revisar el CLAUDE.md del proyecto para extraer: n efectivo, métricas del modelo, predictores, ORs clave
- [ ] Tener el resumen de research estadístico disponible antes de escribir

---

## Estructura estándar del documento

```latex
\documentclass[11pt,a4paper]{article}

% Paquetes obligatorios
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[spanish,es-tabla]{babel}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{array}
\usepackage{microtype}
\usepackage{hyperref}
```

### Secciones para nota metodológica estadística (orden canónico)

1. **Introducción** — contexto periodístico/investigativo, objetivo del documento
2. **Datos y muestra** — fuente, n total, n efectivo, descripción del diseño
3. **Construcción de la variable dependiente** — cómo se derivó el outcome (incluir fórmula si es binaria desde Likert)
4. **Especificación del modelo** — familia, link function, función objetivo, penalización si la hay
5. **Selección del hiperparámetro** (si aplica) — CV, grilla, criterio, resultado
6. **Ajuste del modelo** — métricas (pseudo-R², log-loss, accuracy); incluir fórmula del pseudo-R²
7. **Interpretación de los coeficientes** — odds ratios, interacciones, tabla booktabs
8. **Limitaciones** — listado numerado, honesto y explícito

---

## Reglas de estilo y formato LaTeX

### Ecuaciones
- Usar `\begin{equation}` para ecuaciones numeradas (las principales del modelo)
- Usar `\[ ... \]` para ecuaciones de display sin número (corolarios, definiciones secundarias)
- Usar `\begin{align}` para sistemas de ecuaciones
- Nombrar los parámetros con `\boldsymbol{\beta}` para vectores, `\beta_k` para escalares
- Función logística siempre como `\sigma(z) = \frac{1}{1+\exp(-z)}`

### Tablas
- **Siempre** usar el entorno `{tabular}` con `\toprule`, `\midrule`, `\bottomrule` de `booktabs`
- Nunca usar líneas verticales (`|`)
- Agrupar filas por bloque temático con `\multicolumn{2}{l}{\textit{Bloque}}`
- Caption sobre la tabla (`\caption` antes de `\begin{tabular}`)
- Decimales con coma para español: escribir `2{,}110` no `2.110`

### Texto académico en español
- Usar `\emph{}` para términos técnicos en inglés (*odds ratio*, *log-loss*, *ridge*)
- Mantener los nombres de variables en `\texttt{}` cuando sean literales del código
- Usar `\textbf{}` para la primera aparición de conceptos clave en párrafos definitorios
- `\paragraph{Nombre.}` para subsecciones sin número dentro de una sección

### Referencias bibliográficas
- Usar `\begin{thebibliography}{9}` (sin BibTeX a menos que el proyecto lo configure)
- Formato: `Apellido, I. (año). "Título". \textit{Revista}, vol(num), pp–pp.`
- Citar en texto como `\cite{clave}` sin espacios adicionales

---

## Números decimales en español

En documentos en español con `babel[spanish]`, usar la coma decimal. En LaTeX esto requiere proteger con llaves dentro de modo matemático:
- Correcto: `C = 0{,}5`, `R^2 = 0{,}3685`
- Incorrecto: `C = 0.5` (usará punto, inconsistente con el texto en español)

---

## Checklist de revisión antes de guardar

- [ ] El archivo compila sin errores (verificar sintaxis de `\begin`/`\end` pareados)
- [ ] Todos los `\label{}` tienen su `\ref{}` correspondiente
- [ ] Las tablas tienen `[htbp]` como specifier de posición
- [ ] La bibliografía está al final, antes de `\end{document}`
- [ ] El archivo está guardado en `doc/` con nombre descriptivo (ej. `nota_metodologica.tex`)
- [ ] No hay `\usepackage` duplicados

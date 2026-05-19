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

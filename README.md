# 📊 Widget "Build a Voter" - IVE Uruguay

Widget interactivo estilo [The Economist](https://www.economist.com/interactive/us-2024-election/build-a-voter) que calcula la probabilidad de apoyar el derecho a decidir sobre el embarazo (IVE) según las características demográficas del usuario.

## 🎯 ¿Qué hace?

El usuario selecciona sus características:
- **Edad** (18-85 años)
- **Sexo** (Hombre/Mujer)
- **Nivel educativo** (5 niveles)
- **Religiosidad** (4 niveles)
- **Región** (Montevideo/Interior)
- **Tiene hijos** (Sí/No)

Y el widget muestra:
- **Probabilidad personalizada** de apoyar el IVE (0-100%)
- Comparación con el **promedio nacional**
- Comparación con **otros grupos demográficos**

## 🚀 Instalación rápida

```bash
# 1. Navegar a la carpeta
cd ive_widget

# 2. Crear entorno virtual (opcional pero recomendado)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Entrenar el modelo (genera model_coefficients.json)
python train_model.py

# 5. Ejecutar la app
streamlit run app.py
```

## 📁 Estructura del proyecto

```
ive_widget/
├── app.py                    # App Streamlit (interfaz web)
├── train_model.py            # Script para entrenar el modelo
├── model_coefficients.json   # Coeficientes exportados (se genera)
├── requirements.txt          # Dependencias Python
└── README.md                 # Este archivo
```

## 🌐 Deploy en Streamlit Cloud (GRATIS)

Para publicar y obtener una URL embebible:

### Paso 1: Subir a GitHub
```bash
# Crear repositorio en GitHub y subir
git init
git add .
git commit -m "Widget IVE El Observador"
git remote add origin https://github.com/TU_USUARIO/ive-widget.git
git push -u origin main
```

### Paso 2: Deploy en Streamlit Cloud
1. Ir a [share.streamlit.io](https://share.streamlit.io)
2. Conectar con tu cuenta de GitHub
3. Seleccionar el repositorio `ive-widget`
4. Configurar:
   - **Main file path:** `app.py`
   - **Python version:** 3.11
5. Click en "Deploy"

### Paso 3: Obtener URL para embed
Una vez desplegado, tu app tendrá una URL como:
```
https://tu-usuario-ive-widget.streamlit.app
```

### Paso 4: Embeber en El Observador

**No armes el iframe a mano: copiá el de `docs/embed/ive-widget-embed.html`.**
Ahí están la URL correcta, las dos versiones —completa y caja— y las alturas
medidas dentro de un iframe real.

Dos cosas que ese archivo explica y que este README recomendaba mal:

- La URL del embed lleva **`/~/+/`**. Pedida a secas, Streamlit Cloud devuelve un
  envoltorio que carga la app en otro iframe adentro, y ese iframe anidado no se
  redimensiona dentro del iframe de una nota: el widget queda cortado a la altura
  del título.
- El alto **no es 800**: la versión completa necesita 1860 y la de caja 1180. Un
  iframe corto con `scrolling="no"` recorta sin avisar.

```html
<iframe
  src="https://TU-APP.streamlit.app/~/+/?embed=true"
  width="100%"
  height="1860"
  frameborder="0"
  scrolling="no"
  style="border: none;">
</iframe>
```

## 📈 Modelo estadístico

El modelo es una **regresión logística ponderada** que predice:

```
P(Apoyar IVE) = 1 / (1 + exp(-z))

donde z = β₀ + β₁*edad + β₂*sexo + β₃*educación + β₄*religiosidad + β₅*región + β₆*hijos
```

### Variables más influyentes:
1. **Religiosidad** 🙏 - Mayor impacto negativo (más religioso = menor apoyo)
2. **Educación** 🎓 - Mayor educación = mayor apoyo
3. **Sexo** ⚧ - Mujeres apoyan más que hombres

## 🔧 Personalización

### Cambiar colores/estilos
Editar la sección de CSS en `app.py`:
```python
st.markdown("""
<style>
    /* Modificar aquí */
</style>
""", unsafe_allow_html=True)
```

### Agregar más variables
1. Modificar `train_model.py` para incluir nuevas variables
2. Actualizar `app.py` para agregar los selectores correspondientes
3. Re-entrenar el modelo: `python train_model.py`

### Cambiar el texto/idioma
Todo el texto está en `app.py`, simplemente editar los strings.

## 📊 Fuente de datos

- **Encuesta:** El Observador, Uruguay 2025
- **N:** ~3,300 casos ponderados
- **Pregunta IVE:** P174 - "Las mujeres tienen derecho a decidir sobre su embarazo"
- **Ponderador:** `w_norm` (ajustado por diseño muestral)

## 🐛 Troubleshooting

### Error: "No se encontró el archivo de coeficientes"
```bash
python train_model.py  # Ejecutar primero para generar el JSON
```

### Error: "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### La app no carga en el navegador
- Verificar que Streamlit esté corriendo (`streamlit run app.py`)
- Abrir manualmente: http://localhost:8501

## 📝 Licencia

Proyecto de El Observador Uruguay. Uso editorial.

---

**Contacto:** Equipo de Datos, El Observador  
**Inspiración:** [The Economist - Build a Voter](https://www.economist.com/interactive/us-2024-election/build-a-voter)

"""
Widget Interactivo: ¿Cuál es tu probabilidad de apoyar el IVE?
El Observador - Encuesta Uruguay 2025/2026

Basado en el modelo "Build a Voter" de The Economist
Adaptado para medir apoyo a la interrupción voluntaria del embarazo

Autor: El Observador / Equipo de Datos
"""

import streamlit as st
import json
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="¿Apoyas el IVE? | El Observador",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# ESTILOS CSS PERSONALIZADOS
# ============================================================
st.markdown("""
<style>
    /* Fuentes y colores */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Ocultar menú de Streamlit para embed */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Contenedor principal */
    .main > div {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    
    /* Título principal */
    .main-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    
    .subtitle {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    
    /* Barra de probabilidad */
    .prob-container {
        background: linear-gradient(90deg, #e74c3c 0%, #f39c12 25%, #f1c40f 50%, #2ecc71 75%, #27ae60 100%);
        border-radius: 25px;
        height: 50px;
        position: relative;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .prob-indicator {
        position: absolute;
        top: -10px;
        transform: translateX(-50%);
        width: 4px;
        height: 70px;
        background: #1a1a2e;
        border-radius: 2px;
    }
    
    .prob-label {
        position: absolute;
        top: -40px;
        transform: translateX(-50%);
        background: #1a1a2e;
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        white-space: nowrap;
    }
    
    /* Tarjeta de resultado */
    .result-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #3498db;
    }
    
    .result-number {
        font-size: 3rem;
        font-weight: 700;
        color: #2c3e50;
        line-height: 1;
    }
    
    .result-text {
        font-size: 1rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    /* Comparación */
    .comparison {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .comp-item {
        flex: 1;
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .comp-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
    }
    
    .comp-label {
        font-size: 0.85rem;
        color: #888;
    }
    
    /* Selectores */
    .stSelectbox > div > div {
        border-radius: 8px;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: #3498db;
    }
    
    /* Footer */
    .footer-text {
        font-size: 0.75rem;
        color: #999;
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
    
    /* Interpretación */
    .interpretation {
        background: #fff3cd;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
    
    .interpretation strong {
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CARGAR COEFICIENTES DEL MODELO
# ============================================================
@st.cache_data
def load_model():
    model_path = Path(__file__).parent / "model_coefficients.json"
    with open(model_path, 'r', encoding='utf-8') as f:
        return json.load(f)

try:
    MODEL = load_model()
except FileNotFoundError:
    st.error("⚠️ Error: No se encontró el archivo de coeficientes. Ejecuta primero `train_model.py`")
    st.stop()

# ============================================================
# FUNCIÓN DE PREDICCIÓN
# ============================================================
def predict_probability(tramo_edad, es_mujer, nivel_educ, religiosidad, es_montevideo, tiene_hijos, voto):
    """
    Calcula la probabilidad de apoyar el IVE usando regresión logística Ridge.
    P(Y=1) = 1 / (1 + exp(-z))
    donde z = β0 + β1*x1 + β2*x2 + ...
    Referencia voto = "Otros/No votó"
    Modelo con regularización L2 - TODOS los partidos incluidos
    """
    coef = MODEL['coefficients']
    
    # Voto: TODOS los partidos por separado (referencia = Otros/No votó)
    voto_fa = 1 if voto == "fa" else 0
    voto_pn = 1 if voto == "pn" else 0
    voto_pc = 1 if voto == "pc" else 0
    voto_ca = 1 if voto == "ca" else 0
    
    # Calcular logit (z) - Con Ridge, TODOS los coeficientes existen
    z = coef['intercept']
    z += coef['tramo_edad_num'] * tramo_edad
    z += coef['es_mujer'] * es_mujer
    z += coef['nivel_educ_num'] * nivel_educ
    z += coef['religiosidad_num'] * religiosidad
    z += coef['es_montevideo'] * es_montevideo
    z += coef['tiene_hijos'] * tiene_hijos
    z += coef['voto_fa'] * voto_fa
    z += coef['voto_pn'] * voto_pn
    z += coef['voto_pc'] * voto_pc
    z += coef['voto_ca'] * voto_ca
    
    # Función logística
    probability = 1 / (1 + np.exp(-z))
    
    return probability * 100  # Retornar como porcentaje

# ============================================================
# INTERFAZ DE USUARIO
# ============================================================

# Título
st.markdown('<p class="main-title">¿Cuál es tu probabilidad de apoyar el derecho a decidir sobre el embarazo?</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Basado en la encuesta de El Observador a uruguayos. Selecciona tus características:</p>', unsafe_allow_html=True)

# Crear columnas para los selectores
col1, col2 = st.columns(2)

with col1:
    # Tramo de edad
    edad_options = MODEL['variable_ranges']['tramo_edad_num']['labels']
    edad_sel = st.selectbox(
        "👤 Edad",
        options=edad_options,
        index=1,
        help="Selecciona tu tramo de edad"
    )
    tramo_edad = edad_options.index(edad_sel) + 1
    
    # Sexo
    sexo_options = MODEL['variable_ranges']['es_mujer']['labels']
    sexo = st.selectbox(
        "⚧ Sexo",
        options=sexo_options,
        index=0,
        help="Selecciona tu sexo"
    )
    es_mujer = 1 if sexo == "Mujer" else 0
    
    # Nivel educativo
    educ_options = MODEL['variable_ranges']['nivel_educ_num']['labels']
    educacion = st.selectbox(
        "🎓 Nivel educativo",
        options=educ_options,
        index=2,
        help="Selecciona tu nivel educativo más alto"
    )
    nivel_educ = educ_options.index(educacion) + 1
    
    # Voto 2019
    voto_options = MODEL['variable_ranges']['voto_2019']['labels']
    voto_sel = st.selectbox(
        "🗳️ Voto 2019",
        options=voto_options,
        index=0,
        help="¿A quién votaste en las elecciones de 2019?"
    )
    voto_map = {
        "Otros/No votó": "otros",
        "Frente Amplio": "fa",
        "Partido Nacional": "pn",
        "Partido Colorado": "pc",
        "Cabildo Abierto": "ca"
    }
    voto = voto_map[voto_sel]

with col2:
    # Religiosidad
    relig_options = MODEL['variable_ranges']['religiosidad_num']['labels']
    religiosidad = st.selectbox(
        "🙏 Religiosidad",
        options=relig_options,
        index=1,
        help="¿Cuán religioso/a te consideras?"
    )
    religiosidad_num = relig_options.index(religiosidad) + 1
    
    # Región
    region_options = MODEL['variable_ranges']['es_montevideo']['labels']
    region = st.selectbox(
        "📍 Región",
        options=region_options,
        index=0,
        help="¿Dónde vives?"
    )
    es_montevideo = 1 if region == "Montevideo" else 0
    
    # Tiene hijos
    hijos_options = MODEL['variable_ranges']['tiene_hijos']['labels']
    hijos = st.selectbox(
        "👶 ¿Tienes hijos?",
        options=hijos_options,
        index=0,
        help="¿Tienes hijos/as?"
    )
    tiene_hijos = 1 if hijos == "Sí" else 0

# ============================================================
# CALCULAR Y MOSTRAR RESULTADO
# ============================================================

# Calcular probabilidad
prob = predict_probability(tramo_edad, es_mujer, nivel_educ, religiosidad_num, es_montevideo, tiene_hijos, voto)
prob_nacional = MODEL.get('prob_nacional', 78.6)

# Mostrar resultado principal
st.markdown("---")

# Barra de probabilidad visual
bar_html = f"""
<div style="margin: 2rem 0;">
    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
        <span style="color: #e74c3c; font-weight: 600;">En contra</span>
        <span style="color: #27ae60; font-weight: 600;">A favor</span>
    </div>
    <div class="prob-container">
        <div class="prob-indicator" style="left: {prob}%;">
            <div class="prob-label">{prob:.0f}%</div>
        </div>
    </div>
</div>
"""
st.markdown(bar_html, unsafe_allow_html=True)

# Tarjeta de resultado
col_result1, col_result2 = st.columns([2, 1])

with col_result1:
    # Determinar color según probabilidad
    if prob >= 70:
        color = "#27ae60"
        texto = "muy probable que apoyes"
    elif prob >= 55:
        color = "#2ecc71"
        texto = "probable que apoyes"
    elif prob >= 45:
        color = "#f1c40f"
        texto = "dividido/a"
    elif prob >= 30:
        color = "#f39c12"
        texto = "probable que te opongas"
    else:
        color = "#e74c3c"
        texto = "muy probable que te opongas"
    
    st.markdown(f"""
    <div class="result-card" style="border-left-color: {color};">
        <div class="result-number" style="color: {color};">{prob:.0f}%</div>
        <div class="result-text">
            Probabilidad de apoyar el derecho a decidir sobre el embarazo.<br>
            <strong>Es {texto}</strong> al IVE según tus características.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_result2:
    # Comparación con promedio nacional
    diff = prob - prob_nacional
    arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
    diff_color = "#27ae60" if diff > 0 else "#e74c3c" if diff < 0 else "#666"
    
    st.markdown(f"""
    <div class="comp-item">
        <div class="comp-label">Promedio nacional</div>
        <div class="comp-value">{prob_nacional:.0f}%</div>
        <div style="color: {diff_color}; font-weight: 600; margin-top: 0.5rem;">
            {arrow} {abs(diff):.0f}pp vs. tú
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# COMPARACIONES POR GRUPO
# ============================================================
st.markdown("---")
st.markdown("### 📊 Comparación con otros grupos")

stats = MODEL['stats_by_group']

# Tabs para diferentes comparaciones
tab1, tab2, tab3, tab4 = st.tabs(["Por religiosidad", "Por voto 2019", "Por educación", "Por edad"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        val = stats.get('religiosidad_nada', 0)
        st.metric("Nada religioso", f"{val}%", delta=f"{val-prob:.0f}pp vs tú" if val != prob else None)
    with col2:
        val = stats.get('religiosidad_poco', 0)
        st.metric("Poco religioso", f"{val}%", delta=f"{val-prob:.0f}pp vs tú" if val != prob else None)
    with col3:
        val = stats.get('religiosidad_bastante', 0)
        st.metric("Bastante religioso", f"{val}%", delta=f"{val-prob:.0f}pp vs tú" if val != prob else None)
    with col4:
        val = stats.get('religiosidad_mucho', 0)
        st.metric("Muy religioso", f"{val}%", delta=f"{val-prob:.0f}pp vs tú" if val != prob else None)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        val = stats.get('voto_frente_amplio', 0)
        st.metric("Frente Amplio", f"{val}%", delta=f"{val-prob:.0f}pp vs tú" if val != prob else None)
    with col2:
        val = stats.get('voto_partido_nacional', 0)
        st.metric("Partido Nacional", f"{val}%", delta=f"{val-prob:.0f}pp vs tú" if val != prob else None)
    with col3:
        val = stats.get('voto_partido_colorado', 0)
        st.metric("Partido Colorado", f"{val}%", delta=f"{val-prob:.0f}pp vs tú" if val != prob else None)
    with col4:
        val = stats.get('voto_cabildo_abierto', 0)
        st.metric("Cabildo Abierto", f"{val}%", delta=f"{val-prob:.0f}pp vs tú" if val != prob else None)

with tab3:
    col1, col2, col3 = st.columns(3)
    with col1:
        val = stats.get('educacion_primaria', 0)
        st.metric("Primaria", f"{val}%")
    with col2:
        val = stats.get('educacion_ems_comp', 0)
        st.metric("Secundaria", f"{val}%")
    with col3:
        val = stats.get('educacion_ter_comp', 0)
        st.metric("Universitaria", f"{val}%")

with tab4:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        val = stats.get('edad_18-24', 0)
        st.metric("18-24", f"{val}%")
    with col2:
        val = stats.get('edad_25-34', 0)
        st.metric("25-34", f"{val}%")
    with col3:
        val = stats.get('edad_35-44', 0)
        st.metric("35-44", f"{val}%")
    with col4:
        val = stats.get('edad_45-54', 0)
        st.metric("45-54", f"{val}%")
    with col5:
        val = stats.get('edad_55+', 0)
        st.metric("55+", f"{val}%")

# ============================================================
# INTERPRETACIÓN EXPANDIBLE
# ============================================================
with st.expander("ℹ️ ¿Cómo funciona este modelo?"):
    st.markdown("""
    Este widget utiliza un **modelo de regresión logística** entrenado con datos de la encuesta de 
    El Observador realizada en Uruguay.
    
    **Variables más influyentes:**
    
    1. **Religiosidad** 🙏: Es el factor más importante. Las personas más religiosas tienen 
       significativamente menor probabilidad de apoyar el IVE.
    
    2. **Voto político** 🗳️: Votantes del Frente Amplio tienen mayor probabilidad de apoyo 
       que votantes de la Coalición (PN/PC/CA).
    
    3. **Sexo** ⚧: Las mujeres tienden a apoyar más el IVE que los hombres.
    
    4. **Educación** 🎓: Mayor nivel educativo se asocia con mayor apoyo al derecho a decidir.
    
    5. **Región** 📍: Pueden existir diferencias entre Montevideo y el Interior.
    
    **Nota metodológica:**
    - El modelo excluye a quienes respondieron "Ni de acuerdo ni en desacuerdo"
    - Los resultados son probabilidades basadas en correlaciones estadísticas, no predicciones individuales
    - Pseudo R² del modelo: {:.1%}
    """.format(MODEL['model_info']['pseudo_r2']))

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer-text">
    <strong>El Observador</strong> | Encuesta realizada en Uruguay 2025<br>
    Basado en {} respuestas ponderadas | Modelo de regresión logística<br>
    <em>Las probabilidades son estimaciones estadísticas basadas en grupos, no predicciones individuales</em>
</div>
""".format(MODEL['model_info']['n_observations']), unsafe_allow_html=True)

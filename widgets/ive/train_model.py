"""
Entrenamiento del modelo logístico para predecir apoyo al IVE.
El Observador - Encuesta Uruguay 2025/2026

Mejoras v2 (2026-02-06):
- Variables ordinales como dummies completas (no lineales)
- Términos de interacción (mujer*religiosidad, mujer*hijos)
- Cross-validation para selección de C
- Pseudo R² corregido (modelo nulo con media ponderada)
- Balotaje 2024 (Orsi/FA vs Delgado/Coalición) en vez de voto por partido (mejor R² con menos dummies)
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from widgets.ive.config import DATA_FILE, MODEL_COEFFICIENTS_PATH

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import log_loss
import json

OUTPUT_FILE = MODEL_COEFFICIENTS_PATH

W = 'w_norm'  # Ponderador

print("Cargando datos...")
df = pd.read_csv(DATA_FILE)
print(f"Base cargada: {len(df)} casos")

# Limpieza de outliers en edad: hay ~2 casos con valores fuera de rango
# (e.g. años de nacimiento o fechas codificadas como número en vez de edad).
# Los pasamos a NaN para que sean excluidos por dropna() y no contaminen
# tabulaciones cruzadas. No afectan al modelo principal directamente porque
# usamos tramos categóricos, pero sí a `stats_by_group`.
n_outliers_edad = ((df['edad'] < 18) | (df['edad'] > 110)).sum()
if n_outliers_edad:
    print(f"Limpiando {n_outliers_edad} outliers en edad (<18 o >110)")
    df.loc[(df['edad'] < 18) | (df['edad'] > 110), 'edad'] = np.nan

print("\nCreando variables...")

# ============================================================
# TRAMOS DE EDAD (dummies, referencia: 18-24)
# ============================================================
def edad_a_tramo(edad):
    if pd.isna(edad): return np.nan
    if edad < 25: return '18-24'
    if edad < 35: return '25-34'
    if edad < 45: return '35-44'
    if edad < 55: return '45-54'
    return '55+'

df['tramo_edad'] = df['edad'].apply(edad_a_tramo)

# Dummies (referencia: 18-24)
df['edad_25_34'] = (df['tramo_edad'] == '25-34').astype(int)
df['edad_35_44'] = (df['tramo_edad'] == '35-44').astype(int)
df['edad_45_54'] = (df['tramo_edad'] == '45-54').astype(int)
df['edad_55_plus'] = (df['tramo_edad'] == '55+').astype(int)

# Mantener numérico para variable_ranges (UI)
df['tramo_edad_num'] = df['tramo_edad'].map({
    '18-24': 1, '25-34': 2, '35-44': 3, '45-54': 4, '55+': 5
})

# ============================================================
# SEXO
# ============================================================
df['es_mujer'] = (df['sexo'] == 'F').astype(int)

# ============================================================
# NIVEL EDUCATIVO (dummies, referencia: primaria o menos)
# Recodificación a 4 categorías a partir de nivel_educativo (escala 1-10).
# Bachillerato incompleto + Bachillerato completo se colapsan en "Secundaria":
# las tasas crudas son monotónicas (73.3% vs 76.7%) pero el efecto neto
# controlando por confounders (sexo, religión, balotaje, región) es
# indistinguible (gap ~0.024 logit, signo invertido y dentro del ruido).
# La mayor proporción de mujeres y urbanos en bach_comp explica la
# diferencia cruda; no es un efecto educativo neto.
#   1 Sin instrucción / 2 Primaria              -> Primaria o menos (REF)
#   3 CB inc / 4 CB comp / 5 Bach inc / 6 Bach comp -> Secundaria
#   7 Ter inc                                   -> Terciaria incompleta
#   8 Ter comp / 9 Posg inc / 10 Posg comp      -> Terciaria completa+
# ============================================================
def educ_a_cat(n):
    if pd.isna(n): return np.nan
    n = int(n)
    if n in (1, 2): return 'primaria'
    if n in (3, 4, 5, 6): return 'secundaria'
    if n == 7:      return 'ter_incomp'
    if n in (8, 9, 10): return 'ter_comp'
    return np.nan

df['nivel_educ_cat'] = df['nivel_educativo'].apply(educ_a_cat)

# Dummies (referencia: primaria o menos)
df['educ_secundaria'] = (df['nivel_educ_cat'] == 'secundaria').astype(int)
df['educ_ter_incomp'] = (df['nivel_educ_cat'] == 'ter_incomp').astype(int)
df['educ_ter_comp'] = (df['nivel_educ_cat'] == 'ter_comp').astype(int)

# Mantener numérico para variable_ranges (UI): 1..4
df['nivel_educ_num'] = df['nivel_educ_cat'].map({
    'primaria': 1, 'secundaria': 2, 'ter_incomp': 3, 'ter_comp': 4,
})

# ============================================================
# RELIGIOSIDAD (dummies, referencia: nada)
# ============================================================
relig_map = {
    'Nada. Soy ateo / No creo en la religión': 'nada',
    'Poco. Me identifico culturalmente con alguna religión pero no soy practicante ni ella es muy importante en mi vida': 'poco',
    'Bastante. Me identifico con alguna religión y ella es importante en mi vida y mis valores': 'bastante',
    'Mucho. Me identifico con alguna religión y sigo sus prácticas y valores asistiendo a sus rituales y encuentros': 'mucho'
}
df['relig_cat'] = df['P178_Cuan_religioso'].map(relig_map)

# Dummies (referencia: nada)
df['relig_poco'] = (df['relig_cat'] == 'poco').astype(int)
df['relig_bastante'] = (df['relig_cat'] == 'bastante').astype(int)
df['relig_mucho'] = (df['relig_cat'] == 'mucho').astype(int)

# Mantener numérico para variable_ranges (UI)
df['religiosidad_num'] = df['relig_cat'].map({
    'nada': 1, 'poco': 2, 'bastante': 3, 'mucho': 4
})

# ============================================================
# REGIÓN
# ============================================================
df['es_montevideo'] = (df['dpto'] == 19).astype(int)

# ============================================================
# HIJOS
# ============================================================
df['tiene_hijos'] = df['P159_Cuantos_hijos'].apply(
    lambda x: 0 if x == 'Ninguno' else 1 if pd.notna(x) else np.nan
)

# ============================================================
# PERSONAS EN EL HOGAR (dummies, referencia: 1-2)
# ============================================================
def personas_a_cat(n):
    if pd.isna(n): return np.nan
    if n <= 2: return '1-2'
    if n <= 4: return '3-4'
    return '5+'

df['hogar_cat'] = df['cant_personas'].apply(personas_a_cat)

# Dummies (referencia: 1-2)
df['hogar_3_4'] = (df['hogar_cat'] == '3-4').astype(int)
df['hogar_5_plus'] = (df['hogar_cat'] == '5+').astype(int)

# Mantener numérico para variable_ranges (UI)
df['hogar_num'] = df['hogar_cat'].map({'1-2': 1, '3-4': 2, '5+': 3})

# ============================================================
# BALOTAJE 2024 (dummies, referencia: blanco/no votó/no recuerda)
# IdBalotaje: 1=Orsi (FA), 2=Delgado (Coalición), 3=Blanco, 4=No votó, 5=No recuerda
# Coeficientes internos nombrados como martinez/lacalle por compatibilidad histórica
# ============================================================
df['balotaje_martinez'] = (df['IdBalotaje'] == 1).astype(int)
df['balotaje_lacalle'] = (df['IdBalotaje'] == 2).astype(int)

balotaje_labels = {1: 'Orsi (FA)', 2: 'Delgado (Coalición)', 3: 'Blanco', 4: 'No votó', 5: 'No recuerda'}
df['balotaje_label'] = df['IdBalotaje'].map(balotaje_labels)

print(f"\nDistribución balotaje 2024:")
print(df['balotaje_label'].value_counts())

# ============================================================
# TÉRMINOS DE INTERACCIÓN
# ============================================================
df['mujer_x_relig_mucho'] = df['es_mujer'] * df['relig_mucho']
df['mujer_x_tiene_hijos'] = df['es_mujer'] * df['tiene_hijos']

# ============================================================
# VARIABLE OBJETIVO: APOYO AL IVE
# ============================================================
escala_5 = {
    'Totalmente en desacuerdo': 1,
    'En desacuerdo': 2,
    'Ni de acuerdo ni en desacuerdo': 3,
    'De acuerdo': 4,
    'Totalmente de acuerdo': 5
}
df['decidir_embarazo'] = df['P174_Decidir_embarazo'].map(escala_5)

df['favor_ive'] = np.where(
    df['decidir_embarazo'] >= 4, 1,
    np.where(df['decidir_embarazo'] <= 2, 0, np.nan)
)

# Variable secundaria: indicador de neutralidad / NS-NC.
# 1 si Likert == 3 ("ni de acuerdo ni en desacuerdo") o falta respuesta.
# Se usa para entrenar un modelo logit auxiliar que predice la probabilidad
# de no fijar postura, reportada en la UI como dato secundario.
df['es_neutral'] = np.where(
    df['decidir_embarazo'] == 3, 1,
    np.where(df['decidir_embarazo'].isin([1, 2, 4, 5]), 0, 1)
).astype(int)

print(f"\nDistribución favor_ive:")
print(f"  A favor (1):    {(df['favor_ive']==1).sum()}")
print(f"  En contra (0):  {(df['favor_ive']==0).sum()}")
print(f"  Indecisos (NA): {df['favor_ive'].isna().sum()} (excluidos)")

print(f"\nDistribución es_neutral:")
print(f"  Neutral/NSNC (1): {(df['es_neutral']==1).sum()}")
print(f"  Con postura (0):  {(df['es_neutral']==0).sum()}")

# ============================================================
# PREDICTORES (dummies + interacciones)
# ============================================================
PREDICTORS = [
    # Edad (ref: 18-24)
    'edad_25_34', 'edad_35_44', 'edad_45_54', 'edad_55_plus',
    # Género
    'es_mujer',
    # Educación (ref: primaria o menos)
    'educ_secundaria', 'educ_ter_incomp', 'educ_ter_comp',
    # Religiosidad (ref: nada)
    'relig_poco', 'relig_bastante', 'relig_mucho',
    # Región
    'es_montevideo',
    # Hijos
    'tiene_hijos',
    # Personas en hogar (ref: 1-2)
    'hogar_3_4', 'hogar_5_plus',
    # Balotaje 2019 (ref: blanco/no votó/no recuerda)
    'balotaje_martinez', 'balotaje_lacalle',
    # Interacciones
    'mujer_x_relig_mucho', 'mujer_x_tiene_hijos',
]

# Preparar datos
datos = df[['favor_ive', W] + PREDICTORS].dropna()
print(f"\nN válidos para el modelo: {len(datos)}")

X = datos[PREDICTORS].values
y = datos['favor_ive'].values
weights = datos[W].values

# ============================================================
# CROSS-VALIDATION PARA SELECCIÓN DE C
# ============================================================
print("\n" + "="*60)
print("CROSS-VALIDATION PARA SELECCIÓN DE C")
print("="*60)

C_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_C = 1.0
best_score = -np.inf

for C in C_values:
    modelo_cv = LogisticRegression(
        C=C, solver='lbfgs', max_iter=2000, random_state=42
    )
    scores = cross_val_score(
        modelo_cv, X, y, cv=cv, scoring='neg_log_loss',
        params={'sample_weight': weights}
    )
    mean_score = scores.mean()
    print(f"  C={C:6.2f} | CV neg_log_loss: {mean_score:.4f} (+/- {scores.std():.4f})")
    if mean_score > best_score:
        best_score = mean_score
        best_C = C

print(f"\n  Mejor C: {best_C}")

# ============================================================
# MODELO FINAL
# ============================================================
print("\n" + "="*60)
print(f"ENTRENANDO MODELO FINAL (C={best_C})")
print("="*60)

modelo = LogisticRegression(
    C=best_C, solver='lbfgs', max_iter=2000, random_state=42
)
modelo.fit(X, y, sample_weight=weights)

# ============================================================
# MÉTRICAS (Pseudo R² CORREGIDO)
# ============================================================
y_pred_proba = modelo.predict_proba(X)[:, 1]

# Log-likelihood del modelo ajustado
ll_model = -log_loss(y, y_pred_proba, sample_weight=weights, normalize=False)

# FIX: Modelo nulo con media PONDERADA (antes usaba y.mean() sin ponderar)
y_weighted_mean = np.average(y, weights=weights)
ll_null = -log_loss(y, np.full(len(y), y_weighted_mean), sample_weight=weights, normalize=False)

pseudo_r2 = 1 - (ll_model / ll_null)

# Cross-validated score para comparación
cv_scores = cross_val_score(
    LogisticRegression(C=best_C, solver='lbfgs', max_iter=2000, random_state=42),
    X, y, cv=cv, scoring='neg_log_loss',
    params={'sample_weight': weights}
)

print(f"\nPseudo R² (McFadden, corregido) = {pseudo_r2:.4f}")
print(f"Log-Likelihood modelo = {ll_model:.2f}")
print(f"Log-Likelihood nulo   = {ll_null:.2f}")
print(f"Media ponderada y     = {y_weighted_mean:.4f}")
print(f"Accuracy ponderado    = {np.average(modelo.predict(X) == y, weights=weights):.4f}")
print(f"CV neg_log_loss       = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================
# COEFICIENTES
# ============================================================
print("\n" + "-"*60)
print("COEFICIENTES Y ODDS RATIOS")
print("-"*60)

coeficientes = pd.DataFrame({
    'Variable': ['intercept'] + PREDICTORS,
    'Coef': [modelo.intercept_[0]] + list(modelo.coef_[0]),
})
coeficientes['Odds Ratio'] = np.exp(coeficientes['Coef'])
coeficientes['|Coef|'] = np.abs(coeficientes['Coef'])
coeficientes = coeficientes.sort_values('|Coef|', ascending=False)

print(coeficientes.to_string(index=False))

# ============================================================
# MODELO SECUNDARIO: P(neutral / NS-NC)
# Mismos 21 predictores, misma penalización Ridge, mismos pesos.
# C fijo en best_C (no se vuelve a hacer CV; es un modelo descriptivo
# auxiliar, no un producto principal).
# ============================================================
print("\n" + "="*60)
print(f"ENTRENANDO MODELO DE NEUTRALIDAD (C={best_C})")
print("="*60)

datos_neu = df[['es_neutral', W] + PREDICTORS].dropna()
X_neu = datos_neu[PREDICTORS].values
y_neu = datos_neu['es_neutral'].values
w_neu = datos_neu[W].values

modelo_neu = LogisticRegression(
    C=best_C, solver='lbfgs', max_iter=2000, random_state=42
)
modelo_neu.fit(X_neu, y_neu, sample_weight=w_neu)

y_neu_pred = modelo_neu.predict_proba(X_neu)[:, 1]
ll_neu_model = -log_loss(y_neu, y_neu_pred, sample_weight=w_neu, normalize=False)
y_neu_mean = np.average(y_neu, weights=w_neu)
ll_neu_null = -log_loss(
    y_neu, np.full(len(y_neu), y_neu_mean),
    sample_weight=w_neu, normalize=False
)
pseudo_r2_neu = 1 - (ll_neu_model / ll_neu_null)

print(f"N modelo neutral: {len(datos_neu)}")
print(f"Tasa neutral ponderada: {y_neu_mean:.4f}")
print(f"Pseudo R² (McFadden) neutral: {pseudo_r2_neu:.4f}")

prob_neutral_nacional = float(np.average(y_neu_pred, weights=w_neu)) * 100

# ============================================================
# ESTADÍSTICAS POR GRUPO
# ============================================================
stats_by_group = {}

# Por balotaje
for id_bal, label in [(1, 'martinez'), (2, 'lacalle')]:
    subset = df[(df['IdBalotaje'] == id_bal) & df['favor_ive'].notna()]
    if len(subset) > 0:
        prob = np.average(subset['favor_ive'], weights=subset[W])
        stats_by_group[f'balotaje_{label}'] = round(prob * 100, 1)

# Por religiosidad
for cat, label in [('nada', 'nada'), ('poco', 'poco'), ('bastante', 'bastante'), ('mucho', 'mucho')]:
    subset = df[(df['relig_cat'] == cat) & df['favor_ive'].notna()]
    if len(subset) > 0:
        prob = np.average(subset['favor_ive'], weights=subset[W])
        stats_by_group[f'religiosidad_{label}'] = round(prob * 100, 1)

# Por nivel educativo (4 categorías)
for cat in ['primaria', 'secundaria', 'ter_incomp', 'ter_comp']:
    subset = df[(df['nivel_educ_cat'] == cat) & df['favor_ive'].notna()]
    if len(subset) > 0:
        prob = np.average(subset['favor_ive'], weights=subset[W])
        stats_by_group[f'educacion_{cat}'] = round(prob * 100, 1)

# Por personas en hogar
for cat, label in [('1-2', '1_2'), ('3-4', '3_4'), ('5+', '5_plus')]:
    subset = df[(df['hogar_cat'] == cat) & df['favor_ive'].notna()]
    if len(subset) > 0:
        prob = np.average(subset['favor_ive'], weights=subset[W])
        stats_by_group[f'hogar_{label}'] = round(prob * 100, 1)

# Por tramo de edad
for tramo in ['18-24', '25-34', '35-44', '45-54', '55+']:
    subset = df[(df['tramo_edad'] == tramo) & df['favor_ive'].notna()]
    if len(subset) > 0:
        prob = np.average(subset['favor_ive'], weights=subset[W])
        stats_by_group[f'edad_{tramo}'] = round(prob * 100, 1)

# ============================================================
# EXPORTAR COEFICIENTES
# ============================================================
# Outputs to widgets/ive/model_coefficients.json (package-local copy).
# Root model_coefficients.json is a legacy copy to be removed in platform cleanup.
print("\n" + "="*60)
print("EXPORTANDO COEFICIENTES")
print("="*60)

output = {
    "coefficients": {
        "intercept": round(modelo.intercept_[0], 6)
    },
    "odds_ratios": {},
    "coefficients_neutral": {
        "intercept": round(modelo_neu.intercept_[0], 6)
    },
    "odds_ratios_neutral": {},
    "model_info_neutral": {
        "pseudo_r2": round(pseudo_r2_neu, 4),
        "n_observations": int(len(datos_neu)),
        "n_predictors": len(PREDICTORS),
        "regularization": "Ridge (L2)",
        "C": best_C,
        "weighted_mean_y": round(float(y_neu_mean), 4),
        "model_version": 2,
    },
    "prob_neutral_nacional": round(prob_neutral_nacional, 1),
    "model_info": {
        "pseudo_r2": round(pseudo_r2, 4),
        "n_observations": int(len(datos)),
        "n_predictors": len(PREDICTORS),
        "regularization": "Ridge (L2)",
        "C": best_C,
        "cv_neg_log_loss_mean": round(cv_scores.mean(), 4),
        "cv_neg_log_loss_std": round(cv_scores.std(), 4),
        "weighted_mean_y": round(y_weighted_mean, 4),
        "model_version": 2,
    },
    "variable_ranges": {
        "tramo_edad_num": {
            "options": [1, 2, 3, 4, 5],
            "labels": ["18-24 años", "25-34 años", "35-44 años", "45-54 años", "55+ años"],
            "default": 2
        },
        "es_mujer": {"options": [0, 1], "labels": ["Hombre", "Mujer"], "default": 0},
        "nivel_educ_num": {
            "options": [1, 2, 3, 4],
            "labels": [
                "Primaria o menos",
                "Secundaria",
                "Terciaria incompleta",
                "Terciaria completa o más",
            ],
            "default": 2
        },
        "religiosidad_num": {
            "options": [1, 2, 3, 4],
            "labels": ["Nada", "Poco", "Bastante", "Mucho"],
            "default": 2
        },
        "es_montevideo": {"options": [0, 1], "labels": ["Interior", "Montevideo"], "default": 0},
        "tiene_hijos": {"options": [0, 1], "labels": ["No", "Sí"], "default": 0},
        "hogar_num": {
            "options": [1, 2, 3],
            "labels": ["1-2 personas", "3-4 personas", "5 o mas"],
            "default": 2
        },
        "balotaje": {
            "options": ["otros", "martinez", "lacalle"],
            "labels": ["No votó/Blanco", "Orsi (FA)", "Delgado (Coalición)"],
            "default": "otros"
        }
    },
    "stats_by_group": stats_by_group,
    "predictor_names": PREDICTORS,
}

# Exportar todos los coeficientes
for i, param in enumerate(PREDICTORS):
    output["coefficients"][param] = round(modelo.coef_[0][i], 6)
    output["odds_ratios"][param] = round(np.exp(modelo.coef_[0][i]), 4)
    output["coefficients_neutral"][param] = round(modelo_neu.coef_[0][i], 6)
    output["odds_ratios_neutral"][param] = round(np.exp(modelo_neu.coef_[0][i]), 4)

# Probabilidad promedio nacional
prob_promedio = np.average(y_pred_proba, weights=weights)
output["prob_nacional"] = round(prob_promedio * 100, 1)

# Guardar
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n[OK] Coeficientes exportados a: {OUTPUT_FILE}")
print(f"Probabilidad promedio nacional: {prob_promedio*100:.1f}%")
print(f"Predictores: {len(PREDICTORS)}")
print(f"C seleccionado: {best_C}")

print("\n" + "="*60)
print("[OK] Modelo v2 entrenado y exportado exitosamente")
print("="*60)

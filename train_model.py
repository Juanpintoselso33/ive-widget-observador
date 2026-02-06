"""
Entrenamiento del modelo logístico para predecir apoyo al IVE.
El Observador - Encuesta Uruguay 2025/2026

Mejoras v2 (2026-02-06):
- Variables ordinales como dummies completas (no lineales)
- Términos de interacción (mujer*religiosidad, mujer*hijos)
- Cross-validation para selección de C
- Pseudo R² corregido (modelo nulo con media ponderada)
- Balotaje 2019 en vez de voto por partido (mejor R² con menos dummies)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import log_loss
import json
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================
BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / 'base_limpia.csv'
OUTPUT_FILE = Path(__file__).parent / 'model_coefficients.json'

W = 'w_norm'  # Ponderador

print("Cargando datos...")
df = pd.read_csv(DATA_FILE)
print(f"Base cargada: {len(df)} casos")

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
# NIVEL EDUCATIVO (dummies, referencia: primaria)
# ============================================================
educ_map = {
    '1-PRIMARIA': 'primaria',
    '2-EMS INCOMP': 'ems_incomp',
    '3-EMS COMP': 'ems_comp',
    '4-TER INCOMP': 'ter_incomp',
    '5-TER COMP': 'ter_comp'
}
df['nivel_educ_cat'] = df['nivel_educ'].map(educ_map)

# Dummies (referencia: primaria)
df['educ_ems_incomp'] = (df['nivel_educ_cat'] == 'ems_incomp').astype(int)
df['educ_ems_comp'] = (df['nivel_educ_cat'] == 'ems_comp').astype(int)
df['educ_ter_incomp'] = (df['nivel_educ_cat'] == 'ter_incomp').astype(int)
df['educ_ter_comp'] = (df['nivel_educ_cat'] == 'ter_comp').astype(int)

# Mantener numérico para variable_ranges (UI)
df['nivel_educ_num'] = df['nivel_educ'].map({
    '1-PRIMARIA': 1, '2-EMS INCOMP': 2, '3-EMS COMP': 3,
    '4-TER INCOMP': 4, '5-TER COMP': 5
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
# BALOTAJE 2019 (dummies, referencia: blanco/no votó/no recuerda)
# IdBalotaje: 1=Martínez, 2=Lacalle, 3=Blanco, 4=No votó, 5=No recuerda
# ============================================================
df['balotaje_martinez'] = (df['IdBalotaje'] == 1).astype(int)
df['balotaje_lacalle'] = (df['IdBalotaje'] == 2).astype(int)

balotaje_labels = {1: 'Martínez', 2: 'Lacalle', 3: 'Blanco', 4: 'No votó', 5: 'No recuerda'}
df['balotaje_label'] = df['IdBalotaje'].map(balotaje_labels)

print(f"\nDistribución balotaje 2019:")
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

print(f"\nDistribución favor_ive:")
print(f"  A favor (1):    {(df['favor_ive']==1).sum()}")
print(f"  En contra (0):  {(df['favor_ive']==0).sum()}")
print(f"  Indecisos (NA): {df['favor_ive'].isna().sum()} (excluidos)")

# ============================================================
# PREDICTORES (dummies + interacciones)
# ============================================================
PREDICTORS = [
    # Edad (ref: 18-24)
    'edad_25_34', 'edad_35_44', 'edad_45_54', 'edad_55_plus',
    # Género
    'es_mujer',
    # Educación (ref: primaria)
    'educ_ems_incomp', 'educ_ems_comp', 'educ_ter_incomp', 'educ_ter_comp',
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

# Por nivel educativo
for cat, label in [('primaria', 'primaria'), ('ems_incomp', 'ems_incomp'),
                   ('ems_comp', 'ems_comp'), ('ter_incomp', 'ter_incomp'), ('ter_comp', 'ter_comp')]:
    subset = df[(df['nivel_educ_cat'] == cat) & df['favor_ive'].notna()]
    if len(subset) > 0:
        prob = np.average(subset['favor_ive'], weights=subset[W])
        stats_by_group[f'educacion_{label}'] = round(prob * 100, 1)

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
print("\n" + "="*60)
print("EXPORTANDO COEFICIENTES")
print("="*60)

output = {
    "coefficients": {
        "intercept": round(modelo.intercept_[0], 6)
    },
    "odds_ratios": {},
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
            "options": [1, 2, 3, 4, 5],
            "labels": ["Primaria", "EMS incompleta", "EMS completa", "Terciaria incompleta", "Terciaria completa"],
            "default": 3
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
            "labels": ["No votó/Blanco", "Martínez (FA)", "Lacalle (Coalición)"],
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

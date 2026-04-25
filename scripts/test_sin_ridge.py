"""
Diagnóstico: refit del modelo principal de apoyo IVE SIN regularización Ridge,
para ver si la inversión CB > Bach en los coeficientes persiste o si es
artefacto de la penalización L2 con n chico en CB.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE = Path(__file__).resolve().parent.parent.parent / 'base_limpia.csv'
df = pd.read_csv(BASE)

PREDICTORS = [
    'edad_25_34', 'edad_35_44', 'edad_45_54', 'edad_55_plus',
    'es_mujer',
    'educ_cb_incomp', 'educ_cb_comp',
    'educ_bach_incomp', 'educ_bach_comp', 'educ_ter_incomp', 'educ_ter_comp',
    'relig_poco', 'relig_bastante', 'relig_mucho',
    'es_montevideo', 'tiene_hijos',
    'hogar_3_4', 'hogar_5_plus',
    'balotaje_martinez', 'balotaje_lacalle',
    'mujer_x_relig_mucho', 'mujer_x_tiene_hijos',
]


def edad_a_tramo(e):
    if pd.isna(e): return np.nan
    if e < 25: return '18-24'
    if e < 35: return '25-34'
    if e < 45: return '35-44'
    if e < 55: return '45-54'
    return '55+'


def educ_a_cat(n):
    if pd.isna(n): return np.nan
    n = int(n)
    if n in (1, 2): return 'primaria'
    if n == 3: return 'cb_incomp'
    if n == 4: return 'cb_comp'
    if n == 5: return 'bach_incomp'
    if n == 6: return 'bach_comp'
    if n == 7: return 'ter_incomp'
    if n in (8, 9, 10): return 'ter_comp'
    return np.nan


def personas_a_cat(n):
    if pd.isna(n): return np.nan
    if n <= 2: return '1-2'
    if n <= 4: return '3-4'
    return '5+'


df['tramo_edad'] = df['edad'].apply(edad_a_tramo)
df['edad_25_34'] = (df['tramo_edad'] == '25-34').astype(int)
df['edad_35_44'] = (df['tramo_edad'] == '35-44').astype(int)
df['edad_45_54'] = (df['tramo_edad'] == '45-54').astype(int)
df['edad_55_plus'] = (df['tramo_edad'] == '55+').astype(int)

df['es_mujer'] = (df['sexo'] == 'F').astype(int)

df['nivel_educ_cat'] = df['nivel_educativo'].apply(educ_a_cat)
df['educ_cb_incomp'] = (df['nivel_educ_cat'] == 'cb_incomp').astype(int)
df['educ_cb_comp'] = (df['nivel_educ_cat'] == 'cb_comp').astype(int)
df['educ_bach_incomp'] = (df['nivel_educ_cat'] == 'bach_incomp').astype(int)
df['educ_bach_comp'] = (df['nivel_educ_cat'] == 'bach_comp').astype(int)
df['educ_ter_incomp'] = (df['nivel_educ_cat'] == 'ter_incomp').astype(int)
df['educ_ter_comp'] = (df['nivel_educ_cat'] == 'ter_comp').astype(int)

relig_map = {
    'Nada. Soy ateo / No creo en la religión': 'nada',
    'Poco. Me identifico culturalmente con alguna religión pero no soy practicante ni ella es muy importante en mi vida': 'poco',
    'Bastante. Me identifico con alguna religión y ella es importante en mi vida y mis valores': 'bastante',
    'Mucho. Me identifico con alguna religión y sigo sus prácticas y valores asistiendo a sus rituales y encuentros': 'mucho',
}
df['relig_cat'] = df['P178_Cuan_religioso'].map(relig_map)
df['relig_poco'] = (df['relig_cat'] == 'poco').astype(int)
df['relig_bastante'] = (df['relig_cat'] == 'bastante').astype(int)
df['relig_mucho'] = (df['relig_cat'] == 'mucho').astype(int)

df['es_montevideo'] = (df['dpto'] == 19).astype(int)
df['tiene_hijos'] = df['P159_Cuantos_hijos'].apply(
    lambda x: 0 if x == 'Ninguno' else 1 if pd.notna(x) else np.nan
)

df['hogar_cat'] = df['cant_personas'].apply(personas_a_cat)
df['hogar_3_4'] = (df['hogar_cat'] == '3-4').astype(int)
df['hogar_5_plus'] = (df['hogar_cat'] == '5+').astype(int)

df['balotaje_martinez'] = (df['IdBalotaje'] == 1).astype(int)
df['balotaje_lacalle'] = (df['IdBalotaje'] == 2).astype(int)

df['mujer_x_relig_mucho'] = df['es_mujer'] * df['relig_mucho']
df['mujer_x_tiene_hijos'] = df['es_mujer'] * df['tiene_hijos']

escala_5 = {
    'Totalmente en desacuerdo': 0, 'En desacuerdo': 0,
    'Ni de acuerdo ni en desacuerdo': np.nan,
    'De acuerdo': 1, 'Totalmente de acuerdo': 1,
}
df['favor_ive'] = df['P174_Decidir_embarazo'].map(escala_5)

datos = df[['favor_ive', 'w_norm'] + PREDICTORS].dropna()
print(f"N efectivo = {len(datos)}")

X = datos[PREDICTORS].astype(float)
y = datos['favor_ive'].astype(int)
w = datos['w_norm']


def fit_and_print(C_val, label):
    if C_val is None:
        m = LogisticRegression(penalty=None, solver='lbfgs', max_iter=5000, random_state=42)
    else:
        m = LogisticRegression(penalty='l2', C=C_val, solver='lbfgs', max_iter=5000, random_state=42)
    m.fit(X, y, sample_weight=w)
    coefs = dict(zip(PREDICTORS, m.coef_[0]))
    print(f"\n{'=' * 70}")
    print(f"{label}")
    print(f"{'=' * 70}")
    print(f"intercept = {m.intercept_[0]:+.4f}")
    educ_keys = ['educ_cb_incomp', 'educ_cb_comp', 'educ_bach_incomp',
                 'educ_bach_comp', 'educ_ter_incomp', 'educ_ter_comp']
    print("Coeficientes educación (ref: primaria):")
    for k in educ_keys:
        print(f"  {k:20} coef={coefs[k]:+.4f}  OR={np.exp(coefs[k]):.3f}")
    return coefs


fit_and_print(None, "SIN RIDGE (penalty=None) -- MLE puro")
fit_and_print(10.0, "Ridge muy débil (C=10)")
fit_and_print(0.5, "Ridge actual de producción (C=0.5)")

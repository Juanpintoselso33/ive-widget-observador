"""
Screening de variables candidatas para el modelo IVE.
Prueba cada variable de la encuesta como predictor univariado y bivariado
(controlando por las 22 variables ya incluidas) para encontrar las mejores
adiciones al modelo.

Uso: python scripts/screening_variables.py
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATA_FILE = BASE_DIR / 'base_limpia.csv'

W = 'w_norm'

print("Cargando datos...")
df = pd.read_csv(DATA_FILE)
print(f"Base cargada: {len(df)} casos\n")

# ==============================================================
# PREPARAR VARIABLE OBJETIVO
# ==============================================================
escala_5 = {
    'Totalmente en desacuerdo': 1, 'En desacuerdo': 2,
    'Ni de acuerdo ni en desacuerdo': 3, 'De acuerdo': 4,
    'Totalmente de acuerdo': 5
}
df['decidir_embarazo'] = df['P174_Decidir_embarazo'].map(escala_5)
df['favor_ive'] = np.where(
    df['decidir_embarazo'] >= 4, 1,
    np.where(df['decidir_embarazo'] <= 2, 0, np.nan)
)

# ==============================================================
# PREPARAR VARIABLES YA EN EL MODELO (para control)
# ==============================================================
def edad_a_tramo(edad):
    if pd.isna(edad): return np.nan
    if edad < 25: return '18-24'
    if edad < 35: return '25-34'
    if edad < 45: return '35-44'
    if edad < 55: return '45-54'
    return '55+'

df['tramo_edad'] = df['edad'].apply(edad_a_tramo)
df['edad_25_34'] = (df['tramo_edad'] == '25-34').astype(int)
df['edad_35_44'] = (df['tramo_edad'] == '35-44').astype(int)
df['edad_45_54'] = (df['tramo_edad'] == '45-54').astype(int)
df['edad_55_plus'] = (df['tramo_edad'] == '55+').astype(int)
df['es_mujer'] = (df['sexo'] == 'F').astype(int)

educ_map = {
    '1-PRIMARIA': 'primaria', '2-EMS INCOMP': 'ems_incomp',
    '3-EMS COMP': 'ems_comp', '4-TER INCOMP': 'ter_incomp',
    '5-TER COMP': 'ter_comp'
}
df['nivel_educ_cat'] = df['nivel_educ'].map(educ_map)
df['educ_ems_incomp'] = (df['nivel_educ_cat'] == 'ems_incomp').astype(int)
df['educ_ems_comp'] = (df['nivel_educ_cat'] == 'ems_comp').astype(int)
df['educ_ter_incomp'] = (df['nivel_educ_cat'] == 'ter_incomp').astype(int)
df['educ_ter_comp'] = (df['nivel_educ_cat'] == 'ter_comp').astype(int)

relig_map = {
    'Nada. Soy ateo / No creo en la religión': 'nada',
    'Poco. Me identifico culturalmente con alguna religión pero no soy practicante ni ella es muy importante en mi vida': 'poco',
    'Bastante. Me identifico con alguna religión y ella es importante en mi vida y mis valores': 'bastante',
    'Mucho. Me identifico con alguna religión y sigo sus prácticas y valores asistiendo a sus rituales y encuentros': 'mucho'
}
df['relig_cat'] = df['P178_Cuan_religioso'].map(relig_map)
df['relig_poco'] = (df['relig_cat'] == 'poco').astype(int)
df['relig_bastante'] = (df['relig_cat'] == 'bastante').astype(int)
df['relig_mucho'] = (df['relig_cat'] == 'mucho').astype(int)
df['es_montevideo'] = (df['dpto'] == 19).astype(int)
df['tiene_hijos'] = df['P159_Cuantos_hijos'].apply(
    lambda x: 0 if x == 'Ninguno' else 1 if pd.notna(x) else np.nan
)

def personas_a_cat(n):
    if pd.isna(n): return np.nan
    if n <= 2: return '1-2'
    if n <= 4: return '3-4'
    return '5+'
df['hogar_cat'] = df['cant_personas'].apply(personas_a_cat)
df['hogar_3_4'] = (df['hogar_cat'] == '3-4').astype(int)
df['hogar_5_plus'] = (df['hogar_cat'] == '5+').astype(int)

voto_map = {
    1: 'PN', 2: 'PC', 3: 'FA', 4: 'CA',
    5: 'Otros', 6: 'NV', 7: 'NV', 8: 'NV', 9: 'NV', 10: 'NV', 11: 'NV'
}
df['voto_2019'] = df['IdOpcionElecciones2019'].map(voto_map)
df['voto_fa'] = (df['voto_2019'] == 'FA').astype(int)
df['voto_pn'] = (df['voto_2019'] == 'PN').astype(int)
df['voto_pc'] = (df['voto_2019'] == 'PC').astype(int)
df['voto_ca'] = (df['voto_2019'] == 'CA').astype(int)
df['mujer_x_relig_mucho'] = df['es_mujer'] * df['relig_mucho']
df['mujer_x_tiene_hijos'] = df['es_mujer'] * df['tiene_hijos']

CURRENT_PREDICTORS = [
    'edad_25_34', 'edad_35_44', 'edad_45_54', 'edad_55_plus',
    'es_mujer',
    'educ_ems_incomp', 'educ_ems_comp', 'educ_ter_incomp', 'educ_ter_comp',
    'relig_poco', 'relig_bastante', 'relig_mucho',
    'es_montevideo', 'tiene_hijos',
    'hogar_3_4', 'hogar_5_plus',
    'voto_fa', 'voto_pn', 'voto_pc', 'voto_ca',
    'mujer_x_relig_mucho', 'mujer_x_tiene_hijos',
]

# ==============================================================
# MODELO BASE (sin variables nuevas) para comparar
# ==============================================================
base_data = df[['favor_ive', W] + CURRENT_PREDICTORS].dropna()
X_base = base_data[CURRENT_PREDICTORS].values
y_base = base_data['favor_ive'].values
w_base = base_data[W].values

modelo_base = LogisticRegression(C=0.5, solver='lbfgs', max_iter=2000, random_state=42)
modelo_base.fit(X_base, y_base, sample_weight=w_base)
y_pred_base = modelo_base.predict_proba(X_base)[:, 1]
ll_base = -log_loss(y_base, y_pred_base, sample_weight=w_base, normalize=False)
ll_null = -log_loss(y_base, np.full(len(y_base), np.average(y_base, weights=w_base)),
                    sample_weight=w_base, normalize=False)
r2_base = 1 - (ll_base / ll_null)

print(f"Modelo base: {len(CURRENT_PREDICTORS)} predictores, R2={r2_base:.4f}, N={len(base_data)}")
print(f"Log-Lik base = {ll_base:.2f}, Log-Lik null = {ll_null:.2f}")
print()

# ==============================================================
# FUNCIONES HELPER
# ==============================================================
def likert5_to_numeric(series):
    mapping = {
        'Totalmente en desacuerdo': 1, 'En desacuerdo': 2,
        'Ni de acuerdo ni en desacuerdo': 3, 'De acuerdo': 4,
        'Totalmente de acuerdo': 5
    }
    return series.map(mapping)

def likert5_to_binary(series):
    """Binariza Likert: 4-5 = 1, 1-2 = 0, 3 = NaN."""
    num = likert5_to_numeric(series)
    return np.where(num >= 4, 1, np.where(num <= 2, 0, np.nan))

def test_variable_univariate(name, series):
    """Test univariado: logit solo con esta variable vs favor_ive."""
    tmp = pd.DataFrame({'y': df['favor_ive'], 'w': df[W], 'x': series}).dropna()
    if len(tmp) < 100 or tmp['y'].nunique() < 2:
        return None
    X = tmp[['x']].values
    y = tmp['y'].values
    w = tmp['w'].values
    try:
        m = LogisticRegression(C=1e6, solver='lbfgs', max_iter=2000, random_state=42)
        m.fit(X, y, sample_weight=w)
        yp = m.predict_proba(X)[:, 1]
        ll = -log_loss(y, yp, sample_weight=w, normalize=False)
        ll_n = -log_loss(y, np.full(len(y), np.average(y, weights=w)),
                         sample_weight=w, normalize=False)
        r2 = 1 - (ll / ll_n)
        coef = m.coef_[0][0]
        return {'name': name, 'r2_univar': round(r2, 4), 'coef': round(coef, 3),
                'N': len(tmp), 'pct_favor': round(np.average(y, weights=w) * 100, 1)}
    except Exception:
        return None

def test_variable_incremental(name, new_cols_series_dict):
    """Test incremental: agrega variable(s) al modelo base y mide mejora en R2."""
    tmp = base_data.copy()
    for col_name, series in new_cols_series_dict.items():
        tmp[col_name] = series.reindex(tmp.index)
    tmp = tmp.dropna()
    if len(tmp) < 100:
        return None

    new_pred = CURRENT_PREDICTORS + list(new_cols_series_dict.keys())
    X = tmp[new_pred].values
    y = tmp['favor_ive'].values
    w = tmp[W].values
    try:
        m = LogisticRegression(C=0.5, solver='lbfgs', max_iter=2000, random_state=42)
        m.fit(X, y, sample_weight=w)
        yp = m.predict_proba(X)[:, 1]
        ll = -log_loss(y, yp, sample_weight=w, normalize=False)
        ll_n = -log_loss(y, np.full(len(y), np.average(y, weights=w)),
                         sample_weight=w, normalize=False)
        r2 = 1 - (ll / ll_n)
        delta_r2 = r2 - r2_base
        # Extract coefficients of new variables
        new_coefs = {}
        for i, col_name in enumerate(new_cols_series_dict.keys()):
            idx = CURRENT_PREDICTORS.__len__() + i
            new_coefs[col_name] = round(m.coef_[0][idx], 3)
        return {'name': name, 'r2_full': round(r2, 4), 'delta_r2': round(delta_r2, 4),
                'N': len(tmp), 'new_coefs': new_coefs}
    except Exception as e:
        return {'name': name, 'error': str(e)}

# ==============================================================
# DEFINIR VARIABLES CANDIDATAS
# ==============================================================
print("=" * 70)
print("SCREENING DE VARIABLES CANDIDATAS")
print("=" * 70)

results_uni = []
results_inc = []

# --- 1. Variables Likert (escala 5 puntos) ---
likert_vars = {
    'P169_Pareja_convive': 'P169_Pareja_convive_sin_casar',
    'P170_Adopcion_mismo_sexo': 'P170_Parejas_mismo_sexo_adoptar',
    'P171_Familia_dos_adultos': 'P171_Familia_dos_adultos',
    'P173_Educ_sexual_oblig': 'P173_Educ_sexual_obligatoria',
    'P175_Genero_escuela': 'P175_Identidades_genero_escuela',
    'P176_Genero_medios': 'P176_Identidades_genero_medios',
}

for short_name, col in likert_vars.items():
    # Como numérica continua
    num = likert5_to_numeric(df[col])
    r = test_variable_univariate(short_name, num)
    if r:
        results_uni.append(r)
    # Incremental como numérica
    series_dict = {f'{short_name}_num': num}
    r_inc = test_variable_incremental(short_name, series_dict)
    if r_inc:
        results_inc.append(r_inc)

# P173 tiene escala diferente
educ_sex_map = {
    'Definitivamente sí': 5, 'Probablemente sí': 4,
    'No estoy seguro/a': 3, 'Probablemente no': 2, 'Definitivamente no': 1
}
df['P173_num'] = df['P173_Educ_sexual_obligatoria'].map(educ_sex_map)
r = test_variable_univariate('P173_Educ_sexual (corr)', df['P173_num'])
if r:
    results_uni.append(r)
r_inc = test_variable_incremental('P173_Educ_sexual (corr)', {'P173_educ_sex_num': df['P173_num']})
if r_inc:
    results_inc.append(r_inc)

# P172 Crianza equitativa
crianza_map = {
    'Sí, totalmente': 5, 'En gran medida': 4,
    'Solo en parte': 3, 'No estoy seguro/a': 2,
    'No, la responsabilidad debe recaer principalmente en la madre': 1
}
df['P172_num'] = df['P172_Crianza_equitativa'].map(crianza_map)
r = test_variable_univariate('P172_Crianza_equitativa', df['P172_num'])
if r:
    results_uni.append(r)
r_inc = test_variable_incremental('P172_Crianza_equitativa', {'P172_crianza_num': df['P172_num']})
if r_inc:
    results_inc.append(r_inc)

# --- 2. Balotaje ---
# 1=Lacalle, 2=Martinez, 3=Blanco, 4=No voto, 5=No recuerda
df['balotaje_martinez'] = (df['IdBalotaje'] == 2).astype(int)
df['balotaje_lacalle'] = (df['IdBalotaje'] == 1).astype(int)
r = test_variable_univariate('Balotaje_Martinez', df['balotaje_martinez'])
if r:
    results_uni.append(r)
r_inc = test_variable_incremental('Balotaje', {
    'balotaje_martinez': df['balotaje_martinez'],
    'balotaje_lacalle': df['balotaje_lacalle']
})
if r_inc:
    results_inc.append(r_inc)

# --- 3. Orientacion sexual ---
df['es_heterosexual'] = (df['P177_Orientacion_sexual'] == 'Heterosexual').astype(int)
df['es_lgb'] = (~df['P177_Orientacion_sexual'].isin(['Heterosexual', 'No estoy seguro/a']) &
                df['P177_Orientacion_sexual'].notna()).astype(int)
r = test_variable_univariate('Orientacion_sexual_LGB', df['es_lgb'])
if r:
    results_uni.append(r)
r_inc = test_variable_incremental('Orientacion_sexual', {'es_lgb': df['es_lgb']})
if r_inc:
    results_inc.append(r_inc)

# --- 4. Condicion de actividad ---
for cat in ['ocupado', 'jubilado_pensionista', 'estudiante', 'desocupado']:
    dummy = (df['cond_act'] == cat).astype(int)
    r = test_variable_univariate(f'cond_act_{cat}', dummy)
    if r:
        results_uni.append(r)

# Incremental: dummies de cond_act (ref: ocupado)
cond_dummies = {}
for cat in ['jubilado_pensionista', 'estudiante', 'desocupado', 'inactivo', 'inactivo_hogar']:
    cond_dummies[f'cond_{cat}'] = (df['cond_act'] == cat).astype(int)
r_inc = test_variable_incremental('cond_act (dummies)', cond_dummies)
if r_inc:
    results_inc.append(r_inc)

# --- 5. Rol en hogar ---
# 1=Jefe, 2=Conyuge, 3=Hijo, 4=Otro pariente, 5=No pariente, 6=Otro
for rol_id, rol_name in [(1, 'jefe'), (6, 'otro'), (2, 'conyuge'), (3, 'hijo')]:
    dummy = (df['IdRolEnHogar'] == rol_id).astype(int)
    r = test_variable_univariate(f'rol_{rol_name}', dummy)
    if r:
        results_uni.append(r)

# --- 6. Menores en hogar ---
df['tiene_menores'] = (df['menores_14'] > 0).astype(int)
r = test_variable_univariate('tiene_menores_14', df['tiene_menores'])
if r:
    results_uni.append(r)
r_inc = test_variable_incremental('tiene_menores_14', {'tiene_menores': df['tiene_menores']})
if r_inc:
    results_inc.append(r_inc)

# --- 7. Mayores 65 en hogar ---
df['tiene_mayores65'] = (df['Mayores65'] > 0).astype(int)
r = test_variable_univariate('tiene_mayores65', df['tiene_mayores65'])
if r:
    results_uni.append(r)
r_inc = test_variable_incremental('tiene_mayores65', {'tiene_mayores65': df['tiene_mayores65']})
if r_inc:
    results_inc.append(r_inc)

# --- 8. Banos en hogar (proxy socioecon) ---
r = test_variable_univariate('banios_hogar', df['BaniosEnHogar'].clip(upper=5))
if r:
    results_uni.append(r)
r_inc = test_variable_incremental('banios_hogar', {'banios': df['BaniosEnHogar'].clip(upper=5)})
if r_inc:
    results_inc.append(r_inc)

# --- 9. Habitaciones ---
r = test_variable_univariate('hab_hog', df['hab_hog'].clip(upper=8))
if r:
    results_uni.append(r)
r_inc = test_variable_incremental('hab_hog', {'hab_hog': df['hab_hog'].clip(upper=8)})
if r_inc:
    results_inc.append(r_inc)

# --- 10. HogarPaga (servicio domestico?) ---
df['hogar_paga'] = (df['HogarPaga'] == 'Si').astype(int)
r = test_variable_univariate('HogarPaga', df['hogar_paga'])
if r:
    results_uni.append(r)
r_inc = test_variable_incremental('HogarPaga', {'hogar_paga': df['hogar_paga']})
if r_inc:
    results_inc.append(r_inc)

# --- 11. Rankings de prioridades vitales ---
for col_short, col_full in [
    ('Ranking_Trabajo', 'P180_Ranking_Trabajo'),
    ('Ranking_Pareja', 'P181_Ranking_Pareja'),
    ('Ranking_Hijos', 'P182_Ranking_Hijos'),
    ('Ranking_Tiempo', 'P183_Ranking_Tiempo'),
    ('Ranking_Comunidad', 'P184_Ranking_Comunidad'),
    ('Ranking_Dinero', 'P185_Ranking_Dinero'),
]:
    r = test_variable_univariate(col_short, df[col_full])
    if r:
        results_uni.append(r)
    r_inc = test_variable_incremental(col_short, {col_short.lower(): df[col_full]})
    if r_inc:
        results_inc.append(r_inc)

# --- 12. Identidad de genero (no binario / trans) ---
df['genero_no_cis'] = (~df['IdentidadGenero'].isin(['Mujer', 'Varon']) &
                        df['IdentidadGenero'].notna()).astype(int)
r = test_variable_univariate('genero_no_cis', df['genero_no_cis'])
if r:
    results_uni.append(r)

# --- 13. Edad hijo mayor (proxy de etapa vital) ---
r = test_variable_univariate('edad_hijo_mayor', df['P166_Edad_hijo_mayor'])
if r:
    results_uni.append(r)
r_inc = test_variable_incremental('edad_hijo_mayor', {'edad_hijo_mayor': df['P166_Edad_hijo_mayor']})
if r_inc:
    results_inc.append(r_inc)

# --- 14. Razon pocos hijos ---
df['razon_economica'] = (df['P165_Razon_pocos_hijos'] == 'Por falta de medios económicos').astype(int)
df['razon_conciliacion'] = (df['P165_Razon_pocos_hijos'].str.contains('conciliaci', na=False)).astype(int)
r = test_variable_univariate('razon_economica', df['razon_economica'])
if r:
    results_uni.append(r)
r = test_variable_univariate('razon_conciliacion', df['razon_conciliacion'])
if r:
    results_uni.append(r)

# --- 15. Estrato socioeconomico ---
# Extract rank number from estrato
df['estrato_rank'] = df['estrato'].str.extract(r'rank-(\d+)').astype(float)
r = test_variable_univariate('estrato_rank', df['estrato_rank'])
if r:
    results_uni.append(r)
r_inc = test_variable_incremental('estrato_rank', {'estrato_rank': df['estrato_rank']})
if r_inc:
    results_inc.append(r_inc)

# ==============================================================
# RESULTADOS
# ==============================================================
print("\n" + "=" * 70)
print("RESULTADOS UNIVARIADOS (R2 del logit simple)")
print("=" * 70)
results_uni.sort(key=lambda x: x['r2_univar'], reverse=True)
print(f"{'Variable':<35} {'R2_univ':>8} {'Coef':>8} {'N':>6} {'%Favor':>7}")
print("-" * 70)
for r in results_uni:
    print(f"{r['name']:<35} {r['r2_univar']:>8.4f} {r['coef']:>8.3f} {r['N']:>6} {r['pct_favor']:>6.1f}%")

print("\n" + "=" * 70)
print(f"RESULTADOS INCREMENTALES (R2 base = {r2_base:.4f})")
print("Mejora en R2 al agregar variable al modelo de 22 predictores")
print("=" * 70)
results_inc.sort(key=lambda x: x.get('delta_r2', -1), reverse=True)
print(f"{'Variable':<35} {'R2_full':>8} {'Delta_R2':>9} {'N':>6} {'Coefs nuevos'}")
print("-" * 70)
for r in results_inc:
    if 'error' in r:
        print(f"{r['name']:<35} ERROR: {r['error']}")
        continue
    coef_str = ', '.join(f"{k}={v:+.3f}" for k, v in r['new_coefs'].items())
    flag = " ***" if r['delta_r2'] >= 0.005 else " **" if r['delta_r2'] >= 0.002 else " *" if r['delta_r2'] >= 0.001 else ""
    print(f"{r['name']:<35} {r['r2_full']:>8.4f} {r['delta_r2']:>+9.4f} {r['N']:>6} {coef_str}{flag}")

print("\n" + "=" * 70)
print("LEYENDA: *** > 0.5pp mejora | ** > 0.2pp | * > 0.1pp")
print("=" * 70)

# ==============================================================
# CORRELACION ENTRE CANDIDATAS Y FAVOR_IVE
# ==============================================================
print("\n" + "=" * 70)
print("TABLA CRUZADA: % A FAVOR IVE POR GRUPO (ponderado)")
print("=" * 70)

cross_tabs = {
    'P170_Adopcion_mismo_sexo': {
        'col': 'P170_Parejas_mismo_sexo_adoptar',
        'cats': ['Totalmente en desacuerdo', 'En desacuerdo',
                 'Ni de acuerdo ni en desacuerdo', 'De acuerdo', 'Totalmente de acuerdo']
    },
    'P173_Educ_sexual': {
        'col': 'P173_Educ_sexual_obligatoria',
        'cats': ['Definitivamente no', 'Probablemente no', 'No estoy seguro/a',
                 'Probablemente sí', 'Definitivamente sí']
    },
    'Balotaje': {
        'col': 'IdBalotaje',
        'cats': [1, 2, 3, 4, 5],
        'labels': ['Lacalle', 'Martinez', 'Blanco', 'No voto', 'No recuerda']
    },
    'Orientacion': {
        'col': 'P177_Orientacion_sexual',
        'cats': ['Heterosexual', 'Bisexual', 'Gay', 'Lesbiana', 'Otro']
    },
    'cond_act': {
        'col': 'cond_act',
        'cats': ['ocupado', 'jubilado_pensionista', 'estudiante', 'desocupado', 'inactivo']
    },
}

for var_name, spec in cross_tabs.items():
    col = spec['col']
    cats = spec['cats']
    labels = spec.get('labels', [str(c) for c in cats])
    print(f"\n--- {var_name} ---")
    for cat, label in zip(cats, labels):
        subset = df[(df[col] == cat) & df['favor_ive'].notna()]
        if len(subset) > 10:
            pct = np.average(subset['favor_ive'], weights=subset[W]) * 100
            print(f"  {label:<40} {pct:5.1f}% (N={len(subset)})")

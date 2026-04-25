"""
Validación interna: modelo logit ordinal (proportional odds) sobre Likert 1-5.

Compara el ranking y signo de los coeficientes de un modelo ordinal entrenado
sobre la escala completa (1-5, sin excluir neutrales) con los coeficientes del
modelo binario de producción (≥4 vs ≤2, neutrales excluidos).

El objetivo NO es reemplazar el modelo binario en producción --- el ordinal es
más sensible a la cardinalidad arbitraria de la escala Likert (asume distancias
iguales entre categorías) y carece de la interpretabilidad directa de
"prob de apoyo". Sólo se usa para verificar que la decisión de dicotomizar y
excluir neutrales no invierte ni distorsiona seriamente la estructura de
asociaciones.

Caveat: statsmodels OrderedModel no soporta sample_weight directamente; este
script ajusta sin pesos. La consistencia de signos y orden de magnitud se
preserva razonablemente entre la versión ponderada y no ponderada del binario,
así que el ranking del ordinal sin pesos sigue siendo informativo.

Uso:
    python scripts/validacion_ordinal.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR.parent / 'base_limpia.csv'
COEF_FILE = BASE_DIR / 'model_coefficients.json'

PREDICTORS = [
    'edad_25_34', 'edad_35_44', 'edad_45_54', 'edad_55_plus',
    'es_mujer',
    'educ_cb', 'educ_bach_incomp', 'educ_bach_comp', 'educ_ter_incomp', 'educ_ter_comp',
    'relig_poco', 'relig_bastante', 'relig_mucho',
    'es_montevideo',
    'tiene_hijos',
    'hogar_3_4', 'hogar_5_plus',
    'balotaje_martinez', 'balotaje_lacalle',
    'mujer_x_relig_mucho', 'mujer_x_tiene_hijos',
]


def edad_a_tramo(edad):
    if pd.isna(edad):
        return np.nan
    if edad < 25: return '18-24'
    if edad < 35: return '25-34'
    if edad < 45: return '35-44'
    if edad < 55: return '45-54'
    return '55+'


def educ_a_cat(n):
    if pd.isna(n):
        return np.nan
    n = int(n)
    if n in (1, 2): return 'primaria'
    if n in (3, 4): return 'cb'
    if n == 5:      return 'bach_incomp'
    if n == 6:      return 'bach_comp'
    if n == 7:      return 'ter_incomp'
    if n in (8, 9, 10): return 'ter_comp'
    return np.nan


def personas_a_cat(n):
    if pd.isna(n):
        return np.nan
    if n <= 2: return '1-2'
    if n <= 4: return '3-4'
    return '5+'


def construir_features(df):
    df = df.copy()
    df['tramo_edad'] = df['edad'].apply(edad_a_tramo)
    df['edad_25_34'] = (df['tramo_edad'] == '25-34').astype(int)
    df['edad_35_44'] = (df['tramo_edad'] == '35-44').astype(int)
    df['edad_45_54'] = (df['tramo_edad'] == '45-54').astype(int)
    df['edad_55_plus'] = (df['tramo_edad'] == '55+').astype(int)

    df['es_mujer'] = (df['sexo'] == 'F').astype(int)

    df['nivel_educ_cat'] = df['nivel_educativo'].apply(educ_a_cat)
    df['educ_cb'] = (df['nivel_educ_cat'] == 'cb').astype(int)
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
        'Totalmente en desacuerdo': 1,
        'En desacuerdo': 2,
        'Ni de acuerdo ni en desacuerdo': 3,
        'De acuerdo': 4,
        'Totalmente de acuerdo': 5,
    }
    df['decidir_embarazo'] = df['P174_Decidir_embarazo'].map(escala_5)
    return df


def main():
    print("=" * 70)
    print("VALIDACIÓN INTERNA: LOGIT ORDINAL (Likert 1-5) vs BINARIO")
    print("=" * 70)

    print("\nCargando datos y construyendo features...")
    df = pd.read_csv(DATA_FILE)
    df = construir_features(df)

    datos = df[['decidir_embarazo'] + PREDICTORS].dropna()
    datos = datos[datos['decidir_embarazo'].isin([1, 2, 3, 4, 5])]
    print(f"N para ordinal (incluye neutrales y extremos): {len(datos)}")

    X = datos[PREDICTORS].astype(float)
    y = datos['decidir_embarazo'].astype(int)

    print("\nDistribución Likert:")
    print(y.value_counts().sort_index())

    print("\nAjustando modelo ordinal (proportional odds, logit link)...")
    model = OrderedModel(y, X, distr='logit')
    res = model.fit(method='bfgs', maxiter=2000, disp=False)

    print(f"\nLog-likelihood ordinal: {res.llf:.2f}")
    print(f"Pseudo R² (McFadden):   {1 - res.llf / res.llnull:.4f}")

    print("\nCargando coeficientes binarios de producción...")
    with open(COEF_FILE, 'r', encoding='utf-8') as f:
        bin_model = json.load(f)
    bin_coef = bin_model['coefficients']

    rows = []
    for var in PREDICTORS:
        b_bin = bin_coef.get(var, 0.0)
        b_ord = float(res.params[var])
        rows.append({
            'variable': var,
            'coef_binario': b_bin,
            'coef_ordinal': b_ord,
            'signo_coincide': (b_bin * b_ord >= 0),
        })
    cmp = pd.DataFrame(rows)

    cmp['rank_bin'] = cmp['coef_binario'].abs().rank(ascending=False).astype(int)
    cmp['rank_ord'] = cmp['coef_ordinal'].abs().rank(ascending=False).astype(int)
    cmp['delta_rank'] = (cmp['rank_bin'] - cmp['rank_ord']).abs()

    print("\n" + "-" * 70)
    print("COMPARACIÓN DE COEFICIENTES (binario producción vs ordinal Likert 1-5)")
    print("-" * 70)
    print(cmp.to_string(index=False))

    n_signo = int(cmp['signo_coincide'].sum())
    rho_global = cmp[['rank_bin', 'rank_ord']].corr(method='spearman').iloc[0, 1]

    # Top-5 por magnitud absoluta en el binario (predictores dominantes)
    top5 = cmp.nlargest(5, cmp['coef_binario'].abs().name) if False else \
           cmp.iloc[cmp['coef_binario'].abs().argsort()[::-1][:5]]
    rho_top5 = top5[['rank_bin', 'rank_ord']].corr(method='spearman').iloc[0, 1]
    n_signo_top5 = int(top5['signo_coincide'].sum())

    print(f"\nVariables con mismo signo (todas):     {n_signo}/{len(cmp)}")
    print(f"Spearman ranks |coef| (todas):         {rho_global:.3f}")
    print(f"Variables con mismo signo (top-5 |coef|): {n_signo_top5}/5")
    print(f"Spearman ranks |coef| (top-5):         {rho_top5:.3f}")
    print("\nTop-5 predictores binarios:")
    print(top5[['variable', 'coef_binario', 'coef_ordinal', 'rank_bin', 'rank_ord']]
          .to_string(index=False))

    if n_signo_top5 == 5 and rho_top5 >= 0.9:
        print("\n[OK] Los predictores dominantes (top-5 por magnitud) preservan signo")
        print("     y orden de ranking en el ordinal. La dicotomización + exclusión")
        print("     de neutrales NO invierte la estructura de asociaciones principal.")
        if n_signo < len(cmp):
            print(f"     Discrepancias menores en {len(cmp) - n_signo} variable(s) de baja magnitud.")
    else:
        print("\n[ATENCIÓN] Inconsistencias en predictores dominantes; revisar antes de publicar.")


if __name__ == '__main__':
    main()

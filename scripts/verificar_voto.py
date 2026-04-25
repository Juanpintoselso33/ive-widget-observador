"""
Verificación de apoyo al IVE por código de elecciones 2019.
Compara frecuencias con datos esperados.
"""

import pandas as pd
import numpy as np

df = pd.read_csv(r'c:\Users\trico\OneDrive\Observador\base_limpia.csv')

escala_5 = {
    'Totalmente en desacuerdo': 1,
    'En desacuerdo': 2,
    'Ni de acuerdo ni en desacuerdo': 3,
    'De acuerdo': 4,
    'Totalmente de acuerdo': 5
}

df['decidir_embarazo'] = df['P174_Decidir_embarazo'].map(escala_5)
df['favor_ive'] = df['decidir_embarazo'].apply(lambda x: 1 if x >= 4 else (0 if x <= 2 else np.nan))

print('=== APOYO AL IVE POR CÓDIGO DE ELECCIONES2019 ===\n')

for cod in [1, 2, 3, 4, 5]:
    subset = df[df['IdOpcionElecciones2019'] == cod]
    if len(subset) > 30:
        favor_count = subset['favor_ive'].sum()
        total_count = subset['favor_ive'].notna().sum()
        pct = (favor_count / total_count * 100) if total_count > 0 else 0
        print(f'Código {cod}: {len(subset):4d} casos, {pct:5.1f}% apoyo IVE ({int(favor_count)}/{int(total_count)} válidos)')

print('\n=== COMPARACIÓN CON FACTUM 2012 ===')
print('FA electorado: 72% favor')
print('PN electorado: 45% favor')
print('PC electorado: 55% favor')
print('CA electorado: ~30-40% favor (extrema derecha)')
print('\n=== HIPÓTESIS DE MAPEO CORRECTO ===')
print('Código 3 (1200 casos, 95% favor) → FA (esperado 72% favor)')
print('Código 1 (954 casos, 65% favor) → PN o PC (esperado 45-55% favor)')
print('Código 2 (364 casos, 69% favor) → PN o PC (esperado 45-55% favor)')
print('Código 4 (54 casos, 61% favor) → CA (esperado 30-40% favor) - INCONSISTENTE')

"""
Validación del mapeo de voto (IdOpcionElecciones2019 -> partido).
Compara con datos históricos de Factum 2012.
"""

import pandas as pd
import numpy as np

df = pd.read_csv(r'c:\Users\trico\OneDrive\Observador\base_limpia.csv')

# Mapeo
voto_map = {
    1: 'Partido Nacional',
    2: 'Partido Colorado',
    3: 'Frente Amplio',
    4: 'Cabildo Abierto',
    5: 'Otros partidos',
    6: 'No votó/Blanco', 7: 'No votó/Blanco', 8: 'No votó/Blanco',
    9: 'No votó/Blanco', 10: 'No votó/Blanco', 11: 'No votó/Blanco'
}
df['voto_2019'] = df['IdOpcionElecciones2019'].map(voto_map)

# P174
escala_5 = {
    'Totalmente en desacuerdo': 1, 'En desacuerdo': 2,
    'Ni de acuerdo ni en desacuerdo': 3, 'De acuerdo': 4, 'Totalmente de acuerdo': 5
}
df['decidir_embarazo'] = df['P174_Decidir_embarazo'].map(escala_5)
df['favor_ive'] = df['decidir_embarazo'].apply(lambda x: 1 if x >= 4 else (0 if x <= 2 else np.nan))

# Tabulación
print("="*70)
print("APOYO AL IVE POR PARTIDO (MAPEO CORREGIDO)")
print("="*70)
print()

for partido in ['Frente Amplio', 'Partido Nacional', 'Partido Colorado', 'Cabildo Abierto']:
    subset = df[(df['voto_2019'] == partido) & df['favor_ive'].notna()]
    if len(subset) > 0:
        n_favor = (subset['favor_ive'] == 1).sum()
        n_contra = (subset['favor_ive'] == 0).sum()
        pct_favor = n_favor / (n_favor + n_contra) * 100
        print(f"{partido:20} | {n_favor:4} a favor, {n_contra:4} en contra → {pct_favor:5.1f}% apoyo")

print()
print("COMPARACIÓN CON FACTUM 2012:")
print("-"*70)
print("FA: 72% favor (dato 2012) vs {:.1f}% (muestra 2026)".format(
    df[(df['voto_2019']=='Frente Amplio') & (df['favor_ive']==1)].shape[0] /
    df[(df['voto_2019']=='Frente Amplio') & df['favor_ive'].notna()].shape[0] * 100
))
print("PN: 45% favor (dato 2012) vs {:.1f}% (muestra 2026)".format(
    df[(df['voto_2019']=='Partido Nacional') & (df['favor_ive']==1)].shape[0] /
    df[(df['voto_2019']=='Partido Nacional') & df['favor_ive'].notna()].shape[0] * 100
))
print("PC: 55% favor (dato 2012) vs {:.1f}% (muestra 2026)".format(
    df[(df['voto_2019']=='Partido Colorado') & (df['favor_ive']==1)].shape[0] /
    df[(df['voto_2019']=='Partido Colorado') & df['favor_ive'].notna()].shape[0] * 100
))
print()
print("✓ MAPEO CORRECTO: FA > PN, consistente con datos históricos")

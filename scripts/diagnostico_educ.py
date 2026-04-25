"""
Diagnóstico: por qué los coeficientes ajustados de CB > Bach al controlar por
confounders, mientras las tasas crudas siguen el orden esperado.

Tabula n, tasa cruda de apoyo IVE, y la composición de religiosidad / edad /
balotaje dentro de cada categoría educativa para ver qué confounder explica
la inversión condicional.
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent / 'base_limpia.csv'
df = pd.read_csv(BASE)

def educ_a_cat(n):
    if pd.isna(n):
        return np.nan
    n = int(n)
    if n in (1, 2): return '1-primaria'
    if n == 3:      return '2-cb_incomp'
    if n == 4:      return '3-cb_comp'
    if n == 5:      return '4-bach_incomp'
    if n == 6:      return '5-bach_comp'
    if n == 7:      return '6-ter_incomp'
    if n in (8, 9, 10): return '7-ter_comp'
    return np.nan

df['educ_cat'] = df['nivel_educativo'].apply(educ_a_cat)

# Likert -> binario
escala_5 = {
    'Totalmente en desacuerdo': 0, 'En desacuerdo': 0,
    'Ni de acuerdo ni en desacuerdo': np.nan,
    'De acuerdo': 1, 'Totalmente de acuerdo': 1,
}
df['favor_ive'] = df['P174_Decidir_embarazo'].map(escala_5)

relig_map = {
    'Nada. Soy ateo / No creo en la religión': 'nada',
    'Poco. Me identifico culturalmente con alguna religión pero no soy practicante ni ella es muy importante en mi vida': 'poco',
    'Bastante. Me identifico con alguna religión y ella es importante en mi vida y mis valores': 'bastante',
    'Mucho. Me identifico con alguna religión y sigo sus prácticas y valores asistiendo a sus rituales y encuentros': 'mucho',
}
df['relig_cat'] = df['P178_Cuan_religioso'].map(relig_map)

W = 'w_norm'

print("=" * 80)
print("DISTRIBUCIÓN DE n_efectivo, n_ponderado, y tasas por categoría educativa")
print("=" * 80)

g = df.groupby('educ_cat', dropna=False)
out = pd.DataFrame({
    'n_sin_pesos': g.size(),
    'n_ponderado': g[W].sum().round(0),
    'apoyo_ive_crudo_%': g.apply(
        lambda x: round(np.average(x.dropna(subset=['favor_ive'])['favor_ive'],
                                    weights=x.dropna(subset=['favor_ive'])[W]) * 100, 1)
        if x['favor_ive'].notna().any() else np.nan,
        include_groups=False,
    ),
})
print(out)

print("\n" + "=" * 80)
print("% MUY RELIGIOSOS (relig_cat == 'mucho') por categoría educativa (ponderado)")
print("=" * 80)
for cat in sorted(df['educ_cat'].dropna().unique()):
    sub = df[df['educ_cat'] == cat].dropna(subset=['relig_cat'])
    if len(sub):
        pct = np.average((sub['relig_cat'] == 'mucho').astype(int), weights=sub[W]) * 100
        print(f"{cat:20}  {pct:5.1f}%  (n={len(sub)})")

print("\n" + "=" * 80)
print("% NADA RELIGIOSOS por categoría educativa (ponderado)")
print("=" * 80)
for cat in sorted(df['educ_cat'].dropna().unique()):
    sub = df[df['educ_cat'] == cat].dropna(subset=['relig_cat'])
    if len(sub):
        pct = np.average((sub['relig_cat'] == 'nada').astype(int), weights=sub[W]) * 100
        print(f"{cat:20}  {pct:5.1f}%  (n={len(sub)})")

print("\n" + "=" * 80)
print("% VOTÓ LACALLE EN BALOTAJE por categoría educativa (ponderado)")
print("=" * 80)
for cat in sorted(df['educ_cat'].dropna().unique()):
    sub = df[df['educ_cat'] == cat].dropna(subset=['IdBalotaje'])
    if len(sub):
        pct = np.average((sub['IdBalotaje'] == 2).astype(int), weights=sub[W]) * 100
        print(f"{cat:20}  {pct:5.1f}%  (n={len(sub)})")

print("\n" + "=" * 80)
print("EDAD MEDIA por categoría educativa (ponderada)")
print("=" * 80)
for cat in sorted(df['educ_cat'].dropna().unique()):
    sub = df[df['educ_cat'] == cat].dropna(subset=['edad'])
    if len(sub):
        m = np.average(sub['edad'], weights=sub[W])
        print(f"{cat:20}  edad_media={m:5.1f}  (n={len(sub)})")

print("\n" + "=" * 80)
print("TASA CRUDA DE APOYO IVE DENTRO DE 'MUY RELIGIOSOS' por cat. educativa")
print("=" * 80)
for cat in sorted(df['educ_cat'].dropna().unique()):
    sub = df[(df['educ_cat'] == cat) & (df['relig_cat'] == 'mucho')].dropna(subset=['favor_ive'])
    if len(sub) > 5:
        pct = np.average(sub['favor_ive'], weights=sub[W]) * 100
        print(f"{cat:20}  apoyo_dentro_muy_relig={pct:5.1f}%  (n={len(sub)})")

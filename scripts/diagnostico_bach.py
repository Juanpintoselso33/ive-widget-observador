"""
Diagnóstico Bach incompleto (codes 3,4,5) vs Bach completo (code 6) en el
esquema actual de 5 categorías. Apunta a entender por qué el coeficiente
ajustado de bach_incomp (~0.519) > bach_comp (~0.495) pese a tasas crudas
monotónicas.

Tabula:
- n, tasa cruda de apoyo IVE
- composición de confounders (religiosidad, balotaje, edad, sexo, montevideo)
- tasa de apoyo dentro de cada confounder

También corre una regresión logit solo con dummies de educ (sin controles)
y otra full controlada, para ver si el efecto cambia de signo o sigue siendo
una inversión robusta.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE = Path(__file__).resolve().parent.parent.parent / 'base_limpia.csv'
df = pd.read_csv(BASE)

W = 'w_norm'


def educ_a_cat(n):
    if pd.isna(n):
        return np.nan
    n = int(n)
    if n in (1, 2): return 'primaria'
    if n in (3, 4, 5): return 'bach_incomp'
    if n == 6: return 'bach_comp'
    if n == 7: return 'ter_incomp'
    if n in (8, 9, 10): return 'ter_comp'
    return np.nan


df['educ_cat'] = df['nivel_educativo'].apply(educ_a_cat)

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

CATS = ['primaria', 'bach_incomp', 'bach_comp', 'ter_incomp', 'ter_comp']

print("=" * 80)
print("DESGLOSE BACH INC (codes 3,4,5) — composición interna")
print("=" * 80)
sub_bachinc = df[df['nivel_educativo'].isin([3, 4, 5])]
for code in [3, 4, 5]:
    s = df[df['nivel_educativo'] == code]
    n = len(s)
    n_w = s[W].sum()
    s_v = s.dropna(subset=['favor_ive'])
    apoyo = (np.average(s_v['favor_ive'], weights=s_v[W]) * 100) if len(s_v) else np.nan
    print(f"  code={code}  n_unweighted={n:4}  n_w={n_w:7.1f}  apoyo_ive={apoyo:.1f}%")

print("\n" + "=" * 80)
print("TASAS CRUDAS PONDERADAS POR CATEGORÍA (5-cat)")
print("=" * 80)
for cat in CATS:
    s = df[df['educ_cat'] == cat]
    n = len(s)
    n_w = s[W].sum()
    s_v = s.dropna(subset=['favor_ive'])
    apoyo = np.average(s_v['favor_ive'], weights=s_v[W]) * 100
    print(f"  {cat:15} n={n:4}  n_w={n_w:7.1f}  apoyo_ive={apoyo:.1f}%")

print("\n" + "=" * 80)
print("COMPOSICIÓN DE CONFOUNDERS (% ponderado dentro de cada cat)")
print("=" * 80)
print(f"{'cat':15} {'%muy_rel':>10} {'%nada_rel':>10} {'%lacalle':>10} "
      f"{'%mvd':>8} {'%mujer':>8} {'edad_med':>10}")
for cat in CATS:
    s = df[df['educ_cat'] == cat]
    sr = s.dropna(subset=['relig_cat'])
    sb = s.dropna(subset=['IdBalotaje'])
    sd = s.dropna(subset=['dpto'])
    ss = s.dropna(subset=['sexo'])
    se = s.dropna(subset=['edad'])
    pct_mucho = np.average((sr['relig_cat'] == 'mucho').astype(int), weights=sr[W]) * 100
    pct_nada = np.average((sr['relig_cat'] == 'nada').astype(int), weights=sr[W]) * 100
    pct_lac = np.average((sb['IdBalotaje'] == 2).astype(int), weights=sb[W]) * 100
    pct_mvd = np.average((sd['dpto'] == 19).astype(int), weights=sd[W]) * 100
    pct_muj = np.average((ss['sexo'] == 'F').astype(int), weights=ss[W]) * 100
    edad_m = np.average(se['edad'], weights=se[W])
    print(f"  {cat:15} {pct_mucho:9.1f}% {pct_nada:9.1f}% {pct_lac:9.1f}% "
          f"{pct_mvd:7.1f}% {pct_muj:7.1f}% {edad_m:9.1f}")

print("\n" + "=" * 80)
print("APOYO IVE POR CAT × RELIGIOSIDAD (% ponderado)")
print("=" * 80)
print(f"{'cat':15} {'nada':>12} {'poco':>12} {'bastante':>12} {'mucho':>12}")
for cat in CATS:
    row = f"  {cat:15}"
    for rel in ['nada', 'poco', 'bastante', 'mucho']:
        s = df[(df['educ_cat'] == cat) & (df['relig_cat'] == rel)].dropna(subset=['favor_ive'])
        if len(s) > 5:
            apoyo = np.average(s['favor_ive'], weights=s[W]) * 100
            row += f"  {apoyo:5.1f}% (n={len(s):3})"
        else:
            row += f"  {'.':>12}"
    print(row)

print("\n" + "=" * 80)
print("APOYO IVE POR CAT × BALOTAJE (% ponderado)")
print("=" * 80)
print(f"{'cat':15} {'martinez':>15} {'lacalle':>15} {'otros':>15}")
for cat in CATS:
    row = f"  {cat:15}"
    for bal in [1, 2, None]:
        if bal is None:
            s = df[(df['educ_cat'] == cat) & (df['IdBalotaje'].isna() |
                                              ~df['IdBalotaje'].isin([1, 2]))].dropna(subset=['favor_ive'])
        else:
            s = df[(df['educ_cat'] == cat) & (df['IdBalotaje'] == bal)].dropna(subset=['favor_ive'])
        if len(s) > 5:
            apoyo = np.average(s['favor_ive'], weights=s[W]) * 100
            row += f"  {apoyo:5.1f}% (n={len(s):3})"
        else:
            row += f"  {'.':>15}"
    print(row)

print("\n" + "=" * 80)
print("REGRESIÓN: SOLO DUMMIES DE EDUC (sin controles), Ridge C=0.5, w_norm")
print("=" * 80)
d = df.dropna(subset=['favor_ive', 'educ_cat']).copy()
for cat in ['bach_incomp', 'bach_comp', 'ter_incomp', 'ter_comp']:
    d[f'educ_{cat}'] = (d['educ_cat'] == cat).astype(int)
X = d[[f'educ_{c}' for c in ['bach_incomp', 'bach_comp', 'ter_incomp', 'ter_comp']]].values
y = d['favor_ive'].astype(int).values
w = d[W].values

m = LogisticRegression(C=0.5, penalty='l2', solver='lbfgs', max_iter=2000, random_state=42)
m.fit(X, y, sample_weight=w)
print(f"intercept = {m.intercept_[0]:.4f}")
for cat, c in zip(['bach_incomp', 'bach_comp', 'ter_incomp', 'ter_comp'], m.coef_[0]):
    print(f"  educ_{cat:14} {c:+.4f}")

print("\n" + "=" * 80)
print("REGRESIÓN: SOLO DUMMIES EDUC, SIN PENALIZACIÓN (MLE puro)")
print("=" * 80)
m_mle = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000)
m_mle.fit(X, y, sample_weight=w)
print(f"intercept = {m_mle.intercept_[0]:.4f}")
for cat, c in zip(['bach_incomp', 'bach_comp', 'ter_incomp', 'ter_comp'], m_mle.coef_[0]):
    print(f"  educ_{cat:14} {c:+.4f}")

print("\n" + "=" * 80)
print("DESGLOSE BACH INC POR CODE INDIVIDUAL × RELIGIOSIDAD")
print("=" * 80)
print(f"{'code':6} {'%muy_rel':>10} {'%nada_rel':>10} {'edad_med':>10} {'apoyo':>10}  n")
for code in [3, 4, 5, 6]:
    s = df[df['nivel_educativo'] == code]
    sr = s.dropna(subset=['relig_cat'])
    se = s.dropna(subset=['edad'])
    sv = s.dropna(subset=['favor_ive'])
    pct_m = np.average((sr['relig_cat'] == 'mucho').astype(int), weights=sr[W]) * 100
    pct_n = np.average((sr['relig_cat'] == 'nada').astype(int), weights=sr[W]) * 100
    edad_m = np.average(se['edad'], weights=se[W])
    apoyo = np.average(sv['favor_ive'], weights=sv[W]) * 100 if len(sv) else np.nan
    print(f"  {code:4} {pct_m:9.1f}% {pct_n:9.1f}% {edad_m:9.1f}  {apoyo:8.1f}%  {len(s)}")

"""
Búsqueda exhaustiva de información sobre IdOpcionElecciones2019 en la muestra original.
"""

import pandas as pd

base = pd.read_excel(r'c:\Users\trico\OneDrive\Observador\muestra con weights 8- ENE 2026.xlsx')

print("="*70)
print("BÚSQUEDA EXHAUSTIVA DE IdOpcionElecciones2019")
print("="*70)

print("\n1. TODAS LAS COLUMNAS DEL ARCHIVO:")
print("-"*70)
for i, col in enumerate(base.columns, 1):
    print(f"{i:3}. {col}")

print("\n2. COLUMNAS QUE CONTIENEN 'ELEC' O 'VOTO' O '2019':")
print("-"*70)
relevantes = [c for c in base.columns if 'elec' in c.lower() or 'voto' in c.lower() or '2019' in str(c).lower()]
if relevantes:
    for col in relevantes:
        print(f"  - {col}")
else:
    print("  No se encontraron columnas con esos términos")

print("\n3. INFO SOBRE IdOpcionElecciones2019:")
print("-"*70)
print(f"Tipo de dato: {base['IdOpcionElecciones2019'].dtype}")
print(f"Valores únicos: {sorted(base['IdOpcionElecciones2019'].dropna().unique())}")
print(f"\nFrecuencias:")
print(base['IdOpcionElecciones2019'].value_counts().sort_index())

print("\n4. PRIMEROS 10 REGISTROS (ID, IdOpcionElecciones2019):")
print("-"*70)
print(base[['ID', 'IdOpcionElecciones2019']].head(10))

print("\n5. BUSCANDO COLUMNAS CON 'VAR' (VARIABLES ORIGINALES):")
print("-"*70)
var_cols = [c for c in base.columns if 'var_' in c.lower()]
print(f"Encontradas {len(var_cols)} columnas var_XXX")
if var_cols:
    print("Primeras 5:", var_cols[:5])
    print("Últimas 5:", var_cols[-5:])

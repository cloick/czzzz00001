# Variables booléennes vs target
bool_cols = df_final.select_dtypes(include=['bool']).columns.tolist()
print(f"Variables booléennes: {len(bool_cols)} variables")
print(bool_cols)

# 1. Calculer l'impact de chaque variable booléenne
bool_impact = {}
for col in bool_cols:
    if col in df_final.columns:
        # Affichage textuel détaillé
        print(f"\n=== {col} ===")
        crosstab = pd.crosstab(df_final[col], df_final['dv_close_code'], normalize='index')
        print(crosstab.round(3))
        
        # Taux de succès par groupe
        success_rate = df_final.groupby(col)['success'].mean()
        print(f"Taux de succès: {success_rate.round(3)}")
        
        # Calculer l'impact (différence de taux de succès)
        if len(success_rate) == 2:
            impact = abs(success_rate[False] - success_rate[True])
            bool_impact[col] = impact
            print(f"Impact (différence): {impact:.3f}")

# 2. Trier par impact décroissant
bool_cols_sorted = sorted(bool_impact.keys(), key=lambda x: bool_impact[x], reverse=True)

print(f"\n=== CLASSEMENT PAR IMPACT ===")
for i, col in enumerate(bool_cols_sorted, 1):
    print(f"{i:2d}. {col}: {bool_impact[col]:.3f}")

# 3. Visualisation de TOUTES les variables booléennes
n_vars = len(bool_cols)
n_cols = 4  # 4 colonnes par ligne
n_rows = (n_vars + n_cols - 1) // n_cols  # Calcul du nombre de lignes nécessaires

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))

# S'assurer que axes est un array 2D
if n_rows == 1:
    axes = axes.reshape(1, -1)
elif n_cols == 1:
    axes = axes.reshape(-1, 1)

# Plotting pour chaque variable (triées par impact)
for i, col in enumerate(bool_cols_sorted):
    row = i // n_cols
    col_idx = i % n_cols
    
    # Taux de succès par catégorie
    success_by_cat = df_final.groupby(col)['success'].mean()
    success_by_cat.plot(kind='bar', ax=axes[row, col_idx], 
                       title=f'{col}\n(Impact: {bool_impact[col]:.3f})')
    axes[row, col_idx].set_ylabel('Taux de succès')
    axes[row, col_idx].set_xlabel('')
    axes[row, col_idx].tick_params(axis='x', rotation=45)

# Cacher les axes vides s'il y en a
for i in range(n_vars, n_rows * n_cols):
    row = i // n_cols
    col_idx = i % n_cols
    axes[row, col_idx].set_visible(False)

plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway

print("=== PHASE 1 : ANALYSE CORRÉLATIONS RAPIDE ===\n")

# 1. CORRÉLATIONS VARIABLES NUMÉRIQUES
print("1. CORRÉLATIONS VARIABLES NUMÉRIQUES")
print("=" * 50)

numerical_vars = ['reassignment_count', 'sys_mod_count', 'u_cab_count', 
                 'u_cab_reservation_count', 'u_reminder_count']

# Vérifier que les colonnes existent
numerical_vars_available = [col for col in numerical_vars if col in df_final.columns]
print(f"Variables numériques disponibles: {numerical_vars_available}")

if len(numerical_vars_available) > 1:
    corr_matrix = df_final[numerical_vars_available].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f')
    plt.title('Corrélations Variables Numériques')
    plt.tight_layout()
    plt.show()
    
    # Identifier corrélations fortes (>0.7)
    high_corr = np.where((np.abs(corr_matrix) > 0.7) & (corr_matrix != 1.0))
    if len(high_corr[0]) > 0:
        print("\n⚠️ CORRÉLATIONS FORTES DÉTECTÉES:")
        for i, j in zip(high_corr[0], high_corr[1]):
            if i < j:  # Éviter doublons
                print(f"  {corr_matrix.index[i]} ↔ {corr_matrix.columns[j]}: {corr_matrix.iloc[i,j]:.3f}")
    else:
        print("\n✅ Pas de corrélations fortes entre variables numériques")

# 2. CHI-2 TESTS POUR VARIABLES CATÉGORIELLES IMPORTANTES
print("\n\n2. TESTS CHI-2 VARIABLES CATÉGORIELLES vs TARGET")
print("=" * 60)

key_categorical_vars = ['dv_risk', 'dv_type', 'dv_impact', 'dv_category', 
                       'dv_u_type_change_silca', 'dv_conflict_status',
                       'dv_u_change_prerequisites', 'dv_u_qualification']

results_chi2 = []

for var in key_categorical_vars:
    if var in df_final.columns and df_final[var].nunique() > 1:
        try:
            # Créer tableau de contingence
            contingency = pd.crosstab(df_final[var], df_final['dv_close_code'])
            
            # Test Chi-2
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            
            # Cramér's V (mesure d'association)
            n = contingency.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
            
            results_chi2.append({
                'Variable': var,
                'Chi2': chi2,
                'p_value': p_value,
                'Cramers_V': cramers_v,
                'Significant': p_value < 0.05
            })
            
            print(f"{var:30s} | Chi2: {chi2:8.2f} | p-value: {p_value:.2e} | Cramér's V: {cramers_v:.3f}")
            
        except Exception as e:
            print(f"{var:30s} | ERREUR: {str(e)}")

# Trier par Cramér's V (force d'association)
results_chi2 = sorted(results_chi2, key=lambda x: x['Cramers_V'], reverse=True)

print(f"\n🏆 TOP VARIABLES PAR FORCE D'ASSOCIATION:")
for i, result in enumerate(results_chi2[:5], 1):
    print(f"{i}. {result['Variable']:25s} | Cramér's V: {result['Cramers_V']:.3f}")

# 3. CORRÉLATIONS VARIABLES BOOLÉENNES IMPORTANTES  
print("\n\n3. CORRÉLATIONS VARIABLES BOOLÉENNES")
print("=" * 50)

bool_vars_important = ['u_psi_update_necessary', 'u_bcr', 'u_bpc', 'u_clp', 
                      'u_cfs', 'u_coordinator_trigger', 'u_emergency']

bool_vars_available = [col for col in bool_vars_important if col in df_final.columns]

if len(bool_vars_available) > 1:
    # Convertir en numérique pour corrélation
    bool_df = df_final[bool_vars_available].astype(int)
    bool_corr = bool_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(bool_corr, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f')
    plt.title('Corrélations Variables Booléennes')
    plt.tight_layout()
    plt.show()
    
    # Identifier corrélations fortes
    high_corr_bool = np.where((np.abs(bool_corr) > 0.5) & (bool_corr != 1.0))
    if len(high_corr_bool[0]) > 0:
        print("\n⚠️ CORRÉLATIONS FORTES VARIABLES BOOLÉENNES:")
        for i, j in zip(high_corr_bool[0], high_corr_bool[1]):
            if i < j:
                print(f"  {bool_corr.index[i]} ↔ {bool_corr.columns[j]}: {bool_corr.iloc[i,j]:.3f}")
    else:
        print("\n✅ Pas de corrélations fortes entre variables booléennes")

# 4. ANALYSE SPÉCIFIQUE : u_bcr, u_bpc, u_clp
print("\n\n4. ANALYSE SPÉCIFIQUE BCR/BPC/CLP")
print("=" * 45)

bcr_vars = ['u_bcr', 'u_bpc', 'u_clp']
bcr_available = [col for col in bcr_vars if col in df_final.columns]

if len(bcr_available) >= 2:
    # Tableau de contingence entre ces variables
    for i in range(len(bcr_available)):
        for j in range(i+1, len(bcr_available)):
            var1, var2 = bcr_available[i], bcr_available[j]
            crosstab = pd.crosstab(df_final[var1], df_final[var2], margins=True)
            print(f"\nTableau croisé {var1} vs {var2}:")
            print(crosstab)
            
            # Pourcentage de recouvrement
            both_true = ((df_final[var1] == True) & (df_final[var2] == True)).sum()
            total_either = ((df_final[var1] == True) | (df_final[var2] == True)).sum()
            if total_either > 0:
                overlap = both_true / total_either * 100
                print(f"Recouvrement: {overlap:.1f}%")

# 5. RÉSUMÉ ET RECOMMANDATIONS
print("\n\n5. RÉSUMÉ ET RECOMMANDATIONS")
print("=" * 40)

print("📊 VARIABLES À GARDER PRIORITAIREMENT:")
if results_chi2:
    for result in results_chi2[:3]:
        if result['Cramers_V'] > 0.1:
            print(f"  ✅ {result['Variable']} (Cramér's V: {result['Cramers_V']:.3f})")

print("\n⚠️ VARIABLES POTENTIELLEMENT REDONDANTES:")
print("  (À vérifier selon corrélations detectées ci-dessus)")

print("\n🎯 PRÊT POUR LA MODÉLISATION !")

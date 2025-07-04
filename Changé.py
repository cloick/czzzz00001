import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 1. CORRIGER LES TYPES (dates en datetime)
print("=== CORRECTION DES TYPES ===")
# Colonnes dates à convertir
date_columns = ['approval_set', 'end_date', 'opened_at', 'start_date']
for col in date_columns:
    if col in df_final.columns:
        df_final[col] = pd.to_datetime(df_final[col], errors='coerce')
        print(f"✅ {col} converti en datetime")

# 2. VÉRIFICATION DES TYPES
print("\n=== TYPES FINAUX ===")
print(df_final.dtypes.value_counts())🎯 ANALYSES PAR TYPE DE VARIABLE :A. VARIABLES NUMÉRIQUES (int64) :# Variables numériques vs target
numerical_cols = df_final.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"Variables numériques: {numerical_cols}")

fig, axes = plt.subplots(len(numerical_cols), 2, figsize=(15, 5*len(numerical_cols)))

for i, col in enumerate(numerical_cols):
    # Distribution
    sns.histplot(df_final[col], ax=axes[i,0])
    axes[i,0].set_title(f'Distribution {col}')
    
    # Boxplot par close_code
    sns.boxplot(data=df_final, x='dv_close_code', y=col, ax=axes[i,1])
    axes[i,1].set_title(f'{col} vs Close Code')
    axes[i,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Corrélations avec target
for col in numerical_cols:
    df_final['success'] = (df_final['dv_close_code'] == 'Succès').astype(int)
    corr = df_final[col].corr(df_final['success'])
    print(f"Corrélation {col} avec succès: {corr:.3f}")


B. VARIABLES BOOLÉENNES :# Variables booléennes vs target


bool_cols = df_final.select_dtypes(include=['bool']).columns.tolist()
print(f"Variables booléennes: {len(bool_cols)} variables")
print(bool_cols)

# Analyse croisée avec target
for col in bool_cols:
    print(f"\n=== {col} ===")
    crosstab = pd.crosstab(df_final[col], df_final['dv_close_code'], normalize='index')
    print(crosstab.round(3))
    
    # Taux de succès par groupe
    success_rate = df_final.groupby(col)['success'].mean()
    print(f"Taux de succès: {success_rate.round(3)}")

# Visualisation des plus importantes (top 6)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(bool_cols[:6]):  # Top 6 seulement
    crosstab = pd.crosstab(df_final[col], df_final['dv_close_code'])
    crosstab.plot(kind='bar', ax=axes[i], title=f'{col} vs Close Code')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()C. VARIABLES CATÉGORIELLES (object) :# Variables catégorielles (hors dates)
categorical_cols = df_final.select_dtypes(include=['object']).columns.tolist()
# Exclure la target et les dates
categorical_cols = [col for col in categorical_cols if col != 'dv_close_code' 
                   and not any(date_word in col.lower() for date_word in ['date', '_at'])]

print(f"Variables catégorielles: {len(categorical_cols)} variables")

# Analyser chaque variable catégorielle
for col in categorical_cols[:5]:  # Premières 5 pour commencer
    print(f"\n=== {col} ===")
    print(f"Nombre de catégories: {df_final[col].nunique()}")
    print(f"Top 5 valeurs:")
    print(df_final[col].value_counts().head())
    
    # Crosstab si pas trop de catégories
    if df_final[col].nunique() <= 10:
        crosstab = pd.crosstab(df_final[col], df_final['dv_close_code'], normalize='index')
        print("Taux par catégorie:")
        print(crosstab.round(3))

# Visualisation des variables catégorielles importantes
variables_importantes = ['dv_type', 'dv_assignment_group', 'dv_conflict_status', 
                        'dv_impact', 'dv_risk']

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for i, col in enumerate(variables_importantes[:5]):
    if col in df_final.columns:
        # Taux de succès par catégorie
        success_by_cat = df_final.groupby(col)['success'].mean().sort_values()
        success_by_cat.plot(kind='bar', ax=axes[i], title=f'Taux succès par {col}')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_ylabel('Taux de succès')

plt.tight_layout()
plt.show()

D. ANALYSE TEMPORELLE :# Variables temporelles
date_columns = ['approval_set', 'end_date', 'opened_at', 'start_date']
date_cols_available = [col for col in date_columns if col in df_final.columns]

print(f"Variables temporelles disponibles: {date_cols_available}")

# 1. Créer des features temporelles
for col in date_cols_available:
    if df_final[col].notna().sum() > 100:  # Si assez de données
        df_final[f'{col}_year'] = df_final[col].dt.year
        df_final[f'{col}_month'] = df_final[col].dt.month
        df_final[f'{col}_day_of_week'] = df_final[col].dt.dayofweek  # 0=lundi
        df_final[f'{col}_hour'] = df_final[col].dt.hour
        df_final[f'{col}_is_weekend'] = df_final[col].dt.dayofweek >= 5

# 2. Analyser les patterns temporels avec opened_at (le plus important)
if 'opened_at' in df_final.columns:
    print("\n=== ANALYSE TEMPORELLE SUR opened_at ===")
    
    # Taux de succès par jour de la semaine
    days = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
    success_by_day = df_final.groupby('opened_at_day_of_week')['success'].mean()
    
    plt.figure(figsize=(15, 10))
    
    # Graphique 1: Taux de succès par jour de semaine
    plt.subplot(2, 3, 1)
    success_by_day.plot(kind='bar')
    plt.title('Taux de succès par jour de semaine')
    plt.xticks(range(7), days, rotation=45)
    plt.ylabel('Taux de succès')
    
    # Graphique 2: Taux de succès par heure
    plt.subplot(2, 3, 2)
    success_by_hour = df_final.groupby('opened_at_hour')['success'].mean()
    success_by_hour.plot(kind='line', marker='o')
    plt.title('Taux de succès par heure')
    plt.xlabel('Heure')
    plt.ylabel('Taux de succès')
    
    # Graphique 3: Taux de succès par mois
    plt.subplot(2, 3, 3)
    success_by_month = df_final.groupby('opened_at_month')['success'].mean()
    success_by_month.plot(kind='bar')
    plt.title('Taux de succès par mois')
    plt.xlabel('Mois')
    plt.ylabel('Taux de succès')
    
    # Graphique 4: Weekend vs Semaine
    plt.subplot(2, 3, 4)
    success_by_weekend = df_final.groupby('opened_at_is_weekend')['success'].mean()
    success_by_weekend.plot(kind='bar')
    plt.title('Taux de succès: Semaine vs Weekend')
    plt.xticks([0, 1], ['Semaine', 'Weekend'], rotation=0)
    plt.ylabel('Taux de succès')
    
    # Graphique 5: Volume de changements par jour de semaine
    plt.subplot(2, 3, 5)
    volume_by_day = df_final['opened_at_day_of_week'].value_counts().sort_index()
    volume_by_day.plot(kind='bar')
    plt.title('Volume de changements par jour')
    plt.xticks(range(7), days, rotation=45)
    plt.ylabel('Nombre de changements')
    
    # Graphique 6: Heatmap Jour x Heure
    plt.subplot(2, 3, 6)
    pivot_data = df_final.pivot_table(values='success', 
                                     index='opened_at_day_of_week', 
                                     columns='opened_at_hour', 
                                     aggfunc='mean')
    sns.heatmap(pivot_data, cmap='RdYlGn', vmin=0.8, vmax=1.0, 
                yticklabels=days, cbar_kws={'label': 'Taux de succès'})
    plt.title('Taux de succès par Jour x Heure')
    
    plt.tight_layout()
    plt.show()

# 3. Analyser les durées (si on a start et end)
if 'start_date' in df_final.columns and 'end_date' in df_final.columns:
    df_final['duree_planifiee'] = (df_final['end_date'] - df_final['start_date']).dt.total_seconds() / 3600  # en heures
    
    print("\n=== ANALYSE DES DURÉES ===")
    print(f"Durée planifiée moyenne: {df_final['duree_planifiee'].mean():.2f} heures")
    
    # Durée vs succès
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_final, x='dv_close_code', y='duree_planifiee')
    plt.title('Durée planifiée vs Résultat')
    plt.xticks(rotation=45)
    plt.show()

## 🎯 **PARFAIT ! Je vais reprendre à partir de la standardisation et de la suite !**

**Puisque vous avez déjà :**
- ✅ Exploration des données
- ✅ Fonction `feature_engineering()` (features temporelles)
- ✅ Fonction `encodage()` (variables catégorielles avec regroupement)
- ✅ Groupes de features (baseline_features, bool_correlated_group, etc.)

**Je démarre directement à partir de la standardisation et la suite du workflow !**

---

## ⚙️ **SECTION 2.4 - NORMALISATION ET STANDARDISATION**

```python
# =================================================================
# 2.4 NORMALISATION ET STANDARDISATION
# =================================================================

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def prepare_features_for_clustering(df, feature_groups=None, scaling_method='standard'):
    """
    Préparer les features finales pour clustering
    
    Args:
        df: DataFrame avec features engineered et encodées
        feature_groups: Dict des groupes de features à inclure
        scaling_method: 'standard', 'minmax', ou 'robust'
    """
    
    print("🔧 Préparation des features pour clustering...")
    
    # Copie de travail
    df_clustering = df.copy()
    
    # 1. Sélection des features selon les groupes choisis
    if feature_groups is None:
        # Configuration par défaut - à ajuster selon vos tests
        feature_groups = {
            'baseline_features': True,
            'bool_correlated_group': True,
            'bool_star': False,  # À tester
            'bool_independant': True,
            'bool_low_impact': False,  # À tester
            'categorical_group': True,
            'categorical_others': False  # Éviter le bruit
        }
    
    selected_features = []
    
    # Ajouter les groupes sélectionnés
    for group_name, include in feature_groups.items():
        if include and group_name in df_clustering.columns:
            print(f"✅ Inclusion du groupe: {group_name}")
            # Si c'est une colonne de type list/array, l'étendre
            if df_clustering[group_name].dtype == 'object':
                # Assuming it's a list of feature names
                group_features = df_clustering[group_name].iloc[0] if isinstance(df_clustering[group_name].iloc[0], list) else [group_name]
                selected_features.extend(group_features)
            else:
                selected_features.append(group_name)
    
    # 2. Vérifier que les features existent
    available_features = [f for f in selected_features if f in df_clustering.columns]
    missing_features = [f for f in selected_features if f not in df_clustering.columns]
    
    print(f"📊 Features disponibles: {len(available_features)}")
    print(f"❌ Features manquantes: {len(missing_features)}")
    if missing_features:
        print(f"   Manquantes: {missing_features[:5]}...")
    
    # 3. Créer la matrice de features
    if not available_features:
        raise ValueError("Aucune feature disponible pour le clustering !")
    
    X = df_clustering[available_features].copy()
    
    # 4. Gestion des valeurs manquantes
    print(f"📊 Valeurs manquantes avant imputation: {X.isnull().sum().sum()}")
    
    # Imputation simple
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    # Numériques: médiane
    for col in numeric_cols:
        if X[col].isnull().any():
            X[col].fillna(X[col].median(), inplace=True)
    
    # Catégorielles: mode
    for col in categorical_cols:
        if X[col].isnull().any():
            X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown', inplace=True)
    
    print(f"📊 Valeurs manquantes après imputation: {X.isnull().sum().sum()}")
    
    # 5. Normalisation/Standardisation
    print(f"🔧 Application de la standardisation: {scaling_method}")
    
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    elif scaling_method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("scaling_method doit être 'standard', 'minmax' ou 'robust'")
    
    # Appliquer seulement sur les colonnes numériques
    X_scaled = X.copy()
    if len(numeric_cols) > 0:
        X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    # 6. Statistiques finales
    print(f"\n📊 MATRICE FINALE POUR CLUSTERING:")
    print(f"   Shape: {X_scaled.shape}")
    print(f"   Features numériques: {len(numeric_cols)}")
    print(f"   Features catégorielles: {len(categorical_cols)}")
    print(f"   Dtypes: {X_scaled.dtypes.value_counts().to_dict()}")
    
    return X_scaled, scaler, available_features

# Test de la fonction
print("🧪 Test de préparation des features...")

# Appliquer feature engineering et encodage si pas déjà fait
if 'opened_hour' not in df.columns:
    df_fe = feature_engineering(df)
else:
    df_fe = df.copy()

if 'dv_assignment_group' in df_fe.select_dtypes(include=['object']).columns:
    df_encoded = encodage(df_fe)
else:
    df_encoded = df_fe.copy()

# Préparer pour clustering
X_clustering, scaler_clustering, feature_names = prepare_features_for_clustering(
    df_encoded, 
    scaling_method='standard'
)

print(f"✅ Matrice prête: {X_clustering.shape}")
print(f"✅ Features utilisées: {len(feature_names)}")
```

---

## 🔬 **SECTION 3 - ANALYSE DE LA STRUCTURE NATURELLE**

```python
# =================================================================
# 3. ANALYSE DE LA STRUCTURE NATURELLE DES DONNÉES
# =================================================================

def analyze_data_structure(X, feature_names, sample_size=2000):
    """
    Analyser la structure naturelle des données pour guider le choix d'algorithme
    
    Args:
        X: Matrice de features standardisées
        feature_names: Noms des features
        sample_size: Taille échantillon pour t-SNE (performance)
    """
    
    print("🔬 Analyse de la structure naturelle des données...")
    
    # 1. ANALYSE PCA - Variance expliquée
    print("\n📊 1. ANALYSE PCA")
    pca_full = PCA()
    X_pca_full = pca_full.fit_transform(X)
    
    # Variance expliquée cumulative
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Nombre de composantes pour 80%, 90%, 95% de variance
    n_comp_80 = np.argmax(cumvar >= 0.80) + 1
    n_comp_90 = np.argmax(cumvar >= 0.90) + 1
    n_comp_95 = np.argmax(cumvar >= 0.95) + 1
    
    print(f"   📈 {n_comp_80} composantes pour 80% variance")
    print(f"   📈 {n_comp_90} composantes pour 90% variance")  
    print(f"   📈 {n_comp_95} composantes pour 95% variance")
    
    # 2. VISUALISATION PCA 2D/3D
    print("\n🎨 2. VISUALISATION PCA")
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X)
    
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X)
    
    # Plot PCA 2D
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # PCA 2D scatter
    axes[0,0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], alpha=0.6, s=20)
    axes[0,0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%})')
    axes[0,0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%})')
    axes[0,0].set_title('🎯 Distribution PCA 2D')
    axes[0,0].grid(True, alpha=0.3)
    
    # Variance expliquée
    axes[0,1].plot(range(1, min(21, len(cumvar)+1)), cumvar[:20], 'bo-')
    axes[0,1].axhline(y=0.8, color='r', linestyle='--', label='80%')
    axes[0,1].axhline(y=0.9, color='orange', linestyle='--', label='90%')
    axes[0,1].axhline(y=0.95, color='g', linestyle='--', label='95%')
    axes[0,1].set_xlabel('Nombre de composantes')
    axes[0,1].set_ylabel('Variance expliquée cumulative')
    axes[0,1].set_title('📈 Variance expliquée PCA')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. t-SNE pour structures non-linéaires
    print("\n🌀 3. ANALYSE t-SNE (échantillon)")
    
    # Échantillonner pour performance
    if len(X) > sample_size:
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[sample_idx]
    else:
        X_sample = X
        sample_idx = range(len(X))
    
    # t-SNE avec différentes perplexités
    for i, perplexity in enumerate([10, 30]):
        print(f"   🔄 t-SNE perplexity={perplexity}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_tsne = tsne.fit_transform(X_sample)
        
        axes[1,i].scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, s=20)
        axes[1,i].set_title(f'🌀 t-SNE (perplexity={perplexity})')
        axes[1,i].set_xlabel('t-SNE 1')
        axes[1,i].set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.show()
    
    # 4. ANALYSE DES DISTANCES pour détecter la forme des clusters
    print("\n📏 4. ANALYSE DES DISTANCES")
    
    # Échantillon pour analyse de distances
    sample_size_dist = min(1000, len(X))
    if len(X) > sample_size_dist:
        dist_idx = np.random.choice(len(X), sample_size_dist, replace=False)
        X_dist = X[dist_idx]
    else:
        X_dist = X
    
    # Calculer matrice de distances
    from sklearn.metrics.pairwise import euclidean_distances
    dist_matrix = euclidean_distances(X_dist)
    
    # Distribution des distances
    distances = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(distances, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Distance euclidienne')
    plt.ylabel('Fréquence')
    plt.title('📊 Distribution des distances')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(distances)
    plt.ylabel('Distance euclidienne')
    plt.title('📦 Box plot des distances')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 5. RECOMMANDATIONS ALGORITHMIQUES
    print("\n💡 5. RECOMMANDATIONS ALGORITHMIQUES")
    
    distance_std = np.std(distances)
    distance_mean = np.mean(distances)
    cv_distance = distance_std / distance_mean
    
    print(f"   📊 Distance moyenne: {distance_mean:.3f}")
    print(f"   📊 Écart-type distances: {distance_std:.3f}")
    print(f"   📊 Coefficient de variation: {cv_distance:.3f}")
    
    # Heuristiques de recommandation
    recommendations = []
    
    if cv_distance < 0.3:
        recommendations.append("✅ Données homogènes → K-Means recommandé")
    elif cv_distance > 0.7:
        recommendations.append("⚠️ Données hétérogènes → DBSCAN ou GMM recommandés")
    else:
        recommendations.append("📊 Données moyennement variables → Tester K-Means et GMM")
    
    # Analyse de la variance PCA
    if pca_2d.explained_variance_ratio_[0] > 0.6:
        recommendations.append("📈 Forte variance PC1 → Structure linéaire possible")
    
    if sum(pca_2d.explained_variance_ratio_) < 0.3:
        recommendations.append("🔄 Faible variance 2D → Données très complexes, considérer réduction de dimension")
    
    print("\n🎯 RECOMMANDATIONS:")
    for rec in recommendations:
        print(f"   {rec}")
    
    return {
        'pca_2d': X_pca_2d,
        'pca_3d': X_pca_3d,
        'pca_variance_ratio': pca_2d.explained_variance_ratio_,
        'n_components': {'80%': n_comp_80, '90%': n_comp_90, '95%': n_comp_95},
        'distance_stats': {'mean': distance_mean, 'std': distance_std, 'cv': cv_distance},
        'recommendations': recommendations,
        'sample_indices': sample_idx if len(X) > sample_size else None
    }

# Lancer l'analyse
print("🚀 Lancement de l'analyse de structure...")
structure_analysis = analyze_data_structure(X_clustering, feature_names)
```

**Voulez-vous que je continue avec la section suivante (détermination du nombre optimal de clusters) ?** 🎯

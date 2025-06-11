# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import hdbscan
from sklearn.metrics import silhouette_score
import warnings

# Supprimer TOUS les warnings
warnings.filterwarnings("ignore")

print("🚀 CLUSTERING AVEC OPTIMISATION INTENSIVE - 100 ÉVALUATIONS")
print("🎯 Recherche exhaustive pour le MEILLEUR résultat possible")
print("="*80)

# Read recipe inputs
dataset = dataiku.Dataset("incident_prepared_embeddings_creation")
df = dataset.get_dataframe()

print(f"Dataset chargé : {len(df)} lignes, {len(df.columns)} colonnes")

# Récupérer les colonnes d'embeddings
embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
print(f"Colonnes d'embeddings trouvées : {len(embedding_cols)}")

embeddings = df[embedding_cols].values
print(f"Shape des embeddings : {embeddings.shape}")

# Variables catégorielles avec encodage robuste
cat_vars = ['Priorité', 'Service métier', 'Cat1', 'Cat2', 'Groupe affecté']
cat_features = []

print("\n🔧 Encodage des variables catégorielles...")
for col in cat_vars:
    if col in df.columns:
        unique_vals = df[col].fillna('INCONNU').unique()
        mapping = {val: i for i, val in enumerate(unique_vals)}
        encoded = df[col].fillna('INCONNU').map(mapping)
        cat_features.append(encoded.values)
        print(f"  ✅ {col}: {len(unique_vals)} valeurs uniques")

# Combiner features avec pondération optimisée
if cat_features:
    cat_features_array = np.array(cat_features).T
    scaler = StandardScaler()
    cat_features_scaled = scaler.fit_transform(cat_features_array)
    
    # Pondération renforcée
    weights = np.array([0.8, 3.0, 1.5, 1.5, 4.0])[:len(cat_features)]  # Plus de poids
    cat_features_weighted = cat_features_scaled * weights
    features_combined = np.hstack([embeddings, cat_features_weighted])
    print(f"✅ Features combinées avec pondération renforcée : {features_combined.shape}")
else:
    features_combined = embeddings

# UMAP optimisé et robuste
print("\n🗺️  Réduction dimensionnelle UMAP optimisée...")

try:
    from umap import UMAP
    
    # Configuration haute performance
    import numba
    numba.config.THREADING_LAYER = 'workqueue'
    
    reducer_2d = UMAP(
        n_components=2,
        n_neighbors=min(50, len(features_combined) - 1),  # Plus de voisins
        min_dist=0.05,      # Plus précis
        metric='cosine',
        random_state=42,
        init='spectral',    # Meilleure initialisation
        verbose=False,
        low_memory=False,   # Utiliser toute la mémoire disponible
        n_epochs=500        # Plus d'époques pour convergence
    )
    
    print("🔄 UMAP 2D haute qualité en cours...")
    embeddings_2d = reducer_2d.fit_transform(features_combined)
    
    reducer_3d = UMAP(
        n_components=3,
        n_neighbors=min(50, len(features_combined) - 1),
        min_dist=0.05,
        metric='cosine',
        random_state=42,
        init='spectral',
        verbose=False,
        low_memory=False,
        n_epochs=500
    )
    
    print("🔄 UMAP 3D haute qualité en cours...")
    embeddings_3d = reducer_3d.fit_transform(features_combined)
    
    print(f"✅ UMAP 2D haute qualité : {embeddings_2d.shape}")
    print(f"✅ UMAP 3D haute qualité : {embeddings_3d.shape}")
    
except Exception as e:
    print(f"⚠️  UMAP échoué, fallback PCA optimisé : {e}")
    from sklearn.decomposition import PCA
    
    # PCA avec plus de composantes intermédiaires
    pca_2d = PCA(n_components=2, random_state=42, whiten=True)
    embeddings_2d = pca_2d.fit_transform(features_combined)
    
    pca_3d = PCA(n_components=3, random_state=42, whiten=True)
    embeddings_3d = pca_3d.fit_transform(features_combined)

# OPTIMISATION INTENSIVE - 100 ÉVALUATIONS
print(f"\n🔬 OPTIMISATION BAYÉSIENNE INTENSIVE")
print("="*70)

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt import gp_minimize
    
    # ESPACE DE RECHERCHE ÉLARGI
    space = [
        Integer(30, 1000, name='min_cluster_size'),        # Très large gamme
        Integer(3, 200, name='min_samples'),               # Très large gamme  
        Real(0.05, 4.0, name='cluster_selection_epsilon'), # Gamme étendue
        Categorical(['euclidean', 'manhattan', 'cosine'], name='metric'),  # Test différentes métriques
        Categorical(['eom', 'leaf'], name='cluster_selection_method')      # Méthodes de sélection
    ]
    
    # Objectifs multiples
    target_ranges = [
        {'min': 10, 'max': 25, 'weight': 1.0, 'name': '10-25 clusters'},
        {'min': 25, 'max': 50, 'weight': 0.8, 'name': '25-50 clusters'},  
        {'min': 50, 'max': 80, 'weight': 0.6, 'name': '50-80 clusters'}
    ]
    
    print(f"🎯 ESPACE DE RECHERCHE ÉLARGI :")
    print(f"   min_cluster_size : [30, 1000]")
    print(f"   min_samples : [3, 200]")
    print(f"   cluster_selection_epsilon : [0.05, 4.0]")
    print(f"   metric : ['euclidean', 'manhattan', 'cosine']")
    print(f"   cluster_selection_method : ['eom', 'leaf']")
    print(f"🎯 OBJECTIFS MULTIPLES testés en parallèle")
    
    # Variables pour tracker les meilleurs résultats
    best_results = []
    evaluation_count = 0
    
    @use_named_args(space)
    def objective_intensive(min_cluster_size, min_samples, cluster_selection_epsilon, metric, cluster_selection_method):
        global evaluation_count, best_results
        evaluation_count += 1
        
        # Validation des paramètres
        if min_samples > min_cluster_size:
            return 100.0
        
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=int(min_cluster_size),
                min_samples=int(min_samples),
                cluster_selection_epsilon=float(cluster_selection_epsilon),
                metric=metric,
                cluster_selection_method=cluster_selection_method
            )
            
            labels = clusterer.fit_predict(embeddings_2d)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = (labels == -1).sum() / len(labels)
            
            # Calculer silhouette
            silhouette = -2.0  # Valeur par défaut très basse
            if n_clusters > 1:
                mask = labels != -1
                if mask.sum() > n_clusters and mask.sum() < 20000:  # Limite performance
                    try:
                        silhouette = silhouette_score(embeddings_2d[mask], labels[mask])
                    except:
                        silhouette = -2.0
            
            # Fonction objectif sophistiquée multi-objectifs
            best_objective = 200.0
            best_target = None
            
            for target in target_ranges:
                if target['min'] <= n_clusters <= target['max']:
                    # Dans une plage cible
                    cluster_penalty = 0  # Pas de pénalité
                    
                    # Bonus pour silhouette positive
                    silhouette_bonus = max(0, silhouette * 5.0)
                    
                    # Pénalité pour bruit excessif
                    noise_penalty = max(0, (noise_ratio - 0.1) * 10.0)
                    
                    # Score composite
                    objective_value = (
                        cluster_penalty + noise_penalty - silhouette_bonus
                    ) / target['weight']  # Pondération par préférence
                    
                    if objective_value < best_objective:
                        best_objective = objective_value
                        best_target = target['name']
                else:
                    # Hors plage - pénalité basée sur distance
                    distance = min(abs(n_clusters - target['min']), abs(n_clusters - target['max']))
                    objective_value = 50.0 + distance * 2.0
                    
                    if objective_value < best_objective:
                        best_objective = objective_value
                        best_target = target['name']
            
            # Tracker les bons résultats
            if silhouette > 0.1 and noise_ratio < 0.3:
                result_info = {
                    'eval': evaluation_count,
                    'params': f"MCS={min_cluster_size}, MS={min_samples}, ε={cluster_selection_epsilon:.2f}, {metric}, {cluster_selection_method}",
                    'clusters': n_clusters,
                    'noise': noise_ratio,
                    'silhouette': silhouette,
                    'objective': best_objective,
                    'target': best_target
                }
                best_results.append(result_info)
                
                # Garder seulement les 10 meilleurs
                best_results.sort(key=lambda x: x['objective'])
                best_results = best_results[:10]
            
            # Logs détaillés pour les bons résultats
            if n_clusters <= 100 and (silhouette > 0.0 or evaluation_count % 10 == 0):
                print(f"[{evaluation_count:3d}/100] MCS={min_cluster_size:3d}, MS={min_samples:2d}, ε={cluster_selection_epsilon:.2f}, "
                      f"{metric[:4]}, {cluster_selection_method}: {n_clusters:2d} clusters, {noise_ratio:.1%} bruit, "
                      f"sil={silhouette:.3f}, obj={best_objective:.2f} ({best_target})")
            elif evaluation_count % 25 == 0:
                print(f"[{evaluation_count:3d}/100] Progress checkpoint - {len(best_results)} good results found so far...")
            
            return best_objective
            
        except Exception as e:
            return 150.0
    
    print(f"\n🚀 DÉMARRAGE OPTIMISATION INTENSIVE (100 évaluations)")
    print(f"⏱️  Temps estimé : 45-60 minutes")
    print(f"🎯 Recherche du meilleur compromis clusters/silhouette/bruit")
    print("-" * 70)
    
    # Optimisation avec acquisition function améliorée
    result = gp_minimize(
        objective_intensive,
        space,
        n_calls=100,                    # 100 ÉVALUATIONS !
        n_initial_points=15,            # Plus de points initiaux aléatoires
        acq_func='EI',                  # Expected Improvement
        acq_optimizer='sampling',       # Échantillonnage pour exploration
        random_state=42,
        verbose=False,
        n_jobs=1                        # Single thread pour stabilité
    )
    
    # Récupérer les meilleurs paramètres
    best_mcs, best_ms, best_eps, best_metric, best_method = result.x
    
    print(f"\n" + "="*70)
    print(f"🏆 OPTIMISATION TERMINÉE - {evaluation_count} évaluations effectuées")
    print(f"📊 {len(best_results)} résultats de qualité identifiés")
    print(f"\n🥇 MEILLEURS PARAMÈTRES GLOBAUX :")
    print(f"  min_cluster_size : {best_mcs}")
    print(f"  min_samples : {best_ms}")
    print(f"  cluster_selection_epsilon : {best_eps:.3f}")
    print(f"  metric : {best_metric}")
    print(f"  cluster_selection_method : {best_method}")
    print(f"  Score objectif : {result.fun:.4f}")
    
    # Afficher top 5 des meilleurs résultats
    if best_results:
        print(f"\n🏅 TOP 5 DES MEILLEURS RÉSULTATS :")
        for i, res in enumerate(best_results[:5], 1):
            print(f"  {i}. {res['clusters']:2d} clusters, sil={res['silhouette']:.3f}, "
                  f"bruit={res['noise']:.1%} → {res['target']}")
    
    # Utiliser les meilleurs paramètres
    final_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=best_mcs,
        min_samples=best_ms,
        cluster_selection_epsilon=best_eps,
        metric=best_metric,
        cluster_selection_method=best_method
    )
    
    optimization_used = "Optimisation intensive (100 évaluations)"
    
except Exception as e:
    print(f"❌ Optimisation échouée : {e}")
    print("🔄 Fallback vers paramètres haute qualité")
    
    # Paramètres de fallback optimisés
    final_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=150,
        min_samples=25,
        cluster_selection_epsilon=1.2,
        metric='cosine',
        cluster_selection_method='eom'
    )
    
    optimization_used = "Paramètres haute qualité (fallback)"

# CLUSTERING FINAL AVEC ANALYSE COMPLÈTE
print(f"\n🎯 CLUSTERING FINAL HAUTE QUALITÉ")
print("="*50)

cluster_labels = final_clusterer.fit_predict(embeddings_2d)

# Statistiques complètes
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = (cluster_labels == -1).sum()
noise_ratio = n_noise / len(cluster_labels)

print(f"📊 RÉSULTATS FINAUX ({optimization_used}) :")
print(f"  🎯 Clusters détectés : {n_clusters}")
print(f"  🔇 Points de bruit : {n_noise} ({noise_ratio:.2%})")

# Calculs de qualité étendus
silhouette_final = None
if n_clusters > 1:
    mask = cluster_labels != -1
    if mask.sum() > n_clusters:
        try:
            silhouette_final = silhouette_score(embeddings_2d[mask], cluster_labels[mask])
            print(f"  📈 Score de silhouette : {silhouette_final:.4f}")
            
            if silhouette_final > 0.3:
                print(f"  🏆 EXCELLENT! Silhouette > 0.3")
            elif silhouette_final > 0.1:
                print(f"  ✅ BON! Silhouette > 0.1")
            elif silhouette_final > 0:
                print(f"  🟡 Acceptable - Silhouette positive")
            else:
                print(f"  ⚠️  Silhouette négative")
        except Exception as e:
            print(f"  ⚠️  Score de silhouette : Non calculable ({e})")

# Évaluation qualitative
print(f"\n🎖️  ÉVALUATION QUALITÉ :")
quality_score = 0

if 15 <= n_clusters <= 60:
    print(f"  ✅ Nombre de clusters optimal")
    quality_score += 3
elif 10 <= n_clusters <= 80:
    print(f"  🟡 Nombre de clusters acceptable")  
    quality_score += 2
else:
    print(f"  ⚠️  Nombre de clusters à optimiser")
    quality_score += 1

if noise_ratio < 0.05:
    print(f"  ✅ Très peu de bruit (<5%)")
    quality_score += 3
elif noise_ratio < 0.15:
    print(f"  🟡 Bruit acceptable (<15%)")
    quality_score += 2
else:
    print(f"  ⚠️  Bruit élevé (>15%)")
    quality_score += 1

if silhouette_final and silhouette_final > 0.2:
    print(f"  ✅ Excellente séparation des clusters")
    quality_score += 3
elif silhouette_final and silhouette_final > 0:
    print(f"  🟡 Séparation acceptable")
    quality_score += 2
else:
    print(f"  ⚠️  Amélioration de séparation possible")
    quality_score += 1

print(f"  🎯 Score qualité global : {quality_score}/9")

# Ajouter au dataframe
df['cluster'] = cluster_labels
df['umap_x'] = embeddings_2d[:, 0]
df['umap_y'] = embeddings_2d[:, 1]
df['umap_z'] = embeddings_3d[:, 0]

# Statistiques de distribution avancées
print(f"\n📊 ANALYSE DE DISTRIBUTION :")
cluster_counts = pd.Series(cluster_labels).value_counts().sort_values(ascending=False)

print(f"  📈 Plus gros cluster : {cluster_counts.iloc[0]} tickets")
print(f"  📉 Plus petit cluster : {cluster_counts.iloc[-1] if len(cluster_counts) > 1 else 'N/A'} tickets")
print(f"  📊 Médiane taille cluster : {cluster_counts.median():.0f} tickets")
print(f"  📊 Écart-type taille : {cluster_counts.std():.0f} tickets")

# TOP 20 clusters
print(f"\n📊 TOP 20 CLUSTERS :")
for i, (cluster, count) in enumerate(cluster_counts.head(20).items()):
    if cluster == -1:
        print(f"  Bruit      : {count:5d} tickets ({count/len(df)*100:.1f}%)")
    else:
        print(f"  Cluster {cluster:2d} : {count:5d} tickets ({count/len(df)*100:.1f}%)")

if len(cluster_counts) > 20:
    remaining = len(cluster_counts) - 20
    remaining_tickets = cluster_counts.iloc[20:].sum()
    print(f"  ... + {remaining} autres clusters avec {remaining_tickets} tickets")

# Sauvegarde
print(f"\n💾 SAUVEGARDE...")
output = dataiku.Dataset("incident_with_clusters_intensive")
output.write_with_schema(df)

print(f"\n" + "="*80)
print(f"🎉 CLUSTERING INTENSIF TERMINÉ AVEC SUCCÈS!")
print(f"📁 Dataset : incident_with_clusters_intensive")
print(f"🏆 Résultat obtenu après recherche exhaustive de 100 évaluations")
print(f"🎯 Prêt pour l'analyse fine des causes et le déploiement!")
print("="*80)

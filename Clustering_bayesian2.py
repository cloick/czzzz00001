# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import hdbscan
from sklearn.metrics import silhouette_score

print("🎯 CLUSTERING AVEC OPTIMISATION BAYÉSIENNE CORRIGÉE")
print("="*60)

# Read recipe inputs
dataset = dataiku.Dataset("incident_prepared_embeddings_creation")
df = dataset.get_dataframe()

print(f"Dataset chargé : {len(df)} lignes, {len(df.columns)} colonnes")

# Récupérer les colonnes d'embeddings
embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
print(f"Colonnes d'embeddings trouvées : {len(embedding_cols)}")

if len(embedding_cols) == 0:
    raise ValueError("Aucune colonne d'embedding trouvée!")

embeddings = df[embedding_cols].values
print(f"Shape des embeddings : {embeddings.shape}")

# Récupérer et encoder les variables catégorielles
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
    else:
        print(f"  ⚠️  {col} non trouvé dans le dataset")

# Combiner embeddings et features catégorielles
if cat_features:
    cat_features_array = np.array(cat_features).T
    scaler = StandardScaler()
    cat_features_scaled = scaler.fit_transform(cat_features_array)
    weights = np.array([0.5, 2.0, 1.0, 1.0, 3.0])[:len(cat_features)]
    cat_features_weighted = cat_features_scaled * weights
    features_combined = np.hstack([embeddings, cat_features_weighted])
    print(f"✅ Features combinées : {features_combined.shape}")
else:
    features_combined = embeddings
    print("⚠️  Utilisation des embeddings seuls")

# UMAP pour réduction dimensionnelle
print("\n🗺️  Réduction dimensionnelle avec UMAP...")
reducer_2d = UMAP(
    n_components=2, 
    random_state=42, 
    n_neighbors=15, 
    min_dist=0.1
)
embeddings_2d = reducer_2d.fit_transform(features_combined)

reducer_3d = UMAP(
    n_components=3, 
    random_state=42, 
    n_neighbors=15, 
    min_dist=0.1
)
embeddings_3d = reducer_3d.fit_transform(features_combined)

print(f"✅ UMAP 2D : {embeddings_2d.shape}")
print(f"✅ UMAP 3D : {embeddings_3d.shape}")

# Optimisation bayésienne CORRIGÉE
print("\n🔬 OPTIMISATION BAYÉSIENNE HDBSCAN (version corrigée)")
print("="*50)

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    
    print("✅ scikit-optimize importé avec succès")
    
    # Définir l'espace de recherche
    space = [
        Integer(50, 500, name='min_cluster_size'),
        Integer(5, 100, name='min_samples'), 
        Real(0.0, 3.0, name='cluster_selection_epsilon')
    ]
    
    cible_clusters = 15
    tolerance = 5
    
    print(f"🎯 Objectif : {cible_clusters} clusters (±{tolerance})")
    print(f"🔍 Espace de recherche :")
    print(f"   min_cluster_size : [50, 500]")
    print(f"   min_samples : [5, 100]") 
    print(f"   cluster_selection_epsilon : [0.0, 3.0]")
    
    # Fonction objective CORRIGÉE (sans n_jobs)
    @use_named_args(space)
    def objective(min_cluster_size, min_samples, cluster_selection_epsilon):
        # Validation des paramètres
        if min_samples > min_cluster_size:
            return 10.0  # Pénalité pour paramètres invalides
        
        try:
            # HDBSCAN SANS le paramètre n_jobs problématique
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=int(min_cluster_size),
                min_samples=int(min_samples),
                cluster_selection_epsilon=float(cluster_selection_epsilon),
                metric='euclidean'
                # SUPPRESSION du paramètre n_jobs qui causait l'erreur
            )
            
            # Effectuer le clustering
            labels = clusterer.fit_predict(embeddings_2d)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = (labels == -1).sum() / len(labels)
            
            # Pénaliser trop de bruit
            if noise_ratio > 0.5:
                return 8.0
            
            # Calculer le score de silhouette
            silhouette = -1
            if n_clusters > 1 and n_clusters < len(embeddings_2d) - 1:
                mask = labels != -1
                if mask.sum() > n_clusters:
                    try:
                        silhouette = silhouette_score(embeddings_2d[mask], labels[mask])
                    except:
                        silhouette = -1
            
            # Écart au nombre cible de clusters
            ecart = abs(n_clusters - cible_clusters)
            
            # Fonction objectif combinée (à minimiser)
            if ecart <= tolerance:
                # Dans la zone cible : privilégier la silhouette
                objective_value = ecart * 1.0 - silhouette
            else:
                # Hors zone : forte pénalité
                objective_value = 10.0 + ecart * 2.0
            
            print(f"MCS={min_cluster_size:3d}, MS={min_samples:2d}, ε={cluster_selection_epsilon:.2f}: "
                  f"{n_clusters:2d} clusters, {noise_ratio:.2%} bruit, sil={silhouette:.3f}, obj={objective_value:.3f}")
            
            return objective_value
            
        except Exception as e:
            print(f"❌ Erreur avec MCS={min_cluster_size}, MS={min_samples}, ε={cluster_selection_epsilon:.2f}: {e}")
            return 15.0  # Pénalité forte en cas d'erreur
    
    # Lancer l'optimisation bayésienne
    print(f"\n🚀 Démarrage de l'optimisation (50 évaluations)...")
    
    result = gp_minimize(
        objective,
        space,
        n_calls=50,  # Réduit pour aller plus vite
        random_state=42,
        verbose=False,  # Réduire le verbosité
        acq_func='EI'  # Expected Improvement
    )
    
    # Récupérer les meilleurs paramètres
    best_mcs, best_ms, best_eps = result.x
    best_objective = result.fun
    
    print(f"\n🏆 OPTIMISATION TERMINÉE!")
    print(f"Meilleur score objectif : {best_objective:.4f}")
    print(f"Meilleurs paramètres :")
    print(f"  min_cluster_size : {best_mcs}")
    print(f"  min_samples : {best_ms}")
    print(f"  cluster_selection_epsilon : {best_eps:.3f}")
    
    # Utiliser les meilleurs paramètres
    final_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=best_mcs,
        min_samples=best_ms,
        cluster_selection_epsilon=best_eps,
        metric='euclidean'
    )
    
    optimization_success = True
    
except ImportError:
    print("❌ scikit-optimize non disponible, utilisation des paramètres optimaux")
    final_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=240,
        min_samples=20,
        cluster_selection_epsilon=1.56,
        metric='euclidean'
    )
    optimization_success = False
    
except Exception as e:
    print(f"❌ Erreur lors de l'optimisation : {e}")
    print("   Utilisation des paramètres optimaux comme fallback")
    final_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=240,
        min_samples=20,
        cluster_selection_epsilon=1.56,
        metric='euclidean'
    )
    optimization_success = False

# Clustering final
print(f"\n🎯 CLUSTERING FINAL")
print("="*30)

cluster_labels = final_clusterer.fit_predict(embeddings_2d)

# Statistiques finales
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = (cluster_labels == -1).sum()
noise_ratio = n_noise / len(cluster_labels)

print(f"📊 RÉSULTATS FINAUX :")
print(f"  🎯 Clusters détectés : {n_clusters}")
print(f"  🔇 Points de bruit : {n_noise} ({noise_ratio:.2%})")

# Score de silhouette final
if n_clusters > 1:
    mask = cluster_labels != -1
    if mask.sum() > n_clusters:
        try:
            silhouette_final = silhouette_score(embeddings_2d[mask], cluster_labels[mask])
            print(f"  📈 Score de silhouette : {silhouette_final:.4f}")
        except Exception as e:
            print(f"  ⚠️  Score de silhouette : Non calculable")

# Validation du résultat
target_clusters = 15
if optimization_success:
    if n_clusters <= target_clusters + 5:
        print(f"  ✅ EXCELLENT! Optimisation réussie")
    else:
        print(f"  🟡 Optimisation partielle")
else:
    print(f"  🔄 Paramètres de fallback utilisés")

# Ajouter les résultats au dataframe
df['cluster'] = cluster_labels
df['umap_x'] = embeddings_2d[:, 0]
df['umap_y'] = embeddings_2d[:, 1]
df['umap_z'] = embeddings_3d[:, 0]

# Distribution des clusters
print(f"\n📊 DISTRIBUTION :")
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
for cluster, count in cluster_counts.head(10).items():  # Top 10
    if cluster == -1:
        print(f"  Bruit : {count:4d} tickets")
    else:
        print(f"  Cluster {cluster:2d} : {count:4d} tickets")

if len(cluster_counts) > 10:
    print(f"  ... et {len(cluster_counts)-10} autres clusters")

# Sauvegarder
print(f"\n💾 SAUVEGARDE...")
output = dataiku.Dataset("incident_with_clusters_bayesian")
output.write_with_schema(df)

print(f"✅ Clustering bayésien terminé!")
print(f"📁 Dataset : incident_with_clusters_bayesian")
print("="*60)

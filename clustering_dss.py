# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from umap import UMAP
import hdbscan
import warnings
warnings.filterwarnings("ignore")

# Configuration des graines pour reproductibilité
np.random.seed(42)

# Read recipe inputs
dataset = dataiku.Dataset("incident_prepared_embeddings_creation")
df = dataset.get_dataframe()

print(f"Dataset chargé : {len(df)} tickets avec {len(df.columns)} colonnes")

# ============================================================================
# 1. PRÉPARATION DES DONNÉES POUR LE CLUSTERING
# ============================================================================

# Récupérer les colonnes d'embeddings
embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
print(f"Colonnes d'embeddings trouvées : {len(embedding_cols)}")

embeddings = df[embedding_cols].values
print(f"Shape des embeddings : {embeddings.shape}")

# Encoder les variables catégorielles si pas déjà fait
cat_vars = ['Priorité', 'Service métier', 'Cat1', 'Cat2', 'Groupe affecté']
cat_features = []

from sklearn.preprocessing import LabelEncoder

for col in cat_vars:
    if col in df.columns:
        # Vérifier si déjà encodé
        if f'{col}_encoded' not in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('INCONNU'))
        
        cat_features.append(df[f'{col}_encoded'].values)

# Préparer les features catégorielles
if cat_features:
    cat_features = np.array(cat_features).T
    print(f"Shape des features catégorielles : {cat_features.shape}")
    
    # Normalisation
    scaler = StandardScaler()
    cat_features_scaled = scaler.fit_transform(cat_features)
    
    # Pondération selon votre expérience
    weights = np.array([0.5, 2.0, 1.0, 1.0, 3.0])  # Priorité, Service, Cat1, Cat2, Groupe
    cat_features_weighted = cat_features_scaled * weights
    
    # Combiner embeddings textuels + features catégorielles
    features_combined = np.hstack([embeddings, cat_features_weighted])
    print(f"Shape des features combinées : {features_combined.shape}")
else:
    features_combined = embeddings
    print("Utilisation des embeddings seuls")

# ============================================================================
# 2. RÉDUCTION DIMENSIONNELLE AVEC UMAP
# ============================================================================

print("\n" + "="*50)
print("RÉDUCTION DIMENSIONNELLE UMAP")
print("="*50)

# UMAP 2D pour le clustering
print("Réduction UMAP 2D...")
reducer_2d = UMAP(
    n_components=2,
    random_state=42,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean'
)
embeddings_2d = reducer_2d.fit_transform(features_combined)
print(f"UMAP 2D terminé : {embeddings_2d.shape}")

# UMAP 3D pour la visualisation
print("Réduction UMAP 3D...")
reducer_3d = UMAP(
    n_components=3,
    random_state=42,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean'
)
embeddings_3d = reducer_3d.fit_transform(features_combined)
print(f"UMAP 3D terminé : {embeddings_3d.shape}")

# ============================================================================
# 3. OPTIMISATION BAYÉSIENNE DES PARAMÈTRES HDBSCAN
# ============================================================================

print("\n" + "="*50)
print("OPTIMISATION BAYÉSIENNE HDBSCAN")
print("="*50)

def install_skopt():
    """Installe scikit-optimize si nécessaire"""
    try:
        import skopt
        print("✅ scikit-optimize déjà disponible")
        return True
    except ImportError:
        print("🔄 Installation de scikit-optimize...")
        try:
            import subprocess
            subprocess.check_call(["pip", "install", "scikit-optimize"])
            print("✅ scikit-optimize installé avec succès")
            return True
        except Exception as e:
            print(f"❌ Échec installation scikit-optimize: {e}")
            return False

def optimiser_hdbscan_bayesian(embeddings_2d, cible_clusters=15, tolerance=3, n_calls=50):
    """Optimisation bayésienne intensive des paramètres HDBSCAN"""
    
    if not install_skopt():
        print("⚠️  Passage à l'optimisation par grille...")
        return optimiser_hdbscan_grille(embeddings_2d, cible_clusters, tolerance)
    
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    
    print(f"🎯 Objectif : {cible_clusters} clusters (±{tolerance})")
    print(f"🔍 {n_calls} évaluations bayésiennes")
    
    # Espace de recherche élargi pour plus de puissance
    space = [
        Integer(50, 500, name='min_cluster_size'),      # Plus large que PC
        Integer(10, 150, name='min_samples'),           # Plus large que PC  
        Real(0.0, 3.0, name='cluster_selection_epsilon') # Plus large que PC
    ]
    
    best_score = float('inf')
    best_params = None
    iteration = 0
    
    @use_named_args(space)
    def objective(min_cluster_size, min_samples, cluster_selection_epsilon):
        nonlocal best_score, best_params, iteration
        iteration += 1
        
        # Contrainte de validité
        if min_samples > min_cluster_size:
            return 20.0
        
        try:
            # Clustering HDBSCAN
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
                metric='euclidean',
                n_jobs=1  # Pour éviter les conflits Dataiku
            )
            
            labels = clusterer.fit_predict(embeddings_2d)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = (labels == -1).sum() / len(labels)
            
            # Pénalités
            if noise_ratio > 0.6:  # Trop de bruit
                return 15.0 + noise_ratio * 10
            
            if n_clusters < 5:  # Trop peu de clusters
                return 10.0 + (5 - n_clusters) * 2
            
            if n_clusters > 30:  # Trop de clusters
                return 10.0 + (n_clusters - 30) * 0.5
            
            # Score de silhouette
            silhouette = -1
            if n_clusters > 1:
                mask = labels != -1
                if mask.sum() > n_clusters:
                    try:
                        silhouette = silhouette_score(embeddings_2d[mask], labels[mask])
                    except:
                        silhouette = -1
            
            # Fonction objectif améliorée
            ecart = abs(n_clusters - cible_clusters)
            
            if ecart <= tolerance:
                # Dans la tolérance : optimiser la silhouette
                objective_value = ecart * 1.0 - silhouette * 2.0
            else:
                # Hors tolérance : pénaliser fortement
                objective_value = 10.0 + ecart * 2.0 - silhouette
            
            # Tracking du meilleur
            if objective_value < best_score:
                best_score = objective_value
                best_params = {
                    'min_cluster_size': min_cluster_size,
                    'min_samples': min_samples,
                    'cluster_selection_epsilon': cluster_selection_epsilon,
                    'n_clusters': n_clusters,
                    'silhouette': silhouette,
                    'noise_ratio': noise_ratio
                }
            
            print(f"[{iteration:2d}/{n_calls}] MCS={min_cluster_size:3d} MS={min_samples:2d} ε={cluster_selection_epsilon:.2f} "
                  f"→ {n_clusters:2d} clusters, sil={silhouette:.3f}, bruit={noise_ratio:.1%}, obj={objective_value:.3f}")
            
            return objective_value
            
        except Exception as e:
            print(f"❌ Erreur : {e}")
            return 25.0
    
    # Lancement de l'optimisation
    print("🚀 Démarrage de l'optimisation bayésienne...")
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        random_state=42,
        n_initial_points=10,  # Points d'exploration initiale
        acq_func='EI',        # Expected Improvement
        verbose=False
    )
    
    print(f"\n🎉 OPTIMISATION TERMINÉE")
    print(f"📊 Meilleur score objectif : {best_score:.4f}")
    
    if best_params:
        print(f"🏆 MEILLEURS PARAMÈTRES :")
        print(f"   min_cluster_size: {best_params['min_cluster_size']}")
        print(f"   min_samples: {best_params['min_samples']}")
        print(f"   cluster_selection_epsilon: {best_params['cluster_selection_epsilon']:.3f}")
        print(f"📈 RÉSULTATS :")
        print(f"   Clusters: {best_params['n_clusters']}")
        print(f"   Silhouette: {best_params['silhouette']:.4f}")
        print(f"   Bruit: {best_params['noise_ratio']:.1%}")
        
        return best_params
    else:
        print("❌ Aucun paramètre optimal trouvé")
        return None

def optimiser_hdbscan_grille(embeddings_2d, cible_clusters=15, tolerance=3):
    """Optimisation par grille de recherche (fallback)"""
    print("🔍 Optimisation par grille de recherche...")
    
    # Grille élargie pour Dataiku
    grille_params = {
        'min_cluster_size': [50, 100, 150, 200, 250, 300, 400],
        'min_samples': [10, 20, 30, 40, 50, 75],
        'cluster_selection_epsilon': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    }
    
    best_params = None
    best_score = float('inf')
    best_ecart = float('inf')
    
    total_combinaisons = (len(grille_params['min_cluster_size']) * 
                         len(grille_params['min_samples']) * 
                         len(grille_params['cluster_selection_epsilon']))
    
    print(f"🔢 Testing {total_combinaisons} combinaisons...")
    
    iteration = 0
    for mcs in grille_params['min_cluster_size']:
        for ms in grille_params['min_samples']:
            for eps in grille_params['cluster_selection_epsilon']:
                iteration += 1
                
                if ms > mcs:  # Skip invalid combinations
                    continue
                
                try:
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=mcs,
                        min_samples=ms,
                        cluster_selection_epsilon=eps,
                        metric='euclidean'
                    )
                    
                    labels = clusterer.fit_predict(embeddings_2d)
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    noise_ratio = (labels == -1).sum() / len(labels)
                    
                    # Score de silhouette
                    silhouette = None
                    if n_clusters > 1:
                        mask = labels != -1
                        if mask.sum() > n_clusters:
                            try:
                                silhouette = silhouette_score(embeddings_2d[mask], labels[mask])
                            except:
                                silhouette = None
                    
                    ecart = abs(n_clusters - cible_clusters)
                    
                    if iteration % 10 == 0:
                        print(f"[{iteration:3d}] MCS={mcs:3d} MS={ms:2d} ε={eps:.1f} "
                              f"→ {n_clusters:2d} clusters, sil={silhouette:.3f if silhouette else 'N/A'}")
                    
                    # Critères de sélection
                    if (ecart <= tolerance and silhouette is not None and 
                        (ecart < best_ecart or (ecart == best_ecart and silhouette > best_score))):
                        best_ecart = ecart
                        best_score = silhouette
                        best_params = {
                            'min_cluster_size': mcs,
                            'min_samples': ms,
                            'cluster_selection_epsilon': eps,
                            'n_clusters': n_clusters,
                            'silhouette': silhouette,
                            'noise_ratio': noise_ratio
                        }
                
                except Exception as e:
                    continue
    
    return best_params

# Lancer l'optimisation
params_optimaux = optimiser_hdbscan_bayesian(
    embeddings_2d, 
    cible_clusters=15, 
    tolerance=3,      # Tolérance plus stricte
    n_calls=50        # Plus d'évaluations pour plus de puissance
)

# ============================================================================
# 4. CLUSTERING FINAL AVEC PARAMÈTRES OPTIMAUX
# ============================================================================

print("\n" + "="*50)
print("CLUSTERING FINAL")
print("="*50)

if params_optimaux:
    print("🎯 Application des paramètres optimaux...")
    final_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=params_optimaux['min_cluster_size'],
        min_samples=params_optimaux['min_samples'],
        cluster_selection_epsilon=params_optimaux['cluster_selection_epsilon'],
        metric='euclidean'
    )
else:
    print("⚠️  Utilisation de paramètres par défaut...")
    final_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=150,
        min_samples=30,
        cluster_selection_epsilon=1.0,
        metric='euclidean'
    )

# Clustering final
cluster_labels = final_clusterer.fit_predict(embeddings_2d)

# Statistiques finales
n_clusters_final = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise_final = (cluster_labels == -1).sum()
noise_ratio_final = n_noise_final / len(cluster_labels)

print(f"📊 RÉSULTATS FINAUX :")
print(f"   Clusters détectés : {n_clusters_final}")
print(f"   Points de bruit : {n_noise_final} ({noise_ratio_final:.1%})")

# Score de silhouette final
if n_clusters_final > 1:
    mask = cluster_labels != -1
    if mask.sum() > n_clusters_final:
        try:
            silhouette_final = silhouette_score(embeddings_2d[mask], cluster_labels[mask])
            print(f"   Score de silhouette : {silhouette_final:.4f}")
        except:
            print("   Score de silhouette : Non calculable")

# ============================================================================
# 5. AJOUT DES RÉSULTATS AU DATAFRAME
# ============================================================================

print("\n📝 Sauvegarde des résultats...")

# Ajouter les clusters
df['cluster'] = cluster_labels

# Ajouter les coordonnées UMAP pour visualisation
df['umap_2d_x'] = embeddings_2d[:, 0]
df['umap_2d_y'] = embeddings_2d[:, 1]
df['umap_3d_x'] = embeddings_3d[:, 0]
df['umap_3d_y'] = embeddings_3d[:, 1]
df['umap_3d_z'] = embeddings_3d[:, 2]

# Statistiques par cluster
print("\n📈 Distribution des clusters :")
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    if cluster_id == -1:
        print(f"   Bruit : {count} tickets")
    else:
        print(f"   Cluster {cluster_id} : {count} tickets")

# Sauvegarder les paramètres optimaux dans le dataframe
if params_optimaux:
    df.attrs['hdbscan_params'] = params_optimaux

print(f"✅ Dataset final : {len(df)} tickets avec {len(df.columns)} colonnes")

# ============================================================================
# 6. WRITE RECIPE OUTPUTS
# ============================================================================

output = dataiku.Dataset("incident_with_clusters")
output.write_with_schema(df)

print("🎉 CLUSTERING TERMINÉ AVEC SUCCÈS !")
print("📂 Dataset de sortie : incident_with_clusters")
print("🔥 Prêt pour l'attribution des causes !")

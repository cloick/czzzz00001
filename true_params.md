# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import hdbscan
from sklearn.metrics import silhouette_score

print("🎯 CLUSTERING AVEC PARAMÈTRES OPTIMAUX (PC LOCAL)")
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
        # Encodage simple par valeurs uniques
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
    
    # Normaliser les features catégorielles
    scaler = StandardScaler()
    cat_features_scaled = scaler.fit_transform(cat_features_array)
    
    # Pondération des features (comme dans votre code PC)
    weights = np.array([0.5, 2.0, 1.0, 1.0, 3.0])[:len(cat_features)]
    cat_features_weighted = cat_features_scaled * weights
    
    # Combiner
    features_combined = np.hstack([embeddings, cat_features_weighted])
    print(f"✅ Features combinées : {features_combined.shape}")
else:
    features_combined = embeddings
    print("⚠️  Utilisation des embeddings seuls")

# UMAP pour réduction dimensionnelle (paramètres identiques à votre PC)
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

# HDBSCAN avec VOS PARAMÈTRES OPTIMAUX exactement
print("\n🎯 Clustering HDBSCAN avec PARAMÈTRES OPTIMAUX...")
print("   (paramètres qui fonctionnaient sur votre PC)")

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=240,        # VOS paramètres optimaux
    min_samples=20,
    cluster_selection_epsilon=1.56,
    metric='euclidean'
)

print("🔄 Application du clustering...")
cluster_labels = clusterer.fit_predict(embeddings_2d)

# Statistiques du clustering
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = (cluster_labels == -1).sum()
noise_ratio = n_noise / len(cluster_labels)

print(f"\n📊 RÉSULTATS DU CLUSTERING:")
print(f"  🎯 Clusters détectés : {n_clusters}")
print(f"  🔇 Points de bruit : {n_noise} ({noise_ratio:.2%})")

# Score de silhouette
silhouette_score_val = None
if n_clusters > 1:
    mask = cluster_labels != -1
    if mask.sum() > n_clusters:
        try:
            silhouette_score_val = silhouette_score(embeddings_2d[mask], cluster_labels[mask])
            print(f"  📈 Score de silhouette : {silhouette_score_val:.4f}")
        except Exception as e:
            print(f"  ⚠️  Score de silhouette : Non calculable ({e})")

# Distribution des clusters
print(f"\n📊 DISTRIBUTION DES CLUSTERS:")
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()

for cluster, count in cluster_counts.items():
    if cluster == -1:
        print(f"  Bruit (cluster -1) : {count:4d} tickets ({count/len(cluster_labels)*100:.1f}%)")
    else:
        print(f"  Cluster {cluster:2d} : {count:4d} tickets ({count/len(cluster_labels)*100:.1f}%)")

# Ajouter les résultats au dataframe
df['cluster'] = cluster_labels
df['umap_x'] = embeddings_2d[:, 0]
df['umap_y'] = embeddings_2d[:, 1]
df['umap_z'] = embeddings_3d[:, 0]  # Première dimension 3D

# Analyse rapide des clusters par tickets fiables
if 'est_fiable' in df.columns:
    print(f"\n🔍 ANALYSE RAPIDE (tickets fiables):")
    
    for cluster in sorted(set(cluster_labels)):
        if cluster == -1:
            continue
            
        cluster_data = df[df['cluster'] == cluster]
        fiables = cluster_data[cluster_data['est_fiable']]
        
        if len(fiables) > 0 and 'cause' in fiables.columns:
            causes = fiables['cause'].value_counts()
            if len(causes) > 0:
                cause_principale = causes.index[0]
                count_principale = causes.iloc[0]
                confidence = count_principale / len(fiables)
                print(f"  Cluster {cluster:2d}: {len(cluster_data):4d} tickets, "
                      f"{len(fiables):2d} fiables → {cause_principale} (conf: {confidence:.2f})")
            else:
                print(f"  Cluster {cluster:2d}: {len(cluster_data):4d} tickets → À déterminer")
        else:
            print(f"  Cluster {cluster:2d}: {len(cluster_data):4d} tickets → À déterminer")

# Comparaison avec l'objectif
print(f"\n🎯 ÉVALUATION PAR RAPPORT À L'OBJECTIF:")
target_clusters = 15
if n_clusters <= target_clusters + 5:  # Tolérance de 5
    print(f"  ✅ EXCELLENT! {n_clusters} clusters détectés (objectif: ~{target_clusters})")
elif n_clusters <= target_clusters + 10:
    print(f"  🟡 BON: {n_clusters} clusters détectés (objectif: ~{target_clusters})")
else:
    print(f"  ❌ TROP: {n_clusters} clusters détectés (objectif: ~{target_clusters})")

if noise_ratio < 0.1:
    print(f"  ✅ Taux de bruit excellent: {noise_ratio:.2%}")
elif noise_ratio < 0.2:
    print(f"  🟡 Taux de bruit acceptable: {noise_ratio:.2%}")
else:
    print(f"  ⚠️  Taux de bruit élevé: {noise_ratio:.2%}")

if silhouette_score_val and silhouette_score_val > 0.5:
    print(f"  ✅ Score de silhouette excellent: {silhouette_score_val:.4f}")
elif silhouette_score_val and silhouette_score_val > 0.3:
    print(f"  🟡 Score de silhouette correct: {silhouette_score_val:.4f}")
elif silhouette_score_val:
    print(f"  ⚠️  Score de silhouette faible: {silhouette_score_val:.4f}")

# Sauvegarder le résultat
print(f"\n💾 SAUVEGARDE...")
output = dataiku.Dataset("incident_with_clusters_optimal")
output.write_with_schema(df)

print(f"✅ Clustering terminé avec succès!")
print(f"📁 Dataset de sortie : incident_with_clusters_optimal ({len(df)} lignes)")
print("="*60)
print("🎉 PRÊT pour l'analyse des causes et la suite du pipeline!")

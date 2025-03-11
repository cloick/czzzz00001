# Importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, adjusted_rand_score
import hdbscan
from umap import UMAP
import re

# Définir les graines aléatoires pour la reproductibilité
np.random.seed(42)

# Fonction de préparation des données - Modifiée pour se concentrer sur les variables catégorielles
def preparer_donnees(df):
    """Prépare les données en n'utilisant que les variables catégorielles"""
    # Copier le DataFrame pour éviter de modifier l'original
    df_prep = df.copy()
    
    # Encodage des caractéristiques catégorielles
    cat_vars = ['Priorité', 'Service métier', 'Cat1', 'Cat2', 'Groupe affecté']
    encoders = {}
    
    for col in cat_vars:
        if col in df_prep.columns:
            le = LabelEncoder()
            df_prep[f'{col}_encoded'] = le.fit_transform(df_prep[col].fillna('INCONNU'))
            encoders[col] = le
    
    # Création de variables pour la cause
    if 'cause' in df_prep.columns:
        le_cause = LabelEncoder()
        df_prep['cause_encoded'] = le_cause.fit_transform(df_prep['cause'].fillna('INCONNU'))
        
        # Sauvegarder les mappings de cause pour une utilisation ultérieure
        cause_mapping = dict(zip(le_cause.transform(le_cause.classes_), le_cause.classes_))
        encoders['cause'] = le_cause
        
    return df_prep, cause_mapping, encoders

# Fonction pour extraire et normaliser les caractéristiques catégorielles
def extraire_caracteristiques_categorielles(df, cat_vars, poids=None):
    """Extrait et normalise les caractéristiques catégorielles avec pondération optionnelle"""
    print(f"Extraction des {len(cat_vars)} variables catégorielles...")
    
    # Valeurs par défaut des poids si non spécifiés
    if poids is None:
        poids = {
            'Groupe affecté': 3.0,    # Plus forte influence basée sur l'information mutuelle
            'Service métier': 2.0,     # Seconde influence la plus forte
            'Cat1': 1.0,
            'Cat2': 1.0,
            'Priorité': 0.5            # Moins d'influence car relation non significative
        }
    
    # Extraction des caractéristiques catégorielles
    cat_features_list = []
    
    for col in cat_vars:
        if f'{col}_encoded' in df.columns:
            # Extraire la variable
            feature = df[f'{col}_encoded'].values.reshape(-1, 1)
            
            # Normaliser
            scaler = StandardScaler()
            feature_norm = scaler.fit_transform(feature)
            
            # Appliquer la pondération
            if col in poids:
                feature_norm = feature_norm * poids[col]
                
            cat_features_list.append(feature_norm)
    
    # Concaténer toutes les caractéristiques catégorielles
    if cat_features_list:
        cat_features = np.hstack(cat_features_list)
        print(f"Dimensions des caractéristiques catégorielles: {cat_features.shape}")
        return cat_features
    else:
        raise ValueError("Aucune caractéristique catégorielle n'a pu être extraite")

# Fonction d'optimisation des hyperparamètres HDBSCAN
def optimiser_hdbscan(features_2d, cible_clusters=15, tolerance=5):
    """Recherche les meilleurs paramètres HDBSCAN pour obtenir environ 15 clusters"""
    
    print(f"Optimisation des paramètres HDBSCAN pour {cible_clusters} clusters (±{tolerance})...")
    
    meilleurs_params = None
    meilleur_score = -1
    meilleur_n_clusters = 0
    meilleur_ecart = float('inf')
    
    # Grille d'hyperparamètres
    grille_params = {
        'min_cluster_size': [10, 20, 30, 50, 100, 200],
        'min_samples': [5, 10, 15, 20, 30],
        'cluster_selection_epsilon': [0.0, 0.1, 0.5, 1.0]
    }
    
    resultats = []
    
    for min_cluster_size in grille_params['min_cluster_size']:
        for min_samples in grille_params['min_samples']:
            for epsilon in grille_params['cluster_selection_epsilon']:
                # Ne tester que si min_samples <= min_cluster_size
                if min_samples > min_cluster_size:
                    continue
                    
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_epsilon=epsilon,
                    metric='euclidean'
                )
                
                labels = clusterer.fit_predict(features_2d)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                noise_ratio = (labels == -1).sum() / len(labels)
                
                # Calculer le score de silhouette si possible
                silhouette = None
                if n_clusters > 1 and n_clusters < len(features_2d) - 1:
                    mask = labels != -1
                    if mask.sum() > n_clusters:
                        try:
                            silhouette = silhouette_score(features_2d[mask], labels[mask])
                        except:
                            silhouette = None
                
                resultats.append({
                    'min_cluster_size': min_cluster_size,
                    'min_samples': min_samples,
                    'epsilon': epsilon,
                    'n_clusters': n_clusters,
                    'noise_ratio': noise_ratio,
                    'silhouette': silhouette
                })
                
                print(f"MCS={min_cluster_size}, MS={min_samples}, ε={epsilon}: {n_clusters} clusters, "
                      f"{noise_ratio:.2%} bruit, silhouette={silhouette:.4f if silhouette else 'N/A'}")
                
                # Vérifier si c'est le meilleur résultat jusqu'à présent
                ecart = abs(n_clusters - cible_clusters)
                
                if (ecart <= tolerance and 
                    silhouette is not None and 
                    (ecart < meilleur_ecart or 
                     (ecart == meilleur_ecart and silhouette > meilleur_score))):
                    meilleur_ecart = ecart
                    meilleur_score = silhouette
                    meilleur_n_clusters = n_clusters
                    meilleurs_params = {
                        'min_cluster_size': min_cluster_size,
                        'min_samples': min_samples,
                        'cluster_selection_epsilon': epsilon
                    }
    
    # Afficher résumé des meilleurs paramètres
    if meilleurs_params:
        print(f"\nMeilleurs paramètres trouvés :")
        print(f"- min_cluster_size: {meilleurs_params['min_cluster_size']}")
        print(f"- min_samples: {meilleurs_params['min_samples']}")
        print(f"- cluster_selection_epsilon: {meilleurs_params['cluster_selection_epsilon']}")
        print(f"Résultat: {meilleur_n_clusters} clusters avec silhouette {meilleur_score:.4f}")
    else:
        print("Aucun ensemble de paramètres n'a produit le nombre de clusters souhaité avec un score de silhouette valide.")
    
    return meilleurs_params, resultats

# Fonction de clustering des caractéristiques catégorielles
def clustering_categoriel(cat_features, cible_clusters=15, optimiser=True, params_hdbscan=None):
    """Effectue le clustering sur les caractéristiques catégorielles uniquement"""
    
    print("Réduction dimensionnelle avec UMAP...")
    reducer = UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=15,
        min_dist=0.1
    )
    features_2d = reducer.fit_transform(cat_features)
    
    # Paramètres HDBSCAN
    if params_hdbscan is not None:
        print("Utilisation des paramètres HDBSCAN fournis...")
        min_cluster_size = params_hdbscan.get('min_cluster_size', 200)
        min_samples = params_hdbscan.get('min_samples', 10)
        epsilon = params_hdbscan.get('cluster_selection_epsilon', 1.0)
    elif optimiser:
        print("Optimisation des paramètres HDBSCAN...")
        meilleurs_params, _ = optimiser_hdbscan(features_2d, cible_clusters=cible_clusters)
        
        if meilleurs_params:
            min_cluster_size = meilleurs_params['min_cluster_size']
            min_samples = meilleurs_params['min_samples']
            epsilon = meilleurs_params['cluster_selection_epsilon']
        else:
            # Paramètres par défaut si l'optimisation échoue
            print("Utilisation des paramètres par défaut...")
            min_cluster_size = 30
            min_samples = 10
            epsilon = 0.5
    else:
        # Paramètres de base
        min_cluster_size = 5
        min_samples = 2
        epsilon = 0.5
    
    print(f"Application de HDBSCAN avec min_cluster_size={min_cluster_size}, min_samples={min_samples}, epsilon={epsilon}...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        metric='euclidean'
    )
    clusters = clusterer.fit_predict(features_2d)
    
    # Statistiques du clustering
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = (clusters == -1).sum()
    
    print(f"Nombre de clusters détectés: {n_clusters}")
    print(f"Nombre de points classés comme bruit: {n_noise} ({n_noise/len(clusters):.2%})")
    
    # Visualisation
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=clusters,
        cmap='tab20',
        s=30,
        alpha=0.7
    )
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'Clustering catégoriel: {n_clusters} clusters détectés')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.tight_layout()
    plt.savefig("clustering_categoriel.png", dpi=300)
    plt.show()
    
    # Mesure de qualité
    if n_clusters > 1:
        mask = clusters != -1
        if mask.sum() > n_clusters:
            silhouette = silhouette_score(features_2d[mask], clusters[mask])
            print(f"Score de silhouette: {silhouette:.4f}")
    
    return clusters, features_2d, clusterer

# Fonction d'étiquetage des clusters
def etiqueter_clusters(cluster_labels, df, cause_mapping):
    """Associe chaque cluster à une cause basée sur les tickets fiables"""
    # Créer un dictionnaire pour stocker la distribution des causes par cluster
    cluster_to_cause_counts = {}
    
    # Pour chaque ticket fiable, ajouter sa cause au comptage de son cluster
    for idx, row in df[df['est_fiable']].iterrows():
        cluster = cluster_labels[df.index.get_loc(idx)]
        cause = row['cause_encoded']
        
        # Ignorer les points de bruit (-1)
        if cluster == -1:
            continue
            
        if cluster not in cluster_to_cause_counts:
            cluster_to_cause_counts[cluster] = {}
            
        if cause not in cluster_to_cause_counts[cluster]:
            cluster_to_cause_counts[cluster][cause] = 0
            
        cluster_to_cause_counts[cluster][cause] += 1
    
    # Déterminer la cause prédominante pour chaque cluster
    cluster_to_cause = {}
    cluster_to_cause_name = {}
    
    for cluster, cause_counts in cluster_to_cause_counts.items():
        if cause_counts:  # Si le cluster contient des tickets fiables
            predominant_cause = max(cause_counts.items(), key=lambda x: x[1])[0]
            cluster_to_cause[cluster] = predominant_cause
            cluster_to_cause_name[cluster] = cause_mapping[predominant_cause]
        else:
            # Pour les clusters sans tickets fiables, marquer comme "À déterminer"
            cluster_to_cause[cluster] = -1
            cluster_to_cause_name[cluster] = "À déterminer"
    
    # Ajouter les clusters qui n'ont pas de tickets fiables
    for cluster in set(cluster_labels) - {-1}:
        if cluster not in cluster_to_cause:
            cluster_to_cause[cluster] = -1
            cluster_to_cause_name[cluster] = "À déterminer"
    
    # Afficher le mapping cluster -> cause
    print("\nMapping des clusters vers les causes:")
    for cluster, cause_name in sorted(cluster_to_cause_name.items()):
        print(f"Cluster {cluster}: {cause_name}")
    
    return cluster_to_cause, cluster_to_cause_name

# Fonction d'analyse des clusters non identifiés
def analyser_clusters_inconnus(df, cluster_labels, features_2d, cluster_to_cause_name):
    """Analyse les clusters étiquetés comme "À déterminer" """
    unknown_clusters = [c for c, name in cluster_to_cause_name.items() if name == "À déterminer"]
    
    if not unknown_clusters:
        print("Tous les clusters ont été associés à une cause connue.")
        return
    
    print(f"\nAnalyse des {len(unknown_clusters)} clusters non identifiés:")
    
    for cluster in unknown_clusters:
        print(f"\n=== Analyse du cluster {cluster} ===")
        
        # Sélectionner les tickets de ce cluster
        mask = cluster_labels == cluster
        cluster_indices = np.where(mask)[0]
        cluster_tickets = df.iloc[cluster_indices]
        
        # Analyser les variables catégorielles dominantes
        cat_vars = ['Priorité', 'Service métier', 'Cat1', 'Cat2', 'Groupe affecté']
        
        print("Caractéristiques catégorielles dominantes:")
        for col in cat_vars:
            if col in cluster_tickets.columns:
                top_values = cluster_tickets[col].value_counts().head(3)
                if not top_values.empty:
                    print(f"  {col}: {', '.join([f'{v} ({c})' for v, c in top_values.items()])}")
        
        print(f"Nombre total de tickets dans ce cluster: {mask.sum()}")

# Fonction principale pour exécuter l'ensemble du processus
def executer_clustering_categoriel(chemin_metis, chemin_tickets_fiables, limit_samples=None, 
                                  optimiser_clusters=True, cible_clusters=15, params_hdbscan=None):
    """Exécute l'ensemble du processus de clustering basé uniquement sur les variables catégorielles"""
    # 1. Chargement et préparation des données
    print("\n==== 1. Chargement et préparation des données ====")
    df_metis = pd.read_csv(chemin_metis)
    df_metis = df_metis[['N° INC', 'Priorité', 'Service métier', 'Cat1', 'Cat2', 
                         'Groupe affecté', 'Notes de résolution', 'cause', 'souscause']]
    
    df_fiable = pd.read_csv(chemin_tickets_fiables)
    df_metis['est_fiable'] = df_metis['N° INC'].isin(df_fiable['N° INC'])
    
    print(f"Nombre total de tickets METIS: {len(df_metis)}")
    print(f"Nombre de tickets fiables: {df_metis['est_fiable'].sum()}")
    
    # Échantillonnage en conservant tous les tickets fiables
    if limit_samples and limit_samples < len(df_metis):
        df_fiables = df_metis[df_metis['est_fiable']]
        df_non_fiables = df_metis[~df_metis['est_fiable']].sample(
            min(limit_samples - len(df_fiables), len(df_metis) - len(df_fiables)),
            random_state=42
        )
        df_sample = pd.concat([df_fiables, df_non_fiables])
        print(f"Échantillon limité à {len(df_sample)} tickets dont {len(df_fiables)} fiables")
    else:
        df_sample = df_metis
    
    # 2. Préparation des données
    print("\n==== 2. Préparation des données ====")
    df_prep, cause_mapping, encoders = preparer_donnees(df_sample)
    
    # 3. Extraction des caractéristiques catégorielles
    print("\n==== 3. Extraction des caractéristiques catégorielles ====")
    cat_vars = ['Priorité', 'Service métier', 'Cat1', 'Cat2', 'Groupe affecté']
    poids = {
        'Groupe affecté': 3.0,
        'Service métier': 2.0,
        'Cat1': 1.0,
        'Cat2': 1.0,
        'Priorité': 0.5
    }
    
    cat_features = extraire_caracteristiques_categorielles(df_prep, cat_vars, poids)
    
    # 4. Clustering des caractéristiques catégorielles
    print("\n==== 4. Clustering des caractéristiques catégorielles ====")
    cluster_labels, features_2d, clusterer = clustering_categoriel(
        cat_features, 
        cible_clusters=cible_clusters,
        optimiser=optimiser_clusters,
        params_hdbscan=params_hdbscan
    )
    
    # 5. Étiquetage des clusters
    print("\n==== 5. Étiquetage des clusters ====")
    cluster_to_cause, cluster_to_cause_name = etiqueter_clusters(
        cluster_labels, df_prep, cause_mapping
    )
    
    # 6. Analyse des clusters non identifiés
    print("\n==== 6. Analyse des clusters non identifiés ====")
    analyser_clusters_inconnus(df_prep, cluster_labels, features_2d, cluster_to_cause_name)
    
    # 7. Évaluation de la qualité du clustering
    print("\n==== 7. Évaluation de la qualité du clustering ====")
    # Calculer l'ARI par rapport aux causes réelles (pour les tickets fiables uniquement)
    mask_fiable = df_prep['est_fiable']
    if mask_fiable.sum() > 0:
        fiable_indices = np.where(mask_fiable)[0]
        ari = adjusted_rand_score(
            df_prep.loc[mask_fiable, 'cause_encoded'],
            cluster_labels[fiable_indices]
        )
        print(f"Indice de Rand ajusté (ARI): {ari:.4f}")
        
    print("\nProcessus de clustering catégoriel terminé avec succès!")
    
    # Ajouter les clusters au DataFrame
    df_prep['cluster'] = cluster_labels
    
    # Renvoyer les résultats pour une utilisation ultérieure
    return {
        'df_prep': df_prep,
        'cluster_labels': cluster_labels,
        'features_2d': features_2d,
        'clusterer': clusterer,
        'cluster_to_cause_name': cluster_to_cause_name
    }

# Visualisation avancée des résultats
def visualiser_resultats(resultats):
    """Visualisation avancée des résultats du clustering"""
    df_prep = resultats['df_prep']
    cluster_labels = resultats['cluster_labels']
    features_2d = resultats['features_2d']
    cluster_to_cause_name = resultats['cluster_to_cause_name']
    
    # Créer une palette de couleurs pour les causes
    causes_uniques = set(cluster_to_cause_name.values())
    n_causes = len(causes_uniques)
    palette = sns.color_palette("hsv", n_causes)
    cause_colors = {cause: palette[i] for i, cause in enumerate(causes_uniques)}
    
    # Préparation des données pour la visualisation
    plot_data = pd.DataFrame({
        'UMAP1': features_2d[:, 0],
        'UMAP2': features_2d[:, 1],
        'Cluster': cluster_labels,
        'Est_fiable': df_prep['est_fiable'],
    })
    
    # Ajouter la cause identifiée
    plot_data['Cause'] = [
        cluster_to_cause_name.get(c, "Bruit") if c != -1 else "Bruit"
        for c in cluster_labels
    ]
    
    # Visualisation principale
    plt.figure(figsize=(14, 10))
    
    # Tracer les points par cause identifiée
    for cause in causes_uniques:
        mask = plot_data['Cause'] == cause
        plt.scatter(
            plot_data.loc[mask, 'UMAP1'],
            plot_data.loc[mask, 'UMAP2'],
            c=[cause_colors[cause]],
            label=cause,
            alpha=0.7,
            s=30
        )
    
    # Mettre en évidence les tickets fiables
    fiables_mask = plot_data['Est_fiable']
    plt.scatter(
        plot_data.loc[fiables_mask, 'UMAP1'],
        plot_data.loc[fiables_mask, 'UMAP2'],
        edgecolor='black',
        facecolor='none',
        s=50,
        linewidth=1,
        alpha=1,
        label='Tickets fiables'
    )
    
    plt.title('Clustering des tickets METIS par cause identifiée (variables catégorielles uniquement)')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("visualisation_clustering_categoriel.png", dpi=300)
    plt.show()
    
    return plot_data

# Exemple d'utilisation
if __name__ == "__main__":
    # Chemins des fichiers
    chemin_metis = "metis_tickets.csv"
    chemin_tickets_fiables = "gdp_tickets.csv"
    
    # Paramètres optimaux précédemment identifiés (si disponibles)
    params_optimaux = {
        'min_cluster_size': 200,
        'min_samples': 10,
        'cluster_selection_epsilon': 1.0
    }
    
    # Exécuter le clustering catégoriel
    resultats = executer_clustering_categoriel(
        chemin_metis=chemin_metis,
        chemin_tickets_fiables=chemin_tickets_fiables,
        limit_samples=20000,
        optimiser_clusters=True,  # Mettre à False si vous avez déjà des paramètres optimaux
        cible_clusters=15,
        params_hdbscan=None  # Utiliser params_optimaux si vous avez des paramètres préalablement optimisés
    )
    
    # Visualiser les résultats
    plot_data = visualiser_resultats(resultats)
    
    # Analyser la distribution des causes
    print("\nDistribution des causes identifiées:")
    cause_distribution = plot_data['Cause'].value_counts()
    print(cause_distribution)
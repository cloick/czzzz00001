import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.preprocessing import StandardScaler
import hdbscan
from umap import UMAP
from collections import Counter
import joblib
from tqdm import tqdm

# =========== CHARGEMENT DES RÉSULTATS DE LA PHASE 1 ===========

def charger_resultats_phase1(chemin_resultats="resultats"):
    """
    Charge les résultats de la Phase 1 à partir des fichiers sauvegardés
    
    Returns:
        dict: Dictionnaire contenant tous les résultats de la Phase 1
    """
    resultats = {}
    
    # Charger les DataFrames
    try:
        resultats['df_prep'] = pd.read_csv(f"{chemin_resultats}/df_prep.csv")
        resultats['df_fiable'] = pd.read_csv(f"{chemin_resultats}/df_fiable.csv")
        print("DataFrames chargés avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement des DataFrames: {e}")
        return None
    
    # Charger les embeddings et autres arrays numpy
    try:
        resultats['embeddings_fiables'] = np.load(f"{chemin_resultats}/embeddings_fiables.npy")
        resultats['embeddings_enrichis'] = np.load(f"{chemin_resultats}/embeddings_enrichis.npy")
        resultats['embeddings_2d'] = np.load(f"{chemin_resultats}/embeddings_2d.npy")
        resultats['clusters'] = np.load(f"{chemin_resultats}/clusters.npy")
        print("Embeddings et clusters chargés avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement des embeddings: {e}")
        return None
    
    return resultats

# =========== OPTIMISATION DES HYPERPARAMÈTRES HDBSCAN ===========

def grille_recherche_hdbscan(embeddings, labels, param_grid, n_samples=None):
    """
    Effectue une recherche en grille pour les hyperparamètres de HDBSCAN
    
    Args:
        embeddings: Embeddings des tickets (enrichis ou non)
        labels: Labels réels (causes) pour évaluation
        param_grid: Dict avec les valeurs des hyperparamètres à tester
        n_samples: Nombre d'échantillons à utiliser (None = tous)
        
    Returns:
        dict: Résultats des différentes configurations testées
    """
    # Si n_samples est spécifié, échantillonner les données
    if n_samples is not None and n_samples < len(embeddings):
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings_sample = embeddings[indices]
        labels_sample = labels[indices] if labels is not None else None
    else:
        embeddings_sample = embeddings
        labels_sample = labels
    
    # Initialiser les résultats
    resultats = []
    
    # Créer toutes les combinaisons de paramètres
    combinaisons = []
    for min_cluster_size in param_grid['min_cluster_size']:
        for min_samples in param_grid['min_samples']:
            for cluster_selection_epsilon in param_grid['cluster_selection_epsilon']:
                for metric in param_grid['metric']:
                    combinaisons.append({
                        'min_cluster_size': min_cluster_size,
                        'min_samples': min_samples,
                        'cluster_selection_epsilon': cluster_selection_epsilon,
                        'metric': metric
                    })
    
    # Exécuter HDBSCAN pour chaque combinaison de paramètres
    print(f"Évaluation de {len(combinaisons)} combinaisons de paramètres...")
    for i, params in enumerate(tqdm(combinaisons)):
        clusterer = hdbscan.HDBSCAN(**params)
        clusters = clusterer.fit_predict(embeddings_sample)
        
        # Calculer les métriques
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = (clusters == -1).sum()
        metriques = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'percent_noise': n_noise / len(embeddings_sample) * 100
        }
        
        # Silhouette score (seulement si plus d'un cluster et des points non-bruit)
        if n_clusters > 1:
            mask = clusters != -1
            if mask.sum() > n_clusters:
                try:
                    silhouette = silhouette_score(
                        embeddings_sample[mask], 
                        clusters[mask]
                    )
                    metriques['silhouette'] = silhouette
                except:
                    metriques['silhouette'] = float('nan')
        
        # ARI et AMI (seulement si des labels sont fournis)
        if labels_sample is not None:
            mask = clusters != -1
            if mask.sum() > 0:
                ari = adjusted_rand_score(labels_sample[mask], clusters[mask])
                ami = adjusted_mutual_info_score(labels_sample[mask], clusters[mask])
                metriques['ari'] = ari
                metriques['ami'] = ami
        
        # Stocker les résultats
        resultats.append({
            'params': params,
            'metrics': metriques
        })
    
    # Trier les résultats par score de silhouette décroissant
    resultats_tries = sorted(
        resultats, 
        key=lambda x: x['metrics'].get('silhouette', -1), 
        reverse=True
    )
    
    return resultats_tries

def visualiser_resultats_optimisation(resultats_tries):
    """
    Visualise les résultats de l'optimisation des hyperparamètres
    
    Args:
        resultats_tries: Résultats triés de la grille de recherche
    """
    # Extraire les données pour visualisation
    silhouettes = [r['metrics'].get('silhouette', 0) for r in resultats_tries if 'silhouette' in r['metrics']]
    n_clusters = [r['metrics']['n_clusters'] for r in resultats_tries]
    percent_noise = [r['metrics']['percent_noise'] for r in resultats_tries]
    
    # Créer une figure avec plusieurs sous-graphiques
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Graphique des scores de silhouette
    sns.barplot(x=list(range(len(silhouettes))), y=silhouettes, ax=axes[0])
    axes[0].set_title('Scores de silhouette pour différentes configurations')
    axes[0].set_xlabel('Configuration (index)')
    axes[0].set_ylabel('Score de silhouette')
    
    # Graphique du nombre de clusters
    sns.barplot(x=list(range(len(n_clusters))), y=n_clusters, ax=axes[1])
    axes[1].set_title('Nombre de clusters pour différentes configurations')
    axes[1].set_xlabel('Configuration (index)')
    axes[1].set_ylabel('Nombre de clusters')
    
    # Graphique du pourcentage de bruit
    sns.barplot(x=list(range(len(percent_noise))), y=percent_noise, ax=axes[2])
    axes[2].set_title('Pourcentage de points classés comme bruit')
    axes[2].set_xlabel('Configuration (index)')
    axes[2].set_ylabel('% de points de bruit')
    
    plt.tight_layout()
    plt.savefig('optimisation_hyperparametres.png', dpi=300)
    plt.show()
    
    # Afficher les meilleurs paramètres
    print("Top 5 configurations:")
    for i, res in enumerate(resultats_tries[:5]):
        print(f"\n{i+1}. Paramètres: {res['params']}")
        print(f"   Métriques: {res['metrics']}")

# =========== VISUALISATION ET ANALYSE DES CLUSTERS ===========

def visualiser_clusters_optimaux(embeddings, clusters, labels=None, titre="Clusters optimaux"):
    """
    Visualise les clusters formés par HDBSCAN avec les paramètres optimaux
    
    Args:
        embeddings: Embeddings des tickets (peut être en haute dimension)
        clusters: Étiquettes de cluster attribuées par HDBSCAN
        labels: Labels réels (causes) pour comparaison
        titre: Titre du graphique
    """
    # Réduction dimensionnelle si nécessaire
    if embeddings.shape[1] > 2:
        print("Réduction dimensionnelle avec UMAP...")
        reducer = UMAP(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings
    
    # Créer une figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Visualisation des clusters
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    scatter1 = ax1.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=clusters, cmap='tab20', s=30, alpha=0.7
    )
    ax1.set_title(f"Clusters détectés par HDBSCAN ({n_clusters} clusters)")
    ax1.set_xlabel('UMAP Dimension 1')
    ax1.set_ylabel('UMAP Dimension 2')
    plt.colorbar(scatter1, ax=ax1, label='Cluster')
    
    # Visualisation des causes réelles (si disponibles)
    if labels is not None:
        scatter2 = ax2.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=labels, cmap='tab20', s=30, alpha=0.7
        )
        ax2.set_title("Causes réelles")
        ax2.set_xlabel('UMAP Dimension 1')
        ax2.set_ylabel('UMAP Dimension 2')
        plt.colorbar(scatter2, ax=ax2, label='Cause')
    
    plt.tight_layout()
    plt.savefig(f"{titre.replace(' ', '_').lower()}.png", dpi=300)
    plt.show()
    
    return embeddings_2d

def analyser_composition_clusters(clusters, df_fiable):
    """
    Analyse la composition de chaque cluster en termes de causes
    
    Args:
        clusters: Clusters assignés par HDBSCAN
        df_fiable: DataFrame des tickets fiables avec leurs causes
        
    Returns:
        dict: Dictionnaire avec la composition de chaque cluster
    """
    composition = {}
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    
    # Pour chaque cluster, calculer la distribution des causes
    for cluster_id in range(-1, n_clusters):
        # Indices des tickets dans ce cluster
        indices_cluster = np.where(clusters == cluster_id)[0]
        
        # Extraire les causes de ces tickets
        causes = df_fiable.iloc[indices_cluster]['cause'].values
        
        # Calculer la distribution
        distribution = Counter(causes)
        total = len(causes)
        
        # Calculer les proportions
        proportions = {cause: count/total*100 for cause, count in distribution.items()}
        
        # Cause majoritaire
        cause_majoritaire = max(distribution.items(), key=lambda x: x[1])[0] if distribution else "Inconnu"
        
        # Homogénéité (pourcentage de la cause majoritaire)
        homogeneite = max(proportions.values()) if proportions else 0
        
        # Stocker les résultats
        composition[cluster_id] = {
            'distribution': distribution,
            'proportions': proportions,
            'cause_majoritaire': cause_majoritaire,
            'homogeneite': homogeneite,
            'taille': total
        }
    
    return composition

def visualiser_composition_clusters(composition, n_clusters):
    """
    Visualise la composition des clusters en termes de causes
    
    Args:
        composition: Dictionnaire avec la composition de chaque cluster
        n_clusters: Nombre de clusters (sans compter le bruit)
    """
    # Préparer les données pour le graphique
    cluster_ids = list(range(n_clusters))  # Exclure le cluster de bruit (-1)
    homogeneites = [composition[i]['homogeneite'] for i in cluster_ids]
    tailles = [composition[i]['taille'] for i in cluster_ids]
    causes_majoritaires = [composition[i]['cause_majoritaire'] for i in cluster_ids]
    
    # Créer un DataFrame pour faciliter la visualisation
    df_viz = pd.DataFrame({
        'cluster_id': cluster_ids,
        'homogeneite': homogeneites,
        'taille': tailles,
        'cause_majoritaire': causes_majoritaires
    })
    
    # Trier par taille de cluster
    df_viz = df_viz.sort_values('taille', ascending=False)
    
    # Créer une figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Homogénéité des clusters
    sns.barplot(x='cluster_id', y='homogeneite', data=df_viz, ax=ax1)
    ax1.set_title('Homogénéité des clusters (% de la cause majoritaire)')
    ax1.set_xlabel('Cluster ID')
    ax1.set_ylabel('Homogénéité (%)')
    ax1.set_ylim(0, 100)
    
    # Taille des clusters
    bars = sns.barplot(x='cluster_id', y='taille', data=df_viz, ax=ax2)
    ax2.set_title('Taille des clusters')
    ax2.set_xlabel('Cluster ID')
    ax2.set_ylabel('Nombre de tickets')
    
    # Ajouter les causes majoritaires comme étiquettes
    for i, bar in enumerate(bars.patches):
        cause = df_viz.iloc[i]['cause_majoritaire']
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2., 
            height + 5, 
            cause, 
            ha='center', 
            va='bottom',
            rotation=45
        )
    
    plt.tight_layout()
    plt.savefig('composition_clusters.png', dpi=300)
    plt.show()

# =========== APPROCHE HIÉRARCHIQUE (CAUSE PUIS SOUS-CAUSE) ===========

def clustering_deux_niveaux(embeddings, df_fiable, params_hdbscan):
    """
    Implémente une approche hiérarchique en deux niveaux:
    1. Clusteriser pour identifier les causes principales
    2. Pour chaque cause, clusteriser pour identifier les sous-causes
    
    Args:
        embeddings: Embeddings enrichis des tickets
        df_fiable: DataFrame des tickets fiables
        params_hdbscan: Paramètres optimaux pour HDBSCAN
        
    Returns:
        dict: Modèles HDBSCAN pour chaque niveau
    """
    print("Application de l'approche à deux niveaux...")
    
    # Niveau 1: Clustering pour les causes principales
    print("Niveau 1: Clustering pour causes principales...")
    clusterer_niveau1 = hdbscan.HDBSCAN(**params_hdbscan)
    clusters_niveau1 = clusterer_niveau1.fit_predict(embeddings)
    
    # Analyser la composition des clusters de niveau 1
    n_clusters_niveau1 = len(set(clusters_niveau1)) - (1 if -1 in clusters_niveau1 else 0)
    composition_niveau1 = analyser_composition_clusters(clusters_niveau1, df_fiable)
    
    # Visualiser la composition des clusters de niveau 1
    visualiser_composition_clusters(composition_niveau1, n_clusters_niveau1)
    
    # Mapping des clusters vers les causes
    mapping_cluster_cause = {}
    for cluster_id in range(n_clusters_niveau1):
        if cluster_id in composition_niveau1:
            mapping_cluster_cause[cluster_id] = composition_niveau1[cluster_id]['cause_majoritaire']
    
    # Niveau 2: Pour chaque cause, clustering pour les sous-causes
    print("Niveau 2: Clustering pour sous-causes...")
    clusterers_niveau2 = {}
    
    # Obtenir toutes les causes uniques
    causes_uniques = df_fiable['cause'].unique()
    
    for cause in causes_uniques:
        # Tickets de cette cause
        indices_cause = df_fiable[df_fiable['cause'] == cause].index
        
        if len(indices_cause) >= params_hdbscan['min_cluster_size']:
            print(f"Clustering pour la cause '{cause}' ({len(indices_cause)} tickets)...")
            
            # Extraire les embeddings correspondants
            embeddings_cause = embeddings[indices_cause]
            
            # Appliquer HDBSCAN pour cette cause
            clusterer_cause = hdbscan.HDBSCAN(**params_hdbscan)
            clusters_cause = clusterer_cause.fit_predict(embeddings_cause)
            
            # Analyser la composition des clusters en termes de sous-causes
            df_cause = df_fiable.iloc[indices_cause]
            
            # Stocker le clusterer
            clusterers_niveau2[cause] = clusterer_cause
        else:
            print(f"Pas assez de tickets pour la cause '{cause}' ({len(indices_cause)} tickets) - seuil min: {params_hdbscan['min_cluster_size']}")
    
    return {
        'niveau1': clusterer_niveau1,
        'niveau2': clusterers_niveau2,
        'mapping_cluster_cause': mapping_cluster_cause
    }

# =========== MODÈLE DE CLASSIFICATION FINAL ===========

def creer_modele_final(clusterers, mapping_cluster_cause):
    """
    Crée un modèle final pour prédire les causes et sous-causes
    
    Args:
        clusterers: Dict contenant les modèles HDBSCAN pour les niveaux 1 et 2
        mapping_cluster_cause: Mapping des clusters de niveau 1 vers les causes
        
    Returns:
        dict: Modèle final avec toutes les informations nécessaires
    """
    return {
        'clusterers': clusterers,
        'mapping_cluster_cause': mapping_cluster_cause
    }

def predire_causes(modele, embeddings):
    """
    Utilise le modèle final pour prédire les causes de nouveaux tickets
    
    Args:
        modele: Modèle final créé par creer_modele_final
        embeddings: Embeddings des nouveaux tickets
        
    Returns:
        list: Causes prédites pour chaque ticket
    """
    # Extraire les composants du modèle
    clusterer_niveau1 = modele['clusterers']['niveau1']
    clusterers_niveau2 = modele['clusterers']['niveau2']
    mapping_cluster_cause = modele['mapping_cluster_cause']
    
    # Prédire les clusters de niveau 1
    clusters_niveau1 = clusterer_niveau1.approximate_predict(embeddings)
    
    # Convertir les clusters en causes
    causes_predites = []
    sous_causes_predites = []
    
    for i, cluster_id in enumerate(clusters_niveau1):
        # Obtenir la cause prédite
        if cluster_id in mapping_cluster_cause:
            cause = mapping_cluster_cause[cluster_id]
        else:
            cause = "Non déterminée"  # Cluster de bruit ou inconnu
        
        causes_predites.append(cause)
        
        # Essayer de prédire la sous-cause si possible
        if cause in clusterers_niveau2:
            clusterer_niveau2 = clusterers_niveau2[cause]
            cluster_niveau2 = clusterer_niveau2.approximate_predict([embeddings[i]])[0]
            
            # Pour simplifier, on ne fait pas de mapping des clusters de niveau 2 ici
            # Dans un système complet, on aurait un mapping similaire pour les sous-causes
            sous_cause = f"Sous-cause du cluster {cluster_niveau2}"
        else:
            sous_cause = "Non déterminée"
        
        sous_causes_predites.append(sous_cause)
    
    return causes_predites, sous_causes_predites

def evaluer_predictions(causes_reelles, causes_predites):
    """
    Évalue les prédictions du modèle
    
    Args:
        causes_reelles: Causes réelles des tickets
        causes_predites: Causes prédites par le modèle
        
    Returns:
        dict: Métriques d'évaluation
    """
    # Calculer l'exactitude
    exactitude = sum(r == p for r, p in zip(causes_reelles, causes_predites)) / len(causes_reelles)
    
    # Calculer la matrice de confusion
    causes_uniques = list(set(causes_reelles) | set(causes_predites))
    matrice_confusion = {}
    
    for cause_reelle in causes_uniques:
        matrice_confusion[cause_reelle] = {}
        for cause_predite in causes_uniques:
            count = sum(1 for r, p in zip(causes_reelles, causes_predites) 
                         if r == cause_reelle and p == cause_predite)
            matrice_confusion[cause_reelle][cause_predite] = count
    
    return {
        'exactitude': exactitude,
        'matrice_confusion': matrice_confusion
    }

def visualiser_matrice_confusion(matrice_confusion):
    """
    Visualise la matrice de confusion
    
    Args:
        matrice_confusion: Matrice de confusion calculée par evaluer_predictions
    """
    # Convertir la matrice de confusion en DataFrame
    causes = sorted(matrice_confusion.keys())
    df_confusion = pd.DataFrame(index=causes, columns=causes)
    
    for cause_reelle in causes:
        for cause_predite in causes:
            df_confusion.loc[cause_reelle, cause_predite] = matrice_confusion[cause_reelle].get(cause_predite, 0)
    
    # Normaliser par ligne
    df_confusion_norm = df_confusion.div(df_confusion.sum(axis=1), axis=0)
    
    # Visualiser
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_confusion_norm, annot=df_confusion, fmt='d', cmap='YlGnBu')
    plt.title('Matrice de confusion')
    plt.xlabel('Cause prédite')
    plt.ylabel('Cause réelle')
    plt.tight_layout()
    plt.savefig('matrice_confusion.png', dpi=300)
    plt.show()

# =========== FONCTION PRINCIPALE PHASE 2 ===========

def executer_phase2(resultats_phase1=None, optimiser=True):
    """
    Exécute la Phase 2 complète: développement du modèle de clustering
    
    Args:
        resultats_phase1: Résultats chargés de la Phase 1 (None = charger automatiquement)
        optimiser: Si True, effectue l'optimisation des hyperparamètres
        
    Returns:
        dict: Modèle final et résultats d'évaluation
    """
    print("\n=============== PHASE 2: DÉVELOPPEMENT DU MODÈLE DE CLUSTERING ===============\n")
    
    # 1. Charger les résultats de la Phase 1 si nécessaire
    if resultats_phase1 is None:
        print("Chargement des résultats de la Phase 1...")
        resultats_phase1 = charger_resultats_phase1()
        if resultats_phase1 is None:
            print("Impossible de charger les résultats de la Phase 1. Arrêt de la Phase 2.")
            return None
    
    # Extraire les données nécessaires
    df_fiable = resultats_phase1['df_fiable']
    embeddings_enrichis = resultats_phase1['embeddings_enrichis']
    
    # 2. Optimisation des hyperparamètres de HDBSCAN (si demandé)
    meilleurs_params = None
    if optimiser:
        print("\nOptimisation des hyperparamètres de HDBSCAN...")
        param_grid = {
            'min_cluster_size': [5, 10, 15, 20, 25],
            'min_samples': [2, 5, 10],
            'cluster_selection_epsilon': [0.0, 0.5, 1.0],
            'metric': ['euclidean', 'manhattan']
        }
        
        # Utiliser les labels de cause pour l'évaluation
        labels = df_fiable['cause_encoded'].values
        
        # Effectuer la recherche en grille
        resultats_optimisation = grille_recherche_hdbscan(
            embeddings_enrichis, 
            labels, 
            param_grid
        )
        
        # Visualiser les résultats de l'optimisation
        visualiser_resultats_optimisation(resultats_optimisation)
        
        # Sélectionner les meilleurs paramètres
        meilleurs_params = resultats_optimisation[0]['params']
        print(f"\nMeilleurs paramètres trouvés: {meilleurs_params}")
    else:
        # Paramètres par défaut basés sur l'exploration précédente
        meilleurs_params = {
            'min_cluster_size': 15,
            'min_samples': 5,
            'cluster_selection_epsilon': 0.5,
            'metric': 'euclidean'
        }
        print(f"\nUtilisation des paramètres par défaut: {meilleurs_params}")
    
    # 3. Appliquer HDBSCAN avec les meilleurs paramètres
    print("\nApplication de HDBSCAN avec les paramètres optimaux...")
    clusterer = hdbscan.HDBSCAN(**meilleurs_params)
    clusters = clusterer.fit_predict(embeddings_enrichis)
    
    # Métriques de qualité
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = (clusters == -1).sum()
    print(f"Nombre de clusters détectés: {n_clusters}")
    print(f"Nombre de points classés comme bruit: {n_noise} ({n_noise/len(clusters)*100:.2f}%)")
    
    # Silhouette score
    if n_clusters > 1:
        mask = clusters != -1
        if mask.sum() > n_clusters:
            silhouette = silhouette_score(embeddings_enrichis[mask], clusters[mask])
            print(f"Score de silhouette: {silhouette:.4f}")
    
    # Visualiser les clusters
    print("\nVisualisation des clusters optimaux...")
    embeddings_2d = visualiser_clusters_optimaux(
        embeddings_enrichis, 
        clusters, 
        df_fiable['cause_encoded'].values
    )
    
    # 4. Analyser la composition des clusters
    print("\nAnalyse de la composition des clusters...")
    composition = analyser_composition_clusters(clusters, df_fiable)
    visualiser_composition_clusters(composition, n_clusters)
    
    # 5. Approche à deux niveaux (cause puis sous-cause)
    print("\nMise en œuvre de l'approche à deux niveaux...")
    clusterers = clustering_deux_niveaux(
        embeddings_enrichis, 
        df_fiable, 
        meilleurs_params
    )
    
    # 6. Créer le modèle final
    modele_final = creer_modele_final(
        clusterers, 
        clusterers['mapping_cluster_cause']
    )
    
    # 7. Évaluation du modèle sur les données fiables (validation)
    print("\nÉvaluation du modèle final...")
    
    # Prédiction sur les données d'entraînement (normalement, on utiliserait un jeu de test)
    causes_predites, _ = predire_causes(modele_final, embeddings_enrichis)
    
    # Évaluation des prédictions
    resultats_evaluation = evaluer_predictions(
        df_fiable['cause'].values,
        causes_predites
    )
    
    print(f"Exactitude: {resultats_evaluation['exactitude']*100:.2f}%")
    
    # Visualiser la matrice de confusion
    visualiser_matrice_confusion(resultats_evaluation['matrice_confusion'])
    
    # 8. Sauvegarder le modèle final
    print("\nSauvegarde du modèle final...")
    os.makedirs("modeles", exist_ok=True)
    joblib.dump(modele_final, "modeles/modele_clustering_metis.joblib")
    
    print("\n=============== PHASE 2 TERMINÉE AVEC SUCCÈS ===============")
    
    return {
        'modele': modele_final,
        'evaluation': resultats_evaluation,
        'clusters': clusters,
        'embeddings_2d': embeddings_2d,
        'composition': composition
    }

# =========== EXÉCUTION DU SCRIPT ===========

if __name__ == "__main__":
    # Exécuter la Phase 2
    resultats_phase2 = executer_phase2(optimiser=True)
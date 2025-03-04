# =========== IMPORTATIONS NÉCESSAIRES ===========
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import hdbscan
from umap import UMAP
from collections import Counter
import joblib
from tqdm import tqdm
from datetime import datetime

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
def grille_recherche_hdbscan(embeddings, labels, param_grid, n_samples=None, checkpoint_file="resultats/optimisation_checkpoint.pkl"):
    """
    Effectue une recherche en grille pour les hyperparamètres de HDBSCAN
    
    Args:
        embeddings: Embeddings des tickets (enrichis ou non)
        labels: Labels réels (causes) pour évaluation
        param_grid: Dict avec les valeurs des hyperparamètres à tester
        n_samples: Nombre d'échantillons à utiliser (None = tous)
        checkpoint_file: Fichier pour sauvegarder les résultats intermédiaires
        
    Returns:
        dict: Résultats des différentes configurations testées
    """
    # Vérifier s'il existe déjà un checkpoint
    if os.path.exists(checkpoint_file):
        try:
            print(f"Chargement des résultats d'optimisation intermédiaires depuis {checkpoint_file}...")
            resultats = joblib.load(checkpoint_file)
            print(f"Reprise de l'optimisation à partir de {len(resultats)} configurations déjà testées.")
        except Exception as e:
            print(f"Échec du chargement du checkpoint: {e}")
            resultats = []
    else:
        resultats = []
    
    # Si n_samples est spécifié, échantillonner les données
    if n_samples is not None and n_samples < len(embeddings):
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings_sample = embeddings[indices]
        labels_sample = labels[indices] if labels is not None else None
    else:
        embeddings_sample = embeddings
        labels_sample = labels
    
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
    
    # Exclure les combinaisons déjà testées
    combinaisons_restantes = []
    for params in combinaisons:
        if not any(r['params'] == params for r in resultats):
            combinaisons_restantes.append(params)
    
    print(f"Évaluation de {len(combinaisons_restantes)} combinaisons de paramètres restantes...")
    
    # Exécuter HDBSCAN pour chaque combinaison de paramètres
    for i, params in enumerate(tqdm(combinaisons_restantes)):
        # Utiliser les paramètres sans ajouter prediction_data pour cette phase
        params_current = params.copy()  
        
        clusterer = hdbscan.HDBSCAN(**params_current)
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
            # Correction: s'assurer qu'il y a au moins 2 points par cluster et au moins 2 clusters distincts
            if mask.sum() >= n_clusters * 2 and len(np.unique(clusters[mask])) >= 2:
                try:
                    silhouette = silhouette_score(
                        embeddings_sample[mask], 
                        clusters[mask]
                    )
                    metriques['silhouette'] = silhouette
                except Exception as e:
                    print(f"Erreur lors du calcul du score de silhouette: {e}")
                    metriques['silhouette'] = float('nan')
        
        # ARI et AMI (seulement si des labels sont fournis)
        if labels_sample is not None:
            mask = clusters != -1
            if mask.sum() > 0:
                try:
                    ari = adjusted_rand_score(labels_sample[mask], clusters[mask])
                    ami = adjusted_mutual_info_score(labels_sample[mask], clusters[mask])
                    metriques['ari'] = ari
                    metriques['ami'] = ami
                except Exception as e:
                    print(f"Erreur lors du calcul de l'ARI ou AMI: {e}")
                    metriques['ari'] = float('nan')
                    metriques['ami'] = float('nan')
        
        # Stocker les résultats
        resultats.append({
            'params': params,
            'metrics': metriques,
            'clusters': clusters.tolist() if n_clusters > 0 else []
        })
        
        # Sauvegarder les résultats intermédiaires toutes les 5 configurations
        if (i + 1) % 5 == 0:
            joblib.dump(resultats, checkpoint_file)
    
    # Sauvegarder les résultats finaux
    joblib.dump(resultats, checkpoint_file)
    
    # Trier les résultats en fonction d'un critère mixte: nombre de clusters et score de silhouette
    # Priorité aux configurations avec au moins 8 clusters (proche du nombre de causes)
    # Pour celles qui ont suffisamment de clusters, trier par silhouette décroissante
    resultats_tries = sorted(
        resultats, 
        key=lambda x: (
            1 if x['metrics'].get('n_clusters', 0) >= 8 else 0,  # Priorité aux configs avec ≥ 8 clusters
            x['metrics'].get('silhouette', -1)  # Puis par score de silhouette
        ), 
        reverse=True
    )
    
    return resultats_tries

def visualiser_resultats_optimisation(resultats_tries, n_top=30):
    """
    Visualise les résultats de l'optimisation des hyperparamètres
    
    Args:
        resultats_tries: Résultats triés de la grille de recherche
        n_top: Nombre de configurations à afficher
    """
    # Limiter aux n_top meilleures configurations
    resultats_top = resultats_tries[:n_top]
    
    # Extraire les données pour visualisation
    silhouettes = [r['metrics'].get('silhouette', 0) for r in resultats_top if 'silhouette' in r['metrics']]
    n_clusters = [r['metrics']['n_clusters'] for r in resultats_top]
    percent_noise = [r['metrics']['percent_noise'] for r in resultats_top]
    
    # S'assurer qu'il y a des données à visualiser
    if len(silhouettes) == 0 or len(n_clusters) == 0 or len(percent_noise) == 0:
        print("Pas assez de données pour visualiser les résultats.")
        return
    
    # Créer une figure avec plusieurs sous-graphiques
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    
    # Graphique des scores de silhouette
    ax1 = sns.barplot(x=list(range(len(silhouettes))), y=silhouettes, ax=axes[0])
    axes[0].set_title('Scores de silhouette pour les meilleures configurations')
    axes[0].set_xlabel('Configuration (index)')
    axes[0].set_ylabel('Score de silhouette')
    
    # Ajouter les valeurs sur les barres
    for i, bar in enumerate(ax1.patches):
        if i < len(silhouettes):
            ax1.text(
                bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 0.01,
                f"{silhouettes[i]:.3f}",
                ha='center'
            )
    
    # Graphique du nombre de clusters
    ax2 = sns.barplot(x=list(range(len(n_clusters))), y=n_clusters, ax=axes[1])
    axes[1].set_title('Nombre de clusters pour les meilleures configurations')
    axes[1].set_xlabel('Configuration (index)')
    axes[1].set_ylabel('Nombre de clusters')
    
    # Ajouter les valeurs sur les barres
    for i, bar in enumerate(ax2.patches):
        if i < len(n_clusters):
            axes[1].text(
                bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 0.2,
                f"{n_clusters[i]}",
                ha='center'
            )
    
    # Graphique du pourcentage de bruit
    ax3 = sns.barplot(x=list(range(len(percent_noise))), y=percent_noise, ax=axes[2])
    axes[2].set_title('Pourcentage de points classés comme bruit')
    axes[2].set_xlabel('Configuration (index)')
    axes[2].set_ylabel('% de points de bruit')
    
    # Ajouter les valeurs sur les barres
    for i, bar in enumerate(ax3.patches):
        if i < len(percent_noise):
            axes[2].text(
                bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 0.5,
                f"{percent_noise[i]:.1f}%",
                ha='center'
            )
    
    plt.tight_layout()
    plt.savefig('optimisation_hyperparametres.png', dpi=300)
    plt.show()
    
    # Afficher les meilleurs paramètres
    print("\nTop 5 configurations:")
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
        
    Returns:
        ndarray: Embeddings 2D pour utilisation ultérieure
    """
    # Réduction dimensionnelle si nécessaire
    if embeddings.shape[1] > 2:
        print("Réduction dimensionnelle avec UMAP...")
        reducer = UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=15,
            min_dist=0.1
        )
        embeddings_2d = reducer.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings
    
    # Créer une figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Visualisation des clusters
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    scatter1 = ax1.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=clusters,
        cmap='tab20',
        s=40,
        alpha=0.8
    )
    ax1.set_title(f"Clusters détectés par HDBSCAN ({n_clusters} clusters)")
    ax1.set_xlabel('UMAP Dimension 1')
    ax1.set_ylabel('UMAP Dimension 2')
    plt.colorbar(scatter1, ax=ax1, label='Cluster')
    
    # Visualisation des causes réelles (si disponibles)
    if labels is not None:
        scatter2 = ax2.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            cmap='tab20',
            s=40,
            alpha=0.8
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
        
        # Vérifier si le cluster contient des tickets
        if len(indices_cluster) == 0:
            continue
            
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
    
    # Vérifier que tous les clusters existent dans composition
    cluster_ids = [i for i in cluster_ids if i in composition]
    
    # Si aucun cluster valide n'est disponible
    if not cluster_ids:
        print("Aucun cluster valide à visualiser.")
        return
    
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
    
    # Trier par taille de cluster décroissante
    df_viz = df_viz.sort_values('taille', ascending=False)
    
    # Créer une figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16))
    
    # Homogénéité des clusters
    bars1 = sns.barplot(x='cluster_id', y='homogeneite', data=df_viz, ax=ax1)
    ax1.set_title('Homogénéité des clusters (% de la cause majoritaire)')
    ax1.set_xlabel('Cluster ID')
    ax1.set_ylabel('Homogénéité (%)')
    ax1.set_ylim(0, 100)
    
    # Ajouter les valeurs sur les barres
    for i, bar in enumerate(bars1.patches):
        if i < len(df_viz):
            ax1.text(
                bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 1,
                f"{df_viz.iloc[i]['homogeneite']:.1f}%",
                ha='center'
            )
    
    # Taille des clusters
    bars2 = sns.barplot(x='cluster_id', y='taille', data=df_viz, ax=ax2)
    ax2.set_title('Taille des clusters')
    ax2.set_xlabel('Cluster ID')
    ax2.set_ylabel('Nombre de tickets')
    
    # Ajouter les causes majoritaires comme étiquettes
    for i, bar in enumerate(bars2.patches):
        if i < len(df_viz):
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
    
    # Afficher un résumé textuel
    print("\nRésumé de la composition des clusters:")
    for i, row in df_viz.iterrows():
        cluster_id = row['cluster_id']
        print(f"Cluster {cluster_id}: {row['taille']} tickets, {row['cause_majoritaire']} ({row['homogeneite']:.1f}% d'homogénéité)")
    
    # Si le cluster de bruit (-1) existe, l'afficher également
    if -1 in composition:
        bruit = composition[-1]
        print(f"Points de bruit: {bruit['taille']} tickets, principalement {bruit['cause_majoritaire']} ({bruit['homogeneite']:.1f}% d'homogénéité)")

# =========== APPROCHE HIÉRARCHIQUE (CAUSE PUIS SOUS-CAUSE) ===========
def clustering_deux_niveaux(embeddings, df_fiable, params_hdbscan, min_tickets_pour_souscauses=5):
    """
    Implémente une approche hiérarchique en deux niveaux:
    1. Clusteriser pour identifier les causes principales
    2. Pour chaque cause, clusteriser pour identifier les sous-causes
    
    Args:
        embeddings: Embeddings enrichis des tickets
        df_fiable: DataFrame des tickets fiables
        params_hdbscan: Paramètres optimaux pour HDBSCAN
        min_tickets_pour_souscauses: Nombre minimum de tickets requis pour le clustering des sous-causes
        
    Returns:
        dict: Modèles HDBSCAN pour chaque niveau
    """
    print("Application de l'approche à deux niveaux...")
    
    # Ajouter prediction_data=True pour permettre la prédiction future
    params_with_pred = params_hdbscan.copy()
    params_with_pred['prediction_data'] = True
    
    # Niveau 1: Clustering pour les causes principales
    print("Niveau 1: Clustering pour causes principales...")
    clusterer_niveau1 = hdbscan.HDBSCAN(**params_with_pred)
    clusters_niveau1 = clusterer_niveau1.fit_predict(embeddings)
    
    # Analyser la composition des clusters de niveau 1
    n_clusters_niveau1 = len(set(clusters_niveau1)) - (1 if -1 in clusters_niveau1 else 0)
    composition_niveau1 = analyser_composition_clusters(clusters_niveau1, df_fiable)
    
    print(f"Nombre de clusters de niveau 1 détectés: {n_clusters_niveau1}")
    
    # Visualiser la composition des clusters de niveau 1
    visualiser_composition_clusters(composition_niveau1, n_clusters_niveau1)
    
    # Mapping des clusters vers les causes
    # On attribue à chaque cluster la cause majoritaire de ses tickets
    mapping_cluster_cause = {}
    for cluster_id in range(-1, n_clusters_niveau1):
        if cluster_id in composition_niveau1:
            mapping_cluster_cause[cluster_id] = composition_niveau1[cluster_id]['cause_majoritaire']
    
    # Niveau 2: Pour chaque cause, clustering pour les sous-causes
    print("\nNiveau 2: Clustering pour sous-causes...")
    clusterers_niveau2 = {}
    mapping_cluster_souscause = {}
    
    # Obtenir toutes les causes uniques
    causes_uniques = df_fiable['cause'].unique()
    
    for cause in causes_uniques:
        # Tickets de cette cause
        mask_cause = df_fiable['cause'] == cause
        indices_cause = df_fiable[mask_cause].index.values
        nb_tickets = len(indices_cause)
        
        if nb_tickets >= min_tickets_pour_souscauses:
            print(f"Clustering pour la cause '{cause}' ({nb_tickets} tickets)...")
            
            # Extraire les embeddings correspondants
            embeddings_cause = embeddings[indices_cause]
            
            # Ajuster les paramètres en fonction du nombre de tickets
            params_niveau2 = params_with_pred.copy()
            params_niveau2['min_cluster_size'] = min(params_with_pred['min_cluster_size'], max(3, nb_tickets // 10))
            
            # Appliquer HDBSCAN pour cette cause
            clusterer_cause = hdbscan.HDBSCAN(**params_niveau2)
            clusters_cause = clusterer_cause.fit_predict(embeddings_cause)
            
            # Nombre de clusters détectés
            n_clusters_cause = len(set(clusters_cause)) - (1 if -1 in clusters_cause else 0)
            print(f"  - {n_clusters_cause} clusters de sous-causes détectés")
            
            # Analyser la composition des clusters en termes de sous-causes
            df_cause = df_fiable.iloc[indices_cause]
            composition_souscauses = analyser_composition_clusters(clusters_cause, df_cause)
            
            # Mapping des clusters de niveau 2 vers les sous-causes
            mapping_souscause = {}
            for sous_cluster_id in range(-1, n_clusters_cause):
                if sous_cluster_id in composition_souscauses:
                    # On utilise la sous-cause majoritaire
                    sous_causes = Counter(df_cause.iloc[np.where(clusters_cause == sous_cluster_id)[0]]['souscause'])
                    mapping_souscause[sous_cluster_id] = sous_causes.most_common(1)[0][0] if sous_causes else "Indéterminée"
            
            # Stocker le clusterer et le mapping
            clusterers_niveau2[cause] = clusterer_cause
            mapping_cluster_souscause[cause] = mapping_souscause
        else:
            print(f"Pas assez de tickets pour la cause '{cause}' ({nb_tickets} tickets) - seuil min: {min_tickets_pour_souscauses}")
    
    return {
        'niveau1': clusterer_niveau1,
        'niveau2': clusterers_niveau2,
        'mapping_cluster_cause': mapping_cluster_cause,
        'mapping_cluster_souscause': mapping_cluster_souscause,
        'clusters_niveau1': clusters_niveau1
    }

# =========== MODÈLE DE CLASSIFICATION FINAL ===========
def creer_modele_final(clusterers, mapping_cluster_cause, mapping_cluster_souscause=None):
    """
    Crée un modèle final pour prédire les causes et sous-causes
    
    Args:
        clusterers: Dict contenant les modèles HDBSCAN pour les niveaux 1 et 2
        mapping_cluster_cause: Mapping des clusters de niveau 1 vers les causes
        mapping_cluster_souscause: Mapping des clusters de niveau 2 vers les sous-causes
        
    Returns:
        dict: Modèle final avec toutes les informations nécessaires
    """
    return {
        'clusterers': clusterers,
        'mapping_cluster_cause': mapping_cluster_cause,
        'mapping_cluster_souscause': mapping_cluster_souscause
    }

def predire_cause_avec_exemplars(clusterer, mapping, embeddings, cause_defaut="Non déterminée"):
    """
    Utilise les exemplars de HDBSCAN pour prédire la cause d'un ticket
    
    Args:
        clusterer: Modèle HDBSCAN entraîné
        mapping: Dictionnaire de mapping cluster -> cause
        embeddings: Embeddings des tickets à prédire
        cause_defaut: Cause à retourner en cas d'échec
        
    Returns:
        list: Liste des causes prédites
    """
    predictions = []
    
    # Vérifier que le clusterer a des exemplars
    if not hasattr(clusterer, 'exemplars_') or len(clusterer.exemplars_) == 0:
        print("Erreur: Le clusterer n'a pas d'exemplars.")
        return [cause_defaut] * len(embeddings)
    
    # Pour chaque ticket
    for embedding in embeddings:
        # Calculer la distance à chaque exemplar
        distances = []
        for cluster_id, exemplars in enumerate(clusterer.exemplars_):
            if exemplars.shape[0] > 0:  # S'assurer qu'il y a des exemplars
                # Distance minimale à un exemplar de ce cluster
                dist = np.min([np.linalg.norm(embedding - exemplar) for exemplar in exemplars])
                distances.append((cluster_id, dist))
        
        if distances:
            # Trouver le cluster avec la distance minimale
            closest_cluster, _ = min(distances, key=lambda x: x[1])
            # Obtenir la cause correspondante
            if closest_cluster in mapping:
                predictions.append(mapping[closest_cluster])
            else:
                predictions.append(cause_defaut)
        else:
            predictions.append(cause_defaut)
    
    return predictions

def predire_causes_sous_causes(modele, embeddings):
    """
    Utilise le modèle final pour prédire les causes et sous-causes de nouveaux tickets
    
    Args:
        modele: Modèle final créé par creer_modele_final
        embeddings: Embeddings des nouveaux tickets
        
    Returns:
        tuple: (causes prédites, sous-causes prédites)
    """
    # Extraire les composants du modèle
    clusterer_niveau1 = modele['clusterers']['niveau1']
    clusterers_niveau2 = modele['clusterers']['niveau2']
    mapping_cluster_cause = modele['mapping_cluster_cause']
    mapping_cluster_souscause = modele.get('mapping_cluster_souscause', {})
    
    # Prédiction des causes de niveau 1
    try:
        # Essayer d'utiliser la méthode membership_vector pour obtenir le cluster le plus proche
        membership_vectors = clusterer_niveau1.membership_vector(embeddings)
        clusters_niveau1 = np.argmax(membership_vectors, axis=1)
        
        # Convertir en liste et assigner -1 aux points avec une appartenance maximale faible
        clusters_niveau1 = [-1 if np.max(membership_vectors[i]) < 0.1 else int(clusters_niveau1[i]) 
                           for i in range(len(clusters_niveau1))]
    except Exception as e:
        print(f"Erreur avec membership_vector: {e}")
        try:
            # Essayer avec approximate_predict
            clusters_niveau1, _ = hdbscan.approximate_predict(clusterer_niveau1, embeddings)
            clusters_niveau1 = clusters_niveau1.tolist()
        except Exception as e:
            print(f"Erreur avec approximate_predict: {e}")
            # Utiliser une approche basée sur les distances aux points existants
            print("Utilisation d'une approche basée sur les distances...")
            
            # Obtenir les labels des clusters existants
            labels = clusterer_niveau1.labels_
            original_data = clusterer_niveau1._raw_data
            
            clusters_niveau1 = []
            for emb in tqdm(embeddings, desc="Prédiction par distance"):
                # Calculer les distances à tous les points d'entraînement
                distances = np.linalg.norm(original_data - emb.reshape(1, -1), axis=1)
                
                # Trouver les k plus proches voisins
                k = min(10, len(distances))
                nearest_indices = np.argsort(distances)[:k]
                nearest_labels = labels[nearest_indices]
                
                # Attribuer au cluster le plus fréquent parmi les voisins (excluant le bruit)
                valid_labels = nearest_labels[nearest_labels >= 0]
                if len(valid_labels) > 0:
                    # Comptage des occurrences de chaque label
                    unique_labels, counts = np.unique(valid_labels, return_counts=True)
                    # Prendre le label le plus fréquent
                    cluster_id = unique_labels[np.argmax(counts)]
                else:
                    # Si tous les voisins sont du bruit, assigner au bruit
                    cluster_id = -1
                
                clusters_niveau1.append(cluster_id)
    
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
        if cause in clusterers_niveau2 and cause != "Non déterminée":
            clusterer_niveau2 = clusterers_niveau2[cause]
            
            try:
                # Essayer avec membership_vector
                membership_vect = clusterer_niveau2.membership_vector([embeddings[i]])
                sous_cluster = np.argmax(membership_vect[0])
                
                # Vérifier si l'appartenance est suffisante
                if np.max(membership_vect[0]) < 0.1:
                    sous_cluster = -1
            except Exception as e:
                try:
                    # Essayer avec approximate_predict
                    sous_cluster, _ = hdbscan.approximate_predict(clusterer_niveau2, [embeddings[i]])
                    sous_cluster = sous_cluster[0]
                except Exception as e:
                    # Utiliser une approche basée sur les distances
                    if hasattr(clusterer_niveau2, '_raw_data'):
                        original_data_niveau2 = clusterer_niveau2._raw_data
                        labels_niveau2 = clusterer_niveau2.labels_
                        
                        # Calculer les distances
                        distances = np.linalg.norm(original_data_niveau2 - embeddings[i].reshape(1, -1), axis=1)
                        
                        # Trouver les k plus proches voisins
                        k = min(5, len(original_data_niveau2))
                        nearest_indices = np.argsort(distances)[:k]
                        nearest_labels = labels_niveau2[nearest_indices]
                        
                        # Attribuer au cluster le plus fréquent
                        valid_labels = nearest_labels[nearest_labels >= 0]
                        if len(valid_labels) > 0:
                            unique_labels, counts = np.unique(valid_labels, return_counts=True)
                            sous_cluster = unique_labels[np.argmax(counts)]
                        else:
                            sous_cluster = -1
                    else:
                        sous_cluster = -1
            
            # Obtenir la sous-cause correspondante
            if cause in mapping_cluster_souscause and sous_cluster in mapping_cluster_souscause[cause]:
                sous_cause = mapping_cluster_souscause[cause][sous_cluster]
            else:
                sous_cause = "Non déterminée"
        else:
            sous_cause = "Non déterminée"
        
        sous_causes_predites.append(sous_cause)
    
    return causes_predites, sous_causes_predites

def evaluer_predictions(causes_reelles, causes_predites, sous_causes_reelles=None, sous_causes_predites=None):
    """
    Évalue les prédictions du modèle
    
    Args:
        causes_reelles: Causes réelles des tickets
        causes_predites: Causes prédites par le modèle
        sous_causes_reelles: Sous-causes réelles (optionnel)
        sous_causes_predites: Sous-causes prédites (optionnel)
        
    Returns:
        dict: Métriques d'évaluation
    """
    # Calculer l'exactitude pour les causes
    exactitude_causes = sum(r == p for r, p in zip(causes_reelles, causes_predites)) / len(causes_reelles)
    
    # Calculer la matrice de confusion pour les causes
    causes_uniques = sorted(list(set(causes_reelles) | set(causes_predites)))
    matrice_confusion = {}
    
    for cause_reelle in causes_uniques:
        matrice_confusion[cause_reelle] = {}
        for cause_predite in causes_uniques:
            count = sum(1 for r, p in zip(causes_reelles, causes_predites) 
                         if r == cause_reelle and p == cause_predite)
            matrice_confusion[cause_reelle][cause_predite] = count
    
    # Calcul des métriques par cause
    metriques_par_cause = {}
    
    for cause in causes_uniques:
        # Convertir en format binaire pour chaque cause
        y_true_binary = [1 if r == cause else 0 for r in causes_reelles]
        y_pred_binary = [1 if p == cause else 0 for p in causes_predites]
        
        # Calculer précision, rappel et F1
        try:
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            rappel = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            
            metriques_par_cause[cause] = {
                'precision': precision,
                'rappel': rappel,
                'f1': f1,
                'support': sum(y_true_binary)
            }
        except Exception as e:
            print(f"Erreur lors du calcul des métriques pour la cause '{cause}': {e}")
    
    # Évaluation des sous-causes si disponibles
    resultats_sous_causes = None
    if sous_causes_reelles is not None and sous_causes_predites is not None:
        exactitude_sous_causes = sum(r == p for r, p in zip(sous_causes_reelles, sous_causes_predites)) / len(sous_causes_reelles)
        
        resultats_sous_causes = {
            'exactitude': exactitude_sous_causes
        }
    
    return {
        'exactitude': exactitude_causes,
        'matrice_confusion': matrice_confusion,
        'metriques_par_cause': metriques_par_cause,
        'resultats_sous_causes': resultats_sous_causes
    }

def visualiser_matrice_confusion(matrice_confusion):
    """
    Visualise la matrice de confusion
    
    Args:
        matrice_confusion: Matrice de confusion calculée par evaluer_predictions
    """
    # Convertir la matrice de confusion en DataFrame
    causes = sorted(matrice_confusion.keys())
    df_confusion = pd.DataFrame(index=causes, columns=causes, dtype=float)
    
    # Remplir avec des zéros d'abord pour éviter les NaN
    df_confusion.fillna(0.0, inplace=True)
    
    for cause_reelle in causes:
        for cause_predite in causes:
            df_confusion.loc[cause_reelle, cause_predite] = float(matrice_confusion[cause_reelle].get(cause_predite, 0))
    
    # Créer une copie pour les annotations qui seront en format entier
    df_annot = df_confusion.copy().astype(int)
    
    # Normaliser par ligne en évitant les divisions par zéro
    df_confusion_norm = df_confusion.copy()
    row_sums = df_confusion.sum(axis=1)
    
    for idx in df_confusion_norm.index:
        if row_sums[idx] > 0:
            df_confusion_norm.loc[idx, :] = df_confusion_norm.loc[idx, :] / row_sums[idx]
    
    # S'assurer que toutes les valeurs sont numériques
    df_confusion_norm = df_confusion_norm.astype(float)
    
    # Visualiser
    plt.figure(figsize=(14, 12))
    
    # Utiliser df_annot pour les annotations (format entier) et df_confusion_norm pour les couleurs
    sns.heatmap(df_confusion_norm, annot=df_annot, fmt='d', cmap='YlGnBu')
    plt.title('Matrice de confusion')
    plt.xlabel('Cause prédite')
    plt.ylabel('Cause réelle')
    plt.tight_layout()
    plt.savefig('matrice_confusion.png', dpi=300)
    plt.show()

def visualiser_metriques_par_cause(metriques_par_cause):
    """
    Visualise les métriques par cause
    
    Args:
        metriques_par_cause: Dictionnaire des métriques par cause
    """
    # Préparer les données
    causes = list(metriques_par_cause.keys())
    precisions = [metriques_par_cause[c]['precision'] for c in causes]
    rappels = [metriques_par_cause[c]['rappel'] for c in causes]
    f1_scores = [metriques_par_cause[c]['f1'] for c in causes]
    supports = [metriques_par_cause[c]['support'] for c in causes]
    
    # Créer un DataFrame
    df_metriques = pd.DataFrame({
        'Cause': causes,
        'Précision': precisions,
        'Rappel': rappels,
        'F1-Score': f1_scores,
        'Support': supports
    })
    
    # Trier par support (nombre d'exemples) décroissant
    df_metriques = df_metriques.sort_values('Support', ascending=False)
    
    # Créer un graphique à barres avec précision, rappel et F1
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Positions des barres
    x = np.arange(len(df_metriques))
    width = 0.25
    
    # Tracer les barres
    bars1 = ax.bar(x - width, df_metriques['Précision'], width, label='Précision')
    bars2 = ax.bar(x, df_metriques['Rappel'], width, label='Rappel')
    bars3 = ax.bar(x + width, df_metriques['F1-Score'], width, label='F1-Score')
    
    # Configurer l'axe X
    ax.set_xlabel('Cause')
    ax.set_xticks(x)
    ax.set_xticklabels(df_metriques['Cause'], rotation=45, ha='right')
    
    # Configurer l'axe Y
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    
    # Ajouter une légende
    ax.legend()
    
    # Ajouter les valeurs du support
    for i, support in enumerate(df_metriques['Support']):
        ax.text(i, 0.05, f"n={support}", ha='center', va='bottom', fontsize=9)
    
    # Ajouter un titre
    plt.title('Précision, Rappel et F1-Score par cause')
    
    plt.tight_layout()
    plt.savefig('metriques_par_cause.png', dpi=300)
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
    # Créer dossiers pour les résultats et modèles
    os.makedirs("resultats", exist_ok=True)
    os.makedirs("modeles", exist_ok=True)
    
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
        # Grille de paramètres étendue et diversifiée
        param_grid = {
            'min_cluster_size': [5, 8, 10, 15, 20],
            'min_samples': [1, 2, 4, 6, 8],
            'cluster_selection_epsilon': [0.0, 0.25, 0.5, 0.75, 1.0],
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
        # Paramètres basés sur les explorations précédentes
        meilleurs_params = {
            'min_cluster_size': 8,
            'min_samples': 2,
            'cluster_selection_epsilon': 0.25,
            'metric': 'euclidean'
        }
        print(f"\nUtilisation des paramètres par défaut: {meilleurs_params}")
    
    # 3. Appliquer HDBSCAN avec les meilleurs paramètres
    print("\nApplication de HDBSCAN avec les paramètres optimaux...")
    
    # Ajouter prediction_data=True pour permettre la prédiction future
    params_with_pred = meilleurs_params.copy()
    params_with_pred['prediction_data'] = True
    clusterer = hdbscan.HDBSCAN(**params_with_pred)
    clusters = clusterer.fit_predict(embeddings_enrichis)
    
    # Métriques de qualité
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = (clusters == -1).sum()
    print(f"Nombre de clusters détectés: {n_clusters}")
    print(f"Nombre de points classés comme bruit: {n_noise} ({n_noise/len(clusters)*100:.2f}%)")
    
    # Silhouette score
    if n_clusters > 1:
        mask = clusters != -1
        if mask.sum() >= n_clusters * 2 and len(np.unique(clusters[mask])) >= 2:
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
        clusterers['mapping_cluster_cause'],
        clusterers.get('mapping_cluster_souscause')
    )
    
    # 7. Évaluation du modèle sur les données fiables (validation)
    print("\nÉvaluation du modèle final...")
    
    # Validation croisée simple (80% entraînement, 20% test)
    # Pour une évaluation plus robuste, on pourrait utiliser k-fold
    train_indices, test_indices = train_test_split(
        np.arange(len(df_fiable)), 
        test_size=0.2, 
        random_state=42, 
        stratify=df_fiable['cause'] if len(df_fiable['cause'].unique()) > 1 else None
    )
    
    # Extraire les données de test
    test_embeddings = embeddings_enrichis[test_indices]
    test_causes = df_fiable.iloc[test_indices]['cause'].values
    test_souscauses = df_fiable.iloc[test_indices]['souscause'].values
    
    # Prédiction sur les données de test
    causes_predites, sous_causes_predites = predire_causes_sous_causes(modele_final, test_embeddings)
    
    # Évaluation des prédictions
    resultats_evaluation = evaluer_predictions(
        test_causes,
        causes_predites,
        test_souscauses,
        sous_causes_predites
    )
    
    print(f"Exactitude (causes): {resultats_evaluation['exactitude']*100:.2f}%")
    if resultats_evaluation['resultats_sous_causes']:
        print(f"Exactitude (sous-causes): {resultats_evaluation['resultats_sous_causes']['exactitude']*100:.2f}%")
    
    # Visualiser la matrice de confusion
    visualiser_matrice_confusion(resultats_evaluation['matrice_confusion'])
    
    # Visualiser les métriques par cause
    visualiser_metriques_par_cause(resultats_evaluation['metriques_par_cause'])
    
    # 8. Sauvegarder le modèle final
    print("\nSauvegarde du modèle final...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chemin_modele = f"modeles/modele_clustering_metis_{timestamp}.joblib"
    joblib.dump(modele_final, chemin_modele)
    
    # Créer un fichier de métadonnées avec les informations sur le modèle
    with open(f"modeles/modele_clustering_metis_{timestamp}_info.txt", "w") as f:
        f.write(f"Modèle de clustering METIS - {timestamp}\n")
        f.write(f"Paramètres HDBSCAN: {meilleurs_params}\n")
        f.write(f"Nombre de clusters niveau 1: {n_clusters}\n")
        f.write(f"Exactitude (causes): {resultats_evaluation['exactitude']*100:.2f}%\n")
        if resultats_evaluation['resultats_sous_causes']:
            f.write(f"Exactitude (sous-causes): {resultats_evaluation['resultats_sous_causes']['exactitude']*100:.2f}%\n")
        f.write("\nMapping clusters -> causes:\n")
        for cluster, cause in clusterers['mapping_cluster_cause'].items():
            f.write(f"  Cluster {cluster} -> {cause}\n")
    
    print(f"Modèle sauvegardé sous: {chemin_modele}")
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
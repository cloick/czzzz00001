# =========== IMPORTATIONS NÉCESSAIRES ===========
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import hdbscan
from umap import UMAP
from collections import Counter
import joblib
from tqdm import tqdm
import time
from datetime import datetime

# =========== CHARGEMENT DES RÉSULTATS DE LA PHASE 1 ===========

def charger_resultats_phase1(chemin_resultats="resultats"):
    """
    Charge les résultats de la Phase 1 à partir des fichiers sauvegardés.
    
    Args:
        chemin_resultats: Chemin vers le dossier contenant les résultats de la Phase 1
        
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
        resultats['embeddings_enrichis'] = np.load(f"{chemin_resultats}/embeddings_enrichis.npy")
        print("Embeddings et clusters chargés avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement des embeddings: {e}")
        return None
    
    return resultats

# =========== PRÉPARATION DES DONNÉES CATÉGORIELLES ===========

def preparer_donnees_categorielles(df):
    """
    Prépare un DataFrame ne contenant que les variables catégorielles.
    
    Args:
        df: DataFrame contenant les données
        
    Returns:
        DataFrame: DataFrame ne contenant que les variables catégorielles encodées
    """
    # Liste des variables catégorielles
    cat_vars = ['Priorité', 'Service métier', 'Cat1', 'Cat2', 'Groupe affecté']
    
    # Créer un nouveau DataFrame pour les variables catégorielles
    df_cat = df[cat_vars].copy()
    
    # Encoder les variables catégorielles
    for col in cat_vars:
        if col in df_cat.columns:
            le = LabelEncoder()
            df_cat[f'{col}_encoded'] = le.fit_transform(df_cat[col].fillna('INCONNU'))
    
    # Ne conserver que les colonnes encodées
    df_cat = df_cat[[f'{col}_encoded' for col in cat_vars if f'{col}_encoded' in df_cat.columns]]
    
    # Standardiser les variables catégorielles encodées
    scaler = StandardScaler()
    df_cat_scaled = pd.DataFrame(
        scaler.fit_transform(df_cat),
        columns=df_cat.columns
    )
    
    return df_cat_scaled

# =========== OPTIMISATION DES HYPERPARAMÈTRES HDBSCAN ===========

def grille_recherche_hdbscan(embeddings, labels, param_grid, n_samples=None, checkpoint_file="resultats/optimisation_checkpoint.pkl"):
    """
    Effectue une recherche en grille pour les hyperparamètres de HDBSCAN.
    
    Args:
        embeddings: Embeddings ou caractéristiques catégorielles standardisées
        labels: Labels réels (causes) pour évaluation
        param_grid: Dict avec les valeurs des hyperparamètres à tester
        n_samples: Nombre d'échantillons à utiliser (None = tous)
        checkpoint_file: Fichier pour sauvegarder les résultats intermédiaires
        
    Returns:
        list: Résultats des différentes configurations testées, triés selon plusieurs critères
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
        # Ne pas inclure prediction_data pour la grille de recherche (plus rapide)
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
            # Vérifier qu'il y a au moins 2 points par cluster en moyenne
            if mask.sum() >= n_clusters * 2:
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
                ari = adjusted_rand_score(labels_sample[mask], clusters[mask])
                ami = adjusted_mutual_info_score(labels_sample[mask], clusters[mask])
                metriques['ari'] = ari
                metriques['ami'] = ami
        
        # Stocker les résultats
        resultats.append({
            'params': params,
            'metrics': metriques
        })
        
        # Sauvegarder les résultats intermédiaires toutes les 5 configurations
        if (i + 1) % 5 == 0:
            joblib.dump(resultats, checkpoint_file)
    
    # Sauvegarder les résultats finaux
    joblib.dump(resultats, checkpoint_file)
    
    # Filtrer les résultats avec un nombre suffisant de clusters (au moins 5)
    resultats_filtrés = [r for r in resultats if r['metrics']['n_clusters'] >= 5]
    
    # Si aucun résultat ne satisfait le critère, utiliser tous les résultats
    if not resultats_filtrés:
        resultats_filtrés = resultats
        print("Attention: Aucune configuration n'a produit au moins 5 clusters.")
    
    # Trier par score de silhouette décroissant parmi les configurations avec suffisamment de clusters
    resultats_tries = sorted(
        resultats_filtrés, 
        key=lambda x: x['metrics'].get('silhouette', -1), 
        reverse=True
    )
    
    return resultats_tries

def visualiser_resultats_optimisation(resultats_tries):
    """
    Visualise les résultats de l'optimisation des hyperparamètres.
    
    Args:
        resultats_tries: Résultats triés de la grille de recherche
    """
    # Extraire les données pour visualisation
    silhouettes = [r['metrics'].get('silhouette', 0) for r in resultats_tries if 'silhouette' in r['metrics']]
    n_clusters = [r['metrics']['n_clusters'] for r in resultats_tries]
    percent_noise = [r['metrics']['percent_noise'] for r in resultats_tries]
    
    # Limiter l'affichage aux 30 meilleures configurations
    max_configs = min(30, len(silhouettes))
    
    # Créer une figure avec plusieurs sous-graphiques
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    
    # Graphique des scores de silhouette
    bars1 = axes[0].bar(range(max_configs), silhouettes[:max_configs], color='royalblue')
    axes[0].set_title('Scores de silhouette pour les meilleures configurations', fontsize=14)
    axes[0].set_xlabel('Configuration (index)', fontsize=12)
    axes[0].set_ylabel('Score de silhouette', fontsize=12)
    # Ajouter les valeurs sur les barres
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        axes[0].annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    # Graphique du nombre de clusters
    bars2 = axes[1].bar(range(max_configs), n_clusters[:max_configs], color='seagreen')
    axes[1].set_title('Nombre de clusters pour les meilleures configurations', fontsize=14)
    axes[1].set_xlabel('Configuration (index)', fontsize=12)
    axes[1].set_ylabel('Nombre de clusters', fontsize=12)
    # Ajouter les valeurs sur les barres
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        axes[1].annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    # Graphique du pourcentage de bruit
    bars3 = axes[2].bar(range(max_configs), percent_noise[:max_configs], color='darkorange')
    axes[2].set_title('Pourcentage de points classés comme bruit', fontsize=14)
    axes[2].set_xlabel('Configuration (index)', fontsize=12)
    axes[2].set_ylabel('% de points de bruit', fontsize=12)
    # Ajouter les valeurs sur les barres
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        axes[2].annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
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
    Visualise les clusters formés par HDBSCAN avec les paramètres optimaux.
    
    Args:
        embeddings: Embeddings ou caractéristiques des tickets
        clusters: Étiquettes de cluster attribuées par HDBSCAN
        labels: Labels réels (causes) pour comparaison
        titre: Titre du graphique
        
    Returns:
        ndarray: Embeddings 2D pour utilisation ultérieure
    """
    # Réduction dimensionnelle si nécessaire
    if embeddings.shape[1] > 2:
        print("Réduction dimensionnelle avec UMAP...")
        reducer = UMAP(n_components=2, random_state=42, min_dist=0.1, n_neighbors=30)
        embeddings_2d = reducer.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings
    
    # Créer une figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    
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
    ax1.set_title(f"Clusters détectés par HDBSCAN ({n_clusters} clusters)", fontsize=16)
    ax1.set_xlabel('UMAP Dimension 1', fontsize=14)
    ax1.set_ylabel('UMAP Dimension 2', fontsize=14)
    
    # Ajouter une annotation avec les hyperparamètres utilisés
    ax1.text(0.01, 0.01, 
             f"Points de bruit: {(clusters == -1).sum()} ({(clusters == -1).sum()/len(clusters)*100:.1f}%)",
             transform=ax1.transAxes, fontsize=12, va='bottom', ha='left',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Créer une légende pour les clusters
    handles, labels_legend = scatter1.legend_elements(num=n_clusters)
    legend1 = ax1.legend(handles, [f'Cluster {i}' for i in range(n_clusters)], 
                         loc="upper right", title="Clusters")
    ax1.add_artist(legend1)
    
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
        ax2.set_title("Causes réelles", fontsize=16)
        ax2.set_xlabel('UMAP Dimension 1', fontsize=14)
        ax2.set_ylabel('UMAP Dimension 2', fontsize=14)
        
        # Créer une légende pour les causes réelles
        unique_labels = np.unique(labels)
        handles, labels_legend = scatter2.legend_elements(num=len(unique_labels))
        legend2 = ax2.legend(handles, [f'Cause {i}' for i in unique_labels], 
                          loc="upper right", title="Causes")
        ax2.add_artist(legend2)
    
    plt.tight_layout()
    plt.savefig(f"{titre.replace(' ', '_').lower()}.png", dpi=300)
    plt.show()
    
    return embeddings_2d

def analyser_composition_clusters(clusters, df_fiable):
    """
    Analyse la composition de chaque cluster en termes de causes.
    
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
    Visualise la composition des clusters en termes de causes.
    
    Args:
        composition: Dictionnaire avec la composition de chaque cluster
        n_clusters: Nombre de clusters (sans compter le bruit)
    """
    # Préparer les données pour le graphique
    cluster_ids = list(range(n_clusters))  # Exclure le cluster de bruit (-1)
    
    # Vérifier que tous les clusters existent dans composition
    cluster_ids = [i for i in cluster_ids if i in composition]
    
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
    
    # Homogénéité des clusters
    bars1 = sns.barplot(x='cluster_id', y='homogeneite', data=df_viz, ax=ax1, palette='viridis')
    ax1.set_title('Homogénéité des clusters (% de la cause majoritaire)', fontsize=16)
    ax1.set_xlabel('Cluster ID', fontsize=14)
    ax1.set_ylabel('Homogénéité (%)', fontsize=14)
    ax1.set_ylim(0, 100)
    
    # Ajouter les causes majoritaires comme annotations
    for i, bar in enumerate(bars1.patches):
        if i < len(df_viz):
            homogeneite = df_viz.iloc[i]['homogeneite']
            cause = df_viz.iloc[i]['cause_majoritaire']
            ax1.text(
                bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 1,
                f"{homogeneite:.1f}%\n({cause})",
                ha='center',
                va='bottom',
                fontsize=10,
                rotation=0,
                color='black'
            )
    
    # Taille des clusters
    bars2 = sns.barplot(x='cluster_id', y='taille', data=df_viz, ax=ax2, palette='plasma')
    ax2.set_title('Taille des clusters', fontsize=16)
    ax2.set_xlabel('Cluster ID', fontsize=14)
    ax2.set_ylabel('Nombre de tickets', fontsize=14)
    
    # Ajouter les tailles comme annotations
    for i, bar in enumerate(bars2.patches):
        if i < len(df_viz):
            taille = df_viz.iloc[i]['taille']
            ax2.text(
                bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 5,
                f"{taille}",
                ha='center',
                va='bottom',
                fontsize=10,
                rotation=0
            )
    
    plt.tight_layout()
    plt.savefig('composition_clusters.png', dpi=300)
    plt.show()
    
    # Retourner un mapping cluster -> cause majoritaire
    mapping_cluster_cause = {cluster_id: composition[cluster_id]['cause_majoritaire'] 
                             for cluster_id in range(n_clusters) 
                             if cluster_id in composition}
    
    return mapping_cluster_cause

# =========== MODÈLE DE CLASSIFICATION FINAL ===========

def creer_modele_final(reducer, clusterer, mapping_cluster_cause):
    """
    Crée un modèle final pour prédire les causes.
    
    Args:
        reducer: Réducteur UMAP utilisé pour la visualisation
        clusterer: Modèle HDBSCAN entraîné
        mapping_cluster_cause: Mapping des clusters vers les causes
        
    Returns:
        dict: Modèle final avec toutes les informations nécessaires
    """
    return {
        'reducer': reducer,
        'clusterer': clusterer,
        'mapping_cluster_cause': mapping_cluster_cause
    }

def predire_causes(modele, X_test):
    """
    Utilise le modèle final pour prédire les causes de nouveaux tickets.
    
    Args:
        modele: Modèle final créé par creer_modele_final
        X_test: Caractéristiques des nouveaux tickets
        
    Returns:
        list: Causes prédites pour chaque ticket
    """
    # Extraire les composants du modèle
    reducer = modele['reducer']
    clusterer = modele['clusterer']
    mapping_cluster_cause = modele['mapping_cluster_cause']
    
    # Réduire les dimensions pour la prédiction (si nécessaire)
    X_reduced = X_test
    if hasattr(reducer, 'transform'):
        X_reduced = reducer.transform(X_test)
    
    # Faire des prédictions avec HDBSCAN (s'il a la méthode de prédiction)
    if hasattr(clusterer, 'approximate_predict'):
        try:
            clusters, _ = clusterer.approximate_predict(X_reduced)
        except Exception as e:
            print(f"Erreur avec approximate_predict: {e}")
            # Utiliser le modèle pour faire une prédiction standard
            clusters = clusterer.fit_predict(X_reduced)
    else:
        # Utiliser le modèle pour faire une prédiction standard
        clusters = clusterer.fit_predict(X_reduced)
    
    # Convertir les clusters en causes
    causes_predites = []
    for cluster_id in clusters:
        if cluster_id in mapping_cluster_cause:
            cause = mapping_cluster_cause[cluster_id]
        else:
            # Utiliser la cause la plus fréquente pour les points de bruit
            # ou clusters sans correspondance
            cause = max(mapping_cluster_cause.values(), key=list(mapping_cluster_cause.values()).count)
        causes_predites.append(cause)
    
    return causes_predites

def evaluer_predictions(causes_reelles, causes_predites):
    """
    Évalue les prédictions du modèle avec des métriques détaillées.
    
    Args:
        causes_reelles: Causes réelles des tickets
        causes_predites: Causes prédites par le modèle
        
    Returns:
        dict: Métriques d'évaluation détaillées
    """
    # Calculer l'exactitude globale
    exactitude = accuracy_score(causes_reelles, causes_predites)
    
    # Calculer la matrice de confusion
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
    
    return {
        'exactitude': exactitude,
        'matrice_confusion': matrice_confusion,
        'metriques_par_cause': metriques_par_cause
    }

def visualiser_matrice_confusion(matrice_confusion):
    """
    Visualise la matrice de confusion.
    
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
    plt.figure(figsize=(16, 14))
    
    # Utiliser df_annot pour les annotations (format entier) et df_confusion_norm pour les couleurs
    heatmap = sns.heatmap(df_confusion_norm, annot=df_annot, fmt='d', cmap='YlGnBu')
    plt.title('Matrice de confusion', fontsize=16)
    plt.xlabel('Cause prédite', fontsize=14)
    plt.ylabel('Cause réelle', fontsize=14)
    
    # Améliorer la lisibilité de l'axe y
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig('matrice_confusion.png', dpi=300)
    plt.show()

def visualiser_metriques_par_cause(metriques_par_cause):
    """
    Visualise les métriques par cause.
    
    Args:
        metriques_par_cause: Dictionnaire des métriques par cause
    """
    # Préparer les données
    causes = list(metriques_par_cause.keys())
    precisions = [metriques_par_cause[c]['precision'] for c in causes]
    rappels = [metriques_par_cause[c]['rappel'] for c in causes]
    f1_scores = [metriques_par_cause[c]['f1'] for c in causes]
    supports = [metriques_par_cause[c]['support'] for c in causes]
    
    # Trier par support (nombre d'exemples)
    indices_tries = sorted(range(len(supports)), key=lambda i: supports[i], reverse=True)
    causes = [causes[i] for i in indices_tries]
    precisions = [precisions[i] for i in indices_tries]
    rappels = [rappels[i] for i in indices_tries]
    f1_scores = [f1_scores[i] for i in indices_tries]
    supports = [supports[i] for i in indices_tries]
    
    # Créer un graphique à barres avec précision, rappel et F1
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Positions des barres
    x = np.arange(len(causes))
    width = 0.25
    
    # Tracer les barres
    bars1 = ax.bar(x - width, precisions, width, label='Précision', color='royalblue')
    bars2 = ax.bar(x, rappels, width, label='Rappel', color='darkorange')
    bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='forestgreen')
    
    # Configurer l'axe X
    ax.set_xlabel('Cause', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(causes, rotation=45, ha='right', fontsize=12)
    
    # Configurer l'axe Y
    ax.set_ylabel('Score', fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ajouter une légende
    ax.legend(fontsize=12)
    
    # Ajouter les valeurs du support
    for i, support in enumerate(supports):
        ax.annotate(f"n={support}", 
                 xy=(i, 0.05), 
                 ha='center', 
                 va='bottom', 
                 fontsize=10,
                 color='dimgray')
    
    # Ajouter un titre
    plt.title('Précision, Rappel et F1-Score par cause', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('metriques_par_cause.png', dpi=300)
    plt.show()

# =========== FONCTION PRINCIPALE PHASE 2 ===========

def executer_phase2(resultats_phase1=None, optimiser=True, validation_croisee=True):
    """
    Exécute la Phase 2 complète: développement du modèle de clustering basé uniquement sur les
    variables catégorielles, sans utiliser le texte des notes de résolution.
    
    Args:
        resultats_phase1: Résultats chargés de la Phase 1 (None = charger automatiquement)
        optimiser: Si True, effectue l'optimisation des hyperparamètres
        validation_croisee: Si True, effectue une validation croisée simple 
        
    Returns:
        dict: Modèle final et résultats d'évaluation
    """
    print("\n=============== PHASE 2: DÉVELOPPEMENT DU MODÈLE DE CLUSTERING ===============\n")
    
    # Créer des dossiers pour les résultats si nécessaire
    os.makedirs("resultats", exist_ok=True)
    os.makedirs("modeles", exist_ok=True)
    
    # 1. Charger les résultats de la Phase 1 si nécessaire
    if resultats_phase1 is None:
        print("Chargement des résultats de la Phase 1...")
        resultats_phase1 = charger_resultats_phase1()
        if resultats_phase1 is None:
            print("Impossible de charger les résultats de la Phase 1. Arrêt de la Phase 2.")
            return None
    
    # Extraire les données catégorielles
    print("Préparation des données catégorielles uniquement...")
    df_fiable = resultats_phase1['df_fiable']
    
    # Préparer les données catégorielles (sans utiliser le texte)
    df_cat = preparer_donnees_categorielles(df_fiable)
    X_cat = df_cat.values
    
    # Utiliser les étiquettes de cause pour l'évaluation
    y = df_fiable['cause_encoded'].values
    
    # Division train/test si validation croisée activée
    if validation_croisee:
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
            X_cat, y, np.arange(len(X_cat)), test_size=0.2, random_state=42, stratify=y
        )
        df_train = df_fiable.iloc[indices_train]
        df_test = df_fiable.iloc[indices_test]
        print(f"Division train/test: {len(X_train)} tickets pour l'entraînement, {len(X_test)} pour le test")
    else:
        X_train = X_cat
        y_train = y
        df_train = df_fiable
        X_test, y_test, df_test = None, None, None
    
    # 2. Optimisation des hyperparamètres de HDBSCAN (si demandé)
    meilleurs_params = None
    if optimiser:
        print("\nOptimisation des hyperparamètres de HDBSCAN...")
        param_grid = {
            'min_cluster_size': [5, 8, 10, 15, 20],
            'min_samples': [1, 2, 5, 10, 15],
            'cluster_selection_epsilon': [0.0, 0.5, 1.0, 1.5, 2.0],
            'metric': ['euclidean', 'manhattan']
        }
        
        # Effectuer la recherche en grille
        resultats_optimisation = grille_recherche_hdbscan(
            X_train, 
            y_train, 
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
            'min_cluster_size': 10,
            'min_samples': 2,
            'cluster_selection_epsilon': 0.5,
            'metric': 'euclidean'
        }
        print(f"\nUtilisation des paramètres par défaut: {meilleurs_params}")
    
    # 3. Appliquer HDBSCAN avec les meilleurs paramètres
    print("\nApplication de HDBSCAN avec les paramètres optimaux...")
    
    # Ajouter prediction_data=True pour permettre la prédiction future
    params_with_pred = meilleurs_params.copy()
    params_with_pred['prediction_data'] = True
    
    clusterer = hdbscan.HDBSCAN(**params_with_pred)
    clusters = clusterer.fit_predict(X_train)
    
    # Réduction dimensionnelle pour visualisation
    reducer = UMAP(n_components=2, random_state=42, min_dist=0.1, n_neighbors=30)
    X_train_2d = reducer.fit_transform(X_train)
    
    # Métriques de qualité
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = (clusters == -1).sum()
    print(f"Nombre de clusters détectés: {n_clusters}")
    print(f"Nombre de points classés comme bruit: {n_noise} ({n_noise/len(clusters)*100:.2f}%)")
    
    # Silhouette score
    if n_clusters > 1:
        mask = clusters != -1
        if mask.sum() >= n_clusters * 2:
            silhouette = silhouette_score(X_train[mask], clusters[mask])
            print(f"Score de silhouette: {silhouette:.4f}")
    
    # Visualiser les clusters
    print("\nVisualisation des clusters optimaux...")
    embeddings_2d = visualiser_clusters_optimaux(
        X_train,
        clusters,
        y_train,
        titre="Clusters optimaux (catégories uniquement)"
    )
    
    # 4. Analyser la composition des clusters
    print("\nAnalyse de la composition des clusters...")
    composition = analyser_composition_clusters(clusters, df_train)
    mapping_cluster_cause = visualiser_composition_clusters(composition, n_clusters)
    
    # 5. Créer le modèle final
    modele_final = creer_modele_final(
        reducer,
        clusterer,
        mapping_cluster_cause
    )
    
    # 6. Évaluation du modèle
    resultats_evaluation = None
    if validation_croisee and X_test is not None:
        print("\nÉvaluation du modèle final sur l'ensemble de test...")
        
        # Prédiction sur les données de test
        causes_predites = predire_causes(modele_final, X_test)
        causes_reelles = df_test['cause'].values
        
        # Évaluation des prédictions
        resultats_evaluation = evaluer_predictions(
            causes_reelles,
            causes_predites
        )
        
        print(f"Exactitude sur l'ensemble de test: {resultats_evaluation['exactitude']*100:.2f}%")
        
        # Visualiser la matrice de confusion
        visualiser_matrice_confusion(resultats_evaluation['matrice_confusion'])
        
        # Visualiser les métriques par cause
        visualiser_metriques_par_cause(resultats_evaluation['metriques_par_cause'])
    else:
        # Évaluation sur l'ensemble d'entraînement
        print("\nÉvaluation du modèle final sur l'ensemble d'entraînement...")
        
        causes_predites = predire_causes(modele_final, X_train)
        causes_reelles = df_train['cause'].values
        
        # Évaluation des prédictions
        resultats_evaluation = evaluer_predictions(
            causes_reelles,
            causes_predites
        )
        
        print(f"Exactitude sur l'ensemble d'entraînement: {resultats_evaluation['exactitude']*100:.2f}%")
        
        # Visualiser la matrice de confusion
        visualiser_matrice_confusion(resultats_evaluation['matrice_confusion'])
        
        # Visualiser les métriques par cause
        visualiser_metriques_par_cause(resultats_evaluation['metriques_par_cause'])
    
    # 7. Sauvegarder le modèle final
    print("\nSauvegarde du modèle final...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chemin_modele = f"modeles/modele_clustering_metis_cat_{timestamp}.joblib"
    joblib.dump(modele_final, chemin_modele)
    print(f"Modèle sauvegardé sous: {chemin_modele}")
    
    # Créer un fichier de métadonnées
    with open(f"modeles/modele_clustering_metis_cat_{timestamp}_info.txt", "w") as f:
        f.write(f"Modèle de clustering METIS (variables catégorielles uniquement) - {timestamp}\n")
        f.write(f"Paramètres HDBSCAN: {meilleurs_params}\n")
        f.write(f"Nombre de clusters: {n_clusters}\n")
        if resultats_evaluation:
            f.write(f"Exactitude: {resultats_evaluation['exactitude']*100:.2f}%\n")
        f.write("\nMapping clusters -> causes:\n")
        for cluster, cause in mapping_cluster_cause.items():
            f.write(f"  Cluster {cluster} -> {cause}\n")
    
    print("\n=============== PHASE 2 TERMINÉE AVEC SUCCÈS ===============")
    
    return {
        'modele': modele_final,
        'evaluation': resultats_evaluation,
        'clusters': clusters,
        'embeddings_2d': embeddings_2d,
        'composition': composition,
        'mapping': mapping_cluster_cause
    }

# =========== EXÉCUTION DU SCRIPT ===========

if __name__ == "__main__":
    # Exécuter la Phase 2 sans utiliser les caractéristiques textuelles
    resultats_phase2 = executer_phase2(optimiser=True, validation_croisee=True)
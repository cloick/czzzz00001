import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, adjusted_rand_score
from transformers import TFCamembertModel, CamembertTokenizer
import hdbscan
from umap import UMAP
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import warnings

# Suppression des avertissements pour une sortie plus propre
warnings.filterwarnings("ignore")

# Définir les graines aléatoires pour la reproductibilité
np.random.seed(42)
tf.random.set_seed(42)

# Créer les dossiers pour les visualisations et rapports s'ils n'existent pas
os.makedirs("visualisations", exist_ok=True)
os.makedirs("rapports", exist_ok=True)

# Chemins des fichiers
chemin_metis = "metis_tickets.csv"
chemin_tickets_fiables = "gdp_tickets.csv"

def install_skopt():
    """
    Installe scikit-optimize s'il n'est pas disponible
    """
    try:
        import skopt
        print("scikit-optimize est déjà installé.")
        return True
    except ImportError:
        print("Installation de scikit-optimize...")
        try:
            import subprocess
            subprocess.check_call(["pip", "install", "scikit-optimize"])
            print("scikit-optimize installé avec succès.")
            return True
        except Exception as e:
            print(f"Échec de l'installation de scikit-optimize: {e}")
            return False

def nettoyer_texte(texte):
    """Nettoie le texte des notes de résolution"""
    if not isinstance(texte, str):
        return ""
    
    # Suppression des caractères spéciaux tout en gardant les lettres, chiffres et espaces
    texte = re.sub(r'[^\w\s]', ' ', texte.lower())
    
    # Suppression des espaces multiples
    texte = re.sub(r'\s+', ' ', texte).strip()
    
    return texte

def preparer_donnees(df):
    """Prépare les données pour l'analyse"""
    # Copier le DataFrame pour éviter de modifier l'original
    df_prep = df.copy()
    
    # Nettoyage du texte
    print("Application du nettoyage textuel...")
    df_prep['notes_resolution_nettoyees'] = df_prep['Notes de résolution'].apply(nettoyer_texte)
    
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

def charger_modele_local():
    """Charge le modèle CamemBERT depuis les fichiers téléchargés localement"""
    MODEL_PATH = "./camembert-base-local"
    print(f"Chargement du modèle depuis: {MODEL_PATH}")
    try:
        tokenizer = CamembertTokenizer.from_pretrained(MODEL_PATH)
        model = TFCamembertModel.from_pretrained(MODEL_PATH)
        print("Modèle chargé avec succès!")
        return tokenizer, model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        raise

def generer_embeddings(textes, tokenizer, modele, batch_size=16):
    """Génère des embeddings pour une liste de textes"""
    MAX_LENGTH = 128
    embeddings = []
    total_batches = (len(textes) + batch_size - 1) // batch_size
    
    print(f"Génération d'embeddings pour {len(textes)} textes en {total_batches} lots...")
    
    # Traitement par lots pour gérer la mémoire
    for i in range(0, len(textes), batch_size):
        if i % (10 * batch_size) == 0:
            print(f"Traitement du lot {i//batch_size+1}/{total_batches}...")
        
        lot_textes = textes[i:i+batch_size]
        
        # Tokenisation
        encodages = tokenizer(
            lot_textes,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        
        # Génération des embeddings
        with tf.device('/CPU:0'):  # Utiliser CPU si GPU non disponible
            sorties = modele(encodages['input_ids'], attention_mask=encodages['attention_mask'])
            # Utiliser la représentation [CLS] comme embedding du document
            lot_embeddings = sorties.last_hidden_state[:, 0, :].numpy()
        
        embeddings.append(lot_embeddings)
    
    # Concaténer tous les lots
    print("Finalisation des embeddings...")
    return np.vstack(embeddings)

def enrichir_embeddings_simple(embeddings, df, cat_vars):
    """
    Version simplifiée de l'enrichissement, comme dans la Phase 1
    """
    print(f"Enrichissement des embeddings avec {len(cat_vars)} variables catégorielles...")
    
    # Extraction des variables catégorielles encodées
    cat_features = np.array([df[f'{col}_encoded'] for col in cat_vars if f'{col}_encoded' in df.columns]).T
    
    # Normalisation des caractéristiques catégorielles
    if cat_features.shape[0] > 0:
        scaler = StandardScaler()
        cat_features = scaler.fit_transform(cat_features)
    
    # Concaténation des embeddings textuels avec les caractéristiques catégorielles
    enriched_embeddings = np.hstack([embeddings, cat_features])
    
    print(f"Dimensions originales des embeddings: {embeddings.shape}")
    print(f"Dimensions des embeddings enrichis: {enriched_embeddings.shape}")
    
    return enriched_embeddings

def optimiser_hdbscan(embeddings_2d, cible_clusters=15, tolerance=5):
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
                
                labels = clusterer.fit_predict(embeddings_2d)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                noise_ratio = (labels == -1).sum() / len(labels)
                
                # Calculer le score de silhouette si possible
                silhouette = None
                if n_clusters > 1 and n_clusters < len(embeddings_2d) - 1:
                    mask = labels != -1
                    if mask.sum() > n_clusters:
                        try:
                            silhouette = silhouette_score(embeddings_2d[mask], labels[mask])
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

def optimiser_hdbscan_bayesian(embeddings_2d, cible_clusters=15, tolerance=5, n_calls=30):
    """Optimisation bayésienne des paramètres HDBSCAN"""
    
    # Vérifier que scikit-optimize est installé
    if not install_skopt():
        print("Utilisation de l'optimisation par grille comme alternative...")
        return optimiser_hdbscan(embeddings_2d, cible_clusters, tolerance)
    
    # Importer les modules de scikit-optimize
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    from skopt.plots import plot_convergence, plot_objective
    
    print(f"Optimisation bayésienne des paramètres HDBSCAN pour {cible_clusters} clusters...")
    
    # Définir l'espace de recherche
    space = [
        Integer(5, 300, name='min_cluster_size'),
        Integer(2, 100, name='min_samples'),
        Real(0.0, 2.0, name='cluster_selection_epsilon')
    ]
    
    # Fonction objective
    @use_named_args(space)
    def objective(min_cluster_size, min_samples, cluster_selection_epsilon):
        # Vérifier la validité des paramètres
        if min_samples > min_cluster_size:
            return 10.0  # Pénalité pour paramètres invalides
            
        # Configurer HDBSCAN avec les paramètres proposés
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric='euclidean'
        )
        
        # Effectuer le clustering
        labels = clusterer.fit_predict(embeddings_2d)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = (labels == -1).sum() / len(labels)
        
        # Pénalité pour trop de bruit
        if noise_ratio > 0.5:
            return 5.0
            
        # Calcul du score de silhouette si possible
        silhouette = None
        if n_clusters > 1 and n_clusters < len(embeddings_2d) - 1:
            mask = labels != -1
            if mask.sum() > n_clusters:
                try:
                    silhouette = silhouette_score(embeddings_2d[mask], labels[mask])
                except:
                    silhouette = -1
        else:
            silhouette = -1
            
        # Notre objectif combine l'écart au nombre cible et le score de silhouette
        ecart = abs(n_clusters - cible_clusters)
        
        # Fonction objectif à minimiser:
        # - Pénalise fortement l'écart au nombre cible
        # - Récompense le score de silhouette élevé
        objective_value = ecart * 2.0 - silhouette if ecart <= tolerance else 10.0 + ecart
        
        print(f"MCS={min_cluster_size}, MS={min_samples}, ε={cluster_selection_epsilon:.2f}: "
              f"{n_clusters} clusters, {noise_ratio:.2%} bruit, silhouette={silhouette:.4f if silhouette != -1 else 'N/A'}, "
              f"objectif={objective_value:.4f}")
              
        return objective_value
    
    # Exécuter l'optimisation bayésienne
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        random_state=42,
        verbose=True
    )
    
    # Récupérer les meilleurs paramètres
    best_mcs, best_ms, best_eps = result.x
    
    print(f"\nMeilleurs paramètres trouvés:")
    print(f"- min_cluster_size: {best_mcs}")
    print(f"- min_samples: {best_ms}")
    print(f"- cluster_selection_epsilon: {best_eps:.2f}")
    
    # Visualiser les résultats
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plot_convergence(result)
    plt.subplot(1, 2, 2)
    plot_objective(result)
    plt.tight_layout()
    plt.savefig('visualisations/bayesian_optimization_hdbscan.png', dpi=300)
    plt.show()
    
    # Vérifier les performances des meilleurs paramètres
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=best_mcs,
        min_samples=best_ms,
        cluster_selection_epsilon=best_eps,
        metric='euclidean'
    )
    
    labels = clusterer.fit_predict(embeddings_2d)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Calculer le score de silhouette final
    silhouette = None
    if n_clusters > 1:
        mask = labels != -1
        if mask.sum() > n_clusters:
            try:
                silhouette = silhouette_score(embeddings_2d[mask], labels[mask])
                print(f"Score de silhouette final: {silhouette:.4f}")
            except:
                pass
    
    return {
        'min_cluster_size': best_mcs,
        'min_samples': best_ms,
        'cluster_selection_epsilon': best_eps
    }

def create_interactive_3d_plot(df_prep, embeddings_3d, cluster_labels, cluster_to_cause_name, params_str):
    """Crée une visualisation 3D interactive avec informations au survol des points"""
    
    # Préparer les données pour Plotly
    df_plot = pd.DataFrame({
        'UMAP1': embeddings_3d[:, 0],
        'UMAP2': embeddings_3d[:, 1],
        'UMAP3': embeddings_3d[:, 2],
        'Cluster': cluster_labels,
        'ID_Ticket': df_prep['N° INC'],
        'Groupe_affecte': df_prep['Groupe affecté'],
        'Service_metier': df_prep['Service métier'],
        'Categorie': df_prep['Cat1'],
        'Sous_categorie': df_prep['Cat2'],
        'Est_fiable': df_prep['est_fiable'],
        'Cause': [cluster_to_cause_name.get(c, "Bruit") if c != -1 else "Bruit" for c in cluster_labels]
    })
    
    # Créer la figure 3D interactive
    fig = px.scatter_3d(
        df_plot,
        x='UMAP1',
        y='UMAP2',
        z='UMAP3',
        color='Cluster',
        hover_name='ID_Ticket',
        hover_data={
            'Cluster': True,
            'Cause': True,
            'Groupe_affecte': True,
            'Service_metier': True,
            'Est_fiable': True,
            'UMAP1': False,
            'UMAP2': False,
            'UMAP3': False
        },
        color_discrete_sequence=px.colors.qualitative.Dark24,
        title=f'Clustering 3D Interactif: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters détectés'
    )
    
    # Améliorer la présentation
    fig.update_layout(
        scene=dict(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            zaxis_title='UMAP Dimension 3'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend_title_text='Cluster'
    )
    
    # Sauvegarder au format HTML pour interactivité
    fig.write_html(f"visualisations/clustering_3d_interactif_{params_str}.html")
    
    # Afficher la visualisation
    fig.show()
    
    return fig

def clustering_phase1_style(embeddings_enrichis, df_prep, cible_clusters=15, optimiser=True, 
                          params_hdbscan=None, optimisation_bayesienne=False):
    """Reproduit l'approche de clustering de la Phase 1 avec optimisation des paramètres"""
    
    print("Réduction dimensionnelle avec UMAP 2D...")
    reducer_2d = UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=15,
        min_dist=0.1
    )
    embeddings_2d = reducer_2d.fit_transform(embeddings_enrichis)
    
    print("Réduction dimensionnelle avec UMAP 3D...")
    reducer_3d = UMAP(
        n_components=3,
        random_state=42,
        n_neighbors=15,
        min_dist=0.1
    )
    embeddings_3d = reducer_3d.fit_transform(embeddings_enrichis)
    
    # Paramètres HDBSCAN
    if params_hdbscan is not None:
        print("Utilisation des paramètres HDBSCAN fournis...")
        min_cluster_size = params_hdbscan.get('min_cluster_size', 200)
        min_samples = params_hdbscan.get('min_samples', 10)
        epsilon = params_hdbscan.get('cluster_selection_epsilon', 1.0)
    elif optimiser:
        if optimisation_bayesienne:
            print("Optimisation bayésienne des paramètres HDBSCAN...")
            meilleurs_params = optimiser_hdbscan_bayesian(embeddings_2d, cible_clusters=cible_clusters)
            
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
            print("Optimisation par grille des paramètres HDBSCAN...")
            meilleurs_params, _ = optimiser_hdbscan(embeddings_2d, cible_clusters=cible_clusters)
            
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
        # Paramètres de la Phase 1 originale
        min_cluster_size = 5
        min_samples = 2
        epsilon = 0.5
    
    params_str = f"mcs{min_cluster_size}_ms{min_samples}_eps{epsilon}"
    
    print(f"Application de HDBSCAN avec min_cluster_size={min_cluster_size}, min_samples={min_samples}, epsilon={epsilon}...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        metric='euclidean'
    )
    clusters = clusterer.fit_predict(embeddings_2d)
    
    # Statistiques du clustering
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = (clusters == -1).sum()
    
    print(f"Nombre de clusters détectés: {n_clusters}")
    print(f"Nombre de points classés comme bruit: {n_noise} ({n_noise/len(clusters):.2%})")
    
    # Visualisation 2D
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=clusters,
        cmap='tab20',
        s=30,
        alpha=0.7
    )
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'Clustering 2D: {n_clusters} clusters détectés')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.tight_layout()
    plt.savefig(f"visualisations/clustering_2d_{params_str}.png", dpi=300)
    plt.show()
    
    # Mesure de qualité
    silhouette = None
    if n_clusters > 1:
        mask = clusters != -1
        if mask.sum() > n_clusters:
            silhouette = silhouette_score(embeddings_2d[mask], clusters[mask])
            print(f"Score de silhouette: {silhouette:.4f}")
    
    return clusters, embeddings_2d, embeddings_3d, clusterer, params_str

def etiqueter_clusters(cluster_labels, df, cause_mapping):
    """
    Associe chaque cluster à une cause basée sur les tickets fiables
    """
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

def analyser_clusters_inconnus(df, cluster_labels, embeddings_2d, cluster_to_cause_name):
    """
    Analyse les clusters étiquetés comme "À déterminer"
    """
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
        
        # Extraire quelques textes représentatifs (plus proches du centre du cluster)
        if len(cluster_indices) > 0:
            center = embeddings_2d[mask].mean(axis=0)
            distances = np.linalg.norm(embeddings_2d[mask] - center, axis=1)
            representative_indices = np.argsort(distances)[:min(3, len(distances))]
            
            print("\nTextes représentatifs:")
            for i, idx in enumerate(representative_indices):
                text_idx = cluster_indices[idx]
                text = df.iloc[text_idx]['notes_resolution_nettoyees']
                print(f"  {i+1}. {text[:150]}...")
                
            print(f"Nombre total de tickets dans ce cluster: {mask.sum()}")

def generer_rapport_analyse(resultats, params_str):
    """
    Génère un rapport d'analyse détaillé au format Markdown
    """
    df_prep = resultats['df_prep']
    cluster_labels = resultats['cluster_labels']
    cluster_to_cause_name = resultats['cluster_to_cause_name']
    
    # Statistiques globales
    n_total = len(df_prep)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = (cluster_labels == -1).sum()
    noise_ratio = n_noise / n_total
    
    # Silhouette score
    silhouette = None
    if n_clusters > 1:
        mask = cluster_labels != -1
        if mask.sum() > n_clusters:
            try:
                silhouette = silhouette_score(resultats['embeddings_2d'][mask], cluster_labels[mask])
            except:
                silhouette = "Non calculable"
    
    # Création du rapport
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rapport_path = f"rapports/Rapport_Analyse_{now}_{params_str}.md"
    
    with open(rapport_path, "w", encoding="utf-8") as f:
        f.write("# Rapport d'Analyse du Clustering des Tickets METIS\n\n")
        
        # Date et heure
        f.write(f"*Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}*\n\n")
        
        # Paramètres et statistiques globales
        f.write("## Paramètres et Statistiques Globales\n\n")
        f.write(f"- **Nombre total de tickets analysés**: {n_total}\n")
        f.write(f"- **Nombre de tickets fiables**: {df_prep['est_fiable'].sum()}\n")
        f.write(f"- **Nombre de clusters identifiés**: {n_clusters}\n")
        f.write(f"- **Points classés comme bruit**: {n_noise} ({noise_ratio:.2%})\n")
        f.write(f"- **Score de silhouette**: {silhouette if silhouette is not None else 'Non calculable'}\n")
        
        # Paramètres HDBSCAN
        params = params_str.split('_')
        f.write("\n### Paramètres HDBSCAN\n\n")
        for param in params:
            if param.startswith('mcs'):
                f.write(f"- **min_cluster_size**: {param[3:]}\n")
            elif param.startswith('ms'):
                f.write(f"- **min_samples**: {param[2:]}\n")
            elif param.startswith('eps'):
                f.write(f"- **cluster_selection_epsilon**: {param[3:]}\n")
        
        # Distribution des clusters
        f.write("\n## Distribution des Clusters\n\n")
        
        # Tableau récapitulatif
        f.write("| Cluster ID | Cause | Nombre de tickets | % du total | % tickets fiables |\n")
        f.write("|------------|-------|------------------|------------|-------------------|\n")
        
        cluster_stats = {}
        for cluster in sorted(set(cluster_labels)):
            if cluster == -1:
                cause = "Bruit"
            else:
                cause = cluster_to_cause_name.get(cluster, "Inconnu")
                
            mask = cluster_labels == cluster
            n_tickets = mask.sum()
            percent_total = n_tickets / n_total * 100
            
            # Calculer le pourcentage de tickets fiables dans ce cluster
            fiables_dans_cluster = (df_prep['est_fiable'] & mask).sum()
            percent_fiables = fiables_dans_cluster / df_prep['est_fiable'].sum() * 100 if df_prep['est_fiable'].sum() > 0 else 0
            
            f.write(f"| {cluster} | {cause} | {n_tickets} | {percent_total:.2f}% | {percent_fiables:.2f}% |\n")
            
            cluster_stats[cluster] = {
                'cause': cause,
                'n_tickets': n_tickets,
                'percent_total': percent_total,
                'fiables_dans_cluster': fiables_dans_cluster,
                'percent_fiables': percent_fiables
            }
        
        # Détails des clusters
        f.write("\n## Détails des Clusters\n\n")
        
        for cluster in sorted(set(cluster_labels)):
            if cluster == -1:
                continue  # Sauter le bruit
                
            cause = cluster_to_cause_name.get(cluster, "Inconnu")
            stats = cluster_stats[cluster]
            
            f.write(f"### Cluster {cluster}: {cause}\n\n")
            f.write(f"- **Nombre de tickets**: {stats['n_tickets']} ({stats['percent_total']:.2f}% du total)\n")
            f.write(f"- **Tickets fiables**: {stats['fiables_dans_cluster']} ({stats['percent_fiables']:.2f}% des tickets fiables)\n\n")
            
            # Échantillon de tickets dans ce cluster
            mask = cluster_labels == cluster
            cluster_tickets = df_prep[mask]
            
            # Nombre de tickets à afficher
            n_to_display = 20 if cause == "À déterminer" else 10
            sample_size = min(n_to_display, len(cluster_tickets))
            
            if sample_size > 0:
                # Pour assurer la diversité, prendre un échantillon aléatoire
                if len(cluster_tickets) > sample_size:
                    # Inclure d'abord les tickets fiables s'il y en a
                    fiables = cluster_tickets[cluster_tickets['est_fiable']]
                    non_fiables = cluster_tickets[~cluster_tickets['est_fiable']]
                    
                    # Nombre de fiables et non fiables à inclure
                    n_fiables = min(len(fiables), sample_size // 2)
                    n_non_fiables = sample_size - n_fiables
                    
                    fiables_sample = fiables.sample(n_fiables) if n_fiables > 0 else pd.DataFrame()
                    non_fiables_sample = non_fiables.sample(n_non_fiables) if n_non_fiables > 0 else pd.DataFrame()
                    
                    sample_tickets = pd.concat([fiables_sample, non_fiables_sample])
                else:
                    sample_tickets = cluster_tickets
                
                f.write(f"#### Échantillon de {len(sample_tickets)} tickets:\n\n")
                for idx, ticket in sample_tickets.iterrows():
                    f.write(f"**Ticket {idx} {'(fiable)' if ticket['est_fiable'] else ''}**:\n")
                    f.write(f"- Groupe affecté: {ticket['Groupe affecté']}\n")
                    f.write(f"- Service métier: {ticket['Service métier']}\n")
                    f.write(f"- Catégories: {ticket['Cat1']} / {ticket['Cat2']}\n")
                    f.write(f"- Résolution: {ticket['notes_resolution_nettoyees']}\n\n")
        
        # Statistiques sur les causes identifiées
        f.write("## Statistiques des Causes Identifiées\n\n")
        cause_counts = {}
        for cluster, cause in cluster_to_cause_name.items():
            if cluster != -1:  # Ignorer le bruit
                if cause not in cause_counts:
                    cause_counts[cause] = 0
                cause_counts[cause] += 1
        
        f.write("| Cause | Nombre de clusters |\n")
        f.write("|-------|--------------------|\n")
        for cause, count in sorted(cause_counts.items()):
            f.write(f"| {cause} | {count} |\n")
        
        print(f"Rapport d'analyse généré: {rapport_path}")
    
    return rapport_path

def executer_clustering_phase1_style(limit_samples=None, optimiser_clusters=True, 
                                    cible_clusters=15, params_hdbscan=None,
                                    optimisation_bayesienne=False):
    """
    Exécute l'ensemble du processus avec approche Phase 1
    
    Args:
        limit_samples: Limite le nombre de tickets à traiter (None = tous)
        optimiser_clusters: Si True, optimise les paramètres HDBSCAN
        cible_clusters: Nombre cible de clusters (causes)
        params_hdbscan: Paramètres HDBSCAN prédéfinis
        optimisation_bayesienne: Si True, utilise l'optimisation bayésienne au lieu de la grille
    """
    # 1. Chargement et préparation des données
    print("\n==== 1. Chargement et préparation des données ====")
    df_metis = pd.read_csv(chemin_metis)
    df_metis = df_metis[['N° INC', 'Priorité', 'Service métier', 'Cat1', 'Cat2', 
                         'Groupe affecté', 'Notes de résolution', 'cause', 'souscause']]
    
    df_fiable = pd.read_csv(chemin_tickets_fiables)
    df_metis['est_fiable'] = df_metis['N° INC'].isin(df_fiable['N° INC'])
    
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
    
    # 3. Chargement du modèle et génération des embeddings
    print("\n==== 3. Chargement du modèle et génération des embeddings ====")
    tokenizer, model = charger_modele_local()
    
    textes = df_prep['notes_resolution_nettoyees'].tolist()
    embeddings = generer_embeddings(textes, tokenizer, model)
    
    # 4. Enrichissement simple des embeddings avec variables catégorielles
    print("\n==== 4. Enrichissement des embeddings ====")
    cat_vars = ['Priorité', 'Service métier', 'Cat1', 'Cat2', 'Groupe affecté']
    
    embeddings_enrichis = enrichir_embeddings_simple(embeddings, df_prep, cat_vars)
    
    # 5. Clustering style Phase 1
    print("\n==== 5. Clustering style Phase 1 ====")
    cluster_labels, embeddings_2d, embeddings_3d, clusterer, params_str = clustering_phase1_style(
        embeddings_enrichis,
        df_prep, 
        cible_clusters=cible_clusters,
        optimiser=optimiser_clusters,   
        params_hdbscan=params_hdbscan,
        optimisation_bayesienne=optimisation_bayesienne
    )
    
    # 6. Étiquetage des clusters
    print("\n==== 6. Étiquetage des clusters ====")
    cluster_to_cause, cluster_to_cause_name = etiqueter_clusters(
        cluster_labels, df_prep, cause_mapping
    )
    
    # 7. Analyse des clusters non identifiés
    print("\n==== 7. Analyse des clusters non identifiés ====")
    analyser_clusters_inconnus(df_prep, cluster_labels, embeddings_2d, cluster_to_cause_name)
    
    # 8. Création de la visualisation 3D interactive
    print("\n==== 8. Création de la visualisation 3D interactive ====")
    fig_3d = create_interactive_3d_plot(df_prep, embeddings_3d, cluster_labels, cluster_to_cause_name, params_str)
    
    # 9. Génération du rapport d'analyse
    print("\n==== 9. Génération du rapport d'analyse ====")
    
    # Ajouter les clusters au DataFrame
    df_prep['cluster'] = cluster_labels
    
    resultats = {
        'df_prep': df_prep,
        'cluster_labels': cluster_labels,
        'embeddings_2d': embeddings_2d,
        'embeddings_3d': embeddings_3d,
        'clusterer': clusterer,
        'cluster_to_cause_name': cluster_to_cause_name
    }
    
    rapport_path = generer_rapport_analyse(resultats, params_str)
    
    print("\nProcessus terminé avec succès!")
    print(f"Rapport détaillé disponible: {rapport_path}")
    print(f"Visualisation 3D interactive disponible: visualisations/clustering_3d_interactif_{params_str}.html")
    
    return resultats

# Exécution avec paramètres optimaux
if __name__ == "__main__":
    # Configuration des paramètres d'exécution
    OPTIMISATION_BAYESIENNE = True
    LIMIT_SAMPLES = 20000
    
    if OPTIMISATION_BAYESIENNE:
        # Exécuter avec optimisation bayésienne
        resultats = executer_clustering_phase1_style(
            limit_samples=LIMIT_SAMPLES,
            optimiser_clusters=True,
            cible_clusters=15,
            optimisation_bayesienne=True
        )
    else:
        # Si on a déjà des paramètres optimaux, les utiliser directement
        params_optimaux = {
            'min_cluster_size': 200,
            'min_samples': 10,
            'cluster_selection_epsilon': 1.0
        }
        
        # Exécuter avec les paramètres optimaux et un échantillon de taille définie
        resultats = executer_clustering_phase1_style(
            limit_samples=LIMIT_SAMPLES,
            optimiser_clusters=False,
            cible_clusters=15,
            params_hdbscan=params_optimaux
        )
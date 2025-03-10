1. Importation des bibliothèques nécessaires

# Importation des bibliothèques
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

# Définir les graines aléatoires pour la reproductibilité
np.random.seed(42)
tf.random.set_seed(42)

2. Chargement et préparation des données

# Chemin des fichiers
chemin_metis = "metis_tickets.csv"
chemin_tickets_fiables = "gdp_tickets.csv"

# Chargement des données
df_metis = pd.read_csv(chemin_metis)
df_metis = df_metis[['N° INC', 'Priorité', 'Service métier', 'Cat1', 'Cat2', 
                     'Groupe affecté', 'Notes de résolution', 'cause', 'souscause']]

df_fiable = pd.read_csv(chemin_tickets_fiables)

# Marquer les tickets fiables dans le DataFrame principal
df_metis['est_fiable'] = df_metis['N° INC'].isin(df_fiable['N° INC'])

print(f"Nombre total de tickets METIS: {len(df_metis)}")
print(f"Nombre de tickets fiables: {df_metis['est_fiable'].sum()}")

3. Fonctions de nettoyage et préparation des données

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

4. Chargement du modèle et génération des embeddings

# Paramètres globaux
MODEL_PATH = "./camembert-base-local"
MAX_LENGTH = 128
BATCH_SIZE = 16

def charger_modele_local():
    """Charge le modèle CamemBERT depuis les fichiers téléchargés localement"""
    print(f"Chargement du modèle depuis: {MODEL_PATH}")
    try:
        tokenizer = CamembertTokenizer.from_pretrained(MODEL_PATH)
        model = TFCamembertModel.from_pretrained(MODEL_PATH)
        print("Modèle chargé avec succès!")
        return tokenizer, model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        raise

def generer_embeddings(textes, tokenizer, modele, batch_size=BATCH_SIZE):
    """Génère des embeddings pour une liste de textes"""
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

5. Enrichissement des embeddings avec les variables catégorielles

def enrichir_embeddings(embeddings, df, cat_vars, poids=None):
    """
    Enrichit les embeddings textuels avec les variables catégorielles pondérées
    """
    print(f"Enrichissement des embeddings avec {len(cat_vars)} variables catégorielles...")
    
    # Valeurs par défaut des poids si non spécifiés
    if poids is None:
        poids = {
            'Groupe affecté': 3.0,    # Plus forte influence basée sur l'information mutuelle
            'Service métier': 2.0,     # Seconde influence la plus forte
            'Cat1': 1.0,
            'Cat2': 1.0,
            'Priorité': 0.5            # Moins d'influence car relation non significative
        }
    
    # Extraction et normalisation des caractéristiques catégorielles
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
        
        # Normaliser les embeddings textuels pour équilibrer avec les caractéristiques catégorielles
        embeddings_norm = StandardScaler().fit_transform(embeddings)
        
        # Concaténer embeddings textuels et caractéristiques catégorielles
        enriched_embeddings = np.hstack([embeddings_norm, cat_features])
    else:
        enriched_embeddings = embeddings
    
    print(f"Dimensions originales des embeddings: {embeddings.shape}")
    print(f"Dimensions des embeddings enrichis: {enriched_embeddings.shape}")
    
    return enriched_embeddings

6. Optimisation du clustering et classification semi-supervisée

def optimiser_min_cluster_size(embeddings_2d, n_causes_cible=15, tolerance=2):
    """
    Recherche le meilleur min_cluster_size pour obtenir environ n_causes_cible clusters
    """
    print(f"Optimisation du paramètre min_cluster_size pour obtenir environ {n_causes_cible} clusters...")
    
    # Grille de valeurs à tester pour min_cluster_size
    # Créons une grille logarithmique pour explorer un large éventail de valeurs
    n_samples = len(embeddings_2d)
    grid_min = max(3, n_samples // (n_causes_cible * 5))
    grid_max = min(n_samples // 10, n_samples // (n_causes_cible // 2))
    
    # Créer une grille logarithmique entre grid_min et grid_max
    grid_values = np.unique(np.geomspace(grid_min, grid_max, num=10, dtype=int))
    
    print(f"Valeurs testées pour min_cluster_size: {grid_values}")
    
    resultats = []
    
    # Tester chaque valeur de la grille
    for min_size in grid_values:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_size,
            min_samples=max(3, min_size // 3),
            cluster_selection_epsilon=0.5,
            metric='euclidean'
        )
        
        labels = clusterer.fit_predict(embeddings_2d)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = (labels == -1).sum() / len(labels)
        
        resultats.append((min_size, n_clusters, noise_ratio))
        print(f"min_cluster_size={min_size}: {n_clusters} clusters, {noise_ratio:.2%} bruit")
    
    # Trouver la valeur qui donne un nombre de clusters le plus proche de n_causes_cible
    best_min_size = None
    best_distance = float('inf')
    
    for min_size, n_clusters, noise_ratio in resultats:
        # Calculer la distance au nombre cible, en pénalisant les ratios de bruit élevés
        if noise_ratio > 0.5:  # Pénaliser fortement si plus de 50% des points sont considérés comme du bruit
            continue
            
        distance = abs(n_clusters - n_causes_cible)
        
        if distance <= tolerance:  # Si on est dans la tolérance
            if best_min_size is None or distance < best_distance or (distance == best_distance and noise_ratio < resultats[resultats.index((best_min_size, n_clusters, noise_ratio))][2]):
                best_distance = distance
                best_min_size = min_size
    
    # Si aucune valeur ne donne un nombre de clusters dans la tolérance, prendre la plus proche
    if best_min_size is None:
        best_min_size = min(resultats, key=lambda x: abs(x[1] - n_causes_cible) + (0.5 if x[2] > 0.5 else 0))[0]
    
    print(f"Meilleure valeur trouvée: min_cluster_size={best_min_size}")
    return best_min_size

def generer_contraintes(df_fiable):
    """
    Génère des contraintes must-link et cannot-link à partir des tickets fiables
    """
    must_link = []
    cannot_link = []
    
    # Obtenir les indices des tickets fiables
    indices_fiables = df_fiable[df_fiable['est_fiable']].index.tolist()
    
    # Pour chaque paire de tickets fiables
    for i in range(len(indices_fiables)):
        for j in range(i+1, len(indices_fiables)):
            idx_i = indices_fiables[i]
            idx_j = indices_fiables[j]
            
            # Si même cause -> must-link
            if df_fiable.loc[idx_i, 'cause_encoded'] == df_fiable.loc[idx_j, 'cause_encoded']:
                must_link.append((idx_i, idx_j))
            # Sinon -> cannot-link
            else:
                cannot_link.append((idx_i, idx_j))
    
    print(f"Contraintes générées: {len(must_link)} must-link, {len(cannot_link)} cannot-link")
    return must_link, cannot_link

def clustering_semi_supervise(embeddings, must_link=None, cannot_link=None, n_causes=15):
    """
    Effectue un clustering semi-supervisé avec contraintes
    """
    print("Réduction dimensionnelle avec UMAP...")
    
    # Réduction dimensionnelle avec UMAP
    reducer = UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=30,
        min_dist=0.1,
        metric='euclidean'
    )
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Optimisation du paramètre min_cluster_size
    min_cluster_size = optimiser_min_cluster_size(embeddings_2d, n_causes_cible=n_causes)
    
    print(f"Configuration de HDBSCAN avec min_cluster_size={min_cluster_size}")
    
    # Configuration de HDBSCAN pour le clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=max(3, min_cluster_size // 3),
        cluster_selection_epsilon=0.5,
        metric='euclidean',
        prediction_data=True
    )
    
    # Appliquer le clustering
    cluster_labels = clusterer.fit_predict(embeddings_2d)
    
    # Afficher les statistiques du clustering
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = (cluster_labels == -1).sum()
    
    print(f"Nombre de clusters détectés: {n_clusters}")
    print(f"Nombre de points classés comme bruit: {n_noise} ({n_noise/len(cluster_labels):.2%})")
    
    # Visualiser les résultats
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=cluster_labels,
        cmap='tab20',
        s=30,
        alpha=0.7
    )
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'Clustering semi-supervisé: {n_clusters} clusters détectés')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.tight_layout()
    plt.savefig("clustering_semi_supervise.png", dpi=300)
    plt.show()
    
    return cluster_labels, embeddings_2d, clusterer

7. Étiquetage et analyse des clusters

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
            
8. Exécution du processus complet

def executer_clustering_niveau1(limit_samples=None):
    """
    Exécute l'ensemble du processus de clustering de niveau 1
    
    Parameters:
    -----------
    limit_samples : int, optional
        Limite le nombre de tickets à traiter pour les tests (None = tous)
    """
    # 1. Chargement et préparation des données
    print("\n==== 1. Chargement et préparation des données ====")
    df_metis = pd.read_csv(chemin_metis)
    df_metis = df_metis[['N° INC', 'Priorité', 'Service métier', 'Cat1', 'Cat2', 
                         'Groupe affecté', 'Notes de résolution', 'cause', 'souscause']]
    
    df_fiable = pd.read_csv(chemin_tickets_fiables)
    df_metis['est_fiable'] = df_metis['N° INC'].isin(df_fiable['N° INC'])
    
    print(f"Nombre total de tickets METIS: {len(df_metis)}")
    print(f"Nombre de tickets fiables: {df_metis['est_fiable'].sum()}")
    
    # Limiter le nombre d'échantillons si demandé
    if limit_samples and limit_samples < len(df_metis):
        # Assurez-vous d'inclure tous les tickets fiables dans l'échantillon
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
    
    # 4. Enrichissement des embeddings avec les variables catégorielles pondérées
    print("\n==== 4. Enrichissement des embeddings ====")
    cat_vars = ['Priorité', 'Service métier', 'Cat1', 'Cat2', 'Groupe affecté']
    poids = {
        'Groupe affecté': 3.0,
        'Service métier': 2.0,
        'Cat1': 1.0,
        'Cat2': 1.0,
        'Priorité': 0.5
    }
    
    embeddings_enrichis = enrichir_embeddings(embeddings, df_prep, cat_vars, poids)
    
    # 5. Génération des contraintes pour le clustering semi-supervisé
    print("\n==== 5. Génération des contraintes ====")
    must_link, cannot_link = generer_contraintes(df_prep)
    
    # 6. Clustering semi-supervisé
    print("\n==== 6. Clustering semi-supervisé ====")
    cluster_labels, embeddings_2d, clusterer = clustering_semi_supervise(
        embeddings_enrichis, must_link, cannot_link, n_causes=15
    )
    
    # 7. Étiquetage des clusters
    print("\n==== 7. Étiquetage des clusters ====")
    cluster_to_cause, cluster_to_cause_name = etiqueter_clusters(
        cluster_labels, df_prep, cause_mapping
    )
    
    # 8. Analyse des clusters non identifiés
    print("\n==== 8. Analyse des clusters non identifiés ====")
    analyser_clusters_inconnus(df_prep, cluster_labels, embeddings_2d, cluster_to_cause_name)
    
    # 9. Évaluation de la qualité du clustering
    print("\n==== 9. Évaluation de la qualité du clustering ====")
    # Calculer le score de silhouette (en excluant les points de bruit)
    mask_non_noise = cluster_labels != -1
    if mask_non_noise.sum() > 0:
        silhouette = silhouette_score(
            embeddings_2d[mask_non_noise], 
            cluster_labels[mask_non_noise]
        )
        print(f"Score de silhouette: {silhouette:.4f}")
    
    # Calculer l'ARI par rapport aux causes réelles (pour les tickets fiables uniquement)
    mask_fiable = df_prep['est_fiable']
    if mask_fiable.sum() > 0:
        fiable_indices = np.where(mask_fiable)[0]
        ari = adjusted_rand_score(
            df_prep.loc[mask_fiable, 'cause_encoded'],
            cluster_labels[fiable_indices]
        )
        print(f"Indice de Rand ajusté (ARI): {ari:.4f}")
        
    print("\nProcessus de clustering de niveau 1 terminé avec succès!")
    
    # Ajouter les clusters au DataFrame
    df_prep['cluster'] = cluster_labels
    
    # Renvoyer les résultats pour une utilisation ultérieure
    return {
        'df_prep': df_prep,
        'cluster_labels': cluster_labels,
        'embeddings_2d': embeddings_2d,
        'clusterer': clusterer,
        'cluster_to_cause_name': cluster_to_cause_name,
        'cause_mapping': cause_mapping
    }


9. Exécution avec un échantillon limité pour les tests

# Exécuter le processus complet avec un échantillon limité pour les tests
# Pour traiter tous les tickets, utiliser limit_samples=None
resultats = executer_clustering_niveau1(limit_samples=5000)

10. Visualisation avancée des résultats

def visualiser_resultats(resultats):
    """
    Visualisation avancée des résultats du clustering
    """
    df_prep = resultats['df_prep']
    cluster_labels = resultats['cluster_labels']
    embeddings_2d = resultats['embeddings_2d']
    cluster_to_cause_name = resultats['cluster_to_cause_name']
    
    # Créer une palette de couleurs pour les causes
    causes_uniques = set(cluster_to_cause_name.values())
    n_causes = len(causes_uniques)
    palette = sns.color_palette("hsv", n_causes)
    cause_colors = {cause: palette[i] for i, cause in enumerate(causes_uniques)}
    
    # Préparation des données pour la visualisation
    plot_data = pd.DataFrame({
        'UMAP1': embeddings_2d[:, 0],
        'UMAP2': embeddings_2d[:, 1],
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
    
    plt.title('Clustering des tickets METIS par cause identifiée')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("visualisation_clustering_causes.png", dpi=300)
    plt.show()
    
    return plot_data

# Visualiser les résultats
plot_data = visualiser_resultats(resultats)

11. Analyse statistique des causes identifiées

def analyser_distribution_causes(resultats):
    """
    Analyse statistique de la distribution des causes
    """
    df_prep = resultats['df_prep']
    cluster_labels = resultats['cluster_labels']
    cluster_to_cause_name = resultats['cluster_to_cause_name']
    
    # Créer une colonne avec la cause identifiée
    df_prep['cause_identifiee'] = [
        cluster_to_cause_name.get(c, "Bruit") if c != -1 else "Bruit"
        for c in cluster_labels
    ]
    
    # Distribtion des causes identifiées
    print("\nDistribution des causes identifiées:")
    cause_counts = df_prep['cause_identifiee'].value_counts()
    
    for cause, count in cause_counts.items():
        print(f"{cause}: {count} tickets ({count/len(df_prep):.2%})")
    
    # Visualisation de la distribution
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(y=df_prep['cause_identifiee'], order=cause_counts.index)
    
    # Ajouter les valeurs sur les barres
    for i, count in enumerate(cause_counts):
        ax.text(count + 5, i, str(count))
    
    plt.title('Distribution des causes identifiées')
    plt.tight_layout()
    plt.savefig("distribution_causes.png", dpi=300)
    plt.show()
    
    # Analyse de confusion pour les tickets fiables
    fiables = df_prep[df_prep['est_fiable']]
    
    if not fiables.empty:
        confusion = pd.crosstab(
            fiables['cause'],
            fiables['cause_identifiee'],
            rownames=['Cause réelle'],
            colnames=['Cause identifiée']
        )
        
        print("\nMatrice de confusion pour les tickets fiables:")
        print(confusion)
        
        # Visualisation de la matrice de confusion
        plt.figure(figsize=(14, 12))
        sns.heatmap(confusion, annot=True, cmap='YlGnBu', fmt='d')
        plt.title('Matrice de confusion pour les tickets fiables')
        plt.tight_layout()
        plt.savefig("confusion_causes.png", dpi=300)
        plt.show()
    
    return df_prep

# Analyser la distribution des causes
df_analyse = analyser_distribution_causes(resultats)


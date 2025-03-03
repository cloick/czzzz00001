# =========== IMPORTATIONS NÉCESSAIRES ===========
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import chi2_contingency
import re
import umap
import hdbscan
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from transformers import TFCamembertModel, CamembertTokenizer
import tensorflow as tf

# =========== PARAMÈTRES GLOBAUX ===========
MODEL_PATH = "./camembert-base"  # Dossier contenant les fichiers téléchargés manuellement
MAX_LENGTH = 128                 # Longueur maximale des textes
BATCH_SIZE = 16                  # Taille des lots pour le traitement
RANDOM_SEED = 42                 # Graine aléatoire pour la reproductibilité
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Création du dossier résultats s'il n'existe pas
os.makedirs("resultats", exist_ok=True)

# =========== FONCTIONS DE CHARGEMENT ET PRÉTRAITEMENT ===========

def charger_donnees(chemin_metis, chemin_tickets_fiables):
    """
    Charge les données METIS et le sous-ensemble de tickets fiables
    """
    # Chargement des données complètes METIS
    df_metis = pd.read_csv(chemin_metis)
    
    # Chargement des tickets fiables (annotations des experts GDP)
    df_fiable = pd.read_csv(chemin_tickets_fiables)
    
    # Marquer les tickets fiables dans le DataFrame principal
    df_metis['est_fiable'] = df_metis['ticket_id'].isin(df_fiable['ticket_id'])
    
    print(f"Nombre total de tickets METIS: {len(df_metis)}")
    print(f"Nombre de tickets fiables: {df_metis['est_fiable'].sum()}")
    
    return df_metis, df_fiable

def nettoyer_texte(texte):
    """
    Nettoie le texte des notes de résolution
    """
    if not isinstance(texte, str):
        return ""
    
    # Suppression des caractères spéciaux tout en gardant les lettres, chiffres et espaces
    texte = re.sub(r'[^\w\s]', ' ', texte.lower())
    
    # Suppression des espaces multiples
    texte = re.sub(r'\s+', ' ', texte).strip()
    
    return texte

def preparer_donnees(df):
    """
    Prépare les données pour l'analyse
    """
    # Copier le DataFrame pour éviter de modifier l'original
    df_prep = df.copy()
    
    # Nettoyage du texte - UTILISATION EXPLICITE de nettoyer_texte
    print("Application du nettoyage textuel...")
    df_prep['notes_resolution_nettoyees'] = df_prep['Notes de résolution'].apply(nettoyer_texte)
    
    # Encodage des caractéristiques catégorielles
    for col in ['Priorité', 'Service métier', 'Cat1', 'Cat2', 'Groupe affecté']:
        if col in df_prep.columns:
            le = LabelEncoder()
            df_prep[f'{col}_encoded'] = le.fit_transform(df_prep[col].fillna('INCONNU'))
    
    # Création de variables pour la cause et la sous-cause
    if 'cause' in df_prep.columns:
        df_prep['cause_encoded'] = LabelEncoder().fit_transform(df_prep['cause'].fillna('INCONNU'))
    
    if 'souscause' in df_prep.columns:
        df_prep['souscause_encoded'] = LabelEncoder().fit_transform(df_prep['souscause'].fillna('INCONNU'))
    
    # Vérification rapide des données nettoyées
    verifier_nettoyage(df_prep, n_exemples=3)
    
    return df_prep

def verifier_nettoyage(df, n_exemples=5):
    """
    Affiche quelques exemples avant/après nettoyage pour vérification
    """
    exemples = df.sample(n_exemples)
    for idx, row in exemples.iterrows():
        print("\n=== Exemple de nettoyage textuel ===")
        print("AVANT:", row['Notes de résolution'][:150], "...")
        print("APRÈS:", row['notes_resolution_nettoyees'][:150], "...")
    
    # Statistiques basiques sur les textes
    longueurs_avant = df['Notes de résolution'].astype(str).apply(len)
    longueurs_apres = df['notes_resolution_nettoyees'].apply(len)
    
    print(f"\nLongueur moyenne avant nettoyage: {longueurs_avant.mean():.1f} caractères")
    print(f"Longueur moyenne après nettoyage: {longueurs_apres.mean():.1f} caractères")
    print(f"Réduction moyenne: {(1 - longueurs_apres.mean()/longueurs_avant.mean())*100:.1f}%")

# =========== FONCTIONS DE GÉNÉRATION D'EMBEDDINGS ===========

def charger_modele_local():
    """
    Charge le modèle CamemBERT depuis les fichiers téléchargés localement
    """
    print(f"Chargement du modèle depuis: {MODEL_PATH}")
    try:
        tokenizer = CamembertTokenizer.from_pretrained(MODEL_PATH)
        model = TFCamembertModel.from_pretrained(MODEL_PATH)
        print("Modèle chargé avec succès!")
        return tokenizer, model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        print("\nAssurez-vous d'avoir téléchargé les fichiers suivants dans le dossier camembert-base:")
        print("- config.json")
        print("- tf_model.h5")
        print("- vocab.json")
        print("- sentencepiece.bpe.model")
        print("- special_tokens_map.json")
        print("- tokenizer_config.json")
        raise

def generer_embeddings(textes, tokenizer, modele, batch_size=8):
    """
    Génère des embeddings pour une liste de textes
    """
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

# =========== FONCTIONS DE VISUALISATION ET ANALYSE (TEXTE) ===========

def visualiser_embeddings(embeddings, labels=None, titre="Visualisation des embeddings"):
    """
    Visualise les embeddings en 2D avec UMAP
    """
    print("Réduction dimensionnelle avec UMAP...")
    # Réduction dimensionnelle avec UMAP
    reducer = umap.UMAP(
        n_components=2, 
        random_state=RANDOM_SEED,
        n_neighbors=15,
        min_dist=0.1
    )
    embeddings_2d = reducer.fit_transform(embeddings)
    
    print("Génération du graphique...")
    # Visualisation
    plt.figure(figsize=(12, 10))
    
    if labels is not None:
        # Scatter plot coloré par cause
        scatter = plt.scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1], 
            c=labels, 
            cmap='tab20', 
            s=30, 
            alpha=0.7
        )
        plt.colorbar(scatter, label='Cause')
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=30, alpha=0.7)
    
    plt.title(titre)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.tight_layout()
    plt.savefig(f"{titre.replace(' ', '_').lower()}.png", dpi=300)
    plt.show()
    
    return embeddings_2d

def analyser_tickets_fiables(df):
    """
    Analyse la distribution des causes et sous-causes dans les tickets fiables
    """
    print("Analyse de la distribution des causes et sous-causes...")
    tickets_fiables = df[df['est_fiable']]
    
    # Distribution des causes
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(y=tickets_fiables['cause'], order=tickets_fiables['cause'].value_counts().index)
    ax.set_ylabel('Cause')
    ax.set_xlabel('Nombre de tickets')
    plt.title('Distribution des causes dans les tickets fiables')
    # Ajouter les valeurs sur les barres
    for p in ax.patches:
        width = p.get_width()
        plt.text(width + 1, p.get_y() + p.get_height()/2, 
                 f'{width:.0f}', ha='left', va='center')
    plt.tight_layout()
    plt.savefig('distribution_causes_fiables.png', dpi=300)
    plt.show()
    
    # Distribution des sous-causes
    plt.figure(figsize=(14, 10))
    ax = sns.countplot(y=tickets_fiables['souscause'], 
                    order=tickets_fiables['souscause'].value_counts().index)
    ax.set_ylabel('Sous-cause')
    ax.set_xlabel('Nombre de tickets')
    plt.title('Distribution des sous-causes dans les tickets fiables')
    for p in ax.patches:
        width = p.get_width()
        plt.text(width + 1, p.get_y() + p.get_height()/2, 
                 f'{width:.0f}', ha='left', va='center')
    plt.tight_layout()
    plt.savefig('distribution_souscauses_fiables.png', dpi=300)
    plt.show()
    
    # Matrice de co-occurrence cause/sous-cause
    print("Génération de la matrice de co-occurrence cause/sous-cause...")
    pivote = pd.crosstab(tickets_fiables['cause'], tickets_fiables['souscause'])
    plt.figure(figsize=(16, 12))
    sns.heatmap(pivote, annot=True, cmap='YlGnBu', fmt='d')
    plt.title('Matrice de co-occurrence cause/sous-cause')
    plt.tight_layout()
    plt.savefig('matrice_cooccurrence.png', dpi=300)
    plt.show()
    
    # Analyse des causes manquantes
    total_causes = 15
    total_sous_causes = 45
    causes_presentes = tickets_fiables['cause'].nunique()
    sous_causes_presentes = tickets_fiables['souscause'].nunique()
    
    print(f"\nAnalyse de la couverture des causes et sous-causes:")
    print(f"Causes présentes: {causes_presentes}/{total_causes} ({causes_presentes/total_causes*100:.1f}%)")
    print(f"Sous-causes présentes: {sous_causes_presentes}/{total_sous_causes} ({sous_causes_presentes/total_sous_causes*100:.1f}%)")
    
    return {
        'causes_presentes': tickets_fiables['cause'].unique(),
        'sous_causes_presentes': tickets_fiables['souscause'].unique(),
        'causes_manquantes': total_causes - causes_presentes,
        'sous_causes_manquantes': total_sous_causes - sous_causes_presentes
    }

def analyser_structure_clusters_initiaux(embeddings_2d, labels=None, titre="Analyse des clusters naturels"):
    """
    Applique HDBSCAN pour détecter les clusters naturels dans l'espace réduit
    """
    print("Application de HDBSCAN pour la détection de clusters naturels...")
    # Application de HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=2,
        cluster_selection_epsilon=0.5,
        metric='euclidean'
    )
    clusters = clusterer.fit_predict(embeddings_2d)
    
    # Compter le nombre de clusters (sans compter -1 qui est le bruit)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    print(f"Nombre de clusters détectés: {n_clusters}")
    print(f"Nombre de points classés comme bruit: {(clusters == -1).sum()}")
    
    # Visualisation des clusters
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
    plt.title(titre)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.tight_layout()
    plt.savefig(f"{titre.replace(' ', '_').lower()}.png", dpi=300)
    plt.show()
    
    # Évaluation de la qualité des clusters
    if n_clusters > 1:  # Si plus d'un cluster (hors bruit)
        # Filtrer les points de bruit pour le calcul du score de silhouette
        mask = clusters != -1
        if mask.sum() > n_clusters:  # Vérifier qu'il reste suffisamment de points
            silhouette = silhouette_score(
                embeddings_2d[mask], 
                clusters[mask]
            )
            print(f"Score de silhouette: {silhouette:.4f}")
    
    # Comparaison avec les causes réelles (si disponibles)
    if labels is not None:
        # Calcul de l'ARI en ignorant les points de bruit
        mask = clusters != -1
        if mask.sum() > 0:
            ari = adjusted_rand_score(labels[mask], clusters[mask])
            print(f"Indice de Rand ajusté (par rapport aux causes réelles): {ari:.4f}")
    
    return clusters

# =========== FONCTIONS D'ANALYSE DES CARACTÉRISTIQUES CATÉGORIELLES ===========

def analyser_importance_variables_categorielles(df_fiable):
    """
    Analyse l'importance des variables catégorielles par rapport aux causes et sous-causes
    en calculant l'information mutuelle et réalisant des tests de chi2
    """
    print("Analyse de l'importance des variables catégorielles...")
    
    # Variables catégorielles à analyser
    cat_vars = ['Priorité', 'Service métier', 'Cat1', 'Cat2', 'Groupe affecté']
    
    # Calcul de l'information mutuelle entre chaque variable catégorielle et la cause
    mi_scores = {}
    for col in cat_vars:
        if col in df_fiable.columns:
            # Calcul de l'information mutuelle
            mi = mutual_info_classif(
                df_fiable[[f'{col}_encoded']], 
                df_fiable['cause_encoded'],
                random_state=42
            )
            mi_scores[col] = mi[0]
    
    # Affichage des scores d'information mutuelle
    plt.figure(figsize=(10, 6))
    bars = plt.bar(mi_scores.keys(), mi_scores.values())
    plt.ylabel('Information mutuelle')
    plt.title('Information mutuelle entre variables catégorielles et causes')
    plt.xticks(rotation=45)
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('importance_variables_categorielles.png', dpi=300)
    plt.show()
    
    # Analyse de contingence avec les causes principales
    print("\nTests d'indépendance Chi² entre variables catégorielles et causes:")
    for col in cat_vars:
        if col in df_fiable.columns:
            # Tableau de contingence
            contingency = pd.crosstab(df_fiable[col], df_fiable['cause'])
            
            # Test chi²
            chi2, p, dof, expected = chi2_contingency(contingency)
            
            print(f"- {col}: chi²={chi2:.2f}, p-value={p:.4f}")
            if p < 0.05:
                print(f"  ✓ Relation significative détectée (p<0.05)")
            else:
                print(f"  ✗ Pas de relation significative détectée (p≥0.05)")
    
    return mi_scores

def visualiser_relations_categories_causes(df_fiable):
    """
    Visualise les relations entre variables catégorielles et causes
    via des heatmaps et des graphiques proportionnels
    """
    print("\nVisualisation des relations entre variables catégorielles et causes...")
    
    # Liste des variables catégorielles à analyser
    cat_vars = ['Priorité', 'Service métier', 'Cat1', 'Cat2', 'Groupe affecté']
    
    for col in cat_vars:
        if col in df_fiable.columns:
            # Création d'un tableau croisé dynamique
            crosstab = pd.crosstab(
                df_fiable[col], 
                df_fiable['cause'], 
                normalize='index',
                margins=False
            )
            
            # Visualisation avec heatmap
            plt.figure(figsize=(14, 8))
            sns.heatmap(crosstab, annot=True, cmap='YlGnBu', fmt='.0%')
            plt.title(f'Distribution des causes par {col}')
            plt.tight_layout()
            plt.savefig(f'relation_{col}_causes.png', dpi=300)
            plt.show()
            
            # Graphique en barres empilées
            crosstab_abs = pd.crosstab(df_fiable[col], df_fiable['cause'])
            crosstab_abs_pct = crosstab_abs.div(crosstab_abs.sum(axis=1), axis=0)
            
            plt.figure(figsize=(14, 8))
            crosstab_abs_pct.plot(kind='bar', stacked=True, colormap='tab20')
            plt.title(f'Proportion des causes par {col}')
            plt.legend(title='Cause', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f'proportion_{col}_causes.png', dpi=300)
            plt.show()

def enrichir_embeddings(embeddings, df, cat_vars, normalisation=True):
    """
    Enrichit les embeddings textuels en les combinant avec les variables catégorielles encodées
    """
    print(f"Enrichissement des embeddings avec {len(cat_vars)} variables catégorielles...")
    
    # Extraction des variables catégorielles encodées
    cat_features = np.array([df[f'{col}_encoded'] for col in cat_vars if f'{col}_encoded' in df.columns]).T
    
    # Normalisation des caractéristiques catégorielles pour équilibrer leur influence
    if normalisation and cat_features.shape[0] > 0:
        scaler = StandardScaler()
        cat_features = scaler.fit_transform(cat_features)
    
    # Concaténation des embeddings textuels avec les caractéristiques catégorielles
    enriched_embeddings = np.hstack([embeddings, cat_features])
    
    print(f"Dimensions originales des embeddings: {embeddings.shape}")
    print(f"Dimensions des embeddings enrichis: {enriched_embeddings.shape}")
    
    return enriched_embeddings

def comparer_clustering(embeddings_texte, embeddings_enrichis, labels=None, titre="Comparaison"):
    """
    Compare les résultats de clustering avec et sans variables catégorielles
    """
    print("\nComparaison des résultats de clustering (texte seul vs. enrichi)...")
    
    # Réduction dimensionnelle pour visualisation
    reducer = umap.UMAP(n_components=2, random_state=42)
    text_2d = reducer.fit_transform(embeddings_texte)
    
    reducer = umap.UMAP(n_components=2, random_state=42)
    enriched_2d = reducer.fit_transform(embeddings_enrichis)
    
    # Clustering HDBSCAN sur les deux types d'embeddings
    clusterer_text = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=2,
        cluster_selection_epsilon=0.5,
        metric='euclidean'
    )
    clusters_text = clusterer_text.fit_predict(text_2d)
    
    clusterer_enriched = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=2,
        cluster_selection_epsilon=0.5,
        metric='euclidean'
    )
    clusters_enriched = clusterer_enriched.fit_predict(enriched_2d)
    
    # Nombre de clusters détectés
    n_clusters_text = len(set(clusters_text)) - (1 if -1 in clusters_text else 0)
    n_clusters_enriched = len(set(clusters_enriched)) - (1 if -1 in clusters_enriched else 0)
    
    # Visualisation des résultats
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Visualisation des clusters basés sur le texte seul
    scatter1 = ax1.scatter(
        text_2d[:, 0], 
        text_2d[:, 1], 
        c=clusters_text, 
        cmap='tab20', 
        s=30, 
        alpha=0.7
    )
    ax1.set_title(f"Clustering avec texte seul\n({n_clusters_text} clusters détectés)")
    ax1.set_xlabel('UMAP Dimension 1')
    ax1.set_ylabel('UMAP Dimension 2')
    plt.colorbar(scatter1, ax=ax1, label='Cluster')
    
    # Visualisation des clusters basés sur les embeddings enrichis
    scatter2 = ax2.scatter(
        enriched_2d[:, 0], 
        enriched_2d[:, 1], 
        c=clusters_enriched, 
        cmap='tab20', 
        s=30, 
        alpha=0.7
    )
    ax2.set_title(f"Clustering avec embeddings enrichis\n({n_clusters_enriched} clusters détectés)")
    ax2.set_xlabel('UMAP Dimension 1')
    ax2.set_ylabel('UMAP Dimension 2')
    plt.colorbar(scatter2, ax=ax2, label='Cluster')
    
    plt.tight_layout()
    plt.savefig(f"{titre.replace(' ', '_').lower()}_comparison.png", dpi=300)
    plt.show()
    
    # Calcul des métriques de qualité
    metrics = {
        'Texte seul': {},
        'Embeddings enrichis': {}
    }
    
    # Score de silhouette
    if n_clusters_text > 1:
        mask_text = clusters_text != -1
        if mask_text.sum() > n_clusters_text:
            silhouette_text = silhouette_score(text_2d[mask_text], clusters_text[mask_text])
            metrics['Texte seul']['Silhouette'] = silhouette_text
    
    if n_clusters_enriched > 1:
        mask_enriched = clusters_enriched != -1
        if mask_enriched.sum() > n_clusters_enriched:
            silhouette_enriched = silhouette_score(enriched_2d[mask_enriched], clusters_enriched[mask_enriched])
            metrics['Embeddings enrichis']['Silhouette'] = silhouette_enriched
    
    # Indice de Rand ajusté (si labels disponibles)
    if labels is not None:
        if n_clusters_text > 1:
            mask_text = clusters_text != -1
            if mask_text.sum() > 0:
                ari_text = adjusted_rand_score(labels[mask_text], clusters_text[mask_text])
                metrics['Texte seul']['ARI'] = ari_text
        
        if n_clusters_enriched > 1:
            mask_enriched = clusters_enriched != -1
            if mask_enriched.sum() > 0:
                ari_enriched = adjusted_rand_score(labels[mask_enriched], clusters_enriched[mask_enriched])
                metrics['Embeddings enrichis']['ARI'] = ari_enriched
    
    # Affichage des métriques
    print("\nMétriques de qualité du clustering:")
    print("  Texte seul:")
    for metric, value in metrics['Texte seul'].items():
        print(f"    - {metric}: {value:.4f}")
    
    print("  Embeddings enrichis:")
    for metric, value in metrics['Embeddings enrichis'].items():
        print(f"    - {metric}: {value:.4f}")
    
    return {
        'clusters_text': clusters_text,
        'clusters_enriched': clusters_enriched,
        'text_2d': text_2d,
        'enriched_2d': enriched_2d,
        'metrics': metrics
    }

# =========== FONCTION PRINCIPALE PHASE 1 COMPLÈTE ===========

def executer_phase1_complete(chemin_metis, chemin_tickets_fiables):
    """
    Exécute la phase 1 complète: préparation, exploration et analyse catégorielle
    """
    print("\n=============== PHASE 1: PRÉPARATION ET EXPLORATION DES DONNÉES ===============\n")
    
    print("1. Chargement des données...")
    df_metis, df_fiable_brut = charger_donnees(chemin_metis, chemin_tickets_fiables)
    
    print("\n2. Préparation des données...")
    df_prep = preparer_donnees(df_metis)
    
    # Isoler les tickets fiables déjà préparés
    df_fiable = df_prep[df_prep['est_fiable']].copy()
    
    print("\n3. Chargement du modèle CamemBERT...")
    tokenizer, modele = charger_modele_local()
    
    print("\n4. Génération des embeddings pour les tickets fiables...")
    textes_fiables = df_fiable['notes_resolution_nettoyees'].tolist()
    embeddings_fiables = generer_embeddings(textes_fiables, tokenizer, modele, batch_size=BATCH_SIZE)
    
    print("\n5. Visualisation des embeddings des tickets fiables...")
    embeddings_2d = visualiser_embeddings(
        embeddings_fiables, 
        df_fiable['cause_encoded'],
        "Embeddings des tickets fiables colorés par cause"
    )
    
    print("\n6. Analyse des distributions dans les tickets fiables...")
    resultats_analyse = analyser_tickets_fiables(df_prep)
    
    print("\n7. Analyse initiale des clusters naturels...")
    clusters = analyser_structure_clusters_initiaux(embeddings_2d, df_fiable['cause_encoded'])
    
    # ===== PARTIE MANQUANTE : ANALYSE DES CARACTÉRISTIQUES CATÉGORIELLES =====
    print("\n8. Analyse de l'importance des variables catégorielles...")
    mi_scores = analyser_importance_variables_categorielles(df_fiable)
    
    print("\n9. Visualisation des relations entre variables catégorielles et causes...")
    visualiser_relations_categories_causes(df_fiable)
    
    print("\n10. Enrichissement des embeddings avec les caractéristiques catégorielles...")
    # Liste des variables catégorielles pour l'enrichissement
    cat_vars = ['Priorité', 'Service métier', 'Cat1', 'Cat2', 'Groupe affecté']
    
    # Création des embeddings enrichis
    embeddings_enrichis = enrichir_embeddings(
        embeddings=embeddings_fiables,
        df=df_fiable,
        cat_vars=cat_vars,
        normalisation=True
    )
    
    print("\n11. Comparaison des résultats de clustering avec et sans variables catégorielles...")
    resultats_comparaison = comparer_clustering(
        embeddings_texte=embeddings_fiables,
        embeddings_enrichis=embeddings_enrichis,
        labels=df_fiable['cause_encoded'].values,
        titre="Comparaison de clustering"
    )
    
    print("\n=============== PHASE 1 TERMINÉE AVEC SUCCÈS ===============")
    
    # Sauvegarder les résultats intermédiaires
    df_prep.to_csv("resultats/df_prep.csv", index=False)
    df_fiable.to_csv("resultats/df_fiable.csv", index=False)
    np.save("resultats/embeddings_fiables.npy", embeddings_fiables)
    np.save("resultats/embeddings_enrichis.npy", embeddings_enrichis)
    np.save("resultats/embeddings_2d.npy", embeddings_2d)
    np.save("resultats/clusters.npy", clusters)
    
    # Retourner les résultats pour utilisation dans les phases suivantes
    return {
        'df_prep': df_prep,
        'df_fiable': df_fiable,
        'embeddings_fiables': embeddings_fiables,
        'embeddings_enrichis': embeddings_enrichis,
        'embeddings_2d': embeddings_2d,
        'clusters': clusters,
        'resultats_analyse': resultats_analyse,
        'mi_scores': mi_scores,
        'resultats_comparaison': resultats_comparaison
    }

# =========== EXÉCUTION DU SCRIPT ===========

if __name__ == "__main__":
    # Vérifier si les modèles sont bien téléchargés
    if not os.path.exists(MODEL_PATH):
        print(f"ERREUR: Le dossier {MODEL_PATH} n'existe pas!")
        print("Veuillez créer ce dossier et y télécharger les fichiers CamemBERT depuis Hugging Face.")
        exit(1)
        
    resultats = executer_phase1_complete(
        chemin_metis="metis_tickets.csv",
        chemin_tickets_fiables="gdp_tickets.csv"
    )
    
    print("\nRésumé des résultats:")
    print(f"- Nombre total de tickets traités: {len(resultats['df_prep'])}")
    print(f"- Nombre de tickets fiables: {len(resultats['df_fiable'])}")
    print(f"- Dimensions des embeddings textuels: {resultats['embeddings_fiables'].shape}")
    print(f"- Dimensions des embeddings enrichis: {resultats['embeddings_enrichis'].shape}")
    
    # Suggérer la prochaine étape
    print("\nPhase 1 terminée avec succès. Vous pouvez maintenant passer à la Phase 2 (développement du modèle de clustering).")clustering 
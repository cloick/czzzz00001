# 2. Chargement et exploration des données

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
# Suppression de l'import pyLDAvis
from wordcloud import WordCloud
import warnings

# Ignorer les avertissements
warnings.filterwarnings('ignore')

# Téléchargement des ressources NLTK nécessaires
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Chargement du fichier Excel généré
file_path = 'enquetes_satisfaction.xlsx'
df = pd.read_excel(file_path)

# Conservation de l'index d'origine pour la traçabilité
df['ligne_source'] = df.index

# Affichage des premières lignes pour vérification
print("Aperçu des données :")
print(df.head())

# Informations sur le DataFrame
print("\nInformations sur le dataset :")
print(f"Nombre d'entrées : {df.shape[0]}")
print(f"Nombre de colonnes : {df.shape[1]}")

# Distribution des humeurs
print("\nDistribution des humeurs :")
humeur_counts = df['Humeur'].value_counts()
print(humeur_counts)

# Visualisation de la distribution des humeurs
plt.figure(figsize=(10, 6))
sns.countplot(x='Humeur', data=df, order=humeur_counts.index)
plt.title('Distribution des humeurs')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Filtrage des données négatives (Plutôt Insatisfaite et Très insatisfaite)
df_negatif = df[df['Humeur'].isin(['Plutôt Insatisfaite', 'Très insatisfaite'])]
print(f"\nNombre d'entrées négatives : {df_negatif.shape[0]}")

# Distribution des périmètres dans les retours négatifs
print("\nDistribution des périmètres dans les retours négatifs :")
print(df_negatif['Perimetre'].value_counts())

# Visualisation des périmètres les plus problématiques
plt.figure(figsize=(12, 6))
sns.countplot(x='Perimetre', data=df_negatif)
plt.title('Périmètres avec le plus de retours négatifs')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Exploration des statuts de traitement des retours négatifs
plt.figure(figsize=(10, 6))
sns.countplot(x='Statut', data=df_negatif)
plt.title('Statut de traitement des retours négatifs')
plt.tight_layout()
plt.show()

# Distribution des sous-périmètres dans les retours négatifs
if 'Sous perimetre' in df_negatif.columns:
    plt.figure(figsize=(12, 6))
    sous_perimetre_counts = df_negatif['Sous perimetre'].value_counts().head(10)  # Top 10 pour lisibilité
    sns.barplot(x=sous_perimetre_counts.values, y=sous_perimetre_counts.index)
    plt.title('Top 10 des sous-périmètres avec le plus de retours négatifs')
    plt.tight_layout()
    plt.show()

# Analyse temporelle des retours négatifs
if 'Date de l\'enquete' in df_negatif.columns:
    # Conversion en datetime si nécessaire
    if not pd.api.types.is_datetime64_any_dtype(df_negatif['Date de l\'enquete']):
        df_negatif['Date de l\'enquete'] = pd.to_datetime(df_negatif['Date de l\'enquete'], errors='coerce')
    
    # Agrégation par mois
    df_negatif['mois'] = df_negatif['Date de l\'enquete'].dt.to_period('M')
    trend_data = df_negatif.groupby('mois').size()
    
    plt.figure(figsize=(12, 6))
    trend_data.plot(kind='bar', color='salmon')
    plt.title('Évolution temporelle des retours négatifs')
    plt.xlabel('Mois')
    plt.ylabel('Nombre de retours négatifs')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Comparaison des humeurs par périmètre
humeur_perimetre = pd.crosstab(df['Perimetre'], df['Humeur'])
humeur_perimetre_pct = humeur_perimetre.div(humeur_perimetre.sum(axis=1), axis=0) * 100

plt.figure(figsize=(14, 8))
humeur_perimetre_pct.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Distribution des humeurs par périmètre (%)')
plt.xlabel('Périmètre')
plt.ylabel('Pourcentage')
plt.legend(title='Humeur')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Comptage des mots dans les verbatims négatifs avant nettoyage
if 'Verbatim' in df_negatif.columns:
    all_words = ' '.join(df_negatif['Verbatim'].dropna()).lower().split()
    word_counts = Counter(all_words)
    common_words = dict(word_counts.most_common(20))
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(common_words.values()), y=list(common_words.keys()))
    plt.title('Mots les plus fréquents dans les verbatims négatifs (avant nettoyage)')
    plt.tight_layout()
    plt.show()
    
    
    
    
#3. Prétraitement des textes pour l'analyse


import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Fonction de prétraitement des textes simplifiée sans NLTK
def preprocess_text_simple(text):
    if not isinstance(text, str):
        return ""
    
    # Conversion en minuscules
    text = text.lower()
    
    # Suppression des caractères spéciaux et des chiffres
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Tokenisation simple par espace
    tokens = text.split()
    
    # Suppression des mots très courts
    tokens = [token for token in tokens if len(token) > 2]
    
    # Liste manuelle des mots vides français courants
    stopwords_fr = ["les", "des", "est", "dans", "pour", "par", "pas", "avec", "sont", "ont", 
                   "mais", "comme", "tout", "plus", "autre", "autres", "nous", "vous", "ils", 
                   "elles", "leur", "cette", "ces", "notre", "nos", "votre", "vos", "elle", 
                   "ils", "elles", "nous", "vous", "leur", "leurs", "mon", "ton", "son", "mes", 
                   "tes", "ses", "qui", "que", "quoi", "dont", "où", "quand", "comment", 
                   "pourquoi", "lequel", "auquel", "duquel", "une", "deux", "trois", "quatre", 
                   "cinq", "six", "sept", "huit", "neuf", "dix", "été", "être", "avoir", "fait", 
                   "faire", "dit", "dire", "cela", "ceci", "celui", "celle", "ceux", "celles", 
                   "très", "peu", "beaucoup", "trop", "bien", "mal", "tous", "toutes", "tout", 
                   "toute", "rien", "chaque", "plusieurs", "certains", "certaines", "même", "aux", 
                   "sur", "sous", "entre", "vers", "chez", "sans", "avant", "après", "pendant", 
                   "depuis", "jusqu", "contre", "malgré", "sauf", "hors", "selon", "ainsi", "alors", 
                   "aussi", "donc", "puis", "ensuite", "enfin", "encore", "toujours", "jamais", 
                   "souvent", "parfois"]
    
    # Filtrage des mots vides
    tokens = [token for token in tokens if token not in stopwords_fr]
    
    return ' '.join(tokens)

# Application du prétraitement aux verbatims négatifs
df_negatif['verbatim_clean'] = df_negatif['Verbatim'].apply(preprocess_text_simple)

# Vérification des premiers verbatims nettoyés
print("\nExemples de verbatims nettoyés :")
for original, cleaned in zip(df_negatif['Verbatim'].head(), df_negatif['verbatim_clean'].head()):
    print(f"Original: {original}")
    print(f"Nettoyé: {cleaned}")
    print()

# Visualisation des mots les plus fréquents dans les verbatims négatifs
all_words = ' '.join(df_negatif['verbatim_clean'].dropna()).split()
word_counts = Counter(all_words)
top_words = dict(word_counts.most_common(30))

plt.figure(figsize=(12, 8))
sns.barplot(x=list(top_words.values()), y=list(top_words.keys()), orient='h')
plt.title('Mots les plus fréquents dans les verbatims négatifs')
plt.xlabel('Nombre d\'occurrences')
plt.tight_layout()
plt.show()

# Création d'un nuage de mots
wordcloud = WordCloud(width=800, 
                     height=400, 
                     background_color='white', 
                     max_words=100, 
                     contour_width=1, 
                     contour_color='steelblue',
                     collocations=False)  # Évite les duplications de bigrammes

wordcloud.generate(' '.join(df_negatif['verbatim_clean']))

plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nuage de mots des verbatims négatifs')
plt.show()

# Analyse de la longueur des verbatims
df_negatif['longueur_verbatim'] = df_negatif['Verbatim'].str.len()
df_negatif['nombre_mots'] = df_negatif['verbatim_clean'].str.split().str.len()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df_negatif['longueur_verbatim'], bins=20, kde=True)
plt.title('Distribution de la longueur des verbatims (caractères)')
plt.xlabel('Nombre de caractères')

plt.subplot(1, 2, 2)
sns.histplot(df_negatif['nombre_mots'], bins=15, kde=True)
plt.title('Distribution du nombre de mots par verbatim')
plt.xlabel('Nombre de mots')

plt.tight_layout()
plt.show()

print("\nStatistiques sur les verbatims:")
print(f"Longueur moyenne (caractères): {df_negatif['longueur_verbatim'].mean():.1f}")
print(f"Nombre moyen de mots: {df_negatif['nombre_mots'].mean():.1f}")
print(f"Verbatim le plus court: {df_negatif['nombre_mots'].min()} mots")
print(f"Verbatim le plus long: {df_negatif['nombre_mots'].max()} mots")


# 4. Extraction de n-grammes fréquents

# Fonction pour extraire les n-grammes les plus fréquents
def extract_ngrams(corpus, n_gram_range=(1, 3), top_n=20):
    vectorizer = CountVectorizer(ngram_range=n_gram_range)
    X = vectorizer.fit_transform(corpus)
    
    # Extraction des noms des n-grammes
    features = vectorizer.get_feature_names_out()
    
    # Somme des occurrences pour chaque n-gramme
    sums = X.sum(axis=0).A1
    
    # Création d'un dictionnaire {n-gramme: nombre d'occurrences}
    ngrams_counts = {features[i]: sums[i] for i in range(len(features))}
    
    # Tri par fréquence décroissante et sélection des top_n
    top_ngrams = dict(sorted(ngrams_counts.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    return top_ngrams

# Extraction des n-grammes les plus fréquents (unigrammes, bigrammes et trigrammes)
top_unigrams = extract_ngrams(df_negatif['verbatim_clean'].dropna(), (1, 1))
top_bigrams = extract_ngrams(df_negatif['verbatim_clean'].dropna(), (2, 2))
top_trigrams = extract_ngrams(df_negatif['verbatim_clean'].dropna(), (3, 3))

# Visualisation des unigrammes les plus fréquents
plt.figure(figsize=(12, 8))
sns.barplot(x=list(top_unigrams.values()), y=list(top_unigrams.keys()))
plt.title('Unigrammes les plus fréquents dans les verbatims négatifs')
plt.tight_layout()
plt.show()

# Visualisation des bigrammes les plus fréquents
plt.figure(figsize=(12, 8))
sns.barplot(x=list(top_bigrams.values()), y=list(top_bigrams.keys()))
plt.title('Bigrammes les plus fréquents dans les verbatims négatifs')
plt.tight_layout()
plt.show()

# Visualisation des trigrammes les plus fréquents
plt.figure(figsize=(12, 8))
sns.barplot(x=list(top_trigrams.values()), y=list(top_trigrams.keys()))
plt.title('Trigrammes les plus fréquents dans les verbatims négatifs')
plt.tight_layout()
plt.show()


# 5. Modélisation thématique (Topic Modeling)




# Préparation des données pour la modélisation thématique
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Fonctions de visualisation des thèmes avec Matplotlib
def visualiser_themes_lda(lda_model, feature_names, n_top_words=10, figsize=(20, 15)):
    """
    Visualise les thèmes LDA avec Matplotlib
    
    Args:
        lda_model: Modèle LDA entrainé
        feature_names: Noms des features (mots) utilisés
        n_top_words: Nombre de mots à afficher par thème
        figsize: Taille de la figure
    """
    # Nombre de thèmes
    n_topics = len(lda_model.components_)
    
    # Création de la figure
    fig, axes = plt.subplots(int(np.ceil(n_topics/2)), 2, figsize=figsize)
    if n_topics == 1:
        axes = np.array([axes])  # Assurer que axes est toujours un tableau
    axes = axes.flatten()  # Conversion en tableau 1D pour faciliter l'indexation
    
    # Pour chaque thème
    for topic_idx, topic in enumerate(lda_model.components_):
        # Récupération des mots les plus importants
        top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        top_weights = [topic[i] for i in top_words_idx]
        
        # Normalisation des poids pour une meilleure visualisation
        top_weights = [w/sum(top_weights) for w in top_weights]
        
        # Création du graphique à barres horizontales
        ax = axes[topic_idx]
        ax.barh(top_words, top_weights, align='center', color='steelblue', alpha=0.8)
        ax.set_title(f'Thème {topic_idx+1}', fontsize=14)
        ax.set_xlabel('Poids relatif', fontsize=12)
        ax.set_xlim(0, max(top_weights) * 1.2)  # Ajustement de l'échelle
        ax.invert_yaxis()  # Pour avoir le mot le plus important en haut
        
    # Suppression des axes inutilisés si nombre impair de thèmes
    for i in range(n_topics, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.suptitle('Visualisation des thèmes LDA', fontsize=16, y=1.02)
    plt.show()
    
    # Retour du résumé textuel des thèmes
    themes_resume = {}
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        themes_resume[f"Thème {topic_idx+1}"] = ", ".join(top_words)
    
    return themes_resume

def visualiser_distribution_themes(doc_topic_distrib):
    """
    Visualise la distribution des documents par thème dominant
    
    Args:
        doc_topic_distrib: Matrice de distribution des documents par thème
    """
    # Détermination du thème dominant pour chaque document
    dominant_topics = doc_topic_distrib.argmax(axis=1) + 1  # +1 pour commencer à 1
    
    # Comptage des occurrences
    topic_counts = Counter(dominant_topics)
    
    # Tri des thèmes par ordre numérique
    sorted_topics = sorted(topic_counts.items())
    topics, counts = zip(*sorted_topics) if sorted_topics else ([], [])
    
    # Création du graphique
    plt.figure(figsize=(12, 6))
    plt.bar(topics, counts, color='steelblue', alpha=0.8)
    plt.title('Distribution des documents par thème dominant', fontsize=14)
    plt.xlabel('Numéro de thème', fontsize=12)
    plt.ylabel('Nombre de documents', fontsize=12)
    plt.xticks(topics)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return dict(sorted_topics)

def visualiser_matrice_terme_theme(lda_model, feature_names, n_terms=15):
    """
    Visualise la matrice terme-thème sous forme de heatmap
    
    Args:
        lda_model: Modèle LDA entrainé
        feature_names: Noms des features (mots)
        n_terms: Nombre de termes à inclure par thème
    """
    # Nombre de thèmes
    n_topics = len(lda_model.components_)
    
    # Pour chaque thème, récupérer les n_terms termes les plus importants
    top_terms = []
    term_weights = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_idx = topic.argsort()[:-n_terms-1:-1]
        for idx in top_idx:
            if feature_names[idx] not in top_terms:
                top_terms.append(feature_names[idx])
    
    # Limiter le nombre de termes pour éviter une visualisation trop chargée
    top_terms = top_terms[:min(50, len(top_terms))]
    
    # Créer une matrice pour la heatmap
    heatmap_matrix = np.zeros((len(top_terms), n_topics))
    
    # Remplir la matrice avec les poids des termes
    for term_idx, term in enumerate(top_terms):
        term_feature_idx = np.where(feature_names == term)[0][0]
        for topic_idx, topic in enumerate(lda_model.components_):
            heatmap_matrix[term_idx, topic_idx] = topic[term_feature_idx]
    
    # Normalisation par colonne
    col_sums = heatmap_matrix.sum(axis=0)
    heatmap_matrix = heatmap_matrix / col_sums[np.newaxis, :]
    
    # Création de la heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        heatmap_matrix,
        yticklabels=top_terms,
        xticklabels=[f'Thème {i+1}' for i in range(n_topics)],
        cmap='YlGnBu',
        annot=False,
        linewidths=.5
    )
    plt.title('Importance des termes par thème', fontsize=14)
    plt.tight_layout()
    plt.show()

# Exécution de la modélisation thématique
# Vecteurisation des textes (avec TF-IDF)
vectorizer = TfidfVectorizer(
    max_df=0.95,
    min_df=2,
    max_features=1000,
    ngram_range=(1, 2)
)

# Appliqué uniquement sur les verbatims nettoyés non vides
valid_docs = df_negatif['verbatim_clean'].dropna().replace('', np.nan).dropna()

if len(valid_docs) == 0:
    print("Aucun document valide pour la modélisation thématique. Vérifiez le prétraitement.")
else:
    print(f"Modélisation thématique sur {len(valid_docs)} documents valides...")
    
    # Transformation des textes en matrice TF-IDF
    X = vectorizer.fit_transform(valid_docs)
    
    # Extraction des noms des features (mots et bigrammes)
    feature_names = vectorizer.get_feature_names_out()
    print(f"Nombre de features extraites : {len(feature_names)}")
    
    # Détermination du nombre optimal de thèmes
    n_topics = min(5, len(valid_docs) // 5)  # Règle empirique simple
    n_topics = max(2, n_topics)  # Au moins 2 thèmes
    print(f"Nombre de thèmes sélectionné : {n_topics}")
    
    # Création et entraînement du modèle LDA - CORRECTION ICI
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=20,
        learning_method='online',
        random_state=42,
        n_jobs=1  # Important: Utiliser un seul cœur pour éviter les erreurs de sérialisation
    )
    
    lda.fit(X)
    
    # Évaluation du modèle
    perplexity = lda.perplexity(X)
    print(f"Perplexité du modèle : {perplexity:.2f} (plus la valeur est basse, meilleur est le modèle)")
    
    # Visualisation des thèmes
    print("\nThèmes identifiés avec les termes les plus importants :")
    topics = visualiser_themes_lda(lda, feature_names)
    
    # Affichage textuel des thèmes
    for theme, terms in topics.items():
        print(f"{theme}: {terms}")
    
    # Transformation des documents en distribution de thèmes
    doc_topic_distrib = lda.transform(X)
    
    # Visualisation de la distribution des documents par thème
    print("\nDistribution des documents par thème :")
    topic_distribution = visualiser_distribution_themes(doc_topic_distrib)
    
    try:
        # Visualisation de la matrice terme-thème
        print("\nImportance des termes par thème :")
        visualiser_matrice_terme_theme(lda, feature_names)
    except Exception as e:
        print(f"Erreur lors de la visualisation de la matrice terme-thème: {e}")
    
    # Attribution du thème dominant à chaque document
    df_negatif_topics = df_negatif.loc[valid_docs.index].copy()
    df_negatif_topics['dominant_topic'] = doc_topic_distrib.argmax(axis=1) + 1
    
    # Analyse des thèmes par périmètre (avec gestion des cas où la colonne n'existe pas)
    if 'Perimetre' in df_negatif_topics.columns:
        print("\nDistribution des thèmes par périmètre :")
        theme_perimetre = pd.crosstab(df_negatif_topics['dominant_topic'], df_negatif_topics['Perimetre'])
        print(theme_perimetre)
        
        # Visualisation de la distribution des thèmes par périmètre
        plt.figure(figsize=(14, 8))
        theme_perimetre_pct = theme_perimetre.div(theme_perimetre.sum(axis=0), axis=1) * 100
        theme_perimetre_pct.plot(kind='bar', stacked=True, colormap='viridis')
        plt.title('Distribution des thèmes par périmètre (%)', fontsize=14)
        plt.xlabel('Thème dominant', fontsize=12)
        plt.ylabel('Pourcentage', fontsize=12)
        plt.legend(title='Périmètre')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    # Exemples de verbatims pour chaque thème
    print("\nExemples de verbatims par thème :")
    for theme in sorted(df_negatif_topics['dominant_topic'].unique()):
        theme_docs = df_negatif_topics[df_negatif_topics['dominant_topic'] == theme]
        print(f"\nThème {theme} - {topics[f'Thème {theme}']} ({len(theme_docs)} documents)")
        print("-" * 80)
        # Affichage de quelques exemples
        for idx, row in theme_docs.sample(min(3, len(theme_docs))).iterrows():
            print(f"Ligne {row['ligne_source']} - Humeur: {row['Humeur']}")
            if 'Perimetre' in row:
                print(f"Périmètre: {row['Perimetre']}")
            print(f"Verbatim: {row['Verbatim']}")
            print("-" * 80)
            
            
# 6. Analyse de cooccurrences et clustering des verbatims


# Création d'une matrice de co-occurrences
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage

# Si nous avons suffisamment de documents valides
if len(valid_docs) >= 5:  # Nombre arbitraire pour assurer un minimum de données
    # Calcul de la similarité cosinus entre les documents
    similarity_matrix = cosine_similarity(X)
    
    # Clustering hiérarchique
    Z = linkage(similarity_matrix, 'ward')
    
    # Visualisation du dendrogramme
    plt.figure(figsize=(12, 8))
    dendrogram(Z, truncate_mode='lastp', p=10, leaf_rotation=90.)
    plt.title('Dendrogramme des verbatims négatifs (similarité de contenu)')
    plt.xlabel('Verbatim ID')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()
    
    # Application d'un seuil pour déterminer les clusters
    from scipy.cluster.hierarchy import fcluster
    max_d = 1.0  # Distance maximum pour former un cluster
    clusters = fcluster(Z, max_d, criterion='distance')
    
    # Ajout des clusters aux données
    df_negatif_topics['cluster'] = np.nan
    df_negatif_topics.loc[valid_docs.index, 'cluster'] = clusters
    
    # Analyse des clusters
    cluster_counts = df_negatif_topics['cluster'].value_counts().sort_index()
    print("\nDistribution des verbatims par cluster :")
    print(cluster_counts)
    
    # Pour chaque cluster, afficher quelques exemples de verbatims
    print("\nExemples de verbatims par cluster :")
    for cluster_id in sorted(df_negatif_topics['cluster'].dropna().unique()):
        cluster_docs = df_negatif_topics[df_negatif_topics['cluster'] == cluster_id]
        print(f"\nCluster {int(cluster_id)} (n={len(cluster_docs)}):")
        for idx, row in cluster_docs.head(3).iterrows():
            print(f"- Ligne {row['ligne_source']}: {row['Verbatim']}")
else:
    print("Nombre insuffisant de documents pour l'analyse de cooccurrences et le clustering.")
    
    
    


# 7. Création d'un tableau de bord pour l'analyse des problématiques récurrentes


# Consolidation des résultats pour créer un tableau de bord des problématiques récurrentes
if 'dominant_topic' in df_negatif_topics.columns:
    # Création d'un DataFrame résumant les problématiques par thème
    theme_summary = pd.DataFrame()
    
    for theme_id in sorted(df_negatif_topics['dominant_topic'].unique()):
        theme_docs = df_negatif_topics[df_negatif_topics['dominant_topic'] == theme_id]
        
        # Top mots-clés pour ce thème
        theme_key_words = topics[f"Thème {theme_id}"]
        
        # Exemples de verbatims pour ce thème
        verbatim_examples = theme_docs['Verbatim'].sample(min(3, len(theme_docs))).tolist()
        
        # Statistiques par périmètre
        perimetre_counts = theme_docs['Perimetre'].value_counts().to_dict()
        top_perimetre = max(perimetre_counts.items(), key=lambda x: x[1])[0] if perimetre_counts else "N/A"
        
        # Informations sur les lignes sources
        lignes_sources = theme_docs['ligne_source'].tolist()
        
        # Construire l'entrée pour ce thème
        theme_entry = {
            'Thème ID': theme_id,
            'Mots-clés': theme_key_words,
            'Nombre de verbatims': len(theme_docs),
            'Périmètre principal': top_perimetre,
            'Exemples de verbatims': verbatim_examples,
            'Numéros de lignes sources': lignes_sources
        }
        
        # Ajouter à notre résumé
        theme_summary = pd.concat([theme_summary, pd.DataFrame([theme_entry])], ignore_index=True)
    
    # Sauvegarde du tableau de bord dans un fichier Excel
    dashboard_file = 'dashboard_problematiques_recurrentes.xlsx'
    theme_summary.to_excel(dashboard_file, index=False)
    print(f"\nTableau de bord des problématiques récurrentes sauvegardé dans '{dashboard_file}'")
    
    # Affichage du tableau de bord
    print("\nTableau de bord des problématiques récurrentes :")
    print(theme_summary[['Thème ID', 'Mots-clés', 'Nombre de verbatims', 'Périmètre principal']])
    
    # Pour chaque thème, afficher les exemples et les lignes sources
    for idx, row in theme_summary.iterrows():
        print(f"\nThème {int(row['Thème ID'])} - {row['Mots-clés']}")
        print(f"Périmètre principal: {row['Périmètre principal']}")
        print("Exemples de verbatims:")
        for i, example in enumerate(row['Exemples de verbatims'], 1):
            print(f"  {i}. {example}")
        print(f"Lignes sources: {', '.join(map(str, row['Numéros de lignes sources'][:10]))}{'...' if len(row['Numéros de lignes sources']) > 10 else ''}")
else:
    print("La modélisation thématique n'a pas été effectuée. Impossible de créer le tableau de bord.")
    
    

# 8. Fonction pour rechercher des verbatims par mots-clés


# Fonction pour rechercher des verbatims contenant certains mots-clés
def search_verbatims(dataframe, keywords, humeur_filter=None):
    """
    Recherche des verbatims contenant des mots-clés spécifiques.
    
    Args:
        dataframe: Le DataFrame contenant les données
        keywords: Liste de mots-clés à rechercher
        humeur_filter: Liste des humeurs à filtrer (None pour toutes)
    
    Returns:
        DataFrame contenant les verbatims correspondants
    """
    # Application du filtre d'humeur si spécifié
    if humeur_filter:
        df_filtered = dataframe[dataframe['Humeur'].isin(humeur_filter)]
    else:
        df_filtered = dataframe.copy()
    
    # Création d'une expression régulière pour la recherche
    pattern = '|'.join(keywords)
    
    # Recherche dans les verbatims
    mask = df_filtered['Verbatim'].str.contains(pattern, case=False, na=False, regex=True)
    results = df_filtered[mask].copy()
    
    return results

# Exemple d'utilisation de la fonction de recherche
keywords_exemple = ['communication', 'délai', 'support']
humeurs_negatives = ['Plutôt Insatisfaite', 'Très insatisfaite']

resultats_recherche = search_verbatims(df, keywords_exemple, humeurs_negatives)

print(f"\nRésultats de la recherche pour les mots-clés {keywords_exemple} dans les avis négatifs:")
print(f"Nombre de résultats: {len(resultats_recherche)}")

if len(resultats_recherche) > 0:
    print("\nExemples de verbatims trouvés:")
    for idx, row in resultats_recherche.head(5).iterrows():
        print(f"Ligne {row['ligne_source']} - {row['Humeur']}: {row['Verbatim']}")
        print(f"  Périmètre: {row['Perimetre']} - {row['Sous perimetre']}")
        print(f"  Collaborateur: {row['Collaborateur qui a emis']}")
        print()
        
        


    
    # 2. Chargement et exploration des données

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
# Suppression de l'import pyLDAvis
from wordcloud import WordCloud
import warnings

# Ignorer les avertissements
warnings.filterwarnings('ignore')

# Téléchargement des ressources NLTK nécessaires
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Chargement du fichier Excel généré
file_path = 'enquetes_satisfaction.xlsx'
df = pd.read_excel(file_path)

# Conservation de l'index d'origine pour la traçabilité
df['ligne_source'] = df.index

# Affichage des premières lignes pour vérification
print("Aperçu des données :")
print(df.head())

# Informations sur le DataFrame
print("\nInformations sur le dataset :")
print(f"Nombre d'entrées : {df.shape[0]}")
print(f"Nombre de colonnes : {df.shape[1]}")

# Distribution des humeurs
print("\nDistribution des humeurs :")
humeur_counts = df['Humeur'].value_counts()
print(humeur_counts)

# Visualisation de la distribution des humeurs
plt.figure(figsize=(10, 6))
sns.countplot(x='Humeur', data=df, order=humeur_counts.index)
plt.title('Distribution des humeurs')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Filtrage des données négatives (Plutôt Insatisfaite et Très insatisfaite)
df_negatif = df[df['Humeur'].isin(['Plutôt Insatisfaite', 'Très insatisfaite'])]
print(f"\nNombre d'entrées négatives : {df_negatif.shape[0]}")

# Distribution des périmètres dans les retours négatifs
print("\nDistribution des périmètres dans les retours négatifs :")
print(df_negatif['Perimetre'].value_counts())

# Visualisation des périmètres les plus problématiques
plt.figure(figsize=(12, 6))
sns.countplot(x='Perimetre', data=df_negatif)
plt.title('Périmètres avec le plus de retours négatifs')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Exploration des statuts de traitement des retours négatifs
plt.figure(figsize=(10, 6))
sns.countplot(x='Statut', data=df_negatif)
plt.title('Statut de traitement des retours négatifs')
plt.tight_layout()
plt.show()

# Distribution des sous-périmètres dans les retours négatifs
if 'Sous perimetre' in df_negatif.columns:
    plt.figure(figsize=(12, 6))
    sous_perimetre_counts = df_negatif['Sous perimetre'].value_counts().head(10)  # Top 10 pour lisibilité
    sns.barplot(x=sous_perimetre_counts.values, y=sous_perimetre_counts.index)
    plt.title('Top 10 des sous-périmètres avec le plus de retours négatifs')
    plt.tight_layout()
    plt.show()

# Analyse temporelle des retours négatifs
if 'Date de l\'enquete' in df_negatif.columns:
    # Conversion en datetime si nécessaire
    if not pd.api.types.is_datetime64_any_dtype(df_negatif['Date de l\'enquete']):
        df_negatif['Date de l\'enquete'] = pd.to_datetime(df_negatif['Date de l\'enquete'], errors='coerce')
    
    # Agrégation par mois
    df_negatif['mois'] = df_negatif['Date de l\'enquete'].dt.to_period('M')
    trend_data = df_negatif.groupby('mois').size()
    
    plt.figure(figsize=(12, 6))
    trend_data.plot(kind='bar', color='salmon')
    plt.title('Évolution temporelle des retours négatifs')
    plt.xlabel('Mois')
    plt.ylabel('Nombre de retours négatifs')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Comparaison des humeurs par périmètre
humeur_perimetre = pd.crosstab(df['Perimetre'], df['Humeur'])
humeur_perimetre_pct = humeur_perimetre.div(humeur_perimetre.sum(axis=1), axis=0) * 100

plt.figure(figsize=(14, 8))
humeur_perimetre_pct.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Distribution des humeurs par périmètre (%)')
plt.xlabel('Périmètre')
plt.ylabel('Pourcentage')
plt.legend(title='Humeur')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Comptage des mots dans les verbatims négatifs avant nettoyage
if 'Verbatim' in df_negatif.columns:
    all_words = ' '.join(df_negatif['Verbatim'].dropna()).lower().split()
    word_counts = Counter(all_words)
    common_words = dict(word_counts.most_common(20))
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(common_words.values()), y=list(common_words.keys()))
    plt.title('Mots les plus fréquents dans les verbatims négatifs (avant nettoyage)')
    plt.tight_layout()
    plt.show()
    
    
    
    
#3. Prétraitement des textes pour l'analyse


import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Fonction de prétraitement des textes simplifiée sans NLTK
def preprocess_text_simple(text):
    if not isinstance(text, str):
        return ""
    
    # Conversion en minuscules
    text = text.lower()
    
    # Suppression des caractères spéciaux et des chiffres
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Tokenisation simple par espace
    tokens = text.split()
    
    # Suppression des mots très courts
    tokens = [token for token in tokens if len(token) > 2]
    
    # Liste manuelle des mots vides français courants
    stopwords_fr = ["les", "des", "est", "dans", "pour", "par", "pas", "avec", "sont", "ont", 
                   "mais", "comme", "tout", "plus", "autre", "autres", "nous", "vous", "ils", 
                   "elles", "leur", "cette", "ces", "notre", "nos", "votre", "vos", "elle", 
                   "ils", "elles", "nous", "vous", "leur", "leurs", "mon", "ton", "son", "mes", 
                   "tes", "ses", "qui", "que", "quoi", "dont", "où", "quand", "comment", 
                   "pourquoi", "lequel", "auquel", "duquel", "une", "deux", "trois", "quatre", 
                   "cinq", "six", "sept", "huit", "neuf", "dix", "été", "être", "avoir", "fait", 
                   "faire", "dit", "dire", "cela", "ceci", "celui", "celle", "ceux", "celles", 
                   "très", "peu", "beaucoup", "trop", "bien", "mal", "tous", "toutes", "tout", 
                   "toute", "rien", "chaque", "plusieurs", "certains", "certaines", "même", "aux", 
                   "sur", "sous", "entre", "vers", "chez", "sans", "avant", "après", "pendant", 
                   "depuis", "jusqu", "contre", "malgré", "sauf", "hors", "selon", "ainsi", "alors", 
                   "aussi", "donc", "puis", "ensuite", "enfin", "encore", "toujours", "jamais", 
                   "souvent", "parfois"]
    
    # Filtrage des mots vides
    tokens = [token for token in tokens if token not in stopwords_fr]
    
    return ' '.join(tokens)

# Application du prétraitement aux verbatims négatifs
df_negatif['verbatim_clean'] = df_negatif['Verbatim'].apply(preprocess_text_simple)

# Vérification des premiers verbatims nettoyés
print("\nExemples de verbatims nettoyés :")
for original, cleaned in zip(df_negatif['Verbatim'].head(), df_negatif['verbatim_clean'].head()):
    print(f"Original: {original}")
    print(f"Nettoyé: {cleaned}")
    print()

# Visualisation des mots les plus fréquents dans les verbatims négatifs
all_words = ' '.join(df_negatif['verbatim_clean'].dropna()).split()
word_counts = Counter(all_words)
top_words = dict(word_counts.most_common(30))

plt.figure(figsize=(12, 8))
sns.barplot(x=list(top_words.values()), y=list(top_words.keys()), orient='h')
plt.title('Mots les plus fréquents dans les verbatims négatifs')
plt.xlabel('Nombre d\'occurrences')
plt.tight_layout()
plt.show()

# Création d'un nuage de mots
wordcloud = WordCloud(width=800, 
                     height=400, 
                     background_color='white', 
                     max_words=100, 
                     contour_width=1, 
                     contour_color='steelblue',
                     collocations=False)  # Évite les duplications de bigrammes

wordcloud.generate(' '.join(df_negatif['verbatim_clean']))

plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nuage de mots des verbatims négatifs')
plt.show()

# Analyse de la longueur des verbatims
df_negatif['longueur_verbatim'] = df_negatif['Verbatim'].str.len()
df_negatif['nombre_mots'] = df_negatif['verbatim_clean'].str.split().str.len()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df_negatif['longueur_verbatim'], bins=20, kde=True)
plt.title('Distribution de la longueur des verbatims (caractères)')
plt.xlabel('Nombre de caractères')

plt.subplot(1, 2, 2)
sns.histplot(df_negatif['nombre_mots'], bins=15, kde=True)
plt.title('Distribution du nombre de mots par verbatim')
plt.xlabel('Nombre de mots')

plt.tight_layout()
plt.show()

print("\nStatistiques sur les verbatims:")
print(f"Longueur moyenne (caractères): {df_negatif['longueur_verbatim'].mean():.1f}")
print(f"Nombre moyen de mots: {df_negatif['nombre_mots'].mean():.1f}")
print(f"Verbatim le plus court: {df_negatif['nombre_mots'].min()} mots")
print(f"Verbatim le plus long: {df_negatif['nombre_mots'].max()} mots")


# 4. Extraction de n-grammes fréquents

# Fonction pour extraire les n-grammes les plus fréquents
def extract_ngrams(corpus, n_gram_range=(1, 3), top_n=20):
    vectorizer = CountVectorizer(ngram_range=n_gram_range)
    X = vectorizer.fit_transform(corpus)
    
    # Extraction des noms des n-grammes
    features = vectorizer.get_feature_names_out()
    
    # Somme des occurrences pour chaque n-gramme
    sums = X.sum(axis=0).A1
    
    # Création d'un dictionnaire {n-gramme: nombre d'occurrences}
    ngrams_counts = {features[i]: sums[i] for i in range(len(features))}
    
    # Tri par fréquence décroissante et sélection des top_n
    top_ngrams = dict(sorted(ngrams_counts.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    return top_ngrams

# Extraction des n-grammes les plus fréquents (unigrammes, bigrammes et trigrammes)
top_unigrams = extract_ngrams(df_negatif['verbatim_clean'].dropna(), (1, 1))
top_bigrams = extract_ngrams(df_negatif['verbatim_clean'].dropna(), (2, 2))
top_trigrams = extract_ngrams(df_negatif['verbatim_clean'].dropna(), (3, 3))

# Visualisation des unigrammes les plus fréquents
plt.figure(figsize=(12, 8))
sns.barplot(x=list(top_unigrams.values()), y=list(top_unigrams.keys()))
plt.title('Unigrammes les plus fréquents dans les verbatims négatifs')
plt.tight_layout()
plt.show()

# Visualisation des bigrammes les plus fréquents
plt.figure(figsize=(12, 8))
sns.barplot(x=list(top_bigrams.values()), y=list(top_bigrams.keys()))
plt.title('Bigrammes les plus fréquents dans les verbatims négatifs')
plt.tight_layout()
plt.show()

# Visualisation des trigrammes les plus fréquents
plt.figure(figsize=(12, 8))
sns.barplot(x=list(top_trigrams.values()), y=list(top_trigrams.keys()))
plt.title('Trigrammes les plus fréquents dans les verbatims négatifs')
plt.tight_layout()
plt.show()


# 5. Modélisation thématique (Topic Modeling)




# Préparation des données pour la modélisation thématique
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Fonctions de visualisation des thèmes avec Matplotlib
def visualiser_themes_lda(lda_model, feature_names, n_top_words=10, figsize=(20, 15)):
    """
    Visualise les thèmes LDA avec Matplotlib
    
    Args:
        lda_model: Modèle LDA entrainé
        feature_names: Noms des features (mots) utilisés
        n_top_words: Nombre de mots à afficher par thème
        figsize: Taille de la figure
    """
    # Nombre de thèmes
    n_topics = len(lda_model.components_)
    
    # Création de la figure
    fig, axes = plt.subplots(int(np.ceil(n_topics/2)), 2, figsize=figsize)
    if n_topics == 1:
        axes = np.array([axes])  # Assurer que axes est toujours un tableau
    axes = axes.flatten()  # Conversion en tableau 1D pour faciliter l'indexation
    
    # Pour chaque thème
    for topic_idx, topic in enumerate(lda_model.components_):
        # Récupération des mots les plus importants
        top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        top_weights = [topic[i] for i in top_words_idx]
        
        # Normalisation des poids pour une meilleure visualisation
        top_weights = [w/sum(top_weights) for w in top_weights]
        
        # Création du graphique à barres horizontales
        ax = axes[topic_idx]
        ax.barh(top_words, top_weights, align='center', color='steelblue', alpha=0.8)
        ax.set_title(f'Thème {topic_idx+1}', fontsize=14)
        ax.set_xlabel('Poids relatif', fontsize=12)
        ax.set_xlim(0, max(top_weights) * 1.2)  # Ajustement de l'échelle
        ax.invert_yaxis()  # Pour avoir le mot le plus important en haut
        
    # Suppression des axes inutilisés si nombre impair de thèmes
    for i in range(n_topics, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.suptitle('Visualisation des thèmes LDA', fontsize=16, y=1.02)
    plt.show()
    
    # Retour du résumé textuel des thèmes
    themes_resume = {}
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        themes_resume[f"Thème {topic_idx+1}"] = ", ".join(top_words)
    
    return themes_resume

def visualiser_distribution_themes(doc_topic_distrib):
    """
    Visualise la distribution des documents par thème dominant
    
    Args:
        doc_topic_distrib: Matrice de distribution des documents par thème
    """
    # Détermination du thème dominant pour chaque document
    dominant_topics = doc_topic_distrib.argmax(axis=1) + 1  # +1 pour commencer à 1
    
    # Comptage des occurrences
    topic_counts = Counter(dominant_topics)
    
    # Tri des thèmes par ordre numérique
    sorted_topics = sorted(topic_counts.items())
    topics, counts = zip(*sorted_topics) if sorted_topics else ([], [])
    
    # Création du graphique
    plt.figure(figsize=(12, 6))
    plt.bar(topics, counts, color='steelblue', alpha=0.8)
    plt.title('Distribution des documents par thème dominant', fontsize=14)
    plt.xlabel('Numéro de thème', fontsize=12)
    plt.ylabel('Nombre de documents', fontsize=12)
    plt.xticks(topics)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return dict(sorted_topics)

def visualiser_matrice_terme_theme(lda_model, feature_names, n_terms=15):
    """
    Visualise la matrice terme-thème sous forme de heatmap
    
    Args:
        lda_model: Modèle LDA entrainé
        feature_names: Noms des features (mots)
        n_terms: Nombre de termes à inclure par thème
    """
    # Nombre de thèmes
    n_topics = len(lda_model.components_)
    
    # Pour chaque thème, récupérer les n_terms termes les plus importants
    top_terms = []
    term_weights = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_idx = topic.argsort()[:-n_terms-1:-1]
        for idx in top_idx:
            if feature_names[idx] not in top_terms:
                top_terms.append(feature_names[idx])
    
    # Limiter le nombre de termes pour éviter une visualisation trop chargée
    top_terms = top_terms[:min(50, len(top_terms))]
    
    # Créer une matrice pour la heatmap
    heatmap_matrix = np.zeros((len(top_terms), n_topics))
    
    # Remplir la matrice avec les poids des termes
    for term_idx, term in enumerate(top_terms):
        term_feature_idx = np.where(feature_names == term)[0][0]
        for topic_idx, topic in enumerate(lda_model.components_):
            heatmap_matrix[term_idx, topic_idx] = topic[term_feature_idx]
    
    # Normalisation par colonne
    col_sums = heatmap_matrix.sum(axis=0)
    heatmap_matrix = heatmap_matrix / col_sums[np.newaxis, :]
    
    # Création de la heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        heatmap_matrix,
        yticklabels=top_terms,
        xticklabels=[f'Thème {i+1}' for i in range(n_topics)],
        cmap='YlGnBu',
        annot=False,
        linewidths=.5
    )
    plt.title('Importance des termes par thème', fontsize=14)
    plt.tight_layout()
    plt.show()

# Exécution de la modélisation thématique
# Vecteurisation des textes (avec TF-IDF)
vectorizer = TfidfVectorizer(
    max_df=0.95,
    min_df=2,
    max_features=1000,
    ngram_range=(1, 2)
)

# Appliqué uniquement sur les verbatims nettoyés non vides
valid_docs = df_negatif['verbatim_clean'].dropna().replace('', np.nan).dropna()

if len(valid_docs) == 0:
    print("Aucun document valide pour la modélisation thématique. Vérifiez le prétraitement.")
else:
    print(f"Modélisation thématique sur {len(valid_docs)} documents valides...")
    
    # Transformation des textes en matrice TF-IDF
    X = vectorizer.fit_transform(valid_docs)
    
    # Extraction des noms des features (mots et bigrammes)
    feature_names = vectorizer.get_feature_names_out()
    print(f"Nombre de features extraites : {len(feature_names)}")
    
    # Détermination du nombre optimal de thèmes
    n_topics = min(5, len(valid_docs) // 5)  # Règle empirique simple
    n_topics = max(2, n_topics)  # Au moins 2 thèmes
    print(f"Nombre de thèmes sélectionné : {n_topics}")
    
    # Création et entraînement du modèle LDA - CORRECTION ICI
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=20,
        learning_method='online',
        random_state=42,
        n_jobs=1  # Important: Utiliser un seul cœur pour éviter les erreurs de sérialisation
    )
    
    lda.fit(X)
    
    # Évaluation du modèle
    perplexity = lda.perplexity(X)
    print(f"Perplexité du modèle : {perplexity:.2f} (plus la valeur est basse, meilleur est le modèle)")
    
    # Visualisation des thèmes
    print("\nThèmes identifiés avec les termes les plus importants :")
    topics = visualiser_themes_lda(lda, feature_names)
    
    # Affichage textuel des thèmes
    for theme, terms in topics.items():
        print(f"{theme}: {terms}")
    
    # Transformation des documents en distribution de thèmes
    doc_topic_distrib = lda.transform(X)
    
    # Visualisation de la distribution des documents par thème
    print("\nDistribution des documents par thème :")
    topic_distribution = visualiser_distribution_themes(doc_topic_distrib)
    
    try:
        # Visualisation de la matrice terme-thème
        print("\nImportance des termes par thème :")
        visualiser_matrice_terme_theme(lda, feature_names)
    except Exception as e:
        print(f"Erreur lors de la visualisation de la matrice terme-thème: {e}")
    
    # Attribution du thème dominant à chaque document
    df_negatif_topics = df_negatif.loc[valid_docs.index].copy()
    df_negatif_topics['dominant_topic'] = doc_topic_distrib.argmax(axis=1) + 1
    
    # Analyse des thèmes par périmètre (avec gestion des cas où la colonne n'existe pas)
    if 'Perimetre' in df_negatif_topics.columns:
        print("\nDistribution des thèmes par périmètre :")
        theme_perimetre = pd.crosstab(df_negatif_topics['dominant_topic'], df_negatif_topics['Perimetre'])
        print(theme_perimetre)
        
        # Visualisation de la distribution des thèmes par périmètre
        plt.figure(figsize=(14, 8))
        theme_perimetre_pct = theme_perimetre.div(theme_perimetre.sum(axis=0), axis=1) * 100
        theme_perimetre_pct.plot(kind='bar', stacked=True, colormap='viridis')
        plt.title('Distribution des thèmes par périmètre (%)', fontsize=14)
        plt.xlabel('Thème dominant', fontsize=12)
        plt.ylabel('Pourcentage', fontsize=12)
        plt.legend(title='Périmètre')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    # Exemples de verbatims pour chaque thème
    print("\nExemples de verbatims par thème :")
    for theme in sorted(df_negatif_topics['dominant_topic'].unique()):
        theme_docs = df_negatif_topics[df_negatif_topics['dominant_topic'] == theme]
        print(f"\nThème {theme} - {topics[f'Thème {theme}']} ({len(theme_docs)} documents)")
        print("-" * 80)
        # Affichage de quelques exemples
        for idx, row in theme_docs.sample(min(3, len(theme_docs))).iterrows():
            print(f"Ligne {row['ligne_source']} - Humeur: {row['Humeur']}")
            if 'Perimetre' in row:
                print(f"Périmètre: {row['Perimetre']}")
            print(f"Verbatim: {row['Verbatim']}")
            print("-" * 80)
            
            
# 6. Analyse de cooccurrences et clustering des verbatims


# Création d'une matrice de co-occurrences
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage

# Si nous avons suffisamment de documents valides
if len(valid_docs) >= 5:  # Nombre arbitraire pour assurer un minimum de données
    # Calcul de la similarité cosinus entre les documents
    similarity_matrix = cosine_similarity(X)
    
    # Clustering hiérarchique
    Z = linkage(similarity_matrix, 'ward')
    
    # Visualisation du dendrogramme
    plt.figure(figsize=(12, 8))
    dendrogram(Z, truncate_mode='lastp', p=10, leaf_rotation=90.)
    plt.title('Dendrogramme des verbatims négatifs (similarité de contenu)')
    plt.xlabel('Verbatim ID')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()
    
    # Application d'un seuil pour déterminer les clusters
    from scipy.cluster.hierarchy import fcluster
    max_d = 1.0  # Distance maximum pour former un cluster
    clusters = fcluster(Z, max_d, criterion='distance')
    
    # Ajout des clusters aux données
    df_negatif_topics['cluster'] = np.nan
    df_negatif_topics.loc[valid_docs.index, 'cluster'] = clusters
    
    # Analyse des clusters
    cluster_counts = df_negatif_topics['cluster'].value_counts().sort_index()
    print("\nDistribution des verbatims par cluster :")
    print(cluster_counts)
    
    # Pour chaque cluster, afficher quelques exemples de verbatims
    print("\nExemples de verbatims par cluster :")
    for cluster_id in sorted(df_negatif_topics['cluster'].dropna().unique()):
        cluster_docs = df_negatif_topics[df_negatif_topics['cluster'] == cluster_id]
        print(f"\nCluster {int(cluster_id)} (n={len(cluster_docs)}):")
        for idx, row in cluster_docs.head(3).iterrows():
            print(f"- Ligne {row['ligne_source']}: {row['Verbatim']}")
else:
    print("Nombre insuffisant de documents pour l'analyse de cooccurrences et le clustering.")
    
    
    


# 7. Création d'un tableau de bord pour l'analyse des problématiques récurrentes


# Consolidation des résultats pour créer un tableau de bord des problématiques récurrentes
if 'dominant_topic' in df_negatif_topics.columns:
    # Création d'un DataFrame résumant les problématiques par thème
    theme_summary = pd.DataFrame()
    
    for theme_id in sorted(df_negatif_topics['dominant_topic'].unique()):
        theme_docs = df_negatif_topics[df_negatif_topics['dominant_topic'] == theme_id]
        
        # Top mots-clés pour ce thème
        theme_key_words = topics[f"Thème {theme_id}"]
        
        # Exemples de verbatims pour ce thème
        verbatim_examples = theme_docs['Verbatim'].sample(min(3, len(theme_docs))).tolist()
        
        # Statistiques par périmètre
        perimetre_counts = theme_docs['Perimetre'].value_counts().to_dict()
        top_perimetre = max(perimetre_counts.items(), key=lambda x: x[1])[0] if perimetre_counts else "N/A"
        
        # Informations sur les lignes sources
        lignes_sources = theme_docs['ligne_source'].tolist()
        
        # Construire l'entrée pour ce thème
        theme_entry = {
            'Thème ID': theme_id,
            'Mots-clés': theme_key_words,
            'Nombre de verbatims': len(theme_docs),
            'Périmètre principal': top_perimetre,
            'Exemples de verbatims': verbatim_examples,
            'Numéros de lignes sources': lignes_sources
        }
        
        # Ajouter à notre résumé
        theme_summary = pd.concat([theme_summary, pd.DataFrame([theme_entry])], ignore_index=True)
    
    # Sauvegarde du tableau de bord dans un fichier Excel
    dashboard_file = 'dashboard_problematiques_recurrentes.xlsx'
    theme_summary.to_excel(dashboard_file, index=False)
    print(f"\nTableau de bord des problématiques récurrentes sauvegardé dans '{dashboard_file}'")
    
    # Affichage du tableau de bord
    print("\nTableau de bord des problématiques récurrentes :")
    print(theme_summary[['Thème ID', 'Mots-clés', 'Nombre de verbatims', 'Périmètre principal']])
    
    # Pour chaque thème, afficher les exemples et les lignes sources
    for idx, row in theme_summary.iterrows():
        print(f"\nThème {int(row['Thème ID'])} - {row['Mots-clés']}")
        print(f"Périmètre principal: {row['Périmètre principal']}")
        print("Exemples de verbatims:")
        for i, example in enumerate(row['Exemples de verbatims'], 1):
            print(f"  {i}. {example}")
        print(f"Lignes sources: {', '.join(map(str, row['Numéros de lignes sources'][:10]))}{'...' if len(row['Numéros de lignes sources']) > 10 else ''}")
else:
    print("La modélisation thématique n'a pas été effectuée. Impossible de créer le tableau de bord.")
    
    

# 8. Fonction pour rechercher des verbatims par mots-clés


# Fonction pour rechercher des verbatims contenant certains mots-clés
def search_verbatims(dataframe, keywords, humeur_filter=None):
    """
    Recherche des verbatims contenant des mots-clés spécifiques.
    
    Args:
        dataframe: Le DataFrame contenant les données
        keywords: Liste de mots-clés à rechercher
        humeur_filter: Liste des humeurs à filtrer (None pour toutes)
    
    Returns:
        DataFrame contenant les verbatims correspondants
    """
    # Application du filtre d'humeur si spécifié
    if humeur_filter:
        df_filtered = dataframe[dataframe['Humeur'].isin(humeur_filter)]
    else:
        df_filtered = dataframe.copy()
    
    # Création d'une expression régulière pour la recherche
    pattern = '|'.join(keywords)
    
    # Recherche dans les verbatims
    mask = df_filtered['Verbatim'].str.contains(pattern, case=False, na=False, regex=True)
    results = df_filtered[mask].copy()
    
    return results

# Exemple d'utilisation de la fonction de recherche
keywords_exemple = ['communication', 'délai', 'support']
humeurs_negatives = ['Plutôt Insatisfaite', 'Très insatisfaite']

resultats_recherche = search_verbatims(df, keywords_exemple, humeurs_negatives)

print(f"\nRésultats de la recherche pour les mots-clés {keywords_exemple} dans les avis négatifs:")
print(f"Nombre de résultats: {len(resultats_recherche)}")

if len(resultats_recherche) > 0:
    print("\nExemples de verbatims trouvés:")
    for idx, row in resultats_recherche.head(5).iterrows():
        print(f"Ligne {row['ligne_source']} - {row['Humeur']}: {row['Verbatim']}")
        print(f"  Périmètre: {row['Perimetre']} - {row['Sous perimetre']}")
        print(f"  Collaborateur: {row['Collaborateur qui a emis']}")
        print()
        
        


    
    

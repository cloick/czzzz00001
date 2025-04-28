
Je vais créer un code complet pour une analyse de données textuelles dans un cadre d'enquêtes de satisfaction. La demande comprend plusieurs éléments :

1. Générer un fichier Excel synthétique (données fictives) qui suivra la structure décrite
2. Charger ce fichier Excel 
3. Analyser les verbatims pour trouver des thématiques récurrentes, en se concentrant sur les avis négatifs
4. Permettre la traçabilité pour remonter aux sources des problèmes

Je vais structurer le code en plusieurs sections :

1. Génération des données fictives avec pandas et sauvegarde en format .xlsx
2. Chargement des données
3. Prétraitement des textes (nettoyage, lemmatisation)
4. Analyse exploratoire des données
5. Modélisation thématique (Topic Modeling) avec des méthodes comme LDA ou NMF
6. Extraction de n-grammes fréquents
7. Visualisation des résultats
8. Traçabilité (lien entre thèmes et lignes sources)

Je vais créer un notebook Jupyter complet et bien documenté, avec un style formel et professionnel adapté à un contexte d'entreprise.

# Analyse de verbatims d'enquêtes de satisfaction

Je vais vous fournir un code complet pour réaliser cette analyse textuelle des verbatims, avec une attention particulière sur les retours négatifs. Le code sera organisé en plusieurs sections pour plus de clarté.

## 1. Génération d'un jeu de données synthétique

Commençons par générer un fichier Excel synthétique qui servira pour notre démonstration :

```python
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Définition des valeurs possibles pour chaque colonne
humeurs = ["Très satisfaite", "Plutôt Satisfaite", "Plutôt Insatisfaite", "Très insatisfaite"]
categorisations = ["A dispatcher", "A traiter", "En cours", "Terminé"]
noms = ["Dupont Xavier", "Martin Sophie", "Leblanc Julie", "Durand Thomas", "Petit Claire", 
        "Moreau Nicolas", "Richard Emilie", "Robert Philippe", "Simon Catherine", "Leroy Antoine"]
perimetres = ["Epargne et assurances", "Data", "Crédits", "Banque au quotidien", "Moyens de paiement"]
sous_perimetres = ["Tribu Patrimoine", "Tribu digital workplace", "Tribu innovation", "Tribu crédit immobilier", "Tribu e-banking"]
chapitres = ["Tribu data hub", "Tribu dématérialisation", "Tribu expérience client", "Tribu conformité", "Tribu IT"]
metiers = ["Relai fabrication", "Scrum master", "Product owner", "Développeur", "Analyste fonctionnel", "Chef de projet", "Architecte", "Designer UX"]
types_enquete = ["IRC Flash Vague 1", "IRC Flash Vague 2", "IRC Flash Vague 3", "Enquête trimestrielle"]
equipes = ["IBC", "QOS", "DEV", "OPS", "ARCHI", "SECU"]
responsables = ["Martin Dupond", "Garcia Laura", "Vasseur Jean", "Lefebvre Marie", "Dubois Patrick"]
soutiens = ["Hughes Timhas", "Rousseau Marie", "Bernard David", "Fournier Sylvie", "Lambert François"]
statuts = ["Terminé", "En cours", "A planifier", "Suspendu"]

# Verbatims positifs et négatifs pour la génération
verbatims_positifs = [
    "Réactif et compétent, toujours à l'écoute.",
    "Excellente collaboration, équipe très efficace.",
    "Communication claire et adaptée à nos besoins.",
    "Grande disponibilité et solutions toujours pertinentes.",
    "Respect des délais et qualité du travail remarquable.",
    "Expertise technique impressionnante et bons conseils.",
    "Très bon accompagnement sur notre projet complexe.",
    "Capacité d'adaptation exceptionnelle face aux changements.",
    "Proactivité et anticipation des problématiques.",
    "Support technique efficace et rapide."
]

verbatims_negatifs = [
    "Très mauvaise communication récurrente de la part des Socles CAGIP sur les sujets transverses.",
    "Délais non respectés à plusieurs reprises, impact fort sur nos activités.",
    "Manque de compétence technique évident sur le projet.",
    "Difficultés à obtenir des réponses claires à nos questions.",
    "Documentation insuffisante et non mise à jour.",
    "Problèmes récurrents de qualité dans les livrables.",
    "Absence de suivi après la mise en production.",
    "Manque d'implication et de proactivité de l'équipe.",
    "Changements fréquents dans l'équipe perturbant la continuité.",
    "Outils mis à disposition obsolètes et inefficaces.",
    "Communication inexistante lors des incidents majeurs.",
    "Procédures trop complexes pour des demandes simples.",
    "Temps de réponse beaucoup trop long pour les tickets critiques.",
    "Absence totale de prise en compte de nos contraintes métier.",
    "Support technique défaillant et peu disponible."
]

# Génération de plans d'action
plans_action = [
    "Échange avec le manager de Tribu DWP pour résolution rapide.",
    "Organisation d'un atelier d'amélioration continue.",
    "Mise en place d'un nouveau processus de communication.",
    "Formation spécifique prévue pour l'équipe concernée.",
    "Revue des procédures internes et simplification.",
    "Création d'un groupe de travail dédié à cette problématique.",
    "Renforcement de l'équipe avec des profils plus expérimentés.",
    "Audit des outils et mise à jour prévue.",
    "Réorganisation du support pour améliorer les temps de réponse.",
    "Plan d'amélioration de la qualité en cours d'élaboration."
]

# Synthèses
syntheses = [
    "Florian signale ne pas avoir trouvé de solution malgré plusieurs relances.",
    "Problème identifié et résolu par mise à jour du framework.",
    "Incident majeur ayant nécessité une intervention de la DSI.",
    "Demande traitée avec succès après escalade au N+2.",
    "Problématique récurrente nécessitant une refonte complète.",
    "Solution temporaire mise en place en attendant le nouveau système.",
    "Formation effectuée, amélioration constatée immédiatement.",
    "Analyse approfondie révélant un problème d'infrastructure.",
    "Communication renforcée entre équipes suite à cet incident.",
    "Résolution nécessitant l'intervention de plusieurs services."
]

# Générer des dates sur les 6 derniers mois
date_fin = datetime.now()
date_debut = date_fin - timedelta(days=180)
dates_possibles = [date_debut + timedelta(days=x) for x in range(0, 180, 15)]  # Une enquête tous les 15 jours

# Fonction pour générer un verbatim en fonction de l'humeur
def generer_verbatim(humeur):
    if humeur in ["Très satisfaite", "Plutôt Satisfaite"]:
        return random.choice(verbatims_positifs)
    else:
        return random.choice(verbatims_negatifs)

# Fonction pour générer le plan d'action et la synthèse selon l'humeur et la catégorisation
def generer_plan_et_synthese(humeur, categorisation):
    if humeur in ["Plutôt Insatisfaite", "Très insatisfaite"]:
        plan = random.choice(plans_action) if categorisation != "A dispatcher" else ""
        synthese = random.choice(syntheses) if categorisation in ["En cours", "Terminé"] else ""
    else:
        plan = ""
        synthese = "Retour positif noté" if categorisation in ["En cours", "Terminé"] else ""
    
    return plan, synthese

# Créer un DataFrame avec 200 entrées
n_entries = 200
data = []

for i in range(n_entries):
    humeur = random.choices(humeurs, weights=[0.4, 0.3, 0.2, 0.1], k=1)[0]  # Distribution pondérée
    categorisation = random.choice(categorisations)
    verbatim = generer_verbatim(humeur)
    collaborateur = random.choice(noms)
    perimetre = random.choice(perimetres)
    sous_perimetre = random.choice(sous_perimetres)
    chapitre = random.choice(chapitres)
    metier = random.choice(metiers)
    date_enquete = random.choice(dates_possibles).strftime('%Y-%m-%d')
    type_enquete = random.choice(types_enquete)
    equipe = random.choice(equipes)
    responsable = random.choice(responsables)
    soutien = random.choice(soutiens)
    statut = random.choice(statuts)
    plan_action, synthese = generer_plan_et_synthese(humeur, categorisation)
    
    data.append({
        'Humeur': humeur,
        'Categorisation': categorisation,
        'Verbatim': verbatim,
        'Collaborateur qui a emis': collaborateur,
        'Perimetre': perimetre,
        'Sous perimetre': sous_perimetre,
        'Chapitre de la sous tribu': chapitre,
        'Metier CATS': metier,
        'Date de l\'enquete': date_enquete,
        'Type d\'Enquete': type_enquete,
        'Equipe BFR concernee': equipe,
        'Responsable BFR': responsable,
        'Soutien de BFR': soutien,
        'Statut': statut,
        'Plan d\'action': plan_action,
        'Synthese et/ou plan d\'actions': synthese
    })

# Création du DataFrame
df = pd.DataFrame(data)

# Sauvegarde dans un fichier Excel
output_file = 'enquetes_satisfaction.xlsx'
df.to_excel(output_file, index=True)  # Garder l'index pour avoir le numéro de ligne

print(f"Fichier '{output_file}' généré avec succès contenant {n_entries} entrées.")
```

## 2. Chargement et exploration des données

Maintenant, chargeons le fichier généré et explorons les données :

```python
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
import pyLDAvis
import pyLDAvis.sklearn
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
```

## 3. Prétraitement des textes pour l'analyse

```python
# Fonction de prétraitement des textes
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Conversion en minuscules
    text = text.lower()
    
    # Suppression des caractères spéciaux et des chiffres
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Tokenisation
    tokens = word_tokenize(text)
    
    # Suppression des stopwords (mots vides)
    stop_words = set(stopwords.words('french'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# Application du prétraitement aux verbatims négatifs
df_negatif['verbatim_clean'] = df_negatif['Verbatim'].apply(preprocess_text)

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
sns.barplot(x=list(top_words.values()), y=list(top_words.keys()))
plt.title('Mots les plus fréquents dans les verbatims négatifs')
plt.tight_layout()
plt.show()

# Création d'un nuage de mots
wordcloud = WordCloud(width=800, height=400, background_color='white', 
                     max_words=100, contour_width=1, contour_color='steelblue')
wordcloud.generate(' '.join(df_negatif['verbatim_clean']))

plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nuage de mots des verbatims négatifs')
plt.show()
```

## 4. Extraction de n-grammes fréquents

```python
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
```

## 5. Modélisation thématique (Topic Modeling)

```python
# Préparation des données pour la modélisation thématique
vectorizer = TfidfVectorizer(
    max_df=0.95,         # Ignorer les termes qui apparaissent dans plus de 95% des documents
    min_df=2,            # Ignorer les termes qui apparaissent dans moins de 2 documents
    max_features=1000,   # Limiter le nombre de features
    stop_words='english' # Déjà fait dans le prétraitement, mais ajout de sécurité
)

# Appliqué uniquement sur les verbatims nettoyés non vides
valid_docs = df_negatif['verbatim_clean'].dropna().replace('', np.nan).dropna()
if len(valid_docs) == 0:
    print("Aucun document valide pour la modélisation thématique. Vérifiez le prétraitement.")
else:
    X = vectorizer.fit_transform(valid_docs)
    feature_names = vectorizer.get_feature_names_out()

    # Nombre optimal de thèmes (à ajuster selon les résultats)
    n_topics = 5

    # Application de LDA (Latent Dirichlet Allocation)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=10,
        learning_method='online',
        random_state=42
    )
    lda.fit(X)

    # Fonction pour afficher les mots les plus représentatifs de chaque thème
    def display_topics(model, feature_names, n_top_words=10):
        topic_dict = {}
        for topic_idx, topic in enumerate(model.components_):
            top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_dict[f"Thème {topic_idx+1}"] = ", ".join(top_words)
        return topic_dict

    # Affichage des thèmes identifiés
    topics = display_topics(lda, feature_names)
    print("\nThèmes identifiés dans les verbatims négatifs :")
    for theme, words in topics.items():
        print(f"{theme}: {words}")

    # Prédiction des thèmes pour chaque document
    doc_topic_distrib = lda.transform(X)
    df_negatif_topics = df_negatif.loc[valid_docs.index].copy()
    df_negatif_topics['dominant_topic'] = doc_topic_distrib.argmax(axis=1) + 1  # +1 pour commencer à 1 au lieu de 0

    # Distribution des documents par thème
    topic_distribution = df_negatif_topics['dominant_topic'].value_counts().sort_index()
    print("\nDistribution des documents par thème :")
    print(topic_distribution)

    # Visualisation de la distribution des documents par thème
    plt.figure(figsize=(10, 6))
    sns.countplot(x='dominant_topic', data=df_negatif_topics, order=sorted(df_negatif_topics['dominant_topic'].unique()))
    plt.title('Distribution des verbatims négatifs par thème')
    plt.xlabel('Thème dominant')
    plt.ylabel('Nombre de verbatims')
    plt.tight_layout()
    plt.show()

    # Visualisation interactive des thèmes avec pyLDAvis (si dans un notebook)
    try:
        vis_data = pyLDAvis.sklearn.prepare(lda, X, vectorizer)
        pyLDAvis.display(vis_data)
        print("\nVisualisation interactive des thèmes générée avec succès.")
    except Exception as e:
        print(f"\nErreur lors de la génération de la visualisation interactive : {str(e)}")
```

## 6. Analyse de cooccurrences et clustering des verbatims

```python
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
```

## 7. Création d'un tableau de bord pour l'analyse des problématiques récurrentes

```python
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
```

## 8. Fonction pour rechercher des verbatims par mots-clés

```python
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
```

## 9. Création d'une interface utilisateur simple pour l'exploration

```python
# Création d'une interface simple pour explorer les résultats
def explorer_problematiques():
    """
    Fonction interactive pour explorer les problématiques. À exécuter dans un notebook Jupyter.
    """
    from ipywidgets import interact, widgets
    
    # Widget pour sélectionner les humeurs
    humeur_widget = widgets.SelectMultiple(
        options=df['Humeur'].unique(),
        value=['Plutôt Insatisfaite', 'Très insatisfaite'],
        description='Humeurs:',
        disabled=False
    )
    
    # Widget pour entrer des mots-clés
    keywords_widget = widgets.Text(
        value='',
        placeholder='Entrez des mots-clés séparés par des virgules',
        description='Mots-clés:',
        disabled=False
    )
    
    # Widget pour sélectionner les périmètres
    perimetre_widget = widgets.SelectMultiple(
        options=df['Perimetre'].unique(),
        value=[],
        description='Périmètres:',
        disabled=False
    )
    
    # Fonction pour afficher les résultats
    def afficher_resultats(humeurs, keywords, perimetres):
        # Filtre par humeurs
        if humeurs:
            df_filtre = df[df['Humeur'].isin(humeurs)]
        else:
            df_filtre = df.copy()
        
        # Filtre par périmètres
        if perimetres:
            df_filtre = df_filtre[df_filtre['Perimetre'].isin(perimetres)]
        
        # Filtre par mots-clés
        if keywords:
            keywords_list = [k.strip() for k in keywords.split(',')]
            if keywords_list:
                pattern = '|'.join(keywords_list)
                mask = df_filtre['Verbatim'].str.contains(pattern, case=False, na=False, regex=True)
                df_filtre = df_filtre[mask]
        
        # Affichage des résultats
        print(f"Nombre de verbatims correspondants: {len(df_filtre)}")
        
        if len(df_filtre) > 0:
            print("\nRésultats:")
            for idx, row in df_filtre.head(10).iterrows():
                print(f"Ligne {row['ligne_source']} - {row['Humeur']}: {row['Verbatim']}")
                print(f"  Périmètre: {row['Perimetre']} - {row['Sous perimetre']}")
                print(f"  Collaborateur: {row['Collaborateur qui a emis']}")
                print()
            
            if len(df_filtre) > 10:
                print(f"... et {len(df_filtre) - 10} autres résultats.")
        
        # Distribution des périmètres dans les résultats
        if len(df_filtre) > 0:
            perimetre_counts = df_filtre['Perimetre'].value_counts()
            plt.figure(figsize=(10, 6))
            sns.barplot(x=perimetre_counts.index, y=perimetre_counts.values)
            plt.title('Distribution des périmètres dans les résultats')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    
    # Création de l'interface interactive
    interact(afficher_resultats, 
             humeurs=humeur_widget, 
             keywords=keywords_widget, 
             perimetres=perimetre_widget)

# Exécution de l'interface (à utiliser dans un notebook Jupyter)
print("\nPour explorer les problématiques interactivement, exécutez la fonction 'explorer_problematiques()' dans une cellule séparée.")
```

## 10. Fonction principale pour exécuter l'ensemble de l'analyse

```python
def analyser_verbatims(fichier_excel, humeurs_a_analyser=None):
    """
    Fonction principale pour analyser les verbatims d'enquêtes de satisfaction.
    
    Args:
        fichier_excel: Chemin vers le fichier Excel
        humeurs_a_analyser: Liste des humeurs à analyser (None pour toutes)
    """
    # Chargement des données
    print(f"Chargement du fichier {fichier_excel}...")
    df = pd.read_excel(fichier_excel)
    
    # Ajout de l'index pour la traçabilité
    df['ligne_source'] = df.index
    
    # Filtrage par humeur si spécifié
    if humeurs_a_analyser:
        df_filtre = df[df['Humeur'].isin(humeurs_a_analyser)]
        print(f"Analyse limitée aux humeurs: {humeurs_a_analyser}")
    else:
        df_filtre = df.copy()
        print("Analyse de toutes les humeurs")
    
    print(f"Nombre de verbatims à analyser: {len(df_filtre)}")
    
    # Prétraitement des textes
    print("Prétraitement des verbatims...")
    df_filtre['verbatim_clean'] = df_filtre['Verbatim'].apply(preprocess_text)
    
    # Extraction de n-grammes
    print("Extraction des n-grammes fréquents...")
    top_bigrams = extract_ngrams(df_filtre['verbatim_clean'].dropna(), (2, 2))
    
    print("Top 10 bigrammes:")
    for bigram, count in list(top_bigrams.items())[:10]:
        print(f"- {bigram}: {count} occurrences")
    
    # Modélisation thématique
    print("\nModélisation thématique...")
    valid_docs = df_filtre['verbatim_clean'].dropna().replace('', np.nan).dropna()
    
    if len(valid_docs) >= 5:  # Vérification du nombre minimum de documents
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
        X = vectorizer.fit_transform(valid_docs)
        feature_names = vectorizer.get_feature_names_out()
        
        # Détermination du nombre optimal de thèmes (simplifié ici)
        n_topics = min(5, len(valid_docs) // 2)  # Heuristique simple
        
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=10,
            learning_method='online',
            random_state=42
        )
        lda.fit(X)
        
        # Affichage des thèmes
        topics = display_topics(lda, feature_names)
        print("\nThèmes identifiés dans les verbatims:")
        for theme, words in topics.items():
            print(f"{theme}: {words}")
        
        # Attribution des thèmes aux documents
        doc_topic_distrib = lda.transform(X)
        df_topics = df_filtre.loc[valid_docs.index].copy()
        df_topics['dominant_topic'] = doc_topic_distrib.argmax(axis=1) + 1
        
        # Création du tableau de bord des problématiques
        print("\nCréation du tableau de bord des problématiques récurrentes...")
        theme_summary = pd.DataFrame()
        
        for theme_id in sorted(df_topics['dominant_topic'].unique()):
            theme_docs = df_topics[df_topics['dominant_topic'] == theme_id]
            
            theme_key_words = topics[f"Thème {theme_id}"]
            verbatim_examples = theme_docs['Verbatim'].sample(min(3, len(theme_docs))).tolist()
            perimetre_counts = theme_docs['Perimetre'].value_counts().to_dict()
            top_perimetre = max(perimetre_counts.items(), key=lambda x: x[1])[0] if perimetre_counts else "N/A"
            lignes_sources = theme_docs['ligne_source'].tolist()
            
            theme_entry = {
                'Thème ID': theme_id,
                'Mots-clés': theme_key_words,
                'Nombre de verbatims': len(theme_docs),
                'Périmètre principal': top_perimetre,
                'Exemples de verbatims': verbatim_examples,
                'Numéros de lignes sources': lignes_sources
            }
            
            theme_summary = pd.concat([theme_summary, pd.DataFrame([theme_entry])], ignore_index=True)
        
        # Sauvegarde du tableau de bord
        output_file = 'dashboard_problematiques_recurrentes.xlsx'
        theme_summary.to_excel(output_file, index=False)
        print(f"Tableau de bord sauvegardé dans '{output_file}'")
        
        # Retour du résumé pour utilisation ultérieure
        return theme_summary
    else:
        print("Nombre insuffisant de documents valides pour la modélisation thématique.")
        return None

# Exemple d'utilisation
# analyser_verbatims('enquetes_satisfaction.xlsx', ['Plutôt Insatisfaite', 'Très insatisfaite'])
```

Ce code complet vous permet d'analyser les verbatims des enquêtes de satisfaction en utilisant des techniques avancées de NLP. Vous pouvez l'exécuter dans un notebook Jupyter et ainsi :

1. Générer un fichier Excel synthétique pour tester
2. Explorer les données et filtrer les avis négatifs
3. Prétraiter les textes pour l'analyse
4. Extraire les n-grammes fréquents et identifier les mots-clés récurrents
5. Utiliser la modélisation thématique pour découvrir automatiquement les problématiques
6. Créer un tableau de bord des problématiques récurrentes
7. Rechercher des verbatims spécifiques avec des mots-clés
8. Explorer interactivement les résultats

La fonction principale `analyser_verbatims()` vous permet d'exécuter l'ensemble du processus d'analyse. Vous pouvez facilement adapter et étendre ce code selon vos besoins spécifiques.
Je vais créer un code complet pour une analyse de données textuelles dans un cadre d'enquêtes de satisfaction. La demande comprend plusieurs éléments :

1. Générer un fichier Excel synthétique (données fictives) qui suivra la structure décrite
2. Charger ce fichier Excel 
3. Analyser les verbatims pour trouver des thématiques récurrentes, en se concentrant sur les avis négatifs
4. Permettre la traçabilité pour remonter aux sources des problèmes

Je vais structurer le code en plusieurs sections :

1. Génération des données fictives avec pandas et sauvegarde en format .xlsx
2. Chargement des données
3. Prétraitement des textes (nettoyage, lemmatisation)
4. Analyse exploratoire des données
5. Modélisation thématique (Topic Modeling) avec des méthodes comme LDA ou NMF
6. Extraction de n-grammes fréquents
7. Visualisation des résultats
8. Traçabilité (lien entre thèmes et lignes sources)

Je vais créer un notebook Jupyter complet et bien documenté, avec un style formel et professionnel adapté à un contexte d'entreprise.

# Analyse de verbatims d'enquêtes de satisfaction

Je vais vous fournir un code complet pour réaliser cette analyse textuelle des verbatims, avec une attention particulière sur les retours négatifs. Le code sera organisé en plusieurs sections pour plus de clarté.

## 1. Génération d'un jeu de données synthétique

Commençons par générer un fichier Excel synthétique qui servira pour notre démonstration :

```python
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Définition des valeurs possibles pour chaque colonne
humeurs = ["Très satisfaite", "Plutôt Satisfaite", "Plutôt Insatisfaite", "Très insatisfaite"]
categorisations = ["A dispatcher", "A traiter", "En cours", "Terminé"]
noms = ["Dupont Xavier", "Martin Sophie", "Leblanc Julie", "Durand Thomas", "Petit Claire", 
        "Moreau Nicolas", "Richard Emilie", "Robert Philippe", "Simon Catherine", "Leroy Antoine"]
perimetres = ["Epargne et assurances", "Data", "Crédits", "Banque au quotidien", "Moyens de paiement"]
sous_perimetres = ["Tribu Patrimoine", "Tribu digital workplace", "Tribu innovation", "Tribu crédit immobilier", "Tribu e-banking"]
chapitres = ["Tribu data hub", "Tribu dématérialisation", "Tribu expérience client", "Tribu conformité", "Tribu IT"]
metiers = ["Relai fabrication", "Scrum master", "Product owner", "Développeur", "Analyste fonctionnel", "Chef de projet", "Architecte", "Designer UX"]
types_enquete = ["IRC Flash Vague 1", "IRC Flash Vague 2", "IRC Flash Vague 3", "Enquête trimestrielle"]
equipes = ["IBC", "QOS", "DEV", "OPS", "ARCHI", "SECU"]
responsables = ["Martin Dupond", "Garcia Laura", "Vasseur Jean", "Lefebvre Marie", "Dubois Patrick"]
soutiens = ["Hughes Timhas", "Rousseau Marie", "Bernard David", "Fournier Sylvie", "Lambert François"]
statuts = ["Terminé", "En cours", "A planifier", "Suspendu"]

# Verbatims positifs et négatifs pour la génération
verbatims_positifs = [
    "Réactif et compétent, toujours à l'écoute.",
    "Excellente collaboration, équipe très efficace.",
    "Communication claire et adaptée à nos besoins.",
    "Grande disponibilité et solutions toujours pertinentes.",
    "Respect des délais et qualité du travail remarquable.",
    "Expertise technique impressionnante et bons conseils.",
    "Très bon accompagnement sur notre projet complexe.",
    "Capacité d'adaptation exceptionnelle face aux changements.",
    "Proactivité et anticipation des problématiques.",
    "Support technique efficace et rapide."
]

verbatims_negatifs = [
    "Très mauvaise communication récurrente de la part des Socles CAGIP sur les sujets transverses.",
    "Délais non respectés à plusieurs reprises, impact fort sur nos activités.",
    "Manque de compétence technique évident sur le projet.",
    "Difficultés à obtenir des réponses claires à nos questions.",
    "Documentation insuffisante et non mise à jour.",
    "Problèmes récurrents de qualité dans les livrables.",
    "Absence de suivi après la mise en production.",
    "Manque d'implication et de proactivité de l'équipe.",
    "Changements fréquents dans l'équipe perturbant la continuité.",
    "Outils mis à disposition obsolètes et inefficaces.",
    "Communication inexistante lors des incidents majeurs.",
    "Procédures trop complexes pour des demandes simples.",
    "Temps de réponse beaucoup trop long pour les tickets critiques.",
    "Absence totale de prise en compte de nos contraintes métier.",
    "Support technique défaillant et peu disponible."
]

# Génération de plans d'action
plans_action = [
    "Échange avec le manager de Tribu DWP pour résolution rapide.",
    "Organisation d'un atelier d'amélioration continue.",
    "Mise en place d'un nouveau processus de communication.",
    "Formation spécifique prévue pour l'équipe concernée.",
    "Revue des procédures internes et simplification.",
    "Création d'un groupe de travail dédié à cette problématique.",
    "Renforcement de l'équipe avec des profils plus expérimentés.",
    "Audit des outils et mise à jour prévue.",
    "Réorganisation du support pour améliorer les temps de réponse.",
    "Plan d'amélioration de la qualité en cours d'élaboration."
]

# Synthèses
syntheses = [
    "Florian signale ne pas avoir trouvé de solution malgré plusieurs relances.",
    "Problème identifié et résolu par mise à jour du framework.",
    "Incident majeur ayant nécessité une intervention de la DSI.",
    "Demande traitée avec succès après escalade au N+2.",
    "Problématique récurrente nécessitant une refonte complète.",
    "Solution temporaire mise en place en attendant le nouveau système.",
    "Formation effectuée, amélioration constatée immédiatement.",
    "Analyse approfondie révélant un problème d'infrastructure.",
    "Communication renforcée entre équipes suite à cet incident.",
    "Résolution nécessitant l'intervention de plusieurs services."
]

# Générer des dates sur les 6 derniers mois
date_fin = datetime.now()
date_debut = date_fin - timedelta(days=180)
dates_possibles = [date_debut + timedelta(days=x) for x in range(0, 180, 15)]  # Une enquête tous les 15 jours

# Fonction pour générer un verbatim en fonction de l'humeur
def generer_verbatim(humeur):
    if humeur in ["Très satisfaite", "Plutôt Satisfaite"]:
        return random.choice(verbatims_positifs)
    else:
        return random.choice(verbatims_negatifs)

# Fonction pour générer le plan d'action et la synthèse selon l'humeur et la catégorisation
def generer_plan_et_synthese(humeur, categorisation):
    if humeur in ["Plutôt Insatisfaite", "Très insatisfaite"]:
        plan = random.choice(plans_action) if categorisation != "A dispatcher" else ""
        synthese = random.choice(syntheses) if categorisation in ["En cours", "Terminé"] else ""
    else:
        plan = ""
        synthese = "Retour positif noté" if categorisation in ["En cours", "Terminé"] else ""
    
    return plan, synthese

# Créer un DataFrame avec 200 entrées
n_entries = 200
data = []

for i in range(n_entries):
    humeur = random.choices(humeurs, weights=[0.4, 0.3, 0.2, 0.1], k=1)[0]  # Distribution pondérée
    categorisation = random.choice(categorisations)
    verbatim = generer_verbatim(humeur)
    collaborateur = random.choice(noms)
    perimetre = random.choice(perimetres)
    sous_perimetre = random.choice(sous_perimetres)
    chapitre = random.choice(chapitres)
    metier = random.choice(metiers)
    date_enquete = random.choice(dates_possibles).strftime('%Y-%m-%d')
    type_enquete = random.choice(types_enquete)
    equipe = random.choice(equipes)
    responsable = random.choice(responsables)
    soutien = random.choice(soutiens)
    statut = random.choice(statuts)
    plan_action, synthese = generer_plan_et_synthese(humeur, categorisation)
    
    data.append({
        'Humeur': humeur,
        'Categorisation': categorisation,
        'Verbatim': verbatim,
        'Collaborateur qui a emis': collaborateur,
        'Perimetre': perimetre,
        'Sous perimetre': sous_perimetre,
        'Chapitre de la sous tribu': chapitre,
        'Metier CATS': metier,
        'Date de l\'enquete': date_enquete,
        'Type d\'Enquete': type_enquete,
        'Equipe BFR concernee': equipe,
        'Responsable BFR': responsable,
        'Soutien de BFR': soutien,
        'Statut': statut,
        'Plan d\'action': plan_action,
        'Synthese et/ou plan d\'actions': synthese
    })

# Création du DataFrame
df = pd.DataFrame(data)

# Sauvegarde dans un fichier Excel
output_file = 'enquetes_satisfaction.xlsx'
df.to_excel(output_file, index=True)  # Garder l'index pour avoir le numéro de ligne

print(f"Fichier '{output_file}' généré avec succès contenant {n_entries} entrées.")
```

## 2. Chargement et exploration des données

Maintenant, chargeons le fichier généré et explorons les données :

```python
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
import pyLDAvis
import pyLDAvis.sklearn
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
```

## 3. Prétraitement des textes pour l'analyse

```python
# Fonction de prétraitement des textes
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Conversion en minuscules
    text = text.lower()
    
    # Suppression des caractères spéciaux et des chiffres
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Tokenisation
    tokens = word_tokenize(text)
    
    # Suppression des stopwords (mots vides)
    stop_words = set(stopwords.words('french'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# Application du prétraitement aux verbatims négatifs
df_negatif['verbatim_clean'] = df_negatif['Verbatim'].apply(preprocess_text)

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
sns.barplot(x=list(top_words.values()), y=list(top_words.keys()))
plt.title('Mots les plus fréquents dans les verbatims négatifs')
plt.tight_layout()
plt.show()

# Création d'un nuage de mots
wordcloud = WordCloud(width=800, height=400, background_color='white', 
                     max_words=100, contour_width=1, contour_color='steelblue')
wordcloud.generate(' '.join(df_negatif['verbatim_clean']))

plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nuage de mots des verbatims négatifs')
plt.show()
```

## 4. Extraction de n-grammes fréquents

```python
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
```

## 5. Modélisation thématique (Topic Modeling)

```python
# Préparation des données pour la modélisation thématique
vectorizer = TfidfVectorizer(
    max_df=0.95,         # Ignorer les termes qui apparaissent dans plus de 95% des documents
    min_df=2,            # Ignorer les termes qui apparaissent dans moins de 2 documents
    max_features=1000,   # Limiter le nombre de features
    stop_words='english' # Déjà fait dans le prétraitement, mais ajout de sécurité
)

# Appliqué uniquement sur les verbatims nettoyés non vides
valid_docs = df_negatif['verbatim_clean'].dropna().replace('', np.nan).dropna()
if len(valid_docs) == 0:
    print("Aucun document valide pour la modélisation thématique. Vérifiez le prétraitement.")
else:
    X = vectorizer.fit_transform(valid_docs)
    feature_names = vectorizer.get_feature_names_out()

    # Nombre optimal de thèmes (à ajuster selon les résultats)
    n_topics = 5

    # Application de LDA (Latent Dirichlet Allocation)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=10,
        learning_method='online',
        random_state=42
    )
    lda.fit(X)

    # Fonction pour afficher les mots les plus représentatifs de chaque thème
    def display_topics(model, feature_names, n_top_words=10):
        topic_dict = {}
        for topic_idx, topic in enumerate(model.components_):
            top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_dict[f"Thème {topic_idx+1}"] = ", ".join(top_words)
        return topic_dict

    # Affichage des thèmes identifiés
    topics = display_topics(lda, feature_names)
    print("\nThèmes identifiés dans les verbatims négatifs :")
    for theme, words in topics.items():
        print(f"{theme}: {words}")

    # Prédiction des thèmes pour chaque document
    doc_topic_distrib = lda.transform(X)
    df_negatif_topics = df_negatif.loc[valid_docs.index].copy()
    df_negatif_topics['dominant_topic'] = doc_topic_distrib.argmax(axis=1) + 1  # +1 pour commencer à 1 au lieu de 0

    # Distribution des documents par thème
    topic_distribution = df_negatif_topics['dominant_topic'].value_counts().sort_index()
    print("\nDistribution des documents par thème :")
    print(topic_distribution)

    # Visualisation de la distribution des documents par thème
    plt.figure(figsize=(10, 6))
    sns.countplot(x='dominant_topic', data=df_negatif_topics, order=sorted(df_negatif_topics['dominant_topic'].unique()))
    plt.title('Distribution des verbatims négatifs par thème')
    plt.xlabel('Thème dominant')
    plt.ylabel('Nombre de verbatims')
    plt.tight_layout()
    plt.show()

    # Visualisation interactive des thèmes avec pyLDAvis (si dans un notebook)
    try:
        vis_data = pyLDAvis.sklearn.prepare(lda, X, vectorizer)
        pyLDAvis.display(vis_data)
        print("\nVisualisation interactive des thèmes générée avec succès.")
    except Exception as e:
        print(f"\nErreur lors de la génération de la visualisation interactive : {str(e)}")
```

## 6. Analyse de cooccurrences et clustering des verbatims

```python
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
```

## 7. Création d'un tableau de bord pour l'analyse des problématiques récurrentes

```python
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
```

## 8. Fonction pour rechercher des verbatims par mots-clés

```python
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
```

## 9. Création d'une interface utilisateur simple pour l'exploration

```python
# Création d'une interface simple pour explorer les résultats
def explorer_problematiques():
    """
    Fonction interactive pour explorer les problématiques. À exécuter dans un notebook Jupyter.
    """
    from ipywidgets import interact, widgets
    
    # Widget pour sélectionner les humeurs
    humeur_widget = widgets.SelectMultiple(
        options=df['Humeur'].unique(),
        value=['Plutôt Insatisfaite', 'Très insatisfaite'],
        description='Humeurs:',
        disabled=False
    )
    
    # Widget pour entrer des mots-clés
    keywords_widget = widgets.Text(
        value='',
        placeholder='Entrez des mots-clés séparés par des virgules',
        description='Mots-clés:',
        disabled=False
    )
    
    # Widget pour sélectionner les périmètres
    perimetre_widget = widgets.SelectMultiple(
        options=df['Perimetre'].unique(),
        value=[],
        description='Périmètres:',
        disabled=False
    )
    
    # Fonction pour afficher les résultats
    def afficher_resultats(humeurs, keywords, perimetres):
        # Filtre par humeurs
        if humeurs:
            df_filtre = df[df['Humeur'].isin(humeurs)]
        else:
            df_filtre = df.copy()
        
        # Filtre par périmètres
        if perimetres:
            df_filtre = df_filtre[df_filtre['Perimetre'].isin(perimetres)]
        
        # Filtre par mots-clés
        if keywords:
            keywords_list = [k.strip() for k in keywords.split(',')]
            if keywords_list:
                pattern = '|'.join(keywords_list)
                mask = df_filtre['Verbatim'].str.contains(pattern, case=False, na=False, regex=True)
                df_filtre = df_filtre[mask]
        
        # Affichage des résultats
        print(f"Nombre de verbatims correspondants: {len(df_filtre)}")
        
        if len(df_filtre) > 0:
            print("\nRésultats:")
            for idx, row in df_filtre.head(10).iterrows():
                print(f"Ligne {row['ligne_source']} - {row['Humeur']}: {row['Verbatim']}")
                print(f"  Périmètre: {row['Perimetre']} - {row['Sous perimetre']}")
                print(f"  Collaborateur: {row['Collaborateur qui a emis']}")
                print()
            
            if len(df_filtre) > 10:
                print(f"... et {len(df_filtre) - 10} autres résultats.")
        
        # Distribution des périmètres dans les résultats
        if len(df_filtre) > 0:
            perimetre_counts = df_filtre['Perimetre'].value_counts()
            plt.figure(figsize=(10, 6))
            sns.barplot(x=perimetre_counts.index, y=perimetre_counts.values)
            plt.title('Distribution des périmètres dans les résultats')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    
    # Création de l'interface interactive
    interact(afficher_resultats, 
             humeurs=humeur_widget, 
             keywords=keywords_widget, 
             perimetres=perimetre_widget)

# Exécution de l'interface (à utiliser dans un notebook Jupyter)
print("\nPour explorer les problématiques interactivement, exécutez la fonction 'explorer_problematiques()' dans une cellule séparée.")
```

## 10. Fonction principale pour exécuter l'ensemble de l'analyse

```python
def analyser_verbatims(fichier_excel, humeurs_a_analyser=None):
    """
    Fonction principale pour analyser les verbatims d'enquêtes de satisfaction.
    
    Args:
        fichier_excel: Chemin vers le fichier Excel
        humeurs_a_analyser: Liste des humeurs à analyser (None pour toutes)
    """
    # Chargement des données
    print(f"Chargement du fichier {fichier_excel}...")
    df = pd.read_excel(fichier_excel)
    
    # Ajout de l'index pour la traçabilité
    df['ligne_source'] = df.index
    
    # Filtrage par humeur si spécifié
    if humeurs_a_analyser:
        df_filtre = df[df['Humeur'].isin(humeurs_a_analyser)]
        print(f"Analyse limitée aux humeurs: {humeurs_a_analyser}")
    else:
        df_filtre = df.copy()
        print("Analyse de toutes les humeurs")
    
    print(f"Nombre de verbatims à analyser: {len(df_filtre)}")
    
    # Prétraitement des textes
    print("Prétraitement des verbatims...")
    df_filtre['verbatim_clean'] = df_filtre['Verbatim'].apply(preprocess_text)
    
    # Extraction de n-grammes
    print("Extraction des n-grammes fréquents...")
    top_bigrams = extract_ngrams(df_filtre['verbatim_clean'].dropna(), (2, 2))
    
    print("Top 10 bigrammes:")
    for bigram, count in list(top_bigrams.items())[:10]:
        print(f"- {bigram}: {count} occurrences")
    
    # Modélisation thématique
    print("\nModélisation thématique...")
    valid_docs = df_filtre['verbatim_clean'].dropna().replace('', np.nan).dropna()
    
    if len(valid_docs) >= 5:  # Vérification du nombre minimum de documents
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
        X = vectorizer.fit_transform(valid_docs)
        feature_names = vectorizer.get_feature_names_out()
        
        # Détermination du nombre optimal de thèmes (simplifié ici)
        n_topics = min(5, len(valid_docs) // 2)  # Heuristique simple
        
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=10,
            learning_method='online',
            random_state=42
        )
        lda.fit(X)
        
        # Affichage des thèmes
        topics = display_topics(lda, feature_names)
        print("\nThèmes identifiés dans les verbatims:")
        for theme, words in topics.items():
            print(f"{theme}: {words}")
        
        # Attribution des thèmes aux documents
        doc_topic_distrib = lda.transform(X)
        df_topics = df_filtre.loc[valid_docs.index].copy()
        df_topics['dominant_topic'] = doc_topic_distrib.argmax(axis=1) + 1
        
        # Création du tableau de bord des problématiques
        print("\nCréation du tableau de bord des problématiques récurrentes...")
        theme_summary = pd.DataFrame()
        
        for theme_id in sorted(df_topics['dominant_topic'].unique()):
            theme_docs = df_topics[df_topics['dominant_topic'] == theme_id]
            
            theme_key_words = topics[f"Thème {theme_id}"]
            verbatim_examples = theme_docs['Verbatim'].sample(min(3, len(theme_docs))).tolist()
            perimetre_counts = theme_docs['Perimetre'].value_counts().to_dict()
            top_perimetre = max(perimetre_counts.items(), key=lambda x: x[1])[0] if perimetre_counts else "N/A"
            lignes_sources = theme_docs['ligne_source'].tolist()
            
            theme_entry = {
                'Thème ID': theme_id,
                'Mots-clés': theme_key_words,
                'Nombre de verbatims': len(theme_docs),
                'Périmètre principal': top_perimetre,
                'Exemples de verbatims': verbatim_examples,
                'Numéros de lignes sources': lignes_sources
            }
            
            theme_summary = pd.concat([theme_summary, pd.DataFrame([theme_entry])], ignore_index=True)
        
        # Sauvegarde du tableau de bord
        output_file = 'dashboard_problematiques_recurrentes.xlsx'
        theme_summary.to_excel(output_file, index=False)
        print(f"Tableau de bord sauvegardé dans '{output_file}'")
        
        # Retour du résumé pour utilisation ultérieure
        return theme_summary
    else:
        print("Nombre insuffisant de documents valides pour la modélisation thématique.")
        return None

# Exemple d'utilisation
# analyser_verbatims('enquetes_satisfaction.xlsx', ['Plutôt Insatisfaite', 'Très insatisfaite'])
```

Ce code complet vous permet d'analyser les verbatims des enquêtes de satisfaction en utilisant des techniques avancées de NLP. Vous pouvez l'exécuter dans un notebook Jupyter et ainsi :

1. Générer un fichier Excel synthétique pour tester
2. Explorer les données et filtrer les avis négatifs
3. Prétraiter les textes pour l'analyse
4. Extraire les n-grammes fréquents et identifier les mots-clés récurrents
5. Utiliser la modélisation thématique pour découvrir automatiquement les problématiques
6. Créer un tableau de bord des problématiques récurrentes
7. Rechercher des verbatims spécifiques avec des mots-clés
8. Explorer interactivement les résultats

La fonction principale `analyser_verbatims()` vous permet d'exécuter l'ensemble du processus d'analyse. Vous pouvez facilement adapter et étendre ce code selon vos besoins spécifiques
Je vais créer un code complet pour une analyse de données textuelles dans un cadre d'enquêtes de satisfaction. La demande comprend plusieurs éléments :

1. Générer un fichier Excel synthétique (données fictives) qui suivra la structure décrite
2. Charger ce fichier Excel 
3. Analyser les verbatims pour trouver des thématiques récurrentes, en se concentrant sur les avis négatifs
4. Permettre la traçabilité pour remonter aux sources des problèmes

Je vais structurer le code en plusieurs sections :

1. Génération des données fictives avec pandas et sauvegarde en format .xlsx
2. Chargement des données
3. Prétraitement des textes (nettoyage, lemmatisation)
4. Analyse exploratoire des données
5. Modélisation thématique (Topic Modeling) avec des méthodes comme LDA ou NMF
6. Extraction de n-grammes fréquents
7. Visualisation des résultats
8. Traçabilité (lien entre thèmes et lignes sources)

Je vais créer un notebook Jupyter complet et bien documenté, avec un style formel et professionnel adapté à un contexte d'entreprise.

# Analyse de verbatims d'enquêtes de satisfaction

Je vais vous fournir un code complet pour réaliser cette analyse textuelle des verbatims, avec une attention particulière sur les retours négatifs. Le code sera organisé en plusieurs sections pour plus de clarté.

## 1. Génération d'un jeu de données synthétique

Commençons par générer un fichier Excel synthétique qui servira pour notre démonstration :

```python
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Définition des valeurs possibles pour chaque colonne
humeurs = ["Très satisfaite", "Plutôt Satisfaite", "Plutôt Insatisfaite", "Très insatisfaite"]
categorisations = ["A dispatcher", "A traiter", "En cours", "Terminé"]
noms = ["Dupont Xavier", "Martin Sophie", "Leblanc Julie", "Durand Thomas", "Petit Claire", 
        "Moreau Nicolas", "Richard Emilie", "Robert Philippe", "Simon Catherine", "Leroy Antoine"]
perimetres = ["Epargne et assurances", "Data", "Crédits", "Banque au quotidien", "Moyens de paiement"]
sous_perimetres = ["Tribu Patrimoine", "Tribu digital workplace", "Tribu innovation", "Tribu crédit immobilier", "Tribu e-banking"]
chapitres = ["Tribu data hub", "Tribu dématérialisation", "Tribu expérience client", "Tribu conformité", "Tribu IT"]
metiers = ["Relai fabrication", "Scrum master", "Product owner", "Développeur", "Analyste fonctionnel", "Chef de projet", "Architecte", "Designer UX"]
types_enquete = ["IRC Flash Vague 1", "IRC Flash Vague 2", "IRC Flash Vague 3", "Enquête trimestrielle"]
equipes = ["IBC", "QOS", "DEV", "OPS", "ARCHI", "SECU"]
responsables = ["Martin Dupond", "Garcia Laura", "Vasseur Jean", "Lefebvre Marie", "Dubois Patrick"]
soutiens = ["Hughes Timhas", "Rousseau Marie", "Bernard David", "Fournier Sylvie", "Lambert François"]
statuts = ["Terminé", "En cours", "A planifier", "Suspendu"]

# Verbatims positifs et négatifs pour la génération
verbatims_positifs = [
    "Réactif et compétent, toujours à l'écoute.",
    "Excellente collaboration, équipe très efficace.",
    "Communication claire et adaptée à nos besoins.",
    "Grande disponibilité et solutions toujours pertinentes.",
    "Respect des délais et qualité du travail remarquable.",
    "Expertise technique impressionnante et bons conseils.",
    "Très bon accompagnement sur notre projet complexe.",
    "Capacité d'adaptation exceptionnelle face aux changements.",
    "Proactivité et anticipation des problématiques.",
    "Support technique efficace et rapide."
]

verbatims_negatifs = [
    "Très mauvaise communication récurrente de la part des Socles CAGIP sur les sujets transverses.",
    "Délais non respectés à plusieurs reprises, impact fort sur nos activités.",
    "Manque de compétence technique évident sur le projet.",
    "Difficultés à obtenir des réponses claires à nos questions.",
    "Documentation insuffisante et non mise à jour.",
    "Problèmes récurrents de qualité dans les livrables.",
    "Absence de suivi après la mise en production.",
    "Manque d'implication et de proactivité de l'équipe.",
    "Changements fréquents dans l'équipe perturbant la continuité.",
    "Outils mis à disposition obsolètes et inefficaces.",
    "Communication inexistante lors des incidents majeurs.",
    "Procédures trop complexes pour des demandes simples.",
    "Temps de réponse beaucoup trop long pour les tickets critiques.",
    "Absence totale de prise en compte de nos contraintes métier.",
    "Support technique défaillant et peu disponible."
]

# Génération de plans d'action
plans_action = [
    "Échange avec le manager de Tribu DWP pour résolution rapide.",
    "Organisation d'un atelier d'amélioration continue.",
    "Mise en place d'un nouveau processus de communication.",
    "Formation spécifique prévue pour l'équipe concernée.",
    "Revue des procédures internes et simplification.",
    "Création d'un groupe de travail dédié à cette problématique.",
    "Renforcement de l'équipe avec des profils plus expérimentés.",
    "Audit des outils et mise à jour prévue.",
    "Réorganisation du support pour améliorer les temps de réponse.",
    "Plan d'amélioration de la qualité en cours d'élaboration."
]

# Synthèses
syntheses = [
    "Florian signale ne pas avoir trouvé de solution malgré plusieurs relances.",
    "Problème identifié et résolu par mise à jour du framework.",
    "Incident majeur ayant nécessité une intervention de la DSI.",
    "Demande traitée avec succès après escalade au N+2.",
    "Problématique récurrente nécessitant une refonte complète.",
    "Solution temporaire mise en place en attendant le nouveau système.",
    "Formation effectuée, amélioration constatée immédiatement.",
    "Analyse approfondie révélant un problème d'infrastructure.",
    "Communication renforcée entre équipes suite à cet incident.",
    "Résolution nécessitant l'intervention de plusieurs services."
]

# Générer des dates sur les 6 derniers mois
date_fin = datetime.now()
date_debut = date_fin - timedelta(days=180)
dates_possibles = [date_debut + timedelta(days=x) for x in range(0, 180, 15)]  # Une enquête tous les 15 jours

# Fonction pour générer un verbatim en fonction de l'humeur
def generer_verbatim(humeur):
    if humeur in ["Très satisfaite", "Plutôt Satisfaite"]:
        return random.choice(verbatims_positifs)
    else:
        return random.choice(verbatims_negatifs)

# Fonction pour générer le plan d'action et la synthèse selon l'humeur et la catégorisation
def generer_plan_et_synthese(humeur, categorisation):
    if humeur in ["Plutôt Insatisfaite", "Très insatisfaite"]:
        plan = random.choice(plans_action) if categorisation != "A dispatcher" else ""
        synthese = random.choice(syntheses) if categorisation in ["En cours", "Terminé"] else ""
    else:
        plan = ""
        synthese = "Retour positif noté" if categorisation in ["En cours", "Terminé"] else ""
    
    return plan, synthese

# Créer un DataFrame avec 200 entrées
n_entries = 200
data = []

for i in range(n_entries):
    humeur = random.choices(humeurs, weights=[0.4, 0.3, 0.2, 0.1], k=1)[0]  # Distribution pondérée
    categorisation = random.choice(categorisations)
    verbatim = generer_verbatim(humeur)
    collaborateur = random.choice(noms)
    perimetre = random.choice(perimetres)
    sous_perimetre = random.choice(sous_perimetres)
    chapitre = random.choice(chapitres)
    metier = random.choice(metiers)
    date_enquete = random.choice(dates_possibles).strftime('%Y-%m-%d')
    type_enquete = random.choice(types_enquete)
    equipe = random.choice(equipes)
    responsable = random.choice(responsables)
    soutien = random.choice(soutiens)
    statut = random.choice(statuts)
    plan_action, synthese = generer_plan_et_synthese(humeur, categorisation)
    
    data.append({
        'Humeur': humeur,
        'Categorisation': categorisation,
        'Verbatim': verbatim,
        'Collaborateur qui a emis': collaborateur,
        'Perimetre': perimetre,
        'Sous perimetre': sous_perimetre,
        'Chapitre de la sous tribu': chapitre,
        'Metier CATS': metier,
        'Date de l\'enquete': date_enquete,
        'Type d\'Enquete': type_enquete,
        'Equipe BFR concernee': equipe,
        'Responsable BFR': responsable,
        'Soutien de BFR': soutien,
        'Statut': statut,
        'Plan d\'action': plan_action,
        'Synthese et/ou plan d\'actions': synthese
    })

# Création du DataFrame
df = pd.DataFrame(data)

# Sauvegarde dans un fichier Excel
output_file = 'enquetes_satisfaction.xlsx'
df.to_excel(output_file, index=True)  # Garder l'index pour avoir le numéro de ligne

print(f"Fichier '{output_file}' généré avec succès contenant {n_entries} entrées.")
```

## 2. Chargement et exploration des données

Maintenant, chargeons le fichier généré et explorons les données :

```python
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
import pyLDAvis
import pyLDAvis.sklearn
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
```

## 3. Prétraitement des textes pour l'analyse

```python
# Fonction de prétraitement des textes
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Conversion en minuscules
    text = text.lower()
    
    # Suppression des caractères spéciaux et des chiffres
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Tokenisation
    tokens = word_tokenize(text)
    
    # Suppression des stopwords (mots vides)
    stop_words = set(stopwords.words('french'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# Application du prétraitement aux verbatims négatifs
df_negatif['verbatim_clean'] = df_negatif['Verbatim'].apply(preprocess_text)

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
sns.barplot(x=list(top_words.values()), y=list(top_words.keys()))
plt.title('Mots les plus fréquents dans les verbatims négatifs')
plt.tight_layout()
plt.show()

# Création d'un nuage de mots
wordcloud = WordCloud(width=800, height=400, background_color='white', 
                     max_words=100, contour_width=1, contour_color='steelblue')
wordcloud.generate(' '.join(df_negatif['verbatim_clean']))

plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nuage de mots des verbatims négatifs')
plt.show()
```

## 4. Extraction de n-grammes fréquents

```python
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
```

## 5. Modélisation thématique (Topic Modeling)

```python
# Préparation des données pour la modélisation thématique
vectorizer = TfidfVectorizer(
    max_df=0.95,         # Ignorer les termes qui apparaissent dans plus de 95% des documents
    min_df=2,            # Ignorer les termes qui apparaissent dans moins de 2 documents
    max_features=1000,   # Limiter le nombre de features
    stop_words='english' # Déjà fait dans le prétraitement, mais ajout de sécurité
)

# Appliqué uniquement sur les verbatims nettoyés non vides
valid_docs = df_negatif['verbatim_clean'].dropna().replace('', np.nan).dropna()
if len(valid_docs) == 0:
    print("Aucun document valide pour la modélisation thématique. Vérifiez le prétraitement.")
else:
    X = vectorizer.fit_transform(valid_docs)
    feature_names = vectorizer.get_feature_names_out()

    # Nombre optimal de thèmes (à ajuster selon les résultats)
    n_topics = 5

    # Application de LDA (Latent Dirichlet Allocation)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=10,
        learning_method='online',
        random_state=42
    )
    lda.fit(X)

    # Fonction pour afficher les mots les plus représentatifs de chaque thème
    def display_topics(model, feature_names, n_top_words=10):
        topic_dict = {}
        for topic_idx, topic in enumerate(model.components_):
            top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_dict[f"Thème {topic_idx+1}"] = ", ".join(top_words)
        return topic_dict

    # Affichage des thèmes identifiés
    topics = display_topics(lda, feature_names)
    print("\nThèmes identifiés dans les verbatims négatifs :")
    for theme, words in topics.items():
        print(f"{theme}: {words}")

    # Prédiction des thèmes pour chaque document
    doc_topic_distrib = lda.transform(X)
    df_negatif_topics = df_negatif.loc[valid_docs.index].copy()
    df_negatif_topics['dominant_topic'] = doc_topic_distrib.argmax(axis=1) + 1  # +1 pour commencer à 1 au lieu de 0

    # Distribution des documents par thème
    topic_distribution = df_negatif_topics['dominant_topic'].value_counts().sort_index()
    print("\nDistribution des documents par thème :")
    print(topic_distribution)

    # Visualisation de la distribution des documents par thème
    plt.figure(figsize=(10, 6))
    sns.countplot(x='dominant_topic', data=df_negatif_topics, order=sorted(df_negatif_topics['dominant_topic'].unique()))
    plt.title('Distribution des verbatims négatifs par thème')
    plt.xlabel('Thème dominant')
    plt.ylabel('Nombre de verbatims')
    plt.tight_layout()
    plt.show()

    # Visualisation interactive des thèmes avec pyLDAvis (si dans un notebook)
    try:
        vis_data = pyLDAvis.sklearn.prepare(lda, X, vectorizer)
        pyLDAvis.display(vis_data)
        print("\nVisualisation interactive des thèmes générée avec succès.")
    except Exception as e:
        print(f"\nErreur lors de la génération de la visualisation interactive : {str(e)}")
```

## 6. Analyse de cooccurrences et clustering des verbatims

```python
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
```

## 7. Création d'un tableau de bord pour l'analyse des problématiques récurrentes

```python
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
```

## 8. Fonction pour rechercher des verbatims par mots-clés

```python
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
```

## 9. Création d'une interface utilisateur simple pour l'exploration

```python
# Création d'une interface simple pour explorer les résultats
def explorer_problematiques():
    """
    Fonction interactive pour explorer les problématiques. À exécuter dans un notebook Jupyter.
    """
    from ipywidgets import interact, widgets
    
    # Widget pour sélectionner les humeurs
    humeur_widget = widgets.SelectMultiple(
        options=df['Humeur'].unique(),
        value=['Plutôt Insatisfaite', 'Très insatisfaite'],
        description='Humeurs:',
        disabled=False
    )
    
    # Widget pour entrer des mots-clés
    keywords_widget = widgets.Text(
        value='',
        placeholder='Entrez des mots-clés séparés par des virgules',
        description='Mots-clés:',
        disabled=False
    )
    
    # Widget pour sélectionner les périmètres
    perimetre_widget = widgets.SelectMultiple(
        options=df['Perimetre'].unique(),
        value=[],
        description='Périmètres:',
        disabled=False
    )
    
    # Fonction pour afficher les résultats
    def afficher_resultats(humeurs, keywords, perimetres):
        # Filtre par humeurs
        if humeurs:
            df_filtre = df[df['Humeur'].isin(humeurs)]
        else:
            df_filtre = df.copy()
        
        # Filtre par périmètres
        if perimetres:
            df_filtre = df_filtre[df_filtre['Perimetre'].isin(perimetres)]
        
        # Filtre par mots-clés
        if keywords:
            keywords_list = [k.strip() for k in keywords.split(',')]
            if keywords_list:
                pattern = '|'.join(keywords_list)
                mask = df_filtre['Verbatim'].str.contains(pattern, case=False, na=False, regex=True)
                df_filtre = df_filtre[mask]
        
        # Affichage des résultats
        print(f"Nombre de verbatims correspondants: {len(df_filtre)}")
        
        if len(df_filtre) > 0:
            print("\nRésultats:")
            for idx, row in df_filtre.head(10).iterrows():
                print(f"Ligne {row['ligne_source']} - {row['Humeur']}: {row['Verbatim']}")
                print(f"  Périmètre: {row['Perimetre']} - {row['Sous perimetre']}")
                print(f"  Collaborateur: {row['Collaborateur qui a emis']}")
                print()
            
            if len(df_filtre) > 10:
                print(f"... et {len(df_filtre) - 10} autres résultats.")
        
        # Distribution des périmètres dans les résultats
        if len(df_filtre) > 0:
            perimetre_counts = df_filtre['Perimetre'].value_counts()
            plt.figure(figsize=(10, 6))
            sns.barplot(x=perimetre_counts.index, y=perimetre_counts.values)
            plt.title('Distribution des périmètres dans les résultats')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    
    # Création de l'interface interactive
    interact(afficher_resultats, 
             humeurs=humeur_widget, 
             keywords=keywords_widget, 
             perimetres=perimetre_widget)

# Exécution de l'interface (à utiliser dans un notebook Jupyter)
print("\nPour explorer les problématiques interactivement, exécutez la fonction 'explorer_problematiques()' dans une cellule séparée.")
```

## 10. Fonction principale pour exécuter l'ensemble de l'analyse

```python
def analyser_verbatims(fichier_excel, humeurs_a_analyser=None):
    """
    Fonction principale pour analyser les verbatims d'enquêtes de satisfaction.
    
    Args:
        fichier_excel: Chemin vers le fichier Excel
        humeurs_a_analyser: Liste des humeurs à analyser (None pour toutes)
    """
    # Chargement des données
    print(f"Chargement du fichier {fichier_excel}...")
    df = pd.read_excel(fichier_excel)
    
    # Ajout de l'index pour la traçabilité
    df['ligne_source'] = df.index
    
    # Filtrage par humeur si spécifié
    if humeurs_a_analyser:
        df_filtre = df[df['Humeur'].isin(humeurs_a_analyser)]
        print(f"Analyse limitée aux humeurs: {humeurs_a_analyser}")
    else:
        df_filtre = df.copy()
        print("Analyse de toutes les humeurs")
    
    print(f"Nombre de verbatims à analyser: {len(df_filtre)}")
    
    # Prétraitement des textes
    print("Prétraitement des verbatims...")
    df_filtre['verbatim_clean'] = df_filtre['Verbatim'].apply(preprocess_text)
    
    # Extraction de n-grammes
    print("Extraction des n-grammes fréquents...")
    top_bigrams = extract_ngrams(df_filtre['verbatim_clean'].dropna(), (2, 2))
    
    print("Top 10 bigrammes:")
    for bigram, count in list(top_bigrams.items())[:10]:
        print(f"- {bigram}: {count} occurrences")
    
    # Modélisation thématique
    print("\nModélisation thématique...")
    valid_docs = df_filtre['verbatim_clean'].dropna().replace('', np.nan).dropna()
    
    if len(valid_docs) >= 5:  # Vérification du nombre minimum de documents
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
        X = vectorizer.fit_transform(valid_docs)
        feature_names = vectorizer.get_feature_names_out()
        
        # Détermination du nombre optimal de thèmes (simplifié ici)
        n_topics = min(5, len(valid_docs) // 2)  # Heuristique simple
        
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=10,
            learning_method='online',
            random_state=42
        )
        lda.fit(X)
        
        # Affichage des thèmes
        topics = display_topics(lda, feature_names)
        print("\nThèmes identifiés dans les verbatims:")
        for theme, words in topics.items():
            print(f"{theme}: {words}")
        
        # Attribution des thèmes aux documents
        doc_topic_distrib = lda.transform(X)
        df_topics = df_filtre.loc[valid_docs.index].copy()
        df_topics['dominant_topic'] = doc_topic_distrib.argmax(axis=1) + 1
        
        # Création du tableau de bord des problématiques
        print("\nCréation du tableau de bord des problématiques récurrentes...")
        theme_summary = pd.DataFrame()
        
        for theme_id in sorted(df_topics['dominant_topic'].unique()):
            theme_docs = df_topics[df_topics['dominant_topic'] == theme_id]
            
            theme_key_words = topics[f"Thème {theme_id}"]
            verbatim_examples = theme_docs['Verbatim'].sample(min(3, len(theme_docs))).tolist()
            perimetre_counts = theme_docs['Perimetre'].value_counts().to_dict()
            top_perimetre = max(perimetre_counts.items(), key=lambda x: x[1])[0] if perimetre_counts else "N/A"
            lignes_sources = theme_docs['ligne_source'].tolist()
            
            theme_entry = {
                'Thème ID': theme_id,
                'Mots-clés': theme_key_words,
                'Nombre de verbatims': len(theme_docs),
                'Périmètre principal': top_perimetre,
                'Exemples de verbatims': verbatim_examples,
                'Numéros de lignes sources': lignes_sources
            }
            
            theme_summary = pd.concat([theme_summary, pd.DataFrame([theme_entry])], ignore_index=True)
        
        # Sauvegarde du tableau de bord
        output_file = 'dashboard_problematiques_recurrentes.xlsx'
        theme_summary.to_excel(output_file, index=False)
        print(f"Tableau de bord sauvegardé dans '{output_file}'")
        
        # Retour du résumé pour utilisation ultérieure
        return theme_summary
    else:
        print("Nombre insuffisant de documents valides pour la modélisation thématique.")
        return None

# Exemple d'utilisation
# analyser_verbatims('enquetes_satisfaction.xlsx', ['Plutôt Insatisfaite', 'Très insatisfaite'])
```

Ce code complet vous permet d'analyser les verbatims des enquêtes de satisfaction en utilisant des techniques avancées de NLP. Vous pouvez l'exécuter dans un notebook Jupyter et ainsi :

1. Générer un fichier Excel synthétique pour tester
2. Explorer les données et filtrer les avis négatifs
3. Prétraiter les textes pour l'analyse
4. Extraire les n-grammes fréquents et identifier les mots-clés récurrents
5. Utiliser la modélisation thématique pour découvrir automatiquement les problématiques
6. Créer un tableau de bord des problématiques récurrentes
7. Rechercher des verbatims spécifiques avec des mots-clés
8. Explorer interactivement les résultats

La fonction principale `analyser_verbatims()` vous permet d'exécuter l'ensemble du processus d'analyse. Vous pouvez facilement adapter et étendre ce code selon vos besoins spécifiques.
Je vais créer un code complet pour une analyse de données textuelles dans un cadre d'enquêtes de satisfaction. La demande comprend plusieurs éléments :

1. Générer un fichier Excel synthétique (données fictives) qui suivra la structure décrite
2. Charger ce fichier Excel 
3. Analyser les verbatims pour trouver des thématiques récurrentes, en se concentrant sur les avis négatifs
4. Permettre la traçabilité pour remonter aux sources des problèmes

Je vais structurer le code en plusieurs sections :

1. Génération des données fictives avec pandas et sauvegarde en format .xlsx
2. Chargement des données
3. Prétraitement des textes (nettoyage, lemmatisation)
4. Analyse exploratoire des données
5. Modélisation thématique (Topic Modeling) avec des méthodes comme LDA ou NMF
6. Extraction de n-grammes fréquents
7. Visualisation des résultats
8. Traçabilité (lien entre thèmes et lignes sources)

Je vais créer un notebook Jupyter complet et bien documenté, avec un style formel et professionnel adapté à un contexte d'entreprise.

# Analyse de verbatims d'enquêtes de satisfaction

Je vais vous fournir un code complet pour réaliser cette analyse textuelle des verbatims, avec une attention particulière sur les retours négatifs. Le code sera organisé en plusieurs sections pour plus de clarté.

## 1. Génération d'un jeu de données synthétique

Commençons par générer un fichier Excel synthétique qui servira pour notre démonstration :

```python
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Définition des valeurs possibles pour chaque colonne
humeurs = ["Très satisfaite", "Plutôt Satisfaite", "Plutôt Insatisfaite", "Très insatisfaite"]
categorisations = ["A dispatcher", "A traiter", "En cours", "Terminé"]
noms = ["Dupont Xavier", "Martin Sophie", "Leblanc Julie", "Durand Thomas", "Petit Claire", 
        "Moreau Nicolas", "Richard Emilie", "Robert Philippe", "Simon Catherine", "Leroy Antoine"]
perimetres = ["Epargne et assurances", "Data", "Crédits", "Banque au quotidien", "Moyens de paiement"]
sous_perimetres = ["Tribu Patrimoine", "Tribu digital workplace", "Tribu innovation", "Tribu crédit immobilier", "Tribu e-banking"]
chapitres = ["Tribu data hub", "Tribu dématérialisation", "Tribu expérience client", "Tribu conformité", "Tribu IT"]
metiers = ["Relai fabrication", "Scrum master", "Product owner", "Développeur", "Analyste fonctionnel", "Chef de projet", "Architecte", "Designer UX"]
types_enquete = ["IRC Flash Vague 1", "IRC Flash Vague 2", "IRC Flash Vague 3", "Enquête trimestrielle"]
equipes = ["IBC", "QOS", "DEV", "OPS", "ARCHI", "SECU"]
responsables = ["Martin Dupond", "Garcia Laura", "Vasseur Jean", "Lefebvre Marie", "Dubois Patrick"]
soutiens = ["Hughes Timhas", "Rousseau Marie", "Bernard David", "Fournier Sylvie", "Lambert François"]
statuts = ["Terminé", "En cours", "A planifier", "Suspendu"]

# Verbatims positifs et négatifs pour la génération
verbatims_positifs = [
    "Réactif et compétent, toujours à l'écoute.",
    "Excellente collaboration, équipe très efficace.",
    "Communication claire et adaptée à nos besoins.",
    "Grande disponibilité et solutions toujours pertinentes.",
    "Respect des délais et qualité du travail remarquable.",
    "Expertise technique impressionnante et bons conseils.",
    "Très bon accompagnement sur notre projet complexe.",
    "Capacité d'adaptation exceptionnelle face aux changements.",
    "Proactivité et anticipation des problématiques.",
    "Support technique efficace et rapide."
]

verbatims_negatifs = [
    "Très mauvaise communication récurrente de la part des Socles CAGIP sur les sujets transverses.",
    "Délais non respectés à plusieurs reprises, impact fort sur nos activités.",
    "Manque de compétence technique évident sur le projet.",
    "Difficultés à obtenir des réponses claires à nos questions.",
    "Documentation insuffisante et non mise à jour.",
    "Problèmes récurrents de qualité dans les livrables.",
    "Absence de suivi après la mise en production.",
    "Manque d'implication et de proactivité de l'équipe.",
    "Changements fréquents dans l'équipe perturbant la continuité.",
    "Outils mis à disposition obsolètes et inefficaces.",
    "Communication inexistante lors des incidents majeurs.",
    "Procédures trop complexes pour des demandes simples.",
    "Temps de réponse beaucoup trop long pour les tickets critiques.",
    "Absence totale de prise en compte de nos contraintes métier.",
    "Support technique défaillant et peu disponible."
]

# Génération de plans d'action
plans_action = [
    "Échange avec le manager de Tribu DWP pour résolution rapide.",
    "Organisation d'un atelier d'amélioration continue.",
    "Mise en place d'un nouveau processus de communication.",
    "Formation spécifique prévue pour l'équipe concernée.",
    "Revue des procédures internes et simplification.",
    "Création d'un groupe de travail dédié à cette problématique.",
    "Renforcement de l'équipe avec des profils plus expérimentés.",
    "Audit des outils et mise à jour prévue.",
    "Réorganisation du support pour améliorer les temps de réponse.",
    "Plan d'amélioration de la qualité en cours d'élaboration."
]

# Synthèses
syntheses = [
    "Florian signale ne pas avoir trouvé de solution malgré plusieurs relances.",
    "Problème identifié et résolu par mise à jour du framework.",
    "Incident majeur ayant nécessité une intervention de la DSI.",
    "Demande traitée avec succès après escalade au N+2.",
    "Problématique récurrente nécessitant une refonte complète.",
    "Solution temporaire mise en place en attendant le nouveau système.",
    "Formation effectuée, amélioration constatée immédiatement.",
    "Analyse approfondie révélant un problème d'infrastructure.",
    "Communication renforcée entre équipes suite à cet incident.",
    "Résolution nécessitant l'intervention de plusieurs services."
]

# Générer des dates sur les 6 derniers mois
date_fin = datetime.now()
date_debut = date_fin - timedelta(days=180)
dates_possibles = [date_debut + timedelta(days=x) for x in range(0, 180, 15)]  # Une enquête tous les 15 jours

# Fonction pour générer un verbatim en fonction de l'humeur
def generer_verbatim(humeur):
    if humeur in ["Très satisfaite", "Plutôt Satisfaite"]:
        return random.choice(verbatims_positifs)
    else:
        return random.choice(verbatims_negatifs)

# Fonction pour générer le plan d'action et la synthèse selon l'humeur et la catégorisation
def generer_plan_et_synthese(humeur, categorisation):
    if humeur in ["Plutôt Insatisfaite", "Très insatisfaite"]:
        plan = random.choice(plans_action) if categorisation != "A dispatcher" else ""
        synthese = random.choice(syntheses) if categorisation in ["En cours", "Terminé"] else ""
    else:
        plan = ""
        synthese = "Retour positif noté" if categorisation in ["En cours", "Terminé"] else ""
    
    return plan, synthese

# Créer un DataFrame avec 200 entrées
n_entries = 200
data = []

for i in range(n_entries):
    humeur = random.choices(humeurs, weights=[0.4, 0.3, 0.2, 0.1], k=1)[0]  # Distribution pondérée
    categorisation = random.choice(categorisations)
    verbatim = generer_verbatim(humeur)
    collaborateur = random.choice(noms)
    perimetre = random.choice(perimetres)
    sous_perimetre = random.choice(sous_perimetres)
    chapitre = random.choice(chapitres)
    metier = random.choice(metiers)
    date_enquete = random.choice(dates_possibles).strftime('%Y-%m-%d')
    type_enquete = random.choice(types_enquete)
    equipe = random.choice(equipes)
    responsable = random.choice(responsables)
    soutien = random.choice(soutiens)
    statut = random.choice(statuts)
    plan_action, synthese = generer_plan_et_synthese(humeur, categorisation)
    
    data.append({
        'Humeur': humeur,
        'Categorisation': categorisation,
        'Verbatim': verbatim,
        'Collaborateur qui a emis': collaborateur,
        'Perimetre': perimetre,
        'Sous perimetre': sous_perimetre,
        'Chapitre de la sous tribu': chapitre,
        'Metier CATS': metier,
        'Date de l\'enquete': date_enquete,
        'Type d\'Enquete': type_enquete,
        'Equipe BFR concernee': equipe,
        'Responsable BFR': responsable,
        'Soutien de BFR': soutien,
        'Statut': statut,
        'Plan d\'action': plan_action,
        'Synthese et/ou plan d\'actions': synthese
    })

# Création du DataFrame
df = pd.DataFrame(data)

# Sauvegarde dans un fichier Excel
output_file = 'enquetes_satisfaction.xlsx'
df.to_excel(output_file, index=True)  # Garder l'index pour avoir le numéro de ligne

print(f"Fichier '{output_file}' généré avec succès contenant {n_entries} entrées.")
```

## 2. Chargement et exploration des données

Maintenant, chargeons le fichier généré et explorons les données :

```python
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
import pyLDAvis
import pyLDAvis.sklearn
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
```

## 3. Prétraitement des textes pour l'analyse

```python
# Fonction de prétraitement des textes
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Conversion en minuscules
    text = text.lower()
    
    # Suppression des caractères spéciaux et des chiffres
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Tokenisation
    tokens = word_tokenize(text)
    
    # Suppression des stopwords (mots vides)
    stop_words = set(stopwords.words('french'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# Application du prétraitement aux verbatims négatifs
df_negatif['verbatim_clean'] = df_negatif['Verbatim'].apply(preprocess_text)

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
sns.barplot(x=list(top_words.values()), y=list(top_words.keys()))
plt.title('Mots les plus fréquents dans les verbatims négatifs')
plt.tight_layout()
plt.show()

# Création d'un nuage de mots
wordcloud = WordCloud(width=800, height=400, background_color='white', 
                     max_words=100, contour_width=1, contour_color='steelblue')
wordcloud.generate(' '.join(df_negatif['verbatim_clean']))

plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nuage de mots des verbatims négatifs')
plt.show()
```

## 4. Extraction de n-grammes fréquents

```python
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
```

## 5. Modélisation thématique (Topic Modeling)

```python
# Préparation des données pour la modélisation thématique
vectorizer = TfidfVectorizer(
    max_df=0.95,         # Ignorer les termes qui apparaissent dans plus de 95% des documents
    min_df=2,            # Ignorer les termes qui apparaissent dans moins de 2 documents
    max_features=1000,   # Limiter le nombre de features
    stop_words='english' # Déjà fait dans le prétraitement, mais ajout de sécurité
)

# Appliqué uniquement sur les verbatims nettoyés non vides
valid_docs = df_negatif['verbatim_clean'].dropna().replace('', np.nan).dropna()
if len(valid_docs) == 0:
    print("Aucun document valide pour la modélisation thématique. Vérifiez le prétraitement.")
else:
    X = vectorizer.fit_transform(valid_docs)
    feature_names = vectorizer.get_feature_names_out()

    # Nombre optimal de thèmes (à ajuster selon les résultats)
    n_topics = 5

    # Application de LDA (Latent Dirichlet Allocation)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=10,
        learning_method='online',
        random_state=42
    )
    lda.fit(X)

    # Fonction pour afficher les mots les plus représentatifs de chaque thème
    def display_topics(model, feature_names, n_top_words=10):
        topic_dict = {}
        for topic_idx, topic in enumerate(model.components_):
            top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_dict[f"Thème {topic_idx+1}"] = ", ".join(top_words)
        return topic_dict

    # Affichage des thèmes identifiés
    topics = display_topics(lda, feature_names)
    print("\nThèmes identifiés dans les verbatims négatifs :")
    for theme, words in topics.items():
        print(f"{theme}: {words}")

    # Prédiction des thèmes pour chaque document
    doc_topic_distrib = lda.transform(X)
    df_negatif_topics = df_negatif.loc[valid_docs.index].copy()
    df_negatif_topics['dominant_topic'] = doc_topic_distrib.argmax(axis=1) + 1  # +1 pour commencer à 1 au lieu de 0

    # Distribution des documents par thème
    topic_distribution = df_negatif_topics['dominant_topic'].value_counts().sort_index()
    print("\nDistribution des documents par thème :")
    print(topic_distribution)

    # Visualisation de la distribution des documents par thème
    plt.figure(figsize=(10, 6))
    sns.countplot(x='dominant_topic', data=df_negatif_topics, order=sorted(df_negatif_topics['dominant_topic'].unique()))
    plt.title('Distribution des verbatims négatifs par thème')
    plt.xlabel('Thème dominant')
    plt.ylabel('Nombre de verbatims')
    plt.tight_layout()
    plt.show()

    # Visualisation interactive des thèmes avec pyLDAvis (si dans un notebook)
    try:
        vis_data = pyLDAvis.sklearn.prepare(lda, X, vectorizer)
        pyLDAvis.display(vis_data)
        print("\nVisualisation interactive des thèmes générée avec succès.")
    except Exception as e:
        print(f"\nErreur lors de la génération de la visualisation interactive : {str(e)}")
```

## 6. Analyse de cooccurrences et clustering des verbatims

```python
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
```

## 7. Création d'un tableau de bord pour l'analyse des problématiques récurrentes

```python
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
```

## 8. Fonction pour rechercher des verbatims par mots-clés

```python
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
```

## 9. Création d'une interface utilisateur simple pour l'exploration

```python
# Création d'une interface simple pour explorer les résultats
def explorer_problematiques():
    """
    Fonction interactive pour explorer les problématiques. À exécuter dans un notebook Jupyter.
    """
    from ipywidgets import interact, widgets
    
    # Widget pour sélectionner les humeurs
    humeur_widget = widgets.SelectMultiple(
        options=df['Humeur'].unique(),
        value=['Plutôt Insatisfaite', 'Très insatisfaite'],
        description='Humeurs:',
        disabled=False
    )
    
    # Widget pour entrer des mots-clés
    keywords_widget = widgets.Text(
        value='',
        placeholder='Entrez des mots-clés séparés par des virgules',
        description='Mots-clés:',
        disabled=False
    )
    
    # Widget pour sélectionner les périmètres
    perimetre_widget = widgets.SelectMultiple(
        options=df['Perimetre'].unique(),
        value=[],
        description='Périmètres:',
        disabled=False
    )
    
    # Fonction pour afficher les résultats
    def afficher_resultats(humeurs, keywords, perimetres):
        # Filtre par humeurs
        if humeurs:
            df_filtre = df[df['Humeur'].isin(humeurs)]
        else:
            df_filtre = df.copy()
        
        # Filtre par périmètres
        if perimetres:
            df_filtre = df_filtre[df_filtre['Perimetre'].isin(perimetres)]
        
        # Filtre par mots-clés
        if keywords:
            keywords_list = [k.strip() for k in keywords.split(',')]
            if keywords_list:
                pattern = '|'.join(keywords_list)
                mask = df_filtre['Verbatim'].str.contains(pattern, case=False, na=False, regex=True)
                df_filtre = df_filtre[mask]
        
        # Affichage des résultats
        print(f"Nombre de verbatims correspondants: {len(df_filtre)}")
        
        if len(df_filtre) > 0:
            print("\nRésultats:")
            for idx, row in df_filtre.head(10).iterrows():
                print(f"Ligne {row['ligne_source']} - {row['Humeur']}: {row['Verbatim']}")
                print(f"  Périmètre: {row['Perimetre']} - {row['Sous perimetre']}")
                print(f"  Collaborateur: {row['Collaborateur qui a emis']}")
                print()
            
            if len(df_filtre) > 10:
                print(f"... et {len(df_filtre) - 10} autres résultats.")
        
        # Distribution des périmètres dans les résultats
        if len(df_filtre) > 0:
            perimetre_counts = df_filtre['Perimetre'].value_counts()
            plt.figure(figsize=(10, 6))
            sns.barplot(x=perimetre_counts.index, y=perimetre_counts.values)
            plt.title('Distribution des périmètres dans les résultats')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    
    # Création de l'interface interactive
    interact(afficher_resultats, 
             humeurs=humeur_widget, 
             keywords=keywords_widget, 
             perimetres=perimetre_widget)

# Exécution de l'interface (à utiliser dans un notebook Jupyter)
print("\nPour explorer les problématiques interactivement, exécutez la fonction 'explorer_problematiques()' dans une cellule séparée.")
```

## 10. Fonction principale pour exécuter l'ensemble de l'analyse

```python
def analyser_verbatims(fichier_excel, humeurs_a_analyser=None):
    """
    Fonction principale pour analyser les verbatims d'enquêtes de satisfaction.
    
    Args:
        fichier_excel: Chemin vers le fichier Excel
        humeurs_a_analyser: Liste des humeurs à analyser (None pour toutes)
    """
    # Chargement des données
    print(f"Chargement du fichier {fichier_excel}...")
    df = pd.read_excel(fichier_excel)
    
    # Ajout de l'index pour la traçabilité
    df['ligne_source'] = df.index
    
    # Filtrage par humeur si spécifié
    if humeurs_a_analyser:
        df_filtre = df[df['Humeur'].isin(humeurs_a_analyser)]
        print(f"Analyse limitée aux humeurs: {humeurs_a_analyser}")
    else:
        df_filtre = df.copy()
        print("Analyse de toutes les humeurs")
    
    print(f"Nombre de verbatims à analyser: {len(df_filtre)}")
    
    # Prétraitement des textes
    print("Prétraitement des verbatims...")
    df_filtre['verbatim_clean'] = df_filtre['Verbatim'].apply(preprocess_text)
    
    # Extraction de n-grammes
    print("Extraction des n-grammes fréquents...")
    top_bigrams = extract_ngrams(df_filtre['verbatim_clean'].dropna(), (2, 2))
    
    print("Top 10 bigrammes:")
    for bigram, count in list(top_bigrams.items())[:10]:
        print(f"- {bigram}: {count} occurrences")
    
    # Modélisation thématique
    print("\nModélisation thématique...")
    valid_docs = df_filtre['verbatim_clean'].dropna().replace('', np.nan).dropna()
    
    if len(valid_docs) >= 5:  # Vérification du nombre minimum de documents
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
        X = vectorizer.fit_transform(valid_docs)
        feature_names = vectorizer.get_feature_names_out()
        
        # Détermination du nombre optimal de thèmes (simplifié ici)
        n_topics = min(5, len(valid_docs) // 2)  # Heuristique simple
        
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=10,
            learning_method='online',
            random_state=42
        )
        lda.fit(X)
        
        # Affichage des thèmes
        topics = display_topics(lda, feature_names)
        print("\nThèmes identifiés dans les verbatims:")
        for theme, words in topics.items():
            print(f"{theme}: {words}")
        
        # Attribution des thèmes aux documents
        doc_topic_distrib = lda.transform(X)
        df_topics = df_filtre.loc[valid_docs.index].copy()
        df_topics['dominant_topic'] = doc_topic_distrib.argmax(axis=1) + 1
        
        # Création du tableau de bord des problématiques
        print("\nCréation du tableau de bord des problématiques récurrentes...")
        theme_summary = pd.DataFrame()
        
        for theme_id in sorted(df_topics['dominant_topic'].unique()):
            theme_docs = df_topics[df_topics['dominant_topic'] == theme_id]
            
            theme_key_words = topics[f"Thème {theme_id}"]
            verbatim_examples = theme_docs['Verbatim'].sample(min(3, len(theme_docs))).tolist()
            perimetre_counts = theme_docs['Perimetre'].value_counts().to_dict()
            top_perimetre = max(perimetre_counts.items(), key=lambda x: x[1])[0] if perimetre_counts else "N/A"
            lignes_sources = theme_docs['ligne_source'].tolist()
            
            theme_entry = {
                'Thème ID': theme_id,
                'Mots-clés': theme_key_words,
                'Nombre de verbatims': len(theme_docs),
                'Périmètre principal': top_perimetre,
                'Exemples de verbatims': verbatim_examples,
                'Numéros de lignes sources': lignes_sources
            }
            
            theme_summary = pd.concat([theme_summary, pd.DataFrame([theme_entry])], ignore_index=True)
        
        # Sauvegarde du tableau de bord
        output_file = 'dashboard_problematiques_recurrentes.xlsx'
        theme_summary.to_excel(output_file, index=False)
        print(f"Tableau de bord sauvegardé dans '{output_file}'")
        
        # Retour du résumé pour utilisation ultérieure
        return theme_summary
    else:
        print("Nombre insuffisant de documents valides pour la modélisation thématique.")
        return None

# Exemple d'utilisation
# analyser_verbatims('enquetes_satisfaction.xlsx', ['Plutôt Insatisfaite', 'Très insatisfaite'])
```

Ce code complet vous permet d'analyser les verbatims des enquêtes de satisfaction en utilisant des techniques avancées de NLP. Vous pouvez l'exécuter dans un notebook Jupyter et ainsi :

1. Générer un fichier Excel synthétique pour tester
2. Explorer les données et filtrer les avis négatifs
3. Prétraiter les textes pour l'analyse
4. Extraire les n-grammes fréquents et identifier les mots-clés récurrents
5. Utiliser la modélisation thématique pour découvrir automatiquement les problématiques
6. Créer un tableau de bord des problématiques récurrentes
7. Rechercher des verbatims spécifiques avec des mots-clés
8. Explorer interactivement les résultats

La fonction principale `analyser_verbatims()` vous permet d'exécuter l'ensemble du processus d'analyse. Vous pouvez facilement adapter et étendre ce code selon vos besoins spécifiques.
Je vais créer un code complet pour une analyse de données textuelles dans un cadre d'enquêtes de satisfaction. La demande comprend plusieurs éléments :

1. Générer un fichier Excel synthétique (données fictives) qui suivra la structure décrite
2. Charger ce fichier Excel 
3. Analyser les verbatims pour trouver des thématiques récurrentes, en se concentrant sur les avis négatifs
4. Permettre la traçabilité pour remonter aux sources des problèmes

Je vais structurer le code en plusieurs sections :

1. Génération des données fictives avec pandas et sauvegarde en format .xlsx
2. Chargement des données
3. Prétraitement des textes (nettoyage, lemmatisation)
4. Analyse exploratoire des données
5. Modélisation thématique (Topic Modeling) avec des méthodes comme LDA ou NMF
6. Extraction de n-grammes fréquents
7. Visualisation des résultats
8. Traçabilité (lien entre thèmes et lignes sources)

Je vais créer un notebook Jupyter complet et bien documenté, avec un style formel et professionnel adapté à un contexte d'entreprise.

# Analyse de verbatims d'enquêtes de satisfaction

Je vais vous fournir un code complet pour réaliser cette analyse textuelle des verbatims, avec une attention particulière sur les retours négatifs. Le code sera organisé en plusieurs sections pour plus de clarté.

## 1. Génération d'un jeu de données synthétique

Commençons par générer un fichier Excel synthétique qui servira pour notre démonstration :

```python
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Définition des valeurs possibles pour chaque colonne
humeurs = ["Très satisfaite", "Plutôt Satisfaite", "Plutôt Insatisfaite", "Très insatisfaite"]
categorisations = ["A dispatcher", "A traiter", "En cours", "Terminé"]
noms = ["Dupont Xavier", "Martin Sophie", "Leblanc Julie", "Durand Thomas", "Petit Claire", 
        "Moreau Nicolas", "Richard Emilie", "Robert Philippe", "Simon Catherine", "Leroy Antoine"]
perimetres = ["Epargne et assurances", "Data", "Crédits", "Banque au quotidien", "Moyens de paiement"]
sous_perimetres = ["Tribu Patrimoine", "Tribu digital workplace", "Tribu innovation", "Tribu crédit immobilier", "Tribu e-banking"]
chapitres = ["Tribu data hub", "Tribu dématérialisation", "Tribu expérience client", "Tribu conformité", "Tribu IT"]
metiers = ["Relai fabrication", "Scrum master", "Product owner", "Développeur", "Analyste fonctionnel", "Chef de projet", "Architecte", "Designer UX"]
types_enquete = ["IRC Flash Vague 1", "IRC Flash Vague 2", "IRC Flash Vague 3", "Enquête trimestrielle"]
equipes = ["IBC", "QOS", "DEV", "OPS", "ARCHI", "SECU"]
responsables = ["Martin Dupond", "Garcia Laura", "Vasseur Jean", "Lefebvre Marie", "Dubois Patrick"]
soutiens = ["Hughes Timhas", "Rousseau Marie", "Bernard David", "Fournier Sylvie", "Lambert François"]
statuts = ["Terminé", "En cours", "A planifier", "Suspendu"]

# Verbatims positifs et négatifs pour la génération
verbatims_positifs = [
    "Réactif et compétent, toujours à l'écoute.",
    "Excellente collaboration, équipe très efficace.",
    "Communication claire et adaptée à nos besoins.",
    "Grande disponibilité et solutions toujours pertinentes.",
    "Respect des délais et qualité du travail remarquable.",
    "Expertise technique impressionnante et bons conseils.",
    "Très bon accompagnement sur notre projet complexe.",
    "Capacité d'adaptation exceptionnelle face aux changements.",
    "Proactivité et anticipation des problématiques.",
    "Support technique efficace et rapide."
]

verbatims_negatifs = [
    "Très mauvaise communication récurrente de la part des Socles CAGIP sur les sujets transverses.",
    "Délais non respectés à plusieurs reprises, impact fort sur nos activités.",
    "Manque de compétence technique évident sur le projet.",
    "Difficultés à obtenir des réponses claires à nos questions.",
    "Documentation insuffisante et non mise à jour.",
    "Problèmes récurrents de qualité dans les livrables.",
    "Absence de suivi après la mise en production.",
    "Manque d'implication et de proactivité de l'équipe.",
    "Changements fréquents dans l'équipe perturbant la continuité.",
    "Outils mis à disposition obsolètes et inefficaces.",
    "Communication inexistante lors des incidents majeurs.",
    "Procédures trop complexes pour des demandes simples.",
    "Temps de réponse beaucoup trop long pour les tickets critiques.",
    "Absence totale de prise en compte de nos contraintes métier.",
    "Support technique défaillant et peu disponible."
]

# Génération de plans d'action
plans_action = [
    "Échange avec le manager de Tribu DWP pour résolution rapide.",
    "Organisation d'un atelier d'amélioration continue.",
    "Mise en place d'un nouveau processus de communication.",
    "Formation spécifique prévue pour l'équipe concernée.",
    "Revue des procédures internes et simplification.",
    "Création d'un groupe de travail dédié à cette problématique.",
    "Renforcement de l'équipe avec des profils plus expérimentés.",
    "Audit des outils et mise à jour prévue.",
    "Réorganisation du support pour améliorer les temps de réponse.",
    "Plan d'amélioration de la qualité en cours d'élaboration."
]

# Synthèses
syntheses = [
    "Florian signale ne pas avoir trouvé de solution malgré plusieurs relances.",
    "Problème identifié et résolu par mise à jour du framework.",
    "Incident majeur ayant nécessité une intervention de la DSI.",
    "Demande traitée avec succès après escalade au N+2.",
    "Problématique récurrente nécessitant une refonte complète.",
    "Solution temporaire mise en place en attendant le nouveau système.",
    "Formation effectuée, amélioration constatée immédiatement.",
    "Analyse approfondie révélant un problème d'infrastructure.",
    "Communication renforcée entre équipes suite à cet incident.",
    "Résolution nécessitant l'intervention de plusieurs services."
]

# Générer des dates sur les 6 derniers mois
date_fin = datetime.now()
date_debut = date_fin - timedelta(days=180)
dates_possibles = [date_debut + timedelta(days=x) for x in range(0, 180, 15)]  # Une enquête tous les 15 jours

# Fonction pour générer un verbatim en fonction de l'humeur
def generer_verbatim(humeur):
    if humeur in ["Très satisfaite", "Plutôt Satisfaite"]:
        return random.choice(verbatims_positifs)
    else:
        return random.choice(verbatims_negatifs)

# Fonction pour générer le plan d'action et la synthèse selon l'humeur et la catégorisation
def generer_plan_et_synthese(humeur, categorisation):
    if humeur in ["Plutôt Insatisfaite", "Très insatisfaite"]:
        plan = random.choice(plans_action) if categorisation != "A dispatcher" else ""
        synthese = random.choice(syntheses) if categorisation in ["En cours", "Terminé"] else ""
    else:
        plan = ""
        synthese = "Retour positif noté" if categorisation in ["En cours", "Terminé"] else ""
    
    return plan, synthese

# Créer un DataFrame avec 200 entrées
n_entries = 200
data = []

for i in range(n_entries):
    humeur = random.choices(humeurs, weights=[0.4, 0.3, 0.2, 0.1], k=1)[0]  # Distribution pondérée
    categorisation = random.choice(categorisations)
    verbatim = generer_verbatim(humeur)
    collaborateur = random.choice(noms)
    perimetre = random.choice(perimetres)
    sous_perimetre = random.choice(sous_perimetres)
    chapitre = random.choice(chapitres)
    metier = random.choice(metiers)
    date_enquete = random.choice(dates_possibles).strftime('%Y-%m-%d')
    type_enquete = random.choice(types_enquete)
    equipe = random.choice(equipes)
    responsable = random.choice(responsables)
    soutien = random.choice(soutiens)
    statut = random.choice(statuts)
    plan_action, synthese = generer_plan_et_synthese(humeur, categorisation)
    
    data.append({
        'Humeur': humeur,
        'Categorisation': categorisation,
        'Verbatim': verbatim,
        'Collaborateur qui a emis': collaborateur,
        'Perimetre': perimetre,
        'Sous perimetre': sous_perimetre,
        'Chapitre de la sous tribu': chapitre,
        'Metier CATS': metier,
        'Date de l\'enquete': date_enquete,
        'Type d\'Enquete': type_enquete,
        'Equipe BFR concernee': equipe,
        'Responsable BFR': responsable,
        'Soutien de BFR': soutien,
        'Statut': statut,
        'Plan d\'action': plan_action,
        'Synthese et/ou plan d\'actions': synthese
    })

# Création du DataFrame
df = pd.DataFrame(data)

# Sauvegarde dans un fichier Excel
output_file = 'enquetes_satisfaction.xlsx'
df.to_excel(output_file, index=True)  # Garder l'index pour avoir le numéro de ligne

print(f"Fichier '{output_file}' généré avec succès contenant {n_entries} entrées.")
```

## 2. Chargement et exploration des données

Maintenant, chargeons le fichier généré et explorons les données :

```python
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
import pyLDAvis
import pyLDAvis.sklearn
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
```

## 3. Prétraitement des textes pour l'analyse

```python
# Fonction de prétraitement des textes
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Conversion en minuscules
    text = text.lower()
    
    # Suppression des caractères spéciaux et des chiffres
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Tokenisation
    tokens = word_tokenize(text)
    
    # Suppression des stopwords (mots vides)
    stop_words = set(stopwords.words('french'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# Application du prétraitement aux verbatims négatifs
df_negatif['verbatim_clean'] = df_negatif['Verbatim'].apply(preprocess_text)

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
sns.barplot(x=list(top_words.values()), y=list(top_words.keys()))
plt.title('Mots les plus fréquents dans les verbatims négatifs')
plt.tight_layout()
plt.show()

# Création d'un nuage de mots
wordcloud = WordCloud(width=800, height=400, background_color='white', 
                     max_words=100, contour_width=1, contour_color='steelblue')
wordcloud.generate(' '.join(df_negatif['verbatim_clean']))

plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nuage de mots des verbatims négatifs')
plt.show()
```

## 4. Extraction de n-grammes fréquents

```python
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
```

## 5. Modélisation thématique (Topic Modeling)

```python
# Préparation des données pour la modélisation thématique
vectorizer = TfidfVectorizer(
    max_df=0.95,         # Ignorer les termes qui apparaissent dans plus de 95% des documents
    min_df=2,            # Ignorer les termes qui apparaissent dans moins de 2 documents
    max_features=1000,   # Limiter le nombre de features
    stop_words='english' # Déjà fait dans le prétraitement, mais ajout de sécurité
)

# Appliqué uniquement sur les verbatims nettoyés non vides
valid_docs = df_negatif['verbatim_clean'].dropna().replace('', np.nan).dropna()
if len(valid_docs) == 0:
    print("Aucun document valide pour la modélisation thématique. Vérifiez le prétraitement.")
else:
    X = vectorizer.fit_transform(valid_docs)
    feature_names = vectorizer.get_feature_names_out()

    # Nombre optimal de thèmes (à ajuster selon les résultats)
    n_topics = 5

    # Application de LDA (Latent Dirichlet Allocation)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=10,
        learning_method='online',
        random_state=42
    )
    lda.fit(X)

    # Fonction pour afficher les mots les plus représentatifs de chaque thème
    def display_topics(model, feature_names, n_top_words=10):
        topic_dict = {}
        for topic_idx, topic in enumerate(model.components_):
            top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_dict[f"Thème {topic_idx+1}"] = ", ".join(top_words)
        return topic_dict

    # Affichage des thèmes identifiés
    topics = display_topics(lda, feature_names)
    print("\nThèmes identifiés dans les verbatims négatifs :")
    for theme, words in topics.items():
        print(f"{theme}: {words}")

    # Prédiction des thèmes pour chaque document
    doc_topic_distrib = lda.transform(X)
    df_negatif_topics = df_negatif.loc[valid_docs.index].copy()
    df_negatif_topics['dominant_topic'] = doc_topic_distrib.argmax(axis=1) + 1  # +1 pour commencer à 1 au lieu de 0

    # Distribution des documents par thème
    topic_distribution = df_negatif_topics['dominant_topic'].value_counts().sort_index()
    print("\nDistribution des documents par thème :")
    print(topic_distribution)

    # Visualisation de la distribution des documents par thème
    plt.figure(figsize=(10, 6))
    sns.countplot(x='dominant_topic', data=df_negatif_topics, order=sorted(df_negatif_topics['dominant_topic'].unique()))
    plt.title('Distribution des verbatims négatifs par thème')
    plt.xlabel('Thème dominant')
    plt.ylabel('Nombre de verbatims')
    plt.tight_layout()
    plt.show()

    # Visualisation interactive des thèmes avec pyLDAvis (si dans un notebook)
    try:
        vis_data = pyLDAvis.sklearn.prepare(lda, X, vectorizer)
        pyLDAvis.display(vis_data)
        print("\nVisualisation interactive des thèmes générée avec succès.")
    except Exception as e:
        print(f"\nErreur lors de la génération de la visualisation interactive : {str(e)}")
```

## 6. Analyse de cooccurrences et clustering des verbatims

```python
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
```

## 7. Création d'un tableau de bord pour l'analyse des problématiques récurrentes

```python
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
```

## 8. Fonction pour rechercher des verbatims par mots-clés

```python
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
```

## 9. Création d'une interface utilisateur simple pour l'exploration

```python
# Création d'une interface simple pour explorer les résultats
def explorer_problematiques():
    """
    Fonction interactive pour explorer les problématiques. À exécuter dans un notebook Jupyter.
    """
    from ipywidgets import interact, widgets
    
    # Widget pour sélectionner les humeurs
    humeur_widget = widgets.SelectMultiple(
        options=df['Humeur'].unique(),
        value=['Plutôt Insatisfaite', 'Très insatisfaite'],
        description='Humeurs:',
        disabled=False
    )
    
    # Widget pour entrer des mots-clés
    keywords_widget = widgets.Text(
        value='',
        placeholder='Entrez des mots-clés séparés par des virgules',
        description='Mots-clés:',
        disabled=False
    )
    
    # Widget pour sélectionner les périmètres
    perimetre_widget = widgets.SelectMultiple(
        options=df['Perimetre'].unique(),
        value=[],
        description='Périmètres:',
        disabled=False
    )
    
    # Fonction pour afficher les résultats
    def afficher_resultats(humeurs, keywords, perimetres):
        # Filtre par humeurs
        if humeurs:
            df_filtre = df[df['Humeur'].isin(humeurs)]
        else:
            df_filtre = df.copy()
        
        # Filtre par périmètres
        if perimetres:
            df_filtre = df_filtre[df_filtre['Perimetre'].isin(perimetres)]
        
        # Filtre par mots-clés
        if keywords:
            keywords_list = [k.strip() for k in keywords.split(',')]
            if keywords_list:
                pattern = '|'.join(keywords_list)
                mask = df_filtre['Verbatim'].str.contains(pattern, case=False, na=False, regex=True)
                df_filtre = df_filtre[mask]
        
        # Affichage des résultats
        print(f"Nombre de verbatims correspondants: {len(df_filtre)}")
        
        if len(df_filtre) > 0:
            print("\nRésultats:")
            for idx, row in df_filtre.head(10).iterrows():
                print(f"Ligne {row['ligne_source']} - {row['Humeur']}: {row['Verbatim']}")
                print(f"  Périmètre: {row['Perimetre']} - {row['Sous perimetre']}")
                print(f"  Collaborateur: {row['Collaborateur qui a emis']}")
                print()
            
            if len(df_filtre) > 10:
                print(f"... et {len(df_filtre) - 10} autres résultats.")
        
        # Distribution des périmètres dans les résultats
        if len(df_filtre) > 0:
            perimetre_counts = df_filtre['Perimetre'].value_counts()
            plt.figure(figsize=(10, 6))
            sns.barplot(x=perimetre_counts.index, y=perimetre_counts.values)
            plt.title('Distribution des périmètres dans les résultats')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    
    # Création de l'interface interactive
    interact(afficher_resultats, 
             humeurs=humeur_widget, 
             keywords=keywords_widget, 
             perimetres=perimetre_widget)

# Exécution de l'interface (à utiliser dans un notebook Jupyter)
print("\nPour explorer les problématiques interactivement, exécutez la fonction 'explorer_problematiques()' dans une cellule séparée.")
```

## 10. Fonction principale pour exécuter l'ensemble de l'analyse

```python
def analyser_verbatims(fichier_excel, humeurs_a_analyser=None):
    """
    Fonction principale pour analyser les verbatims d'enquêtes de satisfaction.
    
    Args:
        fichier_excel: Chemin vers le fichier Excel
        humeurs_a_analyser: Liste des humeurs à analyser (None pour toutes)
    """
    # Chargement des données
    print(f"Chargement du fichier {fichier_excel}...")
    df = pd.read_excel(fichier_excel)
    
    # Ajout de l'index pour la traçabilité
    df['ligne_source'] = df.index
    
    # Filtrage par humeur si spécifié
    if humeurs_a_analyser:
        df_filtre = df[df['Humeur'].isin(humeurs_a_analyser)]
        print(f"Analyse limitée aux humeurs: {humeurs_a_analyser}")
    else:
        df_filtre = df.copy()
        print("Analyse de toutes les humeurs")
    
    print(f"Nombre de verbatims à analyser: {len(df_filtre)}")
    
    # Prétraitement des textes
    print("Prétraitement des verbatims...")
    df_filtre['verbatim_clean'] = df_filtre['Verbatim'].apply(preprocess_text)
    
    # Extraction de n-grammes
    print("Extraction des n-grammes fréquents...")
    top_bigrams = extract_ngrams(df_filtre['verbatim_clean'].dropna(), (2, 2))
    
    print("Top 10 bigrammes:")
    for bigram, count in list(top_bigrams.items())[:10]:
        print(f"- {bigram}: {count} occurrences")
    
    # Modélisation thématique
    print("\nModélisation thématique...")
    valid_docs = df_filtre['verbatim_clean'].dropna().replace('', np.nan).dropna()
    
    if len(valid_docs) >= 5:  # Vérification du nombre minimum de documents
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
        X = vectorizer.fit_transform(valid_docs)
        feature_names = vectorizer.get_feature_names_out()
        
        # Détermination du nombre optimal de thèmes (simplifié ici)
        n_topics = min(5, len(valid_docs) // 2)  # Heuristique simple
        
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=10,
            learning_method='online',
            random_state=42
        )
        lda.fit(X)
        
        # Affichage des thèmes
        topics = display_topics(lda, feature_names)
        print("\nThèmes identifiés dans les verbatims:")
        for theme, words in topics.items():
            print(f"{theme}: {words}")
        
        # Attribution des thèmes aux documents
        doc_topic_distrib = lda.transform(X)
        df_topics = df_filtre.loc[valid_docs.index].copy()
        df_topics['dominant_topic'] = doc_topic_distrib.argmax(axis=1) + 1
        
        # Création du tableau de bord des problématiques
        print("\nCréation du tableau de bord des problématiques récurrentes...")
        theme_summary = pd.DataFrame()
        
        for theme_id in sorted(df_topics['dominant_topic'].unique()):
            theme_docs = df_topics[df_topics['dominant_topic'] == theme_id]
            
            theme_key_words = topics[f"Thème {theme_id}"]
            verbatim_examples = theme_docs['Verbatim'].sample(min(3, len(theme_docs))).tolist()
            perimetre_counts = theme_docs['Perimetre'].value_counts().to_dict()
            top_perimetre = max(perimetre_counts.items(), key=lambda x: x[1])[0] if perimetre_counts else "N/A"
            lignes_sources = theme_docs['ligne_source'].tolist()
            
            theme_entry = {
                'Thème ID': theme_id,
                'Mots-clés': theme_key_words,
                'Nombre de verbatims': len(theme_docs),
                'Périmètre principal': top_perimetre,
                'Exemples de verbatims': verbatim_examples,
                'Numéros de lignes sources': lignes_sources
            }
            
            theme_summary = pd.concat([theme_summary, pd.DataFrame([theme_entry])], ignore_index=True)
        
        # Sauvegarde du tableau de bord
        output_file = 'dashboard_problematiques_recurrentes.xlsx'
        theme_summary.to_excel(output_file, index=False)
        print(f"Tableau de bord sauvegardé dans '{output_file}'")
        
        # Retour du résumé pour utilisation ultérieure
        return theme_summary
    else:
        print("Nombre insuffisant de documents valides pour la modélisation thématique.")
        return None

# Exemple d'utilisation
# analyser_verbatims('enquetes_satisfaction.xlsx', ['Plutôt Insatisfaite', 'Très insatisfaite'])
```

Ce code complet vous permet d'analyser les verbatims des enquêtes de satisfaction en utilisant des techniques avancées de NLP. Vous pouvez l'exécuter dans un notebook Jupyter et ainsi :

1. Générer un fichier Excel synthétique pour tester
2. Explorer les données et filtrer les avis négatifs
3. Prétraiter les textes pour l'analyse
4. Extraire les n-grammes fréquents et identifier les mots-clés récurrents
5. Utiliser la modélisation thématique pour découvrir automatiquement les problématiques
6. Créer un tableau de bord des problématiques récurrentes
7. Rechercher des verbatims spécifiques avec des mots-clés
8. Explorer interactivement les résultats

La fonction principale `analyser_verbatims()` vous permet d'exécuter l'ensemble du processus d'analyse. Vous pouvez facilement adapter et étendre ce code selon vos besoins spécifiques.

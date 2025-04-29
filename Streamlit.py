import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import tempfile
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Analyse des Enquêtes de Satisfaction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0277BD;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.4rem;
        color: #0288D1;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    .footer {
        text-align: center;
        color: #757575;
        font-size: 0.8rem;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour le prétraitement des textes
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

# Fonction pour créer un nuage de mots
def generate_wordcloud(text):
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        contour_width=1,
        contour_color='steelblue',
        collocations=False
    ).generate(text)
    
    return wordcloud

# Fonction pour la modélisation thématique
def create_topic_model(df, n_topics=5, max_features=1000):
    # Vecteurisation des textes
    vectorizer = TfidfVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=max_features,
        ngram_range=(1, 2)
    )
    
    # Transformation des textes en matrice TF-IDF
    X = vectorizer.fit_transform(df['verbatim_clean'])
    
    # Extraction des noms des features
    feature_names = vectorizer.get_feature_names_out()
    
    # Création et entraînement du modèle LDA
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=20,
        learning_method='online',
        random_state=42,
        n_jobs=1
    )
    
    lda.fit(X)
    
    # Transformation des documents en distribution de thèmes
    doc_topic_distrib = lda.transform(X)
    
    # Attribution du thème dominant à chaque document
    df_with_topics = df.copy()
    df_with_topics['dominant_topic'] = doc_topic_distrib.argmax(axis=1) + 1
    
    # Extraction des termes les plus importants pour chaque thème
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-11:-1]  # Top 10 mots
        top_words = [feature_names[i] for i in top_words_idx]
        topics[f"Thème {topic_idx+1}"] = ", ".join(top_words)
    
    return df_with_topics, topics, X, vectorizer, lda

# Fonction pour télécharger le DataFrame
def get_table_download_link(df, filename, link_text):
    """Génère un lien pour télécharger le dataframe en Excel"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    b64 = base64.b64encode(output.getvalue()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# Fonction pour créer un tableau de bord des problématiques récurrentes
def create_dashboard(df_topics, topics):
    theme_summary = pd.DataFrame()
    
    for theme_id in sorted(df_topics['dominant_topic'].unique()):
        theme_docs = df_topics[df_topics['dominant_topic'] == theme_id]
        
        # Top mots-clés pour ce thème
        theme_key_words = topics[f"Thème {theme_id}"]
        
        # Exemples de verbatims pour ce thème
        verbatim_examples = theme_docs['Verbatim'].sample(min(3, len(theme_docs))).tolist()
        
        # Statistiques par PP/Direction CA-TS
        perimetre_counts = theme_docs['PP/Direction CA-TS'].value_counts().to_dict()
        top_perimetre = max(perimetre_counts.items(), key=lambda x: x[1])[0] if perimetre_counts else "N/A"
        
        # Informations sur les lignes sources
        lignes_sources = theme_docs['ligne_source'].tolist()
        
        # Construire l'entrée pour ce thème
        theme_entry = {
            'Thème ID': theme_id,
            'Mots-clés': theme_key_words,
            'Nombre de verbatims': len(theme_docs),
            'PP/Direction principale': top_perimetre,
            'Exemples de verbatims': verbatim_examples,
            'Numéros de lignes sources': lignes_sources
        }
        
        # Ajouter à notre résumé
        theme_summary = pd.concat([theme_summary, pd.DataFrame([theme_entry])], ignore_index=True)
    
    return theme_summary

# Fonction principale pour l'analyse des données
def analyze_data(df):
    # 1. Exploration initiale
    st.markdown('<div class="sub-header">1. Exploration initiale des données</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">Aperçu des données</div>', unsafe_allow_html=True)
        st.dataframe(df.head())
    
    with col2:
        st.markdown('<div class="section-header">Informations sur le dataset</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="highlight">
            <p>Nombre d'entrées : {df.shape[0]}</p>
            <p>Nombre de colonnes : {df.shape[1]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 2. Distribution des humeurs
    st.markdown('<div class="section-header">Distribution des humeurs</div>', unsafe_allow_html=True)
    
    humeur_counts = df['Humeur'].value_counts()
    
    fig = px.bar(
        x=humeur_counts.index,
        y=humeur_counts.values,
        labels={'x': 'Humeur', 'y': 'Nombre de réponses'},
        title='Distribution des humeurs',
        color=humeur_counts.values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis={'categoryorder': 'total descending'},
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. Filtrage des données négatives
    st.markdown('<div class="sub-header">2. Analyse des retours négatifs</div>', unsafe_allow_html=True)
    
    # Permettre à l'utilisateur de sélectionner les valeurs négatives
    all_humeurs = df['Humeur'].unique()
    negative_humeurs = st.multiselect(
        "Sélectionnez les humeurs à considérer comme négatives",
        all_humeurs,
        default=[h for h in all_humeurs if 'insatisfait' in h.lower() or 'négativ' in h.lower()]
    )
    
    df_negatif = df[df['Humeur'].isin(negative_humeurs)]
    
    st.markdown(f"""
    <div class="highlight">
        <p>Nombre d'entrées négatives sélectionnées : {df_negatif.shape[0]}</p>
        <p>Pourcentage du total : {(df_negatif.shape[0] / df.shape[0] * 100):.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 4. Distribution des PP/Direction CA-TS dans les retours négatifs
    if not df_negatif.empty and 'PP/Direction CA-TS' in df_negatif.columns:
        st.markdown('<div class="section-header">PP/Direction CA-TS avec le plus de retours négatifs</div>', unsafe_allow_html=True)
        
        perimetre_counts = df_negatif['PP/Direction CA-TS'].value_counts()
        
        fig = px.bar(
            x=perimetre_counts.index,
            y=perimetre_counts.values,
            labels={'x': 'PP/Direction CA-TS', 'y': 'Nombre de retours négatifs'},
            title='Distribution des retours négatifs par PP/Direction CA-TS',
            color=perimetre_counts.values,
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(
            xaxis={'categoryorder': 'total descending', 'tickangle': 45},
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 5. Statut de traitement
    if not df_negatif.empty and 'Statut' in df_negatif.columns:
        st.markdown('<div class="section-header">Statut de traitement des retours négatifs</div>', unsafe_allow_html=True)
        
        statut_counts = df_negatif['Statut'].value_counts()
        
        fig = px.pie(
            values=statut_counts.values,
            names=statut_counts.index,
            title='Répartition des statuts de traitement',
            color_discrete_sequence=px.colors.sequential.Plasma_r
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 6. Analyse temporelle des retours négatifs
    if not df_negatif.empty and 'Date Enquête' in df_negatif.columns:
        st.markdown('<div class="section-header">Évolution temporelle des retours négatifs</div>', unsafe_allow_html=True)
        
        # Conversion en datetime si nécessaire
        if not pd.api.types.is_datetime64_any_dtype(df_negatif['Date Enquête']):
            df_negatif['Date Enquête'] = pd.to_datetime(df_negatif['Date Enquête'], errors='coerce')
        
        # Agrégation par mois
        df_negatif['mois'] = df_negatif['Date Enquête'].dt.to_period('M').astype(str)
        trend_data = df_negatif.groupby('mois').size().reset_index(name='count')
        
        fig = px.line(
            trend_data, 
            x='mois', 
            y='count',
            markers=True,
            labels={'mois': 'Mois', 'count': 'Nombre de retours négatifs'},
            title='Évolution temporelle des retours négatifs'
        )
        
        fig.update_layout(
            xaxis={'tickangle': 45},
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 7. Comparaison des humeurs par périmètre
    if 'PP/Direction CA-TS' in df.columns:
        st.markdown('<div class="section-header">Distribution des humeurs par PP/Direction CA-TS</div>', unsafe_allow_html=True)
        
        humeur_perim = pd.crosstab(df['PP/Direction CA-TS'], df['Humeur'])
        humeur_perim_pct = humeur_perim.div(humeur_perim.sum(axis=1), axis=0) * 100
        
        # Conversion en format long pour plotly
        humeur_perim_pct_long = humeur_perim_pct.reset_index().melt(
            id_vars=['PP/Direction CA-TS'],
            var_name='Humeur',
            value_name='Pourcentage'
        )
        
        fig = px.bar(
            humeur_perim_pct_long,
            x='PP/Direction CA-TS',
            y='Pourcentage',
            color='Humeur',
            title='Distribution des humeurs par PP/Direction CA-TS (%)',
            labels={'PP/Direction CA-TS': 'PP/Direction CA-TS', 'Pourcentage': 'Pourcentage (%)'}
        )
        
        fig.update_layout(
            xaxis={'categoryorder': 'total descending', 'tickangle': 45},
            height=600,
            barmode='stack'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 8. Prétraitement des textes pour l'analyse
    if not df_negatif.empty and 'Verbatim' in df_negatif.columns:
        st.markdown('<div class="sub-header">3. Analyse des verbatims</div>', unsafe_allow_html=True)
        
        # Application du prétraitement aux verbatims négatifs
        df_negatif['verbatim_clean'] = df_negatif['Verbatim'].apply(preprocess_text_simple)
        
        # Visualisation des exemples de verbatims nettoyés
        st.markdown('<div class="section-header">Exemples de verbatims nettoyés</div>', unsafe_allow_html=True)
        
        examples = pd.DataFrame({
            'Original': df_negatif['Verbatim'].head(5),
            'Nettoyé': df_negatif['verbatim_clean'].head(5)
        })
        
        st.dataframe(examples)
        
        # 9. Visualisation des mots les plus fréquents
        st.markdown('<div class="section-header">Mots les plus fréquents dans les verbatims négatifs</div>', unsafe_allow_html=True)
        
        all_words = ' '.join(df_negatif['verbatim_clean'].dropna()).split()
        word_counts = Counter(all_words)
        top_words = dict(word_counts.most_common(20))
        
        fig = px.bar(
            x=list(top_words.values()),
            y=list(top_words.keys()),
            orientation='h',
            labels={'x': 'Nombre d\'occurrences', 'y': 'Mot'},
            title='Top 20 des mots les plus fréquents',
            color=list(top_words.values()),
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 10. Nuage de mots
        st.markdown('<div class="section-header">Nuage de mots des verbatims négatifs</div>', unsafe_allow_html=True)
        
        wordcloud = generate_wordcloud(' '.join(df_negatif['verbatim_clean'].dropna()))
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        st.pyplot(fig)
        
        # 11. Analyse des n-grammes fréquents
        st.markdown('<div class="sub-header">4. Analyse des expressions récurrentes</div>', unsafe_allow_html=True)
        
        # Extraction des n-grammes fréquents
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header">Bigrammes les plus fréquents</div>', unsafe_allow_html=True)
            top_bigrams = extract_ngrams(df_negatif['verbatim_clean'].dropna(), (2, 2), 15)
            
            fig = px.bar(
                x=list(top_bigrams.values()),
                y=list(top_bigrams.keys()),
                orientation='h',
                labels={'x': 'Nombre d\'occurrences', 'y': 'Bigramme'},
                color=list(top_bigrams.values()),
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="section-header">Trigrammes les plus fréquents</div>', unsafe_allow_html=True)
            top_trigrams = extract_ngrams(df_negatif['verbatim_clean'].dropna(), (3, 3), 15)
            
            fig = px.bar(
                x=list(top_trigrams.values()),
                y=list(top_trigrams.keys()),
                orientation='h',
                labels={'x': 'Nombre d\'occurrences', 'y': 'Trigramme'},
                color=list(top_trigrams.values()),
                color_continuous_scale='Greens'
            )
            
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 12. Modélisation thématique
        st.markdown('<div class="sub-header">5. Modélisation thématique</div>', unsafe_allow_html=True)
        
        # Filtrage des documents valides pour la modélisation
        valid_docs = df_negatif['verbatim_clean'].dropna().replace('', np.nan).dropna()
        
        if len(valid_docs) < 5:
            st.warning("Nombre insuffisant de documents valides pour la modélisation thématique.")
        else:
            # Paramètres de modélisation
            col1, col2 = st.columns(2)
            
            with col1:
                n_topics = st.slider(
                    "Nombre de thèmes à identifier",
                    min_value=2,
                    max_value=min(10, len(valid_docs) // 5),
                    value=min(5, len(valid_docs) // 5),
                    help="Nombre de thèmes à identifier dans les verbatims"
                )
            
            with col2:
                max_features = st.slider(
                    "Nombre maximum de features (mots)",
                    min_value=100,
                    max_value=2000,
                    value=1000,
                    step=100,
                    help="Nombre maximum de mots uniques à considérer"
                )
            
            # Création du modèle
            df_negatif_topics, topics, X, vectorizer, lda = create_topic_model(
                df_negatif.loc[valid_docs.index],
                n_topics,
                max_features
            )
            
            # Affichage des thèmes identifiés
            st.markdown('<div class="section-header">Thèmes identifiés avec les termes les plus importants</div>', unsafe_allow_html=True)
            
            topics_df = pd.DataFrame({
                'Thème': list(topics.keys()),
                'Mots-clés': list(topics.values())
            })
            
            st.dataframe(topics_df)
            
            # Distribution des documents par thème
            st.markdown('<div class="section-header">Distribution des documents par thème</div>', unsafe_allow_html=True)
            
            topic_counts = df_negatif_topics['dominant_topic'].value_counts().sort_index()
            
            fig = px.bar(
                x=topic_counts.index,
                y=topic_counts.values,
                labels={'x': 'Thème', 'y': 'Nombre de documents'},
                title='Nombre de verbatims par thème',
                color=topic_counts.values,
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(height=500)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution des thèmes par PP/Direction CA-TS
            if 'PP/Direction CA-TS' in df_negatif_topics.columns:
                st.markdown('<div class="section-header">Distribution des thèmes par PP/Direction CA-TS</div>', unsafe_allow_html=True)
                
                theme_perimetre = pd.crosstab(df_negatif_topics['dominant_topic'], df_negatif_topics['PP/Direction CA-TS'])
                theme_perimetre_pct = theme_perimetre.div(theme_perimetre.sum(axis=0), axis=1) * 100
                
                # Conversion en format long pour plotly
                theme_perimetre_pct_long = theme_perimetre_pct.reset_index().melt(
                    id_vars=['dominant_topic'],
                    var_name='PP/Direction CA-TS',
                    value_name='Pourcentage'
                )
                
                fig = px.bar(
                    theme_perimetre_pct_long,
                    x='dominant_topic',
                    y='Pourcentage',
                    color='PP/Direction CA-TS',
                    title='Distribution des thèmes par PP/Direction CA-TS (%)',
                    labels={'dominant_topic': 'Thème', 'Pourcentage': 'Pourcentage (%)'}
                )
                
                fig.update_layout(
                    xaxis={'title': 'Thème dominant'},
                    height=600,
                    barmode='stack'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Exemples de verbatims par thème
            st.markdown('<div class="section-header">Exemples de verbatims par thème</div>', unsafe_allow_html=True)
            
            for theme in sorted(df_negatif_topics['dominant_topic'].unique()):
                theme_docs = df_negatif_topics[df_negatif_topics['dominant_topic'] == theme]
                
                st.markdown(f"""
                <div class="card">
                    <h4>Thème {int(theme)} - {topics[f'Thème {theme}']} ({len(theme_docs)} documents)</h4>
                    <hr>
                """, unsafe_allow_html=True)
                
                for i, (idx, row) in enumerate(theme_docs.sample(min(3, len(theme_docs))).iterrows(), 1):
                    st.markdown(f"""
                    <p><strong>Exemple {i}:</strong> {row['Verbatim']}</p>
                    <p><small>PP/Direction CA-TS: {row['PP/Direction CA-TS'] if 'PP/Direction CA-TS' in row else 'N/A'}</small></p>
                    <hr>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 13. Tableau de bord des problématiques récurrentes
            st.markdown('<div class="sub-header">6. Tableau de bord des problématiques récurrentes</div>', unsafe_allow_html=True)
            
            theme_summary = create_dashboard(df_negatif_topics, topics)
            
            # Affichage du résumé
            st.dataframe(theme_summary[['Thème ID', 'Mots-clés', 'Nombre de verbatims', 'PP/Direction principale']])
            
            # Lien de téléchargement du tableau de bord
            st.markdown(get_table_download_link(theme_summary, 'dashboard_problematiques_recurrentes.xlsx', 'Télécharger le tableau de bord complet (Excel)'), unsafe_allow_html=True)
            
            # 14. Tableau détaillé des verbatims par thème
            detailed_df = df_negatif_topics[['ligne_source', 'dominant_topic', 'Verbatim', 'PP/Direction CA-TS', 'Humeur']].copy()
            detailed_df['Mots-clés du thème'] = detailed_df['dominant_topic'].apply(lambda x: topics[f'Thème {x}'])
            
            st.markdown('<div class="section-header">Tableau détaillé des verbatims par thème</div>', unsafe_allow_html=True)
            st.dataframe(detailed_df)
            
            # Lien de téléchargement du tableau détaillé
            st.markdown(get_table_download_link(detailed_df, 'verbatims_par_theme.xlsx', 'Télécharger le tableau détaillé (Excel)'), unsafe_allow_html=True)

# Page principale de l'application
def main():
    st.markdown('<h1 class="main-header">📊 Analyse des Enquêtes de Satisfaction</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    Cette application permet d'analyser les enquêtes de satisfaction, avec un focus particulier sur l'identification
    des problématiques récurrentes dans les retours négatifs. Utilis

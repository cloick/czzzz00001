import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import umap
import plotly.express as px
from tqdm import tqdm

# Fonction pour extraire les motifs potentiels d'une ligne de texte
def extraire_motifs(texte):
    if not isinstance(texte, str):
        return []
    
    # Plusieurs patterns pour capturer différents formats sans les connaître à l'avance
    patterns = [
        r'[A-Z]+\d+[A-Z]',  # Format comme AMT381W
        r'[A-Z0-9_-]+_JOB_[A-Z0-9_-]+',  # Format comme EQQ7INIT-1401_JOB_ZEFEQE
        r'JOBID_[A-Z0-9_]+',  # Format comme JOBID_JOB02567_010
        r'[a-zA-Z]+Service[a-zA-Z]+_[A-Z0-9]+_[A-Za-z]+',  # Format comme macServiceSMS_RTP10_MailGate
        r'[A-Z]{2,}_[A-Z0-9]+',  # Format comme SWMUPTEXE10
        r'[A-Za-z]+_Heap',  # Format comme DataNode Heap
        r'Service\s+[A-Z]+',  # Format comme Service HDFS
        r'[a-z]+\d+[a-z]+\d+',  # Format comme slmupds0631
        r'p\d{4}_[a-z]+_[a-z]+_p\d_s\d{4}',  # Format comme p0014_production_intranet_p1_s2648
        r'[a-zA-Z0-9-]+\d+[a-zA-Z0-9-]+',  # Format générique pour identifier avec numéros
        r'[A-Z]{2,}\d+',  # Format comme HR4You, AG00310
        r'[A-Za-z]+\d{4,}'  # Format comme p0014
    ]
    
    tous_motifs = []
    for pattern in patterns:
        matches = re.findall(pattern, texte)
        tous_motifs.extend(matches)
    
    return tous_motifs

# Fonction pour identifier les motifs communs dans un cluster
def identifier_motifs_cluster(textes, n_plus_communs=5):
    tous_motifs = []
    for texte in textes:
        motifs = extraire_motifs(texte)
        tous_motifs.extend(motifs)
    
    # Obtenir les motifs les plus communs
    motifs_communs = Counter(tous_motifs).most_common(n_plus_communs)
    return motifs_communs

# Fonction principale pour le clustering et la visualisation
def cluster_et_visualiser(df, colonne_description='breve description', n_clusters_a_montrer=10):
    # 1. Vectoriser le texte
    print("Vectorisation des données textuelles...")
    vectoriseur = TfidfVectorizer(
        min_df=2,  # Ignorer les termes qui apparaissent dans moins de 2 documents
        max_df=0.5,  # Ignorer les termes qui apparaissent dans plus de 50% des documents
        stop_words='french',  # Supprimer les mots vides français
        ngram_range=(1, 2)  # Considérer les unigrammes et bigrammes
    )
    
    # Gérer les valeurs NaN en les remplaçant par une chaîne vide
    textes = df[colonne_description].fillna("").astype(str)
    X = vectoriseur.fit_transform(textes)
    print(f"Vectorisation de {X.shape[0]} descriptions avec {X.shape[1]} caractéristiques")
    
    # 2. Clustering avec DBSCAN
    print("Clustering avec DBSCAN...")
    # Le paramètre eps peut nécessiter un ajustement en fonction de vos données
    clustering = DBSCAN(eps=0.5, min_samples=5, metric='cosine', n_jobs=-1)
    labels_clusters = clustering.fit_predict(X)
    
    # Compter le nombre de points dans chaque cluster
    n_clusters = len(set(labels_clusters)) - (1 if -1 in labels_clusters else 0)
    print(f"Trouvé {n_clusters} clusters (en excluant le bruit)")
    
    df_enrichi = df.copy()
    df_enrichi['cluster'] = labels_clusters
    
    # Compter les échantillons dans chaque cluster
    comptage_clusters = Counter(labels_clusters)
    print("Taille des clusters:", sorted([(k, v) for k, v in comptage_clusters.items() if k != -1], 
                                  key=lambda x: x[1], reverse=True))
    
    # 3. Réduction de dimensionnalité avec UMAP pour la visualisation
    print("Réduction des dimensions avec UMAP...")
    reducteur_2d = umap.UMAP(n_components=2, random_state=42, metric='cosine')
    reducteur_3d = umap.UMAP(n_components=3, random_state=42, metric='cosine')
    
    # Appliquer UMAP
    embedding_2d = reducteur_2d.fit_transform(X.toarray())
    embedding_3d = reducteur_3d.fit_transform(X.toarray())
    
    # Ajouter les embeddings au dataframe
    df_enrichi['umap_x'] = embedding_2d[:, 0]
    df_enrichi['umap_y'] = embedding_2d[:, 1]
    df_enrichi['umap_z'] = embedding_3d[:, 0]  # Premier composant pour z
    df_enrichi['umap_z2'] = embedding_3d[:, 1]  # Deuxième composant pour z
    df_enrichi['umap_z3'] = embedding_3d[:, 2]  # Troisième composant pour z
    
    # 4. Visualiser les clusters en 2D
    print("Création de la visualisation 2D...")
    
    # Ne tracer que les plus grands clusters et le bruit
    plus_grands_clusters = [k for k, v in comptage_clusters.most_common() if k != -1][:n_clusters_a_montrer]
    plus_grands_clusters = [-1] + plus_grands_clusters  # Inclure les points de bruit (-1)
    
    df_a_tracer = df_enrichi[df_enrichi['cluster'].isin(plus_grands_clusters)].copy()
    df_a_tracer['cluster'] = df_a_tracer['cluster'].astype(str)
    
    fig_2d = px.scatter(
        df_a_tracer, x='umap_x', y='umap_y', 
        color='cluster',
        title='Visualisation UMAP 2D des Clusters d\'Incidents',
        hover_data=[colonne_description]
    )
    
    # 5. Visualiser en 3D
    print("Création de la visualisation 3D...")
    fig_3d = px.scatter_3d(
        df_a_tracer, x='umap_x', y='umap_y', z='umap_z3',
        color='cluster',
        title='Visualisation UMAP 3D des Clusters d\'Incidents',
        hover_data=[colonne_description]
    )
    
    # 6. Analyser les motifs communs dans chaque cluster
    print("Identification des motifs communs dans chaque cluster...")
    motifs_clusters = {}
    
    for id_cluster in tqdm(plus_grands_clusters):
        if id_cluster == -1:
            continue  # Ignorer les points de bruit
            
        # Obtenir les textes pour ce cluster
        textes_cluster = df_enrichi[df_enrichi['cluster'] == id_cluster][colonne_description].tolist()
        
        # Identifier les motifs communs
        motifs_communs = identifier_motifs_cluster(textes_cluster)
        motifs_clusters[id_cluster] = motifs_communs
        
        print(f"\nCluster {id_cluster} ({len(textes_cluster)} incidents):")
        print("Motifs communs:")
        for motif, compte in motifs_communs:
            print(f"  - {motif}: {compte} occurrences ({compte/len(textes_cluster)*100:.1f}%)")
        
        # Montrer quelques exemples
        print("Exemples d'incidents:")
        for exemple in textes_cluster[:3]:
            if len(exemple) > 100:
                exemple = exemple[:97] + "..."
            print(f"  - {exemple}")
    
    # Générer un rapport de synthèse des formats identifiés par cluster
    print("\nGénération du rapport de synthèse...")
    rapport = "# Analyse des Patterns dans les Logs d'Incidents\n\n"
    
    for id_cluster, motifs in motifs_clusters.items():
        nb_incidents = len(df_enrichi[df_enrichi['cluster'] == id_cluster])
        rapport += f"## Cluster {id_cluster} ({nb_incidents} incidents)\n\n"
        rapport += "### Formats identifiés:\n"
        for motif, compte in motifs:
            rapport += f"- {motif}: {compte} occurrences ({compte/nb_incidents*100:.1f}%)\n"
        
        # Ajouter des exemples
        rapport += "\n### Exemples d'incidents:\n"
        exemples = df_enrichi[df_enrichi['cluster'] == id_cluster][colonne_description].head(3).tolist()
        for exemple in exemples:
            if len(exemple) > 100:
                exemple = exemple[:97] + "..."
            rapport += f"- {exemple}\n"
        rapport += "\n"
    
    # Enregistrer le rapport
    with open("rapport_formats_clusters.md", "w", encoding="utf-8") as f:
        f.write(rapport)
    
    print("Rapport de synthèse enregistré sous 'rapport_formats_clusters.md'")
    
    return fig_2d, fig_3d, motifs_clusters, df_enrichi

# Usage de la fonction
def analyser_incidents(df):
    """
    Fonction principale pour analyser les incidents dans un dataframe
    
    Args:
        df: DataFrame pandas contenant une colonne 'breve description'
    
    Returns:
        DataFrame enrichi avec les clusters et coordonnées UMAP
    """
    # Vérifier que la colonne existe
    if 'breve description' not in df.columns:
        raise ValueError("Le dataframe doit contenir une colonne 'breve description'")
        
    # Exécuter le clustering et la visualisation
    fig_2d, fig_3d, motifs_clusters, df_enrichi = cluster_et_visualiser(df)
    
    # Afficher les visualisations
    fig_2d.show()
    fig_3d.show()
    
    # Enregistrer les visualisations
    fig_2d.write_html("visualisation_clusters_2d.html")
    fig_3d.write_html("visualisation_clusters_3d.html")
    
    print("\nAnalyse terminée.")
    print("Visualisation 2D enregistrée sous 'visualisation_clusters_2d.html'")
    print("Visualisation 3D enregistrée sous 'visualisation_clusters_3d.html'")
    
    return df_enrichi

# Exemple d'utilisation:
# df_enrichi = analyser_incidents(df)
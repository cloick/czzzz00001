import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN
import umap
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import seaborn as sns
import os
from joblib import Parallel, delayed

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
def identifier_motifs_cluster(textes, n_plus_communs=10):
    tous_motifs = []
    for texte in textes:
        motifs = extraire_motifs(texte)
        tous_motifs.extend(motifs)
    
    # Obtenir les motifs les plus communs
    motifs_communs = Counter(tous_motifs).most_common(n_plus_communs)
    return motifs_communs

# Fonction pour calculer les métriques de qualité des clusters
def evaluer_clusters(X, labels):
    """
    Calcule plusieurs métriques pour évaluer la qualité des clusters
    """
    # Nombre de clusters (excluant le bruit -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Si un seul cluster ou aucun, les métriques ne sont pas applicables
    if n_clusters <= 1:
        return {
            "n_clusters": n_clusters,
            "noise_ratio": sum(1 for l in labels if l == -1) / len(labels),
            "silhouette": "N/A (clusters insuffisants)",
            "calinski_harabasz": "N/A (clusters insuffisants)"
        }
    
    # Calculer le score de silhouette (uniquement sur les points non bruités)
    non_noise_mask = labels != -1
    if sum(non_noise_mask) > 1:  # Au moins 2 points sont nécessaires
        try:
            silhouette = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
        except:
            silhouette = "Erreur de calcul"
    else:
        silhouette = "N/A (points insuffisants)"
    
    # Calculer le score de Calinski-Harabasz
    try:
        calinski_harabasz = calinski_harabasz_score(X, labels)
    except:
        calinski_harabasz = "Erreur de calcul"
    
    # Calculer le ratio de bruit (points non attribués à un cluster)
    noise_ratio = sum(1 for l in labels if l == -1) / len(labels)
    
    return {
        "n_clusters": n_clusters,
        "noise_ratio": noise_ratio,
        "silhouette": silhouette,
        "calinski_harabasz": calinski_harabasz
    }

# Fonction pour optimiser les hyperparamètres HDBSCAN
def optimiser_hyperparams_hdbscan(X, param_grid=None):
    """
    Trouve les meilleurs hyperparamètres pour HDBSCAN en utilisant une recherche par grille
    """
    if param_grid is None:
        param_grid = {
            'min_cluster_size': [5, 10, 20, 50],
            'min_samples': [5, 10, 15],
            'cluster_selection_epsilon': [0.0, 0.1, 0.5],
            'cluster_selection_method': ['eom', 'leaf']
        }
    
    grid = ParameterGrid(param_grid)
    resultats = []
    
    print(f"Évaluation de {len(grid)} combinaisons d'hyperparamètres...")
    
    # Fonction pour évaluer une configuration
    def evaluer_config(params):
        try:
            clustering = HDBSCAN(metric='cosine', **params)
            labels = clustering.fit_predict(X)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # Accepter même un seul cluster pour éviter de n'avoir aucun résultat
            if n_clusters == 0:
                print(f"Paramètres {params}: 0 cluster détecté (tous les points classés comme bruit)")
                return {
                    **params,
                    'n_clusters': 0,
                    'noise_ratio': 1.0,
                    'silhouette': 0,
                    'calinski_harabasz': 0
                }
                
            metrics = evaluer_clusters(X, labels)
            print(f"Paramètres {params}: {n_clusters} clusters, silhouette={metrics['silhouette']}")
            return {**params, **metrics}
        except Exception as e:
            print(f"Erreur avec params {params}: {str(e)}")
            return None
    
    # Exécuter en parallèle
    resultats_temp = Parallel(n_jobs=-1)(delayed(evaluer_config)(params) for params in grid)
    # Filtrer les résultats None
    resultats = [r for r in resultats_temp if r is not None]
    
    # Trier par score de silhouette (si disponible)
    resultats_valides = [r for r in resultats if isinstance(r['silhouette'], (int, float))]
    
    # Gérer le cas où aucun résultat n'est valide
    if not resultats and not resultats_valides:
        print("ATTENTION: Aucune combinaison de paramètres n'a donné de résultat valide.")
        print("Utilisation des paramètres par défaut.")
        return {
            'min_cluster_size': 10,
            'min_samples': 5,
            'cluster_selection_epsilon': 0.2,
            'cluster_selection_method': 'eom',
            'n_clusters': 10,
            'noise_ratio': 0.3,
            'silhouette': 'N/A',
            'calinski_harabasz': 'N/A'
        }, []
        
    if resultats_valides:
        resultats_valides.sort(key=lambda x: (-x['silhouette'], x['n_clusters']))
        meilleurs_params = resultats_valides[0]
    else:
        # Trier par nombre de clusters si silhouette n'est pas disponible
        resultats.sort(key=lambda x: abs(x['n_clusters'] - 20))  # ~20 clusters est souvent un bon compromis
        meilleurs_params = resultats[0]
    
    return meilleurs_params, resultats

# Fonction principale pour le clustering et la visualisation
def cluster_et_visualiser(df, colonne_description='breve description', max_clusters=30, n_exemples=10):
    """
    Réalise le clustering des incidents et générer les visualisations et rapports
    
    Args:
        df: DataFrame contenant les incidents
        colonne_description: Nom de la colonne contenant les descriptions
        max_clusters: Nombre maximum de clusters à analyser en détail
        n_exemples: Nombre d'exemples à inclure par cluster dans le rapport
    """
    # 1. Vectoriser le texte
    print("Vectorisation des données textuelles...")
    
    # Combiner des stop words français et anglais
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stop_words_fr = ['le', 'la', 'les', 'du', 'de', 'des', 'un', 'une', 'et', 'est', 'il', 'elle', 
                     'en', 'sur', 'qui', 'que', 'pour', 'dans', 'ce', 'cette', 'ces', 'avec', 'au', 'aux',
                     'ou', 'où', 'par', 'si', 'son', 'sa', 'ses', 'notre', 'nos', 'votre', 'vos',
                     'leur', 'leurs', 'tout', 'tous', 'toute', 'toutes', 'plus', 'moins', 'très']
    
    stop_words = list(ENGLISH_STOP_WORDS) + stop_words_fr + ['re', 've', 'll', 't', 's', 'm', 'd']
    
    vectoriseur = TfidfVectorizer(
        min_df=2,  # Ignorer les termes qui apparaissent dans moins de 2 documents
        max_df=0.7,  # Ignorer les termes qui apparaissent dans plus de 70% des documents
        stop_words=stop_words,
        ngram_range=(1, 2)  # Considérer les unigrammes et bigrammes
    )
    
    # Gérer les valeurs NaN en les remplaçant par une chaîne vide
    textes = df[colonne_description].fillna("").astype(str)
    X = vectoriseur.fit_transform(textes)
    print(f"Vectorisation de {X.shape[0]} descriptions avec {X.shape[1]} caractéristiques")
    
    # 2. Optimiser les hyperparamètres HDBSCAN
    print("Optimisation des hyperparamètres de HDBSCAN...")
    param_grid = {
        'min_cluster_size': [3, 5, 10, 15, 20, 30, 50],
        'min_samples': [1, 3, 5, 10, 20],
        'cluster_selection_epsilon': [0.0, 0.1, 0.2, 0.5, 1.0],
        'cluster_selection_method': ['eom', 'leaf']
    }
    meilleurs_params, tous_resultats = optimiser_hyperparams_hdbscan(X.toarray(), param_grid)
    
    print(f"Meilleurs hyperparamètres trouvés: {meilleurs_params}")
    print(f"Nombre estimé de clusters: {meilleurs_params['n_clusters']}")
    print(f"Score de silhouette: {meilleurs_params['silhouette']}")
    
    # Visualiser les résultats de la recherche d'hyperparamètres
    df_resultats = pd.DataFrame(tous_resultats)
    df_resultats_viz = df_resultats[df_resultats['n_clusters'] <= 50].copy()
    
    # Graphique pour le nombre de clusters vs silhouette score
    if isinstance(df_resultats_viz['silhouette'].iloc[0], (int, float)):
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=df_resultats_viz, x='n_clusters', y='silhouette')
        plt.title('Score de Silhouette vs Nombre de Clusters')
        plt.xlabel('Nombre de Clusters')
        plt.ylabel('Score de Silhouette')
        plt.savefig('hyperparams_silhouette.png')
        plt.close()
    
    # 3. Réaliser le clustering final avec les meilleurs paramètres
    print("Exécution du clustering final avec les paramètres optimaux...")
    best_params = {k: v for k, v in meilleurs_params.items() 
                  if k in ['min_cluster_size', 'min_samples', 'cluster_selection_epsilon', 'cluster_selection_method']}
    
    clustering = HDBSCAN(metric='cosine', **best_params)
    labels_clusters = clustering.fit_predict(X)
    
    # Compter le nombre de points dans chaque cluster
    n_clusters = len(set(labels_clusters)) - (1 if -1 in labels_clusters else 0)
    print(f"Nombre final de clusters: {n_clusters}")
    
    df_enrichi = df.copy()
    df_enrichi['cluster'] = labels_clusters
    
    # Compter les échantillons dans chaque cluster
    comptage_clusters = Counter(labels_clusters)
    clusters_tailles = sorted([(k, v) for k, v in comptage_clusters.items() if k != -1], 
                             key=lambda x: x[1], reverse=True)
    
    print(f"Taille des 10 plus grands clusters: {clusters_tailles[:10]}")
    
    # 4. Réduction de dimensionnalité avec UMAP pour la visualisation
    print("Réduction des dimensions avec UMAP...")
    reducteur_2d = umap.UMAP(n_components=2, random_state=42, metric='cosine')
    reducteur_3d = umap.UMAP(n_components=3, random_state=42, metric='cosine')
    
    # Appliquer UMAP
    embedding_2d = reducteur_2d.fit_transform(X.toarray())
    embedding_3d = reducteur_3d.fit_transform(X.toarray())
    
    # Ajouter les embeddings au dataframe
    df_enrichi['umap_x'] = embedding_2d[:, 0]
    df_enrichi['umap_y'] = embedding_2d[:, 1]
    df_enrichi['umap_z'] = embedding_3d[:, 0]
    df_enrichi['umap_z2'] = embedding_3d[:, 1]
    df_enrichi['umap_z3'] = embedding_3d[:, 2]
    
    # 5. Visualiser les clusters en 2D (tous les clusters)
    print("Création de la visualisation 2D complète...")
    df_enrichi['cluster_str'] = df_enrichi['cluster'].astype(str)
    
    # Vérifier si le nombre de clusters est gérable pour la visualisation
    fig_2d_all = px.scatter(
        df_enrichi, x='umap_x', y='umap_y', 
        color='cluster_str',
        title='Visualisation UMAP 2D de Tous les Clusters d\'Incidents',
        hover_data=[colonne_description]
    )
    fig_2d_all.write_html("visualisation_tous_clusters_2d.html")

    # 6. Visualiser les N plus grands clusters pour plus de clarté
    print(f"Création de la visualisation 2D des {max_clusters} plus grands clusters...")
    
    # Ne tracer que les plus grands clusters et le bruit
    plus_grands_clusters = [k for k, v in comptage_clusters.most_common() if k != -1][:max_clusters]
    plus_grands_clusters = [-1] + plus_grands_clusters  # Inclure les points de bruit (-1)
    
    df_top_clusters = df_enrichi[df_enrichi['cluster'].isin(plus_grands_clusters)].copy()
    
    fig_2d_top = px.scatter(
        df_top_clusters, x='umap_x', y='umap_y', 
        color='cluster_str',
        title=f'Visualisation UMAP 2D des {max_clusters} Plus Grands Clusters',
        hover_data=[colonne_description]
    )
    fig_2d_top.write_html("visualisation_top_clusters_2d.html")
    
    # 7. Visualiser en 3D
    print("Création de la visualisation 3D...")
    fig_3d = px.scatter_3d(
        df_top_clusters, x='umap_x', y='umap_y', z='umap_z3',
        color='cluster_str',
        title=f'Visualisation UMAP 3D des {max_clusters} Plus Grands Clusters',
        hover_data=[colonne_description]
    )
    fig_3d.write_html("visualisation_clusters_3d.html")
    
    # 8. Analyser les motifs communs dans chaque cluster
    print("Analyse des motifs communs dans chaque cluster...")
    
    # Créer un dossier pour les résultats détaillés par cluster
    os.makedirs("clusters_details", exist_ok=True)
    
    # Générer un rapport complet avec tous les clusters
    rapport = "# Analyse des Patterns dans les Logs d'Incidents\n\n"
    rapport += f"## Résumé\n\n"
    rapport += f"- Nombre total d'incidents analysés: {len(df)}\n"
    rapport += f"- Nombre de clusters identifiés: {n_clusters}\n"
    rapport += f"- Score de silhouette: {meilleurs_params['silhouette']}\n"
    rapport += f"- Hyperparamètres HDBSCAN utilisés: {best_params}\n\n"
    
    rapport += f"## Clusters par taille\n\n"
    rapport += "| ID Cluster | Nombre d'incidents | % du total |\n"
    rapport += "|------------|-------------------|------------|\n"
    for cluster_id, count in clusters_tailles:
        percentage = count / len(df) * 100
        rapport += f"| {cluster_id} | {count} | {percentage:.2f}% |\n"
    rapport += f"| -1 (bruit) | {comptage_clusters.get(-1, 0)} | {comptage_clusters.get(-1, 0)/len(df)*100:.2f}% |\n\n"
    
    # Analyser tous les clusters (sauf le bruit)
    tous_motifs_clusters = {}
    
    for id_cluster, count in tqdm(clusters_tailles):
        if id_cluster == -1:
            continue  # Ignorer les points de bruit
            
        # Obtenir les textes pour ce cluster
        textes_cluster = df_enrichi[df_enrichi['cluster'] == id_cluster][colonne_description].tolist()
        
        # Identifier les motifs communs
        motifs_communs = identifier_motifs_cluster(textes_cluster, n_plus_communs=20)
        tous_motifs_clusters[id_cluster] = motifs_communs
        
        # Générer un rapport spécifique pour ce cluster
        rapport_cluster = f"# Cluster {id_cluster}\n\n"
        rapport_cluster += f"## Statistiques\n\n"
        rapport_cluster += f"- Nombre d'incidents: {count} ({count/len(df)*100:.2f}% du total)\n\n"
        
        rapport_cluster += f"## Formats identifiés\n\n"
        rapport_cluster += "| Format | Occurrences | % des incidents du cluster |\n"
        rapport_cluster += "|--------|-------------|---------------------------|\n"
        for motif, compte in motifs_communs:
            rapport_cluster += f"| {motif} | {compte} | {compte/count*100:.2f}% |\n"
        
        rapport_cluster += f"\n## Exemples d'incidents\n\n"
        for i, exemple in enumerate(textes_cluster[:n_exemples]):
            rapport_cluster += f"### Exemple {i+1}\n\n```\n{exemple}\n```\n\n"
        
        # Enregistrer le rapport détaillé du cluster
        with open(f"clusters_details/cluster_{id_cluster}.md", "w", encoding="utf-8") as f:
            f.write(rapport_cluster)
    
    # Ajouter un résumé des motifs les plus importants pour chaque cluster dans le rapport principal
    rapport += "## Motifs identifiés par cluster\n\n"
    
    for id_cluster, count in clusters_tailles[:max_clusters]:  # Limiter aux clusters les plus importants
        motifs = tous_motifs_clusters.get(id_cluster, [])
        if not motifs:
            continue
            
        rapport += f"### Cluster {id_cluster} ({count} incidents)\n\n"
        rapport += "| Format | Occurrences | % des incidents |\n"
        rapport += "|--------|-------------|----------------|\n"
        for motif, compte in motifs[:10]:  # Top 10 motifs
            rapport += f"| {motif} | {compte} | {compte/count*100:.2f}% |\n"
        
        # Ajouter des exemples
        rapport += "\n#### Exemples:\n\n"
        exemples = df_enrichi[df_enrichi['cluster'] == id_cluster][colonne_description].head(n_exemples).tolist()
        for i, exemple in enumerate(exemples):
            rapport += f"{i+1}. ```{exemple}```\n\n"
        
        rapport += f"[Détails complets du cluster {id_cluster}](clusters_details/cluster_{id_cluster}.md)\n\n"
    
    # Section sur les points de bruit
    bruit_count = comptage_clusters.get(-1, 0)
    if bruit_count > 0:
        rapport += "## Points de bruit\n\n"
        rapport += f"- Nombre d'incidents non classifiés: {bruit_count} ({bruit_count/len(df)*100:.2f}% du total)\n\n"
        
        # Exemples de points de bruit
        rapport += "### Exemples de points non classifiés:\n\n"
        exemples_bruit = df_enrichi[df_enrichi['cluster'] == -1][colonne_description].sample(min(10, bruit_count)).tolist()
        for i, exemple in enumerate(exemples_bruit):
            rapport += f"{i+1}. ```{exemple}```\n\n"
    
    # Enregistrer le rapport principal
    with open("rapport_formats_clusters.md", "w", encoding="utf-8") as f:
        f.write(rapport)
    
    print("Rapport de synthèse enregistré sous 'rapport_formats_clusters.md'")
    print("Rapports détaillés par cluster enregistrés dans le dossier 'clusters_details/'")
    
    return fig_2d_all, fig_3d, tous_motifs_clusters, df_enrichi, meilleurs_params

# Usage de la fonction
def analyser_incidents(df, colonne_description='breve description'):
    """
    Fonction principale pour analyser les incidents dans un dataframe
    
    Args:
        df: DataFrame pandas contenant les descriptions d'incidents
        colonne_description: Nom de la colonne contenant les descriptions (défaut: 'breve description')
    
    Returns:
        DataFrame enrichi avec les clusters et coordonnées UMAP
    """
    # Vérifier que la colonne existe
    if colonne_description not in df.columns:
        raise ValueError(f"Le dataframe doit contenir une colonne '{colonne_description}'")
        
    # Exécuter le clustering et la visualisation
    fig_2d, fig_3d, motifs_clusters, df_enrichi, params = cluster_et_visualiser(
        df,
        colonne_description=colonne_description,
        max_clusters=30,  # Analyser en détail les 30 plus grands clusters 
        n_exemples=10     # Inclure 10 exemples par cluster
    )
    
    print("\nAnalyse terminée.")
    print("Visualisation 2D de tous les clusters: 'visualisation_tous_clusters_2d.html'")
    print("Visualisation 2D des principaux clusters: 'visualisation_top_clusters_2d.html'")
    print("Visualisation 3D des principaux clusters: 'visualisation_clusters_3d.html'")
    print("Rapport de synthèse: 'rapport_formats_clusters.md'")
    print("Rapports détaillés par cluster: dossier 'clusters_details/'")
    
    # Créer un fichier d'instructions
    with open("README.md", "w", encoding="utf-8") as f:
        f.write("""# Analyse des clusters d'incidents

## Fichiers générés

- `visualisation_tous_clusters_2d.html` - Visualisation interactive de tous les clusters en 2D
- `visualisation_top_clusters_2d.html` - Visualisation des principaux clusters en 2D
- `visualisation_clusters_3d.html` - Visualisation 3D interactive des principaux clusters
- `rapport_formats_clusters.md` - Rapport de synthèse avec les motifs identifiés
- `clusters_details/` - Dossier contenant une analyse détaillée pour chaque cluster
- `hyperparams_silhouette.png` - Graphique montrant la recherche d'hyperparamètres

## Comment explorer les résultats

1. Consultez d'abord le rapport de synthèse `rapport_formats_clusters.md` pour une vue d'ensemble
2. Explorez les visualisations interactives pour comprendre la disposition des clusters
3. Pour chaque cluster d'intérêt, consultez le rapport détaillé dans `clusters_details/cluster_X.md`

## Métriques de qualité

- Score de Silhouette: {silhouette}
- Nombre de clusters: {n_clusters}
- Hyperparamètres optimaux: {params}
""".format(
            silhouette=params['silhouette'],
            n_clusters=params['n_clusters'],
            params=str({k: v for k, v in params.items() 
                      if k in ['min_cluster_size', 'min_samples', 'cluster_selection_epsilon', 'cluster_selection_method']})
        ))
    
    return df_enrichi

# Exemple d'utilisation:
# df_enrichi = analyser_incidents(df)
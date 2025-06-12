# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

print("📊 VISUALISATION DES EMBEDDINGS ET CLUSTERS")
print("="*60)

# Read data
dataset = dataiku.Dataset("incident_with_clusters_intensive")
df = dataset.get_dataframe()

print(f"Dataset : {len(df)} tickets, {df['cluster'].nunique()-1} clusters")

# Préparer les données pour visualisation
df_viz = df.copy()

# Couleurs pour les clusters (limiter à 20 couleurs distinctes)
n_clusters = df['cluster'].nunique() - (1 if -1 in df['cluster'].values else 0)
print(f"Clusters à visualiser : {n_clusters}")

# 1. SCATTER PLOT 2D INTERACTIF (PLOTLY)
print("\n🎨 Création du scatter plot 2D interactif...")

# Préparer les données
df_viz['cluster_str'] = df_viz['cluster'].astype(str)
df_viz['cluster_str'] = df_viz['cluster_str'].replace('-1', 'Bruit')

# Préparer les notes de résolution pour affichage (tronquées)
if 'Notes de résolution' in df_viz.columns:
    df_viz['notes_courtes'] = df_viz['Notes de résolution'].fillna('').astype(str).apply(
        lambda x: x[:100] + '...' if len(x) > 100 else x
    )

# Informations pour hover (ordre optimisé pour lisibilité)
hover_data = []
if 'N° INC' in df_viz.columns:
    hover_data.append('N° INC')
if 'notes_courtes' in df_viz.columns:
    hover_data.append('notes_courtes')
if 'cause' in df_viz.columns:
    hover_data.append('cause')
if 'est_fiable' in df_viz.columns:
    hover_data.append('est_fiable')
if 'Groupe affecté' in df_viz.columns:
    hover_data.append('Groupe affecté')
if 'Service métier' in df_viz.columns:
    hover_data.append('Service métier')

# Créer le graphique 2D
fig_2d = px.scatter(
    df_viz,
    x='umap_x',
    y='umap_y',
    color='cluster_str',
    title=f'Visualisation UMAP 2D - {n_clusters} Clusters',
    hover_data=hover_data,
    width=1000,
    height=700,
    color_discrete_sequence=px.colors.qualitative.Set3
)

fig_2d.update_traces(marker=dict(size=4, opacity=0.7))
fig_2d.update_layout(
    xaxis_title='UMAP Dimension 1',
    yaxis_title='UMAP Dimension 2',
    legend_title='Cluster',
    showlegend=True if n_clusters <= 20 else False  # Masquer légende si trop de clusters
)

# Sauvegarder le graphique 2D
fig_2d.write_html("visualization_2d_clusters.html")
print("✅ Graphique 2D sauvé : visualization_2d_clusters.html")

# 2. SCATTER PLOT 3D INTERACTIF
print("\n🌐 Création du scatter plot 3D interactif...")

fig_3d = px.scatter_3d(
    df_viz,
    x='umap_x',
    y='umap_y', 
    z='umap_z',
    color='cluster_str',
    title=f'Visualisation UMAP 3D - {n_clusters} Clusters',
    hover_data=hover_data,
    width=1000,
    height=700,
    color_discrete_sequence=px.colors.qualitative.Set3
)

fig_3d.update_traces(marker=dict(size=3, opacity=0.6))
fig_3d.update_layout(
    scene=dict(
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2',
        zaxis_title='UMAP Dimension 3'
    ),
    legend_title='Cluster',
    showlegend=True if n_clusters <= 20 else False
)

# Sauvegarder le graphique 3D
fig_3d.write_html("visualization_3d_clusters.html")
print("✅ Graphique 3D sauvé : visualization_3d_clusters.html")

# 3. ANALYSE PAR CAUSES (si disponible)
if 'cause' in df_viz.columns and 'est_fiable' in df_viz.columns:
    print("\n🎯 Visualisation par causes...")
    
    # Seulement les tickets fiables
    df_fiables = df_viz[df_viz['est_fiable'] == True]
    
    if len(df_fiables) > 0:
        # Données de hover pour les causes
        hover_causes = ['cluster_str', 'notes_courtes', 'Groupe affecté', 'Service métier']
        hover_causes = [col for col in hover_causes if col in df_fiables.columns]
        
        fig_causes = px.scatter(
            df_fiables,
            x='umap_x',
            y='umap_y',
            color='cause',
            title=f'Visualisation par Causes - {df_fiables["cause"].nunique()} causes',
            hover_data=hover_causes,
            width=1000,
            height=700
        )
        
        fig_causes.update_traces(marker=dict(size=5, opacity=0.8))
        fig_causes.update_layout(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            legend_title='Cause'
        )
        
        fig_causes.write_html("visualization_causes.html")
        print("✅ Graphique par causes sauvé : visualization_causes.html")

# 4. HEATMAP DE DENSITÉ
print("\n🔥 Création de la heatmap de densité...")

plt.figure(figsize=(12, 8))
plt.hexbin(df_viz['umap_x'], df_viz['umap_y'], gridsize=50, cmap='Blues', alpha=0.7)
plt.colorbar(label='Densité de tickets')
plt.title('Densité des tickets dans l\'espace UMAP 2D')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.tight_layout()
plt.savefig('density_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Heatmap sauvée : density_heatmap.png")

# 5. ANALYSE DES TAILLES DE CLUSTERS
print("\n📊 Analyse des tailles de clusters...")

cluster_sizes = df_viz[df_viz['cluster'] != -1]['cluster'].value_counts().sort_values(ascending=False)

# Graphique en barres des tailles
fig_sizes = px.bar(
    x=cluster_sizes.index.astype(str),
    y=cluster_sizes.values,
    title='Distribution des tailles de clusters',
    labels={'x': 'Cluster ID', 'y': 'Nombre de tickets'},
    width=1000,
    height=500
)

fig_sizes.update_layout(xaxis_title='Cluster ID', yaxis_title='Nombre de tickets')
fig_sizes.write_html("cluster_sizes.html")
print("✅ Graphique des tailles sauvé : cluster_sizes.html")

# 6. MATRICE DE CONFUSION CLUSTERS vs CAUSES (si disponible)
if 'cause' in df_viz.columns and 'est_fiable' in df_viz.columns:
    print("\n📋 Matrice clusters vs causes...")
    
    df_fiables = df_viz[df_viz['est_fiable'] == True]
    
    if len(df_fiables) > 0:
        # Créer matrice de confusion
        confusion_matrix = pd.crosstab(df_fiables['cluster'], df_fiables['cause'])
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=True, yticklabels=True)
        plt.title('Matrice Clusters vs Causes (tickets fiables)')
        plt.xlabel('Cause')
        plt.ylabel('Cluster')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Matrice de confusion sauvée : confusion_matrix.png")

# 7. DASHBOARD INTERACTIF COMPLET
print("\n🎛️ Création du dashboard interactif...")

# Dashboard avec subplots
fig_dashboard = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Clusters 2D', 'Distribution des tailles', 'Densité par zone', 'Évolution temporelle'),
    specs=[[{"type": "scatter"}, {"type": "bar"}],
           [{"type": "scatter"}, {"type": "scatter"}]]
)

# Subplot 1: Scatter 2D
for cluster in df_viz['cluster'].unique():
    if cluster == -1:
        cluster_data = df_viz[df_viz['cluster'] == cluster]
        fig_dashboard.add_trace(
            go.Scatter(x=cluster_data['umap_x'], y=cluster_data['umap_y'],
                      mode='markers', name='Bruit',
                      marker=dict(color='gray', size=3, opacity=0.5)),
            row=1, col=1
        )
    else:
        cluster_data = df_viz[df_viz['cluster'] == cluster]
        if len(cluster_data) > 0:
            fig_dashboard.add_trace(
                go.Scatter(x=cluster_data['umap_x'], y=cluster_data['umap_y'],
                          mode='markers', name=f'Cluster {cluster}',
                          marker=dict(size=3, opacity=0.7)),
                row=1, col=1
            )

# Subplot 2: Tailles des clusters
fig_dashboard.add_trace(
    go.Bar(x=cluster_sizes.index.astype(str), y=cluster_sizes.values,
           name='Taille clusters', showlegend=False),
    row=1, col=2
)

# Subplot 3: Analyse de densité par quadrants
x_median = df_viz['umap_x'].median()
y_median = df_viz['umap_y'].median()

quadrants = {
    'Q1 (↗)': df_viz[(df_viz['umap_x'] > x_median) & (df_viz['umap_y'] > y_median)],
    'Q2 (↖)': df_viz[(df_viz['umap_x'] <= x_median) & (df_viz['umap_y'] > y_median)],
    'Q3 (↙)': df_viz[(df_viz['umap_x'] <= x_median) & (df_viz['umap_y'] <= y_median)],
    'Q4 (↘)': df_viz[(df_viz['umap_x'] > x_median) & (df_viz['umap_y'] <= y_median)]
}

quad_counts = [len(data) for data in quadrants.values()]
fig_dashboard.add_trace(
    go.Bar(x=list(quadrants.keys()), y=quad_counts,
           name='Densité quadrants', showlegend=False),
    row=2, col=1
)

# Subplot 4: Top 10 clusters
top_clusters = cluster_sizes.head(10)
fig_dashboard.add_trace(
    go.Scatter(x=list(range(len(top_clusters))), y=top_clusters.values,
               mode='lines+markers', name='Top 10 clusters',
               showlegend=False),
    row=2, col=2
)

fig_dashboard.update_layout(
    height=800,
    title_text="Dashboard Clustering METIS",
    showlegend=False  # Trop de clusters pour la légende
)

fig_dashboard.write_html("dashboard_clustering.html")
print("✅ Dashboard complet sauvé : dashboard_clustering.html")

# 8. STATISTIQUES DE VISUALISATION
print("\n📈 Statistiques de visualisation :")
print(f"   📊 Étendue X: {df_viz['umap_x'].min():.2f} à {df_viz['umap_x'].max():.2f}")
print(f"   📊 Étendue Y: {df_viz['umap_y'].min():.2f} à {df_viz['umap_y'].max():.2f}")
print(f"   📊 Étendue Z: {df_viz['umap_z'].min():.2f} à {df_viz['umap_z'].max():.2f}")

# Analyse de séparation
print(f"   🎯 Clusters les mieux séparés :")
for cluster in cluster_sizes.head(5).index:
    cluster_data = df_viz[df_viz['cluster'] == cluster]
    center_x = cluster_data['umap_x'].mean()
    center_y = cluster_data['umap_y'].mean()
    spread = np.sqrt(cluster_data['umap_x'].var() + cluster_data['umap_y'].var())
    print(f"      Cluster {cluster}: centre ({center_x:.2f}, {center_y:.2f}), dispersion {spread:.2f}")

# Créer dataset pour Charts Dataiku
print(f"\n💾 Préparation pour Charts Dataiku...")

# Dataset optimisé pour les charts natifs
df_charts = df_viz[['umap_x', 'umap_y', 'umap_z', 'cluster', 'cluster_str']].copy()

# Ajouter informations métier si disponibles
if 'Groupe affecté' in df_viz.columns:
    df_charts['groupe'] = df_viz['Groupe affecté']
if 'Service métier' in df_viz.columns:
    df_charts['service'] = df_viz['Service métier']
if 'cause' in df_viz.columns:
    df_charts['cause'] = df_viz['cause']
if 'est_fiable' in df_viz.columns:
    df_charts['fiable'] = df_viz['est_fiable']
if 'N° INC' in df_viz.columns:
    df_charts['numero_inc'] = df_viz['N° INC']
if 'notes_courtes' in df_viz.columns:
    df_charts['notes_resolution'] = df_viz['notes_courtes']

# Version complète des notes pour analyse approfondie
if 'Notes de résolution' in df_viz.columns:
    df_charts['notes_completes'] = df_viz['Notes de résolution']

# Sauvegarder pour Charts
output_charts = dataiku.Dataset("embeddings_visualization")
output_charts.write_with_schema(df_charts)

print(f"✅ Dataset pour charts Dataiku créé : embeddings_visualization")
print(f"   📝 Colonnes disponibles pour hover :")
print(f"      • notes_resolution (100 premiers caractères)")
print(f"      • notes_completes (texte intégral)")
print(f"      • numero_inc, cause, groupe, service, fiable")

print(f"\n💡 Utilisation dans Charts Dataiku :")
print(f"   1. Allez sur dataset 'embeddings_visualization'")
print(f"   2. Cliquez 'Charts' → 'Scatter plot 3D'")
print(f"   3. X=umap_x, Y=umap_y, Z=umap_z, Color=cluster_str")
print(f"   4. Dans 'Tooltips', ajoutez : notes_resolution, cause, numero_inc")

print(f"\n" + "="*60)
print(f"🎉 VISUALISATIONS CRÉÉES AVEC SUCCÈS !")
print(f"📁 Fichiers générés :")
print(f"   🌐 visualization_2d_clusters.html - Scatter 2D (avec notes)")
print(f"   🌐 visualization_3d_clusters.html - Scatter 3D (avec notes)")
print(f"   🌐 visualization_causes.html - Par causes (avec notes)")  
print(f"   🌐 dashboard_clustering.html - Dashboard complet")
print(f"   📊 density_heatmap.png - Heatmap de densité")
print(f"   📊 confusion_matrix.png - Matrice clusters/causes")
print(f"   📊 cluster_sizes.html - Distribution des tailles")
print(f"   💾 Dataset: embeddings_visualization (avec notes complètes)")
print(f"\n🎯 Hover enrichi : N° INC + Notes (100 car.) + Cause + Groupe + Service")
print("="*60)

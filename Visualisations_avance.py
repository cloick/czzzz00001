# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

print("📊 VISUALISATION AVEC SAUVEGARDE DANS MANAGED FOLDER")
print("="*65)

# Read data
dataset = dataiku.Dataset("incident_with_clusters_intensive")
df = dataset.get_dataframe()

print(f"Dataset : {len(df)} tickets, {df['cluster'].nunique()-1} clusters")

# ACCÉDER AU MANAGED FOLDER POUR SAUVEGARDER
try:
    folder = dataiku.Folder("visualizations")  # Le folder doit exister
    folder_path = folder.get_path()
    print(f"✅ Managed folder trouvé : {folder_path}")
except:
    print("❌ Managed folder 'visualizations' non trouvé")
    print("🔧 Création nécessaire : Administration → Folders → + New Folder")
    # Continuer sans folder
    folder = None

# Préparer les données pour visualisation
df_viz = df.copy()
n_clusters = df['cluster'].nunique() - (1 if -1 in df['cluster'].values else 0)

# Préparer les notes de résolution pour affichage (tronquées)
if 'Notes de résolution' in df_viz.columns:
    df_viz['notes_courtes'] = df_viz['Notes de résolution'].fillna('').astype(str).apply(
        lambda x: x[:100] + '...' if len(x) > 100 else x
    )

df_viz['cluster_str'] = df_viz['cluster'].astype(str)
df_viz['cluster_str'] = df_viz['cluster_str'].replace('-1', 'Bruit')

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

print(f"📊 Données hover : {', '.join(hover_data)}")

# 1. SCATTER PLOT 2D INTERACTIF
print("\n🎨 Création du scatter plot 2D...")

fig_2d = px.scatter(
    df_viz,
    x='umap_x',
    y='umap_y',
    color='cluster_str',
    title=f'Clustering METIS - Vue 2D - {n_clusters} Clusters',
    hover_data=hover_data,
    width=1200,
    height=800,
    color_discrete_sequence=px.colors.qualitative.Set3
)

fig_2d.update_traces(marker=dict(size=5, opacity=0.7))
fig_2d.update_layout(
    xaxis_title='UMAP Dimension 1',
    yaxis_title='UMAP Dimension 2',
    legend_title='Cluster',
    showlegend=True if n_clusters <= 25 else False,
    title_font_size=16
)

# Sauvegarder dans le folder ou localement
if folder:
    html_path_2d = f"{folder_path}/clustering_2d_interactive.html"
    fig_2d.write_html(html_path_2d)
    print(f"✅ Graphique 2D sauvé dans folder : clustering_2d_interactive.html")
else:
    fig_2d.write_html("clustering_2d_interactive.html")
    print("✅ Graphique 2D sauvé localement : clustering_2d_interactive.html")

# 2. SCATTER PLOT 3D INTERACTIF  
print("\n🌐 Création du scatter plot 3D...")

fig_3d = px.scatter_3d(
    df_viz,
    x='umap_x',
    y='umap_y', 
    z='umap_z',
    color='cluster_str',
    title=f'Clustering METIS - Vue 3D - {n_clusters} Clusters',
    hover_data=hover_data,
    width=1200,
    height=800,
    color_discrete_sequence=px.colors.qualitative.Set3
)

fig_3d.update_traces(marker=dict(size=4, opacity=0.6))
fig_3d.update_layout(
    scene=dict(
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2',
        zaxis_title='UMAP Dimension 3',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    ),
    legend_title='Cluster',
    showlegend=True if n_clusters <= 25 else False,
    title_font_size=16
)

if folder:
    html_path_3d = f"{folder_path}/clustering_3d_interactive.html"
    fig_3d.write_html(html_path_3d)
    print(f"✅ Graphique 3D sauvé dans folder : clustering_3d_interactive.html")
else:
    fig_3d.write_html("clustering_3d_interactive.html")
    print("✅ Graphique 3D sauvé localement : clustering_3d_interactive.html")

# 3. VISUALISATION PAR CAUSES
if 'cause' in df_viz.columns and 'est_fiable' in df_viz.columns:
    print("\n🎯 Visualisation par causes...")
    
    df_fiables = df_viz[df_viz['est_fiable'] == True]
    
    if len(df_fiables) > 0:
        hover_causes = ['cluster_str', 'notes_courtes', 'Groupe affecté', 'Service métier']
        hover_causes = [col for col in hover_causes if col in df_fiables.columns]
        
        fig_causes = px.scatter(
            df_fiables,
            x='umap_x',
            y='umap_y',
            color='cause',
            title=f'Clustering par Causes - {df_fiables["cause"].nunique()} causes identifiées',
            hover_data=hover_causes,
            width=1200,
            height=800
        )
        
        fig_causes.update_traces(marker=dict(size=6, opacity=0.8))
        fig_causes.update_layout(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            legend_title='Cause',
            title_font_size=16
        )
        
        if folder:
            html_path_causes = f"{folder_path}/clustering_by_causes.html"
            fig_causes.write_html(html_path_causes)
            print(f"✅ Graphique causes sauvé dans folder : clustering_by_causes.html")
        else:
            fig_causes.write_html("clustering_by_causes.html")
            print("✅ Graphique causes sauvé localement : clustering_by_causes.html")

# 4. DASHBOARD SYNTHÈSE
print("\n🎛️ Création du dashboard de synthèse...")

# Statistiques pour le dashboard
cluster_sizes = df_viz[df_viz['cluster'] != -1]['cluster'].value_counts().sort_values(ascending=False)

fig_dashboard = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        f'Distribution Spatiale ({n_clusters} clusters)', 
        f'Tailles des Clusters (Top 15)', 
        f'Répartition par Quadrants',
        f'Évolution des Tailles'
    ),
    specs=[[{"type": "scatter"}, {"type": "bar"}],
           [{"type": "bar"}, {"type": "scatter"}]]
)

# Graphique 1: Vue d'ensemble spatiale (échantillon)
sample_size = min(5000, len(df_viz))  # Limiter pour performance
df_sample = df_viz.sample(sample_size, random_state=42)

for cluster in df_sample['cluster'].unique():
    if cluster == -1:
        cluster_data = df_sample[df_sample['cluster'] == cluster]
        if len(cluster_data) > 0:
            fig_dashboard.add_trace(
                go.Scatter(x=cluster_data['umap_x'], y=cluster_data['umap_y'],
                          mode='markers', name='Bruit',
                          marker=dict(color='gray', size=2, opacity=0.4)),
                row=1, col=1
            )
    else:
        cluster_data = df_sample[df_sample['cluster'] == cluster]
        if len(cluster_data) > 0:
            fig_dashboard.add_trace(
                go.Scatter(x=cluster_data['umap_x'], y=cluster_data['umap_y'],
                          mode='markers', name=f'C{cluster}',
                          marker=dict(size=2, opacity=0.6)),
                row=1, col=1
            )

# Graphique 2: Top 15 clusters par taille
top_15 = cluster_sizes.head(15)
fig_dashboard.add_trace(
    go.Bar(x=[f'C{i}' for i in top_15.index], y=top_15.values,
           name='Tickets', showlegend=False,
           marker_color='lightblue'),
    row=1, col=2
)

# Graphique 3: Répartition par quadrants
x_median = df_viz['umap_x'].median()
y_median = df_viz['umap_y'].median()

quadrants = {
    'Nord-Est': len(df_viz[(df_viz['umap_x'] > x_median) & (df_viz['umap_y'] > y_median)]),
    'Nord-Ouest': len(df_viz[(df_viz['umap_x'] <= x_median) & (df_viz['umap_y'] > y_median)]),
    'Sud-Ouest': len(df_viz[(df_viz['umap_x'] <= x_median) & (df_viz['umap_y'] <= y_median)]),
    'Sud-Est': len(df_viz[(df_viz['umap_x'] > x_median) & (df_viz['umap_y'] <= y_median)])
}

fig_dashboard.add_trace(
    go.Bar(x=list(quadrants.keys()), y=list(quadrants.values()),
           name='Quadrants', showlegend=False,
           marker_color='lightgreen'),
    row=2, col=1
)

# Graphique 4: Distribution des tailles (courbe)
sizes_sorted = sorted(cluster_sizes.values, reverse=True)
fig_dashboard.add_trace(
    go.Scatter(x=list(range(1, len(sizes_sorted)+1)), y=sizes_sorted,
               mode='lines+markers', name='Distribution',
               line=dict(color='orange'), showlegend=False),
    row=2, col=2
)

fig_dashboard.update_layout(
    height=900,
    title_text=f"Dashboard Clustering METIS - {len(df)} tickets, {n_clusters} clusters",
    title_font_size=18,
    showlegend=False
)

if folder:
    html_path_dashboard = f"{folder_path}/dashboard_metis_clustering.html"
    fig_dashboard.write_html(html_path_dashboard)
    print(f"✅ Dashboard sauvé dans folder : dashboard_metis_clustering.html")
else:
    fig_dashboard.write_html("dashboard_metis_clustering.html")
    print("✅ Dashboard sauvé localement : dashboard_metis_clustering.html")

# 5. GRAPHIQUES STATIQUES (PNG)
print("\n📊 Création des graphiques statiques...")

# Heatmap de densité
plt.figure(figsize=(12, 8))
plt.hexbin(df_viz['umap_x'], df_viz['umap_y'], gridsize=50, cmap='Blues', alpha=0.7)
plt.colorbar(label='Densité de tickets')
plt.title('Heatmap de Densité - Distribution Spatiale des Tickets')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.tight_layout()

if folder:
    png_path_density = f"{folder_path}/density_heatmap.png"
    plt.savefig(png_path_density, dpi=300, bbox_inches='tight')
    print(f"✅ Heatmap sauvée dans folder : density_heatmap.png")
else:
    plt.savefig("density_heatmap.png", dpi=300, bbox_inches='tight')
    print("✅ Heatmap sauvée localement : density_heatmap.png")
plt.close()

# Distribution des tailles (graphique en barres)
plt.figure(figsize=(15, 6))
top_20 = cluster_sizes.head(20)
plt.bar(range(len(top_20)), top_20.values, color='skyblue', alpha=0.7)
plt.title('Distribution des Tailles - Top 20 Clusters')
plt.xlabel('Clusters (classés par taille)')
plt.ylabel('Nombre de tickets')
plt.xticks(range(len(top_20)), [f'C{i}' for i in top_20.index], rotation=45)
plt.tight_layout()

if folder:
    png_path_sizes = f"{folder_path}/cluster_sizes_distribution.png"
    plt.savefig(png_path_sizes, dpi=300, bbox_inches='tight')
    print(f"✅ Distribution tailles sauvée dans folder : cluster_sizes_distribution.png")
else:
    plt.savefig("cluster_sizes_distribution.png", dpi=300, bbox_inches='tight')
    print("✅ Distribution tailles sauvée localement : cluster_sizes_distribution.png")
plt.close()

# RÉSUMÉ ET INSTRUCTIONS
print(f"\n" + "="*65)
print(f"🎉 VISUALISATIONS CRÉÉES AVEC SUCCÈS !")
print("="*65)

if folder:
    print(f"📁 Fichiers sauvés dans le Managed Folder 'visualizations' :")
    print(f"   🌐 clustering_2d_interactive.html")
    print(f"   🌐 clustering_3d_interactive.html") 
    print(f"   🌐 clustering_by_causes.html")
    print(f"   🌐 dashboard_metis_clustering.html")
    print(f"   📊 density_heatmap.png")
    print(f"   📊 cluster_sizes_distribution.png")
    print(f"\n📋 Pour télécharger :")
    print(f"   1. Allez dans Flow → Managed Folders")
    print(f"   2. Cliquez sur 'visualizations'")
    print(f"   3. Téléchargez les fichiers HTML/PNG")
else:
    print(f"⚠️  Fichiers sauvés localement sur le serveur")
    print(f"💡 Pour les récupérer, créez un Managed Folder 'visualizations'")

print(f"\n🎯 Visualisations disponibles :")
print(f"   • Vue 2D interactive avec hover détaillé")
print(f"   • Vue 3D interactive avec notes de résolution")
print(f"   • Analyse par causes (tickets fiables)")
print(f"   • Dashboard de synthèse multi-graphiques")
print(f"   • Heatmaps et analyses statistiques")

# Créer dataset pour Charts Dataiku
df_charts = df_viz[['umap_x', 'umap_y', 'umap_z', 'cluster', 'cluster_str']].copy()

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
if 'Notes de résolution' in df_viz.columns:
    df_charts['notes_completes'] = df_viz['Notes de résolution']

output_charts = dataiku.Dataset("embeddings_visualization_enhanced")
output_charts.write_with_schema(df_charts)

print(f"\n💾 Dataset créé : embeddings_visualization_enhanced")
print(f"   📊 Utilisable dans Charts Dataiku natifs")
print(f"   📝 Hover avec notes de résolution incluses")
print("="*65)

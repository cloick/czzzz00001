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
import base64
from io import BytesIO
warnings.filterwarnings("ignore")

print("📊 VISUALISATIONS INTÉGRÉES DANS DATASET")
print("="*50)

# Read data
dataset = dataiku.Dataset("incident_with_clusters_intensive")
df = dataset.get_dataframe()

print(f"Dataset : {len(df)} tickets, {df['cluster'].nunique()-1} clusters")

# Préparer les données pour visualisation
df_viz = df.copy()
n_clusters = df['cluster'].nunique() - (1 if -1 in df['cluster'].values else 0)

# Préparer les notes de résolution
if 'Notes de résolution' in df_viz.columns:
    df_viz['notes_courtes'] = df_viz['Notes de résolution'].fillna('').astype(str).apply(
        lambda x: x[:100] + '...' if len(x) > 100 else x
    )

df_viz['cluster_str'] = df_viz['cluster'].astype(str)
df_viz['cluster_str'] = df_viz['cluster_str'].replace('-1', 'Bruit')

# Informations pour hover
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

print(f"📊 Hover data : {', '.join(hover_data)}")

# 1. GRAPHIQUE 2D INTERACTIF (PLOTLY → HTML STRING)
print("\n🎨 Création du scatter plot 2D...")

fig_2d = px.scatter(
    df_viz,
    x='umap_x',
    y='umap_y',
    color='cluster_str',
    title=f'Clustering METIS - Vue 2D - {n_clusters} Clusters',
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
    showlegend=True if n_clusters <= 25 else False
)

# Convertir en HTML string
html_2d = fig_2d.to_html(include_plotlyjs='cdn')
print("✅ Graphique 2D généré")

# 2. GRAPHIQUE 3D INTERACTIF
print("\n🌐 Création du scatter plot 3D...")

fig_3d = px.scatter_3d(
    df_viz,
    x='umap_x',
    y='umap_y', 
    z='umap_z',
    color='cluster_str',
    title=f'Clustering METIS - Vue 3D - Notes de résolution au hover',
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
    showlegend=True if n_clusters <= 25 else False
)

html_3d = fig_3d.to_html(include_plotlyjs='cdn')
print("✅ Graphique 3D généré avec notes de résolution")

# 3. GRAPHIQUE PAR CAUSES
html_causes = ""
if 'cause' in df_viz.columns and 'est_fiable' in df_viz.columns:
    print("\n🎯 Création du graphique par causes...")
    
    df_fiables = df_viz[df_viz['est_fiable'] == True]
    
    if len(df_fiables) > 0:
        fig_causes = px.scatter(
            df_fiables,
            x='umap_x',
            y='umap_y',
            color='cause',
            title=f'Clustering par Causes - {df_fiables["cause"].nunique()} causes',
            hover_data=['cluster_str', 'notes_courtes', 'Groupe affecté'],
            width=1000,
            height=700
        )
        
        fig_causes.update_traces(marker=dict(size=5, opacity=0.8))
        html_causes = fig_causes.to_html(include_plotlyjs='cdn')
        print("✅ Graphique par causes généré")

# 4. STATISTIQUES ET MÉTRIQUES
print("\n📊 Calcul des statistiques...")

cluster_sizes = df_viz[df_viz['cluster'] != -1]['cluster'].value_counts().sort_values(ascending=False)

# Statistiques générales
stats = {
    'total_tickets': len(df),
    'n_clusters': n_clusters,
    'bruit_pct': (df['cluster'] == -1).sum() / len(df) * 100,
    'plus_gros_cluster': cluster_sizes.iloc[0] if len(cluster_sizes) > 0 else 0,
    'plus_petit_cluster': cluster_sizes.iloc[-1] if len(cluster_sizes) > 0 else 0,
    'taille_mediane': cluster_sizes.median() if len(cluster_sizes) > 0 else 0
}

print(f"📈 Statistiques calculées : {stats['n_clusters']} clusters, {stats['bruit_pct']:.1f}% bruit")

# 5. GRAPHIQUE DES TAILLES (MATPLOTLIB → BASE64)
print("\n📊 Création du graphique des tailles...")

plt.figure(figsize=(12, 6))
top_15 = cluster_sizes.head(15)
bars = plt.bar(range(len(top_15)), top_15.values, color='skyblue', alpha=0.7)
plt.title('Distribution des Tailles - Top 15 Clusters')
plt.xlabel('Clusters')
plt.ylabel('Nombre de tickets')
plt.xticks(range(len(top_15)), [f'C{i}' for i in top_15.index])

# Ajouter les valeurs sur les barres
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()

# Convertir en base64
buffer = BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)
image_base64 = base64.b64encode(buffer.getvalue()).decode()
plt.close()

print("✅ Graphique des tailles généré")

# 6. HEATMAP DE DENSITÉ
print("\n🔥 Création de la heatmap...")

plt.figure(figsize=(10, 7))
plt.hexbin(df_viz['umap_x'], df_viz['umap_y'], gridsize=40, cmap='Blues', alpha=0.8)
plt.colorbar(label='Densité')
plt.title('Heatmap de Densité - Distribution Spatiale')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.tight_layout()

# Convertir en base64
buffer2 = BytesIO()
plt.savefig(buffer2, format='png', dpi=150, bbox_inches='tight')
buffer2.seek(0)
heatmap_base64 = base64.b64encode(buffer2.getvalue()).decode()
plt.close()

print("✅ Heatmap générée")

# 7. CRÉER LE DATASET AVEC VISUALISATIONS INTÉGRÉES
print("\n💾 Création du dataset avec visualisations...")

# Dataset principal
df_charts = df_viz[['umap_x', 'umap_y', 'umap_z', 'cluster', 'cluster_str']].copy()

# Ajouter colonnes métier
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

# 8. DATASET SÉPARÉ AVEC LES HTMLS
visualization_data = pd.DataFrame({
    'visualization_type': [
        'scatter_2d', 
        'scatter_3d', 
        'causes_analysis',
        'cluster_sizes_chart',
        'density_heatmap',
        'statistics_summary'
    ],
    'title': [
        'Vue 2D Interactive',
        'Vue 3D avec Notes de Résolution',
        'Analyse par Causes',
        'Distribution des Tailles',
        'Heatmap de Densité',
        'Statistiques Générales'
    ],
    'html_content': [
        html_2d,
        html_3d,
        html_causes if html_causes else "Non disponible - Pas de tickets fiables",
        f"<img src='data:image/png;base64,{image_base64}' style='max-width:100%;'>",
        f"<img src='data:image/png;base64,{heatmap_base64}' style='max-width:100%;'>",
        f"""
        <div style='font-family: Arial, sans-serif; padding: 20px;'>
            <h2>📊 Statistiques du Clustering METIS</h2>
            <ul>
                <li><strong>Total tickets:</strong> {stats['total_tickets']:,}</li>
                <li><strong>Clusters détectés:</strong> {stats['n_clusters']}</li>
                <li><strong>Bruit:</strong> {stats['bruit_pct']:.1f}%</li>
                <li><strong>Plus gros cluster:</strong> {stats['plus_gros_cluster']} tickets</li>
                <li><strong>Plus petit cluster:</strong> {stats['plus_petit_cluster']} tickets</li>
                <li><strong>Taille médiane:</strong> {stats['taille_mediane']:.0f} tickets</li>
            </ul>
        </div>
        """
    ],
    'description': [
        'Visualisation 2D interactive avec hover détaillé',
        'Visualisation 3D avec notes de résolution complètes',
        'Distribution spatiale par causes métier',
        'Graphique en barres des tailles de clusters',
        'Carte de chaleur de la densité spatiale',
        'Métriques clés du clustering'
    ]
})

# Sauvegarder les datasets
output_charts = dataiku.Dataset("embeddings_visualization_final")
output_charts.write_with_schema(df_charts)

output_viz = dataiku.Dataset("interactive_visualizations")
output_viz.write_with_schema(visualization_data)

print("✅ Datasets créés avec succès !")

print(f"\n" + "="*50)
print(f"🎉 VISUALISATIONS INTÉGRÉES CRÉÉES !")
print("="*50)
print(f"📊 Datasets disponibles :")
print(f"   💾 embeddings_visualization_final")
print(f"      → Données pour Charts Dataiku natifs")
print(f"      → Hover avec notes de résolution")
print(f"   🌐 interactive_visualizations") 
print(f"      → 6 visualisations HTML intégrées")
print(f"      → Consultables directement dans Dataiku")

print(f"\n💡 Comment utiliser :")
print(f"   1. Dataset 'interactive_visualizations'")
print(f"   2. Cliquez sur une ligne pour voir la visualisation")
print(f"   3. Colonne 'html_content' = visualisation complète")
print(f"   4. Charts Dataiku natifs avec 'embeddings_visualization_final'")

print(f"\n🎯 Visualisations disponibles :")
for i, row in visualization_data.iterrows():
    print(f"   {i+1}. {row['title']} - {row['description']}")
print("="*50)

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
import os
warnings.filterwarnings("ignore")

print("📊 VISUALISATIONS AVEC MANAGED FOLDER - PERMISSIONS ADMIN")
print("="*65)

# Read data
dataset = dataiku.Dataset("incident_with_clusters_intensive")
df = dataset.get_dataframe()

print(f"Dataset : {len(df)} tickets, {df['cluster'].nunique()-1} clusters")

# ACCÉDER AU MANAGED FOLDER
try:
    # Utiliser le nom exact que vous avez créé - remplacez si différent
    folder_names = ["visualizations", "NbaSKuEW", "viz", "charts"]
    folder = None
    
    for name in folder_names:
        try:
            folder = dataiku.Folder(name)
            folder_path = folder.get_path()
            print(f"✅ Managed folder trouvé : '{name}' → {folder_path}")
            break
        except:
            continue
    
    if folder is None:
        print("❌ Aucun Managed folder trouvé")
        print("🔧 Noms testés :", folder_names)
        print("💡 Vérifiez le nom exact dans votre Flow")
        raise Exception("Managed folder non trouvé")
        
    # Tester les permissions d'écriture
    test_file = os.path.join(folder_path, "test_permissions.txt")
    try:
        with open(test_file, 'w') as f:
            f.write("Test permissions OK")
        os.remove(test_file)
        print("✅ Permissions d'écriture confirmées")
    except Exception as e:
        print(f"❌ Problème de permissions : {e}")
        raise
        
except Exception as e:
    print(f"❌ Erreur Managed folder : {e}")
    print("🔧 Assurez-vous que :")
    print("   1. Le folder existe dans le Flow")
    print("   2. Il est déclaré comme INPUT du recipe")
    print("   3. Les permissions sont correctes")
    raise

# Préparer les données pour visualisation
df_viz = df.copy()
n_clusters = df['cluster'].nunique() - (1 if -1 in df['cluster'].values else 0)

# Préparer les notes de résolution pour affichage (tronquées)
if 'Notes de résolution' in df_viz.columns:
    df_viz['notes_courtes'] = df_viz['Notes de résolution'].fillna('').astype(str).apply(
        lambda x: x[:150] + '...' if len(x) > 150 else x
    )

df_viz['cluster_str'] = df_viz['cluster'].astype(str)
df_viz['cluster_str'] = df_viz['cluster_str'].replace('-1', 'Bruit')

# Informations pour hover (ordre optimisé)
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

print(f"📊 Hover data configuré : {', '.join(hover_data)}")

# 1. SCATTER PLOT 2D INTERACTIF
print("\n🎨 Création du scatter plot 2D...")

fig_2d = px.scatter(
    df_viz,
    x='umap_x',
    y='umap_y',
    color='cluster_str',
    title=f'🎯 Clustering METIS - Vue 2D Interactive<br><sub>{n_clusters} clusters détectés - Hover pour détails</sub>',
    hover_data=hover_data,
    width=1400,
    height=900,
    color_discrete_sequence=px.colors.qualitative.Set3
)

fig_2d.update_traces(marker=dict(size=5, opacity=0.7, line=dict(width=0.5, color='white')))
fig_2d.update_layout(
    xaxis_title='UMAP Dimension 1',
    yaxis_title='UMAP Dimension 2',
    legend_title='Cluster',
    showlegend=True if n_clusters <= 30 else False,
    title_font_size=18,
    template='plotly_white',
    hovermode='closest'
)

# Sauvegarder le graphique 2D
html_path_2d = os.path.join(folder_path, "clustering_2d_interactive.html")
fig_2d.write_html(html_path_2d, include_plotlyjs='cdn')
print(f"✅ Graphique 2D sauvé : clustering_2d_interactive.html")

# 2. SCATTER PLOT 3D INTERACTIF AVEC NOTES
print("\n🌐 Création du scatter plot 3D avec notes de résolution...")

fig_3d = px.scatter_3d(
    df_viz,
    x='umap_x',
    y='umap_y', 
    z='umap_z',
    color='cluster_str',
    title=f'🌐 Clustering METIS - Vue 3D Interactive<br><sub>📝 Notes de résolution au hover - {n_clusters} clusters</sub>',
    hover_data=hover_data,
    width=1400,
    height=900,
    color_discrete_sequence=px.colors.qualitative.Set3
)

fig_3d.update_traces(marker=dict(size=4, opacity=0.6, line=dict(width=0.5, color='white')))
fig_3d.update_layout(
    scene=dict(
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2',
        zaxis_title='UMAP Dimension 3',
        camera=dict(eye=dict(x=1.3, y=1.3, z=1.3)),
        bgcolor='white'
    ),
    legend_title='Cluster',
    showlegend=True if n_clusters <= 30 else False,
    title_font_size=18,
    template='plotly_white'
)

# Sauvegarder le graphique 3D
html_path_3d = os.path.join(folder_path, "clustering_3d_interactive_avec_notes.html")
fig_3d.write_html(html_path_3d, include_plotlyjs='cdn')
print(f"✅ Graphique 3D sauvé : clustering_3d_interactive_avec_notes.html")

# 3. VISUALISATION PAR CAUSES (TICKETS FIABLES)
if 'cause' in df_viz.columns and 'est_fiable' in df_viz.columns:
    print("\n🎯 Visualisation par causes...")
    
    df_fiables = df_viz[df_viz['est_fiable'] == True]
    
    if len(df_fiables) > 0:
        hover_causes = ['cluster_str', 'notes_courtes', 'N° INC', 'Groupe affecté', 'Service métier']
        hover_causes = [col for col in hover_causes if col in df_fiables.columns]
        
        fig_causes = px.scatter(
            df_fiables,
            x='umap_x',
            y='umap_y',
            color='cause',
            title=f'📋 Analyse par Causes Métier<br><sub>{df_fiables["cause"].nunique()} causes identifiées - {len(df_fiables)} tickets fiables</sub>',
            hover_data=hover_causes,
            width=1400,
            height=900,
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        
        fig_causes.update_traces(marker=dict(size=6, opacity=0.8, line=dict(width=1, color='white')))
        fig_causes.update_layout(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            legend_title='Cause Métier',
            title_font_size=18,
            template='plotly_white'
        )
        
        html_path_causes = os.path.join(folder_path, "clustering_par_causes_metier.html")
        fig_causes.write_html(html_path_causes, include_plotlyjs='cdn')
        print(f"✅ Graphique causes sauvé : clustering_par_causes_metier.html")

# 4. DASHBOARD SYNTHÈSE MULTI-GRAPHIQUES
print("\n🎛️ Création du dashboard de synthèse...")

cluster_sizes = df_viz[df_viz['cluster'] != -1]['cluster'].value_counts().sort_values(ascending=False)

# Dashboard avec 4 sous-graphiques
fig_dashboard = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        f'📍 Distribution Spatiale ({n_clusters} clusters)', 
        f'📊 Tailles des Clusters (Top 20)', 
        f'🗺️ Répartition Géographique UMAP',
        f'📈 Courbe de Distribution'
    ),
    specs=[[{"type": "scatter"}, {"type": "bar"}],
           [{"type": "bar"}, {"type": "scatter"}]],
    vertical_spacing=0.12,
    horizontal_spacing=0.1
)

# Graphique 1: Vue d'ensemble spatiale (échantillon pour performance)
sample_size = min(3000, len(df_viz))
df_sample = df_viz.sample(sample_size, random_state=42)

colors = px.colors.qualitative.Set3
for i, cluster in enumerate(sorted(df_sample['cluster'].unique())):
    cluster_data = df_sample[df_sample['cluster'] == cluster]
    if len(cluster_data) > 0:
        color = 'gray' if cluster == -1 else colors[i % len(colors)]
        name = 'Bruit' if cluster == -1 else f'C{cluster}'
        
        fig_dashboard.add_trace(
            go.Scatter(
                x=cluster_data['umap_x'], 
                y=cluster_data['umap_y'],
                mode='markers', 
                name=name,
                marker=dict(color=color, size=3, opacity=0.6),
                showlegend=False
            ),
            row=1, col=1
        )

# Graphique 2: Top 20 clusters par taille
top_20 = cluster_sizes.head(20)
fig_dashboard.add_trace(
    go.Bar(
        x=[f'C{i}' for i in top_20.index], 
        y=top_20.values,
        name='Nb Tickets', 
        showlegend=False,
        marker_color='lightblue',
        text=top_20.values,
        textposition='outside'
    ),
    row=1, col=2
)

# Graphique 3: Répartition par quadrants
x_median = df_viz['umap_x'].median()
y_median = df_viz['umap_y'].median()

quadrants = {
    'Nord-Est (Q1)': len(df_viz[(df_viz['umap_x'] > x_median) & (df_viz['umap_y'] > y_median)]),
    'Nord-Ouest (Q2)': len(df_viz[(df_viz['umap_x'] <= x_median) & (df_viz['umap_y'] > y_median)]),
    'Sud-Ouest (Q3)': len(df_viz[(df_viz['umap_x'] <= x_median) & (df_viz['umap_y'] <= y_median)]),
    'Sud-Est (Q4)': len(df_viz[(df_viz['umap_x'] > x_median) & (df_viz['umap_y'] <= y_median)])
}

fig_dashboard.add_trace(
    go.Bar(
        x=list(quadrants.keys()), 
        y=list(quadrants.values()),
        name='Tickets par zone', 
        showlegend=False,
        marker_color='lightgreen',
        text=list(quadrants.values()),
        textposition='outside'
    ),
    row=2, col=1
)

# Graphique 4: Courbe de distribution des tailles
sizes_sorted = sorted(cluster_sizes.values, reverse=True)
fig_dashboard.add_trace(
    go.Scatter(
        x=list(range(1, len(sizes_sorted)+1)), 
        y=sizes_sorted,
        mode='lines+markers', 
        name='Distribution',
        line=dict(color='orange', width=3),
        marker=dict(size=4),
        showlegend=False
    ),
    row=2, col=2
)

# Mise en forme du dashboard
fig_dashboard.update_layout(
    height=1000,
    title_text=f"📈 Dashboard Clustering METIS - Analyse Complète<br><sub>{len(df):,} tickets • {n_clusters} clusters • {(df['cluster'] == -1).sum()} points de bruit</sub>",
    title_font_size=20,
    showlegend=False,
    template='plotly_white'
)

# Ajuster les axes
fig_dashboard.update_xaxes(title_text="UMAP X", row=1, col=1)
fig_dashboard.update_yaxes(title_text="UMAP Y", row=1, col=1)
fig_dashboard.update_xaxes(title_text="Clusters", row=1, col=2, tickangle=45)
fig_dashboard.update_yaxes(title_text="Nombre de tickets", row=1, col=2)
fig_dashboard.update_xaxes(title_text="Zones UMAP", row=2, col=1, tickangle=45)
fig_dashboard.update_yaxes(title_text="Nombre de tickets", row=2, col=1)
fig_dashboard.update_xaxes(title_text="Rang du cluster", row=2, col=2)
fig_dashboard.update_yaxes(title_text="Taille", row=2, col=2)

html_path_dashboard = os.path.join(folder_path, "dashboard_complet_metis.html")
fig_dashboard.write_html(html_path_dashboard, include_plotlyjs='cdn')
print(f"✅ Dashboard complet sauvé : dashboard_complet_metis.html")

# 5. GRAPHIQUES STATIQUES (PNG) POUR RAPPORTS
print("\n📊 Création des graphiques statiques...")

# Heatmap de densité améliorée
plt.figure(figsize=(14, 10))
plt.hexbin(df_viz['umap_x'], df_viz['umap_y'], gridsize=60, cmap='Blues', alpha=0.8)
cbar = plt.colorbar(label='Densité de tickets')
cbar.ax.tick_params(labelsize=12)
plt.title('🔥 Heatmap de Densité - Distribution Spatiale des Tickets METIS', fontsize=16, fontweight='bold')
plt.xlabel('UMAP Dimension 1', fontsize=14)
plt.ylabel('UMAP Dimension 2', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

png_path_density = os.path.join(folder_path, "heatmap_densite_metis.png")
plt.savefig(png_path_density, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✅ Heatmap sauvée : heatmap_densite_metis.png")

# Distribution des tailles avec statistiques
plt.figure(figsize=(18, 8))
top_25 = cluster_sizes.head(25)
bars = plt.bar(range(len(top_25)), top_25.values, color='skyblue', alpha=0.8, edgecolor='navy', linewidth=0.5)
plt.title('📊 Distribution des Tailles - Top 25 Clusters METIS', fontsize=16, fontweight='bold')
plt.xlabel('Clusters (classés par taille décroissante)', fontsize=14)
plt.ylabel('Nombre de tickets', fontsize=14)
plt.xticks(range(len(top_25)), [f'C{i}\n({v})' for i, v in zip(top_25.index, top_25.values)], rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

# Ajouter la moyenne
moyenne = top_25.mean()
plt.axhline(y=moyenne, color='red', linestyle='--', alpha=0.7, label=f'Moyenne: {moyenne:.0f} tickets')
plt.legend()

# Annotations des extremes
plt.annotate(f'Plus gros:\n{top_25.iloc[0]} tickets', 
             xy=(0, top_25.iloc[0]), xytext=(2, top_25.iloc[0] + 500),
             arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, ha='center')

plt.tight_layout()
png_path_sizes = os.path.join(folder_path, "distribution_tailles_clusters.png")
plt.savefig(png_path_sizes, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✅ Distribution tailles sauvée : distribution_tailles_clusters.png")

# 6. RAPPORT DE SYNTHÈSE HTML
print("\n📄 Génération du rapport de synthèse...")

# Statistiques détaillées
stats = {
    'total_tickets': len(df),
    'n_clusters': n_clusters,
    'n_bruit': (df['cluster'] == -1).sum(),
    'pct_bruit': (df['cluster'] == -1).sum() / len(df) * 100,
    'plus_gros': cluster_sizes.iloc[0] if len(cluster_sizes) > 0 else 0,
    'plus_petit': cluster_sizes.iloc[-1] if len(cluster_sizes) > 0 else 0,
    'mediane': cluster_sizes.median() if len(cluster_sizes) > 0 else 0,
    'moyenne': cluster_sizes.mean() if len(cluster_sizes) > 0 else 0
}

if 'est_fiable' in df.columns:
    stats['tickets_fiables'] = df['est_fiable'].sum()
    stats['pct_fiables'] = df['est_fiable'].sum() / len(df) * 100

rapport_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Rapport Clustering METIS</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }}
        .stat-number {{ font-size: 2em; font-weight: bold; color: #3498db; }}
        .stat-label {{ color: #7f8c8d; font-size: 0.9em; }}
        .visualization-links {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin: 20px 0; }}
        .viz-card {{ background: #fff; border: 2px solid #3498db; padding: 20px; border-radius: 8px; text-align: center; transition: transform 0.2s; }}
        .viz-card:hover {{ transform: translateY(-5px); box-shadow: 0 6px 12px rgba(0,0,0,0.1); }}
        .viz-link {{ text-decoration: none; color: #3498db; font-weight: bold; font-size: 1.1em; }}
        .viz-description {{ color: #7f8c8d; margin-top: 8px; }}
        .info {{ background: #e8f6ff; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Rapport Clustering METIS</h1>
        <p style="text-align: center; color: #7f8c8d; font-size: 1.1em;">Analyse des {stats['total_tickets']:,} tickets d'incidents</p>
        
        <h2>📊 Statistiques Générales</h2>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{stats['total_tickets']:,}</div>
                <div class="stat-label">Total Tickets</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['n_clusters']}</div>
                <div class="stat-label">Clusters Détectés</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['pct_bruit']:.1f}%</div>
                <div class="stat-label">Taux de Bruit</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['plus_gros']:,}</div>
                <div class="stat-label">Plus Gros Cluster</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['mediane']:.0f}</div>
                <div class="stat-label">Taille Médiane</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['moyenne']:.0f}</div>
                <div class="stat-label">Taille Moyenne</div>
            </div>
"""

if 'tickets_fiables' in stats:
    rapport_html += f"""
            <div class="stat-card">
                <div class="stat-number">{stats['tickets_fiables']:,}</div>
                <div class="stat-label">Tickets Fiables</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['pct_fiables']:.1f}%</div>
                <div class="stat-label">% Fiables</div>
            </div>
"""

rapport_html += f"""
        </div>
        
        <h2>🌐 Visualisations Interactives</h2>
        <div class="visualization-links">
            <div class="viz-card">
                <a href="clustering_2d_interactive.html" class="viz-link">📍 Vue 2D Interactive</a>
                <div class="viz-description">Exploration 2D avec hover détaillé</div>
            </div>
            <div class="viz-card">
                <a href="clustering_3d_interactive_avec_notes.html" class="viz-link">🌐 Vue 3D + Notes</a>
                <div class="viz-description">Visualisation 3D avec notes de résolution</div>
            </div>
            <div class="viz-card">
                <a href="clustering_par_causes_metier.html" class="viz-link">📋 Analyse par Causes</a>
                <div class="viz-description">Distribution par causes métier</div>
            </div>
            <div class="viz-card">
                <a href="dashboard_complet_metis.html" class="viz-link">📈 Dashboard Complet</a>
                <div class="viz-description">Vue d'ensemble multi-graphiques</div>
            </div>
        </div>
        
        <h2>📊 Analyses Statiques</h2>
        <div class="visualization-links">
            <div class="viz-card">
                <a href="heatmap_densite_metis.png" class="viz-link">🔥 Heatmap Densité</a>
                <div class="viz-description">Carte de chaleur spatiale</div>
            </div>
            <div class="viz-card">
                <a href="distribution_tailles_clusters.png" class="viz-link">📊 Distribution Tailles</a>
                <div class="viz-description">Graphique des tailles de clusters</div>
            </div>
        </div>
        
        <div class="info">
            <strong>💡 Mode d'emploi :</strong><br>
            • Cliquez sur les liens pour ouvrir les visualisations<br>
            • Les fichiers HTML sont interactifs (zoom, hover, filtres)<br>
            • Les fichiers PNG sont pour les rapports statiques<br>
            • Toutes les visualisations incluent les notes de résolution au hover
        </div>
        
        <p style="text-align: center; color: #7f8c8d; margin-top: 30px;">
            Généré le {pd.Timestamp.now().strftime('%d/%m/%Y à %H:%M')} • Projet METIS • Clustering avec {n_clusters} clusters
        </p>
    </div>
</body>
</html>
"""

rapport_path = os.path.join(folder_path, "rapport_clustering_metis.html")
with open(rapport_path, 'w', encoding='utf-8') as f:
    f.write(rapport_html)
print(f"✅ Rapport de synthèse sauvé : rapport_clustering_metis.html")

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

output_charts = dataiku.Dataset("embeddings_visualization_complete")
output_charts.write_with_schema(df_charts)

print(f"\n" + "="*65)
print(f"🎉 VISUALISATIONS CRÉÉES AVEC SUCCÈS DANS LE MANAGED FOLDER !")
print("="*65)
print(f"📁 Fichiers disponibles dans le folder :")
print(f"   🌐 clustering_2d_interactive.html")
print(f"   🌐 clustering_3d_interactive_avec_notes.html") 
print(f"   🌐 clustering_par_causes_metier.html")
print(f"   🌐 dashboard_complet_metis.html")
print(f"   📄 rapport_clustering_metis.html (INDEX)")
print(f"   📊 heatmap_densite_metis.png")
print(f"   📊 distribution_tailles_clusters.png")

print(f"\n📋 Pour télécharger et consulter :")
print(f"   1. Flow → Folders → Cliquez sur votre folder")
print(f"   2. Téléchargez 'rapport_clustering_metis.html' EN PREMIER")
print(f"   3. Téléchargez les autres fichiers dans le même dossier")
print(f"   4. Ouvrez 'rapport_clustering_metis.html' dans votre navigateur")

print(f"\n💾 Dataset Dataiku créé : embeddings_visualization_complete")
print(f"   📊 Pour Charts natifs avec hover personnalisé")

print(f"\n🎯 Features spéciales :")
print(f"   • Vue 3D avec notes de résolution complètes au hover")
print(f"   • Dashboard multi-graphiques interactif")
print(f"   • Rapport HTML avec navigation intégrée")
print(f"   • Graphiques haute résolution pour rapports")
print("="*65)

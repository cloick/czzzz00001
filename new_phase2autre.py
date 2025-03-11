def clustering_phase1_style(embeddings_enrichis):
    """Reproduces the clustering approach that was successful in Phase 1"""
    
    print("Dimensional reduction with UMAP...")
    reducer = UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=15,  # Changed from 30 to 15 as in Phase 1
        min_dist=0.1
    )
    embeddings_2d = reducer.fit_transform(embeddings_enrichis)
    
    print("Applying HDBSCAN with Phase 1 parameters...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=2,
        cluster_selection_epsilon=0.5,
        metric='euclidean'
    )
    clusters = clusterer.fit_predict(embeddings_2d)
    
    # Clustering statistics
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = (clusters == -1).sum()
    
    print(f"Number of detected clusters: {n_clusters}")
    print(f"Number of points classified as noise: {n_noise} ({n_noise/len(clusters):.2%})")
    
    # Visualization
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
    plt.title(f'Clustering: {n_clusters} clusters detected')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.tight_layout()
    plt.savefig("clustering_phase1_style.png", dpi=300)
    plt.show()
    
    # Quality measurement
    if n_clusters > 1:
        mask = clusters != -1
        if mask.sum() > n_clusters:
            silhouette = silhouette_score(embeddings_2d[mask], clusters[mask])
            print(f"Silhouette score: {silhouette:.4f}")
    
    return clusters, embeddings_2d, clusterer


---------------------------------------------------------------------------------


def enrichir_embeddings_simple(embeddings, df, cat_vars):
    """
    Simplified version of embedding enrichment, as in Phase 1
    """
    print(f"Enriching embeddings with {len(cat_vars)} categorical variables...")
    
    # Extract encoded categorical variables
    cat_features = np.array([df[f'{col}_encoded'] for col in cat_vars if f'{col}_encoded' in df.columns]).T
    
    # Normalize categorical features
    if cat_features.shape[0] > 0:
        scaler = StandardScaler()
        cat_features = scaler.fit_transform(cat_features)
    
    # Concatenate text embeddings with categorical features
    enriched_embeddings = np.hstack([embeddings, cat_features])
    
    print(f"Original embeddings dimensions: {embeddings.shape}")
    print(f"Enriched embeddings dimensions: {enriched_embeddings.shape}")
    
    return enriched_embeddings


-----------------------------------------------------------------------------------------------------------------

def executer_clustering_phase1_style(limit_samples=None):
    """
    Executes the entire process following the Phase 1 approach
    """
    # 1. Loading and preparing data
    print("\n==== 1. Loading and preparing data ====")
    df_metis = pd.read_csv(chemin_metis)
    df_metis = df_metis[['N° INC', 'Priorité', 'Service métier', 'Cat1', 'Cat2', 
                         'Groupe affecté', 'Notes de résolution', 'cause', 'souscause']]
    
    df_fiable = pd.read_csv(chemin_tickets_fiables)
    df_metis['est_fiable'] = df_metis['N° INC'].isin(df_fiable['N° INC'])
    
    # Sample selection while keeping all reliable tickets
    if limit_samples and limit_samples < len(df_metis):
        df_fiables = df_metis[df_metis['est_fiable']]
        df_non_fiables = df_metis[~df_metis['est_fiable']].sample(
            min(limit_samples - len(df_fiables), len(df_metis) - len(df_fiables)),
            random_state=42
        )
        df_sample = pd.concat([df_fiables, df_non_fiables])
        print(f"Sample limited to {len(df_sample)} tickets including {len(df_fiables)} reliable ones")
    else:
        df_sample = df_metis
    
    # 2. Data preparation
    print("\n==== 2. Data preparation ====")
    df_prep, cause_mapping, encoders = preparer_donnees(df_sample)
    
    # 3. Model loading and embedding generation
    print("\n==== 3. Model loading and embedding generation ====")
    tokenizer, model = charger_modele_local()
    
    textes = df_prep['notes_resolution_nettoyees'].tolist()
    embeddings = generer_embeddings(textes, tokenizer, model)
    
    # 4. Simple embedding enrichment with categorical variables
    print("\n==== 4. Embedding enrichment ====")
    cat_vars = ['Priorité', 'Service métier', 'Cat1', 'Cat2', 'Groupe affecté']
    
    embeddings_enrichis = enrichir_embeddings_simple(embeddings, df_prep, cat_vars)
    
    # 5. Phase 1 style clustering
    print("\n==== 5. Phase 1 style clustering ====")
    cluster_labels, embeddings_2d, clusterer = clustering_phase1_style(embeddings_enrichis)
    
    # 6. Cluster labeling
    print("\n==== 6. Cluster labeling ====")
    cluster_to_cause, cluster_to_cause_name = etiqueter_clusters(
        cluster_labels, df_prep, cause_mapping
    )
    
    # 7. Analysis of unidentified clusters
    print("\n==== 7. Analysis of unidentified clusters ====")
    analyser_clusters_inconnus(df_prep, cluster_labels, embeddings_2d, cluster_to_cause_name)
    
    # Add clusters to DataFrame
    df_prep['cluster'] = cluster_labels
    
    print("\nProcess completed successfully!")
    
    return {
        'df_prep': df_prep,
        'cluster_labels': cluster_labels,
        'embeddings_2d': embeddings_2d,
        'clusterer': clusterer,
        'cluster_to_cause_name': cluster_to_cause_name
    }
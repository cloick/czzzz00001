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
    
    ----------------------------------------
    -----------------------------------------
    ----------------------------------------
    
    def optimiser_hdbscan(embeddings_2d, cible_clusters=15, tolerance=5):
    """Recherche les meilleurs paramètres HDBSCAN pour obtenir environ 15 clusters"""
    
    print(f"Optimisation des paramètres HDBSCAN pour {cible_clusters} clusters (±{tolerance})...")
    
    meilleurs_params = None
    meilleur_score = -1
    meilleur_n_clusters = 0
    meilleur_ecart = float('inf')
    
    # Grille d'hyperparamètres
    grille_params = {
        'min_cluster_size': [10, 20, 30, 50, 100, 200],
        'min_samples': [5, 10, 15, 20, 30],
        'cluster_selection_epsilon': [0.0, 0.1, 0.5, 1.0]
    }
    
    resultats = []
    
    for min_cluster_size in grille_params['min_cluster_size']:
        for min_samples in grille_params['min_samples']:
            for epsilon in grille_params['cluster_selection_epsilon']:
                # Ne tester que si min_samples <= min_cluster_size
                if min_samples > min_cluster_size:
                    continue
                    
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_epsilon=epsilon,
                    metric='euclidean'
                )
                
                labels = clusterer.fit_predict(embeddings_2d)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                noise_ratio = (labels == -1).sum() / len(labels)
                
                # Calculer le score de silhouette si possible
                silhouette = None
                if n_clusters > 1 and n_clusters < len(embeddings_2d) - 1:
                    mask = labels != -1
                    if mask.sum() > n_clusters:
                        try:
                            silhouette = silhouette_score(embeddings_2d[mask], labels[mask])
                        except:
                            silhouette = None
                
                resultats.append({
                    'min_cluster_size': min_cluster_size,
                    'min_samples': min_samples,
                    'epsilon': epsilon,
                    'n_clusters': n_clusters,
                    'noise_ratio': noise_ratio,
                    'silhouette': silhouette
                })
                
                print(f"MCS={min_cluster_size}, MS={min_samples}, ε={epsilon}: {n_clusters} clusters, "
                      f"{noise_ratio:.2%} bruit, silhouette={silhouette:.4f if silhouette else 'N/A'}")
                
                # Vérifier si c'est le meilleur résultat jusqu'à présent
                ecart = abs(n_clusters - cible_clusters)
                
                if (ecart <= tolerance and 
                    silhouette is not None and 
                    (ecart < meilleur_ecart or 
                     (ecart == meilleur_ecart and silhouette > meilleur_score))):
                    meilleur_ecart = ecart
                    meilleur_score = silhouette
                    meilleur_n_clusters = n_clusters
                    meilleurs_params = {
                        'min_cluster_size': min_cluster_size,
                        'min_samples': min_samples,
                        'cluster_selection_epsilon': epsilon
                    }
    
    # Afficher résumé des meilleurs paramètres
    if meilleurs_params:
        print(f"\nMeilleurs paramètres trouvés :")
        print(f"- min_cluster_size: {meilleurs_params['min_cluster_size']}")
        print(f"- min_samples: {meilleurs_params['min_samples']}")
        print(f"- cluster_selection_epsilon: {meilleurs_params['cluster_selection_epsilon']}")
        print(f"Résultat: {meilleur_n_clusters} clusters avec silhouette {meilleur_score:.4f}")
    else:
        print("Aucun ensemble de paramètres n'a produit le nombre de clusters souhaité avec un score de silhouette valide.")
    
    return meilleurs_params, resultats


2. Modifier la fonction de clustering pour utiliser les paramètres optimisés


def clustering_phase1_style(embeddings_enrichis, cible_clusters=15, optimiser=True):
    """Reproduit l'approche de clustering de la Phase 1 avec optimisation des paramètres"""
    
    print("Réduction dimensionnelle avec UMAP...")
    reducer = UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=15,
        min_dist=0.1
    )
    embeddings_2d = reducer.fit_transform(embeddings_enrichis)
    
    # Paramètres HDBSCAN
    if optimiser:
        print("Optimisation des paramètres HDBSCAN...")
        meilleurs_params, _ = optimiser_hdbscan(embeddings_2d, cible_clusters=cible_clusters)
        
        if meilleurs_params:
            min_cluster_size = meilleurs_params['min_cluster_size']
            min_samples = meilleurs_params['min_samples']
            epsilon = meilleurs_params['cluster_selection_epsilon']
        else:
            # Paramètres par défaut si l'optimisation échoue
            print("Utilisation des paramètres par défaut...")
            min_cluster_size = 30
            min_samples = 10
            epsilon = 0.5
    else:
        # Paramètres de la Phase 1 originale
        min_cluster_size = 5
        min_samples = 2
        epsilon = 0.5
    
    print(f"Application de HDBSCAN avec min_cluster_size={min_cluster_size}, min_samples={min_samples}, epsilon={epsilon}...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        metric='euclidean'
    )
    clusters = clusterer.fit_predict(embeddings_2d)
    
    # Statistiques du clustering
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = (clusters == -1).sum()
    
    print(f"Nombre de clusters détectés: {n_clusters}")
    print(f"Nombre de points classés comme bruit: {n_noise} ({n_noise/len(clusters):.2%})")
    
    # Visualisation
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
    plt.title(f'Clustering: {n_clusters} clusters détectés')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.tight_layout()
    plt.savefig("clustering_optimise.png", dpi=300)
    plt.show()
    
    # Mesure de qualité
    if n_clusters > 1:
        mask = clusters != -1
        if mask.sum() > n_clusters:
            silhouette = silhouette_score(embeddings_2d[mask], clusters[mask])
            print(f"Score de silhouette: {silhouette:.4f}")
    
    return clusters, embeddings_2d, clusterer



----------------------------------------------------
----------------------------------------------
--------------------------------------------------
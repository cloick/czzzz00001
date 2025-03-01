# Fonction pour créer les modèles de sous-causes (un par cause)
def train_subcause_models(df, text_features, categorical_features, cause_encoder, output_dir):
    """
    Entraîne un modèle spécifique pour les sous-causes de chaque cause.
    
    Args:
        df: DataFrame contenant les données
        text_features: Caractéristiques textuelles
        categorical_features: Caractéristiques catégorielles
        cause_encoder: Encodeur des causes
        output_dir: Répertoire pour sauvegarder les modèles
    
    Returns:
        Dictionnaire des modèles de sous-causes et leurs encodeurs
    """
    subcause_models = {}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Pour chaque cause présente dans les données fiables
    for cause_idx, cause_name in enumerate(cause_encoder.classes_):
        print(f"\nTraitement des sous-causes pour la cause: {cause_name}")
        
        # Filtrer les données pour cette cause
        cause_mask = df['cause_correcte'] == cause_name
        cause_df = df[cause_mask]
        
        # Vérifier s'il y a suffisamment d'exemples
        if len(cause_df) < 10:
            print(f"  Pas assez d'exemples ({len(cause_df)}) pour cette cause, modèle ignoré.")
            continue
        
        # Vérifier le nombre de sous-causes distinctes
        subcauses = cause_df['sous_cause_correcte'].unique()
        if len(subcauses) < 2:
            print(f"  Une seule sous-cause ({subcauses[0]}) pour cette cause, modèle non nécessaire.")
            subcause_models[cause_name] = {
                'model': None,
                'encoder': None,
                'single_subcause': subcauses[0]
            }
            continue
        
        print(f"  Nombre d'exemples: {len(cause_df)}, Nombre de sous-causes: {len(subcauses)}")
        
        # Encodage des sous-causes
        subcause_encoder = LabelEncoder()
        subcause_labels = subcause_encoder.fit_transform(cause_df['sous_cause_correcte'])
        n_subcauses = len(subcause_encoder.classes_)
        
        # Préparation des caractéristiques
        cause_text_features = prepare_text_features(cause_df, ['description', 'notes_resolution'])
        
        if len(categorical_features) > 0:
            cause_categorical_features = categorical_features[cause_mask]
        else:
            cause_categorical_features = np.array([]).reshape(len(cause_df), 0)
        
        # Création du modèle et du répertoire spécifique
        cause_output_dir = os.path.join(output_dir, 'subcauses', cause_name.replace('/', '_'))
        os.makedirs(cause_output_dir, exist_ok=True)
        
        # Validation croisée si suffisamment d'exemples, sinon train/validation split
        if len(cause_df) >= 50:
            print(f"  Utilisation de la validation croisée pour {cause_name}")
            fold_results = cross_validate(
                cause_df,
                cause_text_features,
                cause_categorical_features,
                subcause_labels,
                n_subcauses,
                cause_output_dir
            )
            
            analyze_cross_validation_results(
                fold_results,
                subcause_encoder,
                cause_output_dir
            )
        
        # Entraînement du modèle final pour cette cause
        print(f"  Entraînement du modèle final pour les sous-causes de {cause_name}")
        final_subcause_model = train_final_model(
            cause_df,
            cause_text_features,
            cause_categorical_features,
            subcause_labels,
            n_subcauses,
            cause_output_dir
        )
        
        # Ajout du mécanisme de détection de nouveauté spécifique à cette cause
        optimal_threshold, novelty_params = add_novelty_detection(
            final_subcause_model,
            cause_df,
            cause_text_features,
            cause_categorical_features,
            subcause_labels,
            subcause_encoder,
            cause_output_dir
        )
        
        # Sauvegarde du modèle et de l'encodeur
        subcause_models[cause_name] = {
            'model': final_subcause_model,
            'encoder': subcause_encoder,
            'threshold': optimal_threshold,
            'subcauses': subcause_encoder.classes_.tolist()
        }
        
        # Sauvegarde des informations pour ce modèle de sous-cause
        with open(os.path.join(cause_output_dir, 'subcause_model_info.pkl'), 'wb') as f:
            pickle.dump({
                'cause': cause_name,
                'subcauses': subcause_encoder.classes_.tolist(),
                'threshold': optimal_threshold,
                'n_examples': len(cause_df)
            }, f)
    
    # Sauvegarde du dictionnaire complet des modèles de sous-causes
    with open(os.path.join(output_dir, 'subcause_models_info.pkl'), 'wb') as f:
        pickle.dump({
            cause: {
                'subcauses': info['subcauses'] if 'subcauses' in info else [info.get('single_subcause')],
                'threshold': info.get('threshold')
            } for cause, info in subcause_models.items()
        }, f)
    
    return subcause_models




----------------------------------------------------------------------------------------------------------

def main():
    # ... code existant ...
    
    # Entraînement du modèle final pour les causes
    print("\nEntraînement du modèle final pour les causes...")
    final_cause_model = train_final_model(
        df_fiable, 
        text_features, 
        categorical_features, 
        cause_labels, 
        n_causes, 
        os.path.join(output_dir, 'causes')
    )
    
    # Ajout du mécanisme de détection de nouveauté pour les causes
    print("\nAjout du mécanisme de détection de nouveauté pour les causes...")
    optimal_threshold, novelty_params = add_novelty_detection(
        final_cause_model,
        df_fiable,
        text_features,
        categorical_features,
        cause_labels,
        cause_encoder,
        os.path.join(output_dir, 'causes')
    )
    
    # AJOUT: Entraînement des modèles de sous-causes
    print("\nEntraînement des modèles de sous-causes...")
    subcause_models = train_subcause_models(
        df_fiable,
        text_features,
        categorical_features,
        cause_encoder,
        output_dir
    )
    
    # ... reste du code ...
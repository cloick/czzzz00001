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
    
    # ... reste du code ...++---------------
    -----------------------------------------------------
    ----------------------------------------------------
    
    # Au lieu d'utiliser OneHotEncoder de scikit-learn
# encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
# feature_encoded = encoder.fit_transform(df[[column]])

# Utilisez to_categorical de TensorFlow
from tensorflow.keras.utils import to_categorical

# Pour les étiquettes (labels)
def encode_labels(df, label_column):
    """
    Encode les étiquettes de classe en utilisant TensorFlow.
    
    Args:
        df: DataFrame contenant les données
        label_column: Nom de la colonne contenant les étiquettes
    
    Returns:
        Étiquettes encodées et liste des classes originales
    """
    # Convertir d'abord en valeurs numériques
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    numeric_labels = encoder.fit_transform(df[label_column])
    
    # Convertir en one-hot encoding
    one_hot_labels = to_categorical(numeric_labels)
    
    return one_hot_labels, encoder.classes_

------------------------------------------------------
----------------------------------------------------------------

def prepare_categorical_features(df, categorical_columns):
    """
    Encode les variables catégorielles en utilisant TensorFlow.
    
    Args:
        df: DataFrame contenant les données
        categorical_columns: Liste des colonnes catégorielles
    
    Returns:
        Tableau numpy des caractéristiques catégorielles encodées et dictionnaire des encodeurs
    """
    from sklearn.preprocessing import LabelEncoder
    encoders = {}
    categorical_features = []
    
    for column in categorical_columns:
        if column in df.columns:
            # Création de l'encodeur
            encoder = LabelEncoder()
            
            # Transformation des données en valeurs numériques
            numeric_values = encoder.fit_transform(df[column])
            
            # Conversion en one-hot
            one_hot_values = to_categorical(numeric_values)
            
            # Stockage de l'encodeur
            encoders[column] = encoder
            
            # Ajout aux caractéristiques
            categorical_features.append(one_hot_values)
    
    # Combinaison des caractéristiques catégorielles
    if categorical_features:
        # Reshape pour permettre la concaténation si les dimensions ne correspondent pas
        reshaped_features = [f.reshape(len(df), -1) for f in categorical_features]
        all_categorical_features = np.hstack(reshaped_features)
    else:
        all_categorical_features = np.array([]).reshape(len(df), 0)
    
    return all_categorical_features, encoders

-------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------

# Avant l'appel à cross_validate
print(f"Shape de text_features: {text_features.shape if hasattr(text_features, 'shape') else 'pas un array'}")
print(f"Shape de categorical_features: {categorical_features.shape}")
print(f"Shape de cause_labels: {cause_labels.shape}")
print(f"Nombre de classes uniques: {len(np.unique(cause_labels))}")

-----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------

def train_evaluate_model(df, text_features, categorical_features, labels, n_classes, output_dir):
    """
    Entraîne et évalue le modèle en utilisant une simple division train/test.
    
    Args:
        df: DataFrame contenant les données
        text_features: Caractéristiques textuelles
        categorical_features: Caractéristiques catégorielles
        labels: Étiquettes encodées
        n_classes: Nombre de classes
        output_dir: Répertoire pour sauvegarder les modèles et résultats
    
    Returns:
        Résultats de l'évaluation et le modèle entraîné
    """
    # Création du tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    
    # Division des données en ensembles d'entraînement et de validation
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels  # Pour conserver la distribution des classes
    )
    
    print(f"Taille de l'ensemble d'entraînement: {len(train_idx)}")
    print(f"Taille de l'ensemble de validation: {len(val_idx)}")
    
    # Préparation des données d'entraînement et validation
    train_texts = text_features.iloc[train_idx].values
    val_texts = text_features.iloc[val_idx].values
    
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    
    # Tokenisation des textes
    train_encodings = tokenizer(
        train_texts.tolist(),
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='tf'
    )
    
    val_encodings = tokenizer(
        val_texts.tolist(),
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='tf'
    )
    
    # Préparation des données catégorielles si présentes
    if len(categorical_features) > 0:
        train_categorical = categorical_features[train_idx]
        val_categorical = categorical_features[val_idx]
        
        train_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': train_encodings['input_ids'],
                'attention_mask': train_encodings['attention_mask'],
                'categorical_features': train_categorical
            },
            train_labels
        )).shuffle(len(train_idx)).batch(BATCH_SIZE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': val_encodings['input_ids'],
                'attention_mask': val_encodings['attention_mask'],
                'categorical_features': val_categorical
            },
            val_labels
        )).batch(BATCH_SIZE)
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': train_encodings['input_ids'],
                'attention_mask': train_encodings['attention_mask']
            },
            train_labels
        )).shuffle(len(train_idx)).batch(BATCH_SIZE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': val_encodings['input_ids'],
                'attention_mask': val_encodings['attention_mask']
            },
            val_labels
        )).batch(BATCH_SIZE)
    
    # Création du modèle
    n_categorical = categorical_features.shape[1] if len(categorical_features) > 0 else 0
    model = create_model(n_classes, n_categorical)
    
    # Callbacks pour l'entraînement
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Entraînement du modèle
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Évaluation du modèle
    val_predictions = model.predict(val_dataset)
    val_pred_classes = np.argmax(val_predictions, axis=1)
    
    # Calcul des métriques
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(val_labels, val_pred_classes, output_dict=True)
    
    # Affichage des résultats
    print("\nRésultats de l'évaluation:")
    print(classification_report(val_labels, val_pred_classes))
    
    # Matrice de confusion
    cm = confusion_matrix(val_labels, val_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de confusion')
    plt.ylabel('Étiquette réelle')
    plt.xlabel('Étiquette prédite')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Sauvegarde de l'historique d'entraînement
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Perte')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Précision')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    
    # Sauvegarde des résultats
    with open(os.path.join(output_dir, 'evaluation_results.pkl'), 'wb') as f:
        pickle.dump({
            'report': report,
            'confusion_matrix': cm,
            'history': history.history
        }, f)
    
    return report, model


----------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------


def analyze_results(results, label_encoder, output_dir):
    """
    Analyse les résultats d'évaluation du modèle.
    
    Args:
        results: Dictionnaire contenant les résultats (rapport de classification)
        label_encoder: Encodeur des étiquettes
        output_dir: Répertoire pour sauvegarder les résultats
    """
    # Extraction des métriques globales
    accuracy = results['accuracy']
    macro_f1 = results['macro avg']['f1-score']
    weighted_f1 = results['weighted avg']['f1-score']
    
    print("\nRésultats de l'évaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")
    
    # Analyse des performances par classe
    class_performance = {}
    classes = label_encoder.classes_
    
    for class_idx, class_name in enumerate(classes):
        if str(class_idx) in results:
            class_precision = results[str(class_idx)]['precision']
            class_recall = results[str(class_idx)]['recall']
            class_f1 = results[str(class_idx)]['f1-score']
            
            class_performance[class_name] = {
                'precision': class_precision,
                'recall': class_recall,
                'f1-score': class_f1
            }
    
    # Tri des classes par F1-score
    sorted_classes = sorted(
        class_performance.items(),
        key=lambda x: x[1]['f1-score'],
        reverse=True
    )
    
    # Affichage des performances par classe
    print("\nPerformances par classe (triées par F1-score):")
    for class_name, metrics in sorted_classes:
        print(f"{class_name}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")
    
    # Identification des classes problématiques
    problematic_classes = [
        class_name for class_name, metrics in class_performance.items()
        if metrics['f1-score'] < 0.7
    ]
    
    print("\nClasses potentiellement problématiques (F1-score < 0.7):")
    for class_name in problematic_classes:
        print(f"- {class_name}: F1={class_performance[class_name]['f1-score']:.4f}")
    
    # Sauvegarde des résultats
    with open(os.path.join(output_dir, 'class_performance.pkl'), 'wb') as f:
        pickle.dump({
            'class_performance': class_performance,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1
        }, f)
    
    # Visualisation des performances par classe
    plt.figure(figsize=(12, 8))
    classes = [c[0] for c in sorted_classes]
    f1_scores = [c[1]['f1-score'] for c in sorted_classes]
    
    plt.barh(classes, f1_scores, color='skyblue')
    plt.xlabel('F1-score')
    plt.title('Performance par classe (F1-score)')
    plt.axvline(x=0.7, color='red', linestyle='--', label='Seuil F1=0.7')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_performance.png'))
    
    
    
    
    
    
    
    
    
    
    
    -------------------------------------------------------
    ------------------------------------------------------
    ----------------------------------------------------------
    ----------------------------------------------------------
    
    def train_subcause_models(df, text_features, categorical_features, cause_encoder, subcause_column, output_dir):
    """
    Entraîne un modèle spécifique pour les sous-causes de chaque cause.
    
    Args:
        df: DataFrame contenant les données
        text_features: Caractéristiques textuelles
        categorical_features: Caractéristiques catégorielles
        cause_encoder: Encodeur des causes
        subcause_column: Nom de la colonne contenant les sous-causes
        output_dir: Répertoire pour sauvegarder les modèles
    
    Returns:
        Dictionnaire des modèles de sous-causes et leurs encodeurs
    """
    subcause_models = {}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    
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
        subcauses = cause_df[subcause_column].unique()
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
        subcause_labels = subcause_encoder.fit_transform(cause_df[subcause_column])
        n_subcauses = len(subcause_encoder.classes_)
        
        # Préparation des caractéristiques
        cause_text_features = cause_df['texte_combine']
        
        if len(categorical_features) > 0:
            cause_categorical_features = categorical_features[cause_mask]
        else:
            cause_categorical_features = np.array([]).reshape(len(cause_df), 0)
        
        # Création du répertoire spécifique
        cause_output_dir = os.path.join(output_dir, 'subcauses', cause_name.replace('/', '_'))
        os.makedirs(cause_output_dir, exist_ok=True)
        
        # Entraînement et évaluation du modèle pour cette cause
        print(f"  Entraînement et évaluation du modèle pour les sous-causes de {cause_name}")
        evaluation_results, subcause_model = train_evaluate_model(
            cause_df,
            cause_text_features,
            cause_categorical_features,
            subcause_labels,
            n_subcauses,
            cause_output_dir
        )
        
        # Analyse des résultats
        analyze_results(
            evaluation_results,
            subcause_encoder,
            cause_output_dir
        )
        
        # Ajout du mécanisme de détection de nouveauté spécifique à cette cause
        optimal_threshold, novelty_params = add_novelty_detection(
            subcause_model,
            cause_df,
            cause_text_features,
            cause_categorical_features,
            subcause_labels,
            subcause_encoder,
            cause_output_dir
        )
        
        # Sauvegarde du modèle et de l'encodeur
        subcause_models[cause_name] = {
            'model': subcause_model,
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

    -------------------------------------------------------
    ------------------------------------------------------
    ----------------------------------------------------------
    ----------------------------------------------------------
    
    def main():
    # Création du répertoire de sortie
    output_dir = "metis_classification_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Chargement des données
    df_fiable = load_data("metis_tickets.csv", "gdp_tickets.csv")
    
    # Préparation des caractéristiques
    text_columns = ['description', 'notes_resolution']
    categorical_columns = ['priorite', 'statut', 'type_incident', 'plateforme']
    
    # Définition des colonnes d'étiquettes
    cause_column = 'cause_correcte'
    subcause_column = 'sous_cause_correcte'
    
    text_features = prepare_text_features(df_fiable, text_columns)
    categorical_features, cat_encoders = prepare_categorical_features(df_fiable, categorical_columns)
    
    # Encodage des étiquettes (pour les causes)
    cause_labels, cause_encoder = encode_labels(df_fiable, cause_column)
    n_causes = len(cause_encoder.classes_)
    
    # Entraînement et évaluation du modèle de causes
    print("\nEntraînement et évaluation du modèle de causes...")
    evaluation_results, final_cause_model = train_evaluate_model(
        df_fiable, 
        text_features, 
        categorical_features, 
        cause_labels, 
        n_causes, 
        os.path.join(output_dir, 'causes')
    )
    
    # Analyse des résultats
    print("\nAnalyse des résultats du modèle de causes...")
    analyze_results(
        evaluation_results, 
        cause_encoder, 
        os.path.join(output_dir, 'causes')
    )
    
    # Ajout du mécanisme de détection de nouveauté
    print("\nAjout du mécanisme de détection de nouveauté...")
    optimal_threshold, novelty_params = add_novelty_detection(
        final_cause_model,
        df_fiable,
        text_features,
        categorical_features,
        cause_labels,
        cause_encoder,
        os.path.join(output_dir, 'causes')
    )
    
    # Entraînement des modèles de sous-causes
    print("\nEntraînement des modèles de sous-causes...")
    subcause_models = train_subcause_models(
        df_fiable,
        text_features,
        categorical_features,
        cause_encoder,
        subcause_column,
        output_dir
    )
    
    # Sauvegarde des encodeurs et configurations
    with open(os.path.join(output_dir, 'encoders_and_config.pkl'), 'wb') as f:
        pickle.dump({
            'cause_encoder': cause_encoder,
            'categorical_encoders': cat_encoders,
            'max_length': MAX_LENGTH,
            'model_name': MODEL_NAME,
            'categorical_columns': categorical_columns,
            'text_columns': text_columns,
            'novelty_threshold': optimal_threshold
        }, f)
    
    print("\nEntraînement terminé et modèles sauvegardés avec succès dans", output_dir)
def prepare_text_features(df, text_columns):
    """
    Prépare les caractéristiques textuelles en combinant les colonnes de texte.
    
    Args:
        df: DataFrame contenant les données
        text_columns: Liste des colonnes contenant du texte
    
    Returns:
        Série contenant le texte combiné
    """
    # Vérification des colonnes existantes
    available_columns = [col for col in text_columns if col in df.columns]
    
    if not available_columns:
        raise ValueError(f"Aucune des colonnes textuelles {text_columns} n'existe dans le DataFrame")
    
    # Si une seule colonne textuelle, pas besoin de combiner
    if len(available_columns) == 1:
        df['texte_combine'] = df[available_columns[0]].fillna('')
    else:
        # Combinaison des colonnes textuelles
        df['texte_combine'] = df[available_columns].fillna('').agg(' '.join, axis=1)
    
    return df['texte_combine']

---------------------------------


def create_model(n_classes, categorical_features=None):
    """
    Crée un modèle TensorFlow utilisant un transformateur pré-entraîné.
    
    Args:
        n_classes: Nombre de classes pour la prédiction
        categorical_features: Caractéristiques catégorielles (peut être None ou un array)
    
    Returns:
        Modèle TensorFlow compilé
    """
    # Entrées pour le texte
    input_ids = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='attention_mask')
    
    # Chargement du modèle pré-entraîné
    transformer = TFAutoModel.from_pretrained(MODEL_NAME, local_files_only=True)
    
    # Sortie du transformateur
    sequence_output = transformer(input_ids=input_ids, attention_mask=attention_mask)[0]
    pooled_output = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    text_features = tf.keras.layers.Dropout(0.1)(pooled_output)
    
    # Déterminer si des caractéristiques catégorielles sont présentes
    has_categorical = (categorical_features is not None and 
                      isinstance(categorical_features, np.ndarray) and 
                      categorical_features.size > 0)
    
    # Intégration des caractéristiques catégorielles si présentes
    if has_categorical:
        n_categorical = categorical_features.shape[1]
        categorical_input = tf.keras.layers.Input(shape=(n_categorical,), name='categorical_features')
        combined_features = tf.keras.layers.Concatenate()([text_features, categorical_input])
        inputs = [input_ids, attention_mask, categorical_input]
    else:
        combined_features = text_features
        inputs = [input_ids, attention_mask]
    
    # Couches de classification
    x = tf.keras.layers.Dense(128, activation='relu')(combined_features)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    
    # Construction du modèle
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compilation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


---------------

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
    print("Création du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    
    # Division des données en ensembles d'entraînement et de validation
    print("Division des données...")
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels  # Pour conserver la distribution des classes
    )
    
    print(f"Taille de l'ensemble d'entraînement: {len(train_idx)}")
    print(f"Taille de l'ensemble de validation: {len(val_idx)}")
    
    # Vérification des types et dimensions des entrées
    print(f"\nTypes des entrées:")
    print(f"- text_features: {type(text_features)}")
    print(f"- categorical_features: {type(categorical_features)}")
    print(f"- labels: {type(labels)}")
    
    print(f"\nDimensions des entrées:")
    print(f"- text_features: {len(text_features)} éléments")
    if hasattr(categorical_features, 'shape'):
        print(f"- categorical_features: {categorical_features.shape}")
    else:
        print(f"- categorical_features: type non compatible avec shape")
    if hasattr(labels, 'shape'):
        print(f"- labels: {labels.shape}")
    else:
        print(f"- labels: {len(labels)} éléments")
    
    # Vérifications des indices
    print(f"\nIndices:")
    print(f"- train_idx: {len(train_idx)} éléments, min={min(train_idx)}, max={max(train_idx)}")
    print(f"- val_idx: {len(val_idx)} éléments, min={min(val_idx)}, max={max(val_idx)}")
    
    # Préparation des données d'entraînement et validation
    print("\nPréparation des données d'entraînement et validation...")
    
    # Vérification du type de text_features pour extraction correcte
    if hasattr(text_features, 'iloc'):
        print("text_features est de type pandas Series ou DataFrame")
        train_texts = text_features.iloc[train_idx].values
        val_texts = text_features.iloc[val_idx].values
    else:
        print("text_features est probablement un array ou une liste")
        train_texts = text_features[train_idx]
        val_texts = text_features[val_idx]
    
    print(f"Type de train_texts: {type(train_texts)}")
    print(f"Premier exemple de train_texts: {train_texts[0][:100] if len(train_texts) > 0 else 'vide'}")
    
    # Conversion en liste pour tokenizer
    print("\nConversion des textes en listes...")
    if hasattr(train_texts, 'tolist'):
        train_texts_list = train_texts.tolist()
    else:
        train_texts_list = [str(t) for t in train_texts]
    
    if hasattr(val_texts, 'tolist'):
        val_texts_list = val_texts.tolist()
    else:
        val_texts_list = [str(t) for t in val_texts]
    
    print(f"Longueur de train_texts_list: {len(train_texts_list)}")
    print(f"Longueur de val_texts_list: {len(val_texts_list)}")
    
    # Tokenisation
    print("\nTokenisation des textes...")
    try:
        train_encodings = tokenizer(
            train_texts_list,
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors='tf'
        )
        
        val_encodings = tokenizer(
            val_texts_list,
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors='tf'
        )
        
        print("Tokenisation réussie")
        print(f"Shape de train_encodings['input_ids']: {train_encodings['input_ids'].shape}")
        print(f"Shape de train_encodings['attention_mask']: {train_encodings['attention_mask'].shape}")
        print(f"Shape de val_encodings['input_ids']: {val_encodings['input_ids'].shape}")
        print(f"Shape de val_encodings['attention_mask']: {val_encodings['attention_mask'].shape}")
    except Exception as e:
        print(f"Erreur lors de la tokenisation: {e}")
        print(f"Exemple de texte: {train_texts_list[0] if train_texts_list else 'vide'}")
        raise
    
    # Préparation des labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    print(f"Shape de train_labels: {train_labels.shape if hasattr(train_labels, 'shape') else len(train_labels)}")
    print(f"Shape de val_labels: {val_labels.shape if hasattr(val_labels, 'shape') else len(val_labels)}")
    
    # Vérification des caractéristiques catégorielles
    has_categorical = (
        categorical_features is not None and 
        hasattr(categorical_features, 'shape') and 
        len(categorical_features.shape) > 0 and 
        categorical_features.size > 0
    )
    
    print(f"\nUtilisation de caractéristiques catégorielles: {has_categorical}")
    
    # Préparation des datasets TensorFlow
    print("\nCréation des datasets TensorFlow...")
    
    if has_categorical:
        print("Avec caractéristiques catégorielles")
        train_categorical = categorical_features[train_idx]
        val_categorical = categorical_features[val_idx]
        
        print(f"Shape de train_categorical: {train_categorical.shape}")
        print(f"Shape de val_categorical: {val_categorical.shape}")
        
        # Vérification des dimensions
        print("\nVérification de compatibilité des dimensions:")
        print(f"- train_encodings['input_ids']: {train_encodings['input_ids'].shape}")
        print(f"- train_categorical: {train_categorical.shape}")
        print(f"- train_labels: {train_labels.shape if hasattr(train_labels, 'shape') else 'pas un array'}")
        
        try:
            train_dataset = tf.data.Dataset.from_tensor_slices((
                {
                    'input_ids': train_encodings['input_ids'],
                    'attention_mask': train_encodings['attention_mask'],
                    'categorical_features': train_categorical
                },
                train_labels
            )).shuffle(len(train_idx)).batch(BATCH_SIZE)
            
            print("Dataset d'entraînement créé avec succès")
            
            val_dataset = tf.data.Dataset.from_tensor_slices((
                {
                    'input_ids': val_encodings['input_ids'],
                    'attention_mask': val_encodings['attention_mask'],
                    'categorical_features': val_categorical
                },
                val_labels
            )).batch(BATCH_SIZE)
            
            print("Dataset de validation créé avec succès")
        except Exception as e:
            print(f"Erreur lors de la création des datasets avec caractéristiques catégorielles: {e}")
            
            # Plus de détails sur l'erreur
            for batch_idx, batch in enumerate(train_categorical[:5]):
                print(f"Batch {batch_idx}: {batch.shape}, type: {type(batch)}")
            
            raise
    else:
        print("Sans caractéristiques catégorielles")
        try:
            train_dataset = tf.data.Dataset.from_tensor_slices((
                {
                    'input_ids': train_encodings['input_ids'],
                    'attention_mask': train_encodings['attention_mask']
                },
                train_labels
            )).shuffle(len(train_idx)).batch(BATCH_SIZE)
            
            print("Dataset d'entraînement créé avec succès")
            
            val_dataset = tf.data.Dataset.from_tensor_slices((
                {
                    'input_ids': val_encodings['input_ids'],
                    'attention_mask': val_encodings['attention_mask']
                },
                val_labels
            )).batch(BATCH_SIZE)
            
            print("Dataset de validation créé avec succès")
        except Exception as e:
            print(f"Erreur lors de la création des datasets sans caractéristiques catégorielles: {e}")
            print(f"Shape de train_encodings['input_ids']: {train_encodings['input_ids'].shape}")
            print(f"Shape de train_labels: {train_labels.shape if hasattr(train_labels, 'shape') else 'pas un array'}")
            raise
    
    # Création du modèle
    print("\nCréation du modèle...")
    if has_categorical:
        n_categorical = categorical_features.shape[1]
        print(f"Nombre de caractéristiques catégorielles: {n_categorical}")
        model = create_model(n_classes, n_categorical)
    else:
        print("Modèle sans caractéristiques catégorielles")
        model = create_model(n_classes)
    
    # Résumé du modèle
    print("\nRésumé du modèle:")
    model.summary()
    
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
    print("\nDébut de l'entraînement...")
    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            callbacks=callbacks
        )
        print("Entraînement terminé avec succès")
    except Exception as e:
        print(f"Erreur pendant l'entraînement: {e}")
        
        # Inspection du premier batch
        print("\nInspection des données:")
        for batch in train_dataset.take(1):
            print(f"Structure du batch: {type(batch)}")
            print(f"Clés du batch: {batch[0].keys() if isinstance(batch[0], dict) else 'pas un dict'}")
            if isinstance(batch[0], dict) and 'input_ids' in batch[0]:
                print(f"Shape de input_ids: {batch[0]['input_ids'].shape}")
        
        raise
    
    # Évaluation du modèle
    print("\nÉvaluation du modèle...")
    try:
        val_predictions = model.predict(val_dataset)
        val_pred_classes = np.argmax(val_predictions, axis=1)
        
        # Calcul des métriques
        from sklearn.metrics import classification_report, confusion_matrix
        report = classification_report(val_labels, val_pred_classes, output_dict=True)
        print("Évaluation terminée avec succès")
    except Exception as e:
        print(f"Erreur pendant l'évaluation: {e}")
        raise
    
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
    print("\nSauvegarde des résultats...")
    with open(os.path.join(output_dir, 'evaluation_results.pkl'), 'wb') as f:
        pickle.dump({
            'report': report,
            'confusion_matrix': cm,
            'history': history.history
        }, f)
    
    print("Processus d'entraînement et d'évaluation terminé")
    return report, model
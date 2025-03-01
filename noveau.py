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
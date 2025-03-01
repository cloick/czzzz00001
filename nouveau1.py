def create_model(n_classes, n_categorical_features=0):
    """
    Crée un modèle TensorFlow utilisant un transformateur pré-entraîné.
    
    Args:
        n_classes: Nombre de classes pour la prédiction
        n_categorical_features: Nombre de caractéristiques catégorielles
    
    Returns:
        Modèle TensorFlow compilé
    """
    print(f"Création d'un modèle avec {n_classes} classes et {n_categorical_features} caractéristiques catégorielles")
    
    # Entrées pour le texte
    input_ids = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='attention_mask')
    
    # Chargement du modèle pré-entraîné
    transformer = TFAutoModel.from_pretrained(MODEL_NAME, local_files_only=True)
    
    # Sortie du transformateur
    sequence_output = transformer([input_ids, attention_mask])[0]
    pooled_output = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    text_features = tf.keras.layers.Dropout(0.1)(pooled_output)
    
    # Intégration des caractéristiques catégorielles si présentes
    if n_categorical_features > 0:
        print(f"Ajout d'une entrée pour {n_categorical_features} caractéristiques catégorielles")
        categorical_input = tf.keras.layers.Input(shape=(n_categorical_features,), dtype=tf.float32, name='categorical_features')
        combined_features = tf.keras.layers.Concatenate()([text_features, categorical_input])
        inputs = [input_ids, attention_mask, categorical_input]
    else:
        print("Modèle sans caractéristiques catégorielles")
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
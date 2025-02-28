# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from transformers import AutoTokenizer, TFAutoModel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

# Configuration des paramètres
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-5
MODEL_NAME = "camembert-base"  # Modèle pré-entraîné pour le français
NUM_FOLDS = 5
RANDOM_SEED = 42

# Fonction pour charger et préparer les données
def load_data(metis_file, gdp_file):
    """
    Charge et prépare les données en joignant les tickets METIS avec les étiquettes fiables de GDP.
    
    Args:
        metis_file: Chemin vers le fichier contenant les données METIS
        gdp_file: Chemin vers le fichier contenant les données GDP fiables
    
    Returns:
        DataFrame contenant les données fusionnées
    """
    # Chargement des données METIS
    df_metis = pd.read_csv(metis_file)
    
    # Chargement des données GDP fiables
    df_gdp = pd.read_csv(gdp_file)
    
    # Joindre les données en utilisant l'identifiant du ticket
    df_fiable = df_metis.merge(
        df_gdp[['ticket_id', 'cause_correcte', 'sous_cause_correcte']], 
        on='ticket_id', 
        how='inner'
    )
    
    # Vérification du nombre de tickets récupérés
    print(f"Nombre de tickets fiables récupérés : {len(df_fiable)}")
    
    # Affichage de la distribution des causes
    print("\nDistribution des causes :")
    print(df_fiable['cause_correcte'].value_counts())
    
    # Vérification des valeurs nulles
    print("\nVérification des valeurs nulles :")
    print(df_fiable.isnull().sum())
    
    # Gestion des valeurs nulles pour les colonnes textuelles
    text_columns = ['description', 'notes_resolution', 'details_incident']
    for col in text_columns:
        if col in df_fiable.columns:
            df_fiable[col] = df_fiable[col].fillna('')
    
    return df_fiable

# Fonction pour préparer les caractéristiques textuelles
def prepare_text_features(df, text_columns):
    """
    Prépare les caractéristiques textuelles en combinant les colonnes de texte.
    
    Args:
        df: DataFrame contenant les données
        text_columns: Liste des colonnes contenant du texte
    
    Returns:
        Série contenant le texte combiné
    """
    # Combinaison des colonnes textuelles
    df['texte_combine'] = df[text_columns].fillna('').agg(' '.join, axis=1)
    
    # Tokenisation avec le modèle pré-entraîné
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    return df['texte_combine']

# Fonction pour préparer les caractéristiques catégorielles
def prepare_categorical_features(df, categorical_columns):
    """
    Encode les variables catégorielles en utilisant one-hot encoding.
    
    Args:
        df: DataFrame contenant les données
        categorical_columns: Liste des colonnes catégorielles
    
    Returns:
        Tableau numpy des caractéristiques catégorielles encodées et dictionnaire des encodeurs
    """
    encoders = {}
    categorical_features = []
    
    for column in categorical_columns:
        if column in df.columns:
            # Création de l'encodeur
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            
            # Transformation des données
            feature_encoded = encoder.fit_transform(df[[column]])
            
            # Stockage de l'encodeur
            encoders[column] = encoder
            
            # Ajout aux caractéristiques
            categorical_features.append(feature_encoded)
    
    # Combinaison des caractéristiques catégorielles
    if categorical_features:
        all_categorical_features = np.hstack(categorical_features)
    else:
        all_categorical_features = np.array([]).reshape(len(df), 0)
    
    return all_categorical_features, encoders

# Fonction pour encoder les étiquettes
def encode_labels(df, label_column):
    """
    Encode les étiquettes de classe.
    
    Args:
        df: DataFrame contenant les données
        label_column: Nom de la colonne contenant les étiquettes
    
    Returns:
        Étiquettes encodées et encodeur
    """
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(df[label_column])
    return encoded_labels, encoder

# Création du modèle TensorFlow
def create_model(n_classes, n_categorical_features=0):
    """
    Crée un modèle TensorFlow utilisant un transformateur pré-entraîné.
    
    Args:
        n_classes: Nombre de classes pour la prédiction
        n_categorical_features: Nombre de caractéristiques catégorielles
    
    Returns:
        Modèle TensorFlow compilé
    """
    # Entrées pour le texte
    input_ids = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='attention_mask')
    
    # Chargement du modèle pré-entraîné
    transformer = TFAutoModel.from_pretrained(MODEL_NAME)
    
    # Congélation des couches du transformateur (optionnel, à ajuster selon les performances)
    # for layer in transformer.layers:
    #     layer.trainable = False
    
    # Sortie du transformateur
    sequence_output = transformer(input_ids=input_ids, attention_mask=attention_mask)[0]
    pooled_output = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    text_features = tf.keras.layers.Dropout(0.1)(pooled_output)
    
    # Intégration des caractéristiques catégorielles si présentes
    if n_categorical_features > 0:
        categorical_input = tf.keras.layers.Input(shape=(n_categorical_features,), name='categorical_features')
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

# Fonction pour la validation croisée
def cross_validate(df, text_features, categorical_features, labels, n_classes, output_dir):
    """
    Effectue une validation croisée pour évaluer le modèle.
    
    Args:
        df: DataFrame contenant les données
        text_features: Caractéristiques textuelles
        categorical_features: Caractéristiques catégorielles
        labels: Étiquettes encodées
        n_classes: Nombre de classes
        output_dir: Répertoire pour sauvegarder les modèles et résultats
    
    Returns:
        Résultats de la validation croisée
    """
    # Création du tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Initialisation de la validation croisée stratifiée
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []
    
    # Pour chaque pli de la validation croisée
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\nEntraînement du pli {fold+1}/{NUM_FOLDS}")
        
        # Séparation des données d'entraînement et de validation
        train_texts = text_features.iloc[train_idx].values
        val_texts = text_features.iloc[val_idx].values
        
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        
        if len(categorical_features) > 0:
            train_categorical = categorical_features[train_idx]
            val_categorical = categorical_features[val_idx]
        
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
        
        # Création des datasets TensorFlow
        if len(categorical_features) > 0:
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
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(output_dir, f'model_fold_{fold+1}.h5'),
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
        report = classification_report(val_labels, val_pred_classes, output_dict=True)
        fold_results.append(report)
        
        # Affichage des résultats
        print(f"\nRésultats pour le pli {fold+1}:")
        print(classification_report(val_labels, val_pred_classes))
        
        # Matrice de confusion
        cm = confusion_matrix(val_labels, val_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matrice de confusion - Pli {fold+1}')
        plt.ylabel('Étiquette réelle')
        plt.xlabel('Étiquette prédite')
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_fold_{fold+1}.png'))
        
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
        plt.savefig(os.path.join(output_dir, f'training_history_fold_{fold+1}.png'))
    
    return fold_results

# Fonction pour analyser les résultats de la validation croisée
def analyze_cross_validation_results(fold_results, label_encoder, output_dir):
    """
    Analyse les résultats de la validation croisée.
    
    Args:
        fold_results: Liste des résultats pour chaque pli
        label_encoder: Encodeur des étiquettes
        output_dir: Répertoire pour sauvegarder les résultats
    """
    # Calcul des performances moyennes
    avg_accuracy = np.mean([fold['accuracy'] for fold in fold_results])
    avg_macro_f1 = np.mean([fold['macro avg']['f1-score'] for fold in fold_results])
    avg_weighted_f1 = np.mean([fold['weighted avg']['f1-score'] for fold in fold_results])
    
    print("\nRésultats moyens sur la validation croisée:")
    print(f"Accuracy: {avg_accuracy:.4f}")
    print(f"Macro F1-score: {avg_macro_f1:.4f}")
    print(f"Weighted F1-score: {avg_weighted_f1:.4f}")
    
    # Analyse des performances par classe
    class_performance = {}
    classes = label_encoder.classes_
    
    for class_idx, class_name in enumerate(classes):
        class_precision = np.mean([fold.get(str(class_idx), {}).get('precision', 0) for fold in fold_results])
        class_recall = np.mean([fold.get(str(class_idx), {}).get('recall', 0) for fold in fold_results])
        class_f1 = np.mean([fold.get(str(class_idx), {}).get('f1-score', 0) for fold in fold_results])
        
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
    with open(os.path.join(output_dir, 'cross_validation_results.pkl'), 'wb') as f:
        pickle.dump({
            'fold_results': fold_results,
            'class_performance': class_performance,
            'avg_accuracy': avg_accuracy,
            'avg_macro_f1': avg_macro_f1,
            'avg_weighted_f1': avg_weighted_f1
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

# Fonction pour entraîner le modèle final
def train_final_model(df, text_features, categorical_features, labels, n_classes, output_dir):
    """
    Entraîne le modèle final sur l'ensemble des données fiables.
    
    Args:
        df: DataFrame contenant les données
        text_features: Caractéristiques textuelles
        categorical_features: Caractéristiques catégorielles
        labels: Étiquettes encodées
        n_classes: Nombre de classes
        output_dir: Répertoire pour sauvegarder le modèle
    
    Returns:
        Modèle final entraîné
    """
    # Création du tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Tokenisation des textes
    encodings = tokenizer(
        text_features.tolist(),
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='tf'
    )
    
    # Création du dataset TensorFlow
    if len(categorical_features) > 0:
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
                'categorical_features': categorical_features
            },
            labels
        )).shuffle(len(df)).batch(BATCH_SIZE)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask']
            },
            labels
        )).shuffle(len(df)).batch(BATCH_SIZE)
    
    # Création du modèle
    n_categorical = categorical_features.shape[1] if len(categorical_features) > 0 else 0
    model = create_model(n_classes, n_categorical)
    
    # Callbacks pour l'entraînement
    callbacks = [
        EarlyStopping(monitor='loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(
            filepath=os.path.join(output_dir, 'final_model.h5'),
            monitor='loss',
            save_best_only=True
        )
    ]
    
    # Entraînement du modèle
    history = model.fit(
        dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Sauvegarde de l'historique d'entraînement
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.title('Perte')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train')
    plt.title('Précision')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_model_training_history.png'))
    
    # Sauvegarde du modèle et des encodeurs
    model.save(os.path.join(output_dir, 'final_model_tf'))
    
    return model

# Fonction pour ajouter le mécanisme de détection de nouveauté
def add_novelty_detection(model, df, text_features, categorical_features, labels, label_encoder, output_dir):
    """
    Ajoute un mécanisme de détection de nouveauté au modèle.
    
    Args:
        model: Modèle entraîné
        df: DataFrame contenant les données
        text_features: Caractéristiques textuelles
        categorical_features: Caractéristiques catégorielles
        labels: Étiquettes encodées
        label_encoder: Encodeur des étiquettes
        output_dir: Répertoire pour sauvegarder le modèle et les résultats
    
    Returns:
        Seuil de confiance optimal et paramètres de détection de nouveauté
    """
    # Création du tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Tokenisation des textes
    encodings = tokenizer(
        text_features.tolist(),
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='tf'
    )
    
    # Création des inputs pour la prédiction
    if len(categorical_features) > 0:
        inputs = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'categorical_features': categorical_features
        }
    else:
        inputs = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }
    
    # Obtention des probabilités prédites
    predictions = model.predict(inputs)
    max_probs = np.max(predictions, axis=1)
    
    # Analyse de la distribution des probabilités maximales
    plt.figure(figsize=(10, 6))
    plt.hist(max_probs, bins=50, alpha=0.7)
    plt.title('Distribution des probabilités maximales')
    plt.xlabel('Probabilité maximale')
    plt.ylabel('Nombre de tickets')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'max_probabilities_distribution.png'))
    
    # Calcul de la précision pour différents seuils de confiance
    thresholds = np.arange(0.5, 1, 0.05)
    results = []
    
    for threshold in thresholds:
        # Application du seuil
        mask = max_probs >= threshold
        if np.sum(mask) == 0:
            continue
        
        # Prédictions au-dessus du seuil
        filtered_preds = np.argmax(predictions[mask], axis=1)
        filtered_true = labels[mask]
        
        # Calcul des métriques
        accuracy = np.mean(filtered_preds == filtered_true)
        coverage = np.mean(mask)
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'coverage': coverage,
            'f1': 2 * (accuracy * coverage) / (accuracy + coverage) if (accuracy + coverage) > 0 else 0
        })
    
    # Conversion en DataFrame pour faciliter l'analyse
    results_df = pd.DataFrame(results)
    
    # Identification du seuil optimal (maximisant le F1)
    optimal_idx = results_df['f1'].idxmax()
    optimal_threshold = results_df.loc[optimal_idx, 'threshold']
    
    print(f"\nSeuil de confiance optimal: {optimal_threshold:.2f}")
    print(f"Précision à ce seuil: {results_df.loc[optimal_idx, 'accuracy']:.4f}")
    print(f"Couverture à ce seuil: {results_df.loc[optimal_idx, 'coverage']:.4f}")
    
    # Visualisation de l'impact du seuil
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['threshold'], results_df['accuracy'], label='Précision', marker='o')
    plt.plot(results_df['threshold'], results_df['coverage'], label='Couverture', marker='o')
    plt.plot(results_df['threshold'], results_df['f1'], label='F1 (harmonic mean)', marker='o')
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Seuil optimal ({optimal_threshold:.2f})')
    plt.title('Impact du seuil de confiance')
    plt.xlabel('Seuil de confiance')
    plt.ylabel('Métrique')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'confidence_threshold_analysis.png'))
    
    # Sauvegarde des paramètres de détection de nouveauté
    novelty_params = {
        'optimal_threshold': optimal_threshold,
        'thresholds_analysis': results_df.to_dict('records')
    }
    
    with open(os.path.join(output_dir, 'novelty_detection_params.pkl'), 'wb') as f:
        pickle.dump(novelty_params, f)
    
    return optimal_threshold, novelty_params

# Fonction principale
def main():
    # Création du répertoire de sortie
    output_dir = "metis_classification_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Chargement des données
    df_fiable = load_data("metis_tickets.csv", "gdp_tickets.csv")
    
    # Préparation des caractéristiques
    text_columns = ['description', 'notes_resolution']
    categorical_columns = ['priorite', 'statut', 'type_incident', 'plateforme']
    
    text_features = prepare_text_features(df_fiable, text_columns)
    categorical_features, cat_encoders = prepare_categorical_features(df_fiable, categorical_columns)
    
    # Encodage des étiquettes (pour les causes)
    cause_labels, cause_encoder = encode_labels(df_fiable, 'cause_correcte')
    n_causes = len(cause_encoder.classes_)
    
    # Validation croisée
    print("\nDébut de la validation croisée pour le modèle de causes...")
    fold_results = cross_validate(
        df_fiable, 
        text_features, 
        categorical_features, 
        cause_labels, 
        n_causes, 
        os.path.join(output_dir, 'causes')
    )
    
    # Analyse des résultats
    analyze_cross_validation_results(
        fold_results, 
        cause_encoder, 
        os.path.join(output_dir, 'causes')
    )
    
    # Entraînement du modèle final
    print("\nEntraînement du modèle final pour les causes...")
    final_cause_model = train_final_model(
        df_fiable, 
        text_features, 
        categorical_features, 
        cause_labels, 
        n_causes, 
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

if __name__ == "__main__":
    main()
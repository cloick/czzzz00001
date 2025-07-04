Parfait ! Attaquons l'encodage ! 🚀🔍 ÉTAPE 1 : ANALYSER LES TYPES DE VARIABLES# Identifier les différents types de variables à encoder
print("=== ANALYSE TYPES DE VARIABLES ===")

# Variables object (catégorielles)
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
print(f"Variables catégorielles: {len(categorical_cols)}")

# Analyser la cardinalité de chaque variable catégorielle
print("\nCARDINALITÉ DES VARIABLES CATÉGORIELLES:")
cardinalite_info = []

for col in categorical_cols:
    n_unique = X_train[col].nunique()
    n_missing = X_train[col].isnull().sum()
    cardinalite_info.append({
        'colonne': col,
        'cardinalite': n_unique,
        'missing': n_missing,
        'top_values': X_train[col].value_counts().head(3).to_dict()
    })
    print(f"{col:30s} | {n_unique:4d} catégories | {n_missing:4d} manquants")

# Séparer par cardinalité
low_cardinality = [info['colonne'] for info in cardinalite_info if info['cardinalite'] <= 10]
medium_cardinality = [info['colonne'] for info in cardinalite_info if 10 < info['cardinalite'] <= 50]
high_cardinality = [info['colonne'] for info in cardinalite_info if info['cardinalite'] > 50]

print(f"\n📊 RÉPARTITION:")
print(f"Faible cardinalité (≤10): {len(low_cardinality)} → {low_cardinality}")
print(f"Moyenne cardinalité (11-50): {len(medium_cardinality)} → {medium_cardinality}")
print(f"Haute cardinalité (>50): {len(high_cardinality)} → {high_cardinality}")

# Variables numériques et booléennes (déjà OK)
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
bool_cols = X_train.select_dtypes(include=['bool']).columns.tolist()

print(f"\n✅ Variables numériques (OK): {len(numeric_cols)}")
print(f"✅ Variables booléennes (OK): {len(bool_cols)}")🎯 ÉTAPE 2 : STRATÉGIE D'ENCODAGE PAR CARDINALITÉfrom sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

print("\n=== STRATÉGIES D'ENCODAGE ===")

# STRATÉGIE 1: OneHot pour faible cardinalité
print(f"OneHot Encoding: {low_cardinality}")

# STRATÉGIE 2: Label Encoding pour moyenne cardinalité  
print(f"Label Encoding: {medium_cardinality}")

# STRATÉGIE 3: Target Encoding ou regroupement pour haute cardinalité
print(f"Traitement spécial: {high_cardinality}")🔧 ÉTAPE 3 : IMPLÉMENTATION ENCODAGE# Copier les datasets pour préserver originaux
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

# 1. ONEHOT ENCODING (faible cardinalité)
if low_cardinality:
    print("🔄 OneHot Encoding...")
    for col in low_cardinality:
        # Créer dummies
        train_dummies = pd.get_dummies(X_train_encoded[col], prefix=col, dummy_na=True)
        test_dummies = pd.get_dummies(X_test_encoded[col], prefix=col, dummy_na=True)
        
        # Aligner les colonnes (train et test doivent avoir mêmes colonnes)
        all_columns = train_dummies.columns.union(test_dummies.columns)
        
        for dummy_col in all_columns:
            if dummy_col not in train_dummies.columns:
                train_dummies[dummy_col] = 0
            if dummy_col not in test_dummies.columns:
                test_dummies[dummy_col] = 0
        
        # Réorganiser dans même ordre
        train_dummies = train_dummies[all_columns]
        test_dummies = test_dummies[all_columns]
        
        # Ajouter au dataset et supprimer colonne originale
        X_train_encoded = pd.concat([X_train_encoded.drop(col, axis=1), train_dummies], axis=1)
        X_test_encoded = pd.concat([X_test_encoded.drop(col, axis=1), test_dummies], axis=1)
        
        print(f"  ✅ {col}: {len(all_columns)} nouvelles colonnes")

# 2. LABEL ENCODING (moyenne cardinalité)
if medium_cardinality:
    print("🔄 Label Encoding...")
    for col in medium_cardinality:
        le = LabelEncoder()
        
        # Fit sur train seulement
        X_train_encoded[f'{col}_encoded'] = le.fit_transform(X_train_encoded[col].astype(str))
        
        # Transform test (gérer les nouvelles catégories)
        test_labels = []
        for value in X_test_encoded[col].astype(str):
            if value in le.classes_:
                test_labels.append(le.transform([value])[0])
            else:
                test_labels.append(-1)  # Catégorie inconnue
        
        X_test_encoded[f'{col}_encoded'] = test_labels
        
        # Supprimer colonne originale
        X_train_encoded = X_train_encoded.drop(col, axis=1)
        X_test_encoded = X_test_encoded.drop(col, axis=1)
        
        print(f"  ✅ {col}: encodé en numérique")

print(f"\n🎯 Shape final:")
print(f"Train: {X_train_encoded.shape}")
print(f"Test: {X_test_encoded.shape}")

# Vérifier qu'on n'a plus d'object
remaining_objects = X_train_encoded.select_dtypes(include=['object']).columns.tolist()
print(f"\nColonnes object restantes: {remaining_objects}")

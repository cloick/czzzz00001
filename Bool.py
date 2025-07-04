Exactement ! Très bon ordre ! 🎯📅 ÉTAPE 1 : TRAITEMENT DES DATES# Identifier les colonnes de dates
date_columns = ['approval_set', 'end_date', 'opened_at', 'start_date']
date_cols_available = [col for col in date_columns if col in X_train.columns]

print(f"Colonnes de dates trouvées: {date_cols_available}")

# Convertir en datetime
for col in date_cols_available:
    print(f"Conversion {col}...")
    X_train[col] = pd.to_datetime(X_train[col], errors='coerce')
    X_test[col] = pd.to_datetime(X_test[col], errors='coerce')
    
    print(f"  - Valeurs nulles train: {X_train[col].isnull().sum()}")
    print(f"  - Valeurs nulles test: {X_test[col].isnull().sum()}")🕐 ÉTAPE 2 : CRÉER FEATURES TEMPORELLES# Features temporelles sur opened_at (la plus importante)
if 'opened_at' in X_train.columns:
    print("Création features temporelles...")
    
    # TRAIN
    X_train['opened_at_year'] = X_train['opened_at'].dt.year
    X_train['opened_at_month'] = X_train['opened_at'].dt.month
    X_train['opened_at_day_of_week'] = X_train['opened_at'].dt.dayofweek
    X_train['opened_at_hour'] = X_train['opened_at'].dt.hour
    X_train['opened_at_is_weekend'] = (X_train['opened_at'].dt.dayofweek >= 5).astype(int)
    
    # Features spécialisées identifiées dans notre analyse
    X_train['is_risky_hour'] = X_train['opened_at_hour'].isin([5, 17, 18, 19]).astype(int)
    X_train['is_end_of_day'] = X_train['opened_at_hour'].between(17, 19).astype(int)
    X_train['is_peak_day'] = X_train['opened_at_day_of_week'].isin([0, 1]).astype(int)  # Lun/Mar
    
    # TEST (même transformations)
    X_test['opened_at_year'] = X_test['opened_at'].dt.year
    X_test['opened_at_month'] = X_test['opened_at'].dt.month
    X_test['opened_at_day_of_week'] = X_test['opened_at'].dt.dayofweek
    X_test['opened_at_hour'] = X_test['opened_at'].dt.hour
    X_test['opened_at_is_weekend'] = (X_test['opened_at'].dt.dayofweek >= 5).astype(int)
    X_test['is_risky_hour'] = X_test['opened_at_hour'].isin([5, 17, 18, 19]).astype(int)
    X_test['is_end_of_day'] = X_test['opened_at_hour'].between(17, 19).astype(int)
    X_test['is_peak_day'] = X_test['opened_at_day_of_week'].isin([0, 1]).astype(int)

# Durée planifiée si start_date et end_date disponibles
if 'start_date' in X_train.columns and 'end_date' in X_train.columns:
    X_train['duree_planifiee'] = (X_train['end_date'] - X_train['start_date']).dt.total_seconds() / 3600
    X_test['duree_planifiee'] = (X_test['end_date'] - X_test['start_date']).dt.total_seconds() / 3600

# Créer aussi la variable success pour éventuels calculs
X_train['success'] = (y_train == 'Succès').astype(int)
X_test['success'] = (y_test == 'Succès').astype(int)

print("✅ Features temporelles créées")🔧 ÉTAPE 3 : SUPPRIMER COLONNES DATES ORIGINALES# Supprimer les colonnes dates originales (plus besoin)
cols_to_drop = [col for col in date_cols_available if col in X_train.columns]
X_train = X_train.drop(columns=cols_to_drop)
X_test = X_test.drop(columns=cols_to_drop)

print(f"Colonnes supprimées: {cols_to_drop}")
print(f"Shape final train: {X_train.shape}")
print(f"Shape final test: {X_test.shape}")

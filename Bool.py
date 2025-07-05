# FONCTIONS PREPROCESSING MODULAIRES

def encodage(df):
    """Encoder les variables catégorielles - Différentes stratégies à tester"""
    
    # VERSION 1: Label Encoding simple pour baseline
    from sklearn.preprocessing import LabelEncoder
    
    df_encoded = df.copy()
    
    for col in df_encoded.select_dtypes('object').columns:
        if col != 'dv_close_code':  # Ne pas encoder notre target
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    # VERSION 2: Target Encoding (à tester plus tard)
    # target_means = df.groupby(col)['success'].mean()
    # df_encoded[col] = df[col].map(target_means)
    
    # VERSION 3: One-Hot Encoding (à tester plus tard)
    # df_encoded = pd.get_dummies(df, columns=categorical_cols)
    
    return df_encoded

def feature_engineering(df):
    """Créer nouvelles features - Évolution progressive"""
    
    df_fe = df.copy()
    
    # VERSION 1: Baseline - pas de nouvelles features
    # On garde tel quel pour commencer
    
    # VERSION 2: Features temporelles (à ajouter plus tard)
    # if 'opened_at' in df.columns:
    #     df_fe['opened_hour'] = pd.to_datetime(df['opened_at']).dt.hour
    #     df_fe['is_risky_hour'] = df_fe['opened_hour'].isin([17, 18, 19])
    #     df_fe['is_weekend'] = pd.to_datetime(df['opened_at']).dt.dayofweek >= 5
    
    # VERSION 3: Features d'interaction (à tester plus tard)
    # df_fe['risk_complexity'] = df['dv_risk'] + df['dv_u_type_change_silca']
    
    # VERSION 4: Features historiques équipes (advanced)
    # team_success_rate = df.groupby('dv_assignment_group')['success'].mean()
    # df_fe['team_historical_success'] = df['dv_assignment_group'].map(team_success_rate)
    
    return df_fe

def imputation(df):
    """Gérer valeurs manquantes - Stratégies à comparer"""
    
    # VERSION 1: Drop NA (baseline)
    df_clean = df.dropna(axis=0)
    
    # VERSION 2: Fill NA (à tester)
    # df_clean = df.fillna(-999)  # Valeur sentinelle
    
    # VERSION 3: Fill par mode/médiane (à tester)
    # for col in df.columns:
    #     if df[col].dtype == 'object':
    #         df_clean[col] = df[col].fillna(df[col].mode()[0])
    #     else:
    #         df_clean[col] = df[col].fillna(df[col].median())
    
    # VERSION 4: Imputation sophistiquée (KNN, etc.)
    # from sklearn.impute import KNNImputer
    # imputer = KNNImputer(n_neighbors=5)
    # df_clean = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    return df_clean

def preprocessing(df):
    """Pipeline preprocessing complet"""
    
    df_processed = df.copy()
    
    # Étapes du pipeline
    df_processed = encodage(df_processed)
    df_processed = feature_engineering(df_processed)
    df_processed = imputation(df_processed)
    
    # Séparation X, y
    X = df_processed.drop('dv_close_code', axis=1)
    y = df_processed['dv_close_code']
    
    print("=== DISTRIBUTION TARGET APRÈS PREPROCESSING ===")
    print(y.value_counts())
    print(f"Shape finale: X{X.shape}, y{y.shape}")
    
    return X, y

# TEST DE LA PIPELINE
print("=== TEST PREPROCESSING BASELINE ===")
X_train, y_train = preprocessing(trainset)
X_test, y_test = preprocessing(testset)

print(f"\nFeatures disponibles: {list(X_train.columns)}")

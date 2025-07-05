def preprocessing(df):
    """Pipeline preprocessing complet"""
    
    df_processed = df.copy()
    
    # Étapes du pipeline
    df_processed = encodage(df_processed)
    df_processed = feature_engineering(df_processed)
    df_processed = imputation(df_processed)
    
    # Séparation X, y 
    X = df_processed.drop('dv_close_code', axis=1)
    
    # ENCODER LA TARGET
    from sklearn.preprocessing import LabelEncoder
    le_target = LabelEncoder()
    y = le_target.fit_transform(df_processed['dv_close_code'])
    
    print("=== DISTRIBUTION TARGET APRÈS PREPROCESSING ===")
    print(f"Target encodée: {pd.Series(y).value_counts().sort_index()}")
    print(f"Shape finale: X{X.shape}, y{y.shape}")
    
    return X, y

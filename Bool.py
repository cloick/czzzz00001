def encodage(df):
    """Version avec regroupement pour haute cardinalité"""
    
    df_encoded = df.copy()
    high_cardinality_threshold = 100  # Seuil à ajuster
    
    for col in df_encoded.select_dtypes('object').columns:
        if col != 'dv_close_code':
            
            unique_count = df_encoded[col].nunique()
            
            if unique_count > high_cardinality_threshold:
                # REGROUPEMENT pour haute cardinalité
                print(f"Regroupement {col}: {unique_count} → TOP {high_cardinality_threshold} + Others")
                
                # Garder les TOP catégories les plus fréquentes
                top_categories = df_encoded[col].value_counts().head(high_cardinality_threshold).index
                df_encoded[col] = df_encoded[col].apply(
                    lambda x: x if x in top_categories else 'Others'
                )
                
                print(f"Après regroupement: {df_encoded[col].nunique()} catégories")
            
            # LABEL ENCODING pour toutes les variables
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    return df_encoded

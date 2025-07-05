def encodage(df):
    """Version améliorée avec target encoding pour haute cardinalité"""
    
    df_encoded = df.copy()
    high_cardinality_cols = ['dv_assignment_group']  
    
    for col in df_encoded.select_dtypes('object').columns:
        if col != 'dv_close_code':
            
            if col in high_cardinality_cols and 'dv_close_code' in df_encoded.columns:
                # TARGET ENCODING pour haute cardinalité
                print(f"Target encoding pour {col}")
                
                # Calculer taux de succès par équipe
                target_means = df_encoded.groupby(col)['dv_close_code'].apply(
                    lambda x: (x == 'Succès').mean()
                )
                
                # Remplacer par le taux de succès moyen
                df_encoded[f'{col}_success_rate'] = df_encoded[col].map(target_means)
                
                # Supprimer la colonne originale
                df_encoded = df_encoded.drop(col, axis=1)
                
            else:
                # LABEL ENCODING pour le reste
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    return df_encoded

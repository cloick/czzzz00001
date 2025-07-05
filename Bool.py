def test_class_balancing_approaches():
    """Tester différentes approches pour gérer le déséquilibre des classes"""
    
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTEENN
    from sklearn.feature_selection import SelectKBest, f_classif
    import pandas as pd
    
    print("🚀 Préparation des données avec k=10...")
    
    # Préparer les données avec k=10 (configuration optimale)
    selector = SelectKBest(score_func=f_classif, k=10)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    selected_features = X_train.columns[selector.get_support()]
    X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features)
    X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features)
    
    # Modèle avec hyperparamètres optimaux trouvés
    base_model_params = {
        'criterion': 'entropy', 
        'max_depth': 15,
        'max_features': None,
        'min_samples_leaf': 4,
        'min_samples_split': 8,
        'n_estimators': 869,
        'random_state': 42
    }
    
    print("📊 Distribution originale:")
    print(pd.Series(y_train).value_counts().sort_index())
    
    # =====================================================
    # APPROCHE 1: SMOTE
    # =====================================================
    print("\n" + "="*60)
    print("=== APPROCHE 1: SMOTE ===")
    print("="*60)
    
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_selected, y_train)
    
    print("📊 Distribution après SMOTE:")
    print(pd.Series(y_train_smote).value_counts().sort_index())
    print(f"Taille dataset: {X_train_selected.shape[0]} → {X_train_smote.shape[0]}")
    
    # Convertir en DataFrame pour garder les noms de colonnes
    X_train_smote = pd.DataFrame(X_train_smote, columns=selected_features)
    
    model_smote = RandomForestClassifier(class_weight=None, **base_model_params)
    evaluation(model_smote, X_train_smote, y_train_smote, X_test_selected, y_test)
    
    # =====================================================
    # APPROCHE 2: SMOTEENN (SMOTE + Undersampling)
    # =====================================================
    print("\n" + "="*60)
    print("=== APPROCHE 2: SMOTEENN (SMOTE + Undersampling) ===")
    print("="*60)
    
    smoteenn = SMOTEENN(random_state=42)
    X_train_smoteenn, y_train_smoteenn = smoteenn.fit_resample(X_train_selected, y_train)
    
    print("📊 Distribution après SMOTEENN:")
    print(pd.Series(y_train_smoteenn).value_counts().sort_index())
    print(f"Taille dataset: {X_train_selected.shape[0]} → {X_train_smoteenn.shape[0]}")
    
    # Convertir en DataFrame
    X_train_smoteenn = pd.DataFrame(X_train_smoteenn, columns=selected_features)
    
    model_smoteenn = RandomForestClassifier(class_weight=None, **base_model_params)
    evaluation(model_smoteenn, X_train_smoteenn, y_train_smoteenn, X_test_selected, y_test)
    
    # =====================================================
    # APPROCHE 3: COST-SENSITIVE LEARNING
    # =====================================================
    print("\n" + "="*60)
    print("=== APPROCHE 3: COST-SENSITIVE LEARNING ===")
    print("="*60)
    
    # Calculer les poids inversement proportionnels à la fréquence
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    print("📊 Poids des classes calculés:")
    for classe, poids in class_weight_dict.items():
        print(f"Classe {classe}: {poids:.2f}")
    
    model_weighted = RandomForestClassifier(class_weight=class_weight_dict, **base_model_params)
    evaluation(model_weighted, X_train_selected, y_train, X_test_selected, y_test)
    
    # =====================================================
    # APPROCHE 4: COST-SENSITIVE EXTRÊME
    # =====================================================
    print("\n" + "="*60)
    print("=== APPROCHE 4: COST-SENSITIVE EXTRÊME ===")
    print("="*60)
    
    # Poids manuels très élevés pour classes minoritaires
    extreme_weights = {0: 100, 1: 100, 2: 100, 3: 1, 4: 100}
    
    print("📊 Poids extrêmes:")
    for classe, poids in extreme_weights.items():
        print(f"Classe {classe}: {poids}")
    
    model_extreme = RandomForestClassifier(class_weight=extreme_weights, **base_model_params)
    evaluation(model_extreme, X_train_selected, y_train, X_test_selected, y_test)
    
    print("\n" + "="*60)
    print("=== RÉCAPITULATIF À VENIR ===")
    print("="*60)
    print("Analysez les 4 approches et dites-moi laquelle performe le mieux !")

# Lancer tous les tests
test_class_balancing_approaches()

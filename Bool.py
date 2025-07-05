def optimize_hyperparameters_intensive():
    """Optimisation hyperparamètres INTENSIVE - Version cohérente avec notre code"""
    
    print("🚀 Préparation des données avec k=10 (configuration optimale)...")
    
    # Utiliser SelectKBest k=10 (votre configuration optimale)
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint
    import pandas as pd
    
    selector = SelectKBest(score_func=f_classif, k=10)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Garder les noms des features sélectionnées
    selected_features = X_train.columns[selector.get_support()]
    X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features)
    X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features)
    
    print(f"Features sélectionnées: {list(selected_features)}")
    
    # GRID ÉTENDU pour exploration intensive
    param_distributions = {
        'n_estimators': randint(50, 1000),              
        'max_depth': [5, 10, 15, 20, 25, 30, None],     
        'min_samples_split': randint(2, 30),            
        'min_samples_leaf': randint(1, 15),               
        'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7], 
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None],
        'criterion': ['gini', 'entropy'],               
        'max_samples': [None, 0.8, 0.9, 1.0],          
    }
    
    # Utiliser VOTRE random_state
    rf = RandomForestClassifier(random_state=42)
    
    # RANDOMSEARCH INTENSIF
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=300,                   
        cv=5,                         
        scoring='f1_macro',
        n_jobs=-1,                    
        verbose=2,
        random_state=42,              # ← VOTRE random_state !
        return_train_score=True       
    )
    
    print("🚀 Lancement RandomSearchCV INTENSIF (300 itérations, 5-fold CV)...")
    print("💪 Exploitation maximale de vos ressources...")
    
    # FIT
    random_search.fit(X_train_selected, y_train)
    
    # Résultats
    print("\n🏆 MEILLEURS HYPERPARAMÈTRES :")
    print(random_search.best_params_)
    print(f"\n📊 Meilleur score CV: {random_search.best_score_:.4f}")
    
    # ÉVALUATION avec VOTRE fonction evaluation() existante
    print("\n" + "="*60)
    print("=== ÉVALUATION DU MODÈLE OPTIMAL ===")
    print("="*60)
    
    best_model = random_search.best_estimator_
    
    # Utiliser VOTRE fonction evaluation() avec les données sélectionnées
    evaluation(best_model, X_train_selected, y_train, X_test_selected, y_test)
    
    return random_search, X_train_selected, X_test_selected

# Lancer l'optimisation
random_search, X_train_opt, X_test_opt = optimize_hyperparameters_intensive()

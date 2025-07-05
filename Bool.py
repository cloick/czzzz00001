def evaluation(model, X_train_eval=None, y_train_eval=None, X_test_eval=None, y_test_eval=None):
    """Évaluation complète du modèle avec données optionnelles"""
    
    # Utiliser les données globales par défaut ou celles passées en paramètre
    X_train_use = X_train_eval if X_train_eval is not None else X_train
    y_train_use = y_train_eval if y_train_eval is not None else y_train
    X_test_use = X_test_eval if X_test_eval is not None else X_test
    y_test_use = y_test_eval if y_test_eval is not None else y_test
    
    # Entraînement
    model.fit(X_train_use, y_train_use)
    y_pred = model.predict(X_test_use)
    
    print("=== MATRICE DE CONFUSION ===")
    print(confusion_matrix(y_test_use, y_pred))
    
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test_use, y_pred))
    
    # Learning curves
    print("\n=== LEARNING CURVES ===")
    N, train_score, val_score = learning_curve(model, X_train_use, y_train_use,
                                              cv=4, scoring='f1_macro',
                                               train_sizes=np.linspace(0.1, 1, 10))
    
    plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.legend()
    plt.title('Learning Curves')
    plt.xlabel('Training examples')
    plt.ylabel('F1 Score (macro)')
    plt.show()
    
    # Feature importance si disponible
    if hasattr(model, 'feature_importances_'):
        import pandas as pd
        print("\n=== FEATURE IMPORTANCE ===")
        
        # Utiliser les bons noms de colonnes
        column_names = X_train_use.columns if hasattr(X_train_use, 'columns') else [f'feature_{i}' for i in range(X_train_use.shape[1])]
        
        feature_imp = pd.DataFrame(
            model.feature_importances_, 
            index=column_names
        ).sort_values(0, ascending=False)
        
        feature_imp.head(10).plot.bar(figsize=(12, 6))
        plt.title('Top 10 Feature Importances')
        plt.show()
    
    return model🚀 FONCTION DE TEST FEATURE SELECTION :def test_feature_selection():
    """Tester différents nombres de features avec évaluation complète"""
    
    from sklearn.metrics import f1_score
    k_values = [10, 15, 20, 25, 30, 'all']
    
    for k in k_values:
        print(f"\n{'='*50}")
        print(f"=== TEST SelectKBest k={k} ===")
        print(f"{'='*50}")
        
        if k == 'all':
            # Version actuelle (baseline)
            model = RandomForestClassifier(random_state=0)
            evaluation(model)  # Utilise les variables globales
            
        else:
            # SelectKBest
            selector = SelectKBest(score_func=f_classif, k=k)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            # Convertir en DataFrame pour garder les noms de colonnes
            selected_features = X_train.columns[selector.get_support()]
            X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features)
            X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features)
            
            print(f"Features sélectionnées ({k}): {list(selected_features)}")
            
            # Évaluation complète avec la fonction modifiée
            model = RandomForestClassifier(random_state=0)
            evaluation(model, X_train_selected, y_train, X_test_selected, y_test)

# Lancer les tests
test_feature_selection()

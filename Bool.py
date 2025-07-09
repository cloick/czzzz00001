L'utilisateur veut optimiser les 5 premiers modèles (LogisticRegression, BalancedRF, XGBoost, LightGBM, RandomForest_Balanced) avec RandomizedSearchCV. Je vois que LogisticRegression est arrivé en tête avec 53.4% de recall, suivi de très près par les autres modèles.

Il veut :
1. Optimiser ces 5 modèles avec RandomizedSearchCV
2. Beaucoup d'itérations (200+) 
3. Une comparaison finale avec les meilleurs scores

Je dois créer une fonction qui :
- Définit les hyperparamètres pour chaque modèle
- Lance RandomizedSearchCV pour chacun
- Compare les résultats finaux
- Utilise le même preprocessing (SelectKBest k=5)

Pour chaque modèle, je dois définir les hyperparamètres les plus importants :

1. LogisticRegression : C, penalty, solver, max_iter
2. BalancedRF : n_estimators, max_depth, min_samples_split, min_samples_leaf, sampling_strategy
3. XGBoost : n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda
4. LightGBM : n_estimators, max_depth, learning_rate, num_leaves, min_data_in_leaf, feature_fraction
5. RandomForest_Balanced : n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features

Je vais utiliser 'recall' comme scoring puisque c'est leur objectif principal.

Excellents résultats ! **LogisticRegression** arrive en tête avec 53.4% de recall 🎯

Voici la fonction d'optimisation complète pour les 5 meilleurs modèles :

```python
def optimize_top_models():
    """Optimisation par RandomizedSearchCV des 5 meilleurs modèles"""
    
    # === IMPORTS ===
    global pd
    import numpy as np
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.metrics import confusion_matrix, classification_report
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from imblearn.ensemble import BalancedRandomForestClassifier
    
    # === PREPROCESSING (SelectKBest k=5) ===
    selector = SelectKBest(score_func=f_classif, k=5)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    selected_features = X_train.columns[selector.get_support()]
    X_train_sel = pd.DataFrame(X_train_selected, columns=selected_features)
    X_test_sel = pd.DataFrame(X_test_selected, columns=selected_features)
    
    print(f"Features sélectionnées (5): {list(selected_features)}")
    
    # === DÉFINITION DES HYPERPARAMÈTRES ===
    
    # 1. LogisticRegression (Champion actuel)
    logistic_params = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['liblinear', 'saga', 'lbfgs'],
        'max_iter': [500, 1000, 2000, 3000],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # Pour elasticnet
    }
    
    # 2. BalancedRF
    balanced_rf_params = {
        'n_estimators': [100, 200, 300, 500, 800],
        'max_depth': [3, 5, 7, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 8, 12],
        'sampling_strategy': ['auto', 'majority', 'not minority'],
        'bootstrap': [True, False]
    }
    
    # 3. XGBoost
    xgb_params = {
        'n_estimators': [100, 200, 300, 500, 800],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1, 2],
        'reg_lambda': [0, 0.1, 0.5, 1, 2],
        'min_child_weight': [1, 3, 5, 7]
    }
    
    # 4. LightGBM
    lgb_params = {
        'n_estimators': [100, 200, 300, 500, 800],
        'max_depth': [3, 4, 5, 6, 7, 8, -1],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
        'num_leaves': [20, 31, 50, 80, 100, 150],
        'min_data_in_leaf': [5, 10, 20, 30, 50],
        'feature_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
        'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
        'bagging_freq': [0, 1, 5, 10],
        'reg_alpha': [0, 0.1, 0.5, 1],
        'reg_lambda': [0, 0.1, 0.5, 1]
    }
    
    # 5. RandomForest_Balanced
    rf_params = {
        'n_estimators': [100, 200, 300, 500, 800],
        'max_depth': [3, 5, 7, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 8, 12],
        'max_features': ['sqrt', 'log2', None, 0.5, 0.7],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }
    
    # === MODÈLES ET PARAMÈTRES ===
    models_to_optimize = {
        'LogisticRegression': {
            'model': LogisticRegression(class_weight='balanced', random_state=42),
            'params': logistic_params
        },
        'BalancedRF': {
            'model': BalancedRandomForestClassifier(random_state=42),
            'params': balanced_rf_params
        },
        'XGBoost': {
            'model': XGBClassifier(
                scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                random_state=42,
                eval_metric='logloss'
            ),
            'params': xgb_params
        },
        'LightGBM': {
            'model': LGBMClassifier(
                class_weight='balanced',
                random_state=42,
                verbose=-1
            ),
            'params': lgb_params
        },
        'RandomForest_Balanced': {
            'model': RandomForestClassifier(class_weight='balanced', random_state=42),
            'params': rf_params
        }
    }
    
    # === OPTIMISATION ===
    optimized_results = {}
    best_models = {}
    
    for name, config in models_to_optimize.items():
        print(f"\n{'='*80}")
        print(f"=== OPTIMISATION {name} ===")
        print(f"{'='*80}")
        print(f"Nombre de paramètres à tester: {len(config['params'])}")
        print("🔄 Lancement RandomizedSearchCV (peut prendre du temps...)...")
        
        # RandomizedSearchCV avec beaucoup d'itérations
        search = RandomizedSearchCV(
            estimator=config['model'],
            param_distributions=config['params'],
            scoring='recall',  # Objectif principal
            cv=4,
            n_iter=250,  # Beaucoup d'itérations comme demandé
            random_state=42,
            n_jobs=-1,  # Utiliser tous les cores
            verbose=1
        )
        
        try:
            # Lancement de l'optimisation
            search.fit(X_train_sel, y_train)
            
            # Récupération du meilleur modèle
            best_model = search.best_estimator_
            best_models[name] = best_model
            
            # Prédictions avec le meilleur modèle
            y_pred = best_model.predict(X_test_sel)
            
            # Métriques
            cm = confusion_matrix(y_test, y_pred)
            recall_probleme = cm[0,0] / (cm[0,0] + cm[0,1])
            precision_probleme = cm[0,0] / (cm[0,0] + cm[1,0]) if (cm[0,0] + cm[1,0]) > 0 else 0
            fausses_alertes = cm[1,0]
            f1_probleme = 2 * (precision_probleme * recall_probleme) / (precision_probleme + recall_probleme) if (precision_probleme + recall_probleme) > 0 else 0
            
            optimized_results[name] = {
                'best_score': search.best_score_,
                'best_params': search.best_params_,
                'recall_probleme': recall_probleme,
                'precision_probleme': precision_probleme,
                'fausses_alertes': fausses_alertes,
                'f1_probleme': f1_probleme
            }
            
            print(f"✅ {name} TERMINÉ !")
            print(f"Meilleur score CV: {search.best_score_:.4f}")
            print(f"Recall test: {recall_probleme:.4f}")
            print(f"Precision test: {precision_probleme:.4f}")
            print(f"Meilleurs paramètres: {search.best_params_}")
            
        except Exception as e:
            print(f"❌ Erreur avec {name}: {e}")
            continue
    
    # === COMPARAISON FINALE ===
    print(f"\n{'='*100}")
    print("=== COMPARAISON FINALE DES MODÈLES OPTIMISÉS ===")
    print(f"{'='*100}")
    
    if optimized_results:
        # Créer DataFrame de comparaison
        comparison_df = pd.DataFrame(optimized_results).T
        comparison_df = comparison_df.sort_values('recall_probleme', ascending=False)
        
        print("\n📊 CLASSEMENT PAR RECALL PROBLÈME (Test Set):")
        print(comparison_df[['recall_probleme', 'precision_probleme', 'fausses_alertes', 'f1_probleme', 'best_score']].round(4))
        
        # Recommandations
        best_recall = comparison_df.index[0]
        best_precision = comparison_df.sort_values('precision_probleme', ascending=False).index[0]
        best_f1 = comparison_df.sort_values('f1_probleme', ascending=False).index[0]
        best_cv = comparison_df.sort_values('best_score', ascending=False).index[0]
        
        print(f"\n🏆 CHAMPIONS APRÈS OPTIMISATION:")
        print(f"🎯 Meilleur RECALL (détection): {best_recall} - {comparison_df.loc[best_recall, 'recall_probleme']:.4f}")
        print(f"🔍 Meilleure PRECISION (fiabilité): {best_precision} - {comparison_df.loc[best_precision, 'precision_probleme']:.4f}")
        print(f"⚖️  Meilleur F1 (équilibre): {best_f1} - {comparison_df.loc[best_f1, 'f1_probleme']:.4f}")
        print(f"📈 Meilleur CV Score: {best_cv} - {comparison_df.loc[best_cv, 'best_score']:.4f}")
        
        # Amélioration vs baseline
        baseline_recall = 0.534  # LogisticRegression non optimisé
        best_optimized_recall = comparison_df.loc[best_recall, 'recall_probleme']
        improvement = best_optimized_recall - baseline_recall
        
        print(f"\n📈 AMÉLIORATION vs BASELINE:")
        print(f"Baseline (LogisticRegression): {baseline_recall:.4f}")
        print(f"Meilleur optimisé: {best_optimized_recall:.4f}")
        print(f"Gain: {improvement:+.4f} ({improvement/baseline_recall*100:+.2f}%)")
    
    return optimized_results, best_models

# === LANCEMENT DE L'OPTIMISATION ===
print("🚀 DÉMARRAGE DE L'OPTIMISATION DES 5 MEILLEURS MODÈLES")
print("⏰ Cela peut prendre plusieurs heures...")

optimized_results, best_models = optimize_top_models()
```

**Cette fonction va :**
1. **Optimiser chaque modèle** avec 250 itérations RandomizedSearchCV
2. **Comparer les performances** optimisées
3. **Identifier le champion final** pour votre cas d'usage
4. **Calculer l'amélioration** vs votre baseline actuel

**Temps estimé :** 2-4 heures selon votre machine 🕐

Lancez et tenez-moi au courant des résultats ! 🎯

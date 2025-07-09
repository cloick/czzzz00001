L'utilisateur veut maintenant essayer l'approche bayésienne pour l'optimisation des hyperparamètres. Je vois qu'il a eu des erreurs avec les solveurs LogisticRegression - il y a des incompatibilités entre les solveurs et les penalties. Je dois créer une fonction qui utilise l'optimisation bayésienne (probablement avec scikit-optimize ou optuna) et qui évite ces erreurs de compatibilité.

Les erreurs qu'il a eues :
1. Features constantes détectées
2. l1_ratio parameter seulement utilisé avec penalty='elasticnet'
3. penalty=None ignore les paramètres C et l1_ratio
4. Solver lbfgs supporte seulement 'l2' ou None penalties, pas 'elasticnet'

Je dois donc créer une fonction qui :
1. Utilise l'optimisation bayésienne (probablement avec scikit-optimize)
2. Évite les incompatibilités de paramètres pour LogisticRegression
3. Garde la même structure que son code
4. Utilise les même modèles que dans sa fonction précédente

Je vais utiliser scikit-optimize avec BayesSearchCV qui est plus simple que de créer des espaces de recherche manuels.

Parfait ! L'optimisation bayésienne est souvent plus efficace que RandomizedSearchCV. Voici la version avec **BayesSearchCV** qui évite toutes les erreurs d'incompatibilité :

```python
def optimize_bayesian_approach():
    """Optimisation bayésienne avec gestion parfaite des incompatibilités"""
    
    # === IMPORTS ===
    global pd
    import numpy as np
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from imblearn.ensemble import BalancedRandomForestClassifier
    
    # Import BayesSearchCV
    try:
        from skopt import BayesSearchCV
        from skopt.space import Real, Integer, Categorical
        print("✅ scikit-optimize disponible")
    except ImportError:
        print("❌ Installation requise: pip install scikit-optimize")
        return None, None
    
    # === PREPROCESSING ===
    selector = SelectKBest(score_func=f_classif, k=5)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    selected_features = X_train.columns[selector.get_support()]
    X_train_sel = pd.DataFrame(X_train_selected, columns=selected_features)
    X_test_sel = pd.DataFrame(X_test_selected, columns=selected_features)
    
    print(f"Features sélectionnées (5): {list(selected_features)}")
    
    # === ESPACES DE RECHERCHE BAYÉSIENNE ===
    # Définition des espaces compatibles pour chaque modèle
    
    # 1. LogisticRegression - 4 MODÈLES SÉPARÉS POUR ÉVITER LES CONFLITS
    logistic_models = {
        'LogisticRegression_liblinear_l1': {
            'model': LogisticRegression(
                class_weight='balanced',
                solver='liblinear',
                penalty='l1',
                random_state=42
            ),
            'search_space': {
                'C': Real(0.001, 1000, prior='log-uniform'),
                'max_iter': Integer(500, 3000)
            }
        },
        'LogisticRegression_liblinear_l2': {
            'model': LogisticRegression(
                class_weight='balanced',
                solver='liblinear',
                penalty='l2',
                random_state=42
            ),
            'search_space': {
                'C': Real(0.001, 1000, prior='log-uniform'),
                'max_iter': Integer(500, 3000)
            }
        },
        'LogisticRegression_saga_elasticnet': {
            'model': LogisticRegression(
                class_weight='balanced',
                solver='saga',
                penalty='elasticnet',
                random_state=42
            ),
            'search_space': {
                'C': Real(0.001, 1000, prior='log-uniform'),
                'l1_ratio': Real(0.1, 0.9),
                'max_iter': Integer(500, 3000)
            }
        },
        'LogisticRegression_lbfgs_l2': {
            'model': LogisticRegression(
                class_weight='balanced',
                solver='lbfgs',
                penalty='l2',
                random_state=42
            ),
            'search_space': {
                'C': Real(0.001, 1000, prior='log-uniform'),
                'max_iter': Integer(500, 3000)
            }
        }
    }
    
    # 2. Autres modèles
    other_models = {
        'BalancedRF': {
            'model': BalancedRandomForestClassifier(random_state=42),
            'search_space': {
                'n_estimators': Integer(100, 800),
                'max_depth': Integer(3, 20),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 12),
                'sampling_strategy': Categorical(['auto', 'majority', 'not minority']),
                'bootstrap': Categorical([True, False])
            }
        },
        'XGBoost': {
            'model': XGBClassifier(
                scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                random_state=42,
                eval_metric='logloss'
            ),
            'search_space': {
                'n_estimators': Integer(100, 800),
                'max_depth': Integer(3, 8),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'subsample': Real(0.6, 1.0),
                'colsample_bytree': Real(0.6, 1.0),
                'reg_alpha': Real(0, 2),
                'reg_lambda': Real(0, 2),
                'min_child_weight': Integer(1, 7),
                'gamma': Real(0, 0.4)
            }
        },
        'LightGBM': {
            'model': LGBMClassifier(
                class_weight='balanced',
                random_state=42,
                verbose=-1
            ),
            'search_space': {
                'n_estimators': Integer(100, 800),
                'max_depth': Integer(3, 8),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'num_leaves': Integer(20, 150),
                'min_data_in_leaf': Integer(5, 50),
                'feature_fraction': Real(0.6, 1.0),
                'bagging_fraction': Real(0.6, 1.0),
                'bagging_freq': Integer(0, 10),
                'reg_alpha': Real(0, 1),
                'reg_lambda': Real(0, 1)
            }
        },
        'RandomForest_Balanced': {
            'model': RandomForestClassifier(class_weight='balanced', random_state=42),
            'search_space': {
                'n_estimators': Integer(100, 800),
                'max_depth': Integer(3, 20),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 12),
                'max_features': Categorical(['sqrt', 'log2', None]),
                'bootstrap': Categorical([True, False]),
                'criterion': Categorical(['gini', 'entropy'])
            }
        }
    }
    
    # === FUSION DES MODÈLES ===
    all_models = {**logistic_models, **other_models}
    
    # === OPTIMISATION BAYÉSIENNE ===
    bayesian_results = {}
    best_models = {}
    
    for name, config in all_models.items():
        print(f"\n{'='*80}")
        print(f"=== OPTIMISATION BAYÉSIENNE {name} ===")
        print(f"{'='*80}")
        
        # Nombre d'itérations selon le modèle
        if 'LogisticRegression' in name:
            n_calls = 50  # Moins d'itérations pour LogisticRegression
        else:
            n_calls = 80  # Plus d'itérations pour modèles complexes
        
        print(f"Nombre d'itérations bayésiennes: {n_calls}")
        print("🔄 Lancement BayesSearchCV...")
        
        # BayesSearchCV
        bayes_search = BayesSearchCV(
            estimator=config['model'],
            search_spaces=config['search_space'],
            scoring='recall',
            cv=4,
            n_iter=n_calls,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        try:
            # Lancement de l'optimisation bayésienne
            bayes_search.fit(X_train_sel, y_train)
            
            # Récupération du meilleur modèle
            best_model = bayes_search.best_estimator_
            best_models[name] = best_model
            
            # Prédictions
            y_pred = best_model.predict(X_test_sel)
            
            # Métriques
            cm = confusion_matrix(y_test, y_pred)
            recall_probleme = cm[0,0] / (cm[0,0] + cm[0,1])
            precision_probleme = cm[0,0] / (cm[0,0] + cm[1,0]) if (cm[0,0] + cm[1,0]) > 0 else 0
            fausses_alertes = cm[1,0]
            f1_probleme = 2 * (precision_probleme * recall_probleme) / (precision_probleme + recall_probleme) if (precision_probleme + recall_probleme) > 0 else 0
            
            bayesian_results[name] = {
                'best_score': bayes_search.best_score_,
                'best_params': bayes_search.best_params_,
                'recall_probleme': recall_probleme,
                'precision_probleme': precision_probleme,
                'fausses_alertes': fausses_alertes,
                'f1_probleme': f1_probleme,
                'cv_results': bayes_search.cv_results_
            }
            
            print(f"✅ {name} TERMINÉ !")
            print(f"Meilleur score CV: {bayes_search.best_score_:.4f}")
            print(f"Recall test: {recall_probleme:.4f}")
            print(f"Precision test: {precision_probleme:.4f}")
            print(f"Meilleurs paramètres: {bayes_search.best_params_}")
            
        except Exception as e:
            print(f"❌ Erreur avec {name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # === SÉLECTION DU MEILLEUR LOGISTIC REGRESSION ===
    logistic_results = {k: v for k, v in bayesian_results.items() if 'LogisticRegression' in k}
    if logistic_results:
        best_logistic = max(logistic_results.items(), key=lambda x: x[1]['recall_probleme'])
        print(f"\n🏆 MEILLEUR LOGISTIC REGRESSION: {best_logistic[0]}")
        print(f"Recall: {best_logistic[1]['recall_probleme']:.4f}")
        
        # Garder seulement le meilleur LogisticRegression
        for k in list(bayesian_results.keys()):
            if 'LogisticRegression' in k and k != best_logistic[0]:
                del bayesian_results[k]
                del best_models[k]
        
        # Renommer le meilleur
        bayesian_results['LogisticRegression'] = bayesian_results.pop(best_logistic[0])
        best_models['LogisticRegression'] = best_models.pop(best_logistic[0])
    
    # === COMPARAISON FINALE ===
    print(f"\n{'='*100}")
    print("=== COMPARAISON FINALE - OPTIMISATION BAYÉSIENNE ===")
    print(f"{'='*100}")
    
    if bayesian_results:
        comparison_df = pd.DataFrame(bayesian_results).T
        comparison_df = comparison_df.sort_values('recall_probleme', ascending=False)
        
        print("\n📊 CLASSEMENT PAR RECALL PROBLÈME:")
        print(comparison_df[['recall_probleme', 'precision_probleme', 'fausses_alertes', 'f1_probleme', 'best_score']].round(4))
        
        # Recommandations
        best_recall = comparison_df.index[0]
        best_precision = comparison_df.sort_values('precision_probleme', ascending=False).index[0]
        best_f1 = comparison_df.sort_values('f1_probleme', ascending=False).index[0]
        
        print(f"\n🏆 CHAMPIONS BAYÉSIENS:")
        print(f"🎯 Meilleur RECALL: {best_recall} - {comparison_df.loc[best_recall, 'recall_probleme']:.4f}")
        print(f"🔍 Meilleure PRECISION: {best_precision} - {comparison_df.loc[best_precision, 'precision_probleme']:.4f}")
        print(f"⚖️  Meilleur F1: {best_f1} - {comparison_df.loc[best_f1, 'f1_probleme']:.4f}")
        
        # Comparaison avec RandomizedSearchCV
        print(f"\n📈 COMPARAISON BAYÉSIEN vs RANDOM:")
        print(f"RandomizedSearchCV meilleur recall: 0.5436")
        best_bayesian_recall = comparison_df.loc[best_recall, 'recall_probleme']
        improvement = best_bayesian_recall - 0.5436
        print(f"Bayésien meilleur recall: {best_bayesian_recall:.4f}")
        print(f"Amélioration: {improvement:+.4f} ({improvement/0.5436*100:+.2f}%)")
    
    return bayesian_results, best_models

# === LANCEMENT DE L'OPTIMISATION BAYÉSIENNE ===
print("🔬 LANCEMENT DE L'OPTIMISATION BAYÉSIENNE")
print("🎯 Recherche intelligente des hyperparamètres optimaux")
print("⚡ Plus efficace que RandomizedSearchCV")

bayesian_results, bayesian_models = optimize_bayesian_approach()
```

## 🔬 **Avantages de l'optimisation bayésienne :**

1. **Plus efficace** : Explore intelligemment l'espace des paramètres
2. **Converge plus vite** : Utilise l'historique des essais précédents
3. **Moins d'itérations** : 50-80 itérations vs 150-250 en random
4. **Meilleure optimisation** : Souvent trouve de meilleurs paramètres

## 🎯 **Attentes :**
- Amélioration potentielle de 2-5% du recall
- Découverte de paramètres plus fins
- Temps de calcul réduit

**Installez scikit-optimize si nécessaire :**
```bash
pip install scikit-optimize
```

Lancez et montrez-moi si l'approche bayésienne surpasse votre RandomizedSearchCV ! 🚀

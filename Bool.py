L'utilisateur veut essayer l'optimisation bayésienne pour les hyperparamètres. C'est une approche très intelligente car l'optimisation bayésienne est généralement plus efficace que RandomizedSearchCV pour trouver les hyperparamètres optimaux, surtout avec un nombre limité d'évaluations.

L'optimisation bayésienne utilise des modèles probabilistes (généralement des processus gaussiens) pour modéliser la fonction objective et choisir intelligemment les prochains hyperparamètres à tester, au lieu de les choisir aléatoirement comme dans RandomizedSearchCV.

Les bibliothèques populaires pour cela sont :
- scikit-optimize (skopt) - la plus simple et intégrée avec scikit-learn
- Optuna - très performante et moderne
- Hyperopt - plus ancienne mais solide
- BayesSearchCV de scikit-optimize

Je vais proposer une solution avec scikit-optimize car c'est la plus simple à intégrer avec leur code existant, et aussi Optuna qui est très populaire et performante.

Il faut adapter leur fonction existante pour utiliser BayesSearchCV au lieu de RandomizedSearchCV.

Excellente idée ! 🎯 L'optimisation bayésienne est **beaucoup plus intelligente** que RandomizedSearchCV - elle apprend des évaluations précédentes pour choisir les hyperparamètres les plus prometteurs.

## 🧠 **Pourquoi l'optimisation bayésienne peut être plus efficace :**
- **Apprentissage adaptatif** : Utilise les résultats précédents pour guider la recherche
- **Exploration intelligente** : Balance exploration/exploitation
- **Moins d'évaluations** : Converge plus rapidement vers l'optimum

Voici la fonction adaptée avec **deux approches** :

```python
def optimize_bayesian_approach():
    """Optimisation bayésienne des hyperparamètres"""
    
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
    
    # === APPROCHE 1: SCIKIT-OPTIMIZE (BayesSearchCV) ===
    try:
        from skopt import BayesSearchCV
        from skopt.space import Real, Integer, Categorical
        skopt_available = True
        print("✅ Scikit-optimize disponible")
    except ImportError:
        print("❌ Scikit-optimize non disponible - installer: pip install scikit-optimize")
        skopt_available = False
    
    # === APPROCHE 2: OPTUNA ===
    try:
        import optuna
        optuna_available = True
        print("✅ Optuna disponible")
    except ImportError:
        print("❌ Optuna non disponible - installer: pip install optuna")
        optuna_available = False
    
    if not skopt_available and not optuna_available:
        print("❌ Aucune bibliothèque d'optimisation bayésienne disponible")
        return None, None
    
    # === PREPROCESSING ===
    selector = SelectKBest(score_func=f_classif, k=5)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    selected_features = X_train.columns[selector.get_support()]
    X_train_sel = pd.DataFrame(X_train_selected, columns=selected_features)
    X_test_sel = pd.DataFrame(X_test_selected, columns=selected_features)
    
    print(f"Features sélectionnées (5): {list(selected_features)}")
    
    # === FONCTION POUR SCIKIT-OPTIMIZE ===
    def optimize_with_skopt():
        """Optimisation avec BayesSearchCV"""
        if not skopt_available:
            return {}
        
        # Espaces de recherche pour BayesSearchCV
        search_spaces = {
            'LogisticRegression': {
                'model': LogisticRegression(class_weight='balanced', random_state=42),
                'search_space': {
                    'C': Real(1e-4, 1e4, prior='log-uniform'),
                    'penalty': Categorical(['l1', 'l2', 'elasticnet', None]),
                    'solver': Categorical(['liblinear', 'saga', 'lbfgs']),
                    'max_iter': Integer(500, 3000),
                    'l1_ratio': Real(0.1, 0.9)  # Sera ignoré si pas elasticnet
                }
            },
            'XGBoost': {
                'model': XGBClassifier(
                    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                    random_state=42,
                    eval_metric='logloss'
                ),
                'search_space': {
                    'n_estimators': Integer(50, 1000),
                    'max_depth': Integer(3, 10),
                    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                    'subsample': Real(0.6, 1.0),
                    'colsample_bytree': Real(0.6, 1.0),
                    'reg_alpha': Real(0, 5),
                    'reg_lambda': Real(0, 5),
                    'min_child_weight': Integer(1, 10),
                    'gamma': Real(0, 1)
                }
            },
            'LightGBM': {
                'model': LGBMClassifier(
                    class_weight='balanced',
                    random_state=42,
                    verbose=-1
                ),
                'search_space': {
                    'n_estimators': Integer(50, 1000),
                    'max_depth': Integer(3, 10),
                    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                    'num_leaves': Integer(10, 300),
                    'min_data_in_leaf': Integer(5, 100),
                    'feature_fraction': Real(0.5, 1.0),
                    'bagging_fraction': Real(0.5, 1.0),
                    'bagging_freq': Integer(0, 10),
                    'reg_alpha': Real(0, 5),
                    'reg_lambda': Real(0, 5)
                }
            },
            'BalancedRF': {
                'model': BalancedRandomForestClassifier(random_state=42),
                'search_space': {
                    'n_estimators': Integer(50, 1000),
                    'max_depth': Integer(3, 30),
                    'min_samples_split': Integer(2, 30),
                    'min_samples_leaf': Integer(1, 20),
                    'sampling_strategy': Categorical(['auto', 'majority', 'not minority']),
                    'bootstrap': Categorical([True, False])
                }
            }
        }
        
        skopt_results = {}
        
        for name, config in search_spaces.items():
            print(f"\n{'='*70}")
            print(f"=== OPTIMISATION BAYÉSIENNE {name} (scikit-optimize) ===")
            print(f"{'='*70}")
            
            try:
                # BayesSearchCV avec acquisition function
                bayes_search = BayesSearchCV(
                    estimator=config['model'],
                    search_spaces=config['search_space'],
                    scoring='recall',
                    cv=4,
                    n_iter=100,  # Moins d'itérations car plus efficace
                    random_state=42,
                    n_jobs=-1,
                    verbose=1
                )
                
                # Optimisation
                bayes_search.fit(X_train_sel, y_train)
                
                # Évaluation
                best_model = bayes_search.best_estimator_
                y_pred = best_model.predict(X_test_sel)
                
                # Métriques
                cm = confusion_matrix(y_test, y_pred)
                recall_probleme = cm[0,0] / (cm[0,0] + cm[0,1])
                precision_probleme = cm[0,0] / (cm[0,0] + cm[1,0]) if (cm[0,0] + cm[1,0]) > 0 else 0
                fausses_alertes = cm[1,0]
                f1_probleme = 2 * (precision_probleme * recall_probleme) / (precision_probleme + recall_probleme) if (precision_probleme + recall_probleme) > 0 else 0
                
                skopt_results[name] = {
                    'best_score': bayes_search.best_score_,
                    'best_params': bayes_search.best_params_,
                    'recall_probleme': recall_probleme,
                    'precision_probleme': precision_probleme,
                    'fausses_alertes': fausses_alertes,
                    'f1_probleme': f1_probleme,
                    'model': best_model
                }
                
                print(f"✅ {name} TERMINÉ !")
                print(f"Meilleur score CV: {bayes_search.best_score_:.4f}")
                print(f"Recall test: {recall_probleme:.4f}")
                print(f"Precision test: {precision_probleme:.4f}")
                print(f"Meilleurs paramètres: {bayes_search.best_params_}")
                
            except Exception as e:
                print(f"❌ Erreur avec {name}: {e}")
                continue
        
        return skopt_results
    
    # === FONCTION POUR OPTUNA ===
    def optimize_with_optuna():
        """Optimisation avec Optuna"""
        if not optuna_available:
            return {}
        
        # Supprimer les logs Optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        optuna_results = {}
        
        # Fonction objective pour XGBoost (exemple)
        def objective_xgboost(trial):
            # Suggérer les hyperparamètres
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 1),
                'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
                'random_state': 42,
                'eval_metric': 'logloss'
            }
            
            # Modèle avec les paramètres suggérés
            model = XGBClassifier(**params)
            
            # Validation croisée
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(model, X_train_sel, y_train, cv=4, scoring='recall')
            
            return cv_scores.mean()
        
        # Optimisation XGBoost avec Optuna
        print(f"\n{'='*70}")
        print(f"=== OPTIMISATION BAYÉSIENNE XGBoost (Optuna) ===")
        print(f"{'='*70}")
        
        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective_xgboost, n_trials=100)
            
            # Meilleur modèle
            best_params = study.best_params
            best_xgb = XGBClassifier(**best_params)
            best_xgb.fit(X_train_sel, y_train)
            
            # Évaluation
            y_pred = best_xgb.predict(X_test_sel)
            cm = confusion_matrix(y_test, y_pred)
            recall_probleme = cm[0,0] / (cm[0,0] + cm[0,1])
            precision_probleme = cm[0,0] / (cm[0,0] + cm[1,0]) if (cm[0,0] + cm[1,0]) > 0 else 0
            fausses_alertes = cm[1,0]
            f1_probleme = 2 * (precision_probleme * recall_probleme) / (precision_probleme + recall_probleme) if (precision_probleme + recall_probleme) > 0 else 0
            
            optuna_results['XGBoost_Optuna'] = {
                'best_score': study.best_value,
                'best_params': best_params,
                'recall_probleme': recall_probleme,
                'precision_probleme': precision_probleme,
                'fausses_alertes': fausses_alertes,
                'f1_probleme': f1_probleme,
                'model': best_xgb
            }
            
            print(f"✅ XGBoost Optuna TERMINÉ !")
            print(f"Meilleur score CV: {study.best_value:.4f}")
            print(f"Recall test: {recall_probleme:.4f}")
            print(f"Precision test: {precision_probleme:.4f}")
            print(f"Meilleurs paramètres: {best_params}")
            
        except Exception as e:
            print(f"❌ Erreur avec Optuna: {e}")
        
        return optuna_results
    
    # === LANCEMENT DES OPTIMISATIONS ===
    print("🚀 DÉMARRAGE OPTIMISATION BAYÉSIENNE")
    
    # Scikit-optimize
    skopt_results = optimize_with_skopt()
    
    # Optuna (juste XGBoost pour l'exemple)
    optuna_results = optimize_with_optuna()
    
    # Fusion des résultats
    all_results = {**skopt_results, **optuna_results}
    
    # === COMPARAISON FINALE ===
    if all_results:
        print(f"\n{'='*100}")
        print("=== COMPARAISON OPTIMISATION BAYÉSIENNE ===")
        print(f"{'='*100}")
        
        comparison_df = pd.DataFrame(all_results).T
        comparison_df = comparison_df.sort_values('recall_probleme', ascending=False)
        
        print("\n📊 RÉSULTATS BAYÉSIENS:")
        print(comparison_df[['recall_probleme', 'precision_probleme', 'fausses_alertes', 'f1_probleme', 'best_score']].round(4))
        
        # Comparaison avec RandomizedSearchCV
        print(f"\n📈 COMPARAISON vs RandomizedSearchCV:")
        print(f"RandomizedSearchCV meilleur recall: 0.5436")
        if len(comparison_df) > 0:
            bayes_best = comparison_df.iloc[0]['recall_probleme']
            improvement = bayes_best - 0.5436
            print(f"Bayésien meilleur recall: {bayes_best:.4f}")
            print(f"Amélioration: {improvement:+.4f} ({improvement/0.5436*100:+.2f}%)")
    
    return all_results

# === INSTALLATION DES DÉPENDANCES ===
print("📦 Pour utiliser cette fonction, installez:")
print("pip install scikit-optimize")
print("pip install optuna")
print("\nPuis lancez: optimize_bayesian_approach()")

# Lancement
# results_bayesian = optimize_bayesian_approach()
```

## 🎯 **Avantages de l'optimisation bayésienne :**

1. **Plus efficace** : Converge plus rapidement vers l'optimum
2. **Moins d'évaluations** : 100 itérations vs 250 en RandomizedSearchCV
3. **Apprentissage intelligent** : Utilise les résultats précédents
4. **Acquisition functions** : Balance exploration/exploitation

**Installez les dépendances et lancez !** L'optimisation bayésienne pourrait nous donner ce gain de performance qu'on cherche 🚀

L'utilisateur veut la fonction target business avec le code complet, et il veut 150 itérations pour tous les modèles. Je dois donc reprendre le code complet et intégrer la fonction target business personnalisée.

Je vais :
1. Définir la fonction target business
2. Créer le scorer personnalisé
3. Remettre le code complet avec 150 itérations pour tous
4. Utiliser le scorer personnalisé dans BayesSearchCV

Voici le code complet avec la fonction target business et 150 itérations pour tous :

```python
def optimize_bayesian_business_target():
    """Optimisation bayésienne avec fonction target business"""
    
    # === IMPORTS ===
    global pd
    import numpy as np
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.metrics import confusion_matrix, classification_report, make_scorer
    from sklearn.model_selection import cross_val_predict
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
    
    # === FONCTION TARGET BUSINESS ===
    def business_target_function(estimator, X, y):
        """Optimise selon critères business : recall avec contrainte fausses alertes"""
        try:
            y_pred = cross_val_predict(estimator, X, y, cv=4)
            cm = confusion_matrix(y, y_pred)
            
            if cm.shape == (2,2):
                recall_probleme = cm[0,0] / (cm[0,0] + cm[0,1])
                fausses_alertes = cm[1,0]
                
                # Objectif business : maximiser recall mais minimiser fausses alertes
                # Pénalité progressive pour trop de fausses alertes
                if fausses_alertes > 1500:
                    penalty = 0.3
                elif fausses_alertes > 1000:
                    penalty = 0.1
                else:
                    penalty = 0
                    
                return recall_probleme * (1 - penalty)
            else:
                return 0
        except Exception as e:
            print(f"Erreur dans business_target_function: {e}")
            return 0
    
    # Créer le scorer personnalisé
    business_scorer = make_scorer(business_target_function, greater_is_better=True)
    
    # === PREPROCESSING ===
    selector = SelectKBest(score_func=f_classif, k=5)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    selected_features = X_train.columns[selector.get_support()]
    X_train_sel = pd.DataFrame(X_train_selected, columns=selected_features)
    X_test_sel = pd.DataFrame(X_test_selected, columns=selected_features)
    
    print(f"Features sélectionnées (5): {list(selected_features)}")
    
    # === ESPACES DE RECHERCHE BAYÉSIENNE ===
    
    # 1. LogisticRegression - 4 MODÈLES SÉPARÉS
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
        print(f"=== OPTIMISATION BAYÉSIENNE BUSINESS {name} ===")
        print(f"{'='*80}")
        
        # 150 itérations pour tous comme demandé
        n_calls = 150
        
        print(f"Nombre d'itérations bayésiennes: {n_calls}")
        print("🎯 Fonction target : Recall avec pénalité fausses alertes")
        print("🔄 Lancement BayesSearchCV...")
        
        # BayesSearchCV avec fonction target business
        bayes_search = BayesSearchCV(
            estimator=config['model'],
            search_spaces=config['search_space'],
            scoring=business_scorer,  # Fonction target business
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
            
            # Métriques détaillées
            cm = confusion_matrix(y_test, y_pred)
            recall_probleme = cm[0,0] / (cm[0,0] + cm[0,1])
            precision_probleme = cm[0,0] / (cm[0,0] + cm[1,0]) if (cm[0,0] + cm[1,0]) > 0 else 0
            fausses_alertes = cm[1,0]
            f1_probleme = 2 * (precision_probleme * recall_probleme) / (precision_probleme + recall_probleme) if (precision_probleme + recall_probleme) > 0 else 0
            
            # Calcul du score business
            business_score = recall_probleme
            if fausses_alertes > 1500:
                business_score *= 0.7
            elif fausses_alertes > 1000:
                business_score *= 0.9
            
            bayesian_results[name] = {
                'best_score': bayes_search.best_score_,
                'best_params': bayes_search.best_params_,
                'recall_probleme': recall_probleme,
                'precision_probleme': precision_probleme,
                'fausses_alertes': fausses_alertes,
                'f1_probleme': f1_probleme,
                'business_score': business_score
            }
            
            print(f"✅ {name} TERMINÉ !")
            print(f"Meilleur score CV (business): {bayes_search.best_score_:.4f}")
            print(f"Recall test: {recall_probleme:.4f}")
            print(f"Precision test: {precision_probleme:.4f}")
            print(f"Fausses alertes: {fausses_alertes}")
            print(f"Score business: {business_score:.4f}")
            print(f"Meilleurs paramètres: {bayes_search.best_params_}")
            
        except Exception as e:
            print(f"❌ Erreur avec {name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # === SÉLECTION DU MEILLEUR LOGISTIC REGRESSION ===
    logistic_results = {k: v for k, v in bayesian_results.items() if 'LogisticRegression' in k}
    if logistic_results:
        best_logistic = max(logistic_results.items(), key=lambda x: x[1]['business_score'])
        print(f"\n🏆 MEILLEUR LOGISTIC REGRESSION: {best_logistic[0]}")
        print(f"Business Score: {best_logistic[1]['business_score']:.4f}")
        print(f"Recall: {best_logistic[1]['recall_probleme']:.4f}")
        print(f"Fausses alertes: {best_logistic[1]['fausses_alertes']}")
        
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
    print("=== COMPARAISON FINALE - OPTIMISATION BAYÉSIENNE BUSINESS ===")
    print(f"{'='*100}")
    
    if bayesian_results:
        comparison_df = pd.DataFrame(bayesian_results).T
        comparison_df = comparison_df.sort_values('business_score', ascending=False)
        
        print("\n📊 CLASSEMENT PAR BUSINESS SCORE:")
        print(comparison_df[['business_score', 'recall_probleme', 'precision_probleme', 'fausses_alertes', 'f1_probleme']].round(4))
        
        # Recommandations
        best_business = comparison_df.index[0]
        best_recall = comparison_df.sort_values('recall_probleme', ascending=False).index[0]
        best_precision = comparison_df.sort_values('precision_probleme', ascending=False).index[0]
        
        print(f"\n🏆 CHAMPIONS BUSINESS:")
        print(f"🎯 Meilleur BUSINESS SCORE: {best_business} - {comparison_df.loc[best_business, 'business_score']:.4f}")
        print(f"📈 Meilleur RECALL: {best_recall} - {comparison_df.loc[best_recall, 'recall_probleme']:.4f}")
        print(f"🔍 Meilleure PRECISION: {best_precision} - {comparison_df.loc[best_precision, 'precision_probleme']:.4f}")
        
        # Comparaison avec approches précédentes
        print(f"\n📈 ÉVOLUTION DES PERFORMANCES:")
        print(f"Baseline original: 0.5340")
        print(f"RandomizedSearchCV: 0.5436")
        best_business_score = comparison_df.loc[best_business, 'business_score']
        best_recall_score = comparison_df.loc[best_business, 'recall_probleme']
        print(f"Bayésien Business: {best_business_score:.4f} (recall: {best_recall_score:.4f})")
        
        improvement_vs_baseline = best_recall_score - 0.5340
        improvement_vs_random = best_recall_score - 0.5436
        print(f"Gain vs baseline: {improvement_vs_baseline:+.4f} ({improvement_vs_baseline/0.5340*100:+.2f}%)")
        print(f"Gain vs RandomizedSearchCV: {improvement_vs_random:+.4f} ({improvement_vs_random/0.5436*100:+.2f}%)")
        
        # Analyse des fausses alertes
        print(f"\n🚨 ANALYSE FAUSSES ALERTES:")
        for model in comparison_df.index:
            fausses = comparison_df.loc[model, 'fausses_alertes']
            if fausses > 1500:
                status = "🔴 TRÈS ÉLEVÉ"
            elif fausses > 1000:
                status = "🟠 ÉLEVÉ"
            else:
                status = "🟢 ACCEPTABLE"
            print(f"{model}: {fausses} fausses alertes {status}")
    
    return bayesian_results, best_models

# === LANCEMENT DE L'OPTIMISATION BAYÉSIENNE BUSINESS ===
print("🎯 LANCEMENT DE L'OPTIMISATION BAYÉSIENNE BUSINESS")
print("📊 Fonction target: Recall avec pénalité sur fausses alertes")
print("🔄 150 itérations pour tous les modèles")
print("⏰ Cela peut prendre du temps...")

bayesian_business_results, bayesian_business_models = optimize_bayesian_business_target()
```

## 🎯 **Fonction Target Business expliquée :**

**Objectif :** Maximiser le recall tout en pénalisant les fausses alertes excessives

**Pénalités :**
- **> 1500 fausses alertes** : -30% de score
- **> 1000 fausses alertes** : -10% de score  
- **≤ 1000 fausses alertes** : Pas de pénalité

**Avantage :** Trouve le meilleur équilibre détection/fausses alertes selon vos critères business ! 🚀

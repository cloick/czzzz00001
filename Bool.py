L'utilisateur a une erreur NameError pour 'X_train_del'. En regardant le code que j'ai fourni, je vois que j'ai utilisé X_train_sel et X_test_sel dans la fonction debug, mais il se peut que ces variables ne soient pas définies dans le scope global ou qu'il y ait une typo.

Je dois corriger la fonction debug en m'assurant que:
1. J'utilise les bonnes variables 
2. Je refais le preprocessing si nécessaire
3. Je corrige toute erreur de typo

Je vois que dans les fonctions précédentes, nous utilisions X_train_sel et X_test_sel après le SelectKBest. Il faut que je m'assure que ces variables sont bien définies ou que je les recrée dans la fonction debug.

**Ah ! Erreur de variable.** Voici la version corrigée avec le preprocessing inclus :

```python
def debug_threshold_inconsistency():
    """Identifier l'origine de l'incohérence - Version corrigée"""
    
    # === REFAIRE LE PREPROCESSING POUR ÊTRE SÛR ===
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.metrics import confusion_matrix, classification_report
    import numpy as np
    
    selector = SelectKBest(score_func=f_classif, k=5)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    selected_features = X_train.columns[selector.get_support()]
    X_train_sel = pd.DataFrame(X_train_selected, columns=selected_features)
    X_test_sel = pd.DataFrame(X_test_selected, columns=selected_features)
    
    print(f"Features sélectionnées: {list(selected_features)}")
    print(f"Shape train: {X_train_sel.shape}, Shape test: {X_test_sel.shape}")
    
    # 1. VÉRIFIER LE MODÈLE UTILISÉ
    print("\n=== VÉRIFICATION DU MODÈLE ===")
    model = bayesian_models['LogisticRegression']  # Modèle optimisé
    print(f"Paramètres du modèle: {model.get_params()}")
    
    # 2. REPRODUIRE LE RÉSULTAT D'OPTIMISATION
    print("\n=== ENTRAÎNEMENT ET PRÉDICTION ===")
    model.fit(X_train_sel, y_train)
    y_pred_default = model.predict(X_test_sel)
    
    cm_default = confusion_matrix(y_test, y_pred_default)
    print(f"Matrice confusion predict() défaut: \n{cm_default}")
    
    # 3. VÉRIFIER LES MÉTRIQUES AVEC SEUIL 0.5 MANUEL
    print("\n=== PRÉDICTION AVEC SEUIL 0.5 MANUEL ===")
    y_proba = model.predict_proba(X_test_sel)
    print(f"Shape probabilités: {y_proba.shape}")
    print(f"Premiers exemples proba: \n{y_proba[:5]}")
    
    # Proba de la classe 0 (problème)
    y_proba_probleme = y_proba[:, 0]  
    y_pred_05 = (y_proba_probleme >= 0.5).astype(int)
    
    cm_05 = confusion_matrix(y_test, y_pred_05)
    print(f"Matrice confusion seuil 0.5 manuel: \n{cm_05}")
    
    # 4. CALCULER MÉTRIQUES DES DEUX FAÇONS
    print("\n=== MÉTRIQUES PREDICT() DEFAULT ===")
    print(classification_report(y_test, y_pred_default))
    
    print("\n=== MÉTRIQUES SEUIL 0.5 MANUEL ===")
    print(classification_report(y_test, y_pred_05))
    
    # 5. VÉRIFIER SI MÊME RÉSULTAT
    are_same = np.array_equal(y_pred_default, y_pred_05)
    print(f"\n=== COMPARAISON ===")
    print(f"Prédictions identiques: {are_same}")
    
    if not are_same:
        diff_count = np.sum(y_pred_default != y_pred_05)
        print(f"Nombre de différences: {diff_count}")
        
        # Analyser les différences
        print(f"Défaut - 0: {np.sum(y_pred_default == 0)}, 1: {np.sum(y_pred_default == 1)}")
        print(f"Manuel - 0: {np.sum(y_pred_05 == 0)}, 1: {np.sum(y_pred_05 == 1)}")
    
    # 6. CALCULER MANUELLEMENT LES MÉTRIQUES COMME DANS L'OPTIMISATION
    print("\n=== CALCUL MANUEL MÉTRIQUES (STYLE OPTIMISATION) ===")
    tn, fp, fn, tp = cm_default.ravel()
    
    # Métriques pour classe 0 (problème)
    recall_probleme = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision_probleme = tn / (tn + fn) if (tn + fn) > 0 else 0
    fausses_alertes = fn
    
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    print(f"Recall problème: {recall_probleme:.4f}")
    print(f"Precision problème: {precision_probleme:.4f}")  
    print(f"Fausses alertes: {fausses_alertes}")
    
    return model, y_pred_default, y_pred_05, y_proba

# Lancer le debug
print("🔍 DEBUG DE L'INCOHÉRENCE")
model, pred_default, pred_05, proba = debug_threshold_inconsistency()
```

**Cette version :**
1. **Recrée le preprocessing** pour éviter les erreurs de variables
2. **Compare predict() vs seuil manuel**
3. **Affiche les matrices de confusion** des deux méthodes
4. **Calcule manuellement** les métriques comme dans l'optimisation

**Lancez ça et on va identifier d'où vient l'incohérence !** 🔍

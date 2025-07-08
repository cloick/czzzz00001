def evaluation(model, X_train_eval=None, y_train_eval=None, X_test_eval=None, y_test_eval=None):
    """Évaluation complète du modèle avec données optionnelles - Version binaire optimisée"""
    
    # Utiliser les données globales par défaut ou celles passées en paramètre
    X_train_use = X_train_eval if X_train_eval is not None else X_train
    y_train_use = y_train_eval if y_train_eval is not None else y_train
    X_test_use = X_test_eval if X_test_eval is not None else X_test
    y_test_use = y_test_eval if y_test_eval is not None else y_test
    
    # Entraînement
    model.fit(X_train_use, y_train_use)
    y_pred = model.predict(X_test_use)
    y_proba = model.predict_proba(X_test_use)[:, 1] if hasattr(model, 'predict_proba') else None
    
    print("=== MATRICE DE CONFUSION ===")
    cm = confusion_matrix(y_test_use, y_pred)
    print(cm)
    
    # Métriques détaillées pour problème déséquilibré
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"\nDétail matrice:")
        print(f"TN (Vrai Négatif - Problème détecté): {tn}")
        print(f"FP (Faux Positif - Fausse alerte): {fp}")
        print(f"FN (Faux Négatif - Problème raté): {fn}")
        print(f"TP (Vrai Positif - Succès détecté): {tp}")
        
        # Métriques business
        recall_probleme = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision_probleme = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall_succes = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_succes = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        print(f"\n🎯 MÉTRIQUES BUSINESS:")
        print(f"Recall Problème (détection des échecs): {recall_probleme:.3f}")
        print(f"Precision Problème (fiabilité alertes): {precision_probleme:.3f}")
        print(f"Recall Succès: {recall_succes:.3f}")
        print(f"Precision Succès: {precision_succes:.3f}")
        
        if y_proba is not None:
            auc_score = roc_auc_score(y_test_use, y_proba)
            print(f"ROC-AUC: {auc_score:.3f}")
    
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test_use, y_pred))
    
    # Learning curves
    print("\n=== LEARNING CURVES ===")
    N, train_score, val_score = learning_curve(
        model, X_train_use, y_train_use,
        cv=4, scoring='f1', 
        train_sizes=np.linspace(0.1, 1, 10)
    )
    
    plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), 'o-', label='train score')
    plt.plot(N, val_score.mean(axis=1), 's-', label='validation score')
    plt.fill_between(N, train_score.mean(axis=1) - train_score.std(axis=1),
                     train_score.mean(axis=1) + train_score.std(axis=1), alpha=0.2)
    plt.fill_between(N, val_score.mean(axis=1) - val_score.std(axis=1),
                     val_score.mean(axis=1) + val_score.std(axis=1), alpha=0.2)
    plt.legend()
    plt.title('Learning Curves (F1 Score)')
    plt.xlabel('Training examples')
    plt.ylabel('F1 Score')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Feature importance si disponible
    if hasattr(model, 'feature_importances_'):
        print("\n=== FEATURE IMPORTANCE ===")
        
        # Utiliser les bons noms de colonnes
        column_names = (X_train_use.columns if hasattr(X_train_use, 'columns') 
                       else [f'feature_{i}' for i in range(X_train_use.shape[1])])
        
        feature_imp = pd.DataFrame(
            model.feature_importances_,
            index=column_names,
            columns=['importance']
        ).sort_values('importance', ascending=False)
        
        # Afficher top 15 pour plus de détail
        print("\nTop 15 features:")
        print(feature_imp.head(15))
        
        # Graphique
        feature_imp.head(10).plot.bar(figsize=(12, 6))
        plt.title('Top 10 Feature Importances')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    return model


def test_feature_selection():
    """Tester différents nombres de features avec évaluation complète"""
    
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.metrics import f1_score
    
    k_values = [5, 10, 15, 20, 25, 30, 'all']
    results_summary = []
    
    for k in k_values:
        print(f"\n{'='*60}")
        print(f"=== TEST SelectKBest k={k} ===")
        print(f"{'='*60}")
        
        if k == 'all':
            # Version actuelle (baseline)
            print("🔄 Test avec toutes les features (baseline)")
            model = RandomForestClassifier(
                random_state=42,
                class_weight='balanced'
            )
            result_model = evaluation(model)  # Utilise les variables globales
            
            # Stocker les résultats pour comparaison
            y_pred = result_model.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            results_summary.append({'k': 'all', 'f1_score': f1, 'n_features': X_train.shape[1]})
            
        else:
            # SelectKBest
            print(f"🔄 Sélection des {k} meilleures features")
            selector = SelectKBest(score_func=f_classif, k=k)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            # Convertir en DataFrame pour garder les noms de colonnes
            selected_features = X_train.columns[selector.get_support()]
            X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features)
            X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features)
            
            print(f"Features sélectionnées ({k}): {list(selected_features)}")
            
            # Afficher les scores des features
            feature_scores = pd.DataFrame({
                'feature': X_train.columns,
                'score': selector.scores_
            }).sort_values('score', ascending=False)
            
            print(f"\nTop 10 scores F-statistique:")
            print(feature_scores.head(10))
            
            # Évaluation complète avec la fonction modifiée
            model = RandomForestClassifier(
                random_state=42,
                class_weight='balanced'
            )
            result_model = evaluation(model, X_train_selected, y_train, X_test_selected, y_test)
            
            # Stocker les résultats
            y_pred = result_model.predict(X_test_selected)
            f1 = f1_score(y_test, y_pred)
            results_summary.append({'k': k, 'f1_score': f1, 'n_features': k})
    
    # Résumé comparatif
    print(f"\n{'='*60}")
    print("=== RÉSUMÉ COMPARATIF ===")
    print(f"{'='*60}")
    
    results_df = pd.DataFrame(results_summary)
    print(results_df.sort_values('f1_score', ascending=False))
    
    # Graphique de comparaison
    plt.figure(figsize=(12, 6))
    x_labels = [str(x) for x in results_df['k']]
    plt.bar(x_labels, results_df['f1_score'])
    plt.title('Comparaison F1-Score selon le nombre de features')
    plt.xlabel('Nombre de features (k)')
    plt.ylabel('F1-Score')
    plt.xticks(rotation=45)
    
    # Ajouter les valeurs sur les barres
    for i, v in enumerate(results_df['f1_score']):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Recommandation
    best_k = results_df.loc[results_df['f1_score'].idxmax(), 'k']
    best_score = results_df['f1_score'].max()
    print(f"\n🏆 RECOMMANDATION: k={best_k} (F1-Score: {best_score:.3f})")
    
    return results_df


def quick_test_balanced():
    """Test rapide pour comparer avec et sans class_weight='balanced'"""
    
    print("🚀 COMPARAISON RAPIDE: avec/sans class_weight='balanced'")
    print("="*60)
    
    models = {
        'RF_standard': RandomForestClassifier(random_state=42),
        'RF_balanced': RandomForestClassifier(random_state=42, class_weight='balanced')
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Métriques clés
        cm = confusion_matrix(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            recall_probleme = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision_probleme = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            print(f"F1-Score: {f1:.3f}")
            print(f"Recall Problème: {recall_probleme:.3f}")
            print(f"Precision Problème: {precision_probleme:.3f}")
            
            results[name] = {
                'f1': f1,
                'recall_probleme': recall_probleme,
                'precision_probleme': precision_probleme
            }
    
    # Comparaison
    if len(results) == 2:
        rf_std = results['RF_standard']
        rf_bal = results['RF_balanced']
        
        print(f"\n🎯 AMÉLIORATION avec class_weight='balanced':")
        print(f"F1-Score: {rf_std['f1']:.3f} → {rf_bal['f1']:.3f} ({rf_bal['f1']-rf_std['f1']:+.3f})")
        print(f"Recall Problème: {rf_std['recall_probleme']:.3f} → {rf_bal['recall_probleme']:.3f} ({rf_bal['recall_probleme']-rf_std['recall_probleme']:+.3f})")
        print(f"Precision Problème: {rf_std['precision_probleme']:.3f} → {rf_bal['precision_probleme']:.3f} ({rf_bal['precision_probleme']-rf_std['precision_probleme']:+.3f})")
    
    return results


# Imports nécessaires (à ajouter en haut de votre notebook)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (confusion_matrix, classification_report, f1_score, 
                           roc_auc_score, learning_curve)
from sklearn.model_selection import train_test_split

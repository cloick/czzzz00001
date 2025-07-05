# MODÉLISATION
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt

# VERSION 1: Modèles simples pour baseline
model_1 = RandomForestClassifier(random_state=0)

# VERSION 2: Pipeline avancé (à tester plus tard)
# model_2 = make_pipeline(PolynomialFeatures(2), SelectKBest(f_classif, k=10),
#                       RandomForestClassifier(random_state=0))

# VERSION 3: Autres modèles à tester
# model_3 = DecisionTreeClassifier(random_state=0)
# model_4 = XGBClassifier(random_state=0)

# PROCÉDURE D'ÉVALUATION
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve

def evaluation(model):
    """Évaluation complète du modèle"""
    
    # Entraînement
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("=== MATRICE DE CONFUSION ===")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred))
    
    # Learning curves
    print("\n=== LEARNING CURVES ===")
    N, train_score, val_score = learning_curve(model, X_train, y_train,
                                              cv=4, scoring='f1_macro',  # f1_macro pour multiclasse
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
        feature_imp = pd.DataFrame(
            model.feature_importances_, 
            index=X_train.columns
        ).sort_values(0, ascending=False)
        
        feature_imp.head(10).plot.bar(figsize=(12, 6))
        plt.title('Top 10 Feature Importances')
        plt.show()
    
    return model

# TEST DU MODÈLE BASELINE
print("=== ÉVALUATION MODÈLE BASELINE ===")
evaluation(model_1)

SÉLECTION EXPERTE DE MODÈLES
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

# Preprocessors adaptés
selector = SelectKBest(f_classif, k=10)
scaler_selector = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=10))

# MODÈLES EXPERTS pour classification multiclasse déséquilibrée
dict_of_models = {
    'RandomForest': make_pipeline(selector, RandomForestClassifier(
        random_state=42, class_weight='balanced', n_estimators=200)),
    
    'XGBoost': make_pipeline(selector, XGBClassifier(
        random_state=42, eval_metric='mlogloss', verbosity=0)),
    
    'LightGBM': make_pipeline(selector, LGBMClassifier(
        random_state=42, class_weight='balanced', verbosity=-1)),
    
    'CatBoost': make_pipeline(selector, CatBoostClassifier(
        random_state=42, verbose=False, class_weights='Balanced')),
    
    'ExtraTrees': make_pipeline(selector, ExtraTreesClassifier(
        random_state=42, class_weight='balanced', n_estimators=200)),
    
    'LogisticReg': make_pipeline(scaler_selector, LogisticRegression(
        random_state=42, class_weight='balanced', max_iter=1000)),
    
    'SVM_RBF': make_pipeline(scaler_selector, SVC(
        random_state=42, class_weight='balanced', kernel='rbf', probability=True))
}

# Test expert
for name, model in dict_of_models.items():
    print(f"\n{'='*50}")
    print(f"=== ÉVALUATION {name} ===")
    print(f"{'='*50}")
    evaluation(model)

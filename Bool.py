MODÉLISATION COMPLÈTE SANS WARNINGS
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# PREPROCESSOR INTELLIGENT - Évite les warnings
preprocessor = make_pipeline(
    PolynomialFeatures(2, include_bias=False),  # ← Remis comme vous aviez !
    VarianceThreshold(threshold=0),             # ← Supprime features constantes
    SelectKBest(f_classif, k=10)               # ← Plus de warnings !
)

# SÉLECTION EXPERTE DE MODÈLES
dict_of_models = {
    'RandomForest': make_pipeline(preprocessor, 
                                 RandomForestClassifier(random_state=42, class_weight='balanced')),
    
    'XGBoost': make_pipeline(preprocessor,
                            XGBClassifier(random_state=42, eval_metric='mlogloss', verbosity=0)),
    
    'LightGBM': make_pipeline(preprocessor,
                             LGBMClassifier(random_state=42, class_weight='balanced', verbosity=-1)),
    
    'CatBoost': make_pipeline(preprocessor,
                             CatBoostClassifier(random_state=42, verbose=False, class_weight='balanced')),  # ← Corrigé !
    
    'ExtraTrees': make_pipeline(preprocessor,
                               ExtraTreesClassifier(random_state=42, class_weight='balanced')),
    
    'LogisticReg': make_pipeline(preprocessor, StandardScaler(),
                                LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)),
    
    'SVM_RBF': make_pipeline(preprocessor, StandardScaler(),
                            SVC(random_state=42, class_weight='balanced', probability=True))
}

# TEST SYSTÉMATIQUE
for name, model in dict_of_models.items():
    print(f"\n{'='*50}")
    print(f"=== ÉVALUATION {name} ===")
    print(f"{'='*50}")
    evaluation(model)

# Imports nécessaires
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np

print("=== MODÈLE BASELINE - RANDOM FOREST ===\n")

# Vérifier qu'on a bien tout en numérique
print(f"📊 DONNÉES PRÊTES:")
print(f"X_train_simple shape: {X_train_simple.shape}")
print(f"X_test_simple shape: {X_test_simple.shape}")
print(f"Types non-numériques train: {X_train_simple.select_dtypes(exclude=[np.number]).shape[1]}")
print(f"Types non-numériques test: {X_test_simple.select_dtypes(exclude=[np.number]).shape[1]}")

# Distribution des classes
print(f"\n📈 DISTRIBUTION DES CLASSES:")
print("Train:")
print(y_train.value_counts(normalize=True).round(3))
print("Test:")  
print(y_test.value_counts(normalize=True).round(3))🎯 ENTRAÎNEMENT DU MODÈLE :# 1. ENTRAÎNEMENT RANDOM FOREST
print("\n🤖 ENTRAÎNEMENT RANDOM FOREST...")

# Modèle simple avec paramètres basiques
rf_model = RandomForestClassifier(
    n_estimators=100,        # 100 arbres
    max_depth=10,           # Profondeur limitée pour éviter overfitting
    min_samples_split=50,   # Minimum échantillons pour split
    min_samples_leaf=20,    # Minimum échantillons par feuille
    random_state=42,        # Reproductibilité
    class_weight='balanced', # Gérer déséquilibre des classes
    n_jobs=-1              # Utiliser tous les CPU
)

# Fit sur les données d'entraînement
rf_model.fit(X_train_simple, y_train)
print("✅ Modèle entraîné !")

# 2. PRÉDICTIONS
print("\n🔮 PRÉDICTIONS...")
y_pred_train = rf_model.predict(X_train_simple)
y_pred_test = rf_model.predict(X_test_simple)

# Probabilités pour analyse
y_pred_proba_test = rf_model.predict_proba(X_test_simple)
print("✅ Prédictions générées !")📊 ÉVALUATION MULTI-CLASSES :# 3. MÉTRIQUES MULTI-CLASSES
print("\n=== ÉVALUATION MULTI-CLASSES ===")

# Accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"🎯 ACCURACY:")
print(f"  Train: {train_accuracy:.4f}")
print(f"  Test:  {test_accuracy:.4f}")
print(f"  Écart: {abs(train_accuracy - test_accuracy):.4f}")

# Métriques détaillées (macro average pour classes déséquilibrées)
test_precision_macro = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
test_recall_macro = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
test_f1_macro = f1_score(y_test, y_pred_test, average='macro', zero_division=0)

print(f"\n📊 MÉTRIQUES MACRO (égalité entre classes):")
print(f"  Precision: {test_precision_macro:.4f}")
print(f"  Recall:    {test_recall_macro:.4f}")
print(f"  F1-Score:  {test_f1_macro:.4f}")

# Métriques pondérées (par support de classe)
test_precision_weighted = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
test_recall_weighted = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
test_f1_weighted = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)

print(f"\n⚖️ MÉTRIQUES WEIGHTED (pondérées par classe):")
print(f"  Precision: {test_precision_weighted:.4f}")
print(f"  Recall:    {test_recall_weighted:.4f}")
print(f"  F1-Score:  {test_f1_weighted:.4f}")

# Rapport détaillé par classe
print(f"\n📋 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred_test))🎯 ÉVALUATION BINAIRE (Succès vs Problème) :# 4. ÉVALUATION BINAIRE - Plus business-friendly
print("\n=== ÉVALUATION BINAIRE (SUCCÈS vs PROBLÈME) ===")

# Créer target binaire
y_test_binary = (y_test == 'Succès').astype(int)
y_pred_binary = (y_pred_test == 'Succès').astype(int)

# Métriques binaires
binary_accuracy = accuracy_score(y_test_binary, y_pred_binary)
binary_precision = precision_score(y_test_binary, y_pred_binary)
binary_recall = recall_score(y_test_binary, y_pred_binary)
binary_f1 = f1_score(y_test_binary, y_pred_binary)

print(f"🎯 SUCCÈS vs PROBLÈME:")
print(f"  Accuracy:  {binary_accuracy:.4f}")
print(f"  Precision: {binary_precision:.4f} (des prédictions 'Succès', combien sont vraies)")
print(f"  Recall:    {binary_recall:.4f} (des vrais 'Succès', combien détectés)")
print(f"  F1-Score:  {binary_f1:.4f}")

# Matrice de confusion binaire
print(f"\n📊 MATRICE DE CONFUSION BINAIRE:")
conf_matrix_binary = confusion_matrix(y_test_binary, y_pred_binary)
print("           Pred_Problème  Pred_Succès")
print(f"Vrai_Problème      {conf_matrix_binary[0,0]:4d}        {conf_matrix_binary[0,1]:4d}")
print(f"Vrai_Succès        {conf_matrix_binary[1,0]:4d}        {conf_matrix_binary[1,1]:4d}")🔍 FEATURE IMPORTANCES :# 5. VARIABLES LES PLUS IMPORTANTES
print("\n=== TOP 10 VARIABLES IMPORTANTES ===")

# Récupérer importances
feature_names = X_train_simple.columns
importances = rf_model.feature_importances_

# Créer DataFrame et trier
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("🏆 TOP 10 FEATURES:")
for i, row in feature_importance_df.head(10).iterrows():
    print(f"{row['feature']:30s} | {row['importance']:.4f}")

print(f"\n✅ MODÈLE BASELINE ÉVALUÉ !")

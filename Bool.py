# Vérifier quelles colonnes restent non-encodées dans les versions SIMPLE
remaining_objects = X_train_simple.select_dtypes(include=['object']).columns.tolist()
print(f"Colonnes object restantes dans X_train_simple: {remaining_objects}")

# 1. SUPPRIMER number (identifiant inutile)
if 'number' in X_train_simple.columns:
    X_train_simple = X_train_simple.drop('number', axis=1)
    X_test_simple = X_test_simple.drop('number', axis=1)
    print("✅ 'number' supprimé")

# 2. GÉRER u_template_applied - Variable binaire
if 'u_template_applied' in X_train_simple.columns:
    # Créer variable binaire "has_template"
    X_train_simple['has_template'] = X_train_simple['u_template_applied'].notna().astype(int)
    X_test_simple['has_template'] = X_test_simple['u_template_applied'].notna().astype(int)
    
    # Supprimer originale
    X_train_simple = X_train_simple.drop('u_template_applied', axis=1)
    X_test_simple = X_test_simple.drop('u_template_applied', axis=1)
    
    print("✅ 'u_template_applied' → 'has_template' (binaire)")

# Vérification finale
remaining_objects_final = X_train_simple.select_dtypes(include=['object']).columns.tolist()
print(f"\n📊 VÉRIFICATION FINALE:")
print(f"Colonnes object restantes: {remaining_objects_final}")
print(f"Shape X_train_simple: {X_train_simple.shape}")
print(f"Shape X_test_simple: {X_test_simple.shape}")

if len(remaining_objects_final) == 0:
    print("🎉 TOUTES LES VARIABLES SONT NUMÉRIQUES DANS LA VERSION SIMPLE !")
else:
    print(f"⚠️ Colonnes à traiter encore: {remaining_objects_final}")

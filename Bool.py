# Convertir tous les booléens en 0/1
print("🔄 Conversion booléens en 0/1...")

for col in bool_cols:
    X_train_encoded[col] = X_train_encoded[col].astype(int)
    X_test_encoded[col] = X_test_encoded[col].astype(int)
    print(f"  ✅ {col}: bool → int")

# Vérification finale des types
print(f"\n📊 TYPES FINAUX:")
print("Numériques:", X_train_encoded.select_dtypes(include=[np.number]).shape[1])
print("Object:", X_train_encoded.select_dtypes(include=['object']).shape[1])
print("Booléens:", X_train_encoded.select_dtypes(include=['bool']).shape[1])

# S'assurer qu'il n'y a plus d'object ni de bool
remaining_non_numeric = X_train_encoded.select_dtypes(exclude=[np.number]).columns.tolist()
if remaining_non_numeric:
    print(f"⚠️ Colonnes non-numériques restantes: {remaining_non_numeric}")
else:
    print("✅ TOUTES LES VARIABLES SONT NUMÉRIQUES !")

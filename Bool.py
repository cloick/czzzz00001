from sklearn.model_selection import train_test_split

# Vérifier les proportions actuelles
print("Distribution target:")
print(df['dv_close_code'].value_counts(normalize=True))

# Features et target
X = df.drop(['dv_close_code'], axis=1)  # Toutes sauf target
y = df['dv_close_code']                 # Target multi-classes

# Train/Test split stratifié
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,           # 80/20 split
    random_state=42,         # Reproductibilité
    stratify=y              # Stratification sur target
)

print(f"\nTaille train: {X_train.shape[0]}")
print(f"Taille test: {X_test.shape[0]}")

# Vérifier que stratification a marché
print("\nDistribution train:")
print(y_train.value_counts(normalize=True))
print("\nDistribution test:")
print(y_test.value_counts(normalize=True))

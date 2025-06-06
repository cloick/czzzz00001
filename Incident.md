Parfait ! Maintenant adaptons le code pour accéder à votre structure de dossiers. Voici le code corrigé :

```python
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import numpy as np
import os
import torch
from transformers import CamembertTokenizer, CamembertModel

# Read recipe inputs
dataset = dataiku.Dataset("incident_prepared_encode_categorical")
df = dataset.get_dataframe()

# Accéder au folder "models"
folder = dataiku.Folder("models")
folder_path = folder.get_path()

# Le modèle est dans le sous-dossier camembert_model
model_path = os.path.join(folder_path, "camembert_model")

print(f"Chemin du folder : {folder_path}")
print(f"Chemin du modèle : {model_path}")

# Vérifier que les fichiers sont bien là
print("\nFichiers dans camembert_model:")
if os.path.exists(model_path):
    for file in os.listdir(model_path):
        print(f"  - {file}")
else:
    print("ERREUR: Le dossier camembert_model n'existe pas!")

# Charger le modèle et tokenizer
try:
    print("\nChargement du tokenizer...")
    tokenizer = CamembertTokenizer.from_pretrained(model_path, local_files_only=True)
    
    print("Chargement du modèle...")
    model = CamembertModel.from_pretrained(model_path, local_files_only=True)
    model.eval()  # Mettre en mode évaluation
    
    print("✅ Modèle chargé avec succès!")
    
except Exception as e:
    print(f"❌ Erreur lors du chargement : {e}")
    raise

# Fonction pour générer les embeddings
def generate_embeddings_batch(texts, batch_size=8):  # Réduit à 8 pour économiser la mémoire
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        try:
            # Tokenisation
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # Génération des embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                # Utiliser le token [CLS]
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            embeddings.extend(batch_embeddings)
            
            if i % 100 == 0:
                print(f"Traité {i}/{len(texts)} textes...")
                
        except Exception as e:
            print(f"Erreur sur le batch {i}: {e}")
            # Ajouter des embeddings vides en cas d'erreur
            embeddings.extend(np.zeros((len(batch), 768)))
    
    return np.array(embeddings)

# Préparer les textes
print("\nPréparation des textes...")
texts = df['Notes de résolution'].fillna('').astype(str).tolist()

# Limiter pour test initial
print(f"\nNombre total de textes : {len(texts)}")

# Pour tester, commencer avec un échantillon
TEST_MODE = True  # Mettre à False pour traiter tous les textes
if TEST_MODE:
    texts = texts[:100]  # Tester avec 100 textes d'abord
    print(f"Mode TEST : traitement de {len(texts)} textes seulement")

# Générer les embeddings
print("\nGénération des embeddings...")
embeddings = generate_embeddings_batch(texts)

print(f"✅ Embeddings générés : shape {embeddings.shape}")

# Si en mode test, créer un dataframe réduit
if TEST_MODE:
    df_output = df.head(100).copy()
else:
    df_output = df.copy()

# Ajouter les embeddings au dataframe
print("\nAjout des embeddings au dataframe...")
for i in range(min(embeddings.shape[1], 768)):  # CamemBERT = 768 dimensions
    df_output[f'embedding_{i}'] = embeddings[:, i]

print(f"Colonnes dans le dataframe final : {len(df_output.columns)}")

# Write recipe outputs
print("\nSauvegarde des résultats...")
output = dataiku.Dataset("incident_prepared_embeddings_creation")
output.write_with_schema(df_output)

print("✅ Terminé avec succès!")
```

## 🔧 Points importants :

1. **Mode TEST** : J'ai ajouté un mode test qui traite seulement 100 textes. Une fois que ça marche, mettez `TEST_MODE = False`

2. **Gestion mémoire** : J'ai réduit le batch_size à 8 pour éviter les problèmes de mémoire

3. **Gestion d'erreurs** : Le code gère les erreurs potentielles lors du traitement

4. **Vérification des fichiers** : Le code liste les fichiers pour vérifier qu'ils sont bien là

## 🚨 Si vous avez une erreur :

1. **Vérifiez les logs** pour voir quels fichiers sont détectés
2. **Vérifiez les permissions** sur le folder
3. **Essayez avec un batch_size encore plus petit** (4 ou 2) si problème de mémoire

Lancez d'abord en mode TEST pour vérifier que tout fonctionne !

Je vois que vous avez déjà connecté votre base de données et fait les filtres nécessaires. Maintenant, il faut adapter votre code Python pour Dataiku. Je vais vous guider étape par étape pour :

1. **Créer la variable `est_fiable`**
2. **Adapter votre code pour l'environnement Dataiku**

## 📝 Étape 1 : Créer la variable `est_fiable` dans Dataiku

### Option A : Via un Recipe Python
1. **Créez un Recipe Python** sur votre dataset filtré
2. **Code à utiliser** :

```python
import dataiku
import pandas as pd

# Charger les datasets
input_dataset = dataiku.Dataset("votre_dataset_filtré")  # Remplacez par le nom de votre dataset
df_metis = input_dataset.get_dataframe()

# Charger le fichier des tickets fiables (vous devez d'abord l'importer dans Dataiku)
fiables_dataset = dataiku.Dataset("gdp_tickets")  # Assurez-vous d'avoir importé ce fichier
df_fiables = fiables_dataset.get_dataframe()

# Créer la colonne est_fiable
df_metis['est_fiable'] = df_metis['N° INC'].isin(df_fiables['N° INC'])

# Écrire le résultat
output_dataset = dataiku.Dataset("metis_avec_fiable")
output_dataset.write_with_schema(df_metis)
```

### Option B : Via un Recipe Prepare (plus simple)
1. Dans votre **Recipe Prepare** existant, ajoutez un step :
2. **"Enrich with other datasets"** → **"Flag rows based on join"**
3. Configurez :
   - Join with dataset : `gdp_tickets` (vous devez l'avoir importé)
   - Join key : `N° INC`
   - Output column name : `est_fiable`

## 📊 Étape 2 : Adapter votre code complet pour Dataiku

Voici comment structurer votre code dans Dataiku en plusieurs recipes Python :

### **Recipe 1 : Préparation des données et embeddings**

```python
import dataiku
import pandas as pd
import numpy as np
import re
from transformers import CamembertTokenizer, CamembertModel
import torch

# Configuration
BATCH_SIZE = 16
MAX_LENGTH = 128

# Lecture du dataset
dataset = dataiku.Dataset("metis_avec_fiable")
df = dataset.get_dataframe()

# Fonction de nettoyage
def nettoyer_texte(texte):
    if not isinstance(texte, str):
        return ""
    texte = re.sub(r'[^\w\s]', ' ', texte.lower())
    texte = re.sub(r'\s+', ' ', texte).strip()
    return texte

# Nettoyer les textes
df['notes_resolution_nettoyees'] = df['Notes de résolution'].apply(nettoyer_texte)

# Charger CamemBERT
print("Chargement de CamemBERT...")
tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
model = CamembertModel.from_pretrained('camembert-base')
model.eval()

# Générer les embeddings par batch
def generer_embeddings_batch(textes, batch_size=BATCH_SIZE):
    embeddings = []
    
    for i in range(0, len(textes), batch_size):
        batch = textes[i:i+batch_size]
        
        # Tokenisation
        inputs = tokenizer(
            batch, 
            padding=True, 
            truncation=True, 
            max_length=MAX_LENGTH, 
            return_tensors='pt'
        )
        
        # Génération des embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Utiliser le token [CLS]
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        embeddings.extend(batch_embeddings)
        
        if i % 100 == 0:
            print(f"Traité {i}/{len(textes)} textes...")
    
    return np.array(embeddings)

print("Génération des embeddings...")
embeddings = generer_embeddings_batch(df['notes_resolution_nettoyees'].tolist())

# Sauvegarder les embeddings comme colonnes
for i in range(embeddings.shape[1]):
    df[f'embedding_{i}'] = embeddings[:, i]

# Écrire le résultat
output = dataiku.Dataset("metis_avec_embeddings")
output.write_with_schema(df)
```

### **Recipe 2 : Clustering avec HDBSCAN**

```python
import dataiku
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from umap import UMAP
import hdbscan

# Lire les données avec embeddings
dataset = dataiku.Dataset("metis_avec_embeddings")
df = dataset.get_dataframe()

# Récupérer les colonnes d'embeddings
embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
embeddings = df[embedding_cols].values

# Encoder les variables catégorielles
cat_vars = ['Priorité', 'Service métier', 'Cat1', 'Cat2', 'Groupe affecté']
cat_features = []

for col in cat_vars:
    if col in df.columns:
        le = LabelEncoder()
        encoded = le.fit_transform(df[col].fillna('INCONNU'))
        cat_features.append(encoded)

# Normaliser et combiner
scaler = StandardScaler()
cat_features = np.array(cat_features).T
cat_features_scaled = scaler.fit_transform(cat_features)

# Pondération des features catégorielles
weights = np.array([0.5, 2.0, 1.0, 1.0, 3.0])  # Priorité, Service, Cat1, Cat2, Groupe
cat_features_weighted = cat_features_scaled * weights

# Combiner embeddings et features catégorielles
features_combined = np.hstack([embeddings, cat_features_weighted])

# UMAP pour réduction dimensionnelle
print("Réduction UMAP...")
reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
embeddings_2d = reducer.fit_transform(features_combined)

# HDBSCAN avec vos paramètres optimaux
print("Clustering HDBSCAN...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=240,
    min_samples=20,
    cluster_selection_epsilon=1.56,
    metric='euclidean'
)

df['cluster'] = clusterer.fit_predict(embeddings_2d)

# Statistiques
n_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'].values else 0)
n_noise = (df['cluster'] == -1).sum()
print(f"Nombre de clusters: {n_clusters}")
print(f"Points de bruit: {n_noise} ({n_noise/len(df)*100:.2%})")

# Sauvegarder les coordonnées UMAP
df['umap_x'] = embeddings_2d[:, 0]
df['umap_y'] = embeddings_2d[:, 1]

# Écrire le résultat
output = dataiku.Dataset("metis_avec_clusters")
output.write_with_schema(df)
```

### **Recipe 3 : Attribution des causes aux clusters**

```python
import dataiku
import pandas as pd

# Lire les données
dataset = dataiku.Dataset("metis_avec_clusters")
df = dataset.get_dataframe()

# Analyser les clusters
cluster_analysis = []

for cluster in df['cluster'].unique():
    if cluster == -1:
        continue
        
    cluster_data = df[df['cluster'] == cluster]
    
    # Compter les tickets fiables par cause
    if cluster_data['est_fiable'].any():
        cause_counts = cluster_data[cluster_data['est_fiable']]['cause'].value_counts()
        if len(cause_counts) > 0:
            cause_attribuee = cause_counts.index[0]
            confidence = cause_counts.iloc[0] / cluster_data['est_fiable'].sum()
        else:
            cause_attribuee = "À déterminer"
            confidence = 0
    else:
        cause_attribuee = "À déterminer"
        confidence = 0
    
    # Analyser les caractéristiques dominantes
    groupe_dominant = cluster_data['Groupe affecté'].mode()[0] if len(cluster_data) > 0 else ""
    service_dominant = cluster_data['Service métier'].mode()[0] if len(cluster_data) > 0 else ""
    
    cluster_analysis.append({
        'cluster': cluster,
        'cause_attribuee': cause_attribuee,
        'confidence': confidence,
        'n_tickets': len(cluster_data),
        'n_fiables': cluster_data['est_fiable'].sum(),
        'groupe_dominant': groupe_dominant,
        'service_dominant': service_dominant
    })

# Créer DataFrame d'analyse
df_analysis = pd.DataFrame(cluster_analysis)

# Mapper les causes aux tickets
cause_mapping = dict(zip(df_analysis['cluster'], df_analysis['cause_attribuee']))
df['cause_predite'] = df['cluster'].map(cause_mapping)
df['cause_predite'] = df['cause_predite'].fillna('Bruit')

# Sauvegarder
output_tickets = dataiku.Dataset("metis_final")
output_tickets.write_with_schema(df)

output_analysis = dataiku.Dataset("analyse_clusters")
output_analysis.write_with_schema(df_analysis)
```

## 🎯 Conseils pour l'utilisation dans Dataiku :

1. **Gestion de la mémoire** : Pour de gros volumes, utilisez la lecture par chunks :
```python
for chunk in dataset.iter_dataframes(chunksize=1000):
    # Traiter le chunk
```

2. **Parallélisation** : Utilisez les partitions Dataiku pour traiter en parallèle

3. **Monitoring** : Créez un dashboard Dataiku pour visualiser :
   - Distribution des clusters
   - Score de silhouette
   - Nombre de tickets par cause

4. **Automatisation** : Créez un scenario Dataiku pour automatiser tout le pipeline

Avez-vous des questions sur l'une de ces étapes ?

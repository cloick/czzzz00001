Excellent ! Félicitations pour avoir obtenu la licence Dataiku ! Je vais vous guider pas à pas pour implémenter votre projet de classification des tickets METIS sur Dataiku.

## 🚀 Guide complet Dataiku pour votre projet

### 1. Premiers pas avec Dataiku

**Configuration initiale :**
- Connectez-vous à votre instance Dataiku
- Créez un nouveau projet : "Classification_Tickets_METIS"
- Définissez les permissions et partagez avec votre équipe

### 2. Import et préparation des données

**Étape 1 : Importer vos datasets**
```
1. Cliquez sur "IMPORT DATASET"
2. Choisissez votre source :
   - Fichiers CSV : metis_tickets.csv et gdp_tickets.csv
   - Ou connexion directe à votre base de données
3. Prévisualisez et validez l'import
```

**Étape 2 : Préparer les données**
- Utilisez un recipe "Prepare" pour :
  - Nettoyer le texte des notes de résolution
  - Gérer les valeurs manquantes
  - Créer la colonne 'est_fiable'
  - Encoder les variables catégorielles

### 3. Création des embeddings avec CamemBERT

**Dans Dataiku :**
```python
# Recipe Python pour générer les embeddings
import dataiku
import pandas as pd
from transformers import CamembertTokenizer, CamembertModel
import torch

# Lire le dataset
df = dataiku.Dataset("metis_tickets_prepared").get_dataframe()

# Charger CamemBERT
tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
model = CamembertModel.from_pretrained('camembert-base')

# Fonction pour générer les embeddings
def get_embeddings(texts, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, 
                          max_length=128, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
            # Utiliser le token [CLS]
            batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.extend(batch_embeddings)
    return embeddings

# Appliquer sur vos données
df['embeddings'] = get_embeddings(df['notes_resolution_nettoyees'].tolist())
```

### 4. Implémentation du clustering HDBSCAN

**Créer un recipe Python pour le clustering :**
```python
# Enrichir les embeddings avec les variables catégorielles
from sklearn.preprocessing import StandardScaler
import numpy as np

# Préparer les features catégorielles
cat_features = ['Groupe_affecté_encoded', 'Service_métier_encoded', 
                'Cat1_encoded', 'Cat2_encoded', 'Priorité_encoded']

# Normaliser et pondérer
scaler = StandardScaler()
cat_data = scaler.fit_transform(df[cat_features])

# Pondération comme dans votre approche
weights = {'Groupe_affecté': 3.0, 'Service_métier': 2.0, 
           'Cat1': 1.0, 'Cat2': 1.0, 'Priorité': 0.5}

# Combiner embeddings et features catégorielles
embeddings_enriched = np.hstack([df['embeddings'].tolist(), cat_data])

# UMAP + HDBSCAN avec vos paramètres optimaux
from umap import UMAP
import hdbscan

reducer = UMAP(n_components=2, random_state=42, n_neighbors=15)
embeddings_2d = reducer.fit_transform(embeddings_enriched)

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=240,
    min_samples=20,
    cluster_selection_epsilon=1.56,
    metric='euclidean'
)
df['cluster'] = clusterer.fit_predict(embeddings_2d)
```

### 5. Configuration du modèle LLM dans Dataiku

**Utiliser les LLM Labs de Dataiku :**
1. Allez dans "Lab" > "LLM Experimentation"
2. Configurez votre modèle (CamemBERT ou mDeBERTa)
3. Créez un prompt pour la classification des tickets

### 6. Création du classificateur hybride

**Recipe Python pour le vote pondéré :**
```python
def classify_hybrid(row, llm_predictions, cluster_to_cause):
    # Récupérer les prédictions
    llm_pred = llm_predictions.get(row['N° INC'])
    cluster_pred = cluster_to_cause.get(row['cluster'], 'À déterminer')
    
    # Système de vote pondéré
    weights = {
        'llm': 0.6,
        'clustering': 0.4
    }
    
    # Logique de combinaison
    if cluster_pred == 'À déterminer':
        return llm_pred
    elif llm_pred == cluster_pred:
        return llm_pred
    else:
        # Vote pondéré si désaccord
        # Implémenter votre logique spécifique
        return llm_pred  # ou cluster_pred selon confiance
```

### 7. Création du pipeline de déploiement

**Dans Dataiku Flow :**
1. Créez un scenario pour automatiser :
   - Import des nouvelles données
   - Préparation
   - Génération d'embeddings
   - Clustering
   - Classification LLM
   - Vote hybride
   - Export des résultats

2. Configurez les triggers (temps réel, batch, etc.)

### 8. Monitoring et visualisations

**Créer un dashboard Dataiku :**
- Distribution des clusters
- Score de silhouette en temps réel
- Métriques de performance
- Évolution du volume de tickets

### 9. API pour le déploiement

```python
# Endpoint API dans Dataiku
def predict_ticket_cause(ticket_data):
    # Prétraitement
    # Embedding
    # Prédiction hybride
    return {
        'cause': predicted_cause,
        'confidence': confidence_score,
        'method': 'hybrid'
    }
```

### 💡 Conseils pratiques :

1. **Commencez petit** : Testez d'abord sur un échantillon de 1000 tickets
2. **Utilisez les Visual Recipes** : Dataiku offre des interfaces visuelles pour beaucoup d'opérations
3. **Versioning** : Utilisez le versioning de Dataiku pour tracker vos modèles
4. **Collaboration** : Partagez votre projet avec l'équipe pour review

Avez-vous des questions spécifiques sur l'une de ces étapes ? Par quoi souhaitez-vous commencer ?

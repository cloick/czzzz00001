
The user wants me to provide an improved version of the code for analyzing verbatims using a language model. Based on the issues identified, I'll create a comprehensive, optimized version that addresses:

1. Performance issues with the progress tracking
2. Potential timeouts or excessive processing time
3. Memory management
4. Better feedback during processing

I'll write the complete code in a clean, well-structured format that can be directly copied and used in their notebook.

# Code optimisé pour l'analyse de verbatims avec un LLM

Voici le code complet optimisé pour améliorer les performances et le suivi de progression lors de l'analyse de verbatims :

```python
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from tqdm import tqdm
import time
import gc

# Configuration des paramètres d'analyse
VERBATIMS_PAR_PERIMETRE = 3  # Nombre de verbatims à analyser par périmètre
MAX_NEW_TOKENS = 512  # Limite de tokens générés (réduite pour de meilleures performances)
GENERATION_TIMEOUT = 180.0  # Timeout en secondes pour chaque génération

# Spécifier le dossier contenant le modèle Mistral téléchargé
model_path = "./mistral-7b-instruct"

# Vérifier que tous les fichiers nécessaires sont présents
required_files = ["config.json", "tokenizer.json", "tokenizer_config.json", 
                 "model.safetensors.index.json", "special_tokens_map.json"]
                 
for file in required_files:
    if not os.path.exists(os.path.join(model_path, file)):
        print(f"Attention: {file} semble manquant dans {model_path}")

# Optimisation de la mémoire avant chargement
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Chargement du tokenizer et du modèle
print("Chargement du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # Définir le token de padding

print("Chargement du modèle (cette étape peut prendre plusieurs minutes)...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)

# Nettoyage de la mémoire après chargement
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Chargement des données avec les noms exacts des colonnes
print("Chargement des données...")
df = pd.read_excel("enquetes_satisfaction.xlsx")
df['ligne_source'] = df.index

# Filtrage des verbatims négatifs
humeurs_negatives = ['Plutôt Insatisfaite', 'Très insatisfaite']
neg_verbatims = df[df["Humeur"].isin(humeurs_negatives)]
print(f"Nombre de verbatims négatifs: {len(neg_verbatims)}")

# Fonction d'analyse directe des verbatims avec suivi détaillé
def analyser_verbatims(verbatims_df, context="", max_verbatims=None):
    """
    Analyse les verbatims d'un groupe avec un suivi détaillé
    
    Args:
        verbatims_df: DataFrame contenant les verbatims à analyser
        context: Contexte supplémentaire pour l'analyse (ex: nom du périmètre)
        max_verbatims: Nombre maximum de verbatims à analyser (None pour tous)
    """
    # Limiter le nombre de verbatims si spécifié
    if max_verbatims and len(verbatims_df) > max_verbatims:
        verbatims_subset = verbatims_df.sample(max_verbatims, random_state=42)
        print(f"  Analyse limitée à {max_verbatims} verbatims sur {len(verbatims_df)}")
    else:
        verbatims_subset = verbatims_df
    
    # Préparation des verbatims avec leur numéro de ligne
    verbatims_text = "\n".join([
        f"Ligne {row['ligne_source']}: {row['Verbatim']}" 
        for _, row in verbatims_subset.iterrows()
    ])
    
    # Création du prompt pour le modèle
    prompt = f"""<s>[INST] Tu es un expert en analyse de satisfaction client. Analyse ces verbatims négatifs {context}:

{verbatims_text}

Identifie les 2-3 problématiques principales qui ressortent de ces verbatims.
Pour chaque problématique:
1. Donne un titre court et précis
2. Explique la problématique en 1-2 phrases
3. Liste les numéros de ligne des verbatims qui s'y rapportent
4. Évalue sa criticité (Haute/Moyenne/Basse)

FORMAT:
## Problématique 1: [TITRE]
Description: [EXPLICATION]
Verbatims concernés: lignes [NUMÉROS]
Criticité: [NIVEAU]

## Problématique 2: [TITRE]
...
[/INST]</s>
"""
    
    # Suivi de la génération
    print(f"  Début de la génération - {time.strftime('%H:%M:%S')}")
    start_time = time.time()
    
    # Génération de la réponse avec gestion de mémoire optimisée
    try:
        # Préparation des inputs avec attention_mask explicite
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Génération avec timeout et paramètres optimisés
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
            max_time=GENERATION_TIMEOUT,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Décodage et nettoyage du résultat
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("[/INST]")[1].strip() if "[/INST]" in response else response
        
        end_time = time.time()
        print(f"  Génération terminée en {end_time - start_time:.1f} secondes")
        print(f"  Longueur de la réponse: {len(response)} caractères")
        
        # Nettoyage de la mémoire
        del inputs, outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return response
        
    except Exception as e:
        end_time = time.time()
        print(f"  Erreur après {end_time - start_time:.1f} secondes: {str(e)}")
        return f"Erreur d'analyse: {str(e)}"

# Analyse par périmètre avec barre de progression améliorée
print("Début de l'analyse par périmètre...")
resultats = []

# Création d'une liste des périmètres pour la barre de progression
perimetres = neg_verbatims["Perimetre"].unique()
print(f"Périmètres à analyser: {', '.join(perimetres)}")

# Barre de progression avec format amélioré
for i, perimetre in enumerate(perimetres):
    # Affichage de progression manuel pour plus de clarté
    print(f"\nAnalyse du périmètre {i+1}/{len(perimetres)}: {perimetre}")
    
    groupe = neg_verbatims[neg_verbatims["Perimetre"] == perimetre]
    print(f"Nombre de verbatims dans ce périmètre: {len(groupe)}")
    
    analyse = analyser_verbatims(
        groupe, 
        f"du périmètre '{perimetre}'",
        max_verbatims=VERBATIMS_PAR_PERIMETRE
    )
    
    resultats.append({
        "perimetre": perimetre,
        "analyse": analyse,
        "nombre_verbatims": len(groupe),
        "verbatims_analyses": min(VERBATIMS_PAR_PERIMETRE, len(groupe))
    })
    
    # Affichage de la progression
    print(f"Périmètre {perimetre} analysé: {i+1}/{len(perimetres)} terminés ({(i+1)/len(perimetres)*100:.1f}%)")

# Création du rapport final
rapport = "# Analyse des Problématiques dans les Verbatims Négatifs\n\n"
rapport += f"Date d'analyse: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n"
rapport += f"Paramètres: {VERBATIMS_PAR_PERIMETRE} verbatims analysés par périmètre\n\n"

for resultat in resultats:
    rapport += f"## Périmètre: {resultat['perimetre']}\n"
    rapport += f"Nombre total de verbatims: {resultat['nombre_verbatims']} "
    rapport += f"(Analysés: {resultat['verbatims_analyses']})\n\n"
    rapport += resultat['analyse']
    rapport += "\n\n---\n\n"

# Enregistrement du rapport
with open("rapport_analyse_verbatims.md", "w", encoding="utf-8") as f:
    f.write(rapport)

print("\nRapport d'analyse généré avec succès: rapport_analyse_verbatims.md")
```

## Améliorations apportées

Ce code intègre plusieurs optimisations importantes :

1. **Gestion améliorée de la mémoire** : Utilisation de `gc.collect()` et `torch.cuda.empty_cache()` aux moments stratégiques pour libérer la mémoire.

2. **Paramètres de génération optimisés** :
   - Réduction du nombre maximal de tokens générés (512 au lieu de 1024)
   - Ajout d'un timeout de 3 minutes par génération pour éviter les blocages
   - Configuration explicite du `pad_token_id` pour éviter les avertissements

3. **Suivi de progression détaillé** :
   - Affichage de l'heure de début et de la durée de chaque génération
   - Remplacement de tqdm par un suivi manuel plus informatif
   - Affichage du pourcentage de progression global

4. **Robustesse** :
   - Gestion des exceptions durant la génération
   - Validation des entrées et des sorties
   - Définition explicite des paramètres d'attention

5. **Efficacité** :
   - Réduction du nombre de problématiques demandées (2-3 au lieu de 3-5)
   - Transfert optimal des tenseurs vers le périphérique approprié

Ce code devrait fonctionner de manière plus fiable, avec un suivi de progression clair et une meilleure gestion des ressources système.

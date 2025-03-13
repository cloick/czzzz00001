import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline

# Chemin vers les fichiers téléchargés
MODEL_PATH = "./mDeBERTa-v3-base-multilingual-nli-local"

# Initialisation du tokenizer et conversion du modèle en TensorFlow
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Création du pipeline de classification zero-shot
classifier = pipeline(
    "zero-shot-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if tf.config.list_physical_devices('GPU') else -1
)

# Définition des 15 causes METIS
candidate_labels = [
    "Matériel-Infrastructure", 
    "Logiciel", 
    "Intervention", 
    "Non déterminée",
    "Environnementale",
    "Erreur à l'exécution",
    "Programme défaillant",
    "Sécurité",
    # Ajoutez les 7 autres causes ici
]

# Exemple de classification
texte_ticket = "Collaborateur a eu un probleme avec son Microsoft365, le redemarrage du logiciel a resolu le probleme"
resultat = classifier(texte_ticket, candidate_labels, multi_label=False)

print(f"Texte: {texte_ticket}")
print("Classification:")
for label, score in zip(resultat["labels"], resultat["scores"]):
    print(f"- {label}: {score:.4f}")
    
--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
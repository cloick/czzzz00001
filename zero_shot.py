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

cause et sous cause :
    
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline

def setup_zero_shot_classifier(model_path):
    """
    Initialize the zero-shot classification pipeline with the specified model.
    
    Args:
        model_path (str): Path to the local model files or model name on Hugging Face
        
    Returns:
        pipeline: The zero-shot classification pipeline
    """
    try:
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Create classification pipeline
        classifier = pipeline(
            "zero-shot-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if tf.config.list_physical_devices('GPU') else -1
        )
        
        print(f"Zero-shot classifier successfully initialized from {model_path}")
        return classifier
    
    except Exception as e:
        print(f"Error initializing zero-shot classifier: {e}")
        raise

def define_taxonomy():
    """
    Define the complete taxonomy of causes and sub-causes for METIS tickets.
    
    Returns:
        tuple: (causes, sub_causes) where causes is a list of all main causes and
               sub_causes is a dictionary mapping each cause to its list of sub-causes
    """
    # Define the list of all 15 causes
    causes = [
        "Matériel-Infrastructure", 
        "Logiciel", 
        "Intervention", 
        "Non déterminée",
        "Environnementale",
        "Erreur à l'exécution",
        "Programme défaillant",
        "Sécurité",
        # Add the 7 other causes as needed
        "Réseau",
        "Base de données",
        "Erreur utilisateur",
        "Configuration",
        "Authentification",
        "Permissions",
        "Maintenance"
    ]
    
    # Define the sub-causes for each main cause
    sub_causes = {
        "Matériel-Infrastructure": [
            "Panne", 
            "Pièce défectueuse", 
            "Configuration réseau", 
            "Capacité insuffisante",
            "Périphérique non reconnu"
        ],
        "Logiciel": [
            "Bug", 
            "Paramétrage incorrect", 
            "Configuration", 
            "Version obsolète",
            "Compatibilité",
            "Mise à jour requise"
        ],
        "Intervention": [
            "Et supplier",
            "Et standard",
            "Procédure incorrecte",
            "Paramétrage incorrect",
            "Erreur humaine"
        ],
        "Non déterminée": [
            "Analyse en cours",
            "Information insuffisante",
            "Non reproductible"
        ],
        "Environnementale": [
            "Alimentation électrique", 
            "Température", 
            "Humidité",
            "Ventilation"
        ],
        "Erreur à l'exécution": [
            "Timeout",
            "Mémoire insuffisante",
            "Exception non gérée",
            "Conflit de ressources"
        ],
        "Programme défaillant": [
            "Boucle infinie",
            "Fuite mémoire",
            "Synchronisation",
            "Deadlock"
        ],
        "Sécurité": [
            "Accès non autorisé",
            "Authentification échouée",
            "Attaque malveillante",
            "Vulnérabilité"
        ],
        # Define sub-causes for the remaining causes
        "Réseau": [
            "Connexion perdue",
            "Latence élevée",
            "DNS",
            "Routage"
        ],
        "Base de données": [
            "Corruption",
            "Verrouillage",
            "Performance",
            "Espace disque"
        ],
        "Erreur utilisateur": [
            "Formation insuffisante",
            "Erreur de saisie",
            "Procédure non suivie"
        ],
        "Configuration": [
            "Paramètres incorrects",
            "Conflit de configuration",
            "Valeurs par défaut inadaptées"
        ],
        "Authentification": [
            "Identifiants expirés",
            "Compte verrouillé",
            "SSO défaillant"
        ],
        "Permissions": [
            "Accès insuffisant",
            "Droits excessifs",
            "Groupe incorrect"
        ],
        "Maintenance": [
            "Planifiée",
            "Urgente",
            "Non communiquée"
        ]
    }
    
    return causes, sub_causes

def classify_ticket_hierarchical(classifier, ticket_text, causes, sub_causes, 
                                 multi_label=False, threshold=0.05):
    """
    Perform hierarchical zero-shot classification on a ticket text.
    
    Args:
        classifier: The zero-shot classification pipeline
        ticket_text (str): The text of the ticket to classify
        causes (list): List of all possible main causes
        sub_causes (dict): Dictionary mapping causes to their sub-causes
        multi_label (bool): Whether to allow multiple cause classifications
        threshold (float): Minimum confidence threshold for secondary causes
        
    Returns:
        dict: Classification results including cause and sub-cause predictions
    """
    # Step 1: Classify the main cause
    cause_result = classifier(ticket_text, causes, multi_label=multi_label)
    
    # Extract the primary predicted cause
    primary_cause = cause_result["labels"][0]
    primary_score = cause_result["scores"][0]
    
    # Get additional causes above threshold if multi_label is True
    additional_causes = []
    if multi_label:
        for cause, score in zip(cause_result["labels"][1:], cause_result["scores"][1:]):
            if score >= threshold:
                additional_causes.append((cause, score))
    
    # Step 2: Classify the sub-cause based on the predicted main cause(s)
    sub_cause_predictions = {}
    
    # Function to classify sub-causes for a given cause
    def get_sub_causes(cause, score):
        if cause in sub_causes:
            # Enrich context with the cause information
            enriched_text = f"Incident informatique: {ticket_text}. " \
                           f"Cet incident concerne un problème de {cause}. " \
                           f"Quelle est la sous-cause spécifique?"
            
            sub_result = classifier(enriched_text, sub_causes[cause], multi_label=False)
            
            return {
                "sub_cause": sub_result["labels"][0],
                "sub_score": sub_result["scores"][0],
                "all_sub_causes": list(zip(sub_result["labels"], sub_result["scores"]))
            }
        return None
    
    # Get sub-causes for primary cause
    primary_sub = get_sub_causes(primary_cause, primary_score)
    
    # Get sub-causes for additional causes
    additional_subs = {}
    for cause, score in additional_causes:
        additional_subs[cause] = get_sub_causes(cause, score)
    
    # Compile results
    results = {
        "ticket_text": ticket_text,
        "primary_cause": primary_cause,
        "primary_score": primary_score,
        "primary_sub_cause": primary_sub,
        "all_causes": list(zip(cause_result["labels"], cause_result["scores"])),
        "additional_causes": additional_causes,
        "additional_sub_causes": additional_subs
    }
    
    return results

def format_classification_result(result):
    """
    Format the classification result for display.
    
    Args:
        result (dict): Classification result from classify_ticket_hierarchical
        
    Returns:
        str: Formatted result string
    """
    output = [
        f"Texte du ticket: {result['ticket_text']}",
        "\nClassification des causes:",
    ]
    
    # Format all causes
    for cause, score in result["all_causes"]:
        output.append(f"- {cause}: {score:.4f}")
    
    # Format primary cause and sub-cause
    output.append(f"\nCause principale: {result['primary_cause']} ({result['primary_score']:.4f})")
    
    if result["primary_sub_cause"]:
        primary_sub = result["primary_sub_cause"]
        output.append(f"Sous-cause principale: {primary_sub['sub_cause']} ({primary_sub['sub_score']:.4f})")
        
        output.append("\nToutes les sous-causes possibles:")
        for sub, score in primary_sub["all_sub_causes"]:
            output.append(f"- {sub}: {score:.4f}")
    
    # Format additional causes if any
    if result["additional_causes"]:
        output.append("\nCauses additionnelles potentielles:")
        for cause, score in result["additional_causes"]:
            output.append(f"- {cause} ({score:.4f})")
            
            if cause in result["additional_sub_causes"] and result["additional_sub_causes"][cause]:
                sub = result["additional_sub_causes"][cause]
                output.append(f"  Sous-cause: {sub['sub_cause']} ({sub['sub_score']:.4f})")
    
    return "\n".join(output)

def main():
    """Main function to demonstrate the hierarchical classification."""
    # Set up the model path - adjust based on where you downloaded the model
    MODEL_PATH = "./mDeBERTa-v3-base-multilingual-nli-local"  # or use remote: "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    
    # Set up the classifier
    classifier = setup_zero_shot_classifier(MODEL_PATH)
    
    # Define the taxonomy
    causes, sub_causes = define_taxonomy()
    
    # Example ticket text
    ticket_text = "Collaborateur a eu un probleme avec son Microsoft365, le redemarrage du logiciel a resolu le probleme"
    
    # Classify the ticket
    result = classify_ticket_hierarchical(
        classifier, ticket_text, causes, sub_causes, multi_label=True, threshold=0.05
    )
    
    # Print formatted results
    print(format_classification_result(result))

if __name__ == "__main__":
    main()
    

    
--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------

Enrichissement des descriptions de classes pour améliorer la classification

def define_taxonomy_enriched():
    """
    Définit une taxonomie enrichie des causes et sous-causes avec des descriptions détaillées.
    
    Returns:
        tuple: (causes_with_descriptions, subcauses_with_descriptions)
    """
    # Définition des causes avec descriptions
    causes_with_descriptions = {
        "Matériel-Infrastructure": "Problèmes liés aux composants physiques informatiques, aux périphériques, aux serveurs ou à l'infrastructure matérielle. Ces incidents affectent généralement le fonctionnement physique des équipements.",
        
        "Logiciel": "Dysfonctionnements dans les applications, les systèmes d'exploitation ou autres programmes informatiques. Ces incidents sont liés au comportement anormal des logiciels sans cause matérielle identifiée.",
        
        "Intervention": "Actions techniques planifiées ou réactives réalisées par un technicien ou un prestataire sur les systèmes. Ces incidents sont directement liés à une action humaine technique sur le système.",
        
        "Non déterminée": "Incidents dont la cause n'a pas encore été identifiée ou est en cours d'analyse. L'origine du problème reste inconnue ou nécessite des investigations supplémentaires.",
        
        "Environnementale": "Problèmes liés aux conditions physiques externes comme l'alimentation électrique, la température, l'humidité ou autres facteurs environnementaux affectant les systèmes informatiques.",
        
        "Erreur à l'exécution": "Incidents survenant lors de l'exécution d'un programme ou d'une tâche, souvent liés à des ressources insuffisantes ou des conditions imprévues pendant le fonctionnement normal.",
        
        "Programme défaillant": "Défauts structurels dans le code d'une application qui causent des dysfonctionnements systématiques. Ces problèmes sont intrinsèques au programme et surviennent régulièrement dans certaines conditions.",
        
        "Sécurité": "Incidents liés à la protection des systèmes, aux tentatives d'accès non autorisés, aux violations de politique de sécurité ou aux vulnérabilités exploitées.",
        
        "Réseau": "Problèmes de connectivité, de communication entre systèmes, ou d'infrastructure réseau comme les routeurs, switches, ou la configuration DNS. Ces incidents affectent la transmission des données.",
        
        "Base de données": "Dysfonctionnements liés au stockage, à l'accès ou à l'intégrité des données dans les systèmes de gestion de bases de données.",
        
        "Erreur utilisateur": "Incidents causés par une mauvaise manipulation, une erreur de saisie, ou le non-respect des procédures par un utilisateur non-technique.",
        
        "Configuration": "Problèmes liés aux paramètres incorrects des systèmes, aux réglages inappropriés ou aux incompatibilités de configuration entre différents composants.",
        
        "Authentification": "Difficultés liées à l'identification des utilisateurs, à la validation des identifiants ou aux mécanismes de contrôle d'accès aux systèmes.",
        
        "Permissions": "Incidents relatifs aux droits d'accès, aux autorisations insuffisantes ou excessives, ou à la gestion des privilèges utilisateurs.",
        
        "Maintenance": "Activités planifiées d'entretien, de mise à jour ou d'amélioration des systèmes qui peuvent occasionner des interruptions de service temporaires."
    }
    
    # Définition des sous-causes avec descriptions
    subcauses_with_descriptions = {
        "Matériel-Infrastructure": {
            "Panne": "Arrêt complet ou dysfonctionnement majeur d'un équipement matériel sans possibilité d'utilisation normale.",
            "Pièce défectueuse": "Composant matériel spécifique endommagé ou ne fonctionnant pas conformément aux spécifications.",
            "Configuration réseau": "Problème lié aux paramètres matériels de connexion ou d'interconnexion des équipements.",
            "Capacité insuffisante": "Ressources matérielles inadéquates pour supporter la charge de travail demandée (RAM, stockage, processeur).",
            "Périphérique non reconnu": "Équipement externe non détecté ou non compatible avec le système."
        },
        
        "Logiciel": {
            "Bug": "Défaut de programmation causant un comportement inattendu ou incorrect du logiciel.",
            "Paramétrage incorrect": "Options ou réglages logiciels mal configurés empêchant le fonctionnement normal.",
            "Configuration": "Problème dans les fichiers ou paramètres de configuration du logiciel.",
            "Version obsolète": "Utilisation d'une version ancienne du logiciel causant des incompatibilités ou manquant de fonctionnalités.",
            "Compatibilité": "Conflit entre différentes applications ou avec le système d'exploitation.",
            "Mise à jour requise": "Nécessité d'installer une mise à jour pour corriger un problème connu ou améliorer la sécurité.",
            "Espace disque": "Espace de stockage insuffisant pour l'installation ou le fonctionnement du logiciel."
        },
        
        "Intervention": {
            "Et supplier": "Intervention réalisée par un fournisseur ou prestataire externe ayant causé un incident.",
            "Et standard": "Intervention technique standard ayant entraîné un dysfonctionnement imprévu.",
            "Procédure incorrecte": "Non-respect des procédures techniques établies lors d'une intervention.",
            "Paramétrage incorrect": "Erreur dans la configuration des paramètres lors d'une intervention technique.",
            "Erreur humaine": "Erreur commise par un technicien pendant une intervention technique."
        },
        
        # Continuing with the other sub-causes in the same detailed format
        "Non déterminée": {
            "Analyse en cours": "Investigation active pour déterminer la cause fondamentale du problème.",
            "Information insuffisante": "Manque de données ou de contexte pour identifier clairement la cause.",
            "Non reproductible": "Incident qui ne peut pas être reproduit de façon systématique pour analyse."
        },
        
        # I'll continue with all other sub-causes following the same pattern
        "Environnementale": {
            "Alimentation électrique": "Problèmes liés à l'approvisionnement en électricité, coupures ou variations de tension.",
            "Température": "Conditions thermiques inappropriées affectant le fonctionnement des équipements.",
            "Humidité": "Niveaux d'humidité excessifs ou insuffisants impactant les systèmes informatiques.",
            "Ventilation": "Défaillance ou insuffisance des systèmes de refroidissement ou d'aération."
        },
        
        "Erreur à l'exécution": {
            "Timeout": "Dépassement du délai alloué pour l'exécution d'une opération ou d'une requête.",
            "Mémoire insuffisante": "Ressources mémoire épuisées pendant l'exécution d'un programme.",
            "Exception non gérée": "Erreur imprévue pendant l'exécution non traitée par le programme.",
            "Conflit de ressources": "Compétition entre processus pour l'accès aux mêmes ressources système."
        },
        
        "Programme défaillant": {
            "Boucle infinie": "Code entrant dans une séquence répétitive sans condition de sortie.",
            "Fuite mémoire": "Allocation de mémoire non libérée causant une consommation croissante des ressources.",
            "Synchronisation": "Problèmes de coordination entre différents processus ou threads.",
            "Deadlock": "Situation de blocage mutuel où plusieurs processus s'attendent l'un l'autre."
        },
        
        "Sécurité": {
            "Accès non autorisé": "Tentative réussie d'utilisation d'un système sans les droits requis.",
            "Authentification échouée": "Échec des mécanismes de vérification d'identité.",
            "Attaque malveillante": "Action délibérée visant à compromettre ou perturber les systèmes.",
            "Vulnérabilité": "Faille de sécurité connue susceptible d'être exploitée."
        },
        
        "Réseau": {
            "Connexion perdue": "Interruption complète de la communication réseau.",
            "Latence élevée": "Délais excessifs dans la transmission des données sur le réseau.",
            "DNS": "Problèmes liés à la résolution des noms de domaine ou aux serveurs DNS.",
            "Routage": "Difficultés dans l'acheminement des paquets réseau vers leur destination."
        },
        
        "Base de données": {
            "Corruption": "Altération des données stockées rendant certaines informations inaccessibles ou invalides.",
            "Verrouillage": "Blocage d'accès aux données dû à des verrous concurrents ou non libérés.",
            "Performance": "Temps de réponse anormalement longs pour les requêtes ou opérations.",
            "Espace disque": "Espace de stockage insuffisant pour les données ou les journaux de transaction."
        },
        
        "Erreur utilisateur": {
            "Formation insuffisante": "Manque de connaissances ou de compétences de l'utilisateur pour l'opération concernée.",
            "Erreur de saisie": "Information incorrecte introduite dans le système par l'utilisateur.",
            "Procédure non suivie": "Non-respect des étapes requises ou des bonnes pratiques établies."
        },
        
        "Configuration": {
            "Paramètres incorrects": "Valeurs ou options mal définies dans les fichiers de configuration.",
            "Conflit de configuration": "Paramètres incompatibles entre différents composants ou applications.",
            "Valeurs par défaut inadaptées": "Paramètres standard ne correspondant pas aux besoins spécifiques."
        },
        
        "Authentification": {
            "Identifiants expirés": "Mot de passe ou jeton d'authentification ayant dépassé sa durée de validité.",
            "Compte verrouillé": "Accès bloqué suite à plusieurs tentatives échouées ou pour raison administrative.",
            "SSO défaillant": "Dysfonctionnement du système d'authentification unique inter-applications."
        },
        
        "Permissions": {
            "Accès insuffisant": "Droits manquants pour effectuer une opération requise.",
            "Droits excessifs": "Privilèges trop élevés par rapport au besoin réel, posant un risque de sécurité.",
            "Groupe incorrect": "Appartenance à un groupe de sécurité inapproprié pour le rôle de l'utilisateur."
        },
        
        "Maintenance": {
            "Planifiée": "Intervention de maintenance prévue et communiquée à l'avance.",
            "Urgente": "Maintenance non planifiée requise pour résoudre un problème critique.",
            "Non communiquée": "Intervention technique réalisée sans notification préalable aux utilisateurs."
        }
    }
    
    return causes_with_descriptions, subcauses_with_descriptions



Modification du code de classification :
    
def classify_ticket_hierarchical_enriched(classifier, ticket_text, causes_dict, subcauses_dict, 
                                         multi_label=False, threshold=0.05):
    """
    Réalise une classification hiérarchique zero-shot sur un ticket en utilisant des descriptions enrichies.
    
    Args:
        classifier: Pipeline de classification zero-shot
        ticket_text (str): Texte du ticket à classifier
        causes_dict (dict): Dictionnaire des causes avec leurs descriptions
        subcauses_dict (dict): Dictionnaire des sous-causes avec leurs descriptions
        multi_label (bool): Si True, permet plusieurs classifications de causes
        threshold (float): Seuil minimum de confiance pour les causes secondaires
        
    Returns:
        dict: Résultats de classification incluant cause et sous-cause
    """
    # Préparation des candidats pour la classification des causes
    causes = list(causes_dict.keys())
    # Enrichir le prompt avec les descriptions de causes
    cause_candidates = [f"{cause}: {causes_dict[cause]}" for cause in causes]
    
    # Enrichir le contexte pour la classification
    enriched_context = (
        f"Incident informatique: {ticket_text}\n\n"
        f"Analyser cet incident et déterminer sa cause principale parmi les catégories suivantes:"
    )
    
    # Étape 1: Classifier la cause principale avec contexte enrichi
    cause_result = classifier(
        enriched_context, 
        cause_candidates, 
        multi_label=multi_label
    )
    
    # Extraire la cause principale prédite (en supprimant la description)
    primary_cause_full = cause_result["labels"][0]
    primary_cause = primary_cause_full.split(":")[0].strip()
    primary_score = cause_result["scores"][0]
    
    # Obtenir des causes supplémentaires au-dessus du seuil si multi_label est True
    additional_causes = []
    if multi_label:
        for cause_full, score in zip(cause_result["labels"][1:], cause_result["scores"][1:]):
            if score >= threshold:
                cause = cause_full.split(":")[0].strip()
                additional_causes.append((cause, score))
    
    # Étape 2: Classifier la sous-cause en fonction de la cause principale prédite
    sub_cause_predictions = {}
    
    # Fonction pour classifier les sous-causes pour une cause donnée
    def get_sub_causes(cause, score):
        if cause in subcauses_dict:
            subcause_candidates = [
                f"{subcause}: {description}" 
                for subcause, description in subcauses_dict[cause].items()
            ]
            
            # Contexte enrichi spécifique à la sous-cause
            subcause_context = (
                f"Incident informatique: {ticket_text}\n\n"
                f"Cet incident a été identifié comme un problème de {cause}.\n"
                f"Description de la cause: {causes_dict[cause]}\n\n"
                f"Déterminez la sous-cause spécifique parmi les options suivantes:"
            )
            
            sub_result = classifier(subcause_context, subcause_candidates, multi_label=False)
            
            # Extraire la sous-cause (en supprimant la description)
            subcauses = [label.split(":")[0].strip() for label in sub_result["labels"]]
            
            return {
                "sub_cause": subcauses[0],
                "sub_score": sub_result["scores"][0],
                "all_sub_causes": list(zip(subcauses, sub_result["scores"]))
            }
        return None
    
    # Obtenir les sous-causes pour la cause principale
    primary_sub = get_sub_causes(primary_cause, primary_score)
    
    # Obtenir les sous-causes pour les causes supplémentaires
    additional_subs = {}
    for cause, score in additional_causes:
        additional_subs[cause] = get_sub_causes(cause, score)
    
    # Compiler les résultats
    # Extraire les étiquettes de cause sans descriptions pour le résultat final
    all_causes_clean = [(label.split(":")[0].strip(), score) 
                        for label, score in zip(cause_result["labels"], cause_result["scores"])]
    
    results = {
        "ticket_text": ticket_text,
        "primary_cause": primary_cause,
        "primary_score": primary_score,
        "primary_sub_cause": primary_sub,
        "all_causes": all_causes_clean,
        "additional_causes": additional_causes,
        "additional_sub_causes": additional_subs
    }
    
    return results

Adaptation de la fonction d'evalution 

def evaluer_modele_enrichi(classifier, df_test):
    """
    Évalue le modèle sur le jeu de données de test avec descriptions enrichies.
    
    Args:
        classifier: Pipeline de classification zero-shot
        df_test (pd.DataFrame): DataFrame contenant les tickets de test
        
    Returns:
        dict: Résultats d'évaluation
    """
    # Obtenir les taxonomies enrichies
    causes_dict, subcauses_dict = define_taxonomy_enriched()
    
    predictions = []
    y_true_causes = []
    y_pred_causes = []
    y_true_subcauses = []
    y_pred_subcauses = []
    
    total = len(df_test)
    print(f"Début de l'évaluation enrichie sur {total} tickets...")
    
    for i, row in df_test.iterrows():
        # Afficher la progression
        if (i + 1) % 5 == 0 or i == 0 or i == total - 1:
            print(f"Traitement du ticket {i+1}/{total}...")
        
        # Classifier le ticket avec l'approche enrichie
        result = classify_ticket_hierarchical_enriched(
            classifier, row['text_ticket'], causes_dict, subcauses_dict, multi_label=False
        )
        
        # Stocker les prédictions
        predicted_cause = result['primary_cause']
        predicted_subcause = result['primary_sub_cause']['sub_cause'] if result['primary_sub_cause'] else "N/A"
        
        # Enregistrer les résultats pour l'évaluation
        y_true_causes.append(row['cause_attendue'])
        y_pred_causes.append(predicted_cause)
        
        y_true_subcauses.append(row['souscause_attendue'])
        y_pred_subcauses.append(predicted_subcause)
        
        # Détails de la prédiction
        prediction = {
            'ticket': row['text_ticket'][:50] + "..." if len(row['text_ticket']) > 50 else row['text_ticket'],
            'true_cause': row['cause_attendue'],
            'pred_cause': predicted_cause,
            'cause_correct': predicted_cause == row['cause_attendue'],
            'true_subcause': row['souscause_attendue'],
            'pred_subcause': predicted_subcause,
            'subcause_correct': predicted_subcause == row['souscause_attendue'],
            'cause_score': result['primary_score'],
            'subcause_score': result['primary_sub_cause']['sub_score'] if result['primary_sub_cause'] else 0
        }
        predictions.append(prediction)
    
    # Calculer les métriques
    cause_accuracy = accuracy_score(y_true_causes, y_pred_causes)
    subcause_accuracy = accuracy_score(y_true_subcauses, y_pred_subcauses)
    
    # Créer un DataFrame pour les prédictions détaillées
    df_predictions = pd.DataFrame(predictions)
    
    # Calculer le taux de réussite global (cause ET sous-cause correctes)
    overall_accuracy = (df_predictions['cause_correct'] & df_predictions['subcause_correct']).mean()
    
    # Métriques détaillées pour les causes
    cause_precision, cause_recall, cause_f1, _ = precision_recall_fscore_support(
        y_true_causes, y_pred_causes, average='weighted'
    )
    
    # Résultats
    resultats = {
        'df_predictions': df_predictions,
        'cause_accuracy': cause_accuracy,
        'subcause_accuracy': subcause_accuracy,
        'overall_accuracy': overall_accuracy,
        'cause_precision': cause_precision,
        'cause_recall': cause_recall,
        'cause_f1': cause_f1,
        'y_true_causes': y_true_causes,
        'y_pred_causes': y_pred_causes
    }
    
    return resultats
    

Fonction principale pour exécuter le test avec descriptions enrichies


def tester_modele_enrichi():
    """Fonction principale pour tester le modèle avec descriptions enrichies"""
    # Paramètres
    MODEL_PATH = "./mDeBERTa-v3-base-multilingual-nli-local"
    
    # Configuration
    print("Initialisation du classificateur...")
    classifier = setup_zero_shot_classifier(MODEL_PATH)
    
    # Créer le jeu de données de test
    df_test = creer_dataset_test()
    
    # Évaluer le modèle avec des descriptions enrichies
    print("Évaluation du modèle avec descriptions enrichies...")
    resultats = evaluer_modele_enrichi(classifier, df_test)
    
    # Afficher les résultats
    afficher_resultats(resultats)
    
    # Sauvegarder les prédictions détaillées
    resultats['df_predictions'].to_csv('predictions_enrichies.csv', index=False)
    print("Prédictions détaillées sauvegardées dans 'predictions_enrichies.csv'")
    
    return resultats

if __name__ == "__main__":
    resultats_enrichis = tester_modele_enrichi()
    

    
--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------

Modifications pour Intégrer le Prompt Engineering

def define_taxonomy_simplified():
    """
    Définit une taxonomie avec des descriptions concises et ciblées.
    
    Returns:
        tuple: (causes_simplified, subcauses_simplified)
    """
    # Définitions concises des causes principales
    causes_simplified = {
        "Matériel-Infrastructure": "Problème avec un équipement physique informatique (serveur, ordinateur, imprimante)",
        "Logiciel": "Dysfonctionnement d'une application ou système d'exploitation",
        "Intervention": "Incident suite à une action technique réalisée par un technicien",
        "Non déterminée": "Cause non identifiée, nécessitant plus d'analyse",
        "Environnementale": "Problème lié à l'électricité, la température ou d'autres facteurs externes",
        "Erreur à l'exécution": "Programme qui échoue pendant son fonctionnement",
        "Programme défaillant": "Bug ou défaut structurel dans un logiciel",
        "Sécurité": "Incident lié aux accès non autorisés ou vulnérabilités",
        "Réseau": "Problème de connexion ou communication entre systèmes",
        "Base de données": "Problème d'accès ou d'intégrité des données stockées",
        "Erreur utilisateur": "Erreur causée par une manipulation incorrecte d'un utilisateur",
        "Configuration": "Paramètres système incorrects ou incompatibles",
        "Authentification": "Difficulté à se connecter ou vérifier l'identité",
        "Permissions": "Problème de droits d'accès aux ressources",
        "Maintenance": "Interruption planifiée pour mise à jour ou entretien"
    }
    
    # Définition concise des sous-causes avec exemples concrets
    subcauses_simplified = {
        "Matériel-Infrastructure": {
            "Panne": "Équipement totalement non fonctionnel",
            "Pièce défectueuse": "Composant spécifique défaillant (disque dur, carte, écran)",
            "Configuration réseau": "Problème de paramétrage matériel du réseau",
            "Capacité insuffisante": "Ressources matérielles insuffisantes (mémoire, processeur)",
            "Périphérique non reconnu": "Équipement non détecté par le système"
        },
        "Logiciel": {
            "Bug": "Comportement incorrect du programme",
            "Paramétrage incorrect": "Options logicielles mal configurées",
            "Configuration": "Fichiers de configuration incorrects",
            "Version obsolète": "Logiciel non mis à jour",
            "Compatibilité": "Conflit entre applications",
            "Mise à jour requise": "Mise à jour nécessaire pour fonctionner",
            "Espace disque": "Manque d'espace pour le logiciel"
        },
        
        # Autres sous-causes simplifiées pour les autres catégories...
        "Intervention": {
            "Et supplier": "Intervention d'un prestataire externe problématique",
            "Et standard": "Intervention technique standard ayant causé un problème",
            "Procédure incorrecte": "Non-respect des procédures d'intervention",
            "Paramétrage incorrect": "Mauvais paramètres appliqués pendant l'intervention",
            "Erreur humaine": "Erreur du technicien lors de l'intervention"
        },
        
        "Non déterminée": {
            "Analyse en cours": "Investigation en cours sur la cause",
            "Information insuffisante": "Manque d'informations pour déterminer la cause",
            "Non reproductible": "Problème qui ne peut pas être reproduit pour analyse"
        },
        
        "Environnementale": {
            "Alimentation électrique": "Problème de courant ou coupure électrique",
            "Température": "Surchauffe ou température trop basse",
            "Humidité": "Niveau d'humidité inapproprié",
            "Ventilation": "Problème de refroidissement"
        },
        
        "Erreur à l'exécution": {
            "Timeout": "Dépassement du temps alloué pour l'opération",
            "Mémoire insuffisante": "Mémoire épuisée pendant l'exécution",
            "Exception non gérée": "Erreur imprévue pendant le fonctionnement",
            "Conflit de ressources": "Ressources utilisées par plusieurs processus"
        },
        
        "Programme défaillant": {
            "Boucle infinie": "Programme qui ne termine jamais",
            "Fuite mémoire": "Consommation excessive de mémoire",
            "Synchronisation": "Problème de coordination entre processus",
            "Deadlock": "Blocage mutuel entre processus"
        },
        
        "Sécurité": {
            "Accès non autorisé": "Utilisation du système sans permission",
            "Authentification échouée": "Échec de vérification d'identité",
            "Attaque malveillante": "Tentative délibérée de compromettre le système",
            "Vulnérabilité": "Faille de sécurité exploitable"
        },
        
        "Réseau": {
            "Connexion perdue": "Interruption de la communication réseau",
            "Latence élevée": "Temps de réponse anormalement longs",
            "DNS": "Problème de résolution de noms de domaine",
            "Routage": "Problème d'acheminement des paquets réseau"
        },
        
        "Base de données": {
            "Corruption": "Données endommagées ou invalides",
            "Verrouillage": "Blocage d'accès aux données",
            "Performance": "Requêtes anormalement lentes",
            "Espace disque": "Espace insuffisant pour les données"
        },
        
        "Erreur utilisateur": {
            "Formation insuffisante": "Manque de connaissance pour l'opération",
            "Erreur de saisie": "Information incorrecte introduite",
            "Procédure non suivie": "Non-respect des étapes requises"
        },
        
        "Configuration": {
            "Paramètres incorrects": "Valeurs de configuration erronées",
            "Conflit de configuration": "Paramètres incompatibles entre composants",
            "Valeurs par défaut inadaptées": "Paramètres standard inadéquats"
        },
        
        "Authentification": {
            "Identifiants expirés": "Mot de passe ou jeton expiré",
            "Compte verrouillé": "Accès bloqué après tentatives échouées",
            "SSO défaillant": "Authentification unique inter-applications défaillante"
        },
        
        "Permissions": {
            "Accès insuffisant": "Droits manquants pour l'opération",
            "Droits excessifs": "Privilèges trop élevés",
            "Groupe incorrect": "Appartenance à un groupe inadéquat"
        },
        
        "Maintenance": {
            "Planifiée": "Intervention prévue à l'avance",
            "Urgente": "Maintenance non planifiée pour problème critique",
            "Non communiquée": "Intervention sans notification préalable"
        }
    }
    
    return causes_simplified, subcauses_simplified
    
    
    def classify_with_prompt_engineering(classifier, ticket_text, causes_dict, subcauses_dict):
    """
    Classifie un ticket en utilisant des techniques de prompt engineering.
    
    Args:
        classifier: Pipeline de classification zero-shot
        ticket_text (str): Texte du ticket à classifier
        causes_dict (dict): Dictionnaire des causes avec descriptions simplifiées
        subcauses_dict (dict): Dictionnaire des sous-causes avec descriptions simplifiées
        
    Returns:
        dict: Résultats de classification avec cause et sous-cause
    """
    # 1. Construire un prompt bien structuré pour la classification des causes
    cause_prompt = (
        "En tant qu'expert en support informatique, analysez cet incident et déterminez sa cause principale.\n\n"
        f"Description de l'incident: {ticket_text}\n\n"
        "Choisissez la cause la plus probable parmi les suivantes:\n"
    )
    
    # Construire la liste des candidats pour les causes
    cause_candidates = [f"{cause}" for cause in causes_dict.keys()]
    
    # Classifier la cause principale
    cause_result = classifier(cause_prompt, cause_candidates, multi_label=False)
    
    # Extraire la cause principale prédite
    primary_cause = cause_result["labels"][0]
    primary_score = cause_result["scores"][0]
    
    # 2. Construire un prompt pour la classification des sous-causes
    if primary_cause in subcauses_dict:
        subcause_prompt = (
            f"Incident informatique: {ticket_text}\n\n"
            f"Cet incident a été identifié comme un problème de {primary_cause}.\n"
            f"Quelle est la sous-catégorie spécifique de ce problème?\n"
        )
        
        # Construire la liste des candidats pour les sous-causes
        subcause_candidates = list(subcauses_dict[primary_cause].keys())
        
        # Classifier la sous-cause
        subcause_result = classifier(subcause_prompt, subcause_candidates, multi_label=False)
        
        # Extraire la sous-cause prédite
        primary_subcause = subcause_result["labels"][0]
        primary_subcause_score = subcause_result["scores"][0]
        
        primary_sub = {
            "sub_cause": primary_subcause,
            "sub_score": primary_subcause_score,
            "all_sub_causes": list(zip(subcause_result["labels"], subcause_result["scores"]))
        }
    else:
        primary_sub = None
    
    # Compiler les résultats
    results = {
        "ticket_text": ticket_text,
        "primary_cause": primary_cause,
        "primary_score": primary_score,
        "primary_sub_cause": primary_sub,
        "all_causes": list(zip(cause_result["labels"], cause_result["scores"]))
    }
    
    return results
    
    def evaluer_modele_prompt_engineering(classifier, df_test):
    """
    Évalue le modèle avec prompt engineering sur le jeu de données de test.
    
    Args:
        classifier: Pipeline de classification zero-shot
        df_test (pd.DataFrame): DataFrame contenant les tickets de test
        
    Returns:
        dict: Résultats d'évaluation
    """
    # Obtenir la taxonomie simplifiée
    causes_dict, subcauses_dict = define_taxonomy_simplified()
    
    predictions = []
    y_true_causes = []
    y_pred_causes = []
    y_true_subcauses = []
    y_pred_subcauses = []
    
    total = len(df_test)
    print(f"Début de l'évaluation avec prompt engineering sur {total} tickets...")
    
    for i, row in df_test.iterrows():
        # Afficher la progression
        if (i + 1) % 5 == 0 or i == 0 or i == total - 1:
            print(f"Traitement du ticket {i+1}/{total}...")
        
        # Classifier le ticket avec prompt engineering
        result = classify_with_prompt_engineering(
            classifier, row['text_ticket'], causes_dict, subcauses_dict
        )
        
        # Stocker les prédictions
        predicted_cause = result['primary_cause']
        predicted_subcause = result['primary_sub_cause']['sub_cause'] if result['primary_sub_cause'] else "N/A"
        
        # Enregistrer les résultats pour l'évaluation
        y_true_causes.append(row['cause_attendue'])
        y_pred_causes.append(predicted_cause)
        
        y_true_subcauses.append(row['souscause_attendue'])
        y_pred_subcauses.append(predicted_subcause)
        
        # Détails de la prédiction
        prediction = {
            'ticket': row['text_ticket'][:50] + "..." if len(row['text_ticket']) > 50 else row['text_ticket'],
            'true_cause': row['cause_attendue'],
            'pred_cause': predicted_cause,
            'cause_correct': predicted_cause == row['cause_attendue'],
            'true_subcause': row['souscause_attendue'],
            'pred_subcause': predicted_subcause,
            'subcause_correct': predicted_subcause == row['souscause_attendue'],
            'cause_score': result['primary_score'],
            'subcause_score': result['primary_sub_cause']['sub_score'] if result['primary_sub_cause'] else 0
        }
        predictions.append(prediction)
    
    # Calculer les métriques de performance comme précédemment
    cause_accuracy = accuracy_score(y_true_causes, y_pred_causes)
    subcause_accuracy = accuracy_score(y_true_subcauses, y_pred_subcauses)
    
    # Créer un DataFrame pour les prédictions détaillées
    df_predictions = pd.DataFrame(predictions)
    
    # Calculer le taux de réussite global
    overall_accuracy = (df_predictions['cause_correct'] & df_predictions['subcause_correct']).mean()
    
    # Métriques détaillées pour les causes
    cause_precision, cause_recall, cause_f1, _ = precision_recall_fscore_support(
        y_true_causes, y_pred_causes, average='weighted'
    )
    
    # Résultats
    resultats = {
        'df_predictions': df_predictions,
        'cause_accuracy': cause_accuracy,
        'subcause_accuracy': subcause_accuracy,
        'overall_accuracy': overall_accuracy,
        'cause_precision': cause_precision,
        'cause_recall': cause_recall,
        'cause_f1': cause_f1,
        'y_true_causes': y_true_causes,
        'y_pred_causes': y_pred_causes
    }
    
    return resultats
    
    def tester_modele_prompt_engineering():
    """Fonction principale pour tester le modèle avec prompt engineering"""
    # Paramètres
    MODEL_PATH = "./mDeBERTa-v3-base-multilingual-nli-local"
    
    # Configuration
    print("Initialisation du classificateur...")
    classifier = setup_zero_shot_classifier(MODEL_PATH)
    
    # Créer le jeu de données de test
    df_test = creer_dataset_test()
    
    # Évaluer le modèle avec prompt engineering
    print("Évaluation du modèle avec prompt engineering...")
    resultats = evaluer_modele_prompt_engineering(classifier, df_test)
    
    # Afficher les résultats
    afficher_resultats(resultats)
    
    # Sauvegarder les prédictions détaillées
    resultats['df_predictions'].to_csv('predictions_prompt_engineering.csv', index=False)
    print("Prédictions détaillées sauvegardées dans 'predictions_prompt_engineering.csv'")
    
    return resultats

if __name__ == "__main__":
    resultats_prompt_engineering = tester_modele_prompt_engineering()
    
-------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------

APPROCHE DU FEW SHOT 

def create_representative_examples():
    """
    Crée une collection d'exemples représentatifs pour le few-shot learning.
    Ces exemples illustrent clairement les caractéristiques distinctives de chaque cause.
    
    Returns:
        dict: Dictionnaire des exemples de tickets par cause
    """
    examples_by_cause = {
        "Matériel-Infrastructure": [
            {
                "text": "L'écran de l'ordinateur reste noir au démarrage malgré l'alimentation fonctionnelle. Le voyant de l'écran est allumé mais rien ne s'affiche.",
                "cause": "Matériel-Infrastructure",
                "subcause": "Pièce défectueuse"
            },
            {
                "text": "L'imprimante du service comptabilité n'imprime plus. Nous avons vérifié les câbles et redémarré l'appareil sans succès.",
                "cause": "Matériel-Infrastructure",
                "subcause": "Panne"
            }
        ],
        "Logiciel": [
            {
                "text": "Excel se ferme systématiquement lors de l'ouverture de fichiers volumineux. Message d'erreur: 'Microsoft Excel a cessé de fonctionner'.",
                "cause": "Logiciel",
                "subcause": "Bug"
            },
            {
                "text": "Impossible d'utiliser la nouvelle version de SAP. L'application se lance mais se bloque lors de la saisie des données.",
                "cause": "Logiciel",
                "subcause": "Compatibilité"
            }
        ],
        "Intervention": [
            {
                "text": "Suite à l'intervention du prestataire sur le firewall hier, plusieurs utilisateurs ne peuvent plus accéder à l'intranet.",
                "cause": "Intervention",
                "subcause": "Et supplier"
            },
            {
                "text": "Après la mise à jour de Windows effectuée ce matin par l'équipe technique, plusieurs applications métiers ne fonctionnent plus.",
                "cause": "Intervention",
                "subcause": "Procédure incorrecte"
            }
        ],
        "Environnementale": [
            {
                "text": "Les serveurs du datacenter se sont arrêtés suite à une coupure électrique dans le quartier. Systèmes hors service jusqu'au retour du courant.",
                "cause": "Environnementale",
                "subcause": "Alimentation électrique"
            },
            {
                "text": "La salle serveur affiche une température de 32°C. Plusieurs équipements se sont mis en arrêt de sécurité suite à la défaillance de la climatisation.",
                "cause": "Environnementale",
                "subcause": "Température"
            }
        ],
        "Réseau": [
            {
                "text": "Impossible d'accéder aux ressources partagées et aux sites web externes depuis ce matin. Toutes les applications nécessitant Internet sont inaccessibles.",
                "cause": "Réseau",
                "subcause": "Connexion perdue"
            },
            {
                "text": "Les utilisateurs signalent des temps de réponse anormalement longs pour accéder aux applications web. Les pings vers le serveur montrent des délais de 500ms.",
                "cause": "Réseau",
                "subcause": "Latence élevée"
            }
        ],
        "Base de données": [
            {
                "text": "L'application métier affiche une erreur 'Impossible de se connecter à la base de données'. Les utilisateurs ne peuvent plus accéder à leurs données.",
                "cause": "Base de données",
                "subcause": "Connexion perdue"
            },
            {
                "text": "Les requêtes SQL prennent plus de 2 minutes pour s'exécuter alors qu'elles prenaient habituellement quelques secondes.",
                "cause": "Base de données",
                "subcause": "Performance"
            }
        ],
        # Ajout des autres causes
        "Erreur utilisateur": [
            {
                "text": "L'utilisateur a supprimé par erreur un dossier important sur le partage réseau. Il demande une restauration des données.",
                "cause": "Erreur utilisateur",
                "subcause": "Erreur de saisie"
            },
            {
                "text": "Le collaborateur a modifié des paramètres dans le panneau de configuration sans autorisation, causant un dysfonctionnement.",
                "cause": "Erreur utilisateur",
                "subcause": "Procédure non suivie"
            }
        ],
        "Authentification": [
            {
                "text": "L'utilisateur ne peut plus se connecter à son compte après plusieurs tentatives. Message: 'Compte verrouillé, contactez l'administrateur'.",
                "cause": "Authentification",
                "subcause": "Compte verrouillé"
            },
            {
                "text": "Impossible de se connecter au VPN d'entreprise. Le système indique que le mot de passe a expiré et doit être renouvelé.",
                "cause": "Authentification",
                "subcause": "Identifiants expirés"
            }
        ],
        # Continuer avec les autres causes dans le même format
        "Erreur à l'exécution": [
            {
                "text": "Le rapport financier mensuel ne se termine jamais. Il reste bloqué à 95% depuis plus de 2 heures.",
                "cause": "Erreur à l'exécution",
                "subcause": "Timeout"
            }
        ],
        "Non déterminée": [
            {
                "text": "L'application se comporte de manière erratique. Parfois elle fonctionne, parfois non. Nous analysons encore les logs pour comprendre.",
                "cause": "Non déterminée",
                "subcause": "Analyse en cours"
            }
        ],
        "Programme défaillant": [
            {
                "text": "Le logiciel de comptabilité consomme de plus en plus de mémoire RAM au fil du temps jusqu'à ce que l'ordinateur devienne inutilisable.",
                "cause": "Programme défaillant",
                "subcause": "Fuite mémoire"
            }
        ],
        "Sécurité": [
            {
                "text": "Plusieurs tentatives d'accès non autorisées détectées sur le serveur FTP. Les logs montrent des connexions depuis l'étranger.",
                "cause": "Sécurité",
                "subcause": "Accès non autorisé"
            }
        ],
        "Configuration": [
            {
                "text": "Après changement du fichier de configuration, l'application ne démarre plus. Message d'erreur: 'Paramètre invalide dans config.xml'.",
                "cause": "Configuration",
                "subcause": "Paramètres incorrects"
            }
        ],
        "Permissions": [
            {
                "text": "L'utilisateur ne peut pas accéder aux documents du projet malgré son appartenance à l'équipe. Message: 'Accès refusé'.",
                "cause": "Permissions",
                "subcause": "Accès insuffisant"
            }
        ],
        "Maintenance": [
            {
                "text": "Le système de facturation sera indisponible ce soir de 22h à 2h du matin pour mise à jour planifiée.",
                "cause": "Maintenance",
                "subcause": "Planifiée"
            }
        ]
    }
    
    return examples_by_cause
    
    
    def classify_with_fewshot(classifier, ticket_text, examples_by_cause, causes_dict, subcauses_dict, num_examples=2):
    """
    Classifie un ticket en utilisant l'approche few-shot learning.
    
    Args:
        classifier: Pipeline de classification zero-shot
        ticket_text (str): Texte du ticket à classifier
        examples_by_cause (dict): Dictionnaire d'exemples par cause
        causes_dict (dict): Dictionnaire des causes avec descriptions simplifiées
        subcauses_dict (dict): Dictionnaire des sous-causes avec descriptions simplifiées
        num_examples (int): Nombre d'exemples à inclure pour chaque cause
        
    Returns:
        dict: Résultats de classification avec cause et sous-cause
    """
    # 1. Construire un prompt few-shot pour la classification des causes
    cause_prompt = (
        "Voici des exemples de tickets d'incidents informatiques avec leur classification:\n\n"
    )
    
    # Ajouter des exemples pour aider à la classification
    for cause in causes_dict.keys():
        if cause in examples_by_cause and len(examples_by_cause[cause]) > 0:
            # Limiter le nombre d'exemples par cause
            examples = examples_by_cause[cause][:num_examples]
            for example in examples:
                cause_prompt += f"Incident: {example['text']}\nCause: {example['cause']}\n\n"
    
    # Ajouter le ticket à classifier
    cause_prompt += f"Incident: {ticket_text}\nCause:"
    
    # Construire la liste des candidats pour les causes
    cause_candidates = list(causes_dict.keys())
    
    # Classifier la cause principale
    cause_result = classifier(cause_prompt, cause_candidates, multi_label=False)
    
    # Extraire la cause principale prédite
    primary_cause = cause_result["labels"][0]
    primary_score = cause_result["scores"][0]
    
    # 2. Construire un prompt few-shot pour la classification des sous-causes
    if primary_cause in subcauses_dict:
        subcause_prompt = (
            f"La cause de cet incident a été identifiée comme: {primary_cause}\n\n"
            f"Voici des exemples de sous-causes pour des incidents de type '{primary_cause}':\n\n"
        )
        
        # Ajouter des exemples spécifiques à cette cause
        if primary_cause in examples_by_cause:
            examples = examples_by_cause[primary_cause][:num_examples]
            for example in examples:
                subcause_prompt += f"Incident: {example['text']}\nSous-cause: {example['subcause']}\n\n"
        
        # Ajouter le ticket à classifier
        subcause_prompt += f"Incident: {ticket_text}\nSous-cause:"
        
        # Construire la liste des candidats pour les sous-causes
        subcause_candidates = list(subcauses_dict[primary_cause].keys())
        
        # Classifier la sous-cause
        subcause_result = classifier(subcause_prompt, subcause_candidates, multi_label=False)
        
        # Extraire la sous-cause prédite
        primary_subcause = subcause_result["labels"][0]
        primary_subcause_score = subcause_result["scores"][0]
        
        primary_sub = {
            "sub_cause": primary_subcause,
            "sub_score": primary_subcause_score,
            "all_sub_causes": list(zip(subcause_result["labels"], subcause_result["scores"]))
        }
    else:
        primary_sub = None
    
    # Compiler les résultats
    results = {
        "ticket_text": ticket_text,
        "primary_cause": primary_cause,
        "primary_score": primary_score,
        "primary_sub_cause": primary_sub,
        "all_causes": list(zip(cause_result["labels"], cause_result["scores"]))
    }
    
    return results
    
    
    def evaluer_modele_fewshot(classifier, df_test, num_examples=2):
    """
    Évalue le modèle avec few-shot learning sur le jeu de données de test.
    
    Args:
        classifier: Pipeline de classification zero-shot
        df_test (pd.DataFrame): DataFrame contenant les tickets de test
        num_examples (int): Nombre d'exemples à inclure par cause
        
    Returns:
        dict: Résultats d'évaluation
    """
    # Obtenir la taxonomie simplifiée
    causes_dict, subcauses_dict = define_taxonomy_simplified()
    
    # Créer les exemples représentatifs
    examples_by_cause = create_representative_examples()
    
    predictions = []
    y_true_causes = []
    y_pred_causes = []
    y_true_subcauses = []
    y_pred_subcauses = []
    
    total = len(df_test)
    print(f"Début de l'évaluation avec few-shot learning sur {total} tickets...")
    print(f"Utilisation de {num_examples} exemples par cause.")
    
    for i, row in df_test.iterrows():
        # Afficher la progression
        if (i + 1) % 5 == 0 or i == 0 or i == total - 1:
            print(f"Traitement du ticket {i+1}/{total}...")
        
        # Classifier le ticket avec few-shot learning
        result = classify_with_fewshot(
            classifier, 
            row['text_ticket'], 
            examples_by_cause, 
            causes_dict, 
            subcauses_dict,
            num_examples
        )
        
        # Stocker les prédictions
        predicted_cause = result['primary_cause']
        predicted_subcause = result['primary_sub_cause']['sub_cause'] if result['primary_sub_cause'] else "N/A"
        
        # Enregistrer les résultats pour l'évaluation
        y_true_causes.append(row['cause_attendue'])
        y_pred_causes.append(predicted_cause)
        
        y_true_subcauses.append(row['souscause_attendue'])
        y_pred_subcauses.append(predicted_subcause)
        
        # Détails de la prédiction
        prediction = {
            'ticket': row['text_ticket'][:50] + "..." if len(row['text_ticket']) > 50 else row['text_ticket'],
            'true_cause': row['cause_attendue'],
            'pred_cause': predicted_cause,
            'cause_correct': predicted_cause == row['cause_attendue'],
            'true_subcause': row['souscause_attendue'],
            'pred_subcause': predicted_subcause,
            'subcause_correct': predicted_subcause == row['souscause_attendue'],
            'cause_score': result['primary_score'],
            'subcause_score': result['primary_sub_cause']['sub_score'] if result['primary_sub_cause'] else 0
        }
        predictions.append(prediction)
    
    # Calculer les métriques
    cause_accuracy = accuracy_score(y_true_causes, y_pred_causes)
    subcause_accuracy = accuracy_score(y_true_subcauses, y_pred_subcauses)
    
    # Créer un DataFrame pour les prédictions détaillées
    df_predictions = pd.DataFrame(predictions)
    
    # Calculer le taux de réussite global
    overall_accuracy = (df_predictions['cause_correct'] & df_predictions['subcause_correct']).mean()
    
    # Métriques détaillées pour les causes
    cause_precision, cause_recall, cause_f1, _ = precision_recall_fscore_support(
        y_true_causes, y_pred_causes, average='weighted'
    )
    
    # Résultats
    resultats = {
        'df_predictions': df_predictions,
        'cause_accuracy': cause_accuracy,
        'subcause_accuracy': subcause_accuracy,
        'overall_accuracy': overall_accuracy,
        'cause_precision': cause_precision,
        'cause_recall': cause_recall,
        'cause_f1': cause_f1,
        'y_true_causes': y_true_causes,
        'y_pred_causes': y_pred_causes
    }
    
    return resultats
    
    
    def tester_modele_fewshot(num_examples=2):
    """
    Fonction principale pour tester le modèle avec few-shot learning
    
    Args:
        num_examples (int): Nombre d'exemples à inclure par cause
    
    Returns:
        dict: Résultats de l'évaluation
    """
    # Paramètres
    MODEL_PATH = "./mDeBERTa-v3-base-multilingual-nli-local"
    
    # Configuration
    print("Initialisation du classificateur...")
    classifier = setup_zero_shot_classifier(MODEL_PATH)
    
    # Créer le jeu de données de test
    df_test = creer_dataset_test()
    
    # Évaluer le modèle avec few-shot learning
    print(f"Évaluation du modèle avec few-shot learning ({num_examples} exemples par cause)...")
    resultats = evaluer_modele_fewshot(classifier, df_test, num_examples)
    
    # Afficher les résultats
    afficher_resultats(resultats)
    
    # Sauvegarder les prédictions détaillées
    resultats['df_predictions'].to_csv(f'predictions_fewshot_{num_examples}_examples.csv', index=False)
    print(f"Prédictions détaillées sauvegardées dans 'predictions_fewshot_{num_examples}_examples.csv'")
    
    return resultats

if __name__ == "__main__":
    # Tester avec différents nombres d'exemples
    resultats_fewshot = tester_modele_fewshot(num_examples=2)
    
----------------------------------------------------------------------------------
----------------------------------------------------------------------------------
----------------------------------------------------------------------------------
-------------------------------------------------------------------------------------

Implémentation d'une Approche Hybride: LLM + Variables Catégorielles

def create_category_mappings(training_data):
    """
    Crée des mappings entre les variables catégorielles et les causes basés sur les fréquences observées.
    
    Args:
        training_data (pd.DataFrame): DataFrame contenant les données d'entraînement avec 
                                     'Groupe affecté', 'Service métier', et 'cause'
    
    Returns:
        dict: Dictionnaires de mappings pour chaque variable catégorielle
    """
    # Créer des mappings vides
    groupe_to_cause = {}
    service_to_cause = {}
    
    # Calculer les distributions de probabilité pour "Groupe affecté"
    groupe_cause_counts = training_data.groupby(['Groupe affecté', 'cause']).size().reset_index(name='count')
    for groupe in groupe_cause_counts['Groupe affecté'].unique():
        subset = groupe_cause_counts[groupe_cause_counts['Groupe affecté'] == groupe]
        total = subset['count'].sum()
        probs = {}
        for _, row in subset.iterrows():
            probs[row['cause']] = row['count'] / total
        groupe_to_cause[groupe] = probs
    
    # Calculer les distributions de probabilité pour "Service métier"
    service_cause_counts = training_data.groupby(['Service métier', 'cause']).size().reset_index(name='count')
    for service in service_cause_counts['Service métier'].unique():
        subset = service_cause_counts[service_cause_counts['Service métier'] == service]
        total = subset['count'].sum()
        probs = {}
        for _, row in subset.iterrows():
            probs[row['cause']] = row['count'] / total
        service_to_cause[service] = probs
    
    return {
        'Groupe affecté': groupe_to_cause,
        'Service métier': service_to_cause
    }
    
    
def classify_hybrid(classifier, ticket_text, groupe_affecte, service_metier, 
                   category_mappings, causes_dict, subcauses_dict,
                   llm_weight=0.6, groupe_weight=0.25, service_weight=0.15):
    """
    Classifie un ticket en utilisant à la fois le LLM et les variables catégorielles.
    
    Args:
        classifier: Pipeline de classification zero-shot
        ticket_text (str): Texte du ticket à classifier
        groupe_affecte (str): Valeur de la variable "Groupe affecté"
        service_metier (str): Valeur de la variable "Service métier"
        category_mappings (dict): Mappings entre variables catégorielles et causes
        causes_dict (dict): Dictionnaire des causes avec descriptions
        subcauses_dict (dict): Dictionnaire des sous-causes avec descriptions
        llm_weight (float): Poids attribué à la prédiction du LLM
        groupe_weight (float): Poids attribué à la prédiction basée sur "Groupe affecté"
        service_weight (float): Poids attribué à la prédiction basée sur "Service métier"
        
    Returns:
        dict: Résultats de classification avec cause et sous-cause
    """
    # Liste des causes possibles
    causes = list(causes_dict.keys())
    
    # 1. Obtenir la prédiction du LLM avec prompt engineering
    cause_prompt = (
        "En tant qu'expert en support informatique, analysez cet incident et déterminez sa cause principale.\n\n"
        f"Description de l'incident: {ticket_text}\n\n"
        "Choisissez la cause la plus probable parmi les suivantes:\n"
    )
    
    llm_result = classifier(cause_prompt, causes, multi_label=False)
    llm_predictions = {label: score for label, score in zip(llm_result["labels"], llm_result["scores"])}
    
    # 2. Obtenir les probabilités basées sur "Groupe affecté"
    groupe_predictions = {}
    if groupe_affecte in category_mappings['Groupe affecté']:
        groupe_predictions = category_mappings['Groupe affecté'][groupe_affecte]
    else:
        # Si le groupe n'est pas connu, distribution uniforme
        for cause in causes:
            groupe_predictions[cause] = 1.0 / len(causes)
    
    # 3. Obtenir les probabilités basées sur "Service métier"
    service_predictions = {}
    if service_metier in category_mappings['Service métier']:
        service_predictions = category_mappings['Service métier'][service_metier]
    else:
        # Si le service n'est pas connu, distribution uniforme
        for cause in causes:
            service_predictions[cause] = 1.0 / len(causes)
    
    # 4. Combiner les prédictions avec les poids respectifs
    combined_predictions = {}
    for cause in causes:
        llm_score = llm_predictions.get(cause, 0.0)
        groupe_score = groupe_predictions.get(cause, 0.0)
        service_score = service_predictions.get(cause, 0.0)
        
        combined_score = (
            llm_weight * llm_score + 
            groupe_weight * groupe_score + 
            service_weight * service_score
        )
        
        combined_predictions[cause] = combined_score
    
    # 5. Sélectionner la cause avec le score combiné le plus élevé
    primary_cause = max(combined_predictions.items(), key=lambda x: x[1])[0]
    primary_score = combined_predictions[primary_cause]
    
    # 6. Utiliser le LLM pour prédire la sous-cause basée sur la cause principale
    if primary_cause in subcauses_dict:
        subcause_prompt = (
            f"Incident informatique: {ticket_text}\n\n"
            f"Groupe affecté: {groupe_affecte}\n"
            f"Service métier: {service_metier}\n\n"
            f"Cet incident a été identifié comme un problème de {primary_cause}.\n"
            f"Quelle est la sous-catégorie spécifique de ce problème?\n"
        )
        
        subcause_candidates = list(subcauses_dict[primary_cause].keys())
        subcause_result = classifier(subcause_prompt, subcause_candidates, multi_label=False)
        
        primary_subcause = subcause_result["labels"][0]
        primary_subcause_score = subcause_result["scores"][0]
        
        primary_sub = {
            "sub_cause": primary_subcause,
            "sub_score": primary_subcause_score,
            "all_sub_causes": list(zip(subcause_result["labels"], subcause_result["scores"]))
        }
    else:
        primary_sub = None
    
    # 7. Compiler les résultats
    all_causes_sorted = sorted(combined_predictions.items(), key=lambda x: x[1], reverse=True)
    
    results = {
        "ticket_text": ticket_text,
        "groupe_affecte": groupe_affecte,
        "service_metier": service_metier,
        "primary_cause": primary_cause,
        "primary_score": primary_score,
        "primary_sub_cause": primary_sub,
        "llm_prediction": llm_result["labels"][0],
        "llm_score": llm_result["scores"][0],
        "groupe_prediction": max(groupe_predictions.items(), key=lambda x: x[1])[0] if groupe_predictions else None,
        "service_prediction": max(service_predictions.items(), key=lambda x: x[1])[0] if service_predictions else None,
        "all_causes": all_causes_sorted
    }
    
    return results


def creer_dataset_test_avec_categories():
    """
    Crée un jeu de données de test avec les variables catégorielles.
    
    Returns:
        pd.DataFrame: DataFrame contenant les tickets de test avec variables catégorielles
    """
    # Dictionnaire de base pour les tickets
    data_dict = {
        'text_ticket': [
            "Écran qui reste noir après démarrage du poste de travail. Le voyant d'alimentation est allumé mais rien ne s'affiche.",
            "Impossible d'accéder à SAP. Message d'erreur : \"La connexion à la base de données a échoué\".",
            "Le collaborateur n'arrive pas à se connecter à son compte après 3 tentatives. Son compte est verrouillé.",
            "La mise à jour de Windows 11 a échoué avec le code d'erreur 0x80070070. Espace disque insuffisant sur le lecteur C.",
            "Le serveur de fichiers est inaccessible depuis ce matin suite à une coupure électrique dans le bâtiment B.",
            "L'application CRM se ferme de manière inattendue lors de l'exportation des rapports. Message : violation d'accès mémoire.",
            "Lenteur extrême sur le réseau dans les bureaux du 3ème étage. Les temps de réponse dépassent 200ms.",
            "Une mise à jour de sécurité a été appliquée sur tous les postes du service comptabilité conformément au planning de maintenance.",
            "L'utilisateur ne peut pas accéder au partage réseau. Message : \"Vous n'avez pas les droits suffisants pour accéder à cette ressource\".",
            "L'application Microsoft365 affiche une erreur lors du lancement. Le redémarrage du programme a résolu le problème.",
            "La sauvegarde automatique a échoué car l'utilisateur a modifié le nom du répertoire cible.",
            "Tous les utilisateurs du service RH signalent l'impossibilité d'accéder à l'intranet. Problème identifié au niveau du serveur DNS.",
            "Le système de climatisation de la salle serveur s'est arrêté, causant une surchauffe et l'arrêt automatique de trois serveurs.",
            "L'utilisateur a tenté d'installer un logiciel non autorisé, bloqué par la politique de sécurité.",
            "Suite à l'intervention du prestataire externe sur le firewall, plusieurs services sont devenus inaccessibles.",
            "L'application reste en analyse depuis 3 jours. L'équipe technique n'a pas encore identifié la source du problème.",
            "Lors de l'exécution du rapport financier, le système affiche une erreur de délai d'attente après 30 minutes.",
            "Le VPN ne fonctionne pas correctement. Les utilisateurs peuvent se connecter mais perdent la connexion après quelques minutes.",
            "Suite à la mise à jour du navigateur, l'application web interne affiche des erreurs JavaScript.",
            "L'imprimante du service marketing reste bloquée sur \"En attente\" malgré plusieurs redémarrages."
        ],
        'cause_attendue': [
            "Matériel-Infrastructure",
            "Base de données",
            "Authentification",
            "Logiciel",
            "Environnementale",
            "Programme défaillant",
            "Réseau",
            "Maintenance",
            "Permissions",
            "Logiciel",
            "Erreur utilisateur",
            "Réseau",
            "Environnementale",
            "Sécurité",
            "Intervention",
            "Non déterminée",
            "Erreur à l'exécution",
            "Configuration",
            "Logiciel",
            "Matériel-Infrastructure"
        ],
        'souscause_attendue': [
            "Pièce défectueuse",
            "Connexion perdue",
            "Compte verrouillé",
            "Espace disque",
            "Alimentation électrique",
            "Fuite mémoire",
            "Latence élevée",
            "Planifiée",
            "Accès insuffisant",
            "Bug",
            "Procédure non suivie",
            "DNS",
            "Température",
            "Accès non autorisé",
            "Et supplier",
            "Analyse en cours",
            "Timeout",
            "Paramètres incorrects",
            "Compatibilité",
            "Configuration réseau"
        ],
        # Ajout des variables catégorielles
        'Groupe affecté': [
            "Support-Poste-Travail",
            "Support-Applications",
            "Sécu-IAM",
            "Support-Poste-Travail",
            "Support-Infrastructure",
            "Support-Applications",
            "Support-Réseau",
            "Support-Infrastructure",
            "Support-Réseau",
            "Support-Poste-Travail",
            "Support-Data",
            "Support-Réseau",
            "Support-Infrastructure",
            "Sécu-IAM",
            "Support-Infrastructure",
            "Support-Applications",
            "Support-Applications",
            "Support-Réseau",
            "Support-Applications",
            "Support-Périphériques"
        ],
        'Service métier': [
            "Poste_Utilisateur",
            "ERP_Finance",
            "IAM_Authentification",
            "OS_Windows",
            "Stockage_Fichiers",
            "CRM_Commercial",
            "Réseau_LAN",
            "Patching_Sécurité",
            "Partage_Fichiers",
            "Bureautique_Office",
            "Sauvegarde_Données",
            "Réseau_Intranet",
            "Datacenter_Locaux",
            "Sécurité_EndPoint",
            "Réseau_Sécurité",
            "Applications_Métier",
            "Reporting_Financier",
            "Réseau_VPN",
            "Applications_Web",
            "Périphériques_Impression"
        ]
    }
    
    # Convertir en DataFrame pandas
    df = pd.DataFrame(data_dict)
    
    print("Jeu de données avec variables catégorielles créé:")
    print(df[['text_ticket', 'Groupe affecté', 'Service métier', 'cause_attendue']].head())
    
    # Exporter en CSV
    nom_fichier = "test_tickets_metis_with_categories.csv"
    df.to_csv(nom_fichier, index=False)
    print(f"Jeu de données exporté avec succès vers {nom_fichier}")
    
    return df


def creer_dataset_entrainement():
    """
    Crée un jeu de données d'entraînement fictif pour générer les mappings.
    Dans un cas réel, vous utiliseriez vos 874 tickets fiables.
    
    Returns:
        pd.DataFrame: DataFrame contenant les données d'entraînement
    """
    # Les données ci-dessous sont fictives mais représentatives
    # Dans un cas réel, vous utiliseriez vos 874 tickets fiables
    data = {
        'Groupe affecté': [
            "Support-Poste-Travail", "Support-Poste-Travail", "Support-Poste-Travail", 
            "Support-Applications", "Support-Applications", "Support-Applications",
            "Support-Infrastructure", "Support-Infrastructure", "Support-Infrastructure",
            "Support-Réseau", "Support-Réseau", "Support-Réseau",
            "Sécu-IAM", "Sécu-IAM", "Support-Périphériques"
        ],
        'Service métier': [
            "Poste_Utilisateur", "OS_Windows", "Bureautique_Office",
            "ERP_Finance", "CRM_Commercial", "Applications_Métier",
            "Stockage_Fichiers", "Datacenter_Locaux", "Patching_Sécurité",
            "Réseau_LAN", "Réseau_Intranet", "Réseau_VPN",
            "IAM_Authentification", "Sécurité_EndPoint", "Périphériques_Impression"
        ],
        'cause': [
            "Matériel-Infrastructure", "Logiciel", "Logiciel",
            "Logiciel", "Programme défaillant", "Erreur à l'exécution",
            "Environnementale", "Matériel-Infrastructure", "Maintenance",
            "Réseau", "Réseau", "Configuration",
            "Authentification", "Sécurité", "Matériel-Infrastructure"
        ]
    }
    
    # Ajouter plus d'entrées pour obtenir des distributions plus robustes
    # Ces lignes font que "Support-Poste-Travail" est fortement associé à "Matériel-Infrastructure"
    for _ in range(5):
        data['Groupe affecté'].append("Support-Poste-Travail")
        data['Service métier'].append("Poste_Utilisateur")
        data['cause'].append("Matériel-Infrastructure")
    
    # "Support-Applications" est fortement associé à "Logiciel"
    for _ in range(5):
        data['Groupe affecté'].append("Support-Applications")
        data['Service métier'].append("Applications_Métier")
        data['cause'].append("Logiciel")
    
    # "Support-Réseau" est fortement associé à "Réseau"
    for _ in range(5):
        data['Groupe affecté'].append("Support-Réseau")
        data['Service métier'].append("Réseau_LAN")
        data['cause'].append("Réseau")
    
    return pd.DataFrame(data)


def evaluer_modele_hybride(classifier, df_test, category_mappings,
                         llm_weight=0.6, groupe_weight=0.25, service_weight=0.15):
    """
    Évalue le modèle hybride sur le jeu de données de test.
    
    Args:
        classifier: Pipeline de classification zero-shot
        df_test (pd.DataFrame): DataFrame contenant les tickets de test avec variables catégorielles
        category_mappings (dict): Mappings entre variables catégorielles et causes
        llm_weight (float): Poids attribué à la prédiction du LLM
        groupe_weight (float): Poids attribué à la prédiction basée sur "Groupe affecté"
        service_weight (float): Poids attribué à la prédiction basée sur "Service métier"
        
    Returns:
        dict: Résultats d'évaluation
    """
    # Obtenir la taxonomie simplifiée
    causes_dict, subcauses_dict = define_taxonomy_simplified()
    
    predictions = []
    y_true_causes = []
    y_pred_causes = []
    y_true_subcauses = []
    y_pred_subcauses = []
    
    total = len(df_test)
    print(f"Début de l'évaluation hybride sur {total} tickets...")
    print(f"Poids - LLM: {llm_weight}, Groupe: {groupe_weight}, Service: {service_weight}")
    
    for i, row in df_test.iterrows():
        # Afficher la progression
        if (i + 1) % 5 == 0 or i == 0 or i == total - 1:
            print(f"Traitement du ticket {i+1}/{total}...")
        
        # Classifier le ticket avec l'approche hybride
        result = classify_hybrid(
            classifier, 
            row['text_ticket'], 
            row['Groupe affecté'],
            row['Service métier'],
            category_mappings,
            causes_dict, 
            subcauses_dict,
            llm_weight,
            groupe_weight,
            service_weight
        )
        
        # Stocker les prédictions
        predicted_cause = result['primary_cause']
        predicted_subcause = result['primary_sub_cause']['sub_cause'] if result['primary_sub_cause'] else "N/A"
        
        # Enregistrer les résultats pour l'évaluation
        y_true_causes.append(row['cause_attendue'])
        y_pred_causes.append(predicted_cause)
        
        y_true_subcauses.append(row['souscause_attendue'])
        y_pred_subcauses.append(predicted_subcause)
        
        # Détails de la prédiction
        prediction = {
            'ticket': row['text_ticket'][:50] + "..." if len(row['text_ticket']) > 50 else row['text_ticket'],
            'true_cause': row['cause_attendue'],
            'pred_cause': predicted_cause,
            'cause_correct': predicted_cause == row['cause_attendue'],
            'true_subcause': row['souscause_attendue'],
            'pred_subcause': predicted_subcause,
            'subcause_correct': predicted_subcause == row['souscause_attendue'],
            'cause_score': result['primary_score'],
            'subcause_score': result['primary_sub_cause']['sub_score'] if result['primary_sub_cause'] else 0,
            'llm_prediction': result['llm_prediction'],
            'groupe_prediction': result['groupe_prediction'],
            'service_prediction': result['service_prediction']
        }
        predictions.append(prediction)
    
    # Calculer les métriques
    cause_accuracy = accuracy_score(y_true_causes, y_pred_causes)
    subcause_accuracy = accuracy_score(y_true_subcauses, y_pred_subcauses)
    
    # Créer un DataFrame pour les prédictions détaillées
    df_predictions = pd.DataFrame(predictions)
    
    # Calculer le taux de réussite global
    overall_accuracy = (df_predictions['cause_correct'] & df_predictions['subcause_correct']).mean()
    
    # Métriques détaillées pour les causes
    cause_precision, cause_recall, cause_f1, _ = precision_recall_fscore_support(
        y_true_causes, y_pred_causes, average='weighted'
    )
    
    # Métriques pour comparer l'approche hybride avec le LLM seul
    llm_correct = (df_predictions['llm_prediction'] == df_predictions['true_cause']).mean()
    groupe_correct = (df_predictions['groupe_prediction'] == df_predictions['true_cause']).mean()
    service_correct = (df_predictions['service_prediction'] == df_predictions['true_cause']).mean()
    
    print(f"\nComparaison des précisions:")
    print(f"LLM seul: {llm_correct:.2%}")
    print(f"Groupe affecté seul: {groupe_correct:.2%}")
    print(f"Service métier seul: {service_correct:.2%}")
    print(f"Approche hybride: {cause_accuracy:.2%}")
    
    # Résultats
    resultats = {
        'df_predictions': df_predictions,
        'cause_accuracy': cause_accuracy,
        'subcause_accuracy': subcause_accuracy,
        'overall_accuracy': overall_accuracy,
        'cause_precision': cause_precision,
        'cause_recall': cause_recall,
        'cause_f1': cause_f1,
        'y_true_causes': y_true_causes,
        'y_pred_causes': y_pred_causes,
        'llm_accuracy': llm_correct,
        'groupe_accuracy': groupe_correct,
        'service_accuracy': service_correct
    }
    
    return resultats


def tester_modele_hybride(llm_weight=0.6, groupe_weight=0.25, service_weight=0.15):
    """
    Fonction principale pour tester le modèle hybride.
    
    Args:
        llm_weight (float): Poids attribué à la prédiction du LLM
        groupe_weight (float): Poids attribué à la prédiction basée sur "Groupe affecté"
        service_weight (float): Poids attribué à la prédiction basée sur "Service métier"
        
    Returns:
        dict: Résultats de l'évaluation
    """
    # Paramètres
    MODEL_PATH = "./mDeBERTa-v3-base-multilingual-nli-local"
    
    # Configuration
    print("Initialisation du classificateur...")
    classifier = setup_zero_shot_classifier(MODEL_PATH)
    
    # Créer le jeu de données d'entraînement fictif
    print("Création du jeu de données d'entraînement...")
    df_train = creer_dataset_entrainement()
    
    # Créer les mappings entre variables catégorielles et causes
    print("Création des mappings catégoriels...")
    category_mappings = create_category_mappings(df_train)
    
    # Créer le jeu de données de test avec variables catégorielles
    print("Création du jeu de données de test...")
    df_test = creer_dataset_test_avec_categories()
    
    # Évaluer le modèle hybride
    print("Évaluation du modèle hybride...")
    resultats = evaluer_modele_hybride(
        classifier, 
        df_test, 
        category_mappings,
        llm_weight,
        groupe_weight,
        service_weight
    )
    
    # Afficher les résultats
    afficher_resultats(resultats)
    
    # Sauvegarder les prédictions détaillées
    resultats['df_predictions'].to_csv('predictions_hybride.csv', index=False)
    print("Prédictions détaillées sauvegardées dans 'predictions_hybride.csv'")
    
    return resultats

if __name__ == "__main__":
    # Tester l'approche hybride avec les poids par défaut
    resultats_hybride = tester_modele_hybride()
    
    # On pourrait aussi tester différentes combinaisons de poids
    # resultats_plus_llm = tester_modele_hybride(0.8, 0.1, 0.1)
    # resultats_plus_categories = tester_modele_hybride(0.4, 0.4, 0.2)
    
    
------------------------------------------------------------------------------------
----------------------------------------------------------------------------------
------------------------------------------------------------------------------------

OPTIMISATION LLM + VARIABLES CATEGORIELLES 

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import uniform

def classify_hybrid(classifier, text_ticket, groupe_affecte, service_metier, 
                   category_mappings, causes_dict, subcauses_dict,
                   llm_weight=0.6, groupe_weight=0.25, service_weight=0.15):
    """
    Classifie un ticket en utilisant l'approche hybride.
    
    Args:
        classifier: Classificateur pré-entraîné
        text_ticket: Texte du ticket
        groupe_affecte: Groupe affecté
        service_metier: Service métier
        category_mappings: Mappings entre catégories et causes
        causes_dict: Dictionnaire des causes
        subcauses_dict: Dictionnaire des sous-causes
        llm_weight: Poids pour la prédiction LLM
        groupe_weight: Poids pour la prédiction basée sur le groupe
        service_weight: Poids pour la prédiction basée sur le service
        
    Returns:
        dict: Résultat de la classification
    """
    # 1. Prédiction LLM
    llm_result = classify_with_prompt_engineering(
        classifier, 
        text_ticket, 
        causes_dict, 
        subcauses_dict
    )
    llm_pred = llm_result['primary_cause']
    llm_score = float(llm_result['primary_score'])
    
    # 2. Prédiction basée sur le groupe
    groupe_pred = None
    groupe_score = 0.0
    if groupe_affecte in category_mappings['Groupe affecté']:
        groupe_probs = category_mappings['Groupe affecté'][groupe_affecte]
        if len(groupe_probs) > 0:
            groupe_pred, groupe_score = max(groupe_probs.items(), key=lambda x: x[1])
            groupe_score = float(groupe_score)
    
    # 3. Prédiction basée sur le service
    service_pred = None
    service_score = 0.0
    if service_metier in category_mappings['Service métier']:
        service_probs = category_mappings['Service métier'][service_metier]
        if len(service_probs) > 0:
            service_pred, service_score = max(service_probs.items(), key=lambda x: x[1])
            service_score = float(service_score)
    
    # Combiner les prédictions avec une méthode de vote pondéré
    vote_scores = {}
    
    # Ajouter la prédiction LLM si disponible
    if llm_pred:
        vote_scores[llm_pred] = vote_scores.get(llm_pred, 0) + llm_weight * llm_score
    
    # Ajouter la prédiction du groupe si disponible
    if groupe_pred:
        vote_scores[groupe_pred] = vote_scores.get(groupe_pred, 0) + groupe_weight * groupe_score
    
    # Ajouter la prédiction du service si disponible
    if service_pred:
        vote_scores[service_pred] = vote_scores.get(service_pred, 0) + service_weight * service_score
    
    # Si aucun vote, utiliser la prédiction LLM par défaut
    if not vote_scores:
        final_pred = llm_pred
        final_score = llm_score
    else:
        # Sélectionner la cause avec le score le plus élevé
        final_pred = max(vote_scores.items(), key=lambda x: x[1])[0]
        final_score = vote_scores[final_pred]
    
    return {
        'primary_cause': final_pred,
        'primary_score': final_score,
        'primary_sub_cause': llm_result['primary_sub_cause'],
        'llm_prediction': llm_pred,
        'groupe_prediction': groupe_pred,
        'service_prediction': service_pred,
        'vote_scores': vote_scores
    }

def predict_batch(classifier, X, category_mappings, causes_dict, subcauses_dict, 
                 llm_weight=0.6, groupe_weight=0.25, service_weight=0.15):
    """
    Prédit les causes pour un ensemble de tickets.
    
    Args:
        classifier: Classificateur pré-entraîné
        X: DataFrame contenant 'text_ticket', 'Groupe affecté' et 'Service métier'
        category_mappings: Mappings entre catégories et causes
        causes_dict: Dictionnaire des causes
        subcauses_dict: Dictionnaire des sous-causes
        llm_weight: Poids pour la prédiction LLM
        groupe_weight: Poids pour la prédiction basée sur le groupe
        service_weight: Poids pour la prédiction basée sur le service
        
    Returns:
        np.array: Prédictions de causes
    """
    predictions = []
    
    for _, row in X.iterrows():
        result = classify_hybrid(
            classifier, 
            row['text_ticket'], 
            row['Groupe affecté'],
            row['Service métier'],
            category_mappings,
            causes_dict, 
            subcauses_dict,
            llm_weight,
            groupe_weight,
            service_weight
        )
        predictions.append(result['primary_cause'])
    
    return np.array(predictions)

def custom_scoring(y_true, y_pred):
    """
    Score personnalisé combinant accuracy, recall et f1.
    """
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Score combiné: 40% accuracy, 30% recall, 30% F1
    return 0.4 * acc + 0.3 * recall + 0.3 * f1

def optimize_weights_with_randomsearch(classifier, df_train, df_test, category_mappings, 
                                     causes_dict, subcauses_dict, n_iter=20):
    """
    Optimise les poids en utilisant une approche RandomSearch.
    """
    print("Début de l'optimisation des poids avec RandomSearch...")
    
    # Liste pour stocker les résultats
    results = []
    
    # Générer des combinaisons aléatoires de paramètres
    param_combinations = []
    for _ in range(n_iter):
        llm_w = np.random.uniform(0.1, 0.9)
        groupe_w = np.random.uniform(0.0, 0.9 - llm_w)
        service_w = 1.0 - llm_w - groupe_w
        param_combinations.append({
            'llm_weight': llm_w,
            'groupe_weight': groupe_w,
            'service_weight': service_w
        })
    
    print(f"Test de {len(param_combinations)} combinaisons de poids")
    
    # Évaluer chaque combinaison
    for params in tqdm(param_combinations):
        # Effectuer les prédictions pour obtenir les métriques
        y_pred = predict_batch(
            classifier, 
            df_test, 
            category_mappings, 
            causes_dict, 
            subcauses_dict,
            params['llm_weight'],
            params['groupe_weight'],
            params['service_weight']
        )
        y_true = df_test['cause_attendue'].values
        
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        score = 0.4 * acc + 0.3 * recall + 0.3 * f1
        
        print(f"Poids: LLM={params['llm_weight']:.2f}, Groupe={params['groupe_weight']:.2f}, Service={params['service_weight']:.2f} → Score={score:.4f}, Accuracy={acc:.4f}")
        
        results.append((params, score, acc, recall, f1))
    
    # Trouver la meilleure combinaison
    results.sort(key=lambda x: x[1], reverse=True)
    best_params, best_score, best_acc, best_recall, best_f1 = results[0]
    
    print("\nMeilleurs poids trouvés:")
    print(f"- LLM weight: {best_params['llm_weight']:.2f}")
    print(f"- Groupe weight: {best_params['groupe_weight']:.2f}")
    print(f"- Service weight: {best_params['service_weight']:.2f}")
    print(f"- Score combiné: {best_score:.4f}")
    print(f"- Accuracy: {best_acc:.4f}")
    print(f"- Recall: {best_recall:.4f}")
    print(f"- F1 score: {best_f1:.4f}")
    
    # Visualiser les résultats
    visualize_weight_optimization([p for p, _, _, _, _ in results], [s for _, s, _, _, _ in results])
    
    return {
        'best_params': best_params,
        'score': best_score,
        'accuracy': best_acc,
        'recall': best_recall,
        'f1': best_f1,
        'all_results': results
    }

def optimize_weights_bayesian(classifier, df_train, df_test, category_mappings, 
                            causes_dict, subcauses_dict, n_iter=20):
    """
    Optimise les poids en utilisant l'approche bayésienne (scikit-optimize).
    Nécessite l'installation de scikit-optimize (pip install scikit-optimize)
    """
    # Vérification et installation de scikit-optimize si nécessaire
    try:
        import skopt
        from skopt import gp_minimize
        from skopt.space import Real
        from skopt.utils import use_named_args
        from skopt.plots import plot_convergence, plot_objective
    except ImportError:
        print("Erreur: scikit-optimize n'est pas installé.")
        print("Installation de scikit-optimize en cours...")
        
        try:
            import subprocess
            subprocess.check_call(["pip", "install", "scikit-optimize"])
            
            # Réessayer l'importation après installation
            from skopt import gp_minimize
            from skopt.space import Real
            from skopt.utils import use_named_args
            from skopt.plots import plot_convergence, plot_objective
            print("scikit-optimize installé avec succès.")
        except Exception as e:
            print(f"Échec de l'installation: {str(e)}")
            print("Utilisation de la méthode RandomSearch comme alternative...")
            return optimize_weights_with_randomsearch(
                classifier, df_train, df_test, category_mappings, 
                causes_dict, subcauses_dict, n_iter
            )
    
    print("Début de l'optimisation bayésienne des poids...")
    
    # Définir l'espace de recherche
    space = [
        Real(0.1, 0.9, name='llm_weight'),
        Real(0.0, 0.7, name='groupe_weight'),
        Real(0.0, 0.7, name='service_weight')
    ]
    
    # Fonction objectif à optimiser
    @use_named_args(space)
    def objective(llm_weight, groupe_weight, service_weight):
        # S'assurer que les poids somment à 1
        total = llm_weight + groupe_weight + service_weight
        llm_w = llm_weight / total
        groupe_w = groupe_weight / total
        service_w = service_weight / total
        
        # Effectuer les prédictions sur l'ensemble de test
        y_pred = predict_batch(
            classifier, 
            df_test, 
            category_mappings, 
            causes_dict, 
            subcauses_dict,
            llm_w,
            groupe_w,
            service_w
        )
        y_true = df_test['cause_attendue'].values
        
        # Calculer le score personnalisé (négatif car gp_minimize minimise)
        score = -custom_scoring(y_true, y_pred)
        
        print(f"Poids: LLM={llm_w:.2f}, Groupe={groupe_w:.2f}, Service={service_w:.2f} → Score={-score:.4f}")
        
        return score
    
    try:
        # Exécuter l'optimisation bayésienne
        result = gp_minimize(
            objective,
            space,
            n_calls=n_iter,
            random_state=42,
            verbose=True
        )
        
        # Extraire les meilleurs paramètres
        best_llm_w, best_groupe_w, best_service_w = result.x
        
        # Normaliser les poids
        total = best_llm_w + best_groupe_w + best_service_w
        best_llm_w /= total
        best_groupe_w /= total
        best_service_w /= total
        
        # Calculer les métriques avec les meilleurs poids
        y_pred = predict_batch(
            classifier, 
            df_test, 
            category_mappings, 
            causes_dict, 
            subcauses_dict,
            best_llm_w,
            best_groupe_w,
            best_service_w
        )
        y_true = df_test['cause_attendue'].values
        
        best_acc = accuracy_score(y_true, y_pred)
        best_recall = recall_score(y_true, y_pred, average='weighted')
        best_f1 = f1_score(y_true, y_pred, average='weighted')
        best_score = 0.4 * best_acc + 0.3 * best_recall + 0.3 * best_f1
        
        print("\nMeilleurs poids trouvés (optimisation bayésienne):")
        print(f"- LLM weight: {best_llm_w:.2f}")
        print(f"- Groupe weight: {best_groupe_w:.2f}")
        print(f"- Service weight: {best_service_w:.2f}")
        print(f"- Score combiné: {best_score:.4f}")
        print(f"- Accuracy: {best_acc:.4f}")
        print(f"- Recall: {best_recall:.4f}")
        print(f"- F1 score: {best_f1:.4f}")
        
        # Visualiser les résultats
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plot_convergence(result)
        plt.subplot(1, 2, 2)
        plot_objective(result)
        plt.tight_layout()
        plt.savefig('bayesian_optimization_results.png', dpi=300)
        plt.show()
        
        return {
            'best_params': {
                'llm_weight': best_llm_w,
                'groupe_weight': best_groupe_w,
                'service_weight': best_service_w
            },
            'score': best_score,
            'accuracy': best_acc,
            'recall': best_recall,
            'f1': best_f1,
            'result': result
        }
        
    except Exception as e:
        print(f"Erreur lors de l'optimisation bayésienne: {str(e)}")
        print("Utilisation de la méthode RandomSearch comme alternative...")
        
        # Fallback: utiliser RandomSearch si l'optimisation bayésienne échoue
        return optimize_weights_with_randomsearch(
            classifier, df_train, df_test, category_mappings, 
            causes_dict, subcauses_dict, n_iter
        )

# Conserver la fonction de visualisation existante
def visualize_weight_optimization(param_list, results):
    """
    Visualise les résultats de l'optimisation des poids.
    """
    # Créer un DataFrame pour la visualisation
    df_results = pd.DataFrame(param_list)
    df_results['score'] = results
    
    # Visualiser la relation entre les poids et les performances
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Distribution des scores
    plt.subplot(2, 2, 1)
    sns.histplot(df_results['score'], bins=20, kde=True)
    plt.title('Distribution des scores')
    plt.xlabel('Score combiné')
    plt.ylabel('Fréquence')
    
    # Plot 2: Relation LLM weight vs score
    plt.subplot(2, 2, 2)
    sns.scatterplot(x='llm_weight', y='score', data=df_results, alpha=0.7)
    plt.title('Impact du poids LLM sur le score')
    plt.xlabel('Poids LLM')
    plt.ylabel('Score combiné')
    
    # Plot 3: Graphique 3D des poids
    ax = plt.subplot(2, 2, 3, projection='3d')
    ax.scatter(
        df_results['llm_weight'], 
        df_results['groupe_weight'], 
        df_results['service_weight'], 
        c=df_results['score'], 
        cmap='viridis',
        alpha=0.7
    )
    ax.set_xlabel('Poids LLM')
    ax.set_ylabel('Poids Groupe')
    ax.set_zlabel('Poids Service')
    ax.set_title('Espace des poids')
    
    # Plot 4: Heatmap des scores par paires de poids
    plt.subplot(2, 2, 4)
    pivot = df_results.round({'llm_weight': 1, 'groupe_weight': 1}).groupby(['llm_weight', 'groupe_weight'])['score'].mean().reset_index()
    pivot_table = pivot.pivot(index='llm_weight', columns='groupe_weight', values='score')
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title('Scores moyens par poids (LLM vs Groupe)')
    
    plt.tight_layout()
    plt.savefig('weight_optimization_results.png', dpi=300)
    plt.show()

def test_optimized_weights(best_params):
    """
    Fonction principale pour tester le modèle avec les poids optimisés.
    """
    # Paramètres
    MODEL_PATH = "./mDeBERTa-v3-base-multilingual-nli-local"
    
    # Configuration
    print("Initialisation du classificateur...")
    classifier = setup_zero_shot_classifier(MODEL_PATH)
    
    # Créer le jeu de données d'entraînement fictif
    print("Création du jeu de données d'entraînement...")
    df_train = creer_dataset_entrainement()
    
    # Créer les mappings entre variables catégorielles et causes
    print("Création des mappings catégoriels...")
    category_mappings = create_category_mappings(df_train)
    
    # Créer le jeu de données de test avec variables catégorielles
    print("Création du jeu de données de test...")
    df_test = creer_dataset_test_avec_categories()
    
    # Obtenir la taxonomie simplifiée
    causes_dict, subcauses_dict = define_taxonomy_simplified()
    
    # Évaluer le modèle hybride avec les poids optimisés
    print("Évaluation du modèle hybride avec les poids optimisés...")
    resultats = evaluer_modele_hybride(
        classifier, 
        df_test, 
        category_mappings,
        best_params['llm_weight'],
        best_params['groupe_weight'],
        best_params['service_weight']
    )
    
    # Afficher les résultats
    afficher_resultats(resultats)
    
    # Sauvegarder les prédictions détaillées
    resultats['df_predictions'].to_csv('predictions_hybride_optimized.csv', index=False)
    print("Prédictions détaillées sauvegardées dans 'predictions_hybride_optimized.csv'")
    
    return resultats

if __name__ == "__main__":
    # Paramètres
    MODEL_PATH = "./mDeBERTa-v3-base-multilingual-nli-local"
    
    print("Initialisation du classificateur...")
    classifier = setup_zero_shot_classifier(MODEL_PATH)
    
    # Créer le jeu de données d'entraînement fictif
    print("Création du jeu de données d'entraînement...")
    df_train = creer_dataset_entrainement()
    
    # Créer les mappings entre variables catégorielles et causes
    print("Création des mappings catégoriels...")
    category_mappings = create_category_mappings(df_train)
    
    # Créer le jeu de données de test avec variables catégorielles
    print("Création du jeu de données de test...")
    df_test = creer_dataset_test_avec_categories()
    
    # Obtenir la taxonomie simplifiée
    causes_dict, subcauses_dict = define_taxonomy_simplified()
    
    # Choisir la méthode d'optimisation (décommenter celle que vous souhaitez utiliser)
    
    # Option 1: Optimisation avec RandomSearch
    opt_results = optimize_weights_with_randomsearch(
        classifier, 
        df_train, 
        df_test, 
        category_mappings, 
        causes_dict, 
        subcauses_dict,
        n_iter=1  # Augmenter pour une recherche plus exhaustive
    )
    
    # Option 2: Optimisation bayésienne
    #opt_results = optimize_weights_bayesian(
    #     classifier, 
    #     df_train, 
    #     df_test, 
    #     category_mappings, 
    #     causes_dict, 
    #     subcauses_dict,
    #     n_iter=10  # Augmenter pour une recherche plus exhaustive
    # )
    
    # Tester avec les poids optimisés
    resultats_optimisés = test_optimized_weights(opt_results['best_params'])

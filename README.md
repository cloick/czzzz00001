import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Définition des valeurs possibles pour chaque colonne
humeurs = ["Très satisfaite", "Plutôt Satisfaite", "Plutôt Insatisfaite", "Très insatisfaite"]
categorisations = ["A dispatcher", "A traiter", "En cours", "Terminé"]
noms = ["Dupont Xavier", "Martin Sophie", "Leblanc Julie", "Durand Thomas", "Petit Claire", 
        "Moreau Nicolas", "Richard Emilie", "Robert Philippe", "Simon Catherine", "Leroy Antoine"]
perimetres = ["Epargne et assurances", "Data", "Crédits", "Banque au quotidien", "Moyens de paiement"]
sous_perimetres = ["Tribu Patrimoine", "Tribu digital workplace", "Tribu innovation", "Tribu crédit immobilier", "Tribu e-banking"]
chapitres = ["Tribu data hub", "Tribu dématérialisation", "Tribu expérience client", "Tribu conformité", "Tribu IT"]
metiers = ["Relai fabrication", "Scrum master", "Product owner", "Développeur", "Analyste fonctionnel", "Chef de projet", "Architecte", "Designer UX"]
types_enquete = ["IRC Flash Vague 1", "IRC Flash Vague 2", "IRC Flash Vague 3", "Enquête trimestrielle"]
equipes = ["IBC", "QOS", "DEV", "OPS", "ARCHI", "SECU"]
responsables = ["Martin Dupond", "Garcia Laura", "Vasseur Jean", "Lefebvre Marie", "Dubois Patrick"]
soutiens = ["Hughes Timhas", "Rousseau Marie", "Bernard David", "Fournier Sylvie", "Lambert François"]
statuts = ["Terminé", "En cours", "A planifier", "Suspendu"]

# Verbatims positifs et négatifs pour la génération
verbatims_positifs = [
    "Réactif et compétent, toujours à l'écoute.",
    "Excellente collaboration, équipe très efficace.",
    "Communication claire et adaptée à nos besoins.",
    "Grande disponibilité et solutions toujours pertinentes.",
    "Respect des délais et qualité du travail remarquable.",
    "Expertise technique impressionnante et bons conseils.",
    "Très bon accompagnement sur notre projet complexe.",
    "Capacité d'adaptation exceptionnelle face aux changements.",
    "Proactivité et anticipation des problématiques.",
    "Support technique efficace et rapide."
]

verbatims_negatifs = [
    "Très mauvaise communication récurrente de la part des Socles CAGIP sur les sujets transverses.",
    "Délais non respectés à plusieurs reprises, impact fort sur nos activités.",
    "Manque de compétence technique évident sur le projet.",
    "Difficultés à obtenir des réponses claires à nos questions.",
    "Documentation insuffisante et non mise à jour.",
    "Problèmes récurrents de qualité dans les livrables.",
    "Absence de suivi après la mise en production.",
    "Manque d'implication et de proactivité de l'équipe.",
    "Changements fréquents dans l'équipe perturbant la continuité.",
    "Outils mis à disposition obsolètes et inefficaces.",
    "Communication inexistante lors des incidents majeurs.",
    "Procédures trop complexes pour des demandes simples.",
    "Temps de réponse beaucoup trop long pour les tickets critiques.",
    "Absence totale de prise en compte de nos contraintes métier.",
    "Support technique défaillant et peu disponible."
]

# Génération de plans d'action
plans_action = [
    "Échange avec le manager de Tribu DWP pour résolution rapide.",
    "Organisation d'un atelier d'amélioration continue.",
    "Mise en place d'un nouveau processus de communication.",
    "Formation spécifique prévue pour l'équipe concernée.",
    "Revue des procédures internes et simplification.",
    "Création d'un groupe de travail dédié à cette problématique.",
    "Renforcement de l'équipe avec des profils plus expérimentés.",
    "Audit des outils et mise à jour prévue.",
    "Réorganisation du support pour améliorer les temps de réponse.",
    "Plan d'amélioration de la qualité en cours d'élaboration."
]

# Synthèses
syntheses = [
    "Florian signale ne pas avoir trouvé de solution malgré plusieurs relances.",
    "Problème identifié et résolu par mise à jour du framework.",
    "Incident majeur ayant nécessité une intervention de la DSI.",
    "Demande traitée avec succès après escalade au N+2.",
    "Problématique récurrente nécessitant une refonte complète.",
    "Solution temporaire mise en place en attendant le nouveau système.",
    "Formation effectuée, amélioration constatée immédiatement.",
    "Analyse approfondie révélant un problème d'infrastructure.",
    "Communication renforcée entre équipes suite à cet incident.",
    "Résolution nécessitant l'intervention de plusieurs services."
]

# Générer des dates sur les 6 derniers mois
date_fin = datetime.now()
date_debut = date_fin - timedelta(days=180)
dates_possibles = [date_debut + timedelta(days=x) for x in range(0, 180, 15)]  # Une enquête tous les 15 jours

# Fonction pour générer un verbatim en fonction de l'humeur
def generer_verbatim(humeur):
    if humeur in ["Très satisfaite", "Plutôt Satisfaite"]:
        return random.choice(verbatims_positifs)
    else:
        return random.choice(verbatims_negatifs)

# Fonction pour générer le plan d'action et la synthèse selon l'humeur et la catégorisation
def generer_plan_et_synthese(humeur, categorisation):
    if humeur in ["Plutôt Insatisfaite", "Très insatisfaite"]:
        plan = random.choice(plans_action) if categorisation != "A dispatcher" else ""
        synthese = random.choice(syntheses) if categorisation in ["En cours", "Terminé"] else ""
    else:
        plan = ""
        synthese = "Retour positif noté" if categorisation in ["En cours", "Terminé"] else ""
    
    return plan, synthese

# Créer un DataFrame avec 200 entrées
n_entries = 200
data = []

for i in range(n_entries):
    humeur = random.choices(humeurs, weights=[0.4, 0.3, 0.2, 0.1], k=1)[0]  # Distribution pondérée
    categorisation = random.choice(categorisations)
    verbatim = generer_verbatim(humeur)
    collaborateur = random.choice(noms)
    perimetre = random.choice(perimetres)
    sous_perimetre = random.choice(sous_perimetres)
    chapitre = random.choice(chapitres)
    metier = random.choice(metiers)
    date_enquete = random.choice(dates_possibles).strftime('%Y-%m-%d')
    type_enquete = random.choice(types_enquete)
    equipe = random.choice(equipes)
    responsable = random.choice(responsables)
    soutien = random.choice(soutiens)
    statut = random.choice(statuts)
    plan_action, synthese = generer_plan_et_synthese(humeur, categorisation)
    
    data.append({
        'Humeur': humeur,
        'Categorisation': categorisation,
        'Verbatim': verbatim,
        'Collaborateur qui a emis': collaborateur,
        'Perimetre': perimetre,
        'Sous perimetre': sous_perimetre,
        'Chapitre de la sous tribu': chapitre,
        'Metier CATS': metier,
        'Date de l\'enquete': date_enquete,
        'Type d\'Enquete': type_enquete,
        'Equipe BFR concernee': equipe,
        'Responsable BFR': responsable,
        'Soutien de BFR': soutien,
        'Statut': statut,
        'Plan d\'action': plan_action,
        'Synthese et/ou plan d\'actions': synthese
    })

# Création du DataFrame
df = pd.DataFrame(data)

# Sauvegarde dans un fichier Excel
output_file = 'enquetes_satisfaction.xlsx'
df.to_excel(output_file, index=True)  # Garder l'index pour avoir le numéro de ligne

print(f"Fichier '{output_file}' généré avec succès contenant {n_entries} entrées.")

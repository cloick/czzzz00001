import pandas as pd
import numpy as np

def fusionner_dictionnaires_souscauses(dict1, dict2):
    """
    Fusionne deux dictionnaires de sous-causes en conservant toutes les valeurs
    uniques et en éliminant les valeurs NaN.
    
    Args:
        dict1: Premier dictionnaire avec causes comme clés et listes de sous-causes comme valeurs
        dict2: Second dictionnaire avec la même structure
        
    Returns:
        dict: Dictionnaire fusionné
    """
    dictionnaire_fusionne = {}
    
    # Parcourir toutes les clés du premier dictionnaire
    for cause in dict1:
        # Vérifier si la cause existe dans les deux dictionnaires
        if cause in dict2:
            # Obtenir toutes les sous-causes des deux dictionnaires
            sous_causes_combine = dict1[cause] + dict2[cause]
            # Convertir en liste pour faciliter le filtrage
            if not isinstance(sous_causes_combine, list):
                sous_causes_combine = [sous_causes_combine]
            
            # Filtrer les valeurs NaN et conserver les valeurs uniques
            sous_causes_filtre = [sc for sc in sous_causes_combine 
                                  if not (pd.isna(sc) or sc == 'nan' or sc == 'NaN')]
            sous_causes_unique = list(dict.fromkeys(sous_causes_filtre))  # Préserve l'ordre
            
            # Ajouter au dictionnaire fusionné
            dictionnaire_fusionne[cause] = sous_causes_unique
        else:
            # Si la cause n'existe que dans le premier dictionnaire
            sous_causes = [sc for sc in dict1[cause] 
                           if not (pd.isna(sc) or sc == 'nan' or sc == 'NaN')]
            dictionnaire_fusionne[cause] = sous_causes
    
    # Ajouter les causes qui n'existent que dans le second dictionnaire
    for cause in dict2:
        if cause not in dict1:
            sous_causes = [sc for sc in dict2[cause] 
                          if not (pd.isna(sc) or sc == 'nan' or sc == 'NaN')]
            dictionnaire_fusionne[cause] = sous_causes
    
    return dictionnaire_fusionne

# Exemple d'utilisation
dictionnaire_fusionne = fusionner_dictionnaires_souscauses(
    dictionnaire_causes_souscauses, 
    dictionnaire_causes_souscauses_metis
)
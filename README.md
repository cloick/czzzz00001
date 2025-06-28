Ok, analysons ce besoin step by step :

1. L'utilisateur a une table "ILSI" avec des colonnes importantes :
   - "Id Solution"
   - "Planification test : Tx/2025 ou Tx/2026"
   - "Suivi 2025 Test complet (calculé)"
   - "Suivi 2025 Procédure formalisée" 
   - "Appops"

2. Il veut une mesure qui compte les solutions selon cette logique :
   - Pour chaque solution (Id Solution), il regarde la colonne "Planification test : Tx/2025 ou Tx/2026"
   - Si TOUTES les lignes pour cette solution sont "NA" ou "NR" ou "ILSI CAGIP" ou "ILSI externe", alors on ne compte PAS la solution
   - Si AU MOINS UNE ligne pour cette solution a une autre valeur (comme "T1", "T2", etc.), alors on compte la solution

3. Il veut ensuite des mesures par Appops :
   - Maximum par Appops
   - Minimum par Appops  
   - Moyenne par Appops
   - Une mesure pour l'Appops sélectionné

4. Important : les mesures de max, min, moyenne ne doivent PAS réagir au filtre du segment Appops

Je vais proposer une solution DAX pour cela.

Parfaitement ! Je comprends votre logique. Vous voulez compter une solution seulement si **au moins une** de ses lignes dans "Planification test" a une valeur autre que "NA", "NR", "ILSI CAGIP", "ILSI externe".

Voici les mesures DAX :

## 1. Mesure de base - Comptage des solutions par Appops

```dax
Solutions Comptées = 
VAR SolutionsValides = 
    SUMMARIZE(
        ILSI,
        ILSI[Id Solution],
        ILSI[Appops],
        "ATesteFaire", 
        IF(
            COUNTROWS(
                FILTER(
                    RELATEDTABLE(ILSI),
                    NOT(ILSI[Planification test : Tx/2025 ou Tx/2026] IN {"NA", "NR", "ILSI CAGIP", "ILSI externe"})
                )
            ) > 0,
            1,
            0
        )
    )
RETURN
    SUMX(
        FILTER(SolutionsValides, [ATesteFaire] = 1),
        1
    )
```

## 2. Mesures qui ne réagissent PAS au filtre Appops

```dax
Maximum Global = 
CALCULATE(
    [Solutions Comptées],
    ALL(ILSI[Appops])
)

Minimum Global = 
VAR TodosAppops = VALUES(ALL(ILSI[Appops]))
VAR ResultatParAppops = 
    ADDCOLUMNS(
        TodosAppops,
        "Compte", 
        CALCULATE([Solutions Comptées], ALL(ILSI[Appops]))
    )
RETURN
    MINX(ResultatParAppops, [Compte])

Moyenne Globale = 
VAR TodosAppops = VALUES(ALL(ILSI[Appops]))
VAR ResultatParAppops = 
    ADDCOLUMNS(
        TodosAppops,
        "Compte", 
        CALCULATE([Solutions Comptées], ALL(ILSI[Appops]))
    )
RETURN
    AVERAGEX(ResultatParAppops, [Compte])
```

## 3. Mesure pour l'Appops sélectionné

```dax
Solutions Appops Sélectionné = [Solutions Comptées]
```

Cette dernière mesure réagira naturellement au filtre du segment.

**Comment ça fonctionne :**
- La mesure de base groupe par Id Solution et Appops
- Pour chaque solution, elle vérifie s'il existe au moins une ligne avec une valeur de test différente de vos exclusions
- Les mesures "Globales" utilisent `ALL(ILSI[Appops])` pour ignorer le filtre du segment
- La mesure "Sélectionné" suit normalement le contexte de filtre

Est-ce que cette approche correspond à votre besoin ?

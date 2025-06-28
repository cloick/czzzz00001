Parfaitement ! Je comprends votre logique. Vous voulez compter une solution seulement si au moins une de ses lignes dans "Planification test" a une valeur autre que "NA", "NR", "ILSI CAGIP", "ILSI externe".Voici les mesures DAX :1. Mesure de base - Comptage des solutions par AppopsSolutions Comptées = 
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
    )2. Mesures qui ne réagissent PAS au filtre AppopsMaximum Global = 
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
    AVERAGEX(ResultatParAppops, [Compte])3. Mesure pour l'Appops sélectionnéSolutions Appops Sélectionné = [Solutions Comptées]Cette dernière mesure réagira naturellement au filtre du segment.Comment ça fonctionne :La mesure de base groupe par Id Solution et AppopsPour chaque solution, elle vérifie s'il existe au moins une ligne avec une valeur de test différente de vos exclusionsLes mesures "Globales" utilisent ALL(ILSI[Appops]) pour ignorer le filtre du segmentLa mesure "Sélectionné" suit normalement le contexte de filtreEst-ce que cette approche correspond à votre besoin ?

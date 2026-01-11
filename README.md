Compte AppOps Tribu = 
VAR TribuSelectionnee = ISFILTERED(Appops_secu[Tribu])

VAR TableFiltree = 
    IF(
        TribuSelectionnee,
        // Si Tribu sélectionnée : filtrer sur AppOps ET Tribu
        CALCULATETABLE(
            Feuil1,
            TREATAS(VALUES(Appops_secu[Appops]), Feuil1[AppOps]),
            TREATAS(VALUES(Appops_secu[Tribu]), Feuil1[Tribu])
        ),
        // Si seulement AppOps : filtrer seulement sur AppOps
        CALCULATETABLE(
            Feuil1,
            TREATAS(VALUES(Appops_secu[Appops]), Feuil1[AppOps])
        )
    )

VAR CompteNon = 
    COUNTROWS(FILTER(TableFiltree, Feuil1[Discolation] = "Non"))

VAR APremierOui = 
    IF(
        COUNTROWS(FILTER(TableFiltree, Feuil1[Discolation] = "Oui")) > 0,
        1,
        0
    )

RETURN CompteNon + APremierOui

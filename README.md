Ah oui, encore le problème du `IF` avec des tables ! On ne peut pas stocker une table dans une variable avec `IF`.

Il faut appliquer la logique directement dans les calculs :

---

## Mesure DAX corrigée :

```dax
Compte AppOps Tribu = 
VAR TribuSelectionnee = ISFILTERED(Appops_secu[Tribu])

VAR CompteNon = 
    IF(
        TribuSelectionnee,
        CALCULATE(
            COUNTROWS(FILTER(Feuil1, Feuil1[Discolation] = "Non")),
            TREATAS(VALUES(Appops_secu[Appops]), Feuil1[AppOps]),
            TREATAS(VALUES(Appops_secu[Tribu]), Feuil1[Tribu])
        ),
        CALCULATE(
            COUNTROWS(FILTER(Feuil1, Feuil1[Discolation] = "Non")),
            TREATAS(VALUES(Appops_secu[Appops]), Feuil1[AppOps])
        )
    )

VAR APremierOui = 
    IF(
        TribuSelectionnee,
        IF(
            CALCULATE(
                COUNTROWS(FILTER(Feuil1, Feuil1[Discolation] = "Oui")),
                TREATAS(VALUES(Appops_secu[Appops]), Feuil1[AppOps]),
                TREATAS(VALUES(Appops_secu[Tribu]), Feuil1[Tribu])
            ) > 0,
            1,
            0
        ),
        IF(
            CALCULATE(
                COUNTROWS(FILTER(Feuil1, Feuil1[Discolation] = "Oui")),
                TREATAS(VALUES(Appops_secu[Appops]), Feuil1[AppOps])
            ) > 0,
            1,
            0
        )
    )

RETURN CompteNon + APremierOui
```

**Teste maintenant !** 🎯

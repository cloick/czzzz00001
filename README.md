
Je vais maintenant fournir les trois formules corrigées pour MIN, MAX et AVG qui respectent les filtres de la table Environnement tout en ignorant ceux de la table Appops_.

Pour chaque formule, je vais:
1. Capturer le filtre actuel sur Environnement
2. Utiliser ALLEXCEPT pour préserver les filtres sur la colonne Environnement
3. Réappliquer explicitement le filtre d'Environnement

Commençons avec le MIN, puis le MAX, puis la moyenne.

Voici les trois formules corrigées qui ignorent uniquement les filtres Appops_ tout en respectant les filtres Environnement :

```
MinValueWindows = 
VAR CurrentEnvFilter = TREATAS(VALUES('Environnement'[Environnement]), 'windows_new'[Environnement])
VAR AppopsCount = 
    ADDCOLUMNS(
        ALL('Appops_'[Appops]),
        "Comptage", 
        COALESCE(
            CALCULATE(
                COUNTROWS('windows_new'),
                ALLEXCEPT('windows_new', 'windows_new'[Environnement]),
                CurrentEnvFilter,
                FILTER(
                    ALL('windows_new'),
                    'windows_new'[Appops] = EARLIER('Appops_'[Appops])
                )
            ),
            0
        )
    )
RETURN
    MINX(AppopsCount, [Comptage])
```

```
MaxValueWindows = 
VAR CurrentEnvFilter = TREATAS(VALUES('Environnement'[Environnement]), 'windows_new'[Environnement])
VAR AppopsCount = 
    ADDCOLUMNS(
        ALL('Appops_'[Appops]),
        "Comptage", 
        CALCULATE(
            COUNTROWS('windows_new'),
            ALLEXCEPT('windows_new', 'windows_new'[Environnement]),
            CurrentEnvFilter,
            FILTER(
                ALL('windows_new'),
                'windows_new'[Appops] = EARLIER('Appops_'[Appops])
                && FORMAT('windows_new'[Appops], "") <> ""
            )
        )
    )
VAR AppsWithNonZeroCounts = 
    FILTER(AppopsCount, [Comptage] > 0)
RETURN
    MAXX(AppsWithNonZeroCounts, [Comptage])
```

```
AvgValueWindows = 
VAR CurrentEnvFilter = TREATAS(VALUES('Environnement'[Environnement]), 'windows_new'[Environnement])
VAR AppopsCount = 
    ADDCOLUMNS(
        ALL('Appops_'[Appops]),
        "Comptage", 
        CALCULATE(
            COUNTROWS('windows_new'),
            ALLEXCEPT('windows_new', 'windows_new'[Environnement]),
            CurrentEnvFilter,
            FILTER(
                ALL('windows_new'),
                'windows_new'[Appops] = EARLIER('Appops_'[Appops])
                && FORMAT('windows_new'[Appops], "") <> ""
            )
        )
    )
VAR AppsWithNonZeroCounts = 
    FILTER(AppopsCount, [Comptage] > 0)
RETURN
    AVERAGEX(AppsWithNonZeroCounts, [Comptage])
```

Ces formules utilisent maintenant ALLEXCEPT au lieu de REMOVEFILTERS, ce qui permet de préserver les filtres sur la colonne Environnement tout en supprimant les autres filtres. Le TREATAS capture et réapplique explicitement ces filtres pour garantir qu'ils sont correctement pris en compte.

Ah, je comprends mieux maintenant. Si vous voulez compter chaque NOM_CS une seule fois même s'il apparaît plusieurs fois dans la table, il faut utiliser DISTINCTCOUNTNOBLANK à la place de COUNT. Voici les formules corrigées:

**Pour le MINIMUM** (compte unique de NOM_CS par Appops, où statut = "Terminé"):
```
MinStatutTermine = 
VAR AppopsCount = 
    SUMMARIZE(
        REMOVEFILTERS(obso_planifall),
        obso_planifall[Appops],
        "Count", CALCULATE(
            DISTINCTCOUNTNOBLANK(obso_planifall[NOM_CS]),
            obso_planifall[statut] = "Terminé",
            REMOVEFILTERS(obso_planifall[Appops])
        )
    )
RETURN
    MINX(AppopsCount, [Count])
```

**Pour le MAXIMUM** (compte unique de NOM_CS par Appops, où statut = "Terminé"):
```
MaxStatutTermine = 
VAR AppopsCount = 
    SUMMARIZE(
        REMOVEFILTERS(obso_planifall),
        obso_planifall[Appops],
        "Count", CALCULATE(
            DISTINCTCOUNTNOBLANK(obso_planifall[NOM_CS]),
            obso_planifall[statut] = "Terminé",
            REMOVEFILTERS(obso_planifall[Appops])
        )
    )
RETURN
    MAXX(AppopsCount, [Count])
```

**Pour la MOYENNE** (compte unique de NOM_CS par Appops, où statut = "Terminé"):
```
AvgStatutTermine = 
VAR AppopsCount = 
    SUMMARIZE(
        REMOVEFILTERS(obso_planifall),
        obso_planifall[Appops],
        "Count", CALCULATE(
            DISTINCTCOUNTNOBLANK(obso_planifall[NOM_CS]),
            obso_planifall[statut] = "Terminé",
            REMOVEFILTERS(obso_planifall[Appops])
        )
    )
RETURN
    AVERAGEX(AppopsCount, [Count])
```

Avec DISTINCTCOUNTNOBLANK, chaque valeur unique de NOM_CS n'est comptée qu'une seule fois, même si elle apparaît plusieurs fois dans la table.

Vous avez raison, c'est une erreur de syntaxe. REMOVEFILTERS doit être utilisé dans le contexte CALCULATE et non pas sur la table directement dans SUMMARIZE. Voici les formules corrigées:

**Pour le MINIMUM**:
```
MinStatutTermine = 
VAR AppopsCount = 
    SUMMARIZE(
        obso_planifall,
        obso_planifall[Appops],
        "Count", CALCULATE(
            DISTINCTCOUNTNOBLANK(obso_planifall[NOM_CS]),
            obso_planifall[statut] = "Terminé",
            REMOVEFILTERS(obso_planifall[Appops]),
            REMOVEFILTERS()  // Ou spécifiez les colonnes que vous voulez neutraliser
        )
    )
RETURN
    MINX(AppopsCount, [Count])
```

**Pour le MAXIMUM**:
```
MaxStatutTermine = 
VAR AppopsCount = 
    SUMMARIZE(
        obso_planifall,
        obso_planifall[Appops],
        "Count", CALCULATE(
            DISTINCTCOUNTNOBLANK(obso_planifall[NOM_CS]),
            obso_planifall[statut] = "Terminé",
            REMOVEFILTERS(obso_planifall[Appops]),
            REMOVEFILTERS()  
        )
    )
RETURN
    MAXX(AppopsCount, [Count])
```

**Pour la MOYENNE**:
```
AvgStatutTermine = 
VAR AppopsCount = 
    SUMMARIZE(
        obso_planifall,
        obso_planifall[Appops],
        "Count", CALCULATE(
            DISTINCTCOUNTNOBLANK(obso_planifall[NOM_CS]),
            obso_planifall[statut] = "Terminé",
            REMOVEFILTERS(obso_planifall[Appops]),
            REMOVEFILTERS()  
        )
    )
RETURN
    AVERAGEX(AppopsCount, [Count])
```

Le REMOVEFILTERS est maintenant correctement placé à l'intérieur de CALCULATE, ce qui devrait corriger l'erreur de syntaxe.

Le problème persiste avec le REMOVEFILTERS qui ne semble pas fonctionner correctement. Il y a quelques raisons possibles :

1. La position du REMOVEFILTERS dans la formule n'est peut-être pas optimale
2. Il pourrait y avoir un problème dans la structure de données ou les relations entre tables
3. Le contexte d'évaluation de la formule est peut-être différent de ce que nous pensons

Essayons une approche plus radicale en utilisant ALL au lieu d'ALLSELECTED, et en restructurant la formule pour être sûr que REMOVEFILTERS s'applique correctement :

```
MinValueObsoTermine = 
VAR AllAppops = ALL('obso_planifall'[Appops])
VAR ComptageParAppops =
    ADDCOLUMNS(
        AllAppops,
        "NombreNomCSTermine", 
        CALCULATE(
            DISTINCTCOUNT('obso_planifall'[NOM_CS]),
            'obso_planifall'[statut] = "5 - Terminé",
            ALL('obso_planifall'[Appops])  // Utilisation de ALL au lieu de REMOVEFILTERS
        )
    )
RETURN
    MINX(
        FILTER(ComptageParAppops, [NombreNomCSTermine] > 0),
        [NombreNomCSTermine]
    )
```

Cette approche utilise ALL('obso_planifall'[Appops]) au lieu de REMOVEFILTERS, ce qui peut être plus direct et efficace dans certains contextes.

Une autre possibilité est d'utiliser ALLEXCEPT pour préserver tous les filtres sauf ceux sur Appops :

```
MinValueObsoTermine = 
VAR ComptageParAppops =
    SUMMARIZE(
        ALL('obso_planifall'[Appops]),
        'obso_planifall'[Appops],
        "NombreNomCSTermine", 
        CALCULATE(
            DISTINCTCOUNT('obso_planifall'[NOM_CS]),
            'obso_planifall'[statut] = "5 - Terminé",
            ALLEXCEPT('obso_planifall', 'obso_planifall'[statut])  // Préserve uniquement le filtre statut
        )
    )
RETURN
    MINX(
        FILTER(ComptageParAppops, [NombreNomCSTermine] > 0),
        [NombreNomCSTermine]
    )
```

Je vois qu'il peut y avoir une confusion dans les filtres. Essayons une approche qui garantit clairement que nous comptons par Appops mais ignorons les filtres sur Appops. Pour cela, je vais utiliser une approche en deux temps :

1. Créer une table temporaire qui liste toutes les Appops et leur nombre de NOM_CS, sans aucun filtre
2. Puis utiliser cette table pour les calculs MIN, MAX, et AVG

Je note toutefois que l'utilisateur semble avoir trouvé une formule qui fonctionne partiellement, et qu'il suffit juste de faire en sorte que le REMOVEFILTERS s'applique correctement.

Essayons une approche différente pour garantir que le REMOVEFILTERS fonctionne correctement. Voici les formules corrigées :

```
MinValueObsoTermine = 
VAR AllApps = ALL('obso_planifall'[Appops])
VAR ComptageParAppops =
    ADDCOLUMNS(
        AllApps,
        "NombreNomCSTermine", 
        CALCULATE(
            DISTINCTCOUNT('obso_planifall'[NOM_CS]),
            'obso_planifall'[statut] = "5 - Terminé",
            ALL('obso_planifall'[Appops])  // Utilisation de ALL au lieu de REMOVEFILTERS
        )
    )
RETURN
    MINX(
        FILTER(ComptageParAppops, [NombreNomCSTermine] > 0),
        [NombreNomCSTermine]
    )
```

```
MaxValueObsoTermine = 
VAR AllApps = ALL('obso_planifall'[Appops])
VAR ComptageParAppops =
    ADDCOLUMNS(
        AllApps,
        "NombreNomCSTermine", 
        CALCULATE(
            DISTINCTCOUNT('obso_planifall'[NOM_CS]),
            'obso_planifall'[statut] = "5 - Terminé",
            ALL('obso_planifall'[Appops])
        )
    )
RETURN
    MAXX(ComptageParAppops, [NombreNomCSTermine])
```

```
AvgValueObsoTermine = 
VAR AllApps = ALL('obso_planifall'[Appops])
VAR ComptageParAppops =
    ADDCOLUMNS(
        AllApps,
        "NombreNomCSTermine", 
        CALCULATE(
            DISTINCTCOUNT('obso_planifall'[NOM_CS]),
            'obso_planifall'[statut] = "5 - Terminé",
            ALL('obso_planifall'[Appops])
        )
    )
RETURN
    AVERAGEX(
        FILTER(ComptageParAppops, [NombreNomCSTermine] > 0),
        [NombreNomCSTermine]
    )
```

Les changements clés :
1. Utilisation d'ADDCOLUMNS au lieu de SUMMARIZE
2. Utilisation d'ALL au lieu de REMOVEFILTERS (cette approche est plus directe)
3. Stockage de ALL('obso_planifall'[Appops]) dans une variable pour clarté
4. Filtrage des valeurs nulles pour MIN et AVG

Cette approche devrait mieux garantir que les mesures ignorent complètement les filtres sur Appops.

D'après les trois images que vous m'avez envoyées, je peux voir différentes mesures DAX qui ont fonctionné pour différents scénarios:

1. **Image 1 (MinValueLinux)** - Cette mesure compte les occurrences dans la table 'linux' par Appops, en utilisant VALUES, COALESCE, et ALL pour gérer les filtres.

2. **Image 2 (MinValueAudit)** - Cette mesure utilise une approche avec DerniereDate, CurrentLibelleFilter, TREATAS pour préserver le filtre Libellé tout en supprimant les filtres Appops.

3. **Image 3 (MinValueWindows)** - Cette mesure utilise CurrentEnvFilter, ALLEXCEPT, et FILTER avec EARLIER pour gérer les relations entre tables.

Pour votre problème avec ObsoTermine, nous devons adapter ces approches qui fonctionnent. Le problème principal semble être que REMOVEFILTERS ne fonctionne pas correctement dans votre contexte.

Basé sur ces exemples, je vais vous proposer une solution similaire à celle de l'Image 2 ou l'Image 3, qui semblent fonctionner dans des contextes similaires.

Je comprends - vous avez trois exemples de code DAX qui fonctionnent bien pour ignorer certains filtres.

En étudiant vos exemples, je vois que chacun utilise une technique différente mais efficace pour ignorer les filtres :

1. **Image 1 (MinValueLinux)** - Utilise EARLIER avec ALL pour comparer les Appops
2. **Image 2 (MinValueAudit)** - Utilise TREATAS pour préserver certains filtres tout en en ignorant d'autres
3. **Image 3 (MinValueWindows)** - Utilise ALLEXCEPT avec CurrentEnvFilter pour contrôler précisément les filtres

Pour votre problème avec obso_planifall, adaptons la technique de l'Image 1 qui est plus directe :

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
            FILTER(
                ALL('obso_planifall'),
                'obso_planifall'[Appops] = EARLIER('obso_planifall'[Appops])
            )
        )
    )
RETURN
    MINX(
        FILTER(ComptageParAppops, [NombreNomCSTermine] > 0),
        [NombreNomCSTermine]
    )
```

Cette formule utilise la technique EARLIER avec FILTER(ALL()) qui fonctionne bien dans votre premier exemple. Elle devrait ignorer les filtres Appops tout en comptant correctement le nombre distinct de NOM_CS par Appops.

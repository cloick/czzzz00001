
Le problème persiste malgré les différentes approches que nous avons essayées. Reprenons cette situation depuis le début et analysons-la méthodiquement.

Ce que nous voulons:
1. Prendre toutes les valeurs d'Appops dans 'Appops_' (en ignorant les filtres)
2. Compter combien de fois chaque Appops apparaît dans 'windows_new'
3. Ne garder que les Appops qui apparaissent réellement (comptage > 0)
4. Trouver le maximum de ces comptages

La difficulté principale semble être que des valeurs vides ou NULL dans 'windows_new' sont comptées et associées à des valeurs de 'Appops_', ce qui donne un résultat erroné (127 occurrences de valeurs vides).

Essayons une approche différente en utilisant FORMAT comme suggéré:

```
MaxValueWindows = 
VAR AppopsCount = 
    ADDCOLUMNS(
        ALL('Appops_'[Appops]),
        "Comptage", 
        CALCULATE(
            COUNTROWS('windows_new'),
            REMOVEFILTERS('windows_new'),
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

Cette approche n'a pas fonctionné correctement non plus.

Essayons une autre stratégie. Au lieu d'utiliser FILTER pour la correspondance, utilisons une approche plus directe:

```
MaxValueWindows = 
VAR ValidAppops = 
    FILTER(
        ALL('Appops_'[Appops]),
        FORMAT([Appops], "") <> ""
    )
VAR AppopsCount = 
    ADDCOLUMNS(
        ValidAppops,
        "Comptage", 
        CALCULATE(
            COUNTROWS('windows_new'),
            REMOVEFILTERS('windows_new'),
            'windows_new'[Appops] IN ValidAppops
        )
    )
VAR AppsWithNonZeroCounts = 
    FILTER(AppopsCount, [Comptage] > 0)
RETURN
    MAXX(AppsWithNonZeroCounts, [Comptage])
```

Une autre approche serait de traiter explicitement les deux tables pour s'assurer qu'aucune valeur vide n'interfère:

```
MaxValueWindows = 
VAR NonEmptyAppsFromReference = 
    FILTER(
        VALUES('Appops_'[Appops]),
        FORMAT([Appops], "") <> ""
    )
VAR NonEmptyAppsFromWindows = 
    FILTER(
        VALUES('windows_new'[Appops]),
        FORMAT([Appops], "") <> ""
    )
VAR CommonAppops = 
    INTERSECT(NonEmptyAppsFromReference, NonEmptyAppsFromWindows)
VAR AppopsCount = 
    ADDCOLUMNS(
        CommonAppops,
        "Comptage", 
        CALCULATE(
            COUNTROWS('windows_new'),
            REMOVEFILTERS('windows_new'),
            'windows_new'[Appops] IN CommonAppops
        )
    )
RETURN
    MAXX(AppopsCount, [Comptage])
```

Cette dernière approche utilise INTERSECT pour ne garder que les valeurs qui existent dans les deux tables, puis compte les occurrences uniquement pour ces valeurs communes. Cela devrait éliminer complètement tout problème avec les valeurs vides.

Maintenant, concernant le résultat de 86 avec TREATAS, cela suggère que TREATAS traite les données différemment. Il est possible qu'il regroupe des valeurs que nous considérons comme distinctes ou qu'il capture des correspondances que nos autres formules manquent.

Sans connaître les données exactes, il est difficile de dire pourquoi TREATAS donne 86. Mais l'approche avec INTERSECT pourrait être la plus explicite et la plus contrôlée pour ce cas particulier.

Je comprends les difficultés que vous rencontrez. Utilisons explicitement FORMAT comme vous l'avez suggéré et adoptons une approche différente en utilisant INTERSECT pour identifier précisément les Appops qui existent dans les deux tables:

```
MaxValueWindows = 
VAR NonEmptyAppsFromReference = 
    FILTER(
        ALL('Appops_'[Appops]),
        FORMAT([Appops], "") <> ""
    )
VAR NonEmptyAppsFromWindows = 
    FILTER(
        ALL('windows_new'[Appops]),
        FORMAT([Appops], "") <> ""
    )
VAR CommonAppops = 
    INTERSECT(NonEmptyAppsFromReference, NonEmptyAppsFromWindows)
VAR AppopsCount = 
    ADDCOLUMNS(
        CommonAppops,
        "Comptage", 
        CALCULATE(
            COUNTROWS('windows_new'),
            REMOVEFILTERS('windows_new'),
            'windows_new'[Appops] IN CommonAppops
        )
    )
RETURN
    MAXX(AppopsCount, [Comptage])
```

Cette formule:
1. Identifie toutes les valeurs non-vides dans 'Appops_'
2. Identifie toutes les valeurs non-vides dans 'windows_new'
3. Utilise INTERSECT pour ne garder que les valeurs qui existent dans les deux tables
4. Compte les occurrences uniquement pour ces valeurs communes

Cette approche est beaucoup plus explicite et devrait éliminer complètement les problèmes avec les valeurs vides. Elle devrait également ignorer les filtres contextuels grâce à l'utilisation de ALL et REMOVEFILTERS.


Oui, je comprends bien la nuance pour le MAX. Pour le MAX, nous voulons:

1. Considérer uniquement les Appops qui existent à la fois dans la table 'windows_new' ET dans la table 'Appops_'
2. Ignorer les Appops qui existent dans 'Appops_' mais n'apparaissent pas dans 'windows_new'
3. Assurer que le calcul soit insensible aux filtres contextuels

Pour accomplir cela, nous allons:
1. Utiliser ALL pour ignorer les filtres sur 'Appops_'
2. Compter les occurrences de chaque Appops dans 'windows_new' 
3. Filtrer pour ne garder que les Appops avec un comptage > 0
4. Retourner le MAX

La formule ressemblerait à ceci:
```
MaxValueWindows = 
VAR AppopsCount = 
    ADDCOLUMNS(
        ALL('Appops_'[Appops]),  // ALL ignore tous les filtres
        "Comptage", 
        COALESCE(
            CALCULATE(
                COUNTROWS('windows_new'),
                REMOVEFILTERS('windows_new'),
                FILTER(
                    ALL('windows_new'),
                    'windows_new'[Appops] = EARLIER('Appops_'[Appops])
                )
            ),
            0
        )
    )
VAR AppsWithNonZeroCounts = 
    FILTER(AppopsCount, [Comptage] > 0)
RETURN
    MAXX(AppsWithNonZeroCounts, [Comptage])
```

Cette formule est similaire à celle qui a fonctionné pour le MIN, mais avec l'ajout d'une étape de filtrage qui élimine les comptages à zéro avant de calculer le maximum.

Parfaitement, je comprends votre besoin exact pour le MAX :

Vous voulez trouver le nombre maximum d'occurrences, mais uniquement parmi les Appops qui existent à la fois :
1. Dans votre table de référence 'Appops_'
2. ET dans votre table de données 'windows_new'

Voici la formule qui accomplira cela :

```
MaxValueWindows = 
VAR AppopsCount = 
    ADDCOLUMNS(
        ALL('Appops_'[Appops]),  // ALL ignore tous les filtres
        "Comptage", 
        COALESCE(
            CALCULATE(
                COUNTROWS('windows_new'),
                REMOVEFILTERS('windows_new'),
                FILTER(
                    ALL('windows_new'),
                    'windows_new'[Appops] = EARLIER('Appops_'[Appops])
                )
            ),
            0
        )
    )
VAR AppsWithNonZeroCounts = 
    FILTER(AppopsCount, [Comptage] > 0)
RETURN
    MAXX(AppsWithNonZeroCounts, [Comptage])
```

La clé ici est l'étape intermédiaire avec `FILTER(AppopsCount, [Comptage] > 0)` qui élimine les Appops absentes de 'windows_new' avant de calculer le maximum.

Cette formule vous donnera le nombre maximal d'occurrences tout en restant insensible aux filtres, exactement comme vous le souhaitez.

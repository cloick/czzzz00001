
Très bien, maintenant je comprends mieux. Pour la moyenne, comme pour le maximum, nous voulons uniquement considérer les Appops qui apparaissent à la fois dans:
1. La table de référence 'Appops_'
2. ET la table 'windows_new'

Donc nous devons filtrer pour éliminer les comptages nuls, comme dans la formule du MAX.

Voici la formule correcte pour la moyenne avec cette logique:

Compris, pour la moyenne vous voulez suivre exactement la même logique que pour le maximum - c'est-à-dire ne considérer que les Appops qui apparaissent à la fois dans la table de référence et dans windows_new (donc avec un comptage > 0).

Voici la formule adaptée:

```
AvgValueWindows = 
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
    AVERAGEX(AppsWithNonZeroCounts, [Comptage])
```

La différence clé est l'ajout de la variable `AppsWithNonZeroCounts` qui filtre pour ne conserver que les comptages supérieurs à zéro, exactement comme dans votre mesure MAX qui fonctionne.

Cette formule calculera la moyenne uniquement des Appops qui existent à la fois dans Appops_ et dans windows_new, en ignorant les valeurs vides et celles qui n'apparaissent pas.

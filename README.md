Pour créer cette mesure MinValue qui va compter le nombre d'occurrences de chaque valeur dans la colonne Appops et renvoyer la valeur minimale, voici la formule DAX à utiliser:

```
MinValue = 
VAR AppopsCount = 
    SUMMARIZE(
        'VotreTable',
        'VotreTable'[Appops],
        "Comptage", COUNTX(FILTER('VotreTable', 'VotreTable'[Appops] = EARLIER('VotreTable'[Appops])), 1)
    )
RETURN
    MINX(AppopsCount, [Comptage])
```

Cette formule fonctionne en deux étapes:

1. La variable `AppopsCount` crée une table qui compte combien de fois chaque valeur Appops apparaît dans votre table
2. Ensuite, la fonction MINX trouve la valeur minimale parmi tous ces comptages

Par exemple, si vous avez:
- "data" qui apparaît 1 fois
- "crédits" qui apparaît 3 fois
- "réseau" qui apparaît 7 fois

La mesure MinValue renverra 1, car c'est le nombre minimal d'occurrences.

Si vous voulez également connaître quelle Appops a cette valeur minimale, vous auriez besoin d'une mesure supplémentaire, mais celle-ci vous donnera la valeur minimale comme demandé.



Oui, vous pouvez ajouter un REMOVEFILTERS dans cette mesure pour vous assurer que les filtres contextuels n'affectent pas le calcul. Voici comment vous pourriez modifier la formule:

```
MinValue = 
VAR AppopsCount = 
    SUMMARIZE(
        REMOVEFILTERS('VotreTable'),
        'VotreTable'[Appops],
        "Comptage", COUNTX(FILTER(REMOVEFILTERS('VotreTable'), 'VotreTable'[Appops] = EARLIER('VotreTable'[Appops])), 1)
    )
RETURN
    MINX(AppopsCount, [Comptage])
```

Dans cette version, j'ai ajouté REMOVEFILTERS à deux endroits:
1. Dans la fonction SUMMARIZE, pour ignorer les filtres lors de la création de la table de synthèse
2. Dans la fonction FILTER, pour s'assurer que le comptage de chaque valeur Appops est calculé sur l'ensemble complet de données

Ainsi, peu importe quels filtres sont appliqués dans votre rapport ou visuel, cette mesure calculera toujours le nombre minimal d'occurrences d'une valeur Appops dans l'ensemble de la table, sans tenir compte des filtres contextuels.




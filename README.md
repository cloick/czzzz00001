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





Je comprends mieux maintenant. Si la liste des Appops à compter vient d'une autre table, vous devez ajuster la formule pour prendre en compte cette relation. Voici comment vous pourriez modifier la mesure:

```
MinValue = 
VAR AppopsCount = 
    SUMMARIZE(
        'TableDesAppops',  // La table qui contient la liste des Appops
        'TableDesAppops'[Appops],
        "Comptage", CALCULATE(
            COUNTROWS('VotreTable'),  // La table où vous comptez les occurrences
            REMOVEFILTERS('VotreTable')
        )
    )
RETURN
    MINX(AppopsCount, [Comptage])
```

Dans cette formule:
- 'TableDesAppops' est la table qui contient la liste des Appops que vous voulez analyser
- 'VotreTable' est la table où vous voulez compter les occurrences de ces Appops
- REMOVEFILTERS supprime tous les filtres de contexte sur 'VotreTable'
- CALCULATE compte les lignes dans 'VotreTable' pour chaque valeur Appops
- MINX trouve la valeur minimale parmi tous ces comptages

Si les deux tables sont reliées par une relation dans votre modèle de données, le filtre se propagera automatiquement. Si ce n'est pas le cas, vous devrez peut-être ajouter un filtre explicite:

```
MinValue = 
VAR AppopsCount = 
    SUMMARIZE(
        'TableDesAppops',
        'TableDesAppops'[Appops],
        "Comptage", CALCULATE(
            COUNTROWS('VotreTable'),
            REMOVEFILTERS('VotreTable'),
            'VotreTable'[Appops] = 'TableDesAppops'[Appops]  // Relation explicite
        )
    )
RETURN
    MINX(AppopsCount, [Comptage])
```

Cette formule s'assurera de compter correctement les occurrences même si les deux tables ne sont pas reliées dans votre modèle de données.

______


Dans ce cas, c'est logique que vous n'obteniez pas zéro. En effet, la formule DAX que j'ai proposée compte les occurrences d'Appops qui existent dans les deux tables, mais elle ne retourne pas 0 pour les Appops qui n'existent pas dans la table principale.

Voici une formule modifiée qui prendra en compte les valeurs absentes et retournera correctement 0 comme valeur minimale si certaines Appops de votre liste ne sont pas présentes:

```
MinValue = 
VAR AppopsCount = 
    ADDCOLUMNS(
        VALUES('TableDesAppops'[Appops]),
        "Comptage", 
        CALCULATE(
            COUNTROWS('VotreTable'),
            'VotreTable'[Appops] = EARLIER('TableDesAppops'[Appops])
        )
    )
RETURN
    MINX(AppopsCount, [Comptage])
```

Cette formule:
1. Prend chaque valeur unique d'Appops dans votre liste (TableDesAppops)
2. Compte combien de fois elle apparaît dans la table principale (VotreTable)
3. Si une Appops n'apparaît pas du tout, son comptage sera 0
4. Enfin, elle renvoie la valeur minimale parmi tous ces comptages

Si certaines Appops de votre liste ne sont pas présentes dans VotreTable, la mesure devrait maintenant correctement retourner 0.


____


Vous avez raison de poser cette question. Dans la formule que j'ai proposée, la relation entre les tables est implicite à travers cette partie:

```
CALCULATE(
    COUNTROWS('VotreTable'),
    'VotreTable'[Appops] = EARLIER('TableDesAppops'[Appops])
)
```

Ce code dit spécifiquement à DAX de compter les lignes dans 'VotreTable' où la valeur de la colonne 'VotreTable'[Appops] est égale à la valeur actuelle de 'TableDesAppops'[Appops] (c'est ce que fait la fonction EARLIER).

Et oui, vous avez tout à fait raison - puisque les deux tables sont reliées par une relation sur leurs colonnes Appops respectives, DAX utilise automatiquement cette relation lors du calcul. 

Sans cette condition explicite ou cette relation établie dans le modèle, DAX ne saurait pas quelle colonne utiliser pour faire correspondre les valeurs entre les deux tables.

Si vous voulez rendre la formule plus explicite même avec la relation existante, vous pourriez l'écrire ainsi:

```
MinValue = 
VAR AppopsCount = 
    ADDCOLUMNS(
        VALUES('TableDesAppops'[Appops]), 
        "Comptage", 
        CALCULATE(
            COUNTROWS('VotreTable'),
            'VotreTable'[Appops] = EARLIER('TableDesAppops'[Appops])
        )
    )
RETURN
    MINX(AppopsCount, [Comptage])
```

Cette version spécifie clairement que nous comptons les occurrences où les valeurs d'Appops correspondent entre les deux tables.


Je comprends mieux le problème maintenant. Effectivement, la formule renvoie la valeur minimale parmi les comptages non nuls, mais ne tient pas compte des Appops qui sont complètement absentes (qui devraient avoir un comptage de 0).

Pour résoudre ce problème, nous devons être sûrs que la formule inclut également les Appops qui ne sont pas du tout présentes dans la table principale. Voici une formule modifiée:

```
MinValue = 
VAR AppopsCount = 
    ADDCOLUMNS(
        VALUES('TableDesAppops'[Appops]),
        "Comptage", 
        COALESCE(
            CALCULATE(
                COUNTROWS('VotreTable'),
                FILTER(
                    ALL('VotreTable'),
                    'VotreTable'[Appops] = EARLIER('TableDesAppops'[Appops])
                )
            ),
            0
        )
    )
RETURN
    MINX(AppopsCount, [Comptage])
```

Cette formule utilise:
1. FILTER avec ALL pour s'assurer qu'on examine toutes les lignes sans tenir compte des relations existantes
2. COALESCE pour s'assurer que si aucune correspondance n'est trouvée, la valeur sera 0 et non vide
3. MINX pour trouver la valeur minimale parmi tous ces comptages, y compris les zéros

Si vous avez plusieurs Appops qui sont absentes de la table principale (avec un comptage de 0) et d'autres qui sont présentes (avec des comptages de 1 ou plus), cette formule devrait correctement renvoyer 0 comme valeur minimale.



_____


Pour obtenir simplement le nombre maximal d'occurrences d'une Appops dans votre table principale, sans vous référer à la table de référence, voici une formule DAX plus simple:

```
MaxOccurrences = 
VAR CountByAppops = 
    SUMMARIZE(
        'VotreTable',
        'VotreTable'[Appops],
        "NombreOccurrences", COUNTX(CURRENTGROUP(), 1)
    )
RETURN
    MAXX(CountByAppops, [NombreOccurrences])
```

Cette formule:
1. Regroupe les données de 'VotreTable' par la colonne [Appops]
2. Compte le nombre d'occurrences pour chaque valeur d'Appops
3. Retourne la valeur maximale parmi ces comptages

C'est effectivement une approche plus directe pour obtenir le maximum si vous n'avez pas besoin de vous référer à la table de référence. Cette méthode considère uniquement les Appops qui existent réellement dans 'VotreTable', ce qui explique pourquoi vous obtenez 41 et non 127.

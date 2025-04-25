L'erreur indique : "La fonction VALUES attend une expression de référence de colonne ou une expression de référence de table pour l'argument '1'."

Le problème est visible dans votre code. Dans la deuxième image, on voit que vous utilisez:

```
VALUES(REMOVEFILTERS('Appops_'[Appops]))
```

C'est ce qui provoque l'erreur. La fonction VALUES attend directement une référence de colonne ou de table, mais vous lui passez une expression (REMOVEFILTERS) à la place.

Pour résoudre ce problème, il faut restructurer votre code. L'idée est que vous voulez récupérer toutes les valeurs de la colonne Appops de la table Appops_, sans tenir compte des filtres. Il y a deux approches principales pour corriger cela:

1. Appliquer REMOVEFILTERS au niveau du CALCULATE plutôt qu'au niveau des VALUES
2. Utiliser ALL à la place, qui est une meilleure approche dans ce cas

Voici comment corriger ce code:

Je vois l'erreur dans votre formule. Le message indique : "La fonction VALUES attend une expression de référence de colonne ou une expression de référence de table pour l'argument '1'."

Le problème est que vous avez placé REMOVEFILTERS à l'intérieur de VALUES, ce qui n'est pas une construction valide en DAX. La fonction VALUES doit recevoir directement une référence de colonne ou de table.

Voici comment corriger votre formule :

```
MinValueWindows = 
VAR AppopsCount = 
    ADDCOLUMNS(
        VALUES('Appops_'[Appops]),  // D'abord référence directe à la colonne
        "Comptage", 
        COALESCE(
            CALCULATE(
                COUNTROWS('windows_new'),
                REMOVEFILTERS('windows_new'),  // Ici le REMOVEFILTERS est correctement placé
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

Alternativement, vous pourriez utiliser une approche avec ALL qui accomplit la même chose que REMOVEFILTERS pour votre cas :

```
MinValueWindows = 
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
RETURN
    MINX(AppopsCount, [Comptage])
```

L'objectif est le même - vous voulez ignorer tous les filtres sur Appops_, mais la construction syntaxique doit être correcte pour que DAX puisse l'interpréter.


Je vois l'erreur dans le code. Il semble que le message d'erreur indique "Il ne s'agit pas d'une expression DAX valide."

En examinant le code, je remarque une potentielle erreur dans la construction de l'appel EXISTS. Voici ce qui pourrait causer le problème:

1. Dans la ligne 12, l'appel à EXISTS est problématique: 
   ```
   && EXISTS(VALUES('Appops_'[Appops]), 'Appops_'[Appops] = 'windows_new'[Appops])
   ```

La syntaxe d'EXISTS est peut-être incorrecte. Analysons les problèmes possibles:

1. La fonction EXISTS peut être utilisée incorrectement. Dans DAX, la fonction EXISTS a une syntaxe spécifique et attend certains types d'arguments.

2. Il est possible que l'imbrication de VALUES à l'intérieur d'EXISTS crée un problème syntaxique.

3. La condition à l'intérieur d'EXISTS pourrait avoir une syntaxe invalide.

Pour corriger ce problème, je vais proposer une formule modifiée qui utilise une approche légèrement différente mais qui devrait accomplir la même chose, en évitant l'utilisation problématique d'EXISTS:

Je vois l'erreur dans votre formule. Le problème se trouve dans l'utilisation d'EXISTS qui n'est pas correctement structurée ici.

La syntaxe de la fonction EXISTS dans DAX est assez spécifique et votre implémentation actuelle n'est pas valide. Essayons une autre approche :

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
                && NOT(ISBLANK('windows_new'[Appops]))
            )
        )
    )
VAR AppsWithNonZeroCounts = 
    FILTER(AppopsCount, [Comptage] > 0)
RETURN
    MAXX(AppsWithNonZeroCounts, [Comptage])
```

Ou essayez cette version alternative qui utilise TREATAS pour créer une relation virtuelle plus propre :

```
MaxValueWindows = 
VAR AppopsCount = 
    ADDCOLUMNS(
        ALL('Appops_'[Appops]),
        "Comptage", 
        CALCULATE(
            COUNTROWS('windows_new'),
            REMOVEFILTERS('windows_new'),
            TREATAS(VALUES('Appops_'[Appops]), 'windows_new'[Appops])
        )
    )
VAR AppsWithNonZeroCounts = 
    FILTER(AppopsCount, [Comptage] > 0)
RETURN
    MAXX(AppsWithNonZeroCounts, [Comptage])
```

La version avec TREATAS est particulièrement élégante car elle crée une relation virtuelle entre les deux tables qui ignore naturellement les valeurs vides non correspondantes.

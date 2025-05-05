L'utilisateur m'indique qu'il y a un problème avec les mesures que j'ai fournies pour MinValueObsoTermine et MaxValueObsoTermine. Les deux mesures renvoient la même valeur: 1646.

C'est surprenant car les formules sont correctes dans leur structure, mais il y a visiblement un problème de fond.

Analysons ce qui pourrait causer ce problème:

1. Il est possible que toutes les Appops aient exactement le même nombre de NOM_CS avec statut "Terminé" (1646).

2. Il est possible que le filtre sur le statut "Terminé" ne fonctionne pas correctement.

3. Il est possible que DISTINCTCOUNT ne fonctionne pas comme prévu avec cette structure de données.

4. Il pourrait y avoir un problème avec la façon dont nous utilisons ALL et REMOVEFILTERS.

Pour résoudre ce problème, je vais proposer une approche différente qui devrait mieux fonctionner. Au lieu d'utiliser ALL sur 'obso_planifall'[Appops], nous allons d'abord identifier toutes les Appops distinctes, puis compter pour chacune:

Je vois que les deux mesures renvoient la même valeur (1646), ce qui indique un problème dans notre approche.

Le problème vient probablement de notre utilisation d'ALL qui pourrait récupérer des valeurs non pertinentes ou d'une mauvaise interaction avec REMOVEFILTERS.

Essayons une approche différente, en utilisant SUMMARIZE pour d'abord compter clairement les NOM_CS distincts par Appops:

```
MinValueObsoTermine = 
VAR ComptageParAppops = 
    SUMMARIZE(
        ALLSELECTED('obso_planifall'),
        'obso_planifall'[Appops],
        "NombreNomCSTermine", 
        CALCULATE(
            DISTINCTCOUNT('obso_planifall'[NOM_CS]),
            'obso_planifall'[statut] = "Terminé"
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
VAR ComptageParAppops = 
    SUMMARIZE(
        ALLSELECTED('obso_planifall'),
        'obso_planifall'[Appops],
        "NombreNomCSTermine", 
        CALCULATE(
            DISTINCTCOUNT('obso_planifall'[NOM_CS]),
            'obso_planifall'[statut] = "Terminé"
        )
    )
RETURN
    MAXX(ComptageParAppops, [NombreNomCSTermine])
```

Cette approche est différente car:
1. Utilise ALLSELECTED au lieu d'ALL pour préserver certains filtres contextuels
2. Utilise SUMMARIZE directement sur la table complète
3. Simplifie la structure globale
4. Pour le MIN, filtre pour exclure les zéros (qui pourraient fausser le résultat)

Si ces mesures ne fonctionnent toujours pas correctement, nous pourrions avoir besoin de regarder plus en détail la structure de vos données.


________

Il semble que la mesure fonctionne maintenant, mais elle réagit au segment Appops alors que vous souhaitez qu'elle ne réagisse pas à ce filtre.

Le problème vient probablement de l'utilisation de ALLSELECTED qui préserve certains filtres contextuels. Notre objectif est d'ignorer spécifiquement les filtres sur Appops.

Pour corriger cela, je vais modifier les formules pour utiliser une combinaison de ALL pour obtenir toutes les valeurs d'Appops et placer correctement REMOVEFILTERS dans le CALCULATE.

Voici les formules corrigées:

Je comprends le problème - les mesures réagissent au filtre Appops alors qu'elles ne devraient pas. Voici les formules corrigées:

```
MinValueObsoTermine = 
VAR ComptageParAppops = 
    ADDCOLUMNS(
        ALL('obso_planifall'[Appops]),
        "NombreNomCSTermine", 
        CALCULATE(
            DISTINCTCOUNT('obso_planifall'[NOM_CS]),
            'obso_planifall'[statut] = "Terminé",
            REMOVEFILTERS('obso_planifall'[Appops])
        )
    )
VAR ResultatsNonNuls = 
    FILTER(ComptageParAppops, [NombreNomCSTermine] > 0)
RETURN
    IF(
        COUNTROWS(ResultatsNonNuls) > 0,
        MINX(ResultatsNonNuls, [NombreNomCSTermine]),
        0
    )
```

```
MaxValueObsoTermine = 
VAR ComptageParAppops = 
    ADDCOLUMNS(
        ALL('obso_planifall'[Appops]),
        "NombreNomCSTermine", 
        CALCULATE(
            DISTINCTCOUNT('obso_planifall'[NOM_CS]),
            'obso_planifall'[statut] = "Terminé",
            REMOVEFILTERS('obso_planifall'[Appops])
        )
    )
RETURN
    MAXX(ComptageParAppops, [NombreNomCSTermine])
```

```
AvgValueObsoTermine = 
VAR ComptageParAppops = 
    ADDCOLUMNS(
        ALL('obso_planifall'[Appops]),
        "NombreNomCSTermine", 
        CALCULATE(
            DISTINCTCOUNT('obso_planifall'[NOM_CS]),
            'obso_planifall'[statut] = "Terminé",
            REMOVEFILTERS('obso_planifall'[Appops])
        )
    )
VAR ResultatsNonNuls = 
    FILTER(ComptageParAppops, [NombreNomCSTermine] > 0)
RETURN
    AVERAGEX(ResultatsNonNuls, [NombreNomCSTermine])
```

Les changements clés:
1. Utilisation d'ADDCOLUMNS avec ALL pour obtenir toutes les valeurs d'Appops
2. Placement correct de REMOVEFILTERS à l'intérieur du CALCULATE
3. Filtre des valeurs nulles avant d'appliquer MINX et AVERAGEX
4. Vérification supplémentaire pour MIN pour éviter les erreurs si aucun résultat

Ces mesures devraient maintenant ignorer les filtres sur Appops tout en calculant correctement le minimum, le maximum et la moyenne.

22222222222

Le problème persiste avec les mesures MIN et MAX qui renvoient toutes deux la même valeur (1646). Essayons une approche différente.

Le problème pourrait être que nous ne traitons pas correctement les filtres ou que nous approchons mal le calcul.

Je vais essayer une approche complètement différente avec DETAILROWS, qui peut nous donner plus de visibilité sur ce que nous comptons réellement:

```
MinValueObsoTermine = 
VAR AllAppops = ALL('obso_planifall'[Appops])
VAR ComptageComplet = 
    GENERATE(
        AllAppops,
        ADDCOLUMNS(
            ROW("Dummy", 1),
            "NombreNomCSTermine", 
            CALCULATE(
                DISTINCTCOUNT('obso_planifall'[NOM_CS]),
                'obso_planifall'[statut] = "Terminé",
                ALLEXCEPT('obso_planifall', 'obso_planifall'[Appops])
            )
        )
    )
RETURN
    IF(
        HASONEVALUE('obso_planifall'[Appops]),
        CALCULATE(
            DISTINCTCOUNT('obso_planifall'[NOM_CS]),
            'obso_planifall'[statut] = "Terminé"
        ),
        MINX(ComptageComplet, [NombreNomCSTermine])
    )
```

```
MaxValueObsoTermine = 
VAR AllAppops = ALL('obso_planifall'[Appops])
VAR ComptageComplet = 
    GENERATE(
        AllAppops,
        ADDCOLUMNS(
            ROW("Dummy", 1),
            "NombreNomCSTermine", 
            CALCULATE(
                DISTINCTCOUNT('obso_planifall'[NOM_CS]),
                'obso_planifall'[statut] = "Terminé",
                ALLEXCEPT('obso_planifall', 'obso_planifall'[Appops])
            )
        )
    )
RETURN
    IF(
        HASONEVALUE('obso_planifall'[Appops]),
        CALCULATE(
            DISTINCTCOUNT('obso_planifall'[NOM_CS]),
            'obso_planifall'[statut] = "Terminé"
        ),
        MAXX(ComptageComplet, [NombreNomCSTermine])
    )
```

Ces formules plus complexes:
1. Utilisent ALLEXCEPT, qui semble plus fiable que REMOVEFILTERS dans certains contextes
2. Différencient le comportement quand une seule Appops est sélectionnée vs. quand plusieurs/aucune sont sélectionnées
3. Utilisent GENERATE pour plus de contrôle sur la génération de la table de comptage

Si ces mesures ne fonctionnent toujours pas, je vais proposer une approche radicalement différente qui sépare clairement les comptages par Appops:

```
MinValueObsoTermine = 
MINX(
    SUMMARIZE(
        FILTER(
            CALCULATETABLE(
                GROUPBY(
                    FILTER(
                        'obso_planifall',
                        'obso_planifall'[statut] = "Terminé"
                    ),
                    'obso_planifall'[Appops],
                    "NombreNomCS", DISTINCTCOUNT('obso_planifall'[NOM_CS])
                ),
                ALLEXCEPT('obso_planifall', 'obso_planifall'[statut])
            ),
            [NombreNomCS] > 0
        ),
        [Appops],
        "Comptage", [NombreNomCS]
    ),
    [Comptage]
)
```

Cette approche élimine toute ambiguïté en construisant explicitement une table de comptage, groupée par Appops, puis en trouvant le minimum.

Il est aussi possible que la valeur 1646 soit effectivement correcte (peut-être que toutes les Appops ont le même nombre de NOM_CS terminés). Dans ce cas, il pourrait être utile de vérifier cela en créant une mesure temporaire qui affiche tous les comptages:

```
Debug_Comptage_Appops = 
CONCATENATEX(
    ADDCOLUMNS(
        ALL('obso_planifall'[Appops]),
        "NombreNomCSTermine", 
        CALCULATE(
            DISTINCTCOUNT('obso_planifall'[NOM_CS]),
            'obso_planifall'[statut] = "Terminé",
            REMOVEFILTERS('obso_planifall'[Appops])
        )
    ),
    [Appops] & ": " & [NombreNomCSTermine],
    ", "
)
```

Cette mesure de débogage afficherait tous les comptages par Appops pour vérifier si effectivement ils sont tous identiques ou s'il y a un problème avec notre logique.

Examinons plus attentivement notre problème:

Le choix de 1646 comme valeur semble très spécifique. Il est possible que ce ne soit pas un comptage par Appops, mais plutôt le nombre total de NOM_CS uniques avec statut "Terminé" dans toute la table. Si c'est le cas, cela signifierait que notre formule ne filtre pas correctement par Appops comme prévu.

Essayons une autre approche en utilisant ADDCOLUMNS et VALUES:

```
MinValueObsoTermine = 
VAR DistinctAppops = VALUES('obso_planifall'[Appops])
VAR ComptageParAppops = 
    ADDCOLUMNS(
        ALL('obso_planifall'[Appops]),
        "Count", 
        CALCULATE(
            DISTINCTCOUNT('obso_planifall'[NOM_CS]),
            'obso_planifall'[statut] = "Terminé"
        )
    )
RETURN
    MINX(
        FILTER(ComptageParAppops, [Count] > 0),
        [Count]
    )
```

Cette approche simplifie le problème en retirant complètement REMOVEFILTERS et en utilisant CALCULATE avec un filtre simple sur le statut.

Réfléchissons différemment. Peut-être que le problème est que MINX/MAXX fonctionnent sur une colonne de notre table intermédiaire, mais que cette colonne ne contient pas les valeurs que nous attendons.

Essayons une approche très directe avec SUMMARIZE et des agrégations directes:

```
MinValueObsoTermine = 
VAR ComptageParAppops = 
    SUMMARIZE(
        'obso_planifall',
        'obso_planifall'[Appops],
        "Comptage", CALCULATE(
            DISTINCTCOUNT('obso_planifall'[NOM_CS]),
            'obso_planifall'[statut] = "Terminé"
        )
    )
RETURN
    CALCULATE(
        MIN(ComptageParAppops[Comptage]),
        FILTER(ComptageParAppops, [Comptage] > 0)
    )
```

Cette approche crée une table de synthèse explicite avec SUMMARIZE et applique ensuite MIN directement à la colonne appropriée.

Je constate que nous avons un problème persistant avec ces mesures qui renvoient toutes la même valeur (1646). Cette valeur est probablement le nombre total de NOM_CS distincts avec statut "Terminé" dans toute la table, plutôt que les valeurs par Appops.

Essayons une approche complètement différente :

```
MinValueObsoTermine = 
VAR CalculParAppops =
    SUMMARIZE(
        'obso_planifall',
        'obso_planifall'[Appops],
        "NbreNomCSTermine", 
        CALCULATE(
            DISTINCTCOUNT('obso_planifall'[NOM_CS]),
            'obso_planifall'[statut] = "Terminé"
        )
    )
RETURN
    CALCULATE(
        MIN(CalculParAppops[NbreNomCSTermine]),
        FILTER(CalculParAppops, [NbreNomCSTermine] > 0)
    )
```

```
MaxValueObsoTermine = 
VAR CalculParAppops =
    SUMMARIZE(
        'obso_planifall',
        'obso_planifall'[Appops],
        "NbreNomCSTermine", 
        CALCULATE(
            DISTINCTCOUNT('obso_planifall'[NOM_CS]),
            'obso_planifall'[statut] = "Terminé"
        )
    )
RETURN
    MAX(CalculParAppops[NbreNomCSTermine])
```

```
AvgValueObsoTermine = 
VAR CalculParAppops =
    SUMMARIZE(
        'obso_planifall',
        'obso_planifall'[Appops],
        "NbreNomCSTermine", 
        CALCULATE(
            DISTINCTCOUNT('obso_planifall'[NOM_CS]),
            'obso_planifall'[statut] = "Terminé"
        )
    )
RETURN
    AVERAGE(CalculParAppops[NbreNomCSTermine])
```

Cette approche est beaucoup plus directe :
1. Elle crée explicitement une table qui regroupe par Appops
2. Elle calcule le nombre distinct de NOM_CS avec statut "Terminé" pour chaque Appops
3. Elle applique ensuite MIN, MAX ou AVERAGE directement sur cette colonne

Si ces mesures réagissent toujours au filtre Appops, nous devrons ajouter ALLSELECTED ou ALL à l'intérieur du SUMMARIZE.


33333333333333

L'utilisateur me montre une formule DAX pour une mesure appelée "MinValueObsoTermine" et indique que cette formule fonctionne, mais qu'il manque un REMOVEFILTERS sur la colonne Appops.

Voici la formule telle qu'elle apparaît dans l'image :

```
MinValueObsoTermine =
VAR ComptageParAppops =
    SUMMARIZE(
        ALLSELECTED('obso_planifall'),
        'obso_planifall'[Appops],
        "NombreNomCSTermine",
        CALCULATE(
            DISTINCTCOUNT('obso_planifall'[NOM_CS]),
            'obso_planifall'[statut] = "5 - Terminé"
        )
    )
RETURN
    MINX(
        FILTER(ComptageParAppops, [NombreNomCSTermine] > 0),
        [NombreNomCSTermine]
    )
```

Pour résoudre le problème, je dois ajouter un REMOVEFILTERS sur la colonne Appops. La meilleure façon de le faire est de l'ajouter dans le CALCULATE. Voici ma proposition de correction :

```
MinValueObsoTermine =
VAR ComptageParAppops =
    SUMMARIZE(
        ALLSELECTED('obso_planifall'),
        'obso_planifall'[Appops],
        "NombreNomCSTermine",
        CALCULATE(
            DISTINCTCOUNT('obso_planifall'[NOM_CS]),
            'obso_planifall'[statut] = "5 - Terminé",
            REMOVEFILTERS('obso_planifall'[Appops])  // Ajout du REMOVEFILTERS
        )
    )
RETURN
    MINX(
        FILTER(ComptageParAppops, [NombreNomCSTermine] > 0),
        [NombreNomCSTermine]
    )
```

Je vais également préparer des formules similaires pour les mesures MaxValueObsoTermine et AvgValueObsoTermine.

Je vois que la formule fonctionne mais qu'il manque le REMOVEFILTERS sur la colonne Appops. Voici les formules corrigées pour les trois mesures :

```
MinValueObsoTermine = 
VAR ComptageParAppops =
    SUMMARIZE(
        ALLSELECTED('obso_planifall'),
        'obso_planifall'[Appops],
        "NombreNomCSTermine", 
        CALCULATE(
            DISTINCTCOUNT('obso_planifall'[NOM_CS]),
            'obso_planifall'[statut] = "5 - Terminé",
            REMOVEFILTERS('obso_planifall'[Appops])  // Ajout du REMOVEFILTERS
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
VAR ComptageParAppops =
    SUMMARIZE(
        ALLSELECTED('obso_planifall'),
        'obso_planifall'[Appops],
        "NombreNomCSTermine", 
        CALCULATE(
            DISTINCTCOUNT('obso_planifall'[NOM_CS]),
            'obso_planifall'[statut] = "5 - Terminé",
            REMOVEFILTERS('obso_planifall'[Appops])  // Ajout du REMOVEFILTERS
        )
    )
RETURN
    MAXX(ComptageParAppops, [NombreNomCSTermine])
```

```
AvgValueObsoTermine = 
VAR ComptageParAppops =
    SUMMARIZE(
        ALLSELECTED('obso_planifall'),
        'obso_planifall'[Appops],
        "NombreNomCSTermine", 
        CALCULATE(
            DISTINCTCOUNT('obso_planifall'[NOM_CS]),
            'obso_planifall'[statut] = "5 - Terminé",
            REMOVEFILTERS('obso_planifall'[Appops])  // Ajout du REMOVEFILTERS
        )
    )
RETURN
    AVERAGEX(
        FILTER(ComptageParAppops, [NombreNomCSTermine] > 0),
        [NombreNomCSTermine]
    )
```

Ces formules devraient maintenant correctement :
1. Ignorer les filtres sur Appops grâce au REMOVEFILTERS
2. Calculer pour chaque Appops le nombre distinct de NOM_CS avec statut "5 - Terminé"
3. Retourner respectivement le minimum, maximum et moyenne de ces comptages

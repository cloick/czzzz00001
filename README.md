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

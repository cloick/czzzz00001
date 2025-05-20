Je comprends maintenant le problème. Les erreurs que vous avez rencontrées sont liées à certaines syntaxes que j'ai utilisées qui ne sont pas compatibles avec votre version de Power BI. Merci pour les exemples partagés qui sont très instructifs !

Je remarque que dans votre deuxième image, vous utilisez `REMOVEFILTERS` et `TREATAS` combinés de manière efficace, et dans la première image, vous utilisez `ALL` avec `ADDCOLUMNS` suivi d'un `AVERAGEX`. Ce sont d'excellentes approches.

Voici une mesure adaptée à votre syntaxe, inspirée des exemples que vous m'avez partagés :

```
MoyenneTauxLiaisonGlobale = 
VAR DerniereDate = MAX('v_stat_kpi'[dat])
VAR CurrentLibelleFilter = VALUES('v_stat_kpi'[Libellé])

// Calcul pour la moyenne des Appops
RETURN
DIVIDE(
    // Moyenne des valeurs pour "chaine de liaison"
    CALCULATE(
        AVERAGE('v_stat_kpi'[val]),
        'v_stat_kpi'[Libellé] = "Nombre de solutions matière OPEN avec une chaine de liaison",
        'v_stat_kpi'[Libellé] <> "Non pris",
        'v_stat_kpi'[Appops] <> "Non pris",
        'v_stat_kpi'[dat] = DerniereDate,
        REMOVEFILTERS('Appops_'[Appops])
    ),
    
    // Moyenne des valeurs pour "au moins une IDD"
    CALCULATE(
        AVERAGE('v_stat_kpi'[val]),
        'v_stat_kpi'[Libellé] = "Nombre de solutions matière OPEN avec au moins une IDD",
        'v_stat_kpi'[Libellé] <> "Non pris",
        'v_stat_kpi'[Appops] <> "Non pris",
        'v_stat_kpi'[dat] = DerniereDate,
        REMOVEFILTERS('Appops_'[Appops])
    ),
    0
)
```

Alternativement, en s'inspirant plus directement de votre premier exemple avec ADDCOLUMNS et AVERAGEX :

```
MoyenneTauxLiaisonGlobale = 
VAR DerniereDate = MAX('v_stat_kpi'[dat])

// Calcul pour tous les Appops
VAR AppsCount = 
    ADDCOLUMNS(
        ALL('Appops_'[Appops]),
        "Numerateur", 
            CALCULATE(
                VALUES('v_stat_kpi'[val]),
                'v_stat_kpi'[dat] = DerniereDate,
                'v_stat_kpi'[Libellé] = "Nombre de solutions matière OPEN avec une chaine de liaison",
                TREATAS(VALUES('Appops_'[Appops]), 'Appops_'[Appops])
            ),
        "Denominateur", 
            CALCULATE(
                VALUES('v_stat_kpi'[val]),
                'v_stat_kpi'[dat] = DerniereDate,
                'v_stat_kpi'[Libellé] = "Nombre de solutions matière OPEN avec au moins une IDD",
                TREATAS(VALUES('Appops_'[Appops]), 'Appops_'[Appops])
            )
    )

// Filtrer pour éviter les divisions par zéro
VAR AppsWithNonZeroDenominators = 
    FILTER(AppsCount, [Denominateur] > 0)

RETURN
    AVERAGEX(AppsWithNonZeroDenominators, DIVIDE([Numerateur], [Denominateur], 0))
```

Ces deux approches devraient :
1. Ignorer le filtre actuel sur Appops grâce à REMOVEFILTERS ou ALL
2. Calculer soit la moyenne directe des taux, soit calculer le taux pour chaque Appops puis en faire la moyenne
3. Ne pas être affectées par les sélections d'Appops sur la page

Essayez ces formules et voyez laquelle fonctionne le mieux dans votre contexte. Elles sont plus proches de la syntaxe utilisée dans vos exemples.

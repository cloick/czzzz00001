Je vous prie de m'excuser pour cette erreur. Vous avez raison, la mesure réagit encore aux filtres. Pour créer une mesure qui ignore complètement les filtres de contexte et renvoie toujours la même moyenne globale, quelle que soit la sélection, voici une meilleure approche :

```
MoyenneTauxLiaisonFixe = 
CALCULATE(
    // Corps principal du calcul
    VAR DerniereDate = MAX(ALL('v_stat_kpi'[dat])) // Date la plus récente, sans filtre
    RETURN
    AVERAGEX(
        VALUES(ALL(Appops_[Appops])), // Parcours de toutes les Appops, sans filtre
        VAR AppopsCourante = Appops_[Appops]
        VAR Numerateur = 
            CALCULATE(
                VALUES('v_stat_kpi'[val]),
                ALL(), // Supprimer tous les filtres
                'v_stat_kpi'[dat] = DerniereDate,
                'v_stat_kpi'[Libellé] = "Nombre de solutions matière OPEN avec une chaine de liaison",
                Appops_[Appops] = AppopsCourante
            )
        VAR Denominateur = 
            CALCULATE(
                VALUES('v_stat_kpi'[val]),
                ALL(), // Supprimer tous les filtres
                'v_stat_kpi'[dat] = DerniereDate,
                'v_stat_kpi'[Libellé] = "Nombre de solutions matière OPEN avec au moins une IDD",
                Appops_[Appops] = AppopsCourante
            )
        RETURN
            DIVIDE(Numerateur, Denominateur, 0)
    ),
    ALL() // Cette fonction supprime TOUS les filtres de contexte
)
```

L'utilisation de `ALL()` comme modificateur de contexte dans le CALCULATE extérieur garantit que tous les filtres sont supprimés. Cela assure que la mesure ne réagira pas aux sélections sur la page.

Si cette approche ne fonctionne toujours pas, une autre méthode plus directe serait :

```
MoyenneTauxLiaisonFixe = 
VAR DerniereDate = CALCULATE(MAX('v_stat_kpi'[dat]), ALL('v_stat_kpi'))

// Créer une table temporaire de toutes les Appops
VAR ToutesLesAppops = 
    ADDCOLUMNS(
        ALL(Appops_[Appops]),
        "Numerateur", 
            CALCULATE(
                VALUES('v_stat_kpi'[val]),
                ALL(),
                'v_stat_kpi'[dat] = DerniereDate,
                'v_stat_kpi'[Libellé] = "Nombre de solutions matière OPEN avec une chaine de liaison",
                Appops_[Appops] = Appops_[Appops]
            ),
        "Denominateur", 
            CALCULATE(
                VALUES('v_stat_kpi'[val]),
                ALL(),
                'v_stat_kpi'[dat] = DerniereDate,
                'v_stat_kpi'[Libellé] = "Nombre de solutions matière OPEN avec au moins une IDD",
                Appops_[Appops] = Appops_[Appops]
            )
    )

// Calculer la moyenne des ratios, en ignorant les divisions par zéro
VAR MoyenneRatios = 
    AVERAGEX(
        FILTER(ToutesLesAppops, [Denominateur] > 0),
        DIVIDE([Numerateur], [Denominateur], 0)
    )

RETURN
    MoyenneRatios
```

Cette seconde approche est peut-être plus claire et devrait ignorer complètement tous les filtres de contexte. De plus, elle filtre les cas où le dénominateur serait zéro pour éviter de fausser la moyenne.

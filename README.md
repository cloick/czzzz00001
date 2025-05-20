Je comprends parfaitement votre besoin ! Vous souhaitez créer une mesure qui calcule la moyenne globale de ce pourcentage pour toutes les Appops, et cette mesure doit ignorer les filtres de contexte (notamment les filtres sur Appops) pour toujours afficher la moyenne générale.

Pour cela, nous allons utiliser les fonctions ALLSELECTED ou ALL pour supprimer les filtres de contexte. Voici la mesure que je vous propose :

```
MoyenneTauxLiaisonToutes = 
VAR DerniereDate = MAX('v_stat_kpi'[dat])
RETURN
AVERAGEX(
    ALL(Appops_[Appops]), // Ignorer tous les filtres sur Appops
    DIVIDE(
        // Numérateur - Solutions avec chaîne de liaison
        CALCULATE(
            VALUES('v_stat_kpi'[val]),
            'v_stat_kpi'[dat] = DerniereDate,
            'v_stat_kpi'[Libellé] = "Nombre de solutions matière OPEN avec une chaine de liaison",
            TREATAS(VALUES(Appops_[Appops]), Appops_[Appops]) // Pour chaque Appops
        ),
        
        // Dénominateur - Solutions avec au moins une IDD
        CALCULATE(
            VALUES('v_stat_kpi'[val]),
            'v_stat_kpi'[dat] = DerniereDate,
            'v_stat_kpi'[Libellé] = "Nombre de solutions matière OPEN avec au moins une IDD",
            TREATAS(VALUES(Appops_[Appops]), Appops_[Appops]) // Pour chaque Appops
        ),
        0 // Valeur par défaut si division par zéro
    )
)
```

Et si vous voulez également que cette mesure ignore tous les filtres de la page (pas seulement les filtres sur Appops), vous pouvez utiliser la fonction CALCULATE avec ALLSELECTED :

```
MoyenneTauxLiaisonToutes = 
CALCULATE(
    VAR DerniereDate = MAX('v_stat_kpi'[dat])
    RETURN
    AVERAGEX(
        ALL(Appops_[Appops]), // Parcourir toutes les Appops
        DIVIDE(
            // Numérateur - Solutions avec chaîne de liaison
            CALCULATE(
                VALUES('v_stat_kpi'[val]),
                'v_stat_kpi'[dat] = DerniereDate,
                'v_stat_kpi'[Libellé] = "Nombre de solutions matière OPEN avec une chaine de liaison",
                Appops_[Appops] = Appops_[Appops] // Pour chaque Appops en cours d'itération
            ),
            
            // Dénominateur - Solutions avec au moins une IDD
            CALCULATE(
                VALUES('v_stat_kpi'[val]),
                'v_stat_kpi'[dat] = DerniereDate,
                'v_stat_kpi'[Libellé] = "Nombre de solutions matière OPEN avec au moins une IDD",
                Appops_[Appops] = Appops_[Appops] // Pour chaque Appops en cours d'itération
            ),
            0 // Valeur par défaut si division par zéro
        )
    ),
    ALLSELECTED() // Ignorer tous les filtres de contexte
)
```

Cette mesure va :
1. Ignorer tous les filtres actuels grâce à ALLSELECTED()
2. Parcourir toutes les valeurs distinctes de Appops_[Appops]
3. Pour chaque Appops, calculer le taux entre les deux mesures
4. Faire la moyenne de tous ces taux

N'oubliez pas de formater cette mesure en pourcentage dans les propriétés de formatage.

Est-ce que cette approche correspond à ce que vous cherchez ?

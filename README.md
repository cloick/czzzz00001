Pour créer les mesures MaxValueAudit et AvgValueAudit basées sur la même logique que MinValueAudit, je dois simplement remplacer MIN par MAX et AVERAGE respectivement, tout en gardant la même structure avec les filtres.

Le code actuel pour MinValueAudit est:
```
MinValueAudit = 
VAR DerniereDate = MAX('v_stat_kpi'[dat])
VAR CurrentLibelleFilter = VALUES('v_stat_kpi'[Libellé])
RETURN
CALCULATE(
    MIN('v_stat_kpi'[val]),
    'v_stat_kpi'[Libellé] <> "Non pris",
    'v_stat_kpi'[Appops] <> "Non pris",
    'v_stat_kpi'[dat] = DerniereDate,
    REMOVEFILTERS('Appops_'[Appops]),
    TREATAS(CurrentLibelleFilter, 'v_stat_kpi'[Libellé]) // Préserve le filtre Libellé
)
```

Pour MaxValueAudit, on remplace MIN par MAX:
```
MaxValueAudit = 
VAR DerniereDate = MAX('v_stat_kpi'[dat])
VAR CurrentLibelleFilter = VALUES('v_stat_kpi'[Libellé])
RETURN
CALCULATE(
    MAX('v_stat_kpi'[val]),
    'v_stat_kpi'[Libellé] <> "Non pris",
    'v_stat_kpi'[Appops] <> "Non pris",
    'v_stat_kpi'[dat] = DerniereDate,
    REMOVEFILTERS('Appops_'[Appops]),
    TREATAS(CurrentLibelleFilter, 'v_stat_kpi'[Libellé])
)
```

Pour AvgValueAudit, on remplace MIN par AVERAGE:
```
AvgValueAudit = 
VAR DerniereDate = MAX('v_stat_kpi'[dat])
VAR CurrentLibelleFilter = VALUES('v_stat_kpi'[Libellé])
RETURN
CALCULATE(
    AVERAGE('v_stat_kpi'[val]),
    'v_stat_kpi'[Libellé] <> "Non pris",
    'v_stat_kpi'[Appops] <> "Non pris",
    'v_stat_kpi'[dat] = DerniereDate,
    REMOVEFILTERS('Appops_'[Appops]),
    TREATAS(CurrentLibelleFilter, 'v_stat_kpi'[Libellé])
)
```

Tout le reste de la formule reste identique, car nous voulons appliquer exactement la même logique de filtrage et les mêmes conditions. Seule la fonction d'agrégation change (MIN, MAX, AVERAGE).

Merci ! Je suis ravi que ça fonctionne pour vous !

Pour le MAX, voici la formule (simplement remplacer MIN par MAX) :

```
MaxValueAudit = 
VAR DerniereDate = MAX('v_stat_kpi'[dat])
VAR CurrentLibelleFilter = VALUES('v_stat_kpi'[Libellé])
RETURN
CALCULATE(
    MAX('v_stat_kpi'[val]),  // Simplement remplacé MIN par MAX
    'v_stat_kpi'[Libellé] <> "Non pris",
    'v_stat_kpi'[Appops] <> "Non pris",
    'v_stat_kpi'[dat] = DerniereDate,
    REMOVEFILTERS('Appops_'[Appops]),
    TREATAS(CurrentLibelleFilter, 'v_stat_kpi'[Libellé])
)
```

Et pour la MOYENNE :

```
AvgValueAudit = 
VAR DerniereDate = MAX('v_stat_kpi'[dat])
VAR CurrentLibelleFilter = VALUES('v_stat_kpi'[Libellé])
RETURN
CALCULATE(
    AVERAGE('v_stat_kpi'[val]),  // Remplacé MIN par AVERAGE
    'v_stat_kpi'[Libellé] <> "Non pris",
    'v_stat_kpi'[Appops] <> "Non pris",
    'v_stat_kpi'[dat] = DerniereDate,
    REMOVEFILTERS('Appops_'[Appops]),
    TREATAS(CurrentLibelleFilter, 'v_stat_kpi'[Libellé])
)
```

Ces formules gardent exactement la même logique que votre MIN qui fonctionne bien - seule la fonction d'agrégation change. Elles vont :
- Respecter les filtres sur Libellé
- Ignorer les filtres sur Appops
- Filtrer pour exclure "Non pris"
- Se limiter à la date la plus récente

Si vous avez bien des lignes avec "nom de Solution open" dans la colonne [Libellé], essayons une approche différente. Le problème pourrait venir de la combinaison des filtres ou de la façon dont les contextes de filtres interagissent dans la formule.

Voici une nouvelle version qui devrait fonctionner, en séparant plus clairement les différentes parties de la logique :

```
TargetValueAudit = 
VAR DerniereDate = MAX('v_stat_kpi'[dat])
VAR SelectedAppops = SELECTEDVALUE('v_stat_kpi'[Appops], "")

// Nouvelle approche
VAR FilteredTable = 
    FILTER(
        'v_stat_kpi',
        'v_stat_kpi'[Libellé] = "nom de Solution open" &&
        'v_stat_kpi'[Appops] = SelectedAppops &&
        'v_stat_kpi'[Appops] <> "Non pris" &&
        'v_stat_kpi'[dat] = DerniereDate
    )

VAR Result = IF(SelectedAppops <> "", MAXX(FilteredTable, 'v_stat_kpi'[val]), BLANK())

RETURN Result
```

Cette approche utilise FILTER pour créer d'abord un sous-ensemble de la table avec toutes nos conditions, puis extrait la valeur de celle-ci. L'utilisation de MAXX au lieu de VALUES pourrait être plus appropriée selon la structure de vos données.

Si cela ne fonctionne toujours pas, nous pouvons essayer de diagnostiquer davantage en créant des mesures plus simples pour isoler le problème :

```
// Teste uniquement le filtre sur Libellé
Test1 = COUNTROWS(FILTER('v_stat_kpi', 'v_stat_kpi'[Libellé] = "nom de Solution open"))

// Teste la combinaison de filtres
Test2 = 
VAR SelectedAppops = SELECTEDVALUE('v_stat_kpi'[Appops], "")
VAR DerniereDate = MAX('v_stat_kpi'[dat])
RETURN
COUNTROWS(
    FILTER(
        'v_stat_kpi',
        'v_stat_kpi'[Libellé] = "nom de Solution open" &&
        'v_stat_kpi'[Appops] = SelectedAppops &&
        'v_stat_kpi'[dat] = DerniereDate
    )
)
```

Placez ces mesures Test dans votre visuel carte pour voir si elles renvoient des résultats non nuls. Cela vous aidera à identifier quelle partie de la condition pose problème.

Un autre problème potentiel pourrait être avec les espaces ou caractères invisibles dans "nom de Solution open". Vous pourriez essayer cette variante avec TRIM pour éliminer les espaces supplémentaires :

```
TargetValueAudit = 
VAR DerniereDate = MAX('v_stat_kpi'[dat])
VAR SelectedAppops = SELECTEDVALUE('v_stat_kpi'[Appops], "")

VAR FilteredTable = 
    FILTER(
        'v_stat_kpi',
        TRIM('v_stat_kpi'[Libellé]) = "nom de Solution open" &&
        'v_stat_kpi'[Appops] = SelectedAppops &&
        'v_stat_kpi'[Appops] <> "Non pris" &&
        'v_stat_kpi'[dat] = DerniereDate
    )

RETURN IF(SelectedAppops <> "", MAXX(FilteredTable, 'v_stat_kpi'[val]), BLANK())
```

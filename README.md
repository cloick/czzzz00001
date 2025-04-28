Je vois la mesure actuelle dans l'image. Je remarque que:

1. La partie pour FORMAT a été commentée avec //
2. Il y a une condition `'v_stat_kpi'[Appops] <> "Non pris"` à la place
3. Il n'y a pas de capture du filtre Libellé actuel

Je dois modifier cette mesure pour:
1. Garder la structure générale
2. Maintenir le filtre par date récente
3. Ajouter la logique pour conserver les filtres sur Libellé
4. S'assurer que la mesure ignore seulement les filtres sur Appops

Voici la formule que je vais proposer:

Voici la formule modifiée qui respectera les filtres sur Libellé tout en ignorant ceux sur Appops :

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

Les changements principaux sont :

1. Ajout de `VAR CurrentLibelleFilter = VALUES('v_stat_kpi'[Libellé])` pour capturer le filtre actuel sur Libellé
2. Ajout de `TREATAS(CurrentLibelleFilter, 'v_stat_kpi'[Libellé])` qui réapplique ce filtre après le REMOVEFILTERS
3. Conservation de votre condition `'v_stat_kpi'[Appops] <> "Non pris"` comme vous l'avez actuellement
4. Maintien du filtre sur la date la plus récente

Cette formule devrait maintenant :
- Réagir correctement aux sélections du segment Libellé
- Ignorer les sélections du segment Appops
- Toujours filtrer pour exclure "Non pris" et se limiter à la date la plus récente

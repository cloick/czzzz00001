Analysons ce que fait actuellement cette formule DAX:

```
MinValueAudit = 
CALCULATE(
    MIN('v_stat_kpi'[val]),
    'v_stat_kpi'[Libellé] <> "Non pris",
    'v_stat_kpi'[Appops] <> "",
    REMOVEFILTERS('Appops_'[Appops])
)
```

Cette formule calcule le minimum de la colonne [val] en appliquant plusieurs filtres:
1. Exclure les lignes où [Libellé] est "Non pris"
2. Exclure les lignes où [Appops] est vide
3. Ignorer tous les filtres sur la colonne Appops de la table Appops_

L'utilisateur veut:
1. Remplacer la vérification `'v_stat_kpi'[Appops] <> ""` par la technique FORMAT que nous avons utilisée précédemment (FORMAT([Appops], "") <> "")
2. Ajouter un filtre supplémentaire pour ne considérer que les lignes ayant la date la plus récente dans la colonne "dat"

Pour trouver la date la plus récente, nous pouvons utiliser MAX('v_stat_kpi'[dat]) et ensuite filtrer sur cette valeur.

Voici la formule modifiée:

Voici la formule DAX améliorée avec la technique FORMAT et le filtre pour la date la plus récente :

```
MinValueAudit = 
VAR DerniereDate = MAX('v_stat_kpi'[dat])
RETURN
CALCULATE(
    MIN('v_stat_kpi'[val]),
    'v_stat_kpi'[Libellé] <> "Non pris",
    FORMAT('v_stat_kpi'[Appops], "") <> "",  // Technique FORMAT au lieu de <>
    'v_stat_kpi'[dat] = DerniereDate,  // Filtre pour considérer uniquement la date la plus récente
    REMOVEFILTERS('Appops_'[Appops])
)
```

Cette formule améliorée :
1. Utilise la variable `DerniereDate` pour déterminer la date la plus récente dans la colonne [dat]
2. Remplace la condition `<> ""` par la technique FORMAT qui est plus robuste pour détecter tous les types de valeurs vides
3. Ajoute un filtre pour ne considérer que les lignes correspondant à la date la plus récente
4. Conserve l'exclusion des lignes "Non pris" et le REMOVEFILTERS pour ignorer les filtres sur Appops_

Cette nouvelle version sera plus précise pour détecter les valeurs vides et ne prendra en compte que les données les plus récentes pour votre calcul de minimum.

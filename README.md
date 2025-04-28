Je comprends mieux maintenant. Pour une visualisation de type jauge, la structure typique est:
- Une valeur actuelle (par exemple la moyenne, minimum, ou une valeur spécifique)
- Une valeur cible (target)

La personne veut que la targetValue soit simplement la valeur exacte (colonne [val]) correspondant à la sélection d'Appops et de Libellé spécifiques. Ce n'est pas une agrégation comme MIN, MAX ou AVERAGE.

Dans ce cas, il s'agit plutôt d'obtenir une valeur spécifique de la colonne [val] qui correspond exactement aux sélections de filtres d'Appops et de Libellé (et aussi la date la plus récente). Dans une jauge, cette valeur servirait de "cible" que l'on souhaite atteindre.

Je vais corriger ma formule pour qu'elle corresponde à ce besoin.

Vous avez raison, j'ai mal interprété l'objectif de cette mesure. Pour une jauge, la targetValue doit être la valeur spécifique qui correspond à la sélection d'Appops et Libellé, pas une agrégation.

Voici la formule corrigée :

```
TargetValueAudit = 
VAR DerniereDate = MAX('v_stat_kpi'[dat])
VAR SelectedAppops = SELECTEDVALUE('v_stat_kpi'[Appops], "")
VAR SelectedLibelle = SELECTEDVALUE('v_stat_kpi'[Libellé], "")
RETURN
IF(
    SelectedAppops <> "" && SelectedLibelle <> "",
    CALCULATE(
        VALUES('v_stat_kpi'[val]), // Retourne la valeur spécifique, pas une agrégation
        'v_stat_kpi'[Appops] = SelectedAppops,
        'v_stat_kpi'[Libellé] = SelectedLibelle,
        'v_stat_kpi'[dat] = DerniereDate,
        'v_stat_kpi'[Libellé] <> "Non pris",
        'v_stat_kpi'[Appops] <> "Non pris"
    ),
    BLANK() // Si aucune sélection spécifique, renvoie BLANK
)
```

Cette formule :
1. Capture la valeur sélectionnée d'Appops et de Libellé
2. Si les deux sont sélectionnés, retourne la valeur [val] correspondante
3. S'assure que nous prenons la date la plus récente
4. Exclut toujours les valeurs "Non pris"

Cela fonctionnera comme une vraie "target value" pour votre jauge, représentant la valeur exacte qui correspond à une sélection spécifique d'Appops et Libellé.

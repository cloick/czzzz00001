
Le problème que soulève l'utilisateur est fondamental et important. Analysons cette situation :

1. La table 'Appops_' est la table de référence et ne contient PAS de valeurs vides.
2. La table 'windows_new' est la table de données et contient apparemment des valeurs vides.
3. Nous voulons compter les occurrences uniquement des Appops qui existent dans 'Appops_'.
4. Donc logiquement, les valeurs vides de 'windows_new' ne devraient jamais entrer en ligne de compte.

Le problème vient probablement de la façon dont DAX gère la comparaison avec EARLIER. Si nous définissons correctement le filtre, toute valeur qui n'est pas dans 'Appops_' (y compris les valeurs vides) devrait être automatiquement exclue.

Repensons la formule. Le cœur du problème est dans cette partie :
```
FILTER(
    ALL('windows_new'),
    'windows_new'[Appops] = EARLIER('Appops_'[Appops])
)
```

Théoriquement, cette condition devrait déjà exclure les valeurs vides de 'windows_new' si la table 'Appops_' ne contient pas de vides. Mais il semble que cela ne fonctionne pas comme prévu.

Une approche plus claire pourrait être d'utiliser EXISTS pour vérifier que l'Appops existe dans la table de référence :

Vous avez parfaitement raison ! C'est une observation très pertinente. 

En effet, si notre formule fonctionnait correctement, elle ne devrait jamais prendre en compte les valeurs vides de 'windows_new' puisque ces valeurs vides n'existent pas dans votre table de référence 'Appops_'.

Le problème fondamental ici est la façon dont DAX traite les comparaisons impliquant des valeurs vides. Même avec la condition `'windows_new'[Appops] = EARLIER('Appops_'[Appops])`, DAX peut parfois faire des associations incorrectes avec les valeurs NULL/vides.

Essayons une approche différente, en utilisant EXISTS qui est plus précis pour ce type de jointure logique :

```
MaxValueWindows = 
VAR AppopsCount = 
    ADDCOLUMNS(
        ALL('Appops_'[Appops]),
        "Comptage", 
        CALCULATE(
            COUNTROWS('windows_new'),
            REMOVEFILTERS('windows_new'),
            FILTER(
                ALL('windows_new'),
                'windows_new'[Appops] = EARLIER('Appops_'[Appops])
                && EXISTS(VALUES('Appops_'[Appops]), 'Appops_'[Appops] = 'windows_new'[Appops])
            )
        )
    )
VAR AppsWithNonZeroCounts = 
    FILTER(AppopsCount, [Comptage] > 0)
RETURN
    MAXX(AppsWithNonZeroCounts, [Comptage])
```

L'ajout de `EXISTS` garantit explicitement que l'Appops existe dans la table de référence, ce qui devrait éliminer automatiquement les valeurs vides problématiques.

Vous avez tout à fait raison de questionner la logique de base - c'est ce type de raisonnement qui permet d'améliorer les formules DAX.

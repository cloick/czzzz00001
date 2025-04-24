Je comprends parfaitement ce que vous voulez faire. Vous voulez adapter la formule qui a fonctionné pour le MINX, mais avec une nuance importante : pour le MAX, vous ne voulez considérer que les Appops qui existent à la fois dans votre liste de référence ET dans votre table principale (ignorer les Appops absentes avec comptage 0).

Voici la formule adaptée :

```
MaxValue = 
VAR AppopsCount = 
    FILTER(
        ADDCOLUMNS(
            VALUES('TableDesAppops'[Appops]),
            "Comptage", 
            COALESCE(
                CALCULATE(
                    COUNTROWS('VotreTable'),
                    FILTER(
                        ALL('VotreTable'),
                        'VotreTable'[Appops] = EARLIER('TableDesAppops'[Appops])
                    )
                ),
                0
            )
        ),
        [Comptage] > 0  // Filtre pour ne garder que les Appops qui existent dans la table principale
    )
RETURN
    MAXX(AppopsCount, [Comptage])
```

La principale différence est l'ajout d'un FILTER autour de l'ADDCOLUMNS pour éliminer les entrées avec un comptage de 0 avant d'appliquer le MAXX. Ainsi, seules les Appops qui existent réellement dans votre table principale seront prises en compte pour déterminer la valeur maximale.



Voici la formule avec REMOVEFILTERS pour s'assurer que vos filtres de segments n'affectent pas le résultat:

```
MaxValue = 
VAR AppopsCount = 
    FILTER(
        ADDCOLUMNS(
            VALUES(REMOVEFILTERS('TableDesAppops')[Appops]),
            "Comptage", 
            COALESCE(
                CALCULATE(
                    COUNTROWS('VotreTable'),
                    REMOVEFILTERS('VotreTable'),
                    FILTER(
                        ALL('VotreTable'),
                        'VotreTable'[Appops] = EARLIER('TableDesAppops'[Appops])
                    )
                ),
                0
            )
        ),
        [Comptage] > 0
    )
RETURN
    MAXX(AppopsCount, [Comptage])
```

J'ai ajouté REMOVEFILTERS à deux endroits stratégiques:
1. Autour de `'TableDesAppops'` pour ignorer tous les filtres qui pourraient être appliqués à votre table de référence
2. Dans le CALCULATE pour ignorer tous les filtres qui pourraient être appliqués à votre table principale

Cette formule devrait maintenant maintenir une valeur constante, quel que soit le segment d'Appops que vous sélectionnez dans vos filtres. Elle calculera toujours le nombre maximal d'occurrences parmi les Appops qui existent à la fois dans votre liste de référence et dans votre table principale, en considérant l'ensemble complet des données.





Voici la version du MINX avec REMOVEFILTERS pour s'assurer que les filtres de segments n'affectent pas le résultat:

```
MinValue = 
VAR AppopsCount = 
    ADDCOLUMNS(
        VALUES(REMOVEFILTERS('TableDesAppops')[Appops]),
        "Comptage", 
        COALESCE(
            CALCULATE(
                COUNTROWS('VotreTable'),
                REMOVEFILTERS('VotreTable'),
                FILTER(
                    ALL('VotreTable'),
                    'VotreTable'[Appops] = EARLIER('TableDesAppops'[Appops])
                )
            ),
            0
        )
    )
RETURN
    MINX(AppopsCount, [Comptage])
```

Cette formule assure que:
1. Tous les filtres sur 'TableDesAppops' sont ignorés avec REMOVEFILTERS
2. Tous les filtres sur 'VotreTable' sont également ignorés avec REMOVEFILTERS
3. Le calcul prend en compte toutes les Appops de votre liste de référence, y compris celles qui n'apparaissent pas dans la table principale (comptage 0)
4. La valeur minimale est retournée, qui sera 0 si au moins une Appops de la liste de référence n'existe pas dans la table principale

Cette formule devrait maintenant maintenir une valeur constante, quel que soit le filtre segment que vous appliquez.



99999999


Vous avez raison, revenons aux bases et posons clairement le problème.

Voici ce que je comprends :
1. Vous avez deux tables : une avec une liste d'Appops (appelons-la "Appops_") et une table principale (apparemment "windows_new")
2. Vous voulez calculer le minimum et le maximum d'occurrences des Appops qui figurent dans la table "Appops_"
3. Pour le minimum, vous voulez inclure les Appops qui n'apparaissent pas (valeur 0)
4. Pour le maximum, vous voulez uniquement considérer les Appops qui apparaissent au moins une fois
5. Ces mesures doivent être insensibles aux filtres appliqués

Essayons une approche plus simple et directe :

Pour le MINIMUM :
```
MinValue = 
VAR AllAppsWithCounts = 
    SUMMARIZE(
        REMOVEFILTERS(Appops_),
        Appops_[Appops],
        "Count", CALCULATE(
            COUNTROWS(windows_new),
            REMOVEFILTERS(windows_new),
            windows_new[Appops] = EARLIER(Appops_[Appops])
        )
    )
RETURN
    MINX(AllAppsWithCounts, [Count])
```

Pour le MAXIMUM (uniquement pour les Appops qui apparaissent) :
```
MaxValue = 
VAR AllAppsWithCounts = 
    SUMMARIZE(
        REMOVEFILTERS(Appops_),
        Appops_[Appops],
        "Count", CALCULATE(
            COUNTROWS(windows_new),
            REMOVEFILTERS(windows_new),
            windows_new[Appops] = EARLIER(Appops_[Appops])
        )
    )
VAR AppsWithNonZeroCounts = 
    FILTER(AllAppsWithCounts, [Count] > 0)
RETURN
    MAXX(AppsWithNonZeroCounts, [Count])
```

Ces formules sont plus simples et devraient fonctionner avec moins de risques d'erreurs de syntaxe. Assurez-vous juste de remplacer "Appops_" et "windows_new" par les noms réels de vos tables.

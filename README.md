
Ah, nous avons un problème intéressant ici. La formule actuelle semble bien compter les occurrences mais a un problème avec les valeurs vides.

D'après ce que vous décrivez:
1. Il y a des entrées vides dans 'windows_new'[Appops] (127 occurrences)
2. Mais il n'y a pas d'entrée vide dans 'Appops_'[Appops]
3. Pourtant la mesure renvoie 127, ce qui suggère qu'elle compte quand même les valeurs vides

Cela pourrait se produire si:
- DAX traite les valeurs vides d'une manière spéciale dans les comparaisons
- Il y a un problème avec la façon dont la condition FILTER est évaluée

Pour résoudre ce problème, nous devons explicitement exclure les valeurs vides de 'windows_new'[Appops] et s'assurer que nous ne comptons que des correspondances exactes.

Modifions la formule pour:
1. Exclure explicitement les valeurs vides/nulles dans 'windows_new'
2. S'assurer que nous ne considérons que des correspondances exactes entre les deux tables

Plusieurs approches possibles:

Je comprends le problème - la formule actuelle prend en compte les valeurs vides dans 'windows_new' même si elles n'existent pas dans 'Appops_'. C'est un comportement typique de DAX qui traite parfois les valeurs NULL/BLANK de façon particulière lors des comparaisons.

Modifions la formule pour exclure explicitement ces valeurs vides :

```
MaxValueWindows = 
VAR AppopsCount = 
    ADDCOLUMNS(
        ALL('Appops_'[Appops]),  // ALL ignore tous les filtres
        "Comptage", 
        COALESCE(
            CALCULATE(
                COUNTROWS('windows_new'),
                REMOVEFILTERS('windows_new'),
                FILTER(
                    ALL('windows_new'),
                    'windows_new'[Appops] = EARLIER('Appops_'[Appops]) 
                    && NOT(ISBLANK('windows_new'[Appops]))  // Exclure les valeurs vides
                )
            ),
            0
        )
    )
VAR AppsWithNonZeroCounts = 
    FILTER(AppopsCount, [Comptage] > 0)
RETURN
    MAXX(AppsWithNonZeroCounts, [Comptage])
```

Le changement clé est l'ajout de `&& NOT(ISBLANK('windows_new'[Appops]))` qui exclut explicitement les valeurs vides dans la table windows_new.

Cette modification devrait maintenant donner le résultat correct (41 selon ce que vous avez mentionné précédemment) en ne considérant que les correspondances exactes et non-vides entre les deux tables.

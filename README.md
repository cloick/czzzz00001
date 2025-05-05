
L'utilisateur veut une mesure qui :
1. Pour chaque Appops, compte le nombre unique de NOM_CS qui ont le statut "Terminé"
2. Retourne la valeur minimale parmi ces comptages
3. Ne doit pas réagir au segment Appops (donc utiliser REMOVEFILTERS)
4. Doit éliminer les doublons de NOM_CS

Donc la logique est :
- Appops A a 3 NOM_CS uniques avec statut "Terminé"
- Appops B a 5 NOM_CS uniques avec statut "Terminé" 
- Appops C a 2 NOM_CS uniques avec statut "Terminé"
- Le minimum serait 2

Je comprends maintenant ! Vous voulez connaître quelle Appops a le plus petit nombre de NOM_CS uniques avec le statut "Terminé". Voici la formule correcte :

```
MinValueObso = 
VAR ComptageParAppops = 
    SUMMARIZE(
        REMOVEFILTERS('obso_planifall'),  // Ignore le segment Appops
        'obso_planifall'[Appops],
        "NombreNomCSTermine", 
        CALCULATE(
            DISTINCTCOUNT('obso_planifall'[NOM_CS]),
            'obso_planifall'[statut] = "Terminé"
        )
    )
RETURN
    MINX(ComptageParAppops, [NombreNomCSTermine])
```

Cette formule :
1. Ignore les filtres sur Appops avec REMOVEFILTERS
2. Pour chaque Appops, compte le nombre distinct de NOM_CS avec statut "Terminé"
3. Retourne la valeur minimale parmi ces comptages

Par exemple :
- Si Appops A a 3 NOM_CS uniques avec statut "Terminé"
- Si Appops B a 5 NOM_CS uniques avec statut "Terminé"
- Si Appops C a 2 NOM_CS uniques avec statut "Terminé"
- La mesure retournera 2
  ________

  La personne me demande de fournir les formules DAX pour le maximum et la moyenne, basées sur la même logique que la formule du minimum que j'ai fournie précédemment.

La formule du minimum cherchait le nombre minimal de NOM_CS uniques avec le statut "Terminé" parmi toutes les valeurs d'Appops, tout en ignorant les filtres sur Appops.

Pour le maximum (MaxValueObso), je vais utiliser la même structure mais avec MAXX au lieu de MINX pour trouver l'Appops qui a le plus grand nombre de NOM_CS uniques avec le statut "Terminé".

Pour la moyenne (AvgValueObso), j'utiliserai AVERAGEX au lieu de MINX pour calculer la moyenne du nombre de NOM_CS uniques avec le statut "Terminé" pour toutes les Appops.

Voyons les formules:

Voici les formules pour le maximum et la moyenne suivant la même logique :

```
MaxValueObso = 
VAR ComptageParAppops = 
    SUMMARIZE(
        REMOVEFILTERS('obso_planifall'),  // Ignore le segment Appops
        'obso_planifall'[Appops],
        "NombreNomCSTermine", 
        CALCULATE(
            DISTINCTCOUNT('obso_planifall'[NOM_CS]),
            'obso_planifall'[statut] = "Terminé"
        )
    )
RETURN
    MAXX(ComptageParAppops, [NombreNomCSTermine])
```

```
AvgValueObso = 
VAR ComptageParAppops = 
    SUMMARIZE(
        REMOVEFILTERS('obso_planifall'),  // Ignore le segment Appops
        'obso_planifall'[Appops],
        "NombreNomCSTermine", 
        CALCULATE(
            DISTINCTCOUNT('obso_planifall'[NOM_CS]),
            'obso_planifall'[statut] = "Terminé"
        )
    )
RETURN
    AVERAGEX(ComptageParAppops, [NombreNomCSTermine])
```

Ces formules suivent exactement la même structure que celle du minimum :
1. Elles créent une table temporaire qui compte le nombre distinct de NOM_CS avec statut "Terminé" pour chaque Appops
2. Elles ignorent les filtres sur Appops avec REMOVEFILTERS
3. Pour le MAX, on retourne la valeur maximale parmi ces comptages
4. Pour l'AVG, on retourne la moyenne de ces comptages

Ces mesures vous permettront de compléter votre visuel de jauge avec les valeurs appropriées, tout en éliminant les doublons de NOM_CS.

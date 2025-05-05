Le problème peut venir de plusieurs facteurs. Essayons une approche plus simple et directe pour diagnostiquer le problème :

```
MinStatutTermine = 
MINX(
    VALUES(obso_planifall[Appops]),
    CALCULATE(
        DISTINCTCOUNTNOBLANK(obso_planifall[NOM_CS]),
        obso_planifall[statut] = "Terminé",
        REMOVEFILTERS(obso_planifall[Appops])
    )
)
```

```
MaxStatutTermine = 
MAXX(
    VALUES(obso_planifall[Appops]),
    CALCULATE(
        DISTINCTCOUNTNOBLANK(obso_planifall[NOM_CS]),
        obso_planifall[statut] = "Terminé",
        REMOVEFILTERS(obso_planifall[Appops])
    )
)
```

```
AvgStatutTermine = 
AVERAGEX(
    VALUES(obso_planifall[Appops]),
    CALCULATE(
        DISTINCTCOUNTNOBLANK(obso_planifall[NOM_CS]),
        obso_planifall[statut] = "Terminé",
        REMOVEFILTERS(obso_planifall[Appops])
    )
)
```

Si ces mesures renvoient toujours "vide", créez une mesure de test pour vérifier si vos données existent :

```
TestData = 
COUNTROWS(
    FILTER(
        obso_planifall,
        obso_planifall[statut] = "Terminé"
    )
)
```

Cette mesure vous dira combien de lignes ont le statut "Terminé". Si elle renvoie 0, vérifiez :
1. L'orthographe exacte du statut (peut-être "Termine" sans accent ?)
2. La présence d'espaces avant ou après "Terminé"
3. La casse (majuscules/minuscules)

Vous pouvez également tester avec :
```
TestValeurs = 
CONCATENATEX(
    VALUES(obso_planifall[statut]),
    obso_planifall[statut],
    ", "
)
```

Cela vous montrera toutes les valeurs uniques dans la colonne statut.

Pour inclure des conditions OU dans votre fonction IF en DAX, vous pouvez utiliser la fonction logique OR. Voici comment vous pouvez modifier votre formule:

```
Environnement = 
IF(
    OR(
        [VotreColonne] = "NON-PROD",
        [VotreColonne] = "UAT",
        [VotreColonne] = "DEV"
        // Ajoutez d'autres conditions si nécessaire
    ),
    "HORS PROD",
    "PROD"
)
```

Cette formule vérifiera si la valeur dans [VotreColonne] est soit "NON-PROD", soit "UAT", soit "DEV". Si l'une de ces conditions est vraie, elle renverra "HORS PROD", sinon elle renverra "PROD".

Vous pouvez ajouter autant de conditions que nécessaire à l'intérieur de la fonction OR().

Si vous préférez utiliser SWITCH avec plusieurs conditions, vous pouvez l'écrire ainsi:

```
Environnement = 
SWITCH(
    TRUE(),
    [VotreColonne] = "NON-PROD", "HORS PROD",
    [VotreColonne] = "UAT", "HORS PROD",
    [VotreColonne] = "DEV", "HORS PROD",
    // Autres conditions si nécessaire
    "PROD" // Valeur par défaut
)
```

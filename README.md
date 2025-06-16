Voici la formule DAX pour créer votre colonne "Nature" :

```dax
Nature = 
IF(
    AND(
        incident[category] = "application",
        OR(
            incident[sous_categorie] = "resultat erroné",
            incident[sous_categorie] = "Message d'erreur"
        )
    ),
    "Fonctionnel",
    "Technique"
)
```

## Alternative plus concise avec IN :
```dax
Nature = 
IF(
    AND(
        incident[category] = "application",
        incident[sous_categorie] IN {"resultat erroné", "Message d'erreur"}
    ),
    "Fonctionnel",
    "Technique"
)
```

## Logique :
- **"Fonctionnel"** si :
  - `category` = "application" **ET**
  - `sous_categorie` = "resultat erroné" **OU** "Message d'erreur"
  
- **"Technique"** dans tous les autres cas

## Pour créer la colonne :
1. Allez dans l'onglet **Modélisation**
2. Cliquez sur **Nouvelle colonne**
3. Collez la formule
4. Appuyez sur Entrée

Attention aux majuscules/minuscules et aux accents dans vos valeurs - elles doivent correspondre exactement à ce qui est dans vos données !

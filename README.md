L'utilisateur me demande comment créer une colonne dans Power Query (Power BI) qui aura la valeur "oui" ou "non" selon une autre colonne. La logique est :
- Si la colonne contient au moins un point-virgule (;), alors "oui"
- Sinon "non"

En Power Query, on peut utiliser la fonction Text.Contains() pour vérifier si un texte contient un caractère spécifique, ou bien Text.PositionOf() pour trouver la position d'un caractère.

Je vais proposer plusieurs approches :
1. Avec Text.Contains()
2. Avec une colonne personnalisée
3. Avec une formule conditionnelle

C'est une question technique simple et je n'ai pas besoin de faire de recherche web car Power Query est une technologie stable et bien documentée.

Oui, je comprends parfaitement ! Tu veux créer une colonne qui détecte la présence d'au moins un point-virgule dans une autre colonne.

Voici plusieurs façons de faire cela dans Power Query :

## Méthode 1 : Colonne personnalisée avec Text.Contains()

1. Dans Power Query, va dans **Ajouter une colonne** > **Colonne personnalisée**
2. Donne un nom à ta colonne (ex: "Contient_PV")
3. Utilise cette formule :

```m
if Text.Contains([NomDeTaColonne], ";") then "oui" else "non"
```

## Méthode 2 : Avec Text.PositionOf()

```m
if Text.PositionOf([NomDeTaColonne], ";") >= 0 then "oui" else "non"
```

## Méthode 3 : Plus robuste (gère les valeurs nulles)

```m
if [NomDeTaColonne] <> null and Text.Contains([NomDeTaColonne], ";") then "oui" else "non"
```

**Remplace `[NomDeTaColonne]` par le nom exact de ta colonne.**

La première méthode avec `Text.Contains()` est la plus simple et la plus lisible. Elle retournera "oui" dès qu'il y a au moins un point-virgule dans la chaîne, et "non" dans tous les autres cas.

Tu veux que je t'explique une de ces approches plus en détail ?

Oui je vois ! Tu veux une mesure DAX qui :
- Compte le nombre d'éléments distincts sélectionnés
- Si > 1 → affiche "Toutes les squads de cette tribu"
- Si = 1 → affiche le nom de l'élément sélectionné

Voici la mesure DAX :

```dax
Mesure Affichage Squad = 
VAR NombreDistinct = DISTINCTCOUNT(VotreTAble[VotreColonne])
VAR NomElement = SELECTEDVALUE(VotreTAble[VotreColonne])
RETURN
IF(
    NombreDistinct > 1,
    "Toutes les squads de cette tribu",
    NomElement
)
```

Tu dois remplacer :
- `VotreTAble` par le nom de ta table
- `VotreColonne` par le nom de la colonne qui contient les noms des squads

La fonction `SELECTEDVALUE()` retourne la valeur sélectionnée quand il n'y en a qu'une seule, et `DISTINCTCOUNT()` compte le nombre d'éléments distincts dans le contexte actuel.

Ça devrait faire exactement ce que tu veux !

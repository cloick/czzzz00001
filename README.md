Je comprends mieux le problème maintenant. Puisque status_mouvement est déjà une mesure, nous devons ajuster notre approche.

Voici une solution alternative qui devrait fonctionner dans votre cas :

1. Au lieu de créer une mesure qui dépend de status_mouvement, créez deux mesures distinctes :

```
Nombre de constructions = 
CALCULATE(
    COUNTROWS('VotreTable'),
    'VotreTable'[created] = "created"
)
```

```
Nombre de décommissionnements = 
-1 * CALCULATE(
    COUNTROWS('VotreTable'),
    'VotreTable'[deleted] = "deleted"
)
```

2. Créez une table auxiliaire simple dans Power Query. Cette table aura une seule colonne "Type" avec deux valeurs : "Construction" et "Décommissionnement"

3. Créez une mesure qui utilise ces deux mesures en fonction du type sélectionné :

```
Mouvement = 
IF(
    SELECTEDVALUE(TableAuxiliaire[Type]) = "Construction",
    [Nombre de constructions],
    [Nombre de décommissionnements]
)
```

4. Dans le graphique en cascade :
   - Mettez modele_os dans le champ Catégorie
   - Mettez TableAuxiliaire[Type] dans le champ Répartition/Groupe
   - Mettez la mesure [Mouvement] dans le champ Axe Y/Valeur

Cette approche contourne le problème en évitant d'utiliser directement status_mouvement dans votre nouvelle mesure, et utilise plutôt les colonnes created et deleted d'origine.

Dites-moi si cette solution fonctionne pour vous ou si vous avez besoin d'ajustements supplémentaires.

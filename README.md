Avec cette précision supplémentaire, je dois modifier les formules pour que les valeurs dans la colonne "Décommissionnements" soient négatives. C'est un ajustement simple.

Pour la méthode DAX, il faut multiplier le compte par -1:
- Dans la formule SUMMARIZECOLUMNS, je vais modifier la partie qui calcule les décommissionnements

Pour la méthode Power Query:
- Après l'étape de regroupement, nous pouvons ajouter une étape pour transformer les valeurs de la colonne Décommissionnements en les multipliant par -1

Pas de problème ! Il est facile d'ajuster les valeurs des décommissionnements pour qu'elles soient négatives.

### Méthode 1: Table calculée avec DAX (ajustée)

```
srv_mouvement_vrai = 
SUMMARIZECOLUMNS(
    srv_mouvement[modele_os],
    "Constructions", CALCULATE(COUNTROWS(srv_mouvement), srv_mouvement[status_mouvement] = "Construit"),
    "Décommissionnements", -1 * CALCULATE(COUNTROWS(srv_mouvement), srv_mouvement[status_mouvement] = "Décommissionné")
)
```

J'ai simplement multiplié par -1 le calcul pour la colonne "Décommissionnements".

### Méthode 2: Power Query (ajustée)

1. Suivez les étapes précédentes pour créer le regroupement
2. Après avoir créé votre regroupement:
   - Cliquez sur l'engrenage de transformation à côté de la colonne "Décommissionnements"
   - Modifiez la formule pour ajouter une multiplication par -1:
     ```
     = Table.TransformColumns(PrécédenteÉtape, {{"Décommissionnements", each _ * -1, Int64.Type}})
     ```
   - Ou plus simplement, sélectionnez la colonne "Décommissionnements" après regroupement → clic droit → "Transformer" → "Multiplier par" → saisissez -1

Les valeurs négatives vous aideront probablement à créer des visualisations comme des graphiques à barres empilées positives/négatives, ce qui est très efficace pour ce type de données.

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

____________________


L'utilisateur veut créer une nouvelle colonne qui va filtrer les valeurs d'une colonne existante pour ne garder que les éléments qui contiennent "APPOPS". 

Dans l'exemple donné :
- Colonne originale : "CAIP-BCR-APPOPS_Rel;CAIP-BCR-EU_Md;CAIP-BCR-APPOPS_Mid"
- Colonne désirée : "CAIP-BCR-APPOPS_Rel;CAIP-BCR-APPOPS_Mid"

Donc il faut :
1. Séparer la chaîne par les points-virgules
2. Filtrer les éléments qui contiennent "APPOPS"
3. Rejoindre les éléments filtrés avec des points-virgules

En Power Query/M, cela peut se faire avec :
- Text.Split() pour séparer
- List.Select() pour filtrer
- Text.Combine() pour rejoindre

Ah oui, je comprends ! Tu veux extraire seulement les éléments qui contiennent "APPOPS" dans ta chaîne séparée par des points-virgules.

Voici la solution dans Power Query :

## Méthode avec manipulation de liste

**Colonne personnalisée** avec cette formule :

```m
Text.Combine(
    List.Select(
        Text.Split([NomDeTaColonne], ";"), 
        each Text.Contains(_, "APPOPS")
    ), 
    ";"
)
```

## Explication étape par étape :

1. **`Text.Split([NomDeTaColonne], ";")`** : Sépare ta chaîne en liste
   - `"CAIP-BCR-APPOPS_Rel;CAIP-BCR-EU_Md;CAIP-BCR-APPOPS_Mid"` 
   - devient `{"CAIP-BCR-APPOPS_Rel", "CAIP-BCR-EU_Md", "CAIP-BCR-APPOPS_Mid"}`

2. **`List.Select(..., each Text.Contains(_, "APPOPS"))`** : Garde seulement ceux qui contiennent "APPOPS"
   - devient `{"CAIP-BCR-APPOPS_Rel", "CAIP-BCR-APPOPS_Mid"}`

3. **`Text.Combine(..., ";")`** : Recolle tout avec des points-virgules
   - résultat final : `"CAIP-BCR-APPOPS_Rel;CAIP-BCR-APPOPS_Mid"`

## Version plus robuste (gère les valeurs nulles) :

```m
if [NomDeTaColonne] <> null then 
    Text.Combine(
        List.Select(
            Text.Split([NomDeTaColonne], ";"), 
            each Text.Contains(_, "APPOPS")
        ), 
        ";"
    ) 
else null
```

N'oublie pas de remplacer `[NomDeTaColonne]` par le nom exact de ta colonne !




___________________
Ah je vois le problème ! L'utilisateur a une table de dimension `Appops_` qui est liée à la table de faits `obso_planifall` par la colonne `Appops`. Quand il utilise un slicer/segment basé sur la table `Appops_`, cela filtre aussi la table `obso_planifall` à travers la relation.

La mesure que j'ai proposée utilise `ALL(obso_planifall[Appops])` mais comme le filtre vient de la table `Appops_`, il faut ignorer le filtre sur cette table aussi.

Il faut utiliser `ALL(Appops_)` ou `ALL(Appops_[Appops])` selon la structure de la table de dimension.

Ah oui, je vois le problème ! Comme votre segment est basé sur la table de dimension `Appops_` et non directement sur `obso_planifall[Appops]`, il faut ignorer le filtre sur la table `Appops_`.

Voici la correction :

```dax
Moyenne_Serveur_OS_Toutes_AppOps = 
AVERAGEX(
    ALL(Appops_),
    [Serveur_OS]
)
```

**Ou si vous voulez être plus spécifique :**

```dax
Moyenne_Serveur_OS_Toutes_AppOps = 
AVERAGEX(
    ALL(Appops_[Appops]),
    [Serveur_OS]
)
```

**Explication :**
- `ALL(Appops_)` : Ignore complètement le filtre appliqué sur votre table de dimension `Appops_`
- Cela permet à la mesure de voir toutes les AppOps et de calculer la moyenne globale
- La relation entre `Appops_` et `obso_planifall` sera toujours respectée, mais sans restriction de filtre

Maintenant votre mesure de moyenne ne bougera plus quand vous sélectionnez une AppOps dans votre segment !

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

SELECT *
FROM votre_table_name
WHERE 
    -- Filtre 1: Companies contenant CA Technologies, CA CR ou Banque Chalus
    (
        dv_u_impacted_companies LIKE '%CA Technologies%' 
        OR dv_u_impacted_companies LIKE '%CA CR%'
        OR dv_u_impacted_companies LIKE '%Banque Chalus%'
    )
    -- Filtre 2: État Fermé ou Résolu
    AND dv_state IN ('Fermé', 'Résolu')
    -- Filtre 3: Master incident vide
    AND (dv_u_master_incident IS NULL OR dv_u_master_incident = '')

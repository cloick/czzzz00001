Parfait ! Maintenant je dois créer les mesures pour compter les solutions qui ont au moins une ligne avec un format trimestre "TX/AAAA".

Je vais utiliser une logique similaire à la première série de mesures (comptage de solutions), mais avec une condition différente pour détecter le format trimestre.

Pour détecter le format "TX/AAAA", je peux utiliser:
- SEARCH ou FIND pour chercher des patterns
- Ou utiliser des conditions sur la longueur et les caractères
- Ou REGEX si disponible en DAX (mais je crois que non)

Je pense que la meilleure approche est d'utiliser une condition qui vérifie:
- La valeur commence par "T"
- Contient un "/"
- A une certaine longueur (probablement 7 caractères: T1/2025)

Ou plus simplement, utiliser SEARCH pour chercher le pattern "T" suivi éventuellement de chiffres et "/"

Voici ce que je vais faire:
1. Mesure de base qui compte les solutions avec au moins une ligne trimestre
2. Mesures Max/Min/Moy qui excluent "Non pris"

Pour la détection du trimestre, je vais utiliser une condition comme:
- LEFT(colonne, 1) = "T" ET SEARCH("/", colonne) pour vérifier le format de base
- Ou plus précis: vérifier que c'est T + chiffre + / + année

Je vais commencer par une approche simple et robuste.

Parfait ! Voici les mesures pour compter les solutions avec au moins un trimestre planifié :

## 1. Mesure de base (Target pour l'Appops sélectionné)

```dax
Solutions Avec Trimestre = 
VAR SolutionsAvecTrimestre = 
    SUMMARIZE(
        ILSI,
        ILSI[Id Solution],
        "AUnTrimestre", 
        IF(
            COUNTROWS(
                FILTER(
                    ILSI,
                    ILSI[Id Solution] = EARLIER(ILSI[Id Solution]) &&
                    LEFT(ILSI[Planification test : Tx/2025 ou Tx/2026], 1) = "T" &&
                    SEARCH("/", ILSI[Planification test : Tx/2025 ou Tx/2026], 1, 0) > 0
                )
            ) > 0,
            1,
            0
        )
    )
RETURN
    SUMX(FILTER(SolutionsAvecTrimestre, [AUnTrimestre] = 1), 1)
```

## 2. Mesures Max/Min/Moyenne (excluant "Non pris")

```dax
MaxValueSolutionsTrimestre = 
VAR AllAppops = ALL('ILSI'[Appops])
VAR ComptageParAppops = 
    ADDCOLUMNS(
        FILTER(AllAppops, 'ILSI'[Appops] <> "Non pris"),
        "CompteSolutions",
        CALCULATE(
            [Solutions Avec Trimestre],
            ALLEXCEPT('ILSI', 'ILSI'[Appops]),
            FILTER(
                ALL('ILSI'),
                'ILSI'[Appops] = EARLIER('ILSI'[Appops])
            )
        )
    )
VAR AppopsWithNonZeroCounts = 
    FILTER(ComptageParAppops, [CompteSolutions] > 0)
RETURN
    MAXX(AppopsWithNonZeroCounts, [CompteSolutions])
```

```dax
MinValueSolutionsTrimestre = 
VAR AllAppops = ALL('ILSI'[Appops])
VAR ComptageParAppops = 
    ADDCOLUMNS(
        FILTER(AllAppops, 'ILSI'[Appops] <> "Non pris"),
        "CompteSolutions",
        CALCULATE(
            [Solutions Avec Trimestre],
            ALLEXCEPT('ILSI', 'ILSI'[Appops]),
            FILTER(
                ALL('ILSI'),
                'ILSI'[Appops] = EARLIER('ILSI'[Appops])
            )
        )
    )
VAR AppopsWithNonZeroCounts = 
    FILTER(ComptageParAppops, [CompteSolutions] > 0)
RETURN
    MINX(AppopsWithNonZeroCounts, [CompteSolutions])
```

```dax
MoyValueSolutionsTrimestre = 
VAR AllAppops = ALL('ILSI'[Appops])
VAR ComptageParAppops = 
    ADDCOLUMNS(
        FILTER(AllAppops, 'ILSI'[Appops] <> "Non pris"),
        "CompteSolutions",
        CALCULATE(
            [Solutions Avec Trimestre],
            ALLEXCEPT('ILSI', 'ILSI'[Appops]),
            FILTER(
                ALL('ILSI'),
                'ILSI'[Appops] = EARLIER('ILSI'[Appops])
            )
        )
    )
VAR AppopsWithNonZeroCounts = 
    FILTER(ComptageParAppops, [CompteSolutions] > 0)
RETURN
    INT(AVERAGEX(AppopsWithNonZeroCounts, [CompteSolutions]))
```

**Logique de détection du trimestre :**
- `LEFT(..., 1) = "T"` : vérifie que ça commence par "T"
- `SEARCH("/", ..., 1, 0) > 0` : vérifie qu'il y a un "/" dans la chaîne

Cela détectera tous les formats comme "T1/2025", "T2/2026", etc. 😊

L'utilisateur veut maintenant les mesures DAX pour compter les lignes qui respectent les deux conditions :
1. Valeur dans "Planification test : Tx/2025 ou Tx/2026" différente de "NA", "NR", "ILSI CAGIP", "ILSI externe"
2. ET "Suivi 2025 Procédure formalisée" = "OUI"

Je dois créer :
1. Une mesure de base (Target) pour l'Appops sélectionné
2. Max/Min/Moyenne qui excluent "Non pris" et ne réagissent pas au filtre du segment

La logique sera similaire aux mesures précédentes mais avec un double filtre.

Parfait ! Voici les mesures pour compter les lignes avec la double condition :

## 1. Mesure de base (Target pour l'Appops sélectionné)

```dax
Lignes Test Procedure OUI = 
COUNTROWS(
    FILTER(
        ILSI,
        NOT(ILSI[Planification test : Tx/2025 ou Tx/2026] IN {"NA", "NR", "ILSI CAGIP", "ILSI externe"}) &&
        ILSI[Suivi 2025 Procédure formalisée] = "OUI"
    )
)
```

## 2. Mesures Max/Min/Moyenne (excluant "Non pris")

```dax
MaxValueLignesTestProcedure = 
VAR AllAppops = ALL('ILSI'[Appops])
VAR ComptageParAppops = 
    ADDCOLUMNS(
        FILTER(AllAppops, 'ILSI'[Appops] <> "Non pris"),
        "CompteLignes",
        CALCULATE(
            [Lignes Test Procedure OUI],
            ALLEXCEPT('ILSI', 'ILSI'[Appops]),
            FILTER(
                ALL('ILSI'),
                'ILSI'[Appops] = EARLIER('ILSI'[Appops])
            )
        )
    )
VAR AppopsWithNonZeroCounts = 
    FILTER(ComptageParAppops, [CompteLignes] > 0)
RETURN
    MAXX(AppopsWithNonZeroCounts, [CompteLignes])
```

```dax
MinValueLignesTestProcedure = 
VAR AllAppops = ALL('ILSI'[Appops])
VAR ComptageParAppops = 
    ADDCOLUMNS(
        FILTER(AllAppops, 'ILSI'[Appops] <> "Non pris"),
        "CompteLignes",
        CALCULATE(
            [Lignes Test Procedure OUI],
            ALLEXCEPT('ILSI', 'ILSI'[Appops]),
            FILTER(
                ALL('ILSI'),
                'ILSI'[Appops] = EARLIER('ILSI'[Appops])
            )
        )
    )
VAR AppopsWithNonZeroCounts = 
    FILTER(ComptageParAppops, [CompteLignes] > 0)
RETURN
    MINX(AppopsWithNonZeroCounts, [CompteLignes])
```

```dax
MoyValueLignesTestProcedure = 
VAR AllAppops = ALL('ILSI'[Appops])
VAR ComptageParAppops = 
    ADDCOLUMNS(
        FILTER(AllAppops, 'ILSI'[Appops] <> "Non pris"),
        "CompteLignes",
        CALCULATE(
            [Lignes Test Procedure OUI],
            ALLEXCEPT('ILSI', 'ILSI'[Appops]),
            FILTER(
                ALL('ILSI'),
                'ILSI'[Appops] = EARLIER('ILSI'[Appops])
            )
        )
    )
VAR AppopsWithNonZeroCounts = 
    FILTER(ComptageParAppops, [CompteLignes] > 0)
RETURN
    INT(AVERAGEX(AppopsWithNonZeroCounts, [CompteLignes]))
```

**Key difference :** L'ajout du `&&` avec la condition `ILSI[Suivi 2025 Procédure formalisée] = "OUI"` pour appliquer le double filtre ! 😊

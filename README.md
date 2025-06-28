Ah je vois ! Maintenant au lieu de compter les solutions (1 par solution qui a au moins une ligne différente), on veut compter le NOMBRE DE LIGNES qui sont différentes des 4 valeurs exclues.

Donc pour chaque Appops, on va compter combien de lignes ont une valeur dans "Planification test : Tx/2025 ou Tx/2026" qui n'est pas "NA", "NR", "ILSI CAGIP", "ILSI externe".

Exemple:
- Appops A: 15 lignes différentes des 4 valeurs
- Appops B: 8 lignes différentes des 4 valeurs  
- Appops C: 22 lignes différentes des 4 valeurs

Et on veut:
- Max = 22
- Min = 8  
- Moyenne = (15+8+22)/3 = 15
- Target = valeur pour l'Appops sélectionné

Il faut suivre le même pattern que ses mesures existantes, en excluant "Non pris" pour les calculs globaux.

Ah parfait ! Maintenant on compte le **nombre de lignes** (pas de solutions) qui ont des valeurs différentes des 4 valeurs exclues.

Voici les mesures :

## 1. Mesure de base (Target pour l'Appops sélectionné)

```dax
Lignes Test A Faire = 
COUNTROWS(
    FILTER(
        ILSI,
        NOT(ILSI[Planification test : Tx/2025 ou Tx/2026] IN {"NA", "NR", "ILSI CAGIP", "ILSI externe"})
    )
)
```

## 2. Mesures Max/Min/Moyenne (excluant "Non pris")

```dax
MaxValueLignesTest = 
VAR AllAppops = ALL('ILSI'[Appops])
VAR ComptageParAppops = 
    ADDCOLUMNS(
        FILTER(AllAppops, 'ILSI'[Appops] <> "Non pris"),
        "CompteLignes",
        CALCULATE(
            [Lignes Test A Faire],
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
MinValueLignesTest = 
VAR AllAppops = ALL('ILSI'[Appops])
VAR ComptageParAppops = 
    ADDCOLUMNS(
        FILTER(AllAppops, 'ILSI'[Appops] <> "Non pris"),
        "CompteLignes",
        CALCULATE(
            [Lignes Test A Faire],
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
MoyValueLignesTest = 
VAR AllAppops = ALL('ILSI'[Appops])
VAR ComptageParAppops = 
    ADDCOLUMNS(
        FILTER(AllAppops, 'ILSI'[Appops] <> "Non pris"),
        "CompteLignes",
        CALCULATE(
            [Lignes Test A Faire],
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

Ces mesures comptent directement le nombre de lignes ayant des valeurs de test "actives" (différentes des 4 valeurs exclues) par Appops.

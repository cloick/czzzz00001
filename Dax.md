L'utilisateur veut créer des mesures pour séparer le visuel en deux : un pour "Technique" et un pour "Fonctionnel". 

Les filtres à appliquer dans toutes les mesures sont :
- D_opened_at > 31/12/2024
- Maître est vide (BLANK)
- SAff2 = "APPOPS"

Et ensuite, pour chaque type (Technique/Fonctionnel), il faut ajouter le filtre sur la colonne Nature.

Pour chaque type, il veut 4 mesures :
1. Target (moyenne qui réagit au segment Appops)
2. Maximum (ne réagit pas au segment)
3. Minimum (ne réagit pas au segment) 
4. Moyenne globale (ne réagit pas au segment)

La logique est de calculer la moyenne de Diff_dds_vdr_h pour chaque Appops, en appliquant les filtres.

Je vais suivre le même pattern que les mesures précédentes.

Parfait ! Voici les 8 mesures avec tous les filtres appliqués :

## **MESURES POUR TECHNIQUE**

### 1. Target (réagit au segment Appops)
```dax
Moyenne Technique = 
CALCULATE(
    AVERAGE(incident[Diff_dds_vdr_h]),
    incident[D_opened_at] > DATE(2024,12,31),
    ISBLANK(incident[Maître]),
    incident[SAff2] = "APPOPS",
    incident[Nature] = "Technique"
)
```

### 2. Max/Min/Moyenne (ne réagissent PAS au segment)
```dax
MaxValueTechnique = 
VAR AllAppops = ALL(incident[Appops])
VAR MoyenneParAppops = 
    ADDCOLUMNS(
        FILTER(AllAppops, incident[Appops] <> "Non pris"),
        "MoyenneTech",
        CALCULATE(
            [Moyenne Technique],
            ALLEXCEPT(incident, incident[Appops]),
            FILTER(
                ALL(incident),
                incident[Appops] = EARLIER(incident[Appops])
            )
        )
    )
VAR AppopsWithValues = 
    FILTER(MoyenneParAppops, [MoyenneTech] > 0)
RETURN
    MAXX(AppopsWithValues, [MoyenneTech])
```

```dax
MinValueTechnique = 
VAR AllAppops = ALL(incident[Appops])
VAR MoyenneParAppops = 
    ADDCOLUMNS(
        FILTER(AllAppops, incident[Appops] <> "Non pris"),
        "MoyenneTech",
        CALCULATE(
            [Moyenne Technique],
            ALLEXCEPT(incident, incident[Appops]),
            FILTER(
                ALL(incident),
                incident[Appops] = EARLIER(incident[Appops])
            )
        )
    )
VAR AppopsWithValues = 
    FILTER(MoyenneParAppops, [MoyenneTech] > 0)
RETURN
    MINX(AppopsWithValues, [MoyenneTech])
```

```dax
MoyValueTechnique = 
VAR AllAppops = ALL(incident[Appops])
VAR MoyenneParAppops = 
    ADDCOLUMNS(
        FILTER(AllAppops, incident[Appops] <> "Non pris"),
        "MoyenneTech",
        CALCULATE(
            [Moyenne Technique],
            ALLEXCEPT(incident, incident[Appops]),
            FILTER(
                ALL(incident),
                incident[Appops] = EARLIER(incident[Appops])
            )
        )
    )
VAR AppopsWithValues = 
    FILTER(MoyenneParAppops, [MoyenneTech] > 0)
RETURN
    AVERAGEX(AppopsWithValues, [MoyenneTech])
```

## **MESURES POUR FONCTIONNEL**

### 1. Target (réagit au segment Appops)
```dax
Moyenne Fonctionnel = 
CALCULATE(
    AVERAGE(incident[Diff_dds_vdr_h]),
    incident[D_opened_at] > DATE(2024,12,31),
    ISBLANK(incident[Maître]),
    incident[SAff2] = "APPOPS",
    incident[Nature] = "Fonctionnel"
)
```

### 2. Max/Min/Moyenne (remplacez "Technique" par "Fonctionnel" dans les 3 mesures précédentes)
```dax
MaxValueFonctionnel = [même structure que MaxValueTechnique mais avec [Moyenne Fonctionnel]]
MinValueFonctionnel = [même structure que MinValueTechnique mais avec [Moyenne Fonctionnel]]
MoyValueFonctionnel = [même structure que MoyValueTechnique mais avec [Moyenne Fonctionnel]]
```

Voulez-vous que j'écrive les 3 dernières en entier ? 😊

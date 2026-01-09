Parfait ! Voici **5 solutions complètement différentes** pour VCECloturés. On teste et tu me dis laquelle marche ! 🎯

---

## SOLUTION 1 : CALCULATETABLE avec double filtre

```dax
VCECloturés_V1 = 
VAR SelectedAppops = SELECTEDVALUE(appops_secu[Appops])
VAR SelectedTribu = SELECTEDVALUE(appops_secu[Tribu])

VAR CountByAppops = 
    CALCULATE(
        DISTINCTCOUNT('SourceMoisM-1-Appops360'[OriginalId]),
        'SourceMoisM-1-Appops360'[EstClôturé] = "Oui"
    )

VAR CountByTribu = 
    DISTINCTCOUNT(
        CALCULATETABLE(
            VALUES('SourceMoisM-1-Appops360'[OriginalId]),
            'SourceMoisM-1-Appops360'[EstClôturé] = "Oui",
            'SourceMoisM-1-Appops360'[Appops] = SelectedAppops,
            'SourceMoisM-1-Appops360'[Tribu] = SelectedTribu
        )
    )

RETURN
    IF(
        NOT(ISBLANK(SelectedTribu)),
        CountByTribu,
        CountByAppops
    )
```

---

## SOLUTION 2 : COUNTX + FILTER

```dax
VCECloturés_V2 = 
VAR SelectedAppops = SELECTEDVALUE(appops_secu[Appops])
VAR SelectedTribu = SELECTEDVALUE(appops_secu[Tribu])

VAR CountByAppops = 
    CALCULATE(
        DISTINCTCOUNT('SourceMoisM-1-Appops360'[OriginalId]),
        'SourceMoisM-1-Appops360'[EstClôturé] = "Oui"
    )

VAR CountByTribu = 
    COUNTX(
        VALUES('SourceMoisM-1-Appops360'[OriginalId]),
        IF(
            CALCULATE(
                COUNTROWS('SourceMoisM-1-Appops360'),
                'SourceMoisM-1-Appops360'[EstClôturé] = "Oui",
                'SourceMoisM-1-Appops360'[Appops] = SelectedAppops,
                'SourceMoisM-1-Appops360'[Tribu] = SelectedTribu
            ) > 0,
            1,
            BLANK()
        )
    )

RETURN
    IF(
        NOT(ISBLANK(SelectedTribu)),
        CountByTribu,
        CountByAppops
    )
```

---

## SOLUTION 3 : ALL + double FILTER

```dax
VCECloturés_V3 = 
VAR SelectedAppops = SELECTEDVALUE(appops_secu[Appops])
VAR SelectedTribu = SELECTEDVALUE(appops_secu[Tribu])

VAR CountByAppops = 
    CALCULATE(
        DISTINCTCOUNT('SourceMoisM-1-Appops360'[OriginalId]),
        'SourceMoisM-1-Appops360'[EstClôturé] = "Oui"
    )

VAR CountByTribu = 
    CALCULATE(
        DISTINCTCOUNT('SourceMoisM-1-Appops360'[OriginalId]),
        FILTER(
            ALL('SourceMoisM-1-Appops360'),
            'SourceMoisM-1-Appops360'[EstClôturé] = "Oui"
            && 'SourceMoisM-1-Appops360'[Appops] = SelectedAppops
            && 'SourceMoisM-1-Appops360'[Tribu] = SelectedTribu
        )
    )

RETURN
    IF(
        NOT(ISBLANK(SelectedTribu)),
        CountByTribu,
        CountByAppops
    )
```

---

## SOLUTION 4 : KEEPFILTERS pour préserver contexte

```dax
VCECloturés_V4 = 
VAR SelectedAppops = SELECTEDVALUE(appops_secu[Appops])
VAR SelectedTribu = SELECTEDVALUE(appops_secu[Tribu])

VAR CountByAppops = 
    CALCULATE(
        DISTINCTCOUNT('SourceMoisM-1-Appops360'[OriginalId]),
        'SourceMoisM-1-Appops360'[EstClôturé] = "Oui"
    )

VAR CountByTribu = 
    CALCULATE(
        DISTINCTCOUNT('SourceMoisM-1-Appops360'[OriginalId]),
        'SourceMoisM-1-Appops360'[EstClôturé] = "Oui",
        KEEPFILTERS('SourceMoisM-1-Appops360'[Appops] = SelectedAppops),
        KEEPFILTERS('SourceMoisM-1-Appops360'[Tribu] = SelectedTribu)
    )

RETURN
    IF(
        NOT(ISBLANK(SelectedTribu)),
        CountByTribu,
        CountByAppops
    )
```

---

## SOLUTION 5 : ADDCOLUMNS + COUNTROWS (la plus simple)

```dax
VCECloturés_V5 = 
VAR SelectedAppops = SELECTEDVALUE(appops_secu[Appops])
VAR SelectedTribu = SELECTEDVALUE(appops_secu[Tribu])

VAR CountByAppops = 
    CALCULATE(
        DISTINCTCOUNT('SourceMoisM-1-Appops360'[OriginalId]),
        'SourceMoisM-1-Appops360'[EstClôturé] = "Oui"
    )

VAR TableFiltrée = 
    FILTER(
        'SourceMoisM-1-Appops360',
        'SourceMoisM-1-Appops360'[EstClôturé] = "Oui"
        && 'SourceMoisM-1-Appops360'[Appops] = SelectedAppops
        && 'SourceMoisM-1-Appops360'[Tribu] = SelectedTribu
    )

VAR CountByTribu = 
    COUNTROWS(
        DISTINCT(
            SELECTCOLUMNS(
                TableFiltrée,
                "ID", 'SourceMoisM-1-Appops360'[OriginalId]
            )
        )
    )

RETURN
    IF(
        NOT(ISBLANK(SelectedTribu)),
        CountByTribu,
        CountByAppops
    )
```

---

## TEST

**Créé les 5 mesures (V1, V2, V3, V4, V5)**

**Affiche-les dans 5 cartes côte à côte**

**Sélectionne AppOps_D > Tribu_D**

**→ Dis-moi laquelle(s) affiche(nt) 8 !** 🎯

Une fois qu'on sait laquelle marche, on l'applique aussi à VCEDetectés ! 💪

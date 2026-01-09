Voici la correction complète ! 👍

---

## VCECloturés (version corrigée complète)

```dax
VCECloturés = 
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
        'SourceMoisM-1-Appops360'[Appops] = SelectedAppops,
        'SourceMoisM-1-Appops360'[Tribu] = SelectedTribu
    )

RETURN
    IF(
        NOT(ISBLANK(SelectedTribu)),
        CountByTribu,
        CountByAppops
    )
```

---

## VCEDetectés (version corrigée complète)

```dax
VCEDetectés = 
VAR SelectedAppops = SELECTEDVALUE(appops_secu[Appops])
VAR SelectedTribu = SELECTEDVALUE(appops_secu[Tribu])

VAR CountByAppops = 
    CALCULATE(
        DISTINCTCOUNT('SourceMoisActuel-Appops360'[OriginalId]),
        'SourceMoisActuel-Appops360'[EstDetecté] = "Oui"
    )

VAR CountByTribu = 
    CALCULATE(
        DISTINCTCOUNT('SourceMoisActuel-Appops360'[OriginalId]),
        'SourceMoisActuel-Appops360'[EstDetecté] = "Oui",
        'SourceMoisActuel-Appops360'[Appops] = SelectedAppops,
        'SourceMoisActuel-Appops360'[Tribu] = SelectedTribu
    )

RETURN
    IF(
        NOT(ISBLANK(SelectedTribu)),
        CountByTribu,
        CountByAppops
    )
```

---

## Ce qui a changé

**Avant (dans CountByTribu) :**
```dax
ALL('SourceMoisM-1-Appops360'[Appops]),
USERELATIONSHIP(appops_secu[Tribu], 'SourceMoisM-1-Appops360'[Tribu])
```

**Après (dans CountByTribu) :**
```dax
'SourceMoisM-1-Appops360'[Appops] = SelectedAppops,
'SourceMoisM-1-Appops360'[Tribu] = SelectedTribu
```

**Et ajout au début :**
```dax
VAR SelectedAppops = SELECTEDVALUE(appops_secu[Appops])
```

---

**Copie-colle ces deux mesures et teste !** 🎯

Normalement AppOps_D > Tribu_D devrait maintenant donner **8** et non 283 ! 💪

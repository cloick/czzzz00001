Ah oui ! **Je vois le problème !** 🔍

En enlevant `USERELATIONSHIP`, la relation **INACTIVE** sur Tribu ne fonctionne plus !

**Le problème :**
```dax
'SourceMoisM-1-Appops360'[Tribu] = SelectedTribu
```
→ Ça ne marche pas car il n'y a pas de relation active entre `appops_secu[Tribu]` et la table ! ❌

---

## Solution : Garder USERELATIONSHIP + Forcer le contexte Appops

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
        'SourceMoisM-1-Appops360'[Appops] = SelectedAppops,  // ← Force le contexte Appops
        USERELATIONSHIP(appops_secu[Tribu], 'SourceMoisM-1-Appops360'[Tribu])  // ← Active relation Tribu
    )

RETURN
    IF(
        NOT(ISBLANK(SelectedTribu)),
        CountByTribu,
        CountByAppops
    )
```

---

## VCEDetectés (même correction)

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
        USERELATIONSHIP(appops_secu[Tribu], 'SourceMoisActuel-Appops360'[Tribu])
    )

RETURN
    IF(
        NOT(ISBLANK(SelectedTribu)),
        CountByTribu,
        CountByAppops
    )
```

---

## Explication

**Maintenant on a les DEUX filtres :**

1. **`[Appops] = SelectedAppops`** → Force le filtre sur l'Appops sélectionnée
2. **`USERELATIONSHIP(...)`** → Active la relation inactive sur Tribu

**Résultat :**
- AppOps_D > Tribu_D → Filtre sur **AppOps_D ET Tribu_D** = 8 ✅

---

**Teste cette version !** 🎯

Normalement maintenant :
- Sélection Tribu → Affiche un nombre (pas vide)
- AppOps_D > Tribu_D → Affiche 8 (pas 283)

Dis-moi ! 💪

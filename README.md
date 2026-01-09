Ah ! Je vois le problème ! 🔍

**La vérification `ComboExists` ne fonctionne pas correctement.**

Le souci : elle essaie de filtrer directement sur `MoisActuel[Appops]` et `MoisActuel[Tribu]` en même temps, mais avec les relations actives/inactives, ça ne marche pas.

---

## Solution : Vérification SANS utiliser les relations

On va vérifier si la combo existe **en ignorant complètement les relations** :

### VCEDetectés (version corrigée)

```dax
VCEDetectés = 
VAR SelectedAppops = SELECTEDVALUE(appops_secu[Appops])
VAR SelectedTribu = SELECTEDVALUE(appops_secu[Tribu])

// Vérifie si combo existe en IGNORANT les relations
VAR ComboExists = 
    IF(
        NOT(ISBLANK(SelectedTribu)),
        COUNTROWS(
            FILTER(
                ALL('MoisActuel'),
                'MoisActuel'[Appops] = SelectedAppops
                && 'MoisActuel'[Tribu] = SelectedTribu
            )
        ) > 0,
        TRUE
    )

VAR CountByAppops = 
    CALCULATE(
        DISTINCTCOUNT('MoisActuel'[OriginalId]),
        'MoisActuel'[EstDetecté] = "Oui"
    )

VAR CountByTribu = 
    CALCULATE(
        DISTINCTCOUNT('MoisActuel'[OriginalId]),
        'MoisActuel'[EstDetecté] = "Oui",
        ALL('MoisActuel'[Appops]),
        USERELATIONSHIP(appops_secu[Tribu], 'MoisActuel'[Tribu])
    )

RETURN
    SWITCH(
        TRUE(),
        NOT(ISBLANK(SelectedTribu)) && NOT(ComboExists), BLANK(),
        NOT(ISBLANK(SelectedTribu)), CountByTribu,
        CountByAppops
    )
```

---

### VCECloturés (version corrigée)

```dax
VCECloturés = 
VAR SelectedAppops = SELECTEDVALUE(appops_secu[Appops])
VAR SelectedTribu = SELECTEDVALUE(appops_secu[Tribu])

// Vérifie si combo existe en IGNORANT les relations
VAR ComboExists = 
    IF(
        NOT(ISBLANK(SelectedTribu)),
        COUNTROWS(
            FILTER(
                ALL('MoisPrecedent'),
                'MoisPrecedent'[Appops] = SelectedAppops
                && 'MoisPrecedent'[Tribu] = SelectedTribu
            )
        ) > 0,
        TRUE
    )

VAR CountByAppops = 
    CALCULATE(
        DISTINCTCOUNT('MoisPrecedent'[OriginalId]),
        'MoisPrecedent'[EstClôturé] = "Oui"
    )

VAR CountByTribu = 
    CALCULATE(
        DISTINCTCOUNT('MoisPrecedent'[OriginalId]),
        'MoisPrecedent'[EstClôturé] = "Oui",
        ALL('MoisPrecedent'[Appops]),
        USERELATIONSHIP(appops_secu[Tribu], 'MoisPrecedent'[Tribu])
    )

RETURN
    SWITCH(
        TRUE(),
        NOT(ISBLANK(SelectedTribu)) && NOT(ComboExists), BLANK(),
        NOT(ISBLANK(SelectedTribu)), CountByTribu,
        CountByAppops
    )
```

---

## Différence clé

**Avant (ne marchait pas) :**
```dax
VAR ComboExists = 
    CALCULATE(
        COUNTROWS('MoisActuel'),
        'MoisActuel'[Appops] = SelectedAppops,
        'MoisActuel'[Tribu] = SelectedTribu
    ) > 0
```
→ Utilise CALCULATE qui est influencé par les relations actives ❌

**Après (corrigé) :**
```dax
VAR ComboExists = 
    COUNTROWS(
        FILTER(
            ALL('MoisActuel'),  // ← Ignore TOUTES les relations
            'MoisActuel'[Appops] = SelectedAppops
            && 'MoisActuel'[Tribu] = SelectedTribu
        )
    ) > 0
```
→ Utilise FILTER avec ALL = recherche directe dans la table ✅

---

## TESTE CES NOUVELLES VERSIONS

**Remplace tes mesures et refais les tests :**

### Test 1 : AppOps_A (toutes tribus)
```
VCEDetectés = ?
VCECloturés = ?
```

### Test 2 : AppOps_A > Tribu_Alpha (existe)
```
VCEDetectés = ?
VCECloturés = ?
```

### Test 3 : AppOps_A > Tribu_XXX (n'existe PAS)
```
VCEDetectés = ? (devrait être BLANK)
VCECloturés = ? (devrait être BLANK)
```

**→ Dis-moi les résultats !** 🎯

Cette fois ça devrait marcher ! 💪

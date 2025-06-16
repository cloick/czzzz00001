Je vois le problème dans votre formule DAX. Voici la version corrigée :

```dax
Diff_dds_vdr_h = 
IF(
    OR(
        incident[dv_state] = "Résolu",
        incident[dv_state] = "Fermé"
    ),
    DATEDIFF(
        incident[u_loss_date],
        incident[u_effective_resolved_at],
        HOUR
    ),
    BLANK()
)
```

## Corrections apportées :

1. **Syntaxe OR** : `OR(condition1, condition2)` au lieu de `or(incident[dv_state]="Résolu",incident[dv_state]="Fermé")`

2. **Guillemets** : Utilisation de guillemets droits `"` au lieu de guillemets courbes

3. **Structure IF** : Formatage propre avec indentation

4. **BLANK()** : Plus propre que de laisser vide pour la valeur else

## Alternative plus concise :
```dax
Diff_dds_vdr_h = 
IF(
    incident[dv_state] IN {"Résolu", "Fermé"},
    DATEDIFF(
        incident[u_loss_date],
        incident[u_effective_resolved_at],
        HOUR
    ),
    BLANK()
)
```

La fonction `IN` est souvent plus lisible quand on teste plusieurs valeurs sur la même colonne.

Testez cette version corrigée !

😅 **Ah non ! C'est l'inverse du problème !**

Le souci : `ALL(obso_planifall)` à l'extérieur **écrase** les filtres `obso_planifall[Appops] = CurrentAppops` à l'intérieur !

---

## SOLUTION : ALL() à l'INTÉRIEUR, pas à l'extérieur

```dax
Moyenne_Serveur_OS_Toutes_AppOps = 
AVERAGEX(
    FILTER(
        ALL(obso_planifall[Appops]),
        obso_planifall[Appops] <> "Non pris"
    ),
    VAR CurrentAppops = obso_planifall[Appops]
    RETURN
        DIVIDE(
            CALCULATE(
                DISTINCTCOUNT(obso_planifall[NOM_CS]),
                ALL(obso_planifall),  // ← Ignore TOUS les filtres externes
                obso_planifall[Appops] = CurrentAppops,  // ← Puis filtre sur cette Appops
                obso_planifall[statut_obso] IN {"Obsolète majeur", "Obsolète"},
                obso_planifall[type_composant] = "OS"
            ),
            CALCULATE(
                DISTINCTCOUNT(obso_planifall[NOM_CS]),
                ALL(obso_planifall),  // ← Ignore TOUS les filtres externes
                obso_planifall[Appops] = CurrentAppops,  // ← Puis filtre sur cette Appops
                obso_planifall[type_composant] = "OS"
            ),
            0
        )
)
```

---

## Explication de l'ordre des filtres

**Dans CALCULATE, l'ordre compte !**

```dax
CALCULATE(
    DISTINCTCOUNT(...),
    ALL(obso_planifall),           // 1. Enlève TOUS les filtres
    obso_planifall[Appops] = CurrentAppops,  // 2. Puis applique CE filtre
    ...autres filtres...
)
```

**Résultat :**
- Ignore les filtres externes (Tribu, Appops du slicer)
- Calcule uniquement pour `CurrentAppops`

---

## Comportement attendu

```
Aucun filtre → Moyenne = 7%
Filtre AppOps_A → Moyenne = 7% ✅ (fixe)
Filtre Tribu_1 → Moyenne = 7% ✅ (fixe)
```

---

**Teste cette version ! Normalement tu devrais retrouver 7% et ça doit rester fixe !** 🎯

Dis-moi ce que ça donne ! 💪

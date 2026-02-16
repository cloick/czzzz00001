**Ah ! Le problème : il manque `ALL()` pour ignorer les filtres externes de Tribu et Appops !** 🔍

---

## Version CORRIGÉE (ignore Appops ET Tribu)

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
                ALL(obso_planifall[Appops]),  // ← AJOUTER
                ALL(obso_planifall[Tribu]),   // ← AJOUTER
                obso_planifall[Appops] = CurrentAppops,
                obso_planifall[statut] IN {"en maintenant", "en avenir"},
                obso_planifall[statut_obso] IN {"Obsolète majeur", "Obsolète"},
                obso_planifall[type_composant] = "OS"
            ),
            CALCULATE(
                DISTINCTCOUNT(obso_planifall[NOM_CS]),
                ALL(obso_planifall[Appops]),  // ← AJOUTER
                ALL(obso_planifall[Tribu]),   // ← AJOUTER
                obso_planifall[Appops] = CurrentAppops,
                obso_planifall[statut] IN {"en maintenant", "en avenir"},
                obso_planifall[type_composant] = "OS"
            ),
            0
        )
)
```

---

## Explication

**Dans chaque CALCULATE, on doit :**
1. `ALL(obso_planifall[Appops])` → Enlève le filtre externe Appops
2. `ALL(obso_planifall[Tribu])` → **Enlève le filtre externe Tribu** ✅
3. `obso_planifall[Appops] = CurrentAppops` → Réapplique le filtre sur l'Appops en cours d'itération
4. Garde les filtres hardcodés (statut, statut_obso, type_composant)

---

## Comportement attendu

```
┌─────────────────────────────────────────────────────────┐
│  Aucun filtre                                           │
└─────────────────────────────────────────────────────────┘
Serveur_OS = 8%
Moyenne = 7%  ✅ (fixe)

┌─────────────────────────────────────────────────────────┐
│  Filtre : AppOps_A                                      │
└─────────────────────────────────────────────────────────┘
Serveur_OS = 4%  ✅ (change)
Moyenne = 7%  ✅ (fixe)

┌─────────────────────────────────────────────────────────┐
│  Filtre : AppOps_A > Tribu_1                            │
└─────────────────────────────────────────────────────────┘
Serveur_OS = 4%  ✅ (change)
Moyenne = 7%  ✅ (DOIT RESTER FIXE maintenant !)
```

---

**Teste cette version ! Maintenant la moyenne devrait rester à 7% même quand tu sélectionnes une Tribu !** 🎯💪

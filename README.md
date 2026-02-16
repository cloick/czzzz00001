Ah oui ! **Le problème : il manque `ALL(obso_planifall)` pour ignorer TOUS les filtres !** 🔍

---

## Version CORRIGÉE (figée sur tous les filtres)

```dax
Moyenne_Serveur_OS_Toutes_AppOps = 
CALCULATE(
    AVERAGEX(
        FILTER(
            VALUES(obso_planifall[Appops]),
            obso_planifall[Appops] <> "Non pris"
        ),
        VAR CurrentAppops = obso_planifall[Appops]
        RETURN
            DIVIDE(
                CALCULATE(
                    DISTINCTCOUNT(obso_planifall[NOM_CS]),
                    obso_planifall[Appops] = CurrentAppops,
                    obso_planifall[statut_obso] IN {"Obsolète majeur", "Obsolète"},
                    obso_planifall[type_composant] = "OS"
                ),
                CALCULATE(
                    DISTINCTCOUNT(obso_planifall[NOM_CS]),
                    obso_planifall[Appops] = CurrentAppops,
                    obso_planifall[type_composant] = "OS"
                ),
                0
            )
    ),
    ALL(obso_planifall)  // ← AJOUTER CECI pour ignorer TOUS les filtres !
)
```

---

## Explication

**La structure complète :**

```dax
CALCULATE(
    AVERAGEX(...),
    ALL(obso_planifall)  // ← Ignore TOUS les filtres (Appops ET Tribu)
)
```

**Ce qui se passe :**
1. `ALL(obso_planifall)` enlève TOUS les filtres externes
2. `AVERAGEX` itère sur TOUTES les Appops (sauf "Non pris")
3. Pour chaque Appops, recalcule le ratio complet
4. Fait la moyenne de tous ces ratios

**Résultat :** Valeur FIGÉE qui ne bouge jamais ! ✅

---

## Comportement attendu

```
┌─────────────────────────────────────────────────────────┐
│  Aucun filtre                                           │
└─────────────────────────────────────────────────────────┘
Serveur_OS = 8%
Moyenne_Serveur_OS_Toutes_AppOps = 7%  (fixe)

┌─────────────────────────────────────────────────────────┐
│  Filtre : AppOps_A                                      │
└─────────────────────────────────────────────────────────┘
Serveur_OS = 4%  ✅ (change)
Moyenne_Serveur_OS_Toutes_AppOps = 7%  ✅ (fixe)

┌─────────────────────────────────────────────────────────┐
│  Filtre : AppOps_A > Tribu_1                            │
└─────────────────────────────────────────────────────────┘
Serveur_OS = 4%  ✅ (change)
Moyenne_Serveur_OS_Toutes_AppOps = 7%  ✅ (DOIT RESTER 7%)
```

---

**Teste cette version ! Maintenant la moyenne devrait rester à 7% peu importe les filtres !** 🎯

Dis-moi si ça marche ! 💪

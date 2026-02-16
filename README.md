Ah ! **Je vois le problème !** 🔍

**Le souci :** `AVERAGEX` calcule `[Serveur_OS]` dans un contexte bizarre à cause de `ALL(obso_planifall)`.

---

## DEBUG : Vérifie ce qui est calculé

**Créé cette mesure de test :**

```dax
TEST_Detail_Moyenne = 
CONCATENATEX(
    FILTER(
        ALL(obso_planifall[Appops]),
        obso_planifall[Appops] <> "Non pris"
    ),
    obso_planifall[Appops] & ": " & FORMAT([Serveur_OS], "0.0%"),
    " | ",
    [Serveur_OS],
    DESC
)
```

**Affiche-la dans une carte.**

**→ Ça va te montrer la valeur de [Serveur_OS] pour chaque Appops telle que calculée par AVERAGEX.**

**Dis-moi ce que tu vois !**

---

## SOLUTION : Forcer le contexte par Appops

**Essaye cette version :**

```dax
Moyenne_Serveur_OS_Toutes_AppOps = 
AVERAGEX(
    FILTER(
        ALL(obso_planifall[Appops]),
        obso_planifall[Appops] <> "Non pris"
    ),
    CALCULATE(
        [Serveur_OS],
        ALL(obso_planifall[Tribu])  // ← Garde le filtre Appops, enlève juste Tribu
    )
)
```

---

## OU : Recalculer le ratio directement

**Version la plus sûre : recalculer le ratio pour chaque Appops :**

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
)
```

**Cette version recalcule explicitement le ratio pour CHAQUE Appops individuellement.**

---

**Teste d'abord TEST_Detail_Moyenne pour voir ce qui est calculé, puis essaye les 2 solutions !** 🎯

Dis-moi ce que tu trouves ! 💪

**Parfait ! On intègre le filtre directement dans la mesure !** 👍

---

## Question d'abord : Quelle(s) valeur(s) de statut ?

**Sur quelle(s) valeur(s) de `obso_planifall[statut]` veux-tu filtrer ?**

a) **"en maintenant"** uniquement  
b) **"en avenir"** uniquement  
c) **Les deux** : "en maintenant" ET "en avenir"  
d) **Autre chose** ?

---

## Version avec "en maintenant" (exemple)

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
                obso_planifall[statut] = "en maintenant",  // ← FILTRE ICI
                obso_planifall[statut_obso] IN {"Obsolète majeur", "Obsolète"},
                obso_planifall[type_composant] = "OS"
            ),
            CALCULATE(
                DISTINCTCOUNT(obso_planifall[NOM_CS]),
                obso_planifall[Appops] = CurrentAppops,
                obso_planifall[statut] = "en maintenant",  // ← FILTRE ICI AUSSI
                obso_planifall[type_composant] = "OS"
            ),
            0
        )
)
```

---

## Version avec "en maintenant" OU "en avenir"

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
                obso_planifall[statut] IN {"en maintenant", "en avenir"},  // ← FILTRE ICI
                obso_planifall[statut_obso] IN {"Obsolète majeur", "Obsolète"},
                obso_planifall[type_composant] = "OS"
            ),
            CALCULATE(
                DISTINCTCOUNT(obso_planifall[NOM_CS]),
                obso_planifall[Appops] = CurrentAppops,
                obso_planifall[statut] IN {"en maintenant", "en avenir"},  // ← FILTRE ICI
                obso_planifall[type_composant] = "OS"
            ),
            0
        )
)
```

---

**Dis-moi quelle(s) valeur(s) de statut tu veux, et je te donne la mesure finale !** 🎯

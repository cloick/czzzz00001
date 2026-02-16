😤 **OK, on arrête de deviner ! Il faut DÉBOGUER pour comprendre ce qui se passe !**

---

## MESURES DE DEBUG

**Créé ces 3 mesures pour voir ce qui est calculé :**

### 1. Combien d'Appops sont comptées ?

```dax
DEBUG_NbAppops = 
COUNTROWS(
    FILTER(
        ALL(obso_planifall[Appops]),
        obso_planifall[Appops] <> "Non pris"
    )
)
```

**→ Devrait donner 10 si tu as 10 Appops (sauf "Non pris")**

---

### 2. Quelle est la SOMME des ratios ?

```dax
DEBUG_SommeRatios = 
SUMX(
    FILTER(
        ALL(obso_planifall[Appops]),
        obso_planifall[Appops] <> "Non pris"
    ),
    VAR CurrentAppops = obso_planifall[Appops]
    RETURN
        DIVIDE(
            CALCULATE(
                DISTINCTCOUNT(obso_planifall[NOM_CS]),
                ALL(obso_planifall),
                obso_planifall[Appops] = CurrentAppops,
                obso_planifall[statut_obso] IN {"Obsolète majeur", "Obsolète"},
                obso_planifall[type_composant] = "OS"
            ),
            CALCULATE(
                DISTINCTCOUNT(obso_planifall[NOM_CS]),
                ALL(obso_planifall),
                obso_planifall[Appops] = CurrentAppops,
                obso_planifall[type_composant] = "OS"
            ),
            0
        )
)
```

**→ Si 10 Appops, devrait donner environ 0.65 (65%) pour obtenir moyenne de 6.5%**

---

### 3. Détail par Appops (la plus importante !)

```dax
DEBUG_DetailAppops = 
CONCATENATEX(
    FILTER(
        ALL(obso_planifall[Appops]),
        obso_planifall[Appops] <> "Non pris"
    ),
    VAR CurrentAppops = obso_planifall[Appops]
    VAR Ratio = 
        DIVIDE(
            CALCULATE(
                DISTINCTCOUNT(obso_planifall[NOM_CS]),
                ALL(obso_planifall),
                obso_planifall[Appops] = CurrentAppops,
                obso_planifall[statut_obso] IN {"Obsolète majeur", "Obsolète"},
                obso_planifall[type_composant] = "OS"
            ),
            CALCULATE(
                DISTINCTCOUNT(obso_planifall[NOM_CS]),
                ALL(obso_planifall),
                obso_planifall[Appops] = CurrentAppops,
                obso_planifall[type_composant] = "OS"
            ),
            0
        )
    RETURN
        CurrentAppops & ": " & FORMAT(Ratio, "0.0%"),
    UNICHAR(10),  // Retour à la ligne
    Ratio,
    DESC
)
```

**→ Ça va lister TOUTES les Appops avec leur ratio calculé**

---

## TESTE CES 3 MESURES

**Affiche-les dans des cartes et donne-moi les résultats :**

1. **DEBUG_NbAppops** = ?
2. **DEBUG_SommeRatios** = ?
3. **DEBUG_DetailAppops** = ? (copie-colle toute la liste)

**Avec ces infos, on va comprendre exactement pourquoi ça donne 10 au lieu de 7 !** 🔍

Vas-y, teste et envoie-moi les résultats ! 💪

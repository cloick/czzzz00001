Je vais vous aider à créer cette mesure DAX pour extraire la date de votre colonne Source.Name.

Pour extraire la date du format "2025-02-27-obso_planifall.xlsx", vous pouvez utiliser les fonctions de texte en DAX. Voici la mesure que vous pouvez créer :

```
date_extr = 
VAR DateTexte = LEFT(planif_M[Source.Name], 10)
RETURN DATE(
    VALUE(LEFT(DateTexte, 4)), 
    VALUE(MID(DateTexte, 6, 2)), 
    VALUE(RIGHT(DateTexte, 2))
)
```

Cette mesure fonctionne ainsi :
1. Elle extrait les 10 premiers caractères de la colonne Source.Name (qui correspondent à "2025-02-27")
2. Puis convertit cette chaîne en une vraie date en extrayant l'année (4 premiers caractères), le mois (2 caractères après la position 6) et le jour (2 derniers caractères)

Si vous avez besoin d'adapter cette formule pour différents formats ou si vous avez d'autres questions, n'hésitez pas à me le faire savoir.

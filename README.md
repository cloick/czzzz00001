
Je comprends ce que vous cherchez à faire pour compléter votre jauge. Vous voulez une mesure "valeur cible" qui:

1. Réagit au filtre segment Appops (contrairement aux autres mesures qui ignorent ce filtre)
2. Si une Appops spécifique est sélectionnée dans le segment, elle doit afficher le nombre d'occurrences de cette Appops
3. Cette mesure doit quand même respecter les filtres Environnement comme les autres mesures

Pour cette valeur cible, nous n'utiliserons pas ALL('Appops_'[Appops]) car nous voulons que la mesure soit sensible à la sélection de l'utilisateur. Nous aurons besoin de:

1. Capturer le filtre actuel sur Appops 
2. Respecter également le filtre Environnement comme pour les autres mesures
3. Compter les occurrences de l'Appops sélectionnée

Si plusieurs Appops sont sélectionnées ou si aucune n'est sélectionnée, nous devrons décider d'un comportement par défaut (soit somme, soit moyenne, soit valeur maximale des sélections).

Je comprends parfaitement ce que vous cherchez à faire. Pour compléter votre jauge, vous avez besoin d'une mesure "valeur cible" qui, contrairement aux autres, doit réagir à la sélection du segment Appops.

Voici la mesure pour votre valeur cible :

```
ValeurCibleWindows = 
VAR CurrentEnvFilter = TREATAS(VALUES('Environnement'[Environnement]), 'windows_new'[Environnement])
VAR CurrentAppsSelection = VALUES('Appops_'[Appops]) // Capture la sélection actuelle d'Appops
VAR SelectedCount = 
    CALCULATE(
        COUNTROWS('windows_new'),
        CurrentEnvFilter,
        FILTER(
            ALL('windows_new'),
            'windows_new'[Appops] IN CurrentAppsSelection
            && FORMAT('windows_new'[Appops], "") <> ""
        )
    )
RETURN
    // Si une Appops est sélectionnée, renvoie son nombre d'occurrences
    IF(
        COUNTROWS(CurrentAppsSelection) > 0,
        SelectedCount,
        // Sinon, valeur par défaut (0 ou autre valeur souhaitée)
        0
    )
```

Cette mesure :
1. Respecte les filtres Environnement comme les autres mesures
2. Compte uniquement les occurrences des Appops sélectionnées dans le segment
3. Si aucune Appops n'est sélectionnée, renvoie 0 (ou une autre valeur par défaut que vous pourriez préférer)

Cette mesure complètera parfaitement votre jauge car elle fournira la valeur cible qui réagit au segment Appops.

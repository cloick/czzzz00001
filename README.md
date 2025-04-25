Oui, je comprends parfaitement ce que vous souhaitez faire. Vous voulez que votre mesure soit "insensible" aux sélections faites dans un segment/slicer d'Appops, c'est-à-dire qu'elle continue à calculer le minimum parmi TOUTES les Appops, même si l'utilisateur n'en a sélectionné que quelques-unes.
Pour réaliser cela, il faut ajouter la fonction REMOVEFILTERS à votre mesure. Voici la modification à apporter:
MinValueWindows = 
VAR AppopsCount = 
    ADDCOLUMNS(
        VALUES(REMOVEFILTERS('Appops_'[Appops])),  // Ajout de REMOVEFILTERS ici
        "Comptage", 
        COALESCE(
            CALCULATE(
                COUNTROWS('windows_new'),
                REMOVEFILTERS('windows_new'),  // Ajout de REMOVEFILTERS ici aussi
                FILTER(
                    ALL('windows_new'),
                    'windows_new'[Appops] = EARLIER('Appops_'[Appops])
                )
            ),
            0
        )
    )
RETURN
    MINX(AppopsCount, [Comptage])
Les changements clés sont:

VALUES(REMOVEFILTERS('Appops_'[Appops])) - Ignore les filtres sur la table Appops_, garantissant que toutes les valeurs d'Appops sont considérées, peu importe ce qui est sélectionné dans le segment.
REMOVEFILTERS('windows_new') - S'assure que les filtres ne sont pas propagés à windows_new via les relations entre tables.

Ces deux modifications ensemble rendront votre mesure totalement insensible aux sélections dans le segment d'Appops, et elle continuera à afficher le même résultat (0 si au moins une Appops est absente).

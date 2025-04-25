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

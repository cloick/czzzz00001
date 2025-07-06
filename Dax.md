MESURES POUR LE VISUEL "TECHNIQUE"1. Mesure Target (pour l'Appops sélectionné)Moyenne Technique = 
CALCULATE(
    AVERAGE(incident[Diff_dds_vdr_h]),
    incident[Nature] = "Technique"
)2. Mesures Max/Min/Moyenne globale (ne réagissent pas au filtre segment)MaxValue Technique = 
VAR AllAppops = ALL(incident[Appops])
VAR MoyenneParAppops = 
    ADDCOLUMNS(
        FILTER(AllAppops, incident[Appops] <> "Non pris"),
        "MoyenneTech",
        CALCULATE(
            AVERAGE(incident[Diff_dds_vdr_h]),
            incident[Nature] = "Technique",
            ALLEXCEPT(incident, incident[Appops]),
            FILTER(
                ALL(incident),
                incident[Appops] = EARLIER(incident[Appops])
            )
        )
    )
VAR AppopsWithValues = 
    FILTER(MoyenneParAppops, NOT(ISBLANK([MoyenneTech])))
RETURN
    MAXX(AppopsWithValues, [MoyenneTech])MinValue Technique = 
VAR AllAppops = ALL(incident[Appops])
VAR MoyenneParAppops = 
    ADDCOLUMNS(
        FILTER(AllAppops, incident[Appops] <> "Non pris"),
        "MoyenneTech",
        CALCULATE(
            AVERAGE(incident[Diff_dds_vdr_h]),
            incident[Nature] = "Technique",
            ALLEXCEPT(incident, incident[Appops]),
            FILTER(
                ALL(incident),
                incident[Appops] = EARLIER(incident[Appops])
            )
        )
    )
VAR AppopsWithValues = 
    FILTER(MoyenneParAppops, NOT(ISBLANK([MoyenneTech])))
RETURN
    MINX(AppopsWithValues, [MoyenneTech])MoyValue Technique = 
VAR AllAppops = ALL(incident[Appops])
VAR MoyenneParAppops = 
    ADDCOLUMNS(
        FILTER(AllAppops, incident[Appops] <> "Non pris"),
        "MoyenneTech",
        CALCULATE(
            AVERAGE(incident[Diff_dds_vdr_h]),
            incident[Nature] = "Technique",
            ALLEXCEPT(incident, incident[Appops]),
            FILTER(
                ALL(incident),
                incident[Appops] = EARLIER(incident[Appops])
            )
        )
    )
VAR AppopsWithValues = 
    FILTER(MoyenneParAppops, NOT(ISBLANK([MoyenneTech])))
RETURN
    AVERAGEX(AppopsWithValues, [MoyenneTech])MESURES POUR LE VISUEL "FONCTIONNEL"1. Mesure TargetMoyenne Fonctionnel = 
CALCULATE(
    AVERAGE(incident[Diff_dds_vdr_h]),
    incident[Nature] = "Fonctionnel"
)2. Mesures Max/Min/Moyenne globaleMaxValue Fonctionnel = 
VAR AllAppops = ALL(incident[Appops])
VAR MoyenneParAppops = 
    ADDCOLUMNS(
        FILTER(AllAppops, incident[Appops] <> "Non pris"),
        "MoyenneFonc",
        CALCULATE(
            AVERAGE(incident[Diff_dds_vdr_h]),
            incident[Nature] = "Fonctionnel",
            ALLEXCEPT(incident, incident[Appops]),
            FILTER(
                ALL(incident),
                incident[Appops] = EARLIER(incident[Appops])
            )
        )
    )
VAR AppopsWithValues = 
    FILTER(MoyenneParAppops, NOT(ISBLANK([MoyenneFonc])))
RETURN
    MAXX(AppopsWithValues, [MoyenneFonc])MinValue Fonctionnel = 
VAR AllAppops = ALL(incident[Appops])
VAR MoyenneParAppops = 
    ADDCOLUMNS(
        FILTER(AllAppops, incident[Appops] <> "Non pris"),
        "MoyenneFonc",
        CALCULATE(
            AVERAGE(incident[Diff_dds_vdr_h]),
            incident[Nature] = "Fonctionnel",
            ALLEXCEPT(incident, incident[Appops]),
            FILTER(
                ALL(incident),
                incident[Appops] = EARLIER(incident[Appops])
            )
        )
    )
VAR AppopsWithValues = 
    FILTER(MoyenneParAppops, NOT(ISBLANK([MoyenneFonc])))
RETURN
    MINX(AppopsWithValues, [MoyenneFonc])MoyValue Fonctionnel = 
VAR AllAppops = ALL(incident[Appops])
VAR MoyenneParAppops = 
    ADDCOLUMNS(
        FILTER(AllAppops, incident[Appops] <> "Non pris"),
        "MoyenneFonc",
        CALCULATE(
            AVERAGE(incident[Diff_dds_vdr_h]),
            incident[Nature] = "Fonctionnel",
            ALLEXCEPT(incident, incident[Appops]),
            FILTER(
                ALL(incident),
                incident[Appops] = EARLIER(incident[Appops])
            )
        )
    )
VAR AppopsWithValues = 
    FILTER(MoyenneParAppops, NOT(ISBLANK([MoyenneFonc])))
RETURN
    AVERAGEX(AppopsWithValues, [MoyenneFonc])

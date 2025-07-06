Parfait ! Voici les 8 mesures avec tous les filtres appliqués :MESURES POUR TECHNIQUE1. Target (réagit au segment Appops)Moyenne Technique = 
CALCULATE(
    AVERAGE(incident[Diff_dds_vdr_h]),
    incident[D_opened_at] > DATE(2024,12,31),
    ISBLANK(incident[Maître]),
    incident[SAff2] = "APPOPS",
    incident[Nature] = "Technique"
)2. Max/Min/Moyenne (ne réagissent PAS au segment)MaxValueTechnique = 
VAR AllAppops = ALL(incident[Appops])
VAR MoyenneParAppops = 
    ADDCOLUMNS(
        FILTER(AllAppops, incident[Appops] <> "Non pris"),
        "MoyenneTech",
        CALCULATE(
            [Moyenne Technique],
            ALLEXCEPT(incident, incident[Appops]),
            FILTER(
                ALL(incident),
                incident[Appops] = EARLIER(incident[Appops])
            )
        )
    )
VAR AppopsWithValues = 
    FILTER(MoyenneParAppops, [MoyenneTech] > 0)
RETURN
    MAXX(AppopsWithValues, [MoyenneTech])MinValueTechnique = 
VAR AllAppops = ALL(incident[Appops])
VAR MoyenneParAppops = 
    ADDCOLUMNS(
        FILTER(AllAppops, incident[Appops] <> "Non pris"),
        "MoyenneTech",
        CALCULATE(
            [Moyenne Technique],
            ALLEXCEPT(incident, incident[Appops]),
            FILTER(
                ALL(incident),
                incident[Appops] = EARLIER(incident[Appops])
            )
        )
    )
VAR AppopsWithValues = 
    FILTER(MoyenneParAppops, [MoyenneTech] > 0)
RETURN
    MINX(AppopsWithValues, [MoyenneTech])MoyValueTechnique = 
VAR AllAppops = ALL(incident[Appops])
VAR MoyenneParAppops = 
    ADDCOLUMNS(
        FILTER(AllAppops, incident[Appops] <> "Non pris"),
        "MoyenneTech",
        CALCULATE(
            [Moyenne Technique],
            ALLEXCEPT(incident, incident[Appops]),
            FILTER(
                ALL(incident),
                incident[Appops] = EARLIER(incident[Appops])
            )
        )
    )
VAR AppopsWithValues = 
    FILTER(MoyenneParAppops, [MoyenneTech] > 0)
RETURN
    AVERAGEX(AppopsWithValues, [MoyenneTech])MESURES POUR FONCTIONNEL1. Target (réagit au segment Appops)Moyenne Fonctionnel = 
CALCULATE(
    AVERAGE(incident[Diff_dds_vdr_h]),
    incident[D_opened_at] > DATE(2024,12,31),
    ISBLANK(incident[Maître]),
    incident[SAff2] = "APPOPS",
    incident[Nature] = "Fonctionnel"
)2. Max/Min/Moyenne (remplacez "Technique" par "Fonctionnel" dans les 3 mesures précédentes)MaxValueFonctionnel = [même structure que MaxValueTechnique mais avec [Moyenne Fonctionnel]]
MinValueFonctionnel = [même structure que MinValueTechnique mais avec [Moyenne Fonctionnel]]
MoyValueFonctionnel = [même structure que MoyValueTechnique mais avec [Moyenne Fonctionnel]]

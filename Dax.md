L'utilisateur veut que je crée des mesures pour TauxSucces et TauxEchec, avec le même pattern que les mesures précédentes (target, max, min, moyenne globale).

Le visuel est construit avec :
- Axe X : colonne "Appops" de la table "change_request"
- Axe Y : deux mesures "TauxEchec" et "TauxSucces"

Les mesures existantes sont :
- TauxEchec = if ([TauxSucces], 100 - [TauxSucces], "")
- TauxSucces = 100*divide([NbCHGT_TotalSucces], [NbCHGT_TotalEchec]+[NbCHGT_TotalSucces])

Il n'y a aucun filtre sur le visuel.

Il veut que je nomme les mesures avec le pattern "MaxValueMEPTauxSucces" par exemple.

Je dois créer 8 mesures :
- Pour TauxSucces : Target (réagit au segment), Max, Min, Moyenne (ne réagissent pas au segment)
- Pour TauxEchec : Target (réagit au segment), Max, Min, Moyenne (ne réagissent pas au segment)

Je vais suivre le même pattern que pour les mesures précédentes, en utilisant la table "change_request" et la colonne "Appops".

Pour les mesures qui ne réagissent pas au segment, je vais exclure "Non pris" comme dans les mesures précédentes.

Parfait ! Voici les 8 mesures pour TauxSucces et TauxEchec :

## **MESURES POUR TAUXSUCCES**

### 1. Target (réagit au segment Appops)
```dax
TauxSucces MEP = [TauxSucces]
```

### 2. Max/Min/Moyenne (ne réagissent PAS au segment)
```dax
MaxValueMEPTauxSucces = 
VAR AllAppops = ALL(change_request[Appops])
VAR TauxParAppops = 
    ADDCOLUMNS(
        FILTER(AllAppops, change_request[Appops] <> "Non pris"),
        "TauxSuccesAppops",
        CALCULATE(
            [TauxSucces],
            ALLEXCEPT(change_request, change_request[Appops]),
            FILTER(
                ALL(change_request),
                change_request[Appops] = EARLIER(change_request[Appops])
            )
        )
    )
VAR AppopsWithValues = 
    FILTER(TauxParAppops, NOT(ISBLANK([TauxSuccesAppops])))
RETURN
    MAXX(AppopsWithValues, [TauxSuccesAppops])
```

```dax
MinValueMEPTauxSucces = 
VAR AllAppops = ALL(change_request[Appops])
VAR TauxParAppops = 
    ADDCOLUMNS(
        FILTER(AllAppops, change_request[Appops] <> "Non pris"),
        "TauxSuccesAppops",
        CALCULATE(
            [TauxSucces],
            ALLEXCEPT(change_request, change_request[Appops]),
            FILTER(
                ALL(change_request),
                change_request[Appops] = EARLIER(change_request[Appops])
            )
        )
    )
VAR AppopsWithValues = 
    FILTER(TauxParAppops, NOT(ISBLANK([TauxSuccesAppops])))
RETURN
    MINX(AppopsWithValues, [TauxSuccesAppops])
```

```dax
MoyValueMEPTauxSucces = 
VAR AllAppops = ALL(change_request[Appops])
VAR TauxParAppops = 
    ADDCOLUMNS(
        FILTER(AllAppops, change_request[Appops] <> "Non pris"),
        "TauxSuccesAppops",
        CALCULATE(
            [TauxSucces],
            ALLEXCEPT(change_request, change_request[Appops]),
            FILTER(
                ALL(change_request),
                change_request[Appops] = EARLIER(change_request[Appops])
            )
        )
    )
VAR AppopsWithValues = 
    FILTER(TauxParAppops, NOT(ISBLANK([TauxSuccesAppops])))
RETURN
    AVERAGEX(AppopsWithValues, [TauxSuccesAppops])
```

## **MESURES POUR TAUXECHEC**

### 1. Target (réagit au segment Appops)
```dax
TauxEchec MEP = [TauxEchec]
```

### 2. Max/Min/Moyenne (remplacez [TauxSucces] par [TauxEchec] dans les structures précédentes)
```dax
MaxValueMEPTauxEchec = [même structure avec [TauxEchec]]
MinValueMEPTauxEchec = [même structure avec [TauxEchec]]
MoyValueMEPTauxEchec = [même structure avec [TauxEchec]]
```

Voulez-vous que j'écrive les 3 dernières mesures TauxEchec en entier ? 😊

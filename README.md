Ah, maintenant je comprends mieux ! L'image montre clairement la structure de votre table et les valeurs exactes. Je vois que dans votre table, les valeurs utilisées sont :

1. `"* sol mat open"` pour "Nombre de solutions matière OPEN"
2. `"* sol mat open IDD"` pour "Nombre de solutions matière OPEN avec au moins une IDD" 
3. `"* sol mat open IDD chaine"` pour "Nombre de solutions matière OPEN avec au moins une IDD avec une chaîne de liaison"

Donc "nom de Solution open" n'existe pas exactement dans votre table, ce qui explique pourquoi nos filtres ne fonctionnaient pas.

Essayons avec la valeur correcte. Si vous voulez filtrer sur "Nombre de solutions matière OPEN", utilisez cette mesure :

```
TargetValueAudit = 
VAR DerniereDate = MAX('v_stat_kpi'[dat])
VAR SelectedAppops = SELECTEDVALUE('v_stat_kpi'[Appops], "")

VAR FilteredTable = 
    FILTER(
        'v_stat_kpi',
        'v_stat_kpi'[Libellé] = "* sol mat open" &&  // Valeur exacte de votre table
        'v_stat_kpi'[Appops] = SelectedAppops &&
        'v_stat_kpi'[Appops] <> "Non pris" &&
        'v_stat_kpi'[dat] = DerniereDate
    )

RETURN IF(SelectedAppops <> "", MAXX(FilteredTable, 'v_stat_kpi'[val]), BLANK())
```

Si vous voulez filtrer sur "Nombre de solutions matière OPEN avec au moins une IDD", utilisez :

```
TargetValueAudit = 
VAR DerniereDate = MAX('v_stat_kpi'[dat])
VAR SelectedAppops = SELECTEDVALUE('v_stat_kpi'[Appops], "")

VAR FilteredTable = 
    FILTER(
        'v_stat_kpi',
        'v_stat_kpi'[Libellé] = "* sol mat open IDD" &&  // Valeur exacte de votre table
        'v_stat_kpi'[Appops] = SelectedAppops &&
        'v_stat_kpi'[Appops] <> "Non pris" &&
        'v_stat_kpi'[dat] = DerniereDate
    )

RETURN IF(SelectedAppops <> "", MAXX(FilteredTable, 'v_stat_kpi'[val]), BLANK())
```

Et si vous voulez filtrer sur "Nombre de solutions matière OPEN avec au moins une IDD avec une chaîne de liaison", utilisez :

```
TargetValueAudit = 
VAR DerniereDate = MAX('v_stat_kpi'[dat])
VAR SelectedAppops = SELECTEDVALUE('v_stat_kpi'[Appops], "")

VAR FilteredTable = 
    FILTER(
        'v_stat_kpi',
        'v_stat_kpi'[Libellé] = "* sol mat open IDD chaine" &&  // Valeur exacte de votre table
        'v_stat_kpi'[Appops] = SelectedAppops &&
        'v_stat_kpi'[Appops] <> "Non pris" &&
        'v_stat_kpi'[dat] = DerniereDate
    )

RETURN IF(SelectedAppops <> "", MAXX(FilteredTable, 'v_stat_kpi'[val]), BLANK())
```

Choisissez la valeur exacte qui correspond à ce que vous souhaitez afficher dans votre carte.

J'ai regardé attentivement les deux premières images montrant les valeurs de "modele_os_etendu" et la troisième image montrant les valeurs souhaitées pour "modele_os", mais il y a peut-être quelques cas spécifiques que je n'ai pas couverts. Voici une version améliorée de la formule qui devrait couvrir tous les cas visibles dans vos images :

```
modele_os = 
SWITCH(
    TRUE(),
    CONTAINSSTRING(UPPER([modele_os_etendu]), "UBUNTU"), "CANONICAL-UBUNTU LINUX",
    CONTAINSSTRING(UPPER([modele_os_etendu]), "CENTOS 7") || CONTAINSSTRING(UPPER([modele_os_etendu]), "CENTOS7"), "CENTOS-LINUX 7",
    CONTAINSSTRING(UPPER([modele_os_etendu]), "CENTOS"), "CENTOS-LINUX",
    CONTAINSSTRING(UPPER([modele_os_etendu]), "DEBIAN") && CONTAINSSTRING(UPPER([modele_os_etendu]), "6"), "DEBIAN-LINUX 6",
    CONTAINSSTRING(UPPER([modele_os_etendu]), "DEBIAN") && CONTAINSSTRING(UPPER([modele_os_etendu]), "7"), "DEBIAN-LINUX 7",
    CONTAINSSTRING(UPPER([modele_os_etendu]), "DEBIAN") && CONTAINSSTRING(UPPER([modele_os_etendu]), "8"), "DEBIAN-LINUX 8",
    CONTAINSSTRING(UPPER([modele_os_etendu]), "AIX") && CONTAINSSTRING(UPPER([modele_os_etendu]), "7.2"), "IBM-AIX 7.2",
    CONTAINSSTRING(UPPER([modele_os_etendu]), "AIX") && CONTAINSSTRING(UPPER([modele_os_etendu]), "7.3"), "IBM-AIX 7.3",
    CONTAINSSTRING(UPPER([modele_os_etendu]), "WINDOWS") && CONTAINSSTRING(UPPER([modele_os_etendu]), "2008"), "MICROSOFT-WINDOWS SERVER 2008",
    CONTAINSSTRING(UPPER([modele_os_etendu]), "WINDOWS") && CONTAINSSTRING(UPPER([modele_os_etendu]), "2012"), "MICROSOFT-WINDOWS SERVER 2012",
    CONTAINSSTRING(UPPER([modele_os_etendu]), "WINDOWS") && CONTAINSSTRING(UPPER([modele_os_etendu]), "2016"), "MICROSOFT-WINDOWS SERVER 2016",
    CONTAINSSTRING(UPPER([modele_os_etendu]), "WINDOWS") && CONTAINSSTRING(UPPER([modele_os_etendu]), "2019"), "MICROSOFT-WINDOWS SERVER 2019",
    CONTAINSSTRING(UPPER([modele_os_etendu]), "WINDOWS") && CONTAINSSTRING(UPPER([modele_os_etendu]), "2022"), "MICROSOFT-WINDOWS SERVER 2022",
    CONTAINSSTRING(UPPER([modele_os_etendu]), "LINUX") && NOT(CONTAINSSTRING(UPPER([modele_os_etendu]), "RED HAT")) && NOT(CONTAINSSTRING(UPPER([modele_os_etendu]), "DEBIAN")) && NOT(CONTAINSSTRING(UPPER([modele_os_etendu]), "CENTOS")) && NOT(CONTAINSSTRING(UPPER([modele_os_etendu]), "UBUNTU")) && NOT(CONTAINSSTRING(UPPER([modele_os_etendu]), "SUSE")), "ND-LINUX",
    (CONTAINSSTRING(UPPER([modele_os_etendu]), "RED HAT") || CONTAINSSTRING(UPPER([modele_os_etendu]), "REDHAT")) && CONTAINSSTRING(UPPER([modele_os_etendu]), "5"), "REDHAT-ENTREPRISE LINUX 5",
    (CONTAINSSTRING(UPPER([modele_os_etendu]), "RED HAT") || CONTAINSSTRING(UPPER([modele_os_etendu]), "REDHAT")) && CONTAINSSTRING(UPPER([modele_os_etendu]), "6"), "REDHAT-ENTREPRISE LINUX 6",
    (CONTAINSSTRING(UPPER([modele_os_etendu]), "RED HAT") || CONTAINSSTRING(UPPER([modele_os_etendu]), "REDHAT")) && CONTAINSSTRING(UPPER([modele_os_etendu]), "7"), "REDHAT-ENTREPRISE LINUX 7",
    (CONTAINSSTRING(UPPER([modele_os_etendu]), "RED HAT") || CONTAINSSTRING(UPPER([modele_os_etendu]), "REDHAT")) && CONTAINSSTRING(UPPER([modele_os_etendu]), "8"), "REDHAT-ENTREPRISE LINUX 8",
    (CONTAINSSTRING(UPPER([modele_os_etendu]), "RED HAT") || CONTAINSSTRING(UPPER([modele_os_etendu]), "REDHAT")) && CONTAINSSTRING(UPPER([modele_os_etendu]), "9"), "REDHAT-ENTREPRISE LINUX 9",
    CONTAINSSTRING(UPPER([modele_os_etendu]), "SUSE") && CONTAINSSTRING(UPPER([modele_os_etendu]), "10"), "SUSE-LINUX ENTREPRISE 10",
    CONTAINSSTRING(UPPER([modele_os_etendu]), "SUSE") && CONTAINSSTRING(UPPER([modele_os_etendu]), "11"), "SUSE-LINUX ENTREPRISE 11",
    CONTAINSSTRING(UPPER([modele_os_etendu]), "SUSE") && CONTAINSSTRING(UPPER([modele_os_etendu]), "12"), "SUSE-LINUX ENTREPRISE 12",
    CONTAINSSTRING(UPPER([modele_os_etendu]), "ESXI") && CONTAINSSTRING(UPPER([modele_os_etendu]), "6"), "VMWARE-ESXI 6",
    CONTAINSSTRING(UPPER([modele_os_etendu]), "ESXI") && CONTAINSSTRING(UPPER([modele_os_etendu]), "7"), "VMWARE-ESXI 7",
    CONTAINSSTRING(UPPER([modele_os_etendu]), "PHOTON"), "VMWARE-PHOTON",
    UPPER([modele_os_etendu])  // Si aucune correspondance, on retourne la valeur originale en majuscules
)
```

Modifications apportées :
1. J'ai ajouté "REDHAT" comme alternative à "RED HAT" pour couvrir toutes les variations
2. J'ai ajouté des conditions pour les versions spécifiques qui pourraient être écrites sans espace
3. Pour les versions Windows, j'ai simplifié en cherchant à la fois "WINDOWS" et l'année
4. Plutôt que de mettre "AUTRE", le dernier cas renvoie simplement la valeur d'origine en majuscules si aucune correspondance n'est trouvée

Si vous remarquez d'autres cas particuliers à ajouter, n'hésitez pas à me le faire savoir.

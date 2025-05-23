Appops = 
VAR TextToSearch = [metis_app_support_group]
VAR Pos_DEMAT = IFERROR(SEARCH("CAGIP-BCR-APPOPS_DEMAT_PDT_APT", TextToSearch), 9999)
VAR Pos_WIFI = IFERROR(SEARCH("CAGIP-BCR-APPOPS_DISTRI_CLIENTS_WIFI", TextToSearch), 9999)
VAR Pos_PATRIMOINE = IFERROR(SEARCH("CAGIP-BCR-APPOPS_PATRIMOINE_CLIENTS_BANCAIRE", TextToSearch), 9999)
VAR Pos_REGALIEN = IFERROR(SEARCH("CAGIP-BCR-APPOPS_REGALIEN_MONETIQUE", TextToSearch), 9999)
VAR Pos_RH = IFERROR(SEARCH("CAGIP-BCR-APPOPS_RH_INTERNAT_COMCLI_SIINTERNE", TextToSearch), 9999)
VAR Pos_SDAS = IFERROR(SEARCH("CAGIP-BCR-APPOPS_SDAS", TextToSearch), 9999)
VAR Pos_CREDITS = IFERROR(SEARCH("CAGIP-BCR-APPOPS_CREDITS", TextToSearch), 9999)
VAR Pos_DATA = IFERROR(SEARCH("CAGIP-BCR-APPOPS_DATA", TextToSearch), 9999)
VAR Pos_DISTRI_COLLAB = IFERROR(SEARCH("CAGIP-BCR-APPOPS_DISTRI_COLLABORATEUR", TextToSearch), 9999)
VAR Pos_ECHANGES = IFERROR(SEARCH("CAGIP-BCR-APPOPS_ECHANGES_FLUX", TextToSearch), 9999)

VAR MinPosition = MIN(Pos_DEMAT, Pos_WIFI, Pos_PATRIMOINE, Pos_REGALIEN, Pos_RH, Pos_SDAS, Pos_CREDITS, Pos_DATA, Pos_DISTRI_COLLAB, Pos_ECHANGES)

RETURN 
SWITCH(
    TRUE(),
    MinPosition = Pos_DEMAT, "Démat., Poste de travail & Socles APT",
    MinPosition = Pos_WIFI, "Distribution Clients & Wifi",
    MinPosition = Pos_PATRIMOINE, "Patrimoine, Clients, Bancaire",
    MinPosition = Pos_REGALIEN, "Régalien & Monétique",
    MinPosition = Pos_RH, "RH, International, Com. Cli, SI interne",
    MinPosition = Pos_SDAS, "Socles, DEVOPS, Architecture Et SUPPORT",
    MinPosition = Pos_CREDITS, "Crédits",
    MinPosition = Pos_DATA, "Data",
    MinPosition = Pos_DISTRI_COLLAB, "Distribution Collaborateurs",
    MinPosition = Pos_ECHANGES, "Echanges et Flux",
    [metis_app_support_group] // Si aucun pattern trouvé
)

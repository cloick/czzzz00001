# Étapes détaillées pour créer un graphique en cascade de mouvements de serveurs

Voici les étapes précises pour créer le graphique en cascade qui montre les serveurs construits (en vert) et décommissionnés (en jaune) par système d'exploitation :

## 1. Préparation des mesures DAX

1. Dans Power BI Desktop, cliquez sur l'onglet "Modélisation"
2. Cliquez sur "Nouvelle mesure"
3. Créez la mesure pour les constructions :
   ```
   Constructions = 
   CALCULATE(
       COUNTROWS('VotreTable'),
       FILTER(
           'VotreTable',
           'VotreTable'[status_mouvement] = "construit"
       )
   )
   ```
4. Cliquez sur "Cocher" pour valider
5. Créez la mesure pour les décommissionnements :
   ```
   Decommissionnements = 
   -1 * CALCULATE(
       COUNTROWS('VotreTable'),
       FILTER(
           'VotreTable',
           'VotreTable'[status_mouvement] = "decommissionné"
       )
   )
   ```
6. Cliquez sur "Cocher" pour valider

## 2. Création du graphique en cascade

1. Dans le canevas du rapport, cliquez sur une zone vide
2. Dans le panneau Visualisations, cliquez sur l'icône "Graphique en cascade" (ressemble à un histogramme avec des barres suspendues)
3. Dans le panneau Champs de la visualisation :
   - Faites glisser `modele_os` dans le champ "Axe de catégorie"
   - Faites glisser `Constructions` dans le champ "Axe Y"
   - Faites glisser `Decommissionnements` dans le champ "Axe Y"

## 3. Personnalisation du graphique

1. Cliquez sur le graphique pour le sélectionner
2. Cliquez sur l'icône Format (pinceau) dans le panneau Visualisations
3. Développez la section "Couleurs de données"
4. Pour les valeurs positives :
   - Changez la couleur en vert (#4CAF50 ou similaire)
5. Pour les valeurs négatives :
   - Changez la couleur en jaune (#FFC107 ou similaire)
6. Développez la section "Étiquettes de données"
   - Activez les étiquettes de données pour afficher les chiffres sur les barres
7. Développez la section "Titre"
   - Changez le titre en "Mouvements de serveurs par OS"

## 4. Ajout d'une légende (optionnel)

1. Dans le panneau Format, développez la section "Légende"
2. Activez la légende
3. Positionnez-la en haut ou en bas du graphique

## 5. Autres personnalisations (optionnel)

1. Axe X :
   - Développez la section "Axe X" dans le panneau Format
   - Personnalisez les polices, couleurs, etc.
2. Grille :
   - Développez la section "Grille Y" 
   - Ajustez la visibilité et le style des lignes de la grille

## 6. Filtres (optionnel)

1. Dans le panneau Filtres :
   - Vous pouvez ajouter des filtres pour se concentrer sur certains systèmes d'exploitation ou périodes si vous avez une colonne de date

Le résultat final devrait ressembler à votre image de référence, avec des barres vertes pour les serveurs construits (positifs) et des barres jaunes pour les serveurs décommissionnés (négatifs), organisés par version du système d'exploitation.

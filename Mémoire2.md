# 4. DÉROULEMENT CHRONOLOGIQUE DE LA MISSION : CONFRONTATION OBJECTIFS-RÉALISATIONS

## 4.1 Analyse des verbatims clients : de l'automatisation basique à l'intelligence contextuelle

L'objectif initial pour ce projet était relativement simple : développer un système automatisé capable de catégoriser les retours clients des enquêtes IRC Flash pour réduire le temps d'analyse manuelle. Les attentes se limitaient à une classification basique des verbatims selon leur niveau de satisfaction et à l'identification des thèmes récurrents.

Les réalisations ont largement dépassé ces objectifs initiaux. Au-delà de la classification attendue, le système développé a intégré des fonctionnalités avancées de clustering hiérarchique et de topic modeling utilisant des techniques LDA et BERTopic. L'évolution vers l'intégration de l'intelligence artificielle générative a permis d'enrichir l'analyse avec des capacités de résumé automatique et d'extraction d'entités. Cette expansion fonctionnelle a généré une réduction de 70% du temps d'analyse et une identification automatique de 95% des problèmes récurrents.

L'écart principal résidait dans la découverte d'un potentiel technique et organisationnel beaucoup plus important que prévu. Cette expérience a également révélé des enjeux de conformité majeurs liés au cadre normatif IA du groupe, particulièrement concernant l'identification individuelle des auteurs de verbatims. Cette problématique non anticipée a nécessité une adaptation des fonctionnalités et un dialogue approfondi avec les équipes de gouvernance, illustrant la complexité de l'intégration de l'IA dans un environnement réglementé.

## 4.2 Fiabilisation des tickets incidents : du clustering simple à l'architecture d'ensemble

Le projet de classification et clustering des tickets ServiceNow était initialement conçu comme une solution technique ponctuelle pour améliorer l'identification des causes racines d'incidents. Les objectifs se limitaient à développer un modèle de classification capable de prédire les causes selon la taxonomie existante, avec une approche d'apprentissage supervisé classique.

La réalisation effective a évolué vers une architecture d'ensemble sophistiquée combinant apprentissage non supervisé et inférence directe via des modèles de langage open source. Cette approche hybride a permis d'atteindre des performances supérieures aux attentes initiales, mais a également révélé des défis techniques et organisationnels non anticipés. L'utilisation de modèles open source de Hugging Face a soulevé des questions de sécurité nécessitant une validation préalable, l'absence de politique claire sur l'usage de ces technologies ayant conduit à une mise en stand-by temporaire du projet.

Cet écart illustre parfaitement les enjeux de l'innovation technologique dans un contexte d'entreprise traditionnelle, où les contraintes de sécurité et les processus de validation peuvent influencer significativement les choix techniques. L'expérience a souligné l'importance d'anticiper ces aspects organisationnels dès la phase de conception pour éviter les blocages en phase d'industrialisation.

## 4.3 Système prédictif ServiceNow : de l'aide à la décision à l'intégration native

L'objectif initial de ce projet était de développer un système d'aide à la décision pour anticiper les risques d'échec des changements informatiques. Les attentes portaient sur un modèle de classification binaire capable de fournir une probabilité d'échec avec quelques recommandations contextuelles, destiné à être utilisé en support des processus existants.

La réalisation a évolué vers un système intégré nativement dans les workflows ServiceNow, combinant prédiction de risque, recherche de similarité et analyse contextuelle approfondie. L'architecture hybride développée, intégrant régression logistique et clustering K-Means, a permis d'atteindre des performances remarquables avec 87% de précision et 82% de recall. L'intégration native dans ServiceNow via une API Flask et le déploiement sur la plateforme Dataiku ont transformé l'outil en solution opérationnelle adoptée par 85% des équipes de changement.

L'écart le plus significatif résidait dans l'ampleur de l'adoption organisationnelle. La formation de 120 collaborateurs et l'intégration dans les processus quotidiens ont généré des gains opérationnels supérieurs aux prévisions, avec une réduction de 25% des incidents post-changement et des économies estimées à 150k€ annuels. Cette expérience a démontré l'importance de l'accompagnement humain dans la réussite des projets d'intelligence artificielle.

## 4.4 Solutions transverses : d'outils spécialisés à une approche systémique

Les projets AppOps 360 et de modernisation des outils d'obsolescence étaient initialement conçus comme des développements techniques spécialisés répondant à des besoins ponctuels. Pour AppOps 360, l'objectif était de créer quelques tableaux de bord pour améliorer la visibilité sur les enjeux transverses. Pour l'outillage d'obsolescence, il s'agissait de moderniser les interfaces de reporting existantes.

La réalisation a évolué vers une approche systémique de transformation des processus opérationnels. AppOps 360 s'est développé en une plateforme intégrée couvrant sept domaines fonctionnels avec une architecture data moderne permettant une vision à 360° des enjeux du cluster. La refonte de l'outillage d'obsolescence a dépassé la simple modernisation pour inclure la conception d'un datawarehouse complet et une transformation des processus de gestion.

Ces évolutions témoignent d'une capacité d'adaptation remarquable de l'organisation face aux opportunités technologiques. L'écart entre objectifs initiaux et réalisations finales illustre la dynamique d'innovation qui s'est développée au sein du cluster, où chaque projet technique a révélé des potentiels d'amélioration plus larges que prévu initialement.

## 4.5 Facteurs explicatifs des écarts constatés

L'analyse transversale de ces projets révèle plusieurs facteurs récurrents qui expliquent les écarts observés entre objectifs initiaux et réalisations effectives. Le contexte organisationnel s'est révélé plus favorable à l'innovation que prévu, avec une ouverture réelle aux nouvelles technologies et un soutien du management pour l'expérimentation. Cette dynamique positive a créé un cercle vertueux où chaque succès technique a généré de nouvelles opportunités d'expansion.

La capacité d'adaptation technique et organisationnelle a constitué un facteur déterminant dans le dépassement des objectifs. L'approche agile adoptée a permis de faire évoluer les solutions en fonction des besoins exprimés et des opportunités identifiées, transformant des projets techniques ponctuels en véritables leviers de transformation organisationnelle.

Ces écarts illustrent parfaitement les mécanismes de transformation progressive d'une organisation traditionnelle vers l'adoption de l'intelligence artificielle. Ils témoignent de l'importance de maintenir une approche flexible et d'accompagner les dimensions techniques par un investissement approprié dans les aspects humains et organisationnels de l'innovation.

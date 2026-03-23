
Salut,

Merci pour ton retour. Voici les éléments que je peux te communiquer sur notre usage dans le cadre d'un POC RAG sur 1 à 3 mois.

─────────────────────────────
INFORMATIONS PRIORITAIRES
─────────────────────────────

📌 Modèles utilisés à date (4 modèles)

1. LLM (texte) — génération de réponses à partir des chunks récupérés
2. VLM (vision-langage) — analyse de documents contenant des images, tableaux, schémas
3. OCR — extraction de texte à partir de documents scannés ou PDFs non natifs
4. Embedding model — vectorisation des documents et des requêtes pour la recherche sémantique

📌 Estimation de la consommation mensuelle en tokens

S'agissant d'un POC en phase exploratoire, il m'est impossible de fournir des estimations fiables de consommation mensuelle en tokens. Le volume dépendra directement des résultats des premières itérations : qualité du chunking, fréquence des requêtes de test, taille des contextes, etc.

Je préfère être transparent sur ce point plutôt que de te communiquer des chiffres non représentatifs. Des estimations réalistes pourront être fournies à l'issue de la première phase d'indexation et de tests, soit environ 3 à 4 semaines après le démarrage.

📌 Modèles prévus d'ici 2027

Il m'est difficile de me projeter avec fiabilité sur ce point. La convergence vers des modèles multimodaux unifiés est déjà en cours — Mistral Small 4 (sorti le 16 mars 2026) unifie par exemple vision, OCR, raisonnement et coding en un seul modèle, et des modèles comme Qwen3-VL ou Phi-4-reasoning-vision vont dans le même sens. D'ici 2027, il est donc tout à fait probable qu'un seul modèle remplace les 3 ou 4 que j'utilise aujourd'hui.

Te communiquer des prévisions figées sur le nombre de modèles serait contre-productif pour nous deux.

─────────────────────────────
INFORMATIONS COMPLÉMENTAIRES
─────────────────────────────

• Répartition tokens entrée/sortie : ~70% input / 30% output sur le LLM (ordre de grandeur)
• Pics de charge : phase d'indexation initiale en batch, puis pics ponctuels lors des sessions de test
• Latence : non critique en phase POC, à surveiller pour une éventuelle mise en production

Dispo si tu veux qu'on en discute de vive voix.

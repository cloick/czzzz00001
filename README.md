Objet : Blocage technique projet [NOM_PROJET] - Clarification politique modèles LLM/HuggingFace nécessaire

Bonjour [Équipe responsable Dataiku],

Suite à mes relances des 3 dernières semaines restées sans réponse, je sollicite votre aide pour débloquer un projet stratégique actuellement à l'arrêt.

Notre projet nécessite la mise en œuvre de fonctionnalités de traitement du langage naturel en français, notamment du clustering de textes et de la classification zero-shot/few-shot. Pour répondre aux exigences qualité, nous avons identifié le besoin d'utiliser des modèles spécialisés français comme CamemBERT pour les embeddings et des petits modèles LLM open source pour la classification. Ces fonctionnalités sont essentielles pour l'analyse sémantique de nos documents métier.

Cependant, nous rencontrons plusieurs blocages techniques majeurs. Bien que les recettes LLM soient visibles dans l'interface Dataiku, elles affichent systématiquement "nothing to select" car aucune connexion LLM n'est configurée sur notre instance. Nous n'avons pas accès à la licence Advanced LLM Mesh et aucune connexion vers OpenAI, Azure ou HuggingFace n'est disponible.

De plus, le proxy d'entreprise bloque l'accès à HuggingFace Hub, rendant impossible l'import direct des modèles via les API Python. La seule solution technique restante serait de télécharger manuellement les modèles CamemBERT et autres depuis HuggingFace pour les importer localement dans Dataiku. Cependant, cette approche soulève des questions de sécurité importantes que je ne souhaite pas entreprendre sans validation officielle.

J'ai donc besoin de clarifications urgentes sur plusieurs points. Quelle est la politique officielle de l'entreprise concernant l'utilisation de modèles HuggingFace ou autres modèles open source externes ? Existe-t-il un processus de validation pour l'import de modèles externes, et si oui, comment l'initier ? Serait-il possible d'obtenir un upgrade vers Advanced LLM Mesh pour notre projet, ou alternativement, de configurer des connexions LLM basiques ? Dans le cas où aucune de ces solutions ne serait possible, comment obtenir une dérogation sécuritaire pour l'import manuel de modèles validés ?

L'impact business devient significatif avec déjà 3 semaines de retard accumulé. Sans clarification rapide, nous devrons soit reporter le projet, soit nous orienter vers des solutions dégradées qui ne répondront pas aux exigences qualité initialement définies.

Je sollicite donc une réponse sous 48h sur la politique concernant les modèles externes, ainsi qu'un point technique avec l'équipe infrastructure si nécessaire. Je reste disponible pour tout complément d'information ou démonstration des besoins techniques.

Merci de votre compréhension et retour rapide.

Cordialement,
[Votre nom]

En copie : [Manager 1], [Manager 2]

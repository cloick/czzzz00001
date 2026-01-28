
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qualité et curation - Prototype v4</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .tab-button.active {
            background-color: #3b82f6;
            color: white;
        }
        .accordion-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }
        .accordion-content.open {
            max-height: 2000px;
        }
    </style>
</head>
<body class="bg-gray-50">
    <!-- Header -->
    <header class="bg-white shadow-sm border-b">
        <div class="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
            <div class="flex items-center gap-4">
                <button class="text-gray-600 hover:text-gray-900 flex items-center gap-2">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/>
                    </svg>
                    Retour
                </button>
                <h1 class="text-2xl font-semibold text-gray-900">Qualité et curation</h1>
            </div>
            <button class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2">
                📊 Monitoring & Performance
            </button>
        </div>
    </header>

    <!-- Vue d'ensemble -->
    <div class="max-w-7xl mx-auto px-6 py-6">
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <!-- Card 1 -->
            <div class="bg-white rounded-lg shadow p-6 border-l-4 border-red-500">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-gray-600 text-sm">Signalés</p>
                        <p class="text-3xl font-bold text-red-600">2</p>
                    </div>
                    <div class="text-4xl">🔴</div>
                </div>
            </div>

            <!-- Card 2 -->
            <div class="bg-white rounded-lg shadow p-6 border-l-4 border-orange-500">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-gray-600 text-sm">Obsolètes</p>
                        <p class="text-3xl font-bold text-orange-600">45</p>
                    </div>
                    <div class="text-4xl">⚠️</div>
                </div>
            </div>

            <!-- Card 3 -->
            <div class="bg-white rounded-lg shadow p-6 border-l-4 border-blue-500">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-gray-600 text-sm">Orphelines</p>
                        <p class="text-3xl font-bold text-blue-600">15</p>
                    </div>
                    <div class="text-4xl">🔗</div>
                </div>
            </div>

            <!-- Card 4 -->
            <div class="bg-white rounded-lg shadow p-6 border-l-4 border-purple-500">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-gray-600 text-sm">Doublons</p>
                        <p class="text-3xl font-bold text-purple-600">8</p>
                    </div>
                    <div class="text-4xl">📋</div>
                </div>
            </div>
        </div>

        <!-- Tabs -->
        <div class="bg-white rounded-lg shadow">
            <div class="border-b border-gray-200">
                <div class="flex">
                    <button class="tab-button active px-6 py-3 font-medium text-gray-700 border-b-2 border-blue-500" onclick="switchTab('signales')">
                        Signalés 🔴 (2)
                    </button>
                    <button class="tab-button px-6 py-3 font-medium text-gray-500 hover:text-gray-700" onclick="switchTab('problemes')">
                        Problèmes détectés ⚠️
                    </button>
                    <button class="tab-button px-6 py-3 font-medium text-gray-500 hover:text-gray-700" onclick="switchTab('suggestions')">
                        Suggestions ✨
                    </button>
                </div>
            </div>

            <!-- Tab Content: Signalés -->
            <div id="signales" class="tab-content active p-6">
                <div class="flex gap-4 mb-6">
                    <input type="text" placeholder="🔍 Rechercher..." class="flex-1 px-4 py-2 border rounded-lg">
                    <select class="px-4 py-2 border rounded-lg">
                        <option>Filtrer: Toutes</option>
                        <option>Non traitées</option>
                        <option>Traitées</option>
                        <option>Ignorées</option>
                    </select>
                    <select class="px-4 py-2 border rounded-lg">
                        <option>Trier par: Date</option>
                        <option>Source</option>
                    </select>
                </div>

                <!-- Exemple de signalement 1 -->
                <div class="border rounded-lg p-5 mb-4 bg-red-50 border-red-200">
                    <div class="mb-3">
                        <p class="font-semibold text-gray-900 mb-1">Question: "Comment créer un bucket S3 avec chiffrement ?"</p>
                        <p class="text-sm text-gray-600">Page source: <span class="text-blue-600 underline">Procédure AWS v2</span></p>
                    </div>
                    
                    <div class="flex gap-4 text-sm text-gray-600 mb-4">
                        <span>👤 Signalé par: marie.dupont@example.com</span>
                        <span>📅 Date: 24/01/2026 14:32</span>
                    </div>

                    <div class="mb-4">
                        <p class="text-sm text-gray-700 mb-2"><strong>Raison:</strong> Réponse obsolète - ne mentionne pas le chiffrement par défaut maintenant activé automatiquement</p>
                    </div>

                    <div class="border-t pt-4">
                        <p class="text-sm font-medium text-gray-700 mb-3">Actions:</p>
                        <div class="flex flex-wrap gap-2 mb-4">
                            <button class="px-4 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 flex items-center gap-2">
                                🗑️ Supprimer de Confluence
                            </button>
                            <button class="px-4 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 flex items-center gap-2">
                                ✏️ Modifier sur Confluence
                            </button>
                        </div>
                        <div class="flex gap-2">
                            <button class="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 font-medium">
                                ✅ Traiter
                            </button>
                            <button class="px-6 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300">
                                ❌ Ignorer
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Exemple de signalement 2 - Cas contradiction -->
                <div class="border rounded-lg p-5 mb-4 bg-orange-50 border-orange-200">
                    <div class="mb-3">
                        <p class="font-semibold text-gray-900 mb-1">Question: "Est-ce que le chiffrement S3 est activé par défaut ?"</p>
                        <p class="text-sm text-gray-600">Page source: <span class="text-blue-600 underline">Guide AWS 2021</span></p>
                    </div>
                    
                    <div class="flex gap-4 text-sm text-gray-600 mb-4">
                        <span>👤 Signalé par: john.smith@example.com</span>
                        <span>📅 Date: 23/01/2026 09:15</span>
                    </div>

                    <div class="mb-4 bg-white border border-orange-300 rounded p-3">
                        <p class="text-sm font-medium text-gray-900 mb-2">⚠️ Plusieurs réponses contradictoires ont été identifiées</p>
                        <p class="text-sm text-gray-700 mb-2"><strong>Réponse (signalée comme incorrecte):</strong></p>
                        <p class="text-sm text-gray-700 italic">"Non, le chiffrement S3 n'est pas activé par défaut. Vous devez explicitement activer SSE-S3 ou SSE-KMS lors de la création du bucket ou via la configuration après création."</p>
                    </div>

                    <div class="mb-4">
                        <p class="text-sm text-gray-700"><strong>Raison:</strong> Information obsolète - AWS active automatiquement le chiffrement SSE-S3 depuis janvier 2023</p>
                    </div>

                    <div class="border-t pt-4">
                        <p class="text-sm font-medium text-gray-700 mb-3">Actions:</p>
                        <div class="flex flex-wrap gap-2 mb-4">
                            <button class="px-4 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 flex items-center gap-2">
                                🗑️ Supprimer de Confluence
                            </button>
                            <button class="px-4 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 flex items-center gap-2">
                                ✏️ Modifier sur Confluence
                            </button>
                        </div>
                        <div class="flex gap-2">
                            <button class="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 font-medium">
                                ✅ Traiter
                            </button>
                            <button class="px-6 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300">
                                ❌ Ignorer
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Tab Content: Problèmes -->
            <div id="problemes" class="tab-content p-6">
                <!-- Exemple d'item problème avec actions complètes -->
                <div class="mb-4">
                    <button class="accordion-trigger w-full flex items-center justify-between p-4 bg-orange-50 border border-orange-200 rounded-lg hover:bg-orange-100" onclick="toggleAccordion('obsoletes')">
                        <span class="font-semibold text-gray-900">Pages obsolètes 🕰️ (45)</span>
                        <svg class="w-5 h-5 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                        </svg>
                    </button>
                    <div id="obsoletes" class="accordion-content border-x border-b rounded-b-lg">
                        <div class="p-4 space-y-3">
                            <div class="border rounded-lg p-4 bg-white">
                                <div class="mb-3">
                                    <p class="font-semibold text-gray-900">Procédure migration 2020</p>
                                    <p class="text-sm text-gray-600">Modifié il y a 4 ans • 0 consultations/an</p>
                                </div>
                                
                                <div class="border-t pt-3 mt-3">
                                    <p class="text-sm font-medium text-gray-700 mb-3">Actions disponibles:</p>
                                    <div class="flex flex-wrap gap-2">
                                        <button class="px-4 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 flex items-center gap-2">
                                            🗑️ Supprimer de Confluence
                                        </button>
                                        <button class="px-4 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 flex items-center gap-2">
                                            ✏️ Modifier sur Confluence
                                        </button>
                                        <div class="flex items-center gap-2">
                                            <span class="text-sm text-gray-600">🏷️ Ajouter étiquette:</span>
                                            <select class="px-3 py-2 border rounded-lg text-sm">
                                                <option>Sélectionner...</option>
                                                <option>🚨 Obsolète</option>
                                                <option>⚠️ À vérifier</option>
                                                <option>📝 Incomplet</option>
                                            </select>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Orphelines -->
                <div class="mb-4">
                    <button class="accordion-trigger w-full flex items-center justify-between p-4 bg-blue-50 border border-blue-200 rounded-lg hover:bg-blue-100" onclick="toggleAccordion('orphelines')">
                        <span class="font-semibold text-gray-900">Orphelines 🔗 (15)</span>
                        <svg class="w-5 h-5 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                        </svg>
                    </button>
                    <div id="orphelines" class="accordion-content border-x border-b rounded-b-lg">
                        <div class="p-4 space-y-2">
                            <div class="flex items-center justify-between p-3 bg-white border rounded">
                                <div>
                                    <p class="font-medium">Guide installation Node 12</p>
                                    <p class="text-sm text-gray-600">Aucun lien entrant</p>
                                </div>
                                <div class="flex gap-2">
                                    <button class="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm">Lier à...</button>
                                    <button class="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 text-sm">Archiver</button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Doublons -->
                <div class="mb-4">
                    <button class="accordion-trigger w-full flex items-center justify-between p-4 bg-purple-50 border border-purple-200 rounded-lg hover:bg-purple-100" onclick="toggleAccordion('doublons')">
                        <span class="font-semibold text-gray-900">Doublons 📋 (8 groupes)</span>
                        <svg class="w-5 h-5 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                        </svg>
                    </button>
                    <div id="doublons" class="accordion-content border-x border-b rounded-b-lg">
                        <div class="p-4">
                            <div class="bg-white border rounded-lg p-4 mb-3">
                                <div class="flex justify-between items-start mb-3">
                                    <p class="font-medium text-gray-900">Groupe 1: Déploiement production</p>
                                    <span class="px-2 py-1 bg-purple-100 text-purple-700 rounded text-sm">Similarité 94%</span>
                                </div>
                                <div class="space-y-2 mb-3 ml-4">
                                    <div class="flex items-center gap-2">
                                        <span class="text-gray-400">├─</span>
                                        <span class="text-blue-600">Déploiement production</span>
                                    </div>
                                    <div class="flex items-center gap-2">
                                        <span class="text-gray-400">├─</span>
                                        <span class="text-blue-600">Process de déploiement prod</span>
                                    </div>
                                    <div class="flex items-center gap-2">
                                        <span class="text-gray-400">└─</span>
                                        <span class="text-blue-600">Guide deploy prod</span>
                                    </div>
                                </div>
                                <div class="flex gap-2">
                                    <button class="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 text-sm">Fusionner</button>
                                    <button class="px-4 py-2 bg-blue-100 text-blue-700 rounded hover:bg-blue-200 text-sm">Comparer</button>
                                    <button class="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300 text-sm">Garder séparé</button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Contradictions -->
                <div class="mb-4">
                    <button class="accordion-trigger w-full flex items-center justify-between p-4 bg-yellow-50 border border-yellow-200 rounded-lg hover:bg-yellow-100" onclick="toggleAccordion('contradictions')">
                        <span class="font-semibold text-gray-900">Contradictions 🆚 (4)</span>
                        <svg class="w-5 h-5 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                        </svg>
                    </button>
                    <div id="contradictions" class="accordion-content border-x border-b rounded-b-lg">
                        <div class="p-4 space-y-2">
                            <div class="bg-white border rounded-lg p-4">
                                <p class="font-medium mb-2">Politique sécurité vs Guidelines sécurité</p>
                                <p class="text-sm text-gray-600 mb-3">→ Règle MFA différente (obligatoire vs recommandé)</p>
                                <div class="flex gap-2">
                                    <button class="px-3 py-1 bg-yellow-600 text-white rounded hover:bg-yellow-700 text-sm">Voir conflit</button>
                                    <button class="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 text-sm">Arbitrer</button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Trous de documentation -->
                <div class="mb-4">
                    <button class="accordion-trigger w-full flex items-center justify-between p-4 bg-green-50 border border-green-200 rounded-lg hover:bg-green-100" onclick="toggleAccordion('trous')">
                        <span class="font-semibold text-gray-900">Trous de documentation 🕳️ (12)</span>
                        <svg class="w-5 h-5 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                        </svg>
                    </button>
                    <div id="trous" class="accordion-content border-x border-b rounded-b-lg">
                        <div class="p-4 space-y-2">
                            <div class="flex items-center justify-between p-3 bg-white border rounded">
                                <div>
                                    <p class="font-medium">Kubernetes</p>
                                    <p class="text-sm text-gray-600">Mentionné 45 fois, pas de page dédiée</p>
                                </div>
                                <div class="flex gap-2">
                                    <button class="px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700 text-sm">Créer page</button>
                                    <button class="px-3 py-1 bg-blue-100 text-blue-700 rounded hover:bg-blue-200 text-sm">Assigner</button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Tab Content: Suggestions -->
            <div id="suggestions" class="tab-content p-6">
                <!-- Propositions de fusion -->
                <div class="mb-6">
                    <button class="accordion-trigger w-full flex items-center justify-between p-4 bg-blue-50 border border-blue-200 rounded-lg hover:bg-blue-100" onclick="toggleAccordion('fusions')">
                        <span class="font-semibold text-gray-900">Propositions de fusion 🔗 (5 groupes)</span>
                        <svg class="w-5 h-5 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                        </svg>
                    </button>
                    <div id="fusions" class="accordion-content border-x border-b rounded-b-lg">
                        <div class="p-4">
                            <div class="bg-white border-2 border-blue-200 rounded-lg p-5">
                                <div class="flex justify-between items-start mb-4">
                                    <h3 class="font-semibold text-lg">Groupe 1: Guide Docker (3 pages)</h3>
                                    <span class="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium">Confiance: Haute</span>
                                </div>
                                
                                <div class="mb-4">
                                    <p class="text-sm font-medium text-gray-700 mb-2">Pages à fusionner:</p>
                                    <ul class="space-y-1 ml-4">
                                        <li class="text-blue-600">• Docker - Installation</li>
                                        <li class="text-blue-600">• Utiliser Docker</li>
                                        <li class="text-blue-600">• Docker best practices</li>
                                    </ul>
                                </div>

                                <div class="bg-gray-50 p-3 rounded mb-4">
                                    <p class="text-sm text-gray-700"><strong>Raison:</strong> Contenu redondant à 87%, structure similaire, mêmes exemples</p>
                                </div>

                                <div class="flex gap-2">
                                    <button class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 font-medium">
                                        ✅ Fusionner
                                    </button>
                                    <button class="px-4 py-2 bg-blue-100 text-blue-700 rounded hover:bg-blue-200">
                                        👁️ Voir proposition
                                    </button>
                                    <button class="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300">
                                        ✏️ Modifier
                                    </button>
                                    <button class="px-4 py-2 bg-red-100 text-red-700 rounded hover:bg-red-200">
                                        ❌ Rejeter
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Propositions de création -->
                <div class="mb-6">
                    <button class="accordion-trigger w-full flex items-center justify-between p-4 bg-green-50 border border-green-200 rounded-lg hover:bg-green-100" onclick="toggleAccordion('creations')">
                        <span class="font-semibold text-gray-900">Propositions de création ➕ (8)</span>
                        <svg class="w-5 h-5 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                        </svg>
                    </button>
                    <div id="creations" class="accordion-content border-x border-b rounded-b-lg">
                        <div class="p-4 space-y-3">
                            <div class="bg-white border rounded-lg p-4">
                                <div class="flex justify-between items-start mb-3">
                                    <h3 class="font-semibold">Authentification OAuth2</h3>
                                    <span class="px-3 py-1 bg-red-100 text-red-700 rounded-full text-sm">Importance: Haute</span>
                                </div>
                                <p class="text-sm text-gray-600 mb-3">Mentionné dans: 12 pages • Recherché: 45 fois ce mois</p>
                                <div class="flex gap-2">
                                    <button class="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700">📝 Créer maintenant</button>
                                    <button class="px-4 py-2 bg-blue-100 text-blue-700 rounded hover:bg-blue-200">📅 Planifier</button>
                                    <button class="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300">Ignorer</button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Suggestions d'étiquettes -->
                <div class="mb-6">
                    <button class="accordion-trigger w-full flex items-center justify-between p-4 bg-pink-50 border border-pink-200 rounded-lg hover:bg-pink-100" onclick="toggleAccordion('etiquettes')">
                        <span class="font-semibold text-gray-900">Suggestions d'étiquettes 🏷️ (23 pages)</span>
                        <svg class="w-5 h-5 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                        </svg>
                    </button>
                    <div id="etiquettes" class="accordion-content border-x border-b rounded-b-lg">
                        <div class="p-4">
                            <div class="bg-white border rounded-lg p-4 mb-3">
                                <div class="flex justify-between items-start mb-3">
                                    <p class="font-medium text-gray-900">Pages sans étiquettes détectées</p>
                                    <span class="px-3 py-1 bg-pink-100 text-pink-700 rounded-full text-sm">23 pages</span>
                                </div>
                                <div class="space-y-3">
                                    <div class="bg-gray-50 p-3 rounded">
                                        <p class="font-medium text-sm mb-2">Page: "Configuration Kubernetes Production"</p>
                                        <p class="text-sm text-gray-600 mb-2">Étiquettes suggérées:</p>
                                        <div class="flex flex-wrap gap-2 mb-2">
                                            <span class="px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs">Infrastructure</span>
                                            <span class="px-2 py-1 bg-green-100 text-green-700 rounded text-xs">Kubernetes</span>
                                            <span class="px-2 py-1 bg-purple-100 text-purple-700 rounded text-xs">Production</span>
                                        </div>
                                        <button class="px-3 py-1 bg-pink-600 text-white rounded hover:bg-pink-700 text-sm">Appliquer</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function switchTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab-button').forEach(btn => {
                btn.classList.remove('active');
            });
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }

        function toggleAccordion(id) {
            const content = document.getElementById(id);
            const trigger = content.previousElementSibling;
            const arrow = trigger.querySelector('svg');
            
            content.classList.toggle('open');
            arrow.classList.toggle('rotate-180');
        }
    </script>
</body>
</html>



--------------------------------------------------------------------------------------------------------------------------------------------------------------


<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Monitoring & Performance - Prototype</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .tab-button.active {
            background-color: #3b82f6;
            color: white;
        }
        .metric-card {
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-2px);
        }
    </style>
</head>
<body class="bg-gray-50">
    <!-- Header -->
    <header class="bg-white shadow-sm border-b">
        <div class="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
            <div class="flex items-center gap-4">
                <button class="text-gray-600 hover:text-gray-900 flex items-center gap-2">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/>
                    </svg>
                    Retour
                </button>
                <h1 class="text-2xl font-semibold text-gray-900">Monitoring & Performance</h1>
            </div>
            <div class="flex gap-2">
                <select class="px-3 py-2 border rounded-lg text-sm">
                    <option>Dernières 24h</option>
                    <option>7 derniers jours</option>
                    <option>30 derniers jours</option>
                    <option>Personnalisé</option>
                </select>
                <button class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                    📥 Exporter rapport
                </button>
            </div>
        </div>
    </header>

    <!-- Vue d'ensemble - KPIs Principaux -->
    <div class="max-w-7xl mx-auto px-6 py-6">
        <!-- Score Global -->
        <div class="bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg shadow-lg p-6 mb-6 text-white">
            <div class="flex items-center justify-between">
                <div>
                    <p class="text-blue-100 text-sm font-medium mb-1">Score Global RAGAS</p>
                    <p class="text-5xl font-bold">87.5%</p>
                    <p class="text-blue-100 text-sm mt-2">
                        <span class="inline-flex items-center">
                            <svg class="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                                <path fill-rule="evenodd" d="M5.293 9.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 7.414V15a1 1 0 11-2 0V7.414L6.707 9.707a1 1 0 01-1.414 0z"/>
                            </svg>
                            +3.2% vs semaine dernière
                        </span>
                    </p>
                </div>
                <div class="text-right">
                    <p class="text-blue-100 text-sm">Basé sur</p>
                    <p class="text-3xl font-bold">2,847</p>
                    <p class="text-blue-100 text-sm">requêtes analysées</p>
                </div>
            </div>
        </div>

        <!-- Métriques Critiques -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <!-- Faithfulness -->
            <div class="bg-white rounded-lg shadow p-5 border-l-4 border-green-500 metric-card">
                <div class="flex items-center justify-between mb-2">
                    <p class="text-gray-600 text-sm font-medium">Faithfulness 🔥</p>
                    <span class="px-2 py-1 bg-green-100 text-green-700 rounded text-xs font-semibold">Excellent</span>
                </div>
                <p class="text-3xl font-bold text-gray-900">92.3%</p>
                <p class="text-xs text-gray-500 mt-1">Anti-hallucination</p>
                <div class="mt-2 flex items-center text-green-600 text-xs">
                    <svg class="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M5.293 9.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 7.414V15a1 1 0 11-2 0V7.414L6.707 9.707a1 1 0 01-1.414 0z"/>
                    </svg>
                    +1.2% cette semaine
                </div>
            </div>

            <!-- Answer Relevance -->
            <div class="bg-white rounded-lg shadow p-5 border-l-4 border-blue-500 metric-card">
                <div class="flex items-center justify-between mb-2">
                    <p class="text-gray-600 text-sm font-medium">Answer Relevance</p>
                    <span class="px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs font-semibold">Bon</span>
                </div>
                <p class="text-3xl font-bold text-gray-900">88.7%</p>
                <p class="text-xs text-gray-500 mt-1">Pertinence réponses</p>
                <div class="mt-2 flex items-center text-blue-600 text-xs">
                    <svg class="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M5.293 9.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 7.414V15a1 1 0 11-2 0V7.414L6.707 9.707a1 1 0 01-1.414 0z"/>
                    </svg>
                    +0.8% cette semaine
                </div>
            </div>

            <!-- Latency P95 -->
            <div class="bg-white rounded-lg shadow p-5 border-l-4 border-yellow-500 metric-card">
                <div class="flex items-center justify-between mb-2">
                    <p class="text-gray-600 text-sm font-medium">Latency P95</p>
                    <span class="px-2 py-1 bg-yellow-100 text-yellow-700 rounded text-xs font-semibold">À surveiller</span>
                </div>
                <p class="text-3xl font-bold text-gray-900">2.8s</p>
                <p class="text-xs text-gray-500 mt-1">Temps de réponse</p>
                <div class="mt-2 flex items-center text-red-600 text-xs">
                    <svg class="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M14.707 10.293a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 111.414-1.414L9 12.586V5a1 1 0 012 0v7.586l2.293-2.293a1 1 0 011.414 0z"/>
                    </svg>
                    +0.3s cette semaine
                </div>
            </div>

            <!-- Satisfaction -->
            <div class="bg-white rounded-lg shadow p-5 border-l-4 border-purple-500 metric-card">
                <div class="flex items-center justify-between mb-2">
                    <p class="text-gray-600 text-sm font-medium">Satisfaction</p>
                    <span class="px-2 py-1 bg-purple-100 text-purple-700 rounded text-xs font-semibold">Excellent</span>
                </div>
                <p class="text-3xl font-bold text-gray-900">91.2%</p>
                <p class="text-xs text-gray-500 mt-1">Feedbacks positifs</p>
                <div class="mt-2 flex items-center text-purple-600 text-xs">
                    <svg class="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M5.293 9.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 7.414V15a1 1 0 11-2 0V7.414L6.707 9.707a1 1 0 01-1.414 0z"/>
                    </svg>
                    +2.1% cette semaine
                </div>
            </div>
        </div>

        <!-- Tabs -->
        <div class="bg-white rounded-lg shadow mb-6">
            <div class="border-b border-gray-200">
                <div class="flex">
                    <button class="tab-button active px-6 py-3 font-medium text-gray-700 border-b-2 border-blue-500" onclick="switchTab('retrieval')">
                        Retrieval 📥
                    </button>
                    <button class="tab-button px-6 py-3 font-medium text-gray-500 hover:text-gray-700" onclick="switchTab('context')">
                        Context 📄
                    </button>
                    <button class="tab-button px-6 py-3 font-medium text-gray-500 hover:text-gray-700" onclick="switchTab('generation')">
                        Generation 💬
                    </button>
                    <button class="tab-button px-6 py-3 font-medium text-gray-500 hover:text-gray-700" onclick="switchTab('performance')">
                        Performance ⚡
                    </button>
                </div>
            </div>

            <!-- Tab Content: Retrieval -->
            <div id="retrieval" class="tab-content active p-6">
                <h3 class="text-lg font-semibold text-gray-900 mb-4">Métriques de Récupération (Vector Search)</h3>
                
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <div class="bg-gray-50 rounded-lg p-4">
                        <p class="text-sm text-gray-600 mb-1">Precision@5</p>
                        <p class="text-2xl font-bold text-gray-900">84.5%</p>
                        <div class="w-full bg-gray-200 rounded-full h-2 mt-2">
                            <div class="bg-blue-600 h-2 rounded-full" style="width: 84.5%"></div>
                        </div>
                    </div>
                    <div class="bg-gray-50 rounded-lg p-4">
                        <p class="text-sm text-gray-600 mb-1">Recall@10</p>
                        <p class="text-2xl font-bold text-gray-900">76.2%</p>
                        <div class="w-full bg-gray-200 rounded-full h-2 mt-2">
                            <div class="bg-green-600 h-2 rounded-full" style="width: 76.2%"></div>
                        </div>
                    </div>
                    <div class="bg-gray-50 rounded-lg p-4">
                        <p class="text-sm text-gray-600 mb-1">F1-Score</p>
                        <p class="text-2xl font-bold text-gray-900">80.1%</p>
                        <div class="w-full bg-gray-200 rounded-full h-2 mt-2">
                            <div class="bg-purple-600 h-2 rounded-full" style="width: 80.1%"></div>
                        </div>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                    <div class="bg-gray-50 rounded-lg p-4">
                        <p class="text-sm text-gray-600 mb-1">MRR (Mean Reciprocal Rank) 🔥</p>
                        <p class="text-2xl font-bold text-gray-900">0.723</p>
                        <p class="text-xs text-gray-500 mt-1">Position moyenne du 1er doc pertinent: 1.4</p>
                    </div>
                    <div class="bg-gray-50 rounded-lg p-4">
                        <p class="text-sm text-gray-600 mb-1">NDCG@10 (Qualité du ranking) 🔥</p>
                        <p class="text-2xl font-bold text-gray-900">0.812</p>
                        <p class="text-xs text-gray-500 mt-1">Meilleure métrique pour évaluer le ranking</p>
                    </div>
                </div>

                <div class="bg-white border rounded-lg p-4">
                    <h4 class="font-semibold text-gray-900 mb-3">Évolution Retrieval Metrics (7 derniers jours)</h4>
                    <canvas id="retrievalChart" height="80"></canvas>
                </div>
            </div>

            <!-- Tab Content: Context -->
            <div id="context" class="tab-content p-6">
                <h3 class="text-lg font-semibold text-gray-900 mb-4">Métriques de Contexte (Context Building)</h3>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                    <div class="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-5 border border-blue-200">
                        <p class="text-sm text-blue-700 font-medium mb-1">Context Precision 🔥</p>
                        <p class="text-3xl font-bold text-blue-900">89.4%</p>
                        <p class="text-xs text-blue-600 mt-2">% de docs pertinents dans le contexte final</p>
                        <div class="w-full bg-blue-200 rounded-full h-2 mt-3">
                            <div class="bg-blue-600 h-2 rounded-full" style="width: 89.4%"></div>
                        </div>
                    </div>
                    <div class="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-5 border border-green-200">
                        <p class="text-sm text-green-700 font-medium mb-1">Context Recall 🔥</p>
                        <p class="text-3xl font-bold text-green-900">82.7%</p>
                        <p class="text-xs text-green-600 mt-2">% de docs pertinents récupérés dans le contexte</p>
                        <div class="w-full bg-green-200 rounded-full h-2 mt-3">
                            <div class="bg-green-600 h-2 rounded-full" style="width: 82.7%"></div>
                        </div>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <div class="bg-gray-50 rounded-lg p-4">
                        <p class="text-sm text-gray-600 mb-1">Context Relevance 🔥</p>
                        <p class="text-2xl font-bold text-gray-900">90.1%</p>
                        <p class="text-xs text-gray-500 mt-1">Pertinence globale du contexte</p>
                    </div>
                    <div class="bg-gray-50 rounded-lg p-4">
                        <p class="text-sm text-gray-600 mb-1">Context Utilization 🔥</p>
                        <p class="text-2xl font-bold text-gray-900">87.3%</p>
                        <p class="text-xs text-gray-500 mt-1">Le LLM utilise bien le contexte</p>
                    </div>
                    <div class="bg-gray-50 rounded-lg p-4">
                        <p class="text-sm text-gray-600 mb-1">Docs dans contexte (moy.)</p>
                        <p class="text-2xl font-bold text-gray-900">4.2</p>
                        <p class="text-xs text-gray-500 mt-1">Après reranking/filtering</p>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                    <div class="bg-white border rounded-lg p-4">
                        <h4 class="font-semibold text-gray-900 mb-3">Context Tokens</h4>
                        <canvas id="contextTokensChart" height="100"></canvas>
                    </div>
                    <div class="bg-white border rounded-lg p-4">
                        <h4 class="font-semibold text-gray-900 mb-3">Distribution Docs par Contexte</h4>
                        <canvas id="contextDocsChart" height="100"></canvas>
                    </div>
                </div>

                <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <h4 class="font-semibold text-blue-900 mb-2">💡 Insights</h4>
                    <ul class="text-sm text-blue-800 space-y-1">
                        <li>• Context Precision élevée (89.4%) → Bon reranking/filtering</li>
                        <li>• Context Utilization à 87.3% → Le LLM utilise bien le contexte fourni</li>
                        <li>• Moyenne de 4.2 docs/contexte → Taille optimale pour éviter confusion</li>
                    </ul>
                </div>
            </div>

            <!-- Tab Content: Generation -->
            <div id="generation" class="tab-content p-6">
                <h3 class="text-lg font-semibold text-gray-900 mb-4">Métriques de Génération (LLM Response)</h3>
                
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <div class="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-5 border border-green-200">
                        <p class="text-sm text-green-700 font-medium mb-1">Faithfulness 🔥</p>
                        <p class="text-3xl font-bold text-green-900">92.3%</p>
                        <p class="text-xs text-green-600 mt-2">Pas d'hallucinations</p>
                        <div class="mt-3">
                            <span class="px-2 py-1 bg-green-200 text-green-800 rounded text-xs font-semibold">Excellent</span>
                        </div>
                    </div>
                    <div class="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-5 border border-blue-200">
                        <p class="text-sm text-blue-700 font-medium mb-1">Answer Relevance 🔥</p>
                        <p class="text-3xl font-bold text-blue-900">88.7%</p>
                        <p class="text-xs text-blue-600 mt-2">Répond à la question</p>
                        <div class="mt-3">
                            <span class="px-2 py-1 bg-blue-200 text-blue-800 rounded text-xs font-semibold">Bon</span>
                        </div>
                    </div>
                    <div class="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-5 border border-purple-200">
                        <p class="text-sm text-purple-700 font-medium mb-1">Correctness</p>
                        <p class="text-3xl font-bold text-purple-900">85.4%</p>
                        <p class="text-xs text-purple-600 mt-2">vs ground truth</p>
                        <div class="mt-3">
                            <span class="px-2 py-1 bg-purple-200 text-purple-800 rounded text-xs font-semibold">Bon</span>
                        </div>
                    </div>
                </div>

                <div class="bg-white border rounded-lg p-4 mb-6">
                    <h4 class="font-semibold text-gray-900 mb-3">Évolution Generation Metrics</h4>
                    <canvas id="generationChart" height="80"></canvas>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                        <h4 class="font-semibold text-yellow-900 mb-2">⚠️ Hallucinations détectées</h4>
                        <p class="text-2xl font-bold text-yellow-900 mb-2">7.7%</p>
                        <p class="text-sm text-yellow-700">des réponses (220 sur 2847 requêtes)</p>
                        <button class="mt-3 px-3 py-1 bg-yellow-600 text-white rounded hover:bg-yellow-700 text-sm">
                            Voir les cas
                        </button>
                    </div>
                    <div class="bg-green-50 border border-green-200 rounded-lg p-4">
                        <h4 class="font-semibold text-green-900 mb-2">✅ Réponses parfaites</h4>
                        <p class="text-2xl font-bold text-green-900 mb-2">79.2%</p>
                        <p class="text-sm text-green-700">Faithfulness + Relevance > 90%</p>
                        <button class="mt-3 px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700 text-sm">
                            Analyser patterns
                        </button>
                    </div>
                </div>
            </div>

            <!-- Tab Content: Performance -->
            <div id="performance" class="tab-content p-6">
                <h3 class="text-lg font-semibold text-gray-900 mb-4">Métriques de Performance</h3>
                
                <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                    <div class="bg-gray-50 rounded-lg p-4">
                        <p class="text-sm text-gray-600 mb-1">Latency P50</p>
                        <p class="text-2xl font-bold text-gray-900">1.8s</p>
                        <p class="text-xs text-gray-500 mt-1">Médiane</p>
                    </div>
                    <div class="bg-gray-50 rounded-lg p-4">
                        <p class="text-sm text-gray-600 mb-1">Latency P95 🔥</p>
                        <p class="text-2xl font-bold text-gray-900">2.8s</p>
                        <p class="text-xs text-gray-500 mt-1">95e percentile</p>
                    </div>
                    <div class="bg-gray-50 rounded-lg p-4">
                        <p class="text-sm text-gray-600 mb-1">Latency P99 🔥</p>
                        <p class="text-2xl font-bold text-gray-900">4.2s</p>
                        <p class="text-xs text-gray-500 mt-1">99e percentile</p>
                    </div>
                    <div class="bg-gray-50 rounded-lg p-4">
                        <p class="text-sm text-gray-600 mb-1">Moyenne</p>
                        <p class="text-2xl font-bold text-gray-900">2.1s</p>
                        <p class="text-xs text-gray-500 mt-1">Temps moyen</p>
                    </div>
                </div>

                <div class="bg-white border rounded-lg p-4 mb-6">
                    <h4 class="font-semibold text-gray-900 mb-3">Distribution Latency (dernières 24h)</h4>
                    <canvas id="latencyChart" height="80"></canvas>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                    <div class="bg-gradient-to-br from-orange-50 to-orange-100 rounded-lg p-5 border border-orange-200">
                        <p class="text-sm text-orange-700 font-medium mb-1">Cost per Query 🔥</p>
                        <p class="text-3xl font-bold text-orange-900">$0.0142</p>
                        <div class="mt-3 space-y-1 text-xs text-orange-700">
                            <p>• Embedding: $0.0001</p>
                            <p>• Reranking: $0.0021</p>
                            <p>• LLM Generation: $0.0120</p>
                        </div>
                    </div>
                    <div class="bg-gradient-to-br from-pink-50 to-pink-100 rounded-lg p-5 border border-pink-200">
                        <p class="text-sm text-pink-700 font-medium mb-1">Coût total (30 jours)</p>
                        <p class="text-3xl font-bold text-pink-900">$1,247</p>
                        <div class="mt-3 text-xs text-pink-700">
                            <p>87,850 requêtes × $0.0142</p>
                            <p class="mt-2">Projection mois prochain: $1,320</p>
                        </div>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div class="bg-white border rounded-lg p-4">
                        <h4 class="font-semibold text-gray-900 mb-3">Throughput</h4>
                        <p class="text-2xl font-bold text-gray-900 mb-2">12.3 req/s</p>
                        <p class="text-sm text-gray-600">Capacité actuelle: 25 req/s</p>
                        <div class="w-full bg-gray-200 rounded-full h-2 mt-3">
                            <div class="bg-green-600 h-2 rounded-full" style="width: 49.2%"></div>
                        </div>
                        <p class="text-xs text-gray-500 mt-2">49% de la capacité utilisée</p>
                    </div>
                    <div class="bg-white border rounded-lg p-4">
                        <h4 class="font-semibold text-gray-900 mb-3">Taux d'erreur</h4>
                        <p class="text-2xl font-bold text-gray-900 mb-2">0.34%</p>
                        <p class="text-sm text-gray-600">97 erreurs sur 28,470 requêtes</p>
                        <div class="mt-3 space-y-1 text-xs text-gray-600">
                            <p>• Timeout: 42 (43%)</p>
                            <p>• Rate limit: 31 (32%)</p>
                            <p>• Autres: 24 (25%)</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Comparaison temporelle -->
        <div class="bg-white rounded-lg shadow p-6">
            <h3 class="text-lg font-semibold text-gray-900 mb-4">Évolution Globale (30 derniers jours)</h3>
            <canvas id="overallTrendChart" height="60"></canvas>
        </div>
    </div>

    <script>
        function switchTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab-button').forEach(btn => {
                btn.classList.remove('active');
            });
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }

        // Retrieval Chart
        const retrievalCtx = document.getElementById('retrievalChart').getContext('2d');
        new Chart(retrievalCtx, {
            type: 'line',
            data: {
                labels: ['J-7', 'J-6', 'J-5', 'J-4', 'J-3', 'J-2', 'J-1', 'Aujourd\'hui'],
                datasets: [
                    {
                        label: 'Precision@5',
                        data: [82, 83, 84, 83.5, 84, 84.2, 84.3, 84.5],
                        borderColor: 'rgb(59, 130, 246)',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.3
                    },
                    {
                        label: 'NDCG@10',
                        data: [79, 80, 80.5, 80.8, 81, 81.2, 81.5, 81.2],
                        borderColor: 'rgb(168, 85, 247)',
                        backgroundColor: 'rgba(168, 85, 247, 0.1)',
                        tension: 0.3
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: { legend: { position: 'top' } },
                scales: { y: { beginAtZero: false, min: 75, max: 90 } }
            }
        });

        // Context Tokens Chart
        const contextTokensCtx = document.getElementById('contextTokensChart').getContext('2d');
        new Chart(contextTokensCtx, {
            type: 'bar',
            data: {
                labels: ['0-500', '500-1000', '1000-1500', '1500-2000', '2000+'],
                datasets: [{
                    label: 'Nombre de requêtes',
                    data: [120, 580, 1240, 720, 187],
                    backgroundColor: 'rgba(59, 130, 246, 0.6)'
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: { y: { beginAtZero: true } }
            }
        });

        // Context Docs Chart
        const contextDocsCtx = document.getElementById('contextDocsChart').getContext('2d');
        new Chart(contextDocsCtx, {
            type: 'doughnut',
            data: {
                labels: ['2 docs', '3 docs', '4 docs', '5 docs', '6+ docs'],
                datasets: [{
                    data: [8, 15, 42, 28, 7],
                    backgroundColor: [
                        'rgba(239, 68, 68, 0.8)',
                        'rgba(251, 146, 60, 0.8)',
                        'rgba(34, 197, 94, 0.8)',
                        'rgba(59, 130, 246, 0.8)',
                        'rgba(168, 85, 247, 0.8)'
                    ]
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { position: 'right' } }
            }
        });

        // Generation Chart
        const generationCtx = document.getElementById('generationChart').getContext('2d');
        new Chart(generationCtx, {
            type: 'line',
            data: {
                labels: ['J-7', 'J-6', 'J-5', 'J-4', 'J-3', 'J-2', 'J-1', 'Aujourd\'hui'],
                datasets: [
                    {
                        label: 'Faithfulness',
                        data: [91, 91.5, 92, 91.8, 92.2, 92.1, 92.3, 92.3],
                        borderColor: 'rgb(34, 197, 94)',
                        backgroundColor: 'rgba(34, 197, 94, 0.1)',
                        tension: 0.3
                    },
                    {
                        label: 'Answer Relevance',
                        data: [87, 87.5, 88, 88.2, 88.5, 88.4, 88.6, 88.7],
                        borderColor: 'rgb(59, 130, 246)',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.3
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: { legend: { position: 'top' } },
                scales: { y: { beginAtZero: false, min: 85, max: 95 } }
            }
        });

        // Latency Chart
        const latencyCtx = document.getElementById('latencyChart').getContext('2d');
        new Chart(latencyCtx, {
            type: 'line',
            data: {
                labels: Array.from({length: 24}, (_, i) => `${i}h`),
                datasets: [
                    {
                        label: 'P50',
                        data: [1.7, 1.8, 1.7, 1.9, 1.8, 2.0, 2.1, 2.2, 2.0, 1.9, 1.8, 1.7, 1.8, 1.9, 2.0, 1.9, 1.8, 1.9, 2.0, 2.1, 1.9, 1.8, 1.7, 1.8],
                        borderColor: 'rgb(34, 197, 94)',
                        tension: 0.3
                    },
                    {
                        label: 'P95',
                        data: [2.5, 2.7, 2.6, 2.8, 2.7, 2.9, 3.0, 3.1, 2.9, 2.8, 2.7, 2.6, 2.7, 2.8, 2.9, 2.8, 2.7, 2.8, 2.9, 3.0, 2.8, 2.7, 2.6, 2.8],
                        borderColor: 'rgb(251, 146, 60)',
                        tension: 0.3
                    },
                    {
                        label: 'P99',
                        data: [3.8, 4.0, 3.9, 4.2, 4.0, 4.3, 4.5, 4.6, 4.3, 4.2, 4.0, 3.9, 4.0, 4.1, 4.3, 4.2, 4.0, 4.1, 4.3, 4.5, 4.2, 4.0, 3.9, 4.2],
                        borderColor: 'rgb(239, 68, 68)',
                        tension: 0.3
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: { legend: { position: 'top' } },
                scales: { y: { beginAtZero: true, title: { display: true, text: 'Secondes' } } }
            }
        });

        // Overall Trend Chart
        const overallCtx = document.getElementById('overallTrendChart').getContext('2d');
        new Chart(overallCtx, {
            type: 'line',
            data: {
                labels: ['J-30', 'J-27', 'J-24', 'J-21', 'J-18', 'J-15', 'J-12', 'J-9', 'J-6', 'J-3', 'Aujourd\'hui'],
                datasets: [
                    {
                        label: 'Score RAGAS',
                        data: [82, 83, 84, 84.5, 85, 85.5, 86, 86.5, 87, 87.2, 87.5],
                        borderColor: 'rgb(59, 130, 246)',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.3,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Satisfaction',
                        data: [87, 88, 88.5, 89, 89.5, 90, 90.2, 90.5, 91, 91.1, 91.2],
                        borderColor: 'rgb(168, 85, 247)',
                        backgroundColor: 'rgba(168, 85, 247, 0.1)',
                        tension: 0.3,
                        yAxisID: 'y'
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: { legend: { position: 'top' } },
                scales: { 
                    y: { beginAtZero: false, min: 75, max: 100, position: 'left', title: { display: true, text: 'Score (%)' } }
                }
            }
        });
    </script>
</body>
</html>

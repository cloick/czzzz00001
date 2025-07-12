

## 🔧 **PARTIE 1/2 - CORRECTIONS FONDAMENTALES :**

// ===================================================================
// CHANGE RISK ANALYZER - VERSION CORRIGÉE PARTIE 1
// ===================================================================

// Variables globales
let currentChangeData = null;
let activeTab = 'team';

// Initialisation au chargement de la page
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Initialisation de Change Risk Analyzer');
    initializeApp();
    setupEventListeners();
    
    // Démarrer la vérification du statut après un court délai
    setTimeout(() => {
        checkSystemStatus();
    }, 1000);
});

// ===================================================================
// INITIALISATION
// ===================================================================

function initializeApp() {
    console.log('🚀 Initialisation de Change Risk Analyzer');
    
    // Afficher un message de bienvenue
    showToast('Bienvenue dans Change Risk Analyzer', 'success');
    
    // Mettre à jour l'interface avec les états de chargement
    updateLoadingStates();
}

function updateLoadingStates() {
    // Mettre les éléments en état de chargement
    const modelStatus = document.getElementById('model-status-text');
    const connectionStatus = document.getElementById('connection-status-text');
    const performance = document.getElementById('model-performance');
    
    if (modelStatus) modelStatus.textContent = '🔄 Vérification...';
    if (connectionStatus) connectionStatus.textContent = '🔄 Vérification...';
    if (performance) performance.innerHTML = '<p>🔄 Chargement...</p>';
}

function setupEventListeners() {
    // Bouton analyser
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', analyzeChange);
    }
    
    // Bouton test
    const testBtn = document.getElementById('test-btn');
    if (testBtn) {
        testBtn.addEventListener('click', testConnection);
    }
    
    // ✅ AJOUT: Bouton diagnostic
    const diagnosticBtn = document.getElementById('diagnostic-btn');
    if (diagnosticBtn) {
        diagnosticBtn.addEventListener('click', showDiagnostic);
    }
    
    // ✅ AJOUT: Bouton exemple
    const exampleBtn = document.getElementById('example-btn');
    if (exampleBtn) {
        exampleBtn.addEventListener('click', setExampleChange);
    }
    
    // Input changement (validation en temps réel)
    const changeInput = document.getElementById('change-ref');
    if (changeInput) {
        changeInput.addEventListener('input', function() {
            // Auto-majuscules
            this.value = this.value.toUpperCase();
            validateChangeReference();
        });
        
        // Touche Entrée pour lancer l'analyse
        changeInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                analyzeChange();
            }
        });
    }
}

// ===================================================================
// FONCTIONS UTILITAIRES CORRIGÉES
// ===================================================================

function showLoading(message = 'Chargement...') {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        const spinner = overlay.querySelector('.loading-spinner p');
        if (spinner) {
            spinner.textContent = message;
        }
        overlay.style.display = 'flex';
    }
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    
    const toast = document.createElement('div');
    toast.className = 'toast ' + type;
    toast.innerHTML = 
        '<div style="display: flex; justify-content: space-between; align-items: center;">' +
            '<span>' + message + '</span>' +
            '<button onclick="this.parentElement.parentElement.remove()" style="background: none; border: none; font-size: 1.2rem; cursor: pointer;">&times;</button>' +
        '</div>';
    
    container.appendChild(toast);
    
    // Auto-suppression après 5 secondes
    setTimeout(() => {
        if (toast.parentElement) {
            toast.remove();
        }
    }, 5000);
}

function validateChangeReference() {
    const input = document.getElementById('change-ref');
    if (!input) return false;
    
    const value = input.value.toUpperCase();
    const pattern = /^CAGIPCHG\d{7}$/;
    
    // Supprimer les classes précédentes
    input.classList.remove('valid', 'invalid');
    
    if (!value) {
        input.style.borderColor = '#e9ecef';
        return true; // Vide = OK
    }
    
    if (pattern.test(value)) {
        input.classList.add('valid');
        input.style.borderColor = '#28a745';
        return true;
    } else {
        input.classList.add('invalid');
        input.style.borderColor = '#dc3545';
        return false;
    }
}

// ===================================================================
// API CALLS - VERSION CORRIGÉE SANS BACKTICKS
// ===================================================================

async function apiCall(endpoint, params = {}) {
    try {
        console.log("🔗 Appel API: " + endpoint, params);
        
        // ✅ SOLUTION CORRECTE : Utiliser getWebAppBackendUrl()
        const baseUrl = getWebAppBackendUrl(endpoint);
        let url = baseUrl;
        
        // Ajouter les paramètres GET si nécessaire
        if (Object.keys(params).length > 0) {
            const searchParams = new URLSearchParams();
            Object.keys(params).forEach(key => {
                if (params[key] !== null && params[key] !== undefined) {
                    searchParams.append(key, String(params[key]));
                }
            });
            url += "?" + searchParams.toString();
        }
        
        console.log("📡 URL générée par Dataiku: " + url);
        
        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            credentials: 'same-origin'
        });
        
        console.log("📡 Status HTTP: " + response.status);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error("❌ Erreur HTTP " + response.status + ":", errorText);
            throw new Error("HTTP " + response.status + ": " + response.statusText);
        }
        
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            const text = await response.text();
            console.warn("⚠️ Réponse non-JSON:", text.substring(0, 200));
            throw new Error("Réponse non-JSON reçue");
        }
        
        const data = await response.json();
        console.log("📥 Réponse " + endpoint + ":", data);
        
        if (data.status === 'error') {
            throw new Error(data.message || 'Erreur inconnue');
        }
        
        return data;
        
    } catch (error) {
        console.error("❌ Erreur API " + endpoint + ":", error);
        throw error;
    }
}

// ===================================================================
// FONCTIONS PRINCIPALES CORRIGÉES
// ===================================================================

async function checkSystemStatus() {
    console.log('🔍 Vérification du statut système...');
    
    try {
        // Vérifier le statut du modèle
        console.log('📊 Vérification du modèle...');
        const modelStatus = await apiCall('get_model_status');
        updateModelStatus(modelStatus);
        
        // Vérifier le statut des connexions
        console.log('🔗 Vérification des connexions...');
        const connectionStatus = await apiCall('get_connection_status');
        updateConnectionStatus(connectionStatus);
        
        console.log('✅ Vérification du statut terminée');
        
    } catch (error) {
        console.error('❌ Erreur vérification statut:', error);
        showToast('Erreur lors de la vérification du statut système', 'error');
        
        // Mettre en erreur les statuts
        updateModelStatusError(error.message);
        updateConnectionStatusError(error.message);
    }
}

## 🔧 **PARTIE 2/2 - FONCTIONS COMPLÈTES ET CORRECTIONS FINALES :**


// ===================================================================
// MISE À JOUR DES STATUTS - VERSION CORRIGÉE
// ===================================================================

function updateModelStatus(status) {
    console.log('📊 Mise à jour statut modèle:', status);
    
    const statusCard = document.getElementById('model-status');
    const statusText = document.getElementById('model-status-text');
    const details = document.getElementById('model-details');
    const performance = document.getElementById('model-performance');
    
    if (!statusCard || !statusText || !performance) {
        console.error('❌ Éléments DOM manquants pour le statut modèle');
        return;
    }
    
    if (status.data && status.data.status === 'Modèle chargé') {
        statusCard.className = 'status-card status-success';
        statusText.textContent = '✅ Opérationnel';
        if (details) {
            details.textContent = "Algorithme: " + status.data.algorithm;
        }
        
        // Performance
        const perf = status.data.training_info && status.data.training_info.performance || {};
        performance.innerHTML = 
            "<p><strong>Recall:</strong> " + (perf.recall || 'N/A') + "</p>" +
            "<p><strong>Precision:</strong> " + (perf.precision || 'N/A') + "</p>" +
            "<p><strong>Features:</strong> " + (status.data.features && status.data.features.count || 'N/A') + "</p>";
    } else {
        statusCard.className = 'status-card status-error';
        statusText.textContent = '❌ Non disponible';
        if (details) {
            details.textContent = 'Modèle non chargé';
        }
        performance.innerHTML = '<p>❌ Non disponible</p>';
    }
}

function updateModelStatusError(errorMsg) {
    const statusCard = document.getElementById('model-status');
    const statusText = document.getElementById('model-status-text');
    const details = document.getElementById('model-details');
    const performance = document.getElementById('model-performance');
    
    if (statusCard) statusCard.className = 'status-card status-error';
    if (statusText) statusText.textContent = '❌ Erreur';
    if (details) details.textContent = errorMsg;
    if (performance) performance.innerHTML = '<p>❌ Erreur de chargement</p>';
}

function updateConnectionStatus(status) {
    console.log('🔗 Mise à jour statut connexion:', status);
    
    const statusCard = document.getElementById('connection-status');
    const statusText = document.getElementById('connection-status-text');
    const details = document.getElementById('connection-details');
    
    if (!statusCard || !statusText) {
        console.error('❌ Éléments DOM manquants pour le statut connexion');
        return;
    }
    
    if (status.data && status.data.status === 'Connecté') {
        statusCard.className = 'status-card status-success';
        statusText.textContent = '✅ Connecté';
        if (details) {
            details.textContent = 'Tables: change_request & incident_filtree';
        }
    } else {
        statusCard.className = 'status-card status-error';
        statusText.textContent = '❌ Erreur';
        if (details) {
            details.textContent = status.data && status.data.error || 'Erreur de connexion';
        }
    }
}

function updateConnectionStatusError(errorMsg) {
    const statusCard = document.getElementById('connection-status');
    const statusText = document.getElementById('connection-status-text');
    const details = document.getElementById('connection-details');
    
    if (statusCard) statusCard.className = 'status-card status-error';
    if (statusText) statusText.textContent = '❌ Erreur';
    if (details) details.textContent = errorMsg;
}

// ===================================================================
// FONCTIONS DE TEST ET DIAGNOSTIC
// ===================================================================

async function testConnection() {
    try {
        showLoading('Test des connexions...');
        
        const result = await apiCall('test_connection');
        
        hideLoading();
        
        if (result.data && result.data.success) {
            showToast('✅ Test de connexion réussi', 'success');
        } else {
            showToast('❌ Test de connexion échoué', 'error');
        }
        
    } catch (error) {
        hideLoading();
        showToast("❌ Erreur de test: " + error.message, 'error');
    }
}

// ✅ FONCTION DIAGNOSTIC AJOUTÉE
async function showDiagnostic() {
    try {
        showLoading('Diagnostic système en cours...');
        
        const result = await apiCall('diagnostic');
        
        hideLoading();
        
        // Créer ou afficher un modal pour le diagnostic
        let modal = document.getElementById('diagnostic-modal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'diagnostic-modal';
            modal.className = 'modal';
            modal.innerHTML = 
                '<div class="modal-content">' +
                    '<div class="modal-header">' +
                        '<h2><i class="fas fa-stethoscope"></i> Diagnostic Système</h2>' +
                        '<button class="modal-close" onclick="closeDiagnostic()">&times;</button>' +
                    '</div>' +
                    '<div class="modal-body" id="diagnostic-content"></div>' +
                '</div>';
            document.body.appendChild(modal);
        }
        
        const content = document.getElementById('diagnostic-content');
        if (content && result.data) {
            content.innerHTML = '<pre>' + JSON.stringify(result.data, null, 2) + '</pre>';
        } else {
            content.innerHTML = '<div class="error-box"><p>❌ Erreur lors du diagnostic</p></div>';
        }
        
        modal.style.display = 'flex';
        
    } catch (error) {
        hideLoading();
        showToast("❌ Erreur diagnostic: " + error.message, 'error');
    }
}

function closeDiagnostic() {
    const modal = document.getElementById('diagnostic-modal');
    if (modal) {
        modal.style.display = 'none';
    }
}

// ===================================================================
// ANALYSE DES CHANGEMENTS - VERSION CORRIGÉE
// ===================================================================

async function analyzeChange() {
    const changeRef = document.getElementById('change-ref').value.trim().toUpperCase();
    
    // Validation
    if (!changeRef) {
        showToast('⚠️ Veuillez saisir une référence de changement', 'warning');
        return;
    }
    
    if (!validateChangeReference()) {
        showToast('❌ Format invalide. Utilisez CAGIPCHG + 7 chiffres', 'error');
        return;
    }
    
    try {
        showLoading("Analyse de " + changeRef + " en cours...");
        
        // Appel API pour analyser le changement
        const result = await apiCall('analyze_change', { change_ref: changeRef });
        
        hideLoading();
        
        if (result.data && result.data.change_found) {
            // ✅ STOCKAGE CORRECT DES DONNÉES
            currentChangeData = result.data;
            window.currentChangeData = result.data; // Pour les autres fonctions
            
            displayAnalysisResults(result.data);
            showToast("✅ Analyse de " + changeRef + " terminée", 'success');
        } else {
            showToast("❌ Changement " + changeRef + " non trouvé", 'error');
            clearResults();
        }
        
    } catch (error) {
        hideLoading();
        showToast("❌ Erreur d'analyse: " + error.message, 'error');
        clearResults();
    }
}

// ===================================================================
// AFFICHAGE DES RÉSULTATS - VERSION CORRIGÉE
// ===================================================================

function displayAnalysisResults(data) {
    displayMainResults(data);
    displayDetailedResults(data);
}

function displayMainResults(data) {
    const resultsDiv = document.getElementById('analysis-results');
    if (!resultsDiv) return;
    
    const analysis = data.detailed_analysis;
    
    resultsDiv.innerHTML = 
        '<div class="text-center mb-3">' +
            '<h3><i class="fas fa-chart-pie"></i> Analyse de ' + data.change_ref + '</h3>' +
        '</div>' +
        '<div class="risk-card">' +
            '<div class="risk-score">' + analysis.risk_score + '%</div>' +
            '<div class="risk-level">Risque d\'échec</div>' +
            '<div class="risk-level">Niveau: ' + analysis.risk_level + '</div>' +
            '<div class="risk-interpretation">' + (analysis.interpretation || 'Analyse terminée') + '</div>' +
        '</div>';
}

function displayDetailedResults(data) {
    const detailedDiv = document.getElementById('detailed-results');
    if (!detailedDiv) return;
    
    const analysis = data.detailed_analysis;
    const change = data.change_data;
    
    // Construire la liste des facteurs de risque
    let riskFactorsHtml = '';
    if (analysis.risk_factors && analysis.risk_factors.length > 0) {
        riskFactorsHtml = '<ul>';
        analysis.risk_factors.forEach(factor => {
            riskFactorsHtml += '<li>' + factor + '</li>';
        });
        riskFactorsHtml += '</ul>';
    } else {
        riskFactorsHtml = '<p>Aucun facteur spécifique détecté</p>';
    }
    
    detailedDiv.innerHTML = 
        '<div class="details-grid">' +
            '<!-- Colonne gauche -->' +
            '<div class="details-section">' +
                '<h4><i class="fas fa-exclamation-triangle"></i> Facteurs de risque</h4>' +
                '<div class="warning-box">' + riskFactorsHtml + '</div>' +
                
                '<h4><i class="fas fa-lightbulb"></i> Recommandations</h4>' +
                '<div class="success-box">' +
                    '<ul>' +
                        '<li>✅ Vérifier la documentation</li>' +
                        '<li>✅ Planifier les tests</li>' +
                        '<li>✅ Préparer le rollback</li>' +
                    '</ul>' +
                '</div>' +
            '</div>' +
            
            '<!-- Colonne droite -->' +
            '<div class="details-section">' +
                '<h4><i class="fas fa-cogs"></i> Caractéristiques techniques</h4>' +
                '<div class="features-list">' +
                    '<p><strong>Type SILCA:</strong> <span>' + (change.dv_u_type_change_silca || 'N/A') + '</span></p>' +
                    '<p><strong>Type de changement:</strong> <span>' + (change.dv_type || 'N/A') + '</span></p>' +
                    '<p><strong>Nombre de CAB:</strong> <span>' + (change.u_cab_count || 'N/A') + '</span></p>' +
                    '<p><strong>Périmètre BCR:</strong> <span>' + (change.u_bcr ? '✅' : '❌') + '</span></p>' +
                    '<p><strong>Périmètre BPC:</strong> <span>' + (change.u_bpc ? '✅' : '❌') + '</span></p>' +
                '</div>' +
                
                '<h4><i class="fas fa-clipboard-list"></i> Métadonnées</h4>' +
                '<div class="info-box">' +
                    '<p><strong>Équipe:</strong> ' + (change.dv_assignment_group || 'N/A') + '</p>' +
                    '<p><strong>CI/Solution:</strong> ' + (change.dv_cmdb_ci || 'N/A') + '</p>' +
                    '<p><strong>Catégorie:</strong> ' + (change.dv_category || 'N/A') + '</p>' +
                    '<p><strong>État:</strong> ' + (change.dv_state || 'N/A') + '</p>' +
                '</div>' +
            '</div>' +
        '</div>' +
        
        '<!-- Onglets contextuels -->' +
        '<div class="tabs-container">' +
            '<hr>' +
            '<h3><i class="fas fa-chart-line"></i> Informations contextuelles</h3>' +
            '<p style="color: #666; font-style: italic;"><i class="fas fa-database"></i> Données extraites des tables ServiceNow réelles</p>' +
            
            '<div class="tabs-buttons">' +
                '<button class="btn-tab active" onclick="switchTab(\'team\')">' +
                    '<i class="fas fa-users"></i> Statistiques équipe' +
                '</button>' +
                '<button class="btn-tab" onclick="switchTab(\'incidents\')">' +
                    '<i class="fas fa-tools"></i> Incidents liés' +
                '</button>' +
                '<button class="btn-tab" onclick="switchTab(\'similar\')">' +
                    '<i class="fas fa-clipboard-list"></i> Changements similaires' +
                '</button>' +
            '</div>' +
            
            '<div id="tab-content" class="tab-content">' +
                '<p style="text-align: center; color: #666; padding: 2rem;">' +
                    '<i class="fas fa-mouse-pointer"></i> Cliquez sur un onglet pour voir les informations contextuelles' +
                '</p>' +
            '</div>' +
        '</div>';
}

// ===================================================================
// FONCTION SWITCHTAB - VERSION CORRIGÉE ET COMPLÈTE
// ===================================================================

async function switchTab(tabName) {
    if (!currentChangeData) {
        showToast('❌ Aucun changement analysé', 'error');
        return;
    }
    
    activeTab = tabName;
    
    // Mettre à jour les boutons d'onglet
    document.querySelectorAll('.btn-tab').forEach(btn => btn.classList.remove('active'));
    
    // Trouver et activer le bon bouton
    const buttons = document.querySelectorAll('.btn-tab');
    buttons.forEach(btn => {
        if (btn.onclick && btn.onclick.toString().includes(tabName)) {
            btn.classList.add('active');
        }
    });
    
    const contentDiv = document.getElementById('tab-content');
    if (!contentDiv) {
        console.error('❌ Élément tab-content introuvable');
        return;
    }
    
    try {
        contentDiv.innerHTML = '<p style="text-align: center;"><i class="fas fa-spinner fa-spin"></i> Chargement...</p>';
        
        let result;
        
        switch(tabName) {
            case 'team':
                const assignmentGroup = currentChangeData.change_data.dv_assignment_group;
                if (!assignmentGroup) {
                    contentDiv.innerHTML = '<div class="warning-box"><p>⚠️ Aucune équipe identifiée</p></div>';
                    return;
                }
                result = await apiCall('get_team_stats', { 
                    assignment_group: assignmentGroup,
                    months_back: 6
                });
                displayTeamStats(result.data);
                break;
                
            case 'incidents':
                const cmdbCi = currentChangeData.change_data.dv_cmdb_ci;
                if (!cmdbCi) {
                    contentDiv.innerHTML = '<div class="warning-box"><p>⚠️ Aucun CI identifié</p></div>';
                    return;
                }
                result = await apiCall('get_incidents', { 
                    cmdb_ci: cmdbCi,
                    months_back: 3
                });
                displayIncidents(result.data);
                break;
                
            case 'similar':
                result = await apiCall('get_similar_changes', { 
                    change_ref: currentChangeData.change_ref,
                    limit: 10
                });
                displaySimilarChanges(result.data);
                break;
                
            default:
                contentDiv.innerHTML = '<div class="error-box"><p>❌ Onglet inconnu: ' + tabName + '</p></div>';
                return;
        }
        
    } catch (error) {
        console.error('❌ Erreur switchTab:', error);
        const contentDiv = document.getElementById('tab-content');
        if (contentDiv) {
            contentDiv.innerHTML = 
                '<div class="error-box">' +
                    '<p><i class="fas fa-exclamation-circle"></i> Erreur: ' + error.message + '</p>' +
                    '<p><small>Cet endpoint pourrait ne pas être implémenté côté backend.</small></p>' +
                '</div>';
        }
    }
}

// ===================================================================
// FONCTIONS D'AFFICHAGE DES ONGLETS - VERSION CORRIGÉE
// ===================================================================

function displayTeamStats(stats) {
    const contentDiv = document.getElementById('tab-content');
    if (!contentDiv) return;
    
    if (stats && !stats.error) {
        const lastFailureText = stats.last_failure_date ? 
            "Il y a " + Math.floor((new Date() - new Date(stats.last_failure_date)) / (1000 * 60 * 60 * 24)) + " jours" :
            'Aucun récent';
        
        contentDiv.innerHTML = 
            '<div class="metrics-grid">' +
                '<div class="metric-card">' +
                    '<div class="metric-value">' + (stats.total_changes || 0) + '</div>' +
                    '<div class="metric-title">Total changements</div>' +
                    '<div class="metric-subtitle">6 derniers mois</div>' +
                '</div>' +
                '<div class="metric-card">' +
                    '<div class="metric-value">' + (stats.success_rate || 0) + '%</div>' +
                    '<div class="metric-title">Taux de succès</div>' +
                '</div>' +
                '<div class="metric-card">' +
                    '<div class="metric-value">' + (stats.failures || 0) + '</div>' +
                    '<div class="metric-title">Échecs</div>' +
                '</div>' +
                '<div class="metric-card">' +
                    '<div class="metric-value">' + lastFailureText + '</div>' +
                    '<div class="metric-title">Dernier échec</div>' +
                '</div>' +
            '</div>';
    } else {
        contentDiv.innerHTML = 
            '<div class="warning-box">' +
                '<p><i class="fas fa-exclamation-triangle"></i> Statistiques équipe non disponibles</p>' +
                '<p><small>Équipe: ' + (currentChangeData.change_data.dv_assignment_group || 'N/A') + '</small></p>' +
            '</div>';
    }
}

function displayIncidents(incidents) {
    const contentDiv = document.getElementById('tab-content');
    if (!contentDiv) return;
    
    if (incidents && incidents.total_incidents !== undefined) {
        const resolutionText = incidents.avg_resolution_hours > 0 ? 
            incidents.avg_resolution_hours + "h" : 'N/A';
        
        contentDiv.innerHTML = 
            '<div class="metrics-grid">' +
                '<div class="metric-card">' +
                    '<div class="metric-value">' + incidents.total_incidents + '</div>' +
                    '<div class="metric-title">Total incidents</div>' +
                    '<div class="metric-subtitle">3 derniers mois</div>' +
                '</div>' +
                '<div class="metric-card">' +
                    '<div class="metric-value">' + (incidents.critical_incidents || 0) + '</div>' +
                    '<div class="metric-title">Incidents critiques</div>' +
                '</div>' +
                '<div class="metric-card">' +
                    '<div class="metric-value">' + resolutionText + '</div>' +
                    '<div class="metric-title">Résolution moyenne</div>' +
                '</div>' +
            '</div>' +
            '<p style="text-align: center; color: #666; margin-top: 1rem;">' +
                '<i class="fas fa-database"></i> Données extraites de la table incident_filtree' +
            '</p>';
    } else {
        contentDiv.innerHTML = 
            '<div class="warning-box">' +
                '<p><i class="fas fa-exclamation-triangle"></i> Aucun incident trouvé</p>' +
                '<p><small>CI: ' + (currentChangeData.change_data.dv_cmdb_ci || 'N/A') + '</small></p>' +
            '</div>';
    }
}

function displaySimilarChanges(similar) {
    const contentDiv = document.getElementById('tab-content');
    if (!contentDiv) return;
    
    if (similar && similar.length > 0) {
        let changesHtml = '';
        
        similar.forEach(change => {
            const closeCode = change.dv_close_code || 'Inconnu';
            let className = 'similar-change ';
            let icon = '';
            
            if (closeCode === 'Succès') {
                className += 'success';
                icon = '✅';
            } else if (closeCode && closeCode.includes('Échec')) {
                className += 'error';
                icon = '❌';
            } else {
                className += 'warning';
                icon = '⚠️';
            }
            
            const durationText = change.duration_hours ? 
                " • Durée: " + change.duration_hours + "h" : '';
            
            changesHtml += 
                '<div class="' + className + '">' +
                    '<p>' +
                        '<strong>' + icon + ' ' + (change.number || 'N/A') + ' - ' + closeCode + '</strong><br>' +
                        '<small>' + (change.short_description || 'Pas de description').substring(0, 100) + '...</small><br>' +
                        '<small style="color: #666;">' +
                            'Similarité: ' + (change.similarity_score || 0) + '%' + durationText +
                        '</small>' +
                    '</p>' +
                '</div>';
        });
        
        contentDiv.innerHTML = 
            '<p><i class="fas fa-clipboard-list"></i> <strong>Historique réel des changements similaires:</strong></p>' +
            changesHtml;
    } else {
        contentDiv.innerHTML = 
            '<div class="info-box">' +
                '<p><i class="fas fa-info-circle"></i> Aucun changement similaire trouvé</p>' +
            '</div>';
    }
}

// ===================================================================
// FONCTIONS UTILITAIRES FINALES
// ===================================================================

function clearResults() {
    const analysisResults = document.getElementById('analysis-results');
    const detailedResults = document.getElementById('detailed-results');
    
    if (analysisResults) analysisResults.innerHTML = '';
    if (detailedResults) detailedResults.innerHTML = '';
    
    currentChangeData = null;
    window.currentChangeData = null;
}

function setExampleChange() {
    const input = document.getElementById('change-ref');
    if (input) {
        input.value = 'CAGIPCHG0290677';
        validateChangeReference();
        showToast('Référence d\'exemple insérée', 'success');
    }
}

// ===================================================================
// FONCTION DE DEBUG CORRIGÉE
// ===================================================================

async function debugURLs() {
    console.log("🔍 DEBUG URLs:");
    console.log("Current URL:", window.location.href);
    console.log("Origin:", window.location.origin);
    console.log("Pathname:", window.location.pathname);
    
    // Test des endpoints
    const endpoints = ['health', 'get_model_status', 'get_connection_status'];
    
    for (const endpoint of endpoints) {
        try {
            const url = "./" + endpoint;
            console.log("Testing: " + url);
            
            const response = await fetch(url, { method: 'HEAD' });
            console.log(endpoint + ": " + response.status);
            
        } catch (error) {
            console.error(endpoint + ": ERROR - " + error.message);
        }
    }
}

// Exposer pour debug
window.debugURLs = debugURLs;
window.switchTab = switchTab;
window.showDiagnostic = showDiagnostic;
window.closeDiagnostic = closeDiagnostic;
window.setExampleChange = setExampleChange;

console.log("✅ Change Risk Analyzer JavaScript chargé complètement");

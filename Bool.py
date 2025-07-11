L'utilisateur veut abandonner Dash qui ne fonctionne pas et passer à une webapp Standard avec HTML/CSS/JS/Python. C'est effectivement une bonne idée car cela semble plus simple et plus fiable.

Je dois adapter le code pour créer :

1. **HTML** : Structure de la page
2. **CSS** : Styles pour un design moderne
3. **JavaScript** : Logique frontend et appels AJAX
4. **Python** : Backend Flask avec endpoints API

L'architecture sera :
- Frontend : HTML/CSS/JS pour l'interface utilisateur
- Backend : Flask avec des endpoints pour :
  - Analyser un changement
  - Tester les connexions
  - Récupérer les statistiques d'équipe
  - Récupérer les incidents
  - Récupérer les changements similaires

Je vais reproduire la même fonctionnalité que la webapp Dash mais avec cette stack plus traditionnelle.

## 🎯 **EXCELLENTE IDÉE ! Passons à la webapp Standard**

**Dash était effectivement problématique. Voici le code pour chaque onglet :**

---

## 📁 **ONGLET HTML :**

```html
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Change Risk Analyzer</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body>
    <!-- Header principal -->
    <div class="main-header">
        <h1><i class="fas fa-search"></i> Change Risk Analyzer</h1>
        <p>Analyseur de risques pour changements ServiceNow • 100% Données Réelles</p>
    </div>

    <!-- Container principal -->
    <div class="container">
        
        <!-- Row pour le statut et la saisie -->
        <div class="main-row">
            
            <!-- Colonne gauche - Statuts -->
            <div class="sidebar">
                <h4><i class="fas fa-robot"></i> Statut du Système</h4>
                
                <!-- Statut modèle -->
                <div id="model-status" class="status-card">
                    <h6>Modèle ML</h6>
                    <p id="model-status-text">Vérification...</p>
                    <small id="model-details"></small>
                </div>
                
                <!-- Statut connexions -->
                <div id="connection-status" class="status-card">
                    <h6>Connexions ServiceNow</h6>
                    <p id="connection-status-text">Vérification...</p>
                    <small id="connection-details"></small>
                </div>
                
                <!-- Performance modèle -->
                <h5><i class="fas fa-chart-line"></i> Performance Modèle</h5>
                <div id="model-performance" class="info-box">
                    <p>Chargement...</p>
                </div>
            </div>
            
            <!-- Colonne droite - Interface principale -->
            <div class="main-content">
                <h4><i class="fas fa-edit"></i> Analyse de Changement</h4>
                
                <!-- Zone de saisie -->
                <div class="input-section">
                    <div class="input-group">
                        <label for="change-ref">Référence du changement :</label>
                        <input type="text" id="change-ref" placeholder="CHG0012345" />
                        <small>Format: CHG + 7 chiffres</small>
                    </div>
                    
                    <div class="button-group">
                        <button id="analyze-btn" class="btn-primary">
                            <i class="fas fa-search"></i> Analyser
                        </button>
                        <button id="test-btn" class="btn-secondary">
                            <i class="fas fa-info-circle"></i> Test
                        </button>
                    </div>
                </div>
                
                <!-- Zone de résultats -->
                <hr>
                <div id="analysis-results"></div>
            </div>
        </div>
        
        <!-- Zone pour les résultats détaillés -->
        <div id="detailed-results"></div>
    </div>

    <!-- Footer -->
    <footer>
        <hr>
        <div class="footer-content">
            <p><i class="fas fa-robot"></i> Change Risk Analyzer • 100% Données Réelles ServiceNow</p>
            <p><em>Tables connectées: change_request_without_empty_columns & incident_filtree</em></p>
            <p><strong>ZERO SIMULATION • ZERO DONNÉES FACTICES</strong></p>
        </div>
    </footer>

    <!-- Loading overlay -->
    <div id="loading-overlay" class="loading-overlay" style="display: none;">
        <div class="loading-spinner">
            <i class="fas fa-spinner fa-spin fa-3x"></i>
            <p>Analyse en cours...</p>
        </div>
    </div>

    <!-- Toast notifications -->
    <div id="toast-container"></div>
</body>
</html>
```

---

## 🎨 **ONGLET CSS :**

```css
/* ===================================================================
   CHANGE RISK ANALYZER - STYLES MODERNES
   ================================================================== */

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: #f8f9fa;
    color: #333;
    line-height: 1.6;
}

/* Header principal */
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.main-header h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 300;
}

.main-header p {
    margin: 0.5rem 0 0 0;
    font-size: 1.1rem;
    opacity: 0.9;
}

/* Container et layout */
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 1rem;
}

.main-row {
    display: flex;
    gap: 2rem;
    align-items: flex-start;
}

.sidebar {
    width: 30%;
    min-width: 300px;
}

.main-content {
    flex: 1;
    min-width: 0;
}

/* Status cards */
.status-card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    border: 1px solid #e9ecef;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}

.status-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

.status-card h6 {
    margin: 0 0 0.5rem 0;
    font-weight: 600;
    color: #495057;
}

.status-card p {
    margin: 0;
    font-weight: 500;
}

.status-card small {
    color: #6c757d;
    font-size: 0.875rem;
}

/* Status couleurs */
.status-success {
    border-left: 4px solid #28a745;
    background: #d4edda;
}

.status-error {
    border-left: 4px solid #dc3545;
    background: #f8d7da;
}

.status-warning {
    border-left: 4px solid #ffc107;
    background: #fff3cd;
}

/* Boxes d'information */
.info-box {
    background: #d1ecf1;
    border: 1px solid #bee5eb;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.success-box {
    background: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.warning-box {
    background: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.error-box {
    background: #f8d7da;
    border: 1px solid #f5c6cb;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

/* Zone de saisie */
.input-section {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}

.input-group {
    margin-bottom: 1rem;
}

.input-group label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: 500;
    color: #495057;
}

.input-group input {
    width: 100%;
    padding: 0.75rem;
    border: 2px solid #e9ecef;
    border-radius: 8px;
    font-size: 1rem;
    transition: border-color 0.3s ease;
}

.input-group input:focus {
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.input-group small {
    color: #6c757d;
    font-size: 0.875rem;
}

.button-group {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
}

/* Boutons */
.btn-primary {
    background: #667eea;
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}

.btn-primary:hover {
    background: #5a6fd8;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.btn-secondary {
    background: #6c757d;
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}

.btn-secondary:hover {
    background: #5a6268;
    transform: translateY(-1px);
}

.btn-tab {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    padding: 0.5rem 1rem;
    border-radius: 5px;
    margin: 0.25rem;
    cursor: pointer;
    transition: all 0.3s ease;
}

.btn-tab:hover {
    background: #e9ecef;
}

.btn-tab.active {
    background: #667eea;
    color: white;
    border-color: #667eea;
}

/* Cartes de résultats */
.risk-card {
    background: white;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border-left: 5px solid #667eea;
    margin: 1rem 0;
    text-align: center;
}

.risk-score {
    font-size: 3rem;
    font-weight: bold;
    color: #667eea;
    margin: 0;
}

.risk-level {
    font-size: 1.5rem;
    margin: 0.5rem 0;
    font-weight: 600;
}

.risk-interpretation {
    font-style: italic;
    color: #6c757d;
    margin: 0;
}

/* Métriques */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

.metric-card {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    text-align: center;
    border: 1px solid #e9ecef;
    transition: transform 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: #667eea;
    margin: 0;
}

.metric-title {
    color: #495057;
    margin: 0.5rem 0 0 0;
}

.metric-subtitle {
    color: #6c757d;
    font-size: 0.875rem;
    margin: 0;
}

/* Détails en colonnes */
.details-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    margin: 2rem 0;
}

.details-section h4 {
    color: #495057;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.features-list {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.features-list p {
    margin: 0.5rem 0;
    display: flex;
    justify-content: space-between;
}

.features-list strong {
    color: #495057;
}

/* Changements similaires */
.similar-change {
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border: 1px solid #dee2e6;
}

.similar-change.success {
    background: #d4edda;
    border-color: #c3e6cb;
}

.similar-change.error {
    background: #f8d7da;
    border-color: #f5c6cb;
}

.similar-change.warning {
    background: #fff3cd;
    border-color: #ffeaa7;
}

/* Onglets */
.tabs-container {
    margin: 2rem 0;
}

.tabs-buttons {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}

.tab-content {
    padding: 1rem;
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    min-height: 200px;
}

/* Loading overlay */
.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.5);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 9999;
}

.loading-spinner {
    background: white;
    padding: 2rem;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

.loading-spinner i {
    color: #667eea;
    margin-bottom: 1rem;
}

/* Toast notifications */
#toast-container {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 10000;
}

.toast {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    margin-bottom: 1rem;
    min-width: 300px;
    border-left: 4px solid #28a745;
    animation: slideIn 0.3s ease;
}

.toast.error {
    border-left-color: #dc3545;
}

.toast.warning {
    border-left-color: #ffc107;
}

@keyframes slideIn {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

/* Footer */
footer {
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 1px solid #dee2e6;
}

.footer-content {
    text-align: center;
    color: #6c757d;
    padding: 2rem;
}

.footer-content p {
    margin: 0.25rem 0;
}

/* Responsive */
@media (max-width: 768px) {
    .main-row {
        flex-direction: column;
    }
    
    .sidebar {
        width: 100%;
    }
    
    .details-grid {
        grid-template-columns: 1fr;
    }
    
    .metrics-grid {
        grid-template-columns: 1fr;
    }
    
    .button-group {
        justify-content: center;
    }
    
    .main-header h1 {
        font-size: 2rem;
    }
}

/* Utilities */
.text-center { text-align: center; }
.text-left { text-align: left; }
.text-right { text-align: right; }
.mt-1 { margin-top: 0.5rem; }
.mt-2 { margin-top: 1rem; }
.mt-3 { margin-top: 1.5rem; }
.mb-1 { margin-bottom: 0.5rem; }
.mb-2 { margin-bottom: 1rem; }
.mb-3 { margin-bottom: 1.5rem; }
.hidden { display: none; }
.visible { display: block; }
```

---

## ⚡ **ONGLET JAVASCRIPT :**

```javascript
// ===================================================================
// CHANGE RISK ANALYZER - LOGIQUE FRONTEND
// ===================================================================

// Variables globales
let currentChangeData = null;
let activeTab = 'team';

// Initialisation au chargement de la page
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    checkSystemStatus();
});

// ===================================================================
// INITIALISATION
// ===================================================================

function initializeApp() {
    console.log('🚀 Initialisation de Change Risk Analyzer');
    
    // Afficher un message de bienvenue
    showToast('Bienvenue dans Change Risk Analyzer', 'success');
}

function setupEventListeners() {
    // Bouton analyser
    document.getElementById('analyze-btn').addEventListener('click', analyzeChange);
    
    // Bouton test
    document.getElementById('test-btn').addEventListener('click', testConnection);
    
    // Input changement (validation en temps réel)
    document.getElementById('change-ref').addEventListener('input', validateChangeReference);
    
    // Touche Entrée pour lancer l'analyse
    document.getElementById('change-ref').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            analyzeChange();
        }
    });
}

// ===================================================================
// FONCTIONS UTILITAIRES
// ===================================================================

function showLoading(message = 'Chargement...') {
    const overlay = document.getElementById('loading-overlay');
    const spinner = overlay.querySelector('.loading-spinner p');
    spinner.textContent = message;
    overlay.style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}

function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span>${message}</span>
            <button onclick="this.parentElement.parentElement.remove()" style="background: none; border: none; font-size: 1.2rem; cursor: pointer;">&times;</button>
        </div>
    `;
    
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
    const value = input.value.toUpperCase();
    const pattern = /^CHG\d{7}$/;
    
    if (value && !pattern.test(value)) {
        input.style.borderColor = '#dc3545';
        return false;
    } else {
        input.style.borderColor = '#28a745';
        return true;
    }
}

// ===================================================================
// API CALLS
// ===================================================================

async function apiCall(endpoint, params = {}) {
    try {
        const url = new URL(getWebAppBackendUrl(endpoint));
        Object.keys(params).forEach(key => url.searchParams.append(key, params[key]));
        
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.status === 'error') {
            throw new Error(data.message || 'Erreur inconnue');
        }
        
        return data;
        
    } catch (error) {
        console.error(`Erreur API ${endpoint}:`, error);
        throw error;
    }
}

// ===================================================================
// FONCTIONS PRINCIPALES
// ===================================================================

async function checkSystemStatus() {
    try {
        // Vérifier le statut du modèle
        const modelStatus = await apiCall('get_model_status');
        updateModelStatus(modelStatus);
        
        // Vérifier le statut des connexions
        const connectionStatus = await apiCall('get_connection_status');
        updateConnectionStatus(connectionStatus);
        
    } catch (error) {
        console.error('Erreur vérification statut:', error);
        showToast('Erreur lors de la vérification du statut système', 'error');
    }
}

function updateModelStatus(status) {
    const statusCard = document.getElementById('model-status');
    const statusText = document.getElementById('model-status-text');
    const details = document.getElementById('model-details');
    const performance = document.getElementById('model-performance');
    
    if (status.data.status === 'Modèle chargé') {
        statusCard.className = 'status-card status-success';
        statusText.textContent = '✅ Opérationnel';
        details.textContent = `Algorithme: ${status.data.algorithm}`;
        
        // Performance
        const perf = status.data.training_info?.performance || {};
        performance.innerHTML = `
            <p><strong>Recall:</strong> ${perf.recall || 'N/A'}</p>
            <p><strong>Precision:</strong> ${perf.precision || 'N/A'}</p>
            <p><strong>Features:</strong> ${status.data.features?.count || 'N/A'}</p>
        `;
    } else {
        statusCard.className = 'status-card status-error';
        statusText.textContent = '❌ Non disponible';
        details.textContent = 'Modèle non chargé';
        performance.innerHTML = '<p>❌ Non disponible</p>';
    }
}

function updateConnectionStatus(status) {
    const statusCard = document.getElementById('connection-status');
    const statusText = document.getElementById('connection-status-text');
    const details = document.getElementById('connection-details');
    
    if (status.data.status === 'Connecté') {
        statusCard.className = 'status-card status-success';
        statusText.textContent = '✅ Connecté';
        details.textContent = 'Tables: change_request & incident_filtree';
    } else {
        statusCard.className = 'status-card status-error';
        statusText.textContent = '❌ Erreur';
        details.textContent = status.data.error || 'Erreur de connexion';
    }
}

async function testConnection() {
    try {
        showLoading('Test des connexions...');
        
        const result = await apiCall('test_connection');
        
        hideLoading();
        
        if (result.data.success) {
            showToast('✅ Test de connexion réussi', 'success');
        } else {
            showToast('❌ Test de connexion échoué', 'error');
        }
        
    } catch (error) {
        hideLoading();
        showToast(`❌ Erreur de test: ${error.message}`, 'error');
    }
}

async function analyzeChange() {
    const changeRef = document.getElementById('change-ref').value.trim().toUpperCase();
    
    // Validation
    if (!changeRef) {
        showToast('⚠️ Veuillez saisir une référence de changement', 'warning');
        return;
    }
    
    if (!validateChangeReference()) {
        showToast('❌ Format invalide. Utilisez CHG + 7 chiffres', 'error');
        return;
    }
    
    try {
        showLoading(`Analyse de ${changeRef} en cours...`);
        
        // Appel API pour analyser le changement
        const result = await apiCall('analyze_change', { change_ref: changeRef });
        
        hideLoading();
        
        if (result.data.change_found) {
            currentChangeData = result.data;
            displayAnalysisResults(result.data);
            showToast(`✅ Analyse de ${changeRef} terminée`, 'success');
        } else {
            showToast(`❌ Changement ${changeRef} non trouvé`, 'error');
            clearResults();
        }
        
    } catch (error) {
        hideLoading();
        showToast(`❌ Erreur d'analyse: ${error.message}`, 'error');
        clearResults();
    }
}

// ===================================================================
// AFFICHAGE DES RÉSULTATS
// ===================================================================

function displayAnalysisResults(data) {
    // Résultats principaux
    displayMainResults(data);
    
    // Résultats détaillés
    displayDetailedResults(data);
}

function displayMainResults(data) {
    const resultsDiv = document.getElementById('analysis-results');
    const analysis = data.detailed_analysis;
    
    resultsDiv.innerHTML = `
        <div class="text-center mb-3">
            <h3><i class="fas fa-chart-pie"></i> Analyse de ${data.change_ref}</h3>
        </div>
        
        <div class="risk-card">
            <div class="risk-score">${analysis.risk_color} ${analysis.risk_score}%</div>
            <div class="risk-level">Risque d'échec</div>
            <div class="risk-level">Niveau: ${analysis.risk_level}</div>
            <div class="risk-interpretation">${analysis.interpretation}</div>
        </div>
    `;
}

function displayDetailedResults(data) {
    const detailedDiv = document.getElementById('detailed-results');
    const analysis = data.detailed_analysis;
    const change = data.change_data;
    
    detailedDiv.innerHTML = `
        <div class="details-grid">
            <!-- Colonne gauche -->
            <div class="details-section">
                <h4><i class="fas fa-exclamation-triangle"></i> Facteurs de risque</h4>
                <div class="warning-box">
                    ${analysis.risk_factors.length > 0 ? 
                        '<ul>' + analysis.risk_factors.map(factor => `<li>${factor}</li>`).join('') + '</ul>' :
                        '<p>Aucun facteur spécifique détecté</p>'
                    }
                </div>
                
                <h4><i class="fas fa-lightbulb"></i> Recommandations</h4>
                <div class="success-box">
                    <ul>
                        ${analysis.recommendations.map(rec => `<li>✅ ${rec}</li>`).join('')}
                    </ul>
                </div>
            </div>
            
            <!-- Colonne droite -->
            <div class="details-section">
                <h4><i class="fas fa-cogs"></i> Caractéristiques techniques</h4>
                <div class="features-list">
                    <p><strong>Type SILCA:</strong> <span>${change.dv_u_type_change_silca || 'N/A'}</span></p>
                    <p><strong>Type de changement:</strong> <span>${change.dv_type || 'N/A'}</span></p>
                    <p><strong>Nombre de CAB:</strong> <span>${change.u_cab_count || 'N/A'}</span></p>
                    <p><strong>Périmètre BCR:</strong> <span>${change.u_bcr ? '✅' : '❌'}</span></p>
                    <p><strong>Périmètre BPC:</strong> <span>${change.u_bpc ? '✅' : '❌'}</span></p>
                </div>
                
                <h4><i class="fas fa-clipboard-list"></i> Métadonnées</h4>
                <div class="info-box">
                    <p><strong>Équipe:</strong> ${change.dv_assignment_group || 'N/A'}</p>
                    <p><strong>CI/Solution:</strong> ${change.dv_cmdb_ci || 'N/A'}</p>
                    <p><strong>Catégorie:</strong> ${change.dv_category || 'N/A'}</p>
                    <p><strong>État:</strong> ${change.dv_state || 'N/A'}</p>
                </div>
            </div>
        </div>
        
        <!-- Onglets contextuels -->
        <div class="tabs-container">
            <hr>
            <h3><i class="fas fa-chart-line"></i> Informations contextuelles</h3>
            <p style="color: #666; font-style: italic;"><i class="fas fa-database"></i> Données extraites des tables ServiceNow réelles</p>
            
            <div class="tabs-buttons">
                <button class="btn-tab active" onclick="switchTab('team')">
                    <i class="fas fa-users"></i> Statistiques équipe
                </button>
                <button class="btn-tab" onclick="switchTab('incidents')">
                    <i class="fas fa-tools"></i> Incidents liés
                </button>
                <button class="btn-tab" onclick="switchTab('similar')">
                    <i class="fas fa-clipboard-list"></i> Changements similaires
                </button>
            </div>
            
            <div id="tab-content" class="tab-content">
                <p style="text-align: center; color: #666; padding: 2rem;">
                    <i class="fas fa-mouse-pointer"></i> Cliquez sur un onglet pour voir les informations contextuelles
                </p>
            </div>
        </div>
    `;
}

async function switchTab(tabName) {
    if (!currentChangeData) return;
    
    activeTab = tabName;
    
    // Mettre à jour les boutons
    document.querySelectorAll('.btn-tab').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    const contentDiv = document.getElementById('tab-content');
    
    try {
        showLoading('Chargement des données contextuelles...');
        
        let result;
        
        switch(tabName) {
            case 'team':
                result = await apiCall('get_team_stats', { 
                    assignment_group: currentChangeData.change_data.dv_assignment_group 
                });
                displayTeamStats(result.data);
                break;
                
            case 'incidents':
                result = await apiCall('get_incidents', { 
                    cmdb_ci: currentChangeData.change_data.dv_cmdb_ci 
                });
                displayIncidents(result.data);
                break;
                
            case 'similar':
                result = await apiCall('get_similar_changes', { 
                    change_ref: currentChangeData.change_ref 
                });
                displaySimilarChanges(result.data);
                break;
        }
        
        hideLoading();
        
    } catch (error) {
        hideLoading();
        contentDiv.innerHTML = `
            <div class="error-box">
                <p><i class="fas fa-exclamation-circle"></i> Erreur lors du chargement: ${error.message}</p>
            </div>
        `;
    }
}

function displayTeamStats(stats) {
    const contentDiv = document.getElementById('tab-content');
    
    if (stats && !stats.error) {
        const lastFailureText = stats.last_failure_date ? 
            `Il y a ${Math.floor((new Date() - new Date(stats.last_failure_date)) / (1000 * 60 * 60 * 24))} jours` :
            'Aucun récent';
        
        contentDiv.innerHTML = `
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">${stats.total_changes}</div>
                    <div class="metric-title">Total changements</div>
                    <div class="metric-subtitle">6 derniers mois</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${stats.success_rate}%</div>
                    <div class="metric-title">Taux de succès</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${stats.failures}</div>
                    <div class="metric-title">Échecs</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${lastFailureText}</div>
                    <div class="metric-title">Dernier échec</div>
                </div>
            </div>
        `;
    } else {
        contentDiv.innerHTML = `
            <div class="warning-box">
                <p><i class="fas fa-exclamation-triangle"></i> Statistiques équipe non disponibles</p>
            </div>
        `;
    }
}

function displayIncidents(incidents) {
    const contentDiv = document.getElementById('tab-content');
    
    if (incidents) {
        const resolutionText = incidents.avg_resolution_hours > 0 ? 
            `${incidents.avg_resolution_hours}h` : 'N/A';
        
        contentDiv.innerHTML = `
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">${incidents.total_incidents}</div>
                    <div class="metric-title">Total incidents</div>
                    <div class="metric-subtitle">3 derniers mois</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${incidents.critical_incidents}</div>
                    <div class="metric-title">Incidents critiques</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${resolutionText}</div>
                    <div class="metric-title">Résolution moyenne</div>
                </div>
            </div>
            <p style="text-align: center; color: #666; margin-top: 1rem;">
                <i class="fas fa-database"></i> Données extraites de la table incident_filtree
            </p>
        `;
    } else {
        contentDiv.innerHTML = `
            <div class="warning-box">
                <p><i class="fas fa-exclamation-triangle"></i> Aucun incident trouvé pour cette solution/CI</p>
            </div>
        `;
    }
}

function displaySimilarChanges(similar) {
    const contentDiv = document.getElementById('tab-content');
    
    if (similar && similar.length > 0) {
        const changesHtml = similar.map(change => {
            const closeCode = change.dv_close_code;
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
                ` • Durée: ${change.duration_hours}h` : '';
            
            return `
                <div class="${className}">
                    <p>
                        <strong>${icon} ${change.number} - ${closeCode}</strong><br>
                        <small>${change.short_description.substring(0, 100)}...</small><br>
                        <small style="color: #666;">
                            Similarité: ${change.similarity_score}%${durationText}
                        </small>
                    </p>
                </div>
            `;
        }).join('');
        
        contentDiv.innerHTML = `
            <p><i class="fas fa-clipboard-list"></i> <strong>Historique réel des changements similaires:</strong></p>
            ${changesHtml}
        `;
    } else {
        contentDiv.innerHTML = `
            <div class="info-box">
                <p><i class="fas fa-info-circle"></i> Aucun changement similaire trouvé dans l'historique</p>
            </div>
        `;
    }
}

function clearResults() {
    document.getElementById('analysis-results').innerHTML = '';
    document.getElementById('detailed-results').innerHTML = '';
    currentChangeData = null;
}

// ===================================================================
// FONCTIONS UTILITAIRES SUPPLÉMENTAIRES
// ===================================================================

// Auto-complétion majuscules pour le champ changement
document.addEventListener('DOMContentLoaded', function() {
    const changeInput = document.getElementById('change-ref');
    if (changeInput) {
        changeInput.addEventListener('input', function() {
            this.value = this.value.toUpperCase();
        });
    }
});
```

---

## 🐍 **ONGLET PYTHON :**

```python
"""
WEBAPP BACKEND - ANALYSE DES RISQUES DE CHANGEMENT SERVICENOW
Flask backend avec endpoints API
"""

import dataiku
import pandas as pd
import json
import traceback
from flask import request, jsonify
from datetime import datetime

# Import de nos modules
from change_risk_predictor import ChangeRiskPredictor
from servicenow_connector import ServiceNowConnector

# ===================================================================
# INITIALISATION DES COMPOSANTS
# ===================================================================

# Initialisation globale
try:
    predictor = ChangeRiskPredictor()
    connector = ServiceNowConnector()
    init_success = True
    init_error = ""
except Exception as e:
    predictor = None
    connector = None
    init_success = False
    init_error = str(e)

# ===================================================================
# UTILITAIRES
# ===================================================================

def create_response(data=None, status="ok", message=""):
    """Créer une réponse API standardisée"""
    response = {
        "status": status,
        "message": message,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }
    return json.dumps(response)

def handle_error(error, endpoint_name):
    """Gestion standardisée des erreurs"""
    error_msg = str(error)
    print(f"❌ Erreur dans {endpoint_name}: {error_msg}")
    print(traceback.format_exc())
    
    return create_response(
        status="error",
        message=f"Erreur dans {endpoint_name}: {error_msg}"
    )

# ===================================================================
# ENDPOINTS API
# ===================================================================

@app.route('/get_model_status')
def get_model_status():
    """Récupérer le statut du modèle ML"""
    
    try:
        if not init_success:
            return create_response(
                data={"status": "Erreur d'initialisation", "error": init_error},
                status="error"
            )
        
        model_info = predictor.get_model_info()
        
        return create_response(data=model_info)
        
    except Exception as e:
        return handle_error(e, "get_model_status")

@app.route('/get_connection_status')
def get_connection_status():
    """Vérifier le statut des connexions ServiceNow"""
    
    try:
        if not init_success:
            return create_response(
                data={"status": "Erreur", "error": init_error},
                status="error"
            )
        
        connection_status = connector.get_connection_status()
        
        return create_response(data=connection_status)
        
    except Exception as e:
        return handle_error(e, "get_connection_status")

@app.route('/test_connection')
def test_connection():
    """Tester les connexions système"""
    
    try:
        if not init_success:
            return create_response(
                data={"success": False, "error": init_error},
                status="error"
            )
        
        # Test du modèle
        model_info = predictor.get_model_info()
        model_ok = model_info.get("status") == "Modèle chargé"
        
        # Test des connexions
        connection_status = connector.get_connection_status()
        connection_ok = connection_status.get("status") == "Connecté"
        
        success = model_ok and connection_ok
        
        return create_response(data={
            "success": success,
            "model_status": model_ok,
            "connection_status": connection_ok,
            "details": {
                "model": model_info,
                "connection": connection_status
            }
        })
        
    except Exception as e:
        return handle_error(e, "test_connection")

@app.route('/analyze_change')
def analyze_change():
    """Analyser un changement spécifique"""
    
    try:
        if not init_success:
            return create_response(
                status="error",
                message="Système non initialisé"
            )
        
        # Récupérer la référence du changement
        change_ref = request.args.get('change_ref')
        
        if not change_ref:
            return create_response(
                status="error", 
                message="Référence de changement manquante"
            )
        
        # Validation du format
        if not connector.validate_change_reference(change_ref):
            return create_response(
                status="error",
                message="Format de référence invalide"
            )
        
        # Récupération des données
        change_data = connector.get_change_data(change_ref)
        
        if not change_data:
            return create_response(data={
                "change_found": False,
                "change_ref": change_ref
            })
        
        # Analyse ML
        detailed_analysis = predictor.get_detailed_analysis(change_data)
        
        return create_response(data={
            "change_found": True,
            "change_ref": change_ref,
            "change_data": change_data,
            "detailed_analysis": detailed_analysis
        })
        
    except Exception as e:
        return handle_error(e, "analyze_change")

@app.route('/get_team_stats')
def get_team_stats():
    """Récupérer les statistiques d'une équipe"""
    
    try:
        if not init_success:
            return create_response(
                status="error",
                message="Système non initialisé"
            )
        
        assignment_group = request.args.get('assignment_group')
        
        if not assignment_group:
            return create_response(
                status="error",
                message="Nom d'équipe manquant"
            )
        
        team_stats = connector.get_team_statistics(assignment_group)
        
        return create_response(data=team_stats)
        
    except Exception as e:
        return handle_error(e, "get_team_stats")

@app.route('/get_incidents')
def get_incidents():
    """Récupérer les incidents liés à un CI"""
    
    try:
        if not init_success:
            return create_response(
                status="error",
                message="Système non initialisé"
            )
        
        cmdb_ci = request.args.get('cmdb_ci')
        
        if not cmdb_ci:
            return create_response(
                status="error",
                message="CI manquant"
            )
        
        incidents_data = connector.get_solution_incidents(cmdb_ci)
        
        return create_response(data=incidents_data)
        
    except Exception as e:
        return handle_error(e, "get_incidents")

@app.route('/get_similar_changes')
def get_similar_changes():
    """Récupérer les changements similaires"""
    
    try:
        if not init_success:
            return create_response(
                status="error",
                message="Système non initialisé"
            )
        
        change_ref = request.args.get('change_ref')
        
        if not change_ref:
            return create_response(
                status="error",
                message="Référence de changement manquante"
            )
        
        # Récupérer les données du changement d'abord
        change_data = connector.get_change_data(change_ref)
        
        if not change_data:
            return create_response(
                status="error",
                message="Changement non trouvé"
            )
        
        # Chercher les changements similaires
        similar_changes = connector.find_similar_changes(change_data)
        
        return create_response(data=similar_changes)
        
    except Exception as e:
        return handle_error(e, "get_similar_changes")

# ===================================================================
# ENDPOINT DE DIAGNOSTIC
# ===================================================================

@app.route('/diagnostic')
def diagnostic():
    """Endpoint de diagnostic complet du système"""
    
    try:
        diagnostic_info = {
            "initialization": {
                "success": init_success,
                "error": init_error if not init_success else None
            },
            "components": {},
            "datasets": {},
            "timestamp": datetime.now().isoformat()
        }
        
        if init_success:
            # Test du prédicteur
            try:
                model_info = predictor.get_model_info()
                diagnostic_info["components"]["predictor"] = {
                    "status": "OK",
                    "details": model_info
                }
            except Exception as e:
                diagnostic_info["components"]["predictor"] = {
                    "status": "ERROR",
                    "error": str(e)
                }
            
            # Test du connecteur
            try:
                connection_status = connector.get_connection_status()
                diagnostic_info["components"]["connector"] = {
                    "status": "OK",
                    "details": connection_status
                }
            except Exception as e:
                diagnostic_info["components"]["connector"] = {
                    "status": "ERROR",
                    "error": str(e)
                }
            
            # Test des datasets
            datasets_to_test = [
                "change_request_without_empty_columns",
                "incident_filtree"
            ]
            
            for dataset_name in datasets_to_test:
                try:
                    dataset = dataiku.Dataset(dataset_name)
                    df = dataset.get_dataframe(limit=1)
                    diagnostic_info["datasets"][dataset_name] = {
                        "status": "OK",
                        "columns": len(df.columns),
                        "sample_available": True
                    }
                except Exception as e:
                    diagnostic_info["datasets"][dataset_name] = {
                        "status": "ERROR",
                        "error": str(e)
                    }
        
        return create_response(data=diagnostic_info)
        
    except Exception as e:
        return handle_error(e, "diagnostic")

# ===================================================================
# ENDPOINT DE SANTÉ
# ===================================================================

@app.route('/health')
def health():
    """Endpoint de santé simple"""
    
    return create_response(data={
        "status": "alive",
        "initialization": init_success,
        "timestamp": datetime.now().isoformat()
    })

# ===================================================================
# GESTION DES ERREURS GLOBALES
# ===================================================================

@app.errorhandler(404)
def not_found(error):
    return create_response(
        status="error",
        message="Endpoint non trouvé"
    ), 404

@app.errorhandler(500)
def internal_error(error):
    return create_response(
        status="error",
        message="Erreur interne du serveur"
    ), 500

# ===================================================================
# POINT D'ENTRÉE
# ===================================================================

if __name__ == '__main__':
    print("🚀 Démarrage du backend Change Risk Analyzer")
    print(f"Initialisation: {'✅ OK' if init_success else '❌ ERROR'}")
    if not init_success:
        print(f"Erreur: {init_error}")
```

## 🎯 **MAINTENANT VOUS AVEZ UNE WEBAPP COMPLÈTE !**

### **✅ AVANTAGES DE CETTE APPROCHE :**
- **Plus stable** que Dash
- **Interface moderne** et responsive
- **API REST** claire et documentée
- **Gestion d'erreurs** robuste
- **100% données réelles** ServiceNow

### **🚀 PROCHAINES ÉTAPES :**
1. **Sauvegarder** chaque onglet avec son code
2. **Démarrer** la webapp
3. **Tester** les fonctionnalités

**Cette webapp devrait marcher du premier coup !** 🎯

J'ai créé une webapp complète avec HTML/CSS/JavaScript/Python qui reproduit toutes les fonctionnalités de la version Dash mais avec une architecture plus stable et traditionnelle.

**Structure:**

1. **HTML:** Interface utilisateur complète avec structure sémantique
2. **CSS:** Design moderne avec gradients, animations, responsive design
3. **JavaScript:** Logique frontend avec appels AJAX, gestion d'état, validation
4. **Python:** Backend Flask avec endpoints REST clairs

**Fonctionnalités:**
- Vérification du statut système
- Analyse de changements en temps réel
- Affichage des résultats avec score de risque
- Onglets contextuels (statistiques équipe, incidents, changements similaires)
- Gestion d'erreurs robuste
- Interface responsive
- Notifications toast
- Loading states

**Avantages par rapport à Dash:**
- Plus stable (pas de problèmes de layout)
- Plus de contrôle sur l'interface
- Meilleure performance
- Plus facile à déboguer
- Architecture claire frontend/backend

L'utilisateur devrait pouvoir copier-coller chaque code dans l'onglet correspondant et avoir une webapp fonctionnelle immédiatement.L'utilisateur veut abandonner Dash qui ne fonctionne pas et passer à une webapp Standard avec HTML/CSS/JS/Python. C'est effectivement une bonne idée car cela semble plus simple et plus fiable.

Je dois adapter le code pour créer :

1. **HTML** : Structure de la page
2. **CSS** : Styles pour un design moderne
3. **JavaScript** : Logique frontend et appels AJAX
4. **Python** : Backend Flask avec endpoints API

L'architecture sera :
- Frontend : HTML/CSS/JS pour l'interface utilisateur
- Backend : Flask avec des endpoints pour :
  - Analyser un changement
  - Tester les connexions
  - Récupérer les statistiques d'équipe
  - Récupérer les incidents
  - Récupérer les changements similaires

Je vais reproduire la même fonctionnalité que la webapp Dash mais avec cette stack plus traditionnelle.

## 🎯 **EXCELLENTE IDÉE ! Passons à la webapp Standard**

**Dash était effectivement problématique. Voici le code pour chaque onglet :**

---

## 📁 **ONGLET HTML :**

```html
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Change Risk Analyzer</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body>
    <!-- Header principal -->
    <div class="main-header">
        <h1><i class="fas fa-search"></i> Change Risk Analyzer</h1>
        <p>Analyseur de risques pour changements ServiceNow • 100% Données Réelles</p>
    </div>

    <!-- Container principal -->
    <div class="container">
        
        <!-- Row pour le statut et la saisie -->
        <div class="main-row">
            
            <!-- Colonne gauche - Statuts -->
            <div class="sidebar">
                <h4><i class="fas fa-robot"></i> Statut du Système</h4>
                
                <!-- Statut modèle -->
                <div id="model-status" class="status-card">
                    <h6>Modèle ML</h6>
                    <p id="model-status-text">Vérification...</p>
                    <small id="model-details"></small>
                </div>
                
                <!-- Statut connexions -->
                <div id="connection-status" class="status-card">
                    <h6>Connexions ServiceNow</h6>
                    <p id="connection-status-text">Vérification...</p>
                    <small id="connection-details"></small>
                </div>
                
                <!-- Performance modèle -->
                <h5><i class="fas fa-chart-line"></i> Performance Modèle</h5>
                <div id="model-performance" class="info-box">
                    <p>Chargement...</p>
                </div>
            </div>
            
            <!-- Colonne droite - Interface principale -->
            <div class="main-content">
                <h4><i class="fas fa-edit"></i> Analyse de Changement</h4>
                
                <!-- Zone de saisie -->
                <div class="input-section">
                    <div class="input-group">
                        <label for="change-ref">Référence du changement :</label>
                        <input type="text" id="change-ref" placeholder="CHG0012345" />
                        <small>Format: CHG + 7 chiffres</small>
                    </div>
                    
                    <div class="button-group">
                        <button id="analyze-btn" class="btn-primary">
                            <i class="fas fa-search"></i> Analyser
                        </button>
                        <button id="test-btn" class="btn-secondary">
                            <i class="fas fa-info-circle"></i> Test
                        </button>
                    </div>
                </div>
                
                <!-- Zone de résultats -->
                <hr>
                <div id="analysis-results"></div>
            </div>
        </div>
        
        <!-- Zone pour les résultats détaillés -->
        <div id="detailed-results"></div>
    </div>

    <!-- Footer -->
    <footer>
        <hr>
        <div class="footer-content">
            <p><i class="fas fa-robot"></i> Change Risk Analyzer • 100% Données Réelles ServiceNow</p>
            <p><em>Tables connectées: change_request_without_empty_columns & incident_filtree</em></p>
            <p><strong>ZERO SIMULATION • ZERO DONNÉES FACTICES</strong></p>
        </div>
    </footer>

    <!-- Loading overlay -->
    <div id="loading-overlay" class="loading-overlay" style="display: none;">
        <div class="loading-spinner">
            <i class="fas fa-spinner fa-spin fa-3x"></i>
            <p>Analyse en cours...</p>
        </div>
    </div>

    <!-- Toast notifications -->
    <div id="toast-container"></div>
</body>
</html>
```

---

## 🎨 **ONGLET CSS :**

```css
/* ===================================================================
   CHANGE RISK ANALYZER - STYLES MODERNES
   ================================================================== */

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: #f8f9fa;
    color: #333;
    line-height: 1.6;
}

/* Header principal */
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.main-header h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 300;
}

.main-header p {
    margin: 0.5rem 0 0 0;
    font-size: 1.1rem;
    opacity: 0.9;
}

/* Container et layout */
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 1rem;
}

.main-row {
    display: flex;
    gap: 2rem;
    align-items: flex-start;
}

.sidebar {
    width: 30%;
    min-width: 300px;
}

.main-content {
    flex: 1;
    min-width: 0;
}

/* Status cards */
.status-card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    border: 1px solid #e9ecef;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}

.status-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

.status-card h6 {
    margin: 0 0 0.5rem 0;
    font-weight: 600;
    color: #495057;
}

.status-card p {
    margin: 0;
    font-weight: 500;
}

.status-card small {
    color: #6c757d;
    font-size: 0.875rem;
}

/* Status couleurs */
.status-success {
    border-left: 4px solid #28a745;
    background: #d4edda;
}

.status-error {
    border-left: 4px solid #dc3545;
    background: #f8d7da;
}

.status-warning {
    border-left: 4px solid #ffc107;
    background: #fff3cd;
}

/* Boxes d'information */
.info-box {
    background: #d1ecf1;
    border: 1px solid #bee5eb;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.success-box {
    background: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.warning-box {
    background: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.error-box {
    background: #f8d7da;
    border: 1px solid #f5c6cb;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

/* Zone de saisie */
.input-section {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}

.input-group {
    margin-bottom: 1rem;
}

.input-group label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: 500;
    color: #495057;
}

.input-group input {
    width: 100%;
    padding: 0.75rem;
    border: 2px solid #e9ecef;
    border-radius: 8px;
    font-size: 1rem;
    transition: border-color 0.3s ease;
}

.input-group input:focus {
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.input-group small {
    color: #6c757d;
    font-size: 0.875rem;
}

.button-group {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
}

/* Boutons */
.btn-primary {
    background: #667eea;
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}

.btn-primary:hover {
    background: #5a6fd8;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.btn-secondary {
    background: #6c757d;
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}

.btn-secondary:hover {
    background: #5a6268;
    transform: translateY(-1px);
}

.btn-tab {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    padding: 0.5rem 1rem;
    border-radius: 5px;
    margin: 0.25rem;
    cursor: pointer;
    transition: all 0.3s ease;
}

.btn-tab:hover {
    background: #e9ecef;
}

.btn-tab.active {
    background: #667eea;
    color: white;
    border-color: #667eea;
}

/* Cartes de résultats */
.risk-card {
    background: white;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border-left: 5px solid #667eea;
    margin: 1rem 0;
    text-align: center;
}

.risk-score {
    font-size: 3rem;
    font-weight: bold;
    color: #667eea;
    margin: 0;
}

.risk-level {
    font-size: 1.5rem;
    margin: 0.5rem 0;
    font-weight: 600;
}

.risk-interpretation {
    font-style: italic;
    color: #6c757d;
    margin: 0;
}

/* Métriques */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

.metric-card {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    text-align: center;
    border: 1px solid #e9ecef;
    transition: transform 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: #667eea;
    margin: 0;
}

.metric-title {
    color: #495057;
    margin: 0.5rem 0 0 0;
}

.metric-subtitle {
    color: #6c757d;
    font-size: 0.875rem;
    margin: 0;
}

/* Détails en colonnes */
.details-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    margin: 2rem 0;
}

.details-section h4 {
    color: #495057;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.features-list {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.features-list p {
    margin: 0.5rem 0;
    display: flex;
    justify-content: space-between;
}

.features-list strong {
    color: #495057;
}

/* Changements similaires */
.similar-change {
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border: 1px solid #dee2e6;
}

.similar-change.success {
    background: #d4edda;
    border-color: #c3e6cb;
}

.similar-change.error {
    background: #f8d7da;
    border-color: #f5c6cb;
}

.similar-change.warning {
    background: #fff3cd;
    border-color: #ffeaa7;
}

/* Onglets */
.tabs-container {
    margin: 2rem 0;
}

.tabs-buttons {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}

.tab-content {
    padding: 1rem;
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    min-height: 200px;
}

/* Loading overlay */
.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.5);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 9999;
}

.loading-spinner {
    background: white;
    padding: 2rem;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

.loading-spinner i {
    color: #667eea;
    margin-bottom: 1rem;
}

/* Toast notifications */
#toast-container {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 10000;
}

.toast {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    margin-bottom: 1rem;
    min-width: 300px;
    border-left: 4px solid #28a745;
    animation: slideIn 0.3s ease;
}

.toast.error {
    border-left-color: #dc3545;
}

.toast.warning {
    border-left-color: #ffc107;
}

@keyframes slideIn {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

/* Footer */
footer {
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 1px solid #dee2e6;
}

.footer-content {
    text-align: center;
    color: #6c757d;
    padding: 2rem;
}

.footer-content p {
    margin: 0.25rem 0;
}

/* Responsive */
@media (max-width: 768px) {
    .main-row {
        flex-direction: column;
    }
    
    .sidebar {
        width: 100%;
    }
    
    .details-grid {
        grid-template-columns: 1fr;
    }
    
    .metrics-grid {
        grid-template-columns: 1fr;
    }
    
    .button-group {
        justify-content: center;
    }
    
    .main-header h1 {
        font-size: 2rem;
    }
}

/* Utilities */
.text-center { text-align: center; }
.text-left { text-align: left; }
.text-right { text-align: right; }
.mt-1 { margin-top: 0.5rem; }
.mt-2 { margin-top: 1rem; }
.mt-3 { margin-top: 1.5rem; }
.mb-1 { margin-bottom: 0.5rem; }
.mb-2 { margin-bottom: 1rem; }
.mb-3 { margin-bottom: 1.5rem; }
.hidden { display: none; }
.visible { display: block; }
```

---

## ⚡ **ONGLET JAVASCRIPT :**

```javascript
// ===================================================================
// CHANGE RISK ANALYZER - LOGIQUE FRONTEND
// ===================================================================

// Variables globales
let currentChangeData = null;
let activeTab = 'team';

// Initialisation au chargement de la page
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    checkSystemStatus();
});

// ===================================================================
// INITIALISATION
// ===================================================================

function initializeApp() {
    console.log('🚀 Initialisation de Change Risk Analyzer');
    
    // Afficher un message de bienvenue
    showToast('Bienvenue dans Change Risk Analyzer', 'success');
}

function setupEventListeners() {
    // Bouton analyser
    document.getElementById('analyze-btn').addEventListener('click', analyzeChange);
    
    // Bouton test
    document.getElementById('test-btn').addEventListener('click', testConnection);
    
    // Input changement (validation en temps réel)
    document.getElementById('change-ref').addEventListener('input', validateChangeReference);
    
    // Touche Entrée pour lancer l'analyse
    document.getElementById('change-ref').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            analyzeChange();
        }
    });
}

// ===================================================================
// FONCTIONS UTILITAIRES
// ===================================================================

function showLoading(message = 'Chargement...') {
    const overlay = document.getElementById('loading-overlay');
    const spinner = overlay.querySelector('.loading-spinner p');
    spinner.textContent = message;
    overlay.style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}

function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span>${message}</span>
            <button onclick="this.parentElement.parentElement.remove()" style="background: none; border: none; font-size: 1.2rem; cursor: pointer;">&times;</button>
        </div>
    `;
    
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
    const value = input.value.toUpperCase();
    const pattern = /^CHG\d{7}$/;
    
    if (value && !pattern.test(value)) {
        input.style.borderColor = '#dc3545';
        return false;
    } else {
        input.style.borderColor = '#28a745';
        return true;
    }
}

// ===================================================================
// API CALLS
// ===================================================================

async function apiCall(endpoint, params = {}) {
    try {
        const url = new URL(getWebAppBackendUrl(endpoint));
        Object.keys(params).forEach(key => url.searchParams.append(key, params[key]));
        
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.status === 'error') {
            throw new Error(data.message || 'Erreur inconnue');
        }
        
        return data;
        
    } catch (error) {
        console.error(`Erreur API ${endpoint}:`, error);
        throw error;
    }
}

// ===================================================================
// FONCTIONS PRINCIPALES
// ===================================================================

async function checkSystemStatus() {
    try {
        // Vérifier le statut du modèle
        const modelStatus = await apiCall('get_model_status');
        updateModelStatus(modelStatus);
        
        // Vérifier le statut des connexions
        const connectionStatus = await apiCall('get_connection_status');
        updateConnectionStatus(connectionStatus);
        
    } catch (error) {
        console.error('Erreur vérification statut:', error);
        showToast('Erreur lors de la vérification du statut système', 'error');
    }
}

function updateModelStatus(status) {
    const statusCard = document.getElementById('model-status');
    const statusText = document.getElementById('model-status-text');
    const details = document.getElementById('model-details');
    const performance = document.getElementById('model-performance');
    
    if (status.data.status === 'Modèle chargé') {
        statusCard.className = 'status-card status-success';
        statusText.textContent = '✅ Opérationnel';
        details.textContent = `Algorithme: ${status.data.algorithm}`;
        
        // Performance
        const perf = status.data.training_info?.performance || {};
        performance.innerHTML = `
            <p><strong>Recall:</strong> ${perf.recall || 'N/A'}</p>
            <p><strong>Precision:</strong> ${perf.precision || 'N/A'}</p>
            <p><strong>Features:</strong> ${status.data.features?.count || 'N/A'}</p>
        `;
    } else {
        statusCard.className = 'status-card status-error';
        statusText.textContent = '❌ Non disponible';
        details.textContent = 'Modèle non chargé';
        performance.innerHTML = '<p>❌ Non disponible</p>';
    }
}

function updateConnectionStatus(status) {
    const statusCard = document.getElementById('connection-status');
    const statusText = document.getElementById('connection-status-text');
    const details = document.getElementById('connection-details');
    
    if (status.data.status === 'Connecté') {
        statusCard.className = 'status-card status-success';
        statusText.textContent = '✅ Connecté';
        details.textContent = 'Tables: change_request & incident_filtree';
    } else {
        statusCard.className = 'status-card status-error';
        statusText.textContent = '❌ Erreur';
        details.textContent = status.data.error || 'Erreur de connexion';
    }
}

async function testConnection() {
    try {
        showLoading('Test des connexions...');
        
        const result = await apiCall('test_connection');
        
        hideLoading();
        
        if (result.data.success) {
            showToast('✅ Test de connexion réussi', 'success');
        } else {
            showToast('❌ Test de connexion échoué', 'error');
        }
        
    } catch (error) {
        hideLoading();
        showToast(`❌ Erreur de test: ${error.message}`, 'error');
    }
}

async function analyzeChange() {
    const changeRef = document.getElementById('change-ref').value.trim().toUpperCase();
    
    // Validation
    if (!changeRef) {
        showToast('⚠️ Veuillez saisir une référence de changement', 'warning');
        return;
    }
    
    if (!validateChangeReference()) {
        showToast('❌ Format invalide. Utilisez CHG + 7 chiffres', 'error');
        return;
    }
    
    try {
        showLoading(`Analyse de ${changeRef} en cours...`);
        
        // Appel API pour analyser le changement
        const result = await apiCall('analyze_change', { change_ref: changeRef });
        
        hideLoading();
        
        if (result.data.change_found) {
            currentChangeData = result.data;
            displayAnalysisResults(result.data);
            showToast(`✅ Analyse de ${changeRef} terminée`, 'success');
        } else {
            showToast(`❌ Changement ${changeRef} non trouvé`, 'error');
            clearResults();
        }
        
    } catch (error) {
        hideLoading();
        showToast(`❌ Erreur d'analyse: ${error.message}`, 'error');
        clearResults();
    }
}

// ===================================================================
// AFFICHAGE DES RÉSULTATS
// ===================================================================

function displayAnalysisResults(data) {
    // Résultats principaux
    displayMainResults(data);
    
    // Résultats détaillés
    displayDetailedResults(data);
}

function displayMainResults(data) {
    const resultsDiv = document.getElementById('analysis-results');
    const analysis = data.detailed_analysis;
    
    resultsDiv.innerHTML = `
        <div class="text-center mb-3">
            <h3><i class="fas fa-chart-pie"></i> Analyse de ${data.change_ref}</h3>
        </div>
        
        <div class="risk-card">
            <div class="risk-score">${analysis.risk_color} ${analysis.risk_score}%</div>
            <div class="risk-level">Risque d'échec</div>
            <div class="risk-level">Niveau: ${analysis.risk_level}</div>
            <div class="risk-interpretation">${analysis.interpretation}</div>
        </div>
    `;
}

function displayDetailedResults(data) {
    const detailedDiv = document.getElementById('detailed-results');
    const analysis = data.detailed_analysis;
    const change = data.change_data;
    
    detailedDiv.innerHTML = `
        <div class="details-grid">
            <!-- Colonne gauche -->
            <div class="details-section">
                <h4><i class="fas fa-exclamation-triangle"></i> Facteurs de risque</h4>
                <div class="warning-box">
                    ${analysis.risk_factors.length > 0 ? 
                        '<ul>' + analysis.risk_factors.map(factor => `<li>${factor}</li>`).join('') + '</ul>' :
                        '<p>Aucun facteur spécifique détecté</p>'
                    }
                </div>
                
                <h4><i class="fas fa-lightbulb"></i> Recommandations</h4>
                <div class="success-box">
                    <ul>
                        ${analysis.recommendations.map(rec => `<li>✅ ${rec}</li>`).join('')}
                    </ul>
                </div>
            </div>
            
            <!-- Colonne droite -->
            <div class="details-section">
                <h4><i class="fas fa-cogs"></i> Caractéristiques techniques</h4>
                <div class="features-list">
                    <p><strong>Type SILCA:</strong> <span>${change.dv_u_type_change_silca || 'N/A'}</span></p>
                    <p><strong>Type de changement:</strong> <span>${change.dv_type || 'N/A'}</span></p>
                    <p><strong>Nombre de CAB:</strong> <span>${change.u_cab_count || 'N/A'}</span></p>
                    <p><strong>Périmètre BCR:</strong> <span>${change.u_bcr ? '✅' : '❌'}</span></p>
                    <p><strong>Périmètre BPC:</strong> <span>${change.u_bpc ? '✅' : '❌'}</span></p>
                </div>
                
                <h4><i class="fas fa-clipboard-list"></i> Métadonnées</h4>
                <div class="info-box">
                    <p><strong>Équipe:</strong> ${change.dv_assignment_group || 'N/A'}</p>
                    <p><strong>CI/Solution:</strong> ${change.dv_cmdb_ci || 'N/A'}</p>
                    <p><strong>Catégorie:</strong> ${change.dv_category || 'N/A'}</p>
                    <p><strong>État:</strong> ${change.dv_state || 'N/A'}</p>
                </div>
            </div>
        </div>
        
        <!-- Onglets contextuels -->
        <div class="tabs-container">
            <hr>
            <h3><i class="fas fa-chart-line"></i> Informations contextuelles</h3>
            <p style="color: #666; font-style: italic;"><i class="fas fa-database"></i> Données extraites des tables ServiceNow réelles</p>
            
            <div class="tabs-buttons">
                <button class="btn-tab active" onclick="switchTab('team')">
                    <i class="fas fa-users"></i> Statistiques équipe
                </button>
                <button class="btn-tab" onclick="switchTab('incidents')">
                    <i class="fas fa-tools"></i> Incidents liés
                </button>
                <button class="btn-tab" onclick="switchTab('similar')">
                    <i class="fas fa-clipboard-list"></i> Changements similaires
                </button>
            </div>
            
            <div id="tab-content" class="tab-content">
                <p style="text-align: center; color: #666; padding: 2rem;">
                    <i class="fas fa-mouse-pointer"></i> Cliquez sur un onglet pour voir les informations contextuelles
                </p>
            </div>
        </div>
    `;
}

async function switchTab(tabName) {
    if (!currentChangeData) return;
    
    activeTab = tabName;
    
    // Mettre à jour les boutons
    document.querySelectorAll('.btn-tab').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    const contentDiv = document.getElementById('tab-content');
    
    try {
        showLoading('Chargement des données contextuelles...');
        
        let result;
        
        switch(tabName) {
            case 'team':
                result = await apiCall('get_team_stats', { 
                    assignment_group: currentChangeData.change_data.dv_assignment_group 
                });
                displayTeamStats(result.data);
                break;
                
            case 'incidents':
                result = await apiCall('get_incidents', { 
                    cmdb_ci: currentChangeData.change_data.dv_cmdb_ci 
                });
                displayIncidents(result.data);
                break;
                
            case 'similar':
                result = await apiCall('get_similar_changes', { 
                    change_ref: currentChangeData.change_ref 
                });
                displaySimilarChanges(result.data);
                break;
        }
        
        hideLoading();
        
    } catch (error) {
        hideLoading();
        contentDiv.innerHTML = `
            <div class="error-box">
                <p><i class="fas fa-exclamation-circle"></i> Erreur lors du chargement: ${error.message}</p>
            </div>
        `;
    }
}

function displayTeamStats(stats) {
    const contentDiv = document.getElementById('tab-content');
    
    if (stats && !stats.error) {
        const lastFailureText = stats.last_failure_date ? 
            `Il y a ${Math.floor((new Date() - new Date(stats.last_failure_date)) / (1000 * 60 * 60 * 24))} jours` :
            'Aucun récent';
        
        contentDiv.innerHTML = `
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">${stats.total_changes}</div>
                    <div class="metric-title">Total changements</div>
                    <div class="metric-subtitle">6 derniers mois</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${stats.success_rate}%</div>
                    <div class="metric-title">Taux de succès</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${stats.failures}</div>
                    <div class="metric-title">Échecs</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${lastFailureText}</div>
                    <div class="metric-title">Dernier échec</div>
                </div>
            </div>
        `;
    } else {
        contentDiv.innerHTML = `
            <div class="warning-box">
                <p><i class="fas fa-exclamation-triangle"></i> Statistiques équipe non disponibles</p>
            </div>
        `;
    }
}

function displayIncidents(incidents) {
    const contentDiv = document.getElementById('tab-content');
    
    if (incidents) {
        const resolutionText = incidents.avg_resolution_hours > 0 ? 
            `${incidents.avg_resolution_hours}h` : 'N/A';
        
        contentDiv.innerHTML = `
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">${incidents.total_incidents}</div>
                    <div class="metric-title">Total incidents</div>
                    <div class="metric-subtitle">3 derniers mois</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${incidents.critical_incidents}</div>
                    <div class="metric-title">Incidents critiques</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${resolutionText}</div>
                    <div class="metric-title">Résolution moyenne</div>
                </div>
            </div>
            <p style="text-align: center; color: #666; margin-top: 1rem;">
                <i class="fas fa-database"></i> Données extraites de la table incident_filtree
            </p>
        `;
    } else {
        contentDiv.innerHTML = `
            <div class="warning-box">
                <p><i class="fas fa-exclamation-triangle"></i> Aucun incident trouvé pour cette solution/CI</p>
            </div>
        `;
    }
}

function displaySimilarChanges(similar) {
    const contentDiv = document.getElementById('tab-content');
    
    if (similar && similar.length > 0) {
        const changesHtml = similar.map(change => {
            const closeCode = change.dv_close_code;
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
                ` • Durée: ${change.duration_hours}h` : '';
            
            return `
                <div class="${className}">
                    <p>
                        <strong>${icon} ${change.number} - ${closeCode}</strong><br>
                        <small>${change.short_description.substring(0, 100)}...</small><br>
                        <small style="color: #666;">
                            Similarité: ${change.similarity_score}%${durationText}
                        </small>
                    </p>
                </div>
            `;
        }).join('');
        
        contentDiv.innerHTML = `
            <p><i class="fas fa-clipboard-list"></i> <strong>Historique réel des changements similaires:</strong></p>
            ${changesHtml}
        `;
    } else {
        contentDiv.innerHTML = `
            <div class="info-box">
                <p><i class="fas fa-info-circle"></i> Aucun changement similaire trouvé dans l'historique</p>
            </div>
        `;
    }
}

function clearResults() {
    document.getElementById('analysis-results').innerHTML = '';
    document.getElementById('detailed-results').innerHTML = '';
    currentChangeData = null;
}

// ===================================================================
// FONCTIONS UTILITAIRES SUPPLÉMENTAIRES
// ===================================================================

// Auto-complétion majuscules pour le champ changement
document.addEventListener('DOMContentLoaded', function() {
    const changeInput = document.getElementById('change-ref');
    if (changeInput) {
        changeInput.addEventListener('input', function() {
            this.value = this.value.toUpperCase();
        });
    }
});
```

---

## 🐍 **ONGLET PYTHON :**

```python
"""
WEBAPP BACKEND - ANALYSE DES RISQUES DE CHANGEMENT SERVICENOW
Flask backend avec endpoints API
"""

import dataiku
import pandas as pd
import json
import traceback
from flask import request, jsonify
from datetime import datetime

# Import de nos modules
from change_risk_predictor import ChangeRiskPredictor
from servicenow_connector import ServiceNowConnector

# ===================================================================
# INITIALISATION DES COMPOSANTS
# ===================================================================

# Initialisation globale
try:
    predictor = ChangeRiskPredictor()
    connector = ServiceNowConnector()
    init_success = True
    init_error = ""
except Exception as e:
    predictor = None
    connector = None
    init_success = False
    init_error = str(e)

# ===================================================================
# UTILITAIRES
# ===================================================================

def create_response(data=None, status="ok", message=""):
    """Créer une réponse API standardisée"""
    response = {
        "status": status,
        "message": message,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }
    return json.dumps(response)

def handle_error(error, endpoint_name):
    """Gestion standardisée des erreurs"""
    error_msg = str(error)
    print(f"❌ Erreur dans {endpoint_name}: {error_msg}")
    print(traceback.format_exc())
    
    return create_response(
        status="error",
        message=f"Erreur dans {endpoint_name}: {error_msg}"
    )

# ===================================================================
# ENDPOINTS API
# ===================================================================

@app.route('/get_model_status')
def get_model_status():
    """Récupérer le statut du modèle ML"""
    
    try:
        if not init_success:
            return create_response(
                data={"status": "Erreur d'initialisation", "error": init_error},
                status="error"
            )
        
        model_info = predictor.get_model_info()
        
        return create_response(data=model_info)
        
    except Exception as e:
        return handle_error(e, "get_model_status")

@app.route('/get_connection_status')
def get_connection_status():
    """Vérifier le statut des connexions ServiceNow"""
    
    try:
        if not init_success:
            return create_response(
                data={"status": "Erreur", "error": init_error},
                status="error"
            )
        
        connection_status = connector.get_connection_status()
        
        return create_response(data=connection_status)
        
    except Exception as e:
        return handle_error(e, "get_connection_status")

@app.route('/test_connection')
def test_connection():
    """Tester les connexions système"""
    
    try:
        if not init_success:
            return create_response(
                data={"success": False, "error": init_error},
                status="error"
            )
        
        # Test du modèle
        model_info = predictor.get_model_info()
        model_ok = model_info.get("status") == "Modèle chargé"
        
        # Test des connexions
        connection_status = connector.get_connection_status()
        connection_ok = connection_status.get("status") == "Connecté"
        
        success = model_ok and connection_ok
        
        return create_response(data={
            "success": success,
            "model_status": model_ok,
            "connection_status": connection_ok,
            "details": {
                "model": model_info,
                "connection": connection_status
            }
        })
        
    except Exception as e:
        return handle_error(e, "test_connection")

@app.route('/analyze_change')
def analyze_change():
    """Analyser un changement spécifique"""
    
    try:
        if not init_success:
            return create_response(
                status="error",
                message="Système non initialisé"
            )
        
        # Récupérer la référence du changement
        change_ref = request.args.get('change_ref')
        
        if not change_ref:
            return create_response(
                status="error", 
                message="Référence de changement manquante"
            )
        
        # Validation du format
        if not connector.validate_change_reference(change_ref):
            return create_response(
                status="error",
                message="Format de référence invalide"
            )
        
        # Récupération des données
        change_data = connector.get_change_data(change_ref)
        
        if not change_data:
            return create_response(data={
                "change_found": False,
                "change_ref": change_ref
            })
        
        # Analyse ML
        detailed_analysis = predictor.get_detailed_analysis(change_data)
        
        return create_response(data={
            "change_found": True,
            "change_ref": change_ref,
            "change_data": change_data,
            "detailed_analysis": detailed_analysis
        })
        
    except Exception as e:
        return handle_error(e, "analyze_change")

@app.route('/get_team_stats')
def get_team_stats():
    """Récupérer les statistiques d'une équipe"""
    
    try:
        if not init_success:
            return create_response(
                status="error",
                message="Système non initialisé"
            )
        
        assignment_group = request.args.get('assignment_group')
        
        if not assignment_group:
            return create_response(
                status="error",
                message="Nom d'équipe manquant"
            )
        
        team_stats = connector.get_team_statistics(assignment_group)
        
        return create_response(data=team_stats)
        
    except Exception as e:
        return handle_error(e, "get_team_stats")

@app.route('/get_incidents')
def get_incidents():
    """Récupérer les incidents liés à un CI"""
    
    try:
        if not init_success:
            return create_response(
                status="error",
                message="Système non initialisé"
            )
        
        cmdb_ci = request.args.get('cmdb_ci')
        
        if not cmdb_ci:
            return create_response(
                status="error",
                message="CI manquant"
            )
        
        incidents_data = connector.get_solution_incidents(cmdb_ci)
        
        return create_response(data=incidents_data)
        
    except Exception as e:
        return handle_error(e, "get_incidents")

@app.route('/get_similar_changes')
def get_similar_changes():
    """Récupérer les changements similaires"""
    
    try:
        if not init_success:
            return create_response(
                status="error",
                message="Système non initialisé"
            )
        
        change_ref = request.args.get('change_ref')
        
        if not change_ref:
            return create_response(
                status="error",
                message="Référence de changement manquante"
            )
        
        # Récupérer les données du changement d'abord
        change_data = connector.get_change_data(change_ref)
        
        if not change_data:
            return create_response(
                status="error",
                message="Changement non trouvé"
            )
        
        # Chercher les changements similaires
        similar_changes = connector.find_similar_changes(change_data)
        
        return create_response(data=similar_changes)
        
    except Exception as e:
        return handle_error(e, "get_similar_changes")

# ===================================================================
# ENDPOINT DE DIAGNOSTIC
# ===================================================================

@app.route('/diagnostic')
def diagnostic():
    """Endpoint de diagnostic complet du système"""
    
    try:
        diagnostic_info = {
            "initialization": {
                "success": init_success,
                "error": init_error if not init_success else None
            },
            "components": {},
            "datasets": {},
            "timestamp": datetime.now().isoformat()
        }
        
        if init_success:
            # Test du prédicteur
            try:
                model_info = predictor.get_model_info()
                diagnostic_info["components"]["predictor"] = {
                    "status": "OK",
                    "details": model_info
                }
            except Exception as e:
                diagnostic_info["components"]["predictor"] = {
                    "status": "ERROR",
                    "error": str(e)
                }
            
            # Test du connecteur
            try:
                connection_status = connector.get_connection_status()
                diagnostic_info["components"]["connector"] = {
                    "status": "OK",
                    "details": connection_status
                }
            except Exception as e:
                diagnostic_info["components"]["connector"] = {
                    "status": "ERROR",
                    "error": str(e)
                }
            
            # Test des datasets
            datasets_to_test = [
                "change_request_without_empty_columns",
                "incident_filtree"
            ]
            
            for dataset_name in datasets_to_test:
                try:
                    dataset = dataiku.Dataset(dataset_name)
                    df = dataset.get_dataframe(limit=1)
                    diagnostic_info["datasets"][dataset_name] = {
                        "status": "OK",
                        "columns": len(df.columns),
                        "sample_available": True
                    }
                except Exception as e:
                    diagnostic_info["datasets"][dataset_name] = {
                        "status": "ERROR",
                        "error": str(e)
                    }
        
        return create_response(data=diagnostic_info)
        
    except Exception as e:
        return handle_error(e, "diagnostic")

# ===================================================================
# ENDPOINT DE SANTÉ
# ===================================================================

@app.route('/health')
def health():
    """Endpoint de santé simple"""
    
    return create_response(data={
        "status": "alive",
        "initialization": init_success,
        "timestamp": datetime.now().isoformat()
    })

# ===================================================================
# GESTION DES ERREURS GLOBALES
# ===================================================================

@app.errorhandler(404)
def not_found(error):
    return create_response(
        status="error",
        message="Endpoint non trouvé"
    ), 404

@app.errorhandler(500)
def internal_error(error):
    return create_response(
        status="error",
        message="Erreur interne du serveur"
    ), 500

# ===================================================================
# POINT D'ENTRÉE
# ===================================================================

if __name__ == '__main__':
    print("🚀 Démarrage du backend Change Risk Analyzer")
    print(f"Initialisation: {'✅ OK' if init_success else '❌ ERROR'}")
    if not init_success:
        print(f"Erreur: {init_error}")
```

## 🎯 **MAINTENANT VOUS AVEZ UNE WEBAPP COMPLÈTE !**

### **✅ AVANTAGES DE CETTE APPROCHE :**
- **Plus stable** que Dash
- **Interface moderne** et responsive
- **API REST** claire et documentée
- **Gestion d'erreurs** robuste
- **100% données réelles** ServiceNow

### **🚀 PROCHAINES ÉTAPES :**
1. **Sauvegarder** chaque onglet avec son code
2. **Démarrer** la webapp
3. **Tester** les fonctionnalités

**Cette webapp devrait marcher du premier coup !** 🎯

J'ai créé une webapp complète avec HTML/CSS/JavaScript/Python qui reproduit toutes les fonctionnalités de la version Dash mais avec une architecture plus stable et traditionnelle.

**Structure:**

1. **HTML:** Interface utilisateur complète avec structure sémantique
2. **CSS:** Design moderne avec gradients, animations, responsive design
3. **JavaScript:** Logique frontend avec appels AJAX, gestion d'état, validation
4. **Python:** Backend Flask avec endpoints REST clairs

**Fonctionnalités:**
- Vérification du statut système
- Analyse de changements en temps réel
- Affichage des résultats avec score de risque
- Onglets contextuels (statistiques équipe, incidents, changements similaires)
- Gestion d'erreurs robuste
- Interface responsive
- Notifications toast
- Loading states

**Avantages par rapport à Dash:**
- Plus stable (pas de problèmes de layout)
- Plus de contrôle sur l'interface
- Meilleure performance
- Plus facile à déboguer
- Architecture claire frontend/backend

L'utilisateur devrait pouvoir copier-coller chaque code dans l'onglet correspondant et avoir une webapp fonctionnelle immédiatement.L'utilisateur veut abandonner Dash qui ne fonctionne pas et passer à une webapp Standard avec HTML/CSS/JS/Python. C'est effectivement une bonne idée car cela semble plus simple et plus fiable.

Je dois adapter le code pour créer :

1. **HTML** : Structure de la page
2. **CSS** : Styles pour un design moderne
3. **JavaScript** : Logique frontend et appels AJAX
4. **Python** : Backend Flask avec endpoints API

L'architecture sera :
- Frontend : HTML/CSS/JS pour l'interface utilisateur
- Backend : Flask avec des endpoints pour :
  - Analyser un changement
  - Tester les connexions
  - Récupérer les statistiques d'équipe
  - Récupérer les incidents
  - Récupérer les changements similaires

Je vais reproduire la même fonctionnalité que la webapp Dash mais avec cette stack plus traditionnelle.

## 🎯 **EXCELLENTE IDÉE ! Passons à la webapp Standard**

**Dash était effectivement problématique. Voici le code pour chaque onglet :**

---

## 📁 **ONGLET HTML :**

```html
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Change Risk Analyzer</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body>
    <!-- Header principal -->
    <div class="main-header">
        <h1><i class="fas fa-search"></i> Change Risk Analyzer</h1>
        <p>Analyseur de risques pour changements ServiceNow • 100% Données Réelles</p>
    </div>

    <!-- Container principal -->
    <div class="container">
        
        <!-- Row pour le statut et la saisie -->
        <div class="main-row">
            
            <!-- Colonne gauche - Statuts -->
            <div class="sidebar">
                <h4><i class="fas fa-robot"></i> Statut du Système</h4>
                
                <!-- Statut modèle -->
                <div id="model-status" class="status-card">
                    <h6>Modèle ML</h6>
                    <p id="model-status-text">Vérification...</p>
                    <small id="model-details"></small>
                </div>
                
                <!-- Statut connexions -->
                <div id="connection-status" class="status-card">
                    <h6>Connexions ServiceNow</h6>
                    <p id="connection-status-text">Vérification...</p>
                    <small id="connection-details"></small>
                </div>
                
                <!-- Performance modèle -->
                <h5><i class="fas fa-chart-line"></i> Performance Modèle</h5>
                <div id="model-performance" class="info-box">
                    <p>Chargement...</p>
                </div>
            </div>
            
            <!-- Colonne droite - Interface principale -->
            <div class="main-content">
                <h4><i class="fas fa-edit"></i> Analyse de Changement</h4>
                
                <!-- Zone de saisie -->
                <div class="input-section">
                    <div class="input-group">
                        <label for="change-ref">Référence du changement :</label>
                        <input type="text" id="change-ref" placeholder="CHG0012345" />
                        <small>Format: CHG + 7 chiffres</small>
                    </div>
                    
                    <div class="button-group">
                        <button id="analyze-btn" class="btn-primary">
                            <i class="fas fa-search"></i> Analyser
                        </button>
                        <button id="test-btn" class="btn-secondary">
                            <i class="fas fa-info-circle"></i> Test
                        </button>
                    </div>
                </div>
                
                <!-- Zone de résultats -->
                <hr>
                <div id="analysis-results"></div>
            </div>
        </div>
        
        <!-- Zone pour les résultats détaillés -->
        <div id="detailed-results"></div>
    </div>

    <!-- Footer -->
    <footer>
        <hr>
        <div class="footer-content">
            <p><i class="fas fa-robot"></i> Change Risk Analyzer • 100% Données Réelles ServiceNow</p>
            <p><em>Tables connectées: change_request_without_empty_columns & incident_filtree</em></p>
            <p><strong>ZERO SIMULATION • ZERO DONNÉES FACTICES</strong></p>
        </div>
    </footer>

    <!-- Loading overlay -->
    <div id="loading-overlay" class="loading-overlay" style="display: none;">
        <div class="loading-spinner">
            <i class="fas fa-spinner fa-spin fa-3x"></i>
            <p>Analyse en cours...</p>
        </div>
    </div>

    <!-- Toast notifications -->
    <div id="toast-container"></div>
</body>
</html>
```

---

## 🎨 **ONGLET CSS :**

```css
/* ===================================================================
   CHANGE RISK ANALYZER - STYLES MODERNES
   ================================================================== */

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: #f8f9fa;
    color: #333;
    line-height: 1.6;
}

/* Header principal */
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.main-header h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 300;
}

.main-header p {
    margin: 0.5rem 0 0 0;
    font-size: 1.1rem;
    opacity: 0.9;
}

/* Container et layout */
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 1rem;
}

.main-row {
    display: flex;
    gap: 2rem;
    align-items: flex-start;
}

.sidebar {
    width: 30%;
    min-width: 300px;
}

.main-content {
    flex: 1;
    min-width: 0;
}

/* Status cards */
.status-card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    border: 1px solid #e9ecef;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}

.status-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

.status-card h6 {
    margin: 0 0 0.5rem 0;
    font-weight: 600;
    color: #495057;
}

.status-card p {
    margin: 0;
    font-weight: 500;
}

.status-card small {
    color: #6c757d;
    font-size: 0.875rem;
}

/* Status couleurs */
.status-success {
    border-left: 4px solid #28a745;
    background: #d4edda;
}

.status-error {
    border-left: 4px solid #dc3545;
    background: #f8d7da;
}

.status-warning {
    border-left: 4px solid #ffc107;
    background: #fff3cd;
}

/* Boxes d'information */
.info-box {
    background: #d1ecf1;
    border: 1px solid #bee5eb;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.success-box {
    background: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.warning-box {
    background: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.error-box {
    background: #f8d7da;
    border: 1px solid #f5c6cb;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

/* Zone de saisie */
.input-section {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}

.input-group {
    margin-bottom: 1rem;
}

.input-group label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: 500;
    color: #495057;
}

.input-group input {
    width: 100%;
    padding: 0.75rem;
    border: 2px solid #e9ecef;
    border-radius: 8px;
    font-size: 1rem;
    transition: border-color 0.3s ease;
}

.input-group input:focus {
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.input-group small {
    color: #6c757d;
    font-size: 0.875rem;
}

.button-group {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
}

/* Boutons */
.btn-primary {
    background: #667eea;
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}

.btn-primary:hover {
    background: #5a6fd8;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.btn-secondary {
    background: #6c757d;
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}

.btn-secondary:hover {
    background: #5a6268;
    transform: translateY(-1px);
}

.btn-tab {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    padding: 0.5rem 1rem;
    border-radius: 5px;
    margin: 0.25rem;
    cursor: pointer;
    transition: all 0.3s ease;
}

.btn-tab:hover {
    background: #e9ecef;
}

.btn-tab.active {
    background: #667eea;
    color: white;
    border-color: #667eea;
}

/* Cartes de résultats */
.risk-card {
    background: white;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border-left: 5px solid #667eea;
    margin: 1rem 0;
    text-align: center;
}

.risk-score {
    font-size: 3rem;
    font-weight: bold;
    color: #667eea;
    margin: 0;
}

.risk-level {
    font-size: 1.5rem;
    margin: 0.5rem 0;
    font-weight: 600;
}

.risk-interpretation {
    font-style: italic;
    color: #6c757d;
    margin: 0;
}

/* Métriques */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

.metric-card {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    text-align: center;
    border: 1px solid #e9ecef;
    transition: transform 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: #667eea;
    margin: 0;
}

.metric-title {
    color: #495057;
    margin: 0.5rem 0 0 0;
}

.metric-subtitle {
    color: #6c757d;
    font-size: 0.875rem;
    margin: 0;
}

/* Détails en colonnes */
.details-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    margin: 2rem 0;
}

.details-section h4 {
    color: #495057;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.features-list {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.features-list p {
    margin: 0.5rem 0;
    display: flex;
    justify-content: space-between;
}

.features-list strong {
    color: #495057;
}

/* Changements similaires */
.similar-change {
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border: 1px solid #dee2e6;
}

.similar-change.success {
    background: #d4edda;
    border-color: #c3e6cb;
}

.similar-change.error {
    background: #f8d7da;
    border-color: #f5c6cb;
}

.similar-change.warning {
    background: #fff3cd;
    border-color: #ffeaa7;
}

/* Onglets */
.tabs-container {
    margin: 2rem 0;
}

.tabs-buttons {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}

.tab-content {
    padding: 1rem;
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    min-height: 200px;
}

/* Loading overlay */
.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.5);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 9999;
}

.loading-spinner {
    background: white;
    padding: 2rem;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

.loading-spinner i {
    color: #667eea;
    margin-bottom: 1rem;
}

/* Toast notifications */
#toast-container {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 10000;
}

.toast {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    margin-bottom: 1rem;
    min-width: 300px;
    border-left: 4px solid #28a745;
    animation: slideIn 0.3s ease;
}

.toast.error {
    border-left-color: #dc3545;
}

.toast.warning {
    border-left-color: #ffc107;
}

@keyframes slideIn {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

/* Footer */
footer {
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 1px solid #dee2e6;
}

.footer-content {
    text-align: center;
    color: #6c757d;
    padding: 2rem;
}

.footer-content p {
    margin: 0.25rem 0;
}

/* Responsive */
@media (max-width: 768px) {
    .main-row {
        flex-direction: column;
    }
    
    .sidebar {
        width: 100%;
    }
    
    .details-grid {
        grid-template-columns: 1fr;
    }
    
    .metrics-grid {
        grid-template-columns: 1fr;
    }
    
    .button-group {
        justify-content: center;
    }
    
    .main-header h1 {
        font-size: 2rem;
    }
}

/* Utilities */
.text-center { text-align: center; }
.text-left { text-align: left; }
.text-right { text-align: right; }
.mt-1 { margin-top: 0.5rem; }
.mt-2 { margin-top: 1rem; }
.mt-3 { margin-top: 1.5rem; }
.mb-1 { margin-bottom: 0.5rem; }
.mb-2 { margin-bottom: 1rem; }
.mb-3 { margin-bottom: 1.5rem; }
.hidden { display: none; }
.visible { display: block; }
```

---

## ⚡ **ONGLET JAVASCRIPT :**

```javascript
// ===================================================================
// CHANGE RISK ANALYZER - LOGIQUE FRONTEND
// ===================================================================

// Variables globales
let currentChangeData = null;
let activeTab = 'team';

// Initialisation au chargement de la page
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    checkSystemStatus();
});

// ===================================================================
// INITIALISATION
// ===================================================================

function initializeApp() {
    console.log('🚀 Initialisation de Change Risk Analyzer');
    
    // Afficher un message de bienvenue
    showToast('Bienvenue dans Change Risk Analyzer', 'success');
}

function setupEventListeners() {
    // Bouton analyser
    document.getElementById('analyze-btn').addEventListener('click', analyzeChange);
    
    // Bouton test
    document.getElementById('test-btn').addEventListener('click', testConnection);
    
    // Input changement (validation en temps réel)
    document.getElementById('change-ref').addEventListener('input', validateChangeReference);
    
    // Touche Entrée pour lancer l'analyse
    document.getElementById('change-ref').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            analyzeChange();
        }
    });
}

// ===================================================================
// FONCTIONS UTILITAIRES
// ===================================================================

function showLoading(message = 'Chargement...') {
    const overlay = document.getElementById('loading-overlay');
    const spinner = overlay.querySelector('.loading-spinner p');
    spinner.textContent = message;
    overlay.style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}

function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span>${message}</span>
            <button onclick="this.parentElement.parentElement.remove()" style="background: none; border: none; font-size: 1.2rem; cursor: pointer;">&times;</button>
        </div>
    `;
    
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
    const value = input.value.toUpperCase();
    const pattern = /^CHG\d{7}$/;
    
    if (value && !pattern.test(value)) {
        input.style.borderColor = '#dc3545';
        return false;
    } else {
        input.style.borderColor = '#28a745';
        return true;
    }
}

// ===================================================================
// API CALLS
// ===================================================================

async function apiCall(endpoint, params = {}) {
    try {
        const url = new URL(getWebAppBackendUrl(endpoint));
        Object.keys(params).forEach(key => url.searchParams.append(key, params[key]));
        
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.status === 'error') {
            throw new Error(data.message || 'Erreur inconnue');
        }
        
        return data;
        
    } catch (error) {
        console.error(`Erreur API ${endpoint}:`, error);
        throw error;
    }
}

// ===================================================================
// FONCTIONS PRINCIPALES
// ===================================================================

async function checkSystemStatus() {
    try {
        // Vérifier le statut du modèle
        const modelStatus = await apiCall('get_model_status');
        updateModelStatus(modelStatus);
        
        // Vérifier le statut des connexions
        const connectionStatus = await apiCall('get_connection_status');
        updateConnectionStatus(connectionStatus);
        
    } catch (error) {
        console.error('Erreur vérification statut:', error);
        showToast('Erreur lors de la vérification du statut système', 'error');
    }
}

function updateModelStatus(status) {
    const statusCard = document.getElementById('model-status');
    const statusText = document.getElementById('model-status-text');
    const details = document.getElementById('model-details');
    const performance = document.getElementById('model-performance');
    
    if (status.data.status === 'Modèle chargé') {
        statusCard.className = 'status-card status-success';
        statusText.textContent = '✅ Opérationnel';
        details.textContent = `Algorithme: ${status.data.algorithm}`;
        
        // Performance
        const perf = status.data.training_info?.performance || {};
        performance.innerHTML = `
            <p><strong>Recall:</strong> ${perf.recall || 'N/A'}</p>
            <p><strong>Precision:</strong> ${perf.precision || 'N/A'}</p>
            <p><strong>Features:</strong> ${status.data.features?.count || 'N/A'}</p>
        `;
    } else {
        statusCard.className = 'status-card status-error';
        statusText.textContent = '❌ Non disponible';
        details.textContent = 'Modèle non chargé';
        performance.innerHTML = '<p>❌ Non disponible</p>';
    }
}

function updateConnectionStatus(status) {
    const statusCard = document.getElementById('connection-status');
    const statusText = document.getElementById('connection-status-text');
    const details = document.getElementById('connection-details');
    
    if (status.data.status === 'Connecté') {
        statusCard.className = 'status-card status-success';
        statusText.textContent = '✅ Connecté';
        details.textContent = 'Tables: change_request & incident_filtree';
    } else {
        statusCard.className = 'status-card status-error';
        statusText.textContent = '❌ Erreur';
        details.textContent = status.data.error || 'Erreur de connexion';
    }
}

async function testConnection() {
    try {
        showLoading('Test des connexions...');
        
        const result = await apiCall('test_connection');
        
        hideLoading();
        
        if (result.data.success) {
            showToast('✅ Test de connexion réussi', 'success');
        } else {
            showToast('❌ Test de connexion échoué', 'error');
        }
        
    } catch (error) {
        hideLoading();
        showToast(`❌ Erreur de test: ${error.message}`, 'error');
    }
}

async function analyzeChange() {
    const changeRef = document.getElementById('change-ref').value.trim().toUpperCase();
    
    // Validation
    if (!changeRef) {
        showToast('⚠️ Veuillez saisir une référence de changement', 'warning');
        return;
    }
    
    if (!validateChangeReference()) {
        showToast('❌ Format invalide. Utilisez CHG + 7 chiffres', 'error');
        return;
    }
    
    try {
        showLoading(`Analyse de ${changeRef} en cours...`);
        
        // Appel API pour analyser le changement
        const result = await apiCall('analyze_change', { change_ref: changeRef });
        
        hideLoading();
        
        if (result.data.change_found) {
            currentChangeData = result.data;
            displayAnalysisResults(result.data);
            showToast(`✅ Analyse de ${changeRef} terminée`, 'success');
        } else {
            showToast(`❌ Changement ${changeRef} non trouvé`, 'error');
            clearResults();
        }
        
    } catch (error) {
        hideLoading();
        showToast(`❌ Erreur d'analyse: ${error.message}`, 'error');
        clearResults();
    }
}

// ===================================================================
// AFFICHAGE DES RÉSULTATS
// ===================================================================

function displayAnalysisResults(data) {
    // Résultats principaux
    displayMainResults(data);
    
    // Résultats détaillés
    displayDetailedResults(data);
}

function displayMainResults(data) {
    const resultsDiv = document.getElementById('analysis-results');
    const analysis = data.detailed_analysis;
    
    resultsDiv.innerHTML = `
        <div class="text-center mb-3">
            <h3><i class="fas fa-chart-pie"></i> Analyse de ${data.change_ref}</h3>
        </div>
        
        <div class="risk-card">
            <div class="risk-score">${analysis.risk_color} ${analysis.risk_score}%</div>
            <div class="risk-level">Risque d'échec</div>
            <div class="risk-level">Niveau: ${analysis.risk_level}</div>
            <div class="risk-interpretation">${analysis.interpretation}</div>
        </div>
    `;
}

function displayDetailedResults(data) {
    const detailedDiv = document.getElementById('detailed-results');
    const analysis = data.detailed_analysis;
    const change = data.change_data;
    
    detailedDiv.innerHTML = `
        <div class="details-grid">
            <!-- Colonne gauche -->
            <div class="details-section">
                <h4><i class="fas fa-exclamation-triangle"></i> Facteurs de risque</h4>
                <div class="warning-box">
                    ${analysis.risk_factors.length > 0 ? 
                        '<ul>' + analysis.risk_factors.map(factor => `<li>${factor}</li>`).join('') + '</ul>' :
                        '<p>Aucun facteur spécifique détecté</p>'
                    }
                </div>
                
                <h4><i class="fas fa-lightbulb"></i> Recommandations</h4>
                <div class="success-box">
                    <ul>
                        ${analysis.recommendations.map(rec => `<li>✅ ${rec}</li>`).join('')}
                    </ul>
                </div>
            </div>
            
            <!-- Colonne droite -->
            <div class="details-section">
                <h4><i class="fas fa-cogs"></i> Caractéristiques techniques</h4>
                <div class="features-list">
                    <p><strong>Type SILCA:</strong> <span>${change.dv_u_type_change_silca || 'N/A'}</span></p>
                    <p><strong>Type de changement:</strong> <span>${change.dv_type || 'N/A'}</span></p>
                    <p><strong>Nombre de CAB:</strong> <span>${change.u_cab_count || 'N/A'}</span></p>
                    <p><strong>Périmètre BCR:</strong> <span>${change.u_bcr ? '✅' : '❌'}</span></p>
                    <p><strong>Périmètre BPC:</strong> <span>${change.u_bpc ? '✅' : '❌'}</span></p>
                </div>
                
                <h4><i class="fas fa-clipboard-list"></i> Métadonnées</h4>
                <div class="info-box">
                    <p><strong>Équipe:</strong> ${change.dv_assignment_group || 'N/A'}</p>
                    <p><strong>CI/Solution:</strong> ${change.dv_cmdb_ci || 'N/A'}</p>
                    <p><strong>Catégorie:</strong> ${change.dv_category || 'N/A'}</p>
                    <p><strong>État:</strong> ${change.dv_state || 'N/A'}</p>
                </div>
            </div>
        </div>
        
        <!-- Onglets contextuels -->
        <div class="tabs-container">
            <hr>
            <h3><i class="fas fa-chart-line"></i> Informations contextuelles</h3>
            <p style="color: #666; font-style: italic;"><i class="fas fa-database"></i> Données extraites des tables ServiceNow réelles</p>
            
            <div class="tabs-buttons">
                <button class="btn-tab active" onclick="switchTab('team')">
                    <i class="fas fa-users"></i> Statistiques équipe
                </button>
                <button class="btn-tab" onclick="switchTab('incidents')">
                    <i class="fas fa-tools"></i> Incidents liés
                </button>
                <button class="btn-tab" onclick="switchTab('similar')">
                    <i class="fas fa-clipboard-list"></i> Changements similaires
                </button>
            </div>
            
            <div id="tab-content" class="tab-content">
                <p style="text-align: center; color: #666; padding: 2rem;">
                    <i class="fas fa-mouse-pointer"></i> Cliquez sur un onglet pour voir les informations contextuelles
                </p>
            </div>
        </div>
    `;
}

async function switchTab(tabName) {
    if (!currentChangeData) return;
    
    activeTab = tabName;
    
    // Mettre à jour les boutons
    document.querySelectorAll('.btn-tab').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    const contentDiv = document.getElementById('tab-content');
    
    try {
        showLoading('Chargement des données contextuelles...');
        
        let result;
        
        switch(tabName) {
            case 'team':
                result = await apiCall('get_team_stats', { 
                    assignment_group: currentChangeData.change_data.dv_assignment_group 
                });
                displayTeamStats(result.data);
                break;
                
            case 'incidents':
                result = await apiCall('get_incidents', { 
                    cmdb_ci: currentChangeData.change_data.dv_cmdb_ci 
                });
                displayIncidents(result.data);
                break;
                
            case 'similar':
                result = await apiCall('get_similar_changes', { 
                    change_ref: currentChangeData.change_ref 
                });
                displaySimilarChanges(result.data);
                break;
        }
        
        hideLoading();
        
    } catch (error) {
        hideLoading();
        contentDiv.innerHTML = `
            <div class="error-box">
                <p><i class="fas fa-exclamation-circle"></i> Erreur lors du chargement: ${error.message}</p>
            </div>
        `;
    }
}

function displayTeamStats(stats) {
    const contentDiv = document.getElementById('tab-content');
    
    if (stats && !stats.error) {
        const lastFailureText = stats.last_failure_date ? 
            `Il y a ${Math.floor((new Date() - new Date(stats.last_failure_date)) / (1000 * 60 * 60 * 24))} jours` :
            'Aucun récent';
        
        contentDiv.innerHTML = `
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">${stats.total_changes}</div>
                    <div class="metric-title">Total changements</div>
                    <div class="metric-subtitle">6 derniers mois</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${stats.success_rate}%</div>
                    <div class="metric-title">Taux de succès</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${stats.failures}</div>
                    <div class="metric-title">Échecs</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${lastFailureText}</div>
                    <div class="metric-title">Dernier échec</div>
                </div>
            </div>
        `;
    } else {
        contentDiv.innerHTML = `
            <div class="warning-box">
                <p><i class="fas fa-exclamation-triangle"></i> Statistiques équipe non disponibles</p>
            </div>
        `;
    }
}

function displayIncidents(incidents) {
    const contentDiv = document.getElementById('tab-content');
    
    if (incidents) {
        const resolutionText = incidents.avg_resolution_hours > 0 ? 
            `${incidents.avg_resolution_hours}h` : 'N/A';
        
        contentDiv.innerHTML = `
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">${incidents.total_incidents}</div>
                    <div class="metric-title">Total incidents</div>
                    <div class="metric-subtitle">3 derniers mois</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${incidents.critical_incidents}</div>
                    <div class="metric-title">Incidents critiques</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${resolutionText}</div>
                    <div class="metric-title">Résolution moyenne</div>
                </div>
            </div>
            <p style="text-align: center; color: #666; margin-top: 1rem;">
                <i class="fas fa-database"></i> Données extraites de la table incident_filtree
            </p>
        `;
    } else {
        contentDiv.innerHTML = `
            <div class="warning-box">
                <p><i class="fas fa-exclamation-triangle"></i> Aucun incident trouvé pour cette solution/CI</p>
            </div>
        `;
    }
}

function displaySimilarChanges(similar) {
    const contentDiv = document.getElementById('tab-content');
    
    if (similar && similar.length > 0) {
        const changesHtml = similar.map(change => {
            const closeCode = change.dv_close_code;
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
                ` • Durée: ${change.duration_hours}h` : '';
            
            return `
                <div class="${className}">
                    <p>
                        <strong>${icon} ${change.number} - ${closeCode}</strong><br>
                        <small>${change.short_description.substring(0, 100)}...</small><br>
                        <small style="color: #666;">
                            Similarité: ${change.similarity_score}%${durationText}
                        </small>
                    </p>
                </div>
            `;
        }).join('');
        
        contentDiv.innerHTML = `
            <p><i class="fas fa-clipboard-list"></i> <strong>Historique réel des changements similaires:</strong></p>
            ${changesHtml}
        `;
    } else {
        contentDiv.innerHTML = `
            <div class="info-box">
                <p><i class="fas fa-info-circle"></i> Aucun changement similaire trouvé dans l'historique</p>
            </div>
        `;
    }
}

function clearResults() {
    document.getElementById('analysis-results').innerHTML = '';
    document.getElementById('detailed-results').innerHTML = '';
    currentChangeData = null;
}

// ===================================================================
// FONCTIONS UTILITAIRES SUPPLÉMENTAIRES
// ===================================================================

// Auto-complétion majuscules pour le champ changement
document.addEventListener('DOMContentLoaded', function() {
    const changeInput = document.getElementById('change-ref');
    if (changeInput) {
        changeInput.addEventListener('input', function() {
            this.value = this.value.toUpperCase();
        });
    }
});
```

---

## 🐍 **ONGLET PYTHON :**

```python
"""
WEBAPP BACKEND - ANALYSE DES RISQUES DE CHANGEMENT SERVICENOW
Flask backend avec endpoints API
"""

import dataiku
import pandas as pd
import json
import traceback
from flask import request, jsonify
from datetime import datetime

# Import de nos modules
from change_risk_predictor import ChangeRiskPredictor
from servicenow_connector import ServiceNowConnector

# ===================================================================
# INITIALISATION DES COMPOSANTS
# ===================================================================

# Initialisation globale
try:
    predictor = ChangeRiskPredictor()
    connector = ServiceNowConnector()
    init_success = True
    init_error = ""
except Exception as e:
    predictor = None
    connector = None
    init_success = False
    init_error = str(e)

# ===================================================================
# UTILITAIRES
# ===================================================================

def create_response(data=None, status="ok", message=""):
    """Créer une réponse API standardisée"""
    response = {
        "status": status,
        "message": message,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }
    return json.dumps(response)

def handle_error(error, endpoint_name):
    """Gestion standardisée des erreurs"""
    error_msg = str(error)
    print(f"❌ Erreur dans {endpoint_name}: {error_msg}")
    print(traceback.format_exc())
    
    return create_response(
        status="error",
        message=f"Erreur dans {endpoint_name}: {error_msg}"
    )

# ===================================================================
# ENDPOINTS API
# ===================================================================

@app.route('/get_model_status')
def get_model_status():
    """Récupérer le statut du modèle ML"""
    
    try:
        if not init_success:
            return create_response(
                data={"status": "Erreur d'initialisation", "error": init_error},
                status="error"
            )
        
        model_info = predictor.get_model_info()
        
        return create_response(data=model_info)
        
    except Exception as e:
        return handle_error(e, "get_model_status")

@app.route('/get_connection_status')
def get_connection_status():
    """Vérifier le statut des connexions ServiceNow"""
    
    try:
        if not init_success:
            return create_response(
                data={"status": "Erreur", "error": init_error},
                status="error"
            )
        
        connection_status = connector.get_connection_status()
        
        return create_response(data=connection_status)
        
    except Exception as e:
        return handle_error(e, "get_connection_status")

@app.route('/test_connection')
def test_connection():
    """Tester les connexions système"""
    
    try:
        if not init_success:
            return create_response(
                data={"success": False, "error": init_error},
                status="error"
            )
        
        # Test du modèle
        model_info = predictor.get_model_info()
        model_ok = model_info.get("status") == "Modèle chargé"
        
        # Test des connexions
        connection_status = connector.get_connection_status()
        connection_ok = connection_status.get("status") == "Connecté"
        
        success = model_ok and connection_ok
        
        return create_response(data={
            "success": success,
            "model_status": model_ok,
            "connection_status": connection_ok,
            "details": {
                "model": model_info,
                "connection": connection_status
            }
        })
        
    except Exception as e:
        return handle_error(e, "test_connection")

@app.route('/analyze_change')
def analyze_change():
    """Analyser un changement spécifique"""
    
    try:
        if not init_success:
            return create_response(
                status="error",
                message="Système non initialisé"
            )
        
        # Récupérer la référence du changement
        change_ref = request.args.get('change_ref')
        
        if not change_ref:
            return create_response(
                status="error", 
                message="Référence de changement manquante"
            )
        
        # Validation du format
        if not connector.validate_change_reference(change_ref):
            return create_response(
                status="error",
                message="Format de référence invalide"
            )
        
        # Récupération des données
        change_data = connector.get_change_data(change_ref)
        
        if not change_data:
            return create_response(data={
                "change_found": False,
                "change_ref": change_ref
            })
        
        # Analyse ML
        detailed_analysis = predictor.get_detailed_analysis(change_data)
        
        return create_response(data={
            "change_found": True,
            "change_ref": change_ref,
            "change_data": change_data,
            "detailed_analysis": detailed_analysis
        })
        
    except Exception as e:
        return handle_error(e, "analyze_change")

@app.route('/get_team_stats')
def get_team_stats():
    """Récupérer les statistiques d'une équipe"""
    
    try:
        if not init_success:
            return create_response(
                status="error",
                message="Système non initialisé"
            )
        
        assignment_group = request.args.get('assignment_group')
        
        if not assignment_group:
            return create_response(
                status="error",
                message="Nom d'équipe manquant"
            )
        
        team_stats = connector.get_team_statistics(assignment_group)
        
        return create_response(data=team_stats)
        
    except Exception as e:
        return handle_error(e, "get_team_stats")

@app.route('/get_incidents')
def get_incidents():
    """Récupérer les incidents liés à un CI"""
    
    try:
        if not init_success:
            return create_response(
                status="error",
                message="Système non initialisé"
            )
        
        cmdb_ci = request.args.get('cmdb_ci')
        
        if not cmdb_ci:
            return create_response(
                status="error",
                message="CI manquant"
            )
        
        incidents_data = connector.get_solution_incidents(cmdb_ci)
        
        return create_response(data=incidents_data)
        
    except Exception as e:
        return handle_error(e, "get_incidents")

@app.route('/get_similar_changes')
def get_similar_changes():
    """Récupérer les changements similaires"""
    
    try:
        if not init_success:
            return create_response(
                status="error",
                message="Système non initialisé"
            )
        
        change_ref = request.args.get('change_ref')
        
        if not change_ref:
            return create_response(
                status="error",
                message="Référence de changement manquante"
            )
        
        # Récupérer les données du changement d'abord
        change_data = connector.get_change_data(change_ref)
        
        if not change_data:
            return create_response(
                status="error",
                message="Changement non trouvé"
            )
        
        # Chercher les changements similaires
        similar_changes = connector.find_similar_changes(change_data)
        
        return create_response(data=similar_changes)
        
    except Exception as e:
        return handle_error(e, "get_similar_changes")

# ===================================================================
# ENDPOINT DE DIAGNOSTIC
# ===================================================================

@app.route('/diagnostic')
def diagnostic():
    """Endpoint de diagnostic complet du système"""
    
    try:
        diagnostic_info = {
            "initialization": {
                "success": init_success,
                "error": init_error if not init_success else None
            },
            "components": {},
            "datasets": {},
            "timestamp": datetime.now().isoformat()
        }
        
        if init_success:
            # Test du prédicteur
            try:
                model_info = predictor.get_model_info()
                diagnostic_info["components"]["predictor"] = {
                    "status": "OK",
                    "details": model_info
                }
            except Exception as e:
                diagnostic_info["components"]["predictor"] = {
                    "status": "ERROR",
                    "error": str(e)
                }
            
            # Test du connecteur
            try:
                connection_status = connector.get_connection_status()
                diagnostic_info["components"]["connector"] = {
                    "status": "OK",
                    "details": connection_status
                }
            except Exception as e:
                diagnostic_info["components"]["connector"] = {
                    "status": "ERROR",
                    "error": str(e)
                }
            
            # Test des datasets
            datasets_to_test = [
                "change_request_without_empty_columns",
                "incident_filtree"
            ]
            
            for dataset_name in datasets_to_test:
                try:
                    dataset = dataiku.Dataset(dataset_name)
                    df = dataset.get_dataframe(limit=1)
                    diagnostic_info["datasets"][dataset_name] = {
                        "status": "OK",
                        "columns": len(df.columns),
                        "sample_available": True
                    }
                except Exception as e:
                    diagnostic_info["datasets"][dataset_name] = {
                        "status": "ERROR",
                        "error": str(e)
                    }
        
        return create_response(data=diagnostic_info)
        
    except Exception as e:
        return handle_error(e, "diagnostic")

# ===================================================================
# ENDPOINT DE SANTÉ
# ===================================================================

@app.route('/health')
def health():
    """Endpoint de santé simple"""
    
    return create_response(data={
        "status": "alive",
        "initialization": init_success,
        "timestamp": datetime.now().isoformat()
    })

# ===================================================================
# GESTION DES ERREURS GLOBALES
# ===================================================================

@app.errorhandler(404)
def not_found(error):
    return create_response(
        status="error",
        message="Endpoint non trouvé"
    ), 404

@app.errorhandler(500)
def internal_error(error):
    return create_response(
        status="error",
        message="Erreur interne du serveur"
    ), 500

# ===================================================================
# POINT D'ENTRÉE
# ===================================================================

if __name__ == '__main__':
    print("🚀 Démarrage du backend Change Risk Analyzer")
    print(f"Initialisation: {'✅ OK' if init_success else '❌ ERROR'}")
    if not init_success:
        print(f"Erreur: {init_error}")
```

## 🎯 **MAINTENANT VOUS AVEZ UNE WEBAPP COMPLÈTE !**

### **✅ AVANTAGES DE CETTE APPROCHE :**
- **Plus stable** que Dash
- **Interface moderne** et responsive
- **API REST** claire et documentée
- **Gestion d'erreurs** robuste
- **100% données réelles** ServiceNow

### **🚀 PROCHAINES ÉTAPES :**
1. **Sauvegarder** chaque onglet avec son code
2. **Démarrer** la webapp
3. **Tester** les fonctionnalités

**Cette webapp devrait marcher du premier coup !** 🎯

J'ai créé une webapp complète avec HTML/CSS/JavaScript/Python qui reproduit toutes les fonctionnalités de la version Dash mais avec une architecture plus stable et traditionnelle.

**Structure:**

1. **HTML:** Interface utilisateur complète avec structure sémantique
2. **CSS:** Design moderne avec gradients, animations, responsive design
3. **JavaScript:** Logique frontend avec appels AJAX, gestion d'état, validation
4. **Python:** Backend Flask avec endpoints REST clairs

**Fonctionnalités:**
- Vérification du statut système
- Analyse de changements en temps réel
- Affichage des résultats avec score de risque
- Onglets contextuels (statistiques équipe, incidents, changements similaires)
- Gestion d'erreurs robuste
- Interface responsive
- Notifications toast
- Loading states

**Avantages par rapport à Dash:**
- Plus stable (pas de problèmes de layout)
- Plus de contrôle sur l'interface
- Meilleure performance
- Plus facile à déboguer
- Architecture claire frontend/backend

L'utilisateur devrait pouvoir copier-coller chaque code dans l'onglet correspondant et avoir une webapp fonctionnelle immédiatement.

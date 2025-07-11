L'utilisateur veut maintenant le fichier CSS complet avec les corrections pour le format CAGIPCHG0290677 et toutes les améliorations que j'ai ajoutées dans le HTML (modals, diagnostic, boutons supplémentaires, etc.).

Je dois fournir un CSS complet qui inclut :
1. Tous les styles de base existants
2. Les styles pour les nouveaux éléments (modals, diagnostic, boutons, etc.)
3. Les améliorations visuelles
4. Les styles responsive
5. Les animations et transitions
6. Les styles pour les nouvelles fonctionnalités

## 🎨 **ONGLET CSS - VERSION COMPLÈTE AVEC TOUTES LES AMÉLIORATIONS :**

```css
/* ===================================================================
   CHANGE RISK ANALYZER - STYLES MODERNES COMPLETS
   Version avec modals, diagnostic et améliorations UX
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
    min-height: 100vh;
}

/* ===================================================================
   HEADER PRINCIPAL
   ================================================================== */

.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='60' height='60' viewBox='0 0 60 60'%3E%3Cg fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.1'%3E%3Cpath d='m36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E") repeat;
    opacity: 0.1;
    z-index: 0;
}

.main-header h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 300;
    position: relative;
    z-index: 1;
}

.main-header p {
    margin: 0.5rem 0 0 0;
    font-size: 1.1rem;
    opacity: 0.9;
    position: relative;
    z-index: 1;
}

/* ===================================================================
   CONTAINER ET LAYOUT
   ================================================================== */

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 1rem;
}

.main-row {
    display: flex;
    gap: 2rem;
    align-items: flex-start;
    min-height: 600px;
}

.sidebar {
    width: 30%;
    min-width: 300px;
    position: sticky;
    top: 1rem;
}

.main-content {
    flex: 1;
    min-width: 0;
}

/* ===================================================================
   STATUS CARDS
   ================================================================== */

.status-card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border: 1px solid #e9ecef;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    transition: all 0.3s ease;
    position: relative;
}

.status-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
}

.status-card h6 {
    margin: 0 0 0.5rem 0;
    font-weight: 600;
    color: #495057;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.status-card p {
    margin: 0;
    font-weight: 500;
    font-size: 1.1rem;
}

.status-card small {
    color: #6c757d;
    font-size: 0.875rem;
    display: block;
    margin-top: 0.5rem;
}

/* Status couleurs avec animations */
.status-success {
    border-left: 4px solid #28a745;
    background: linear-gradient(135deg, #d4edda 0%, #ffffff 100%);
    animation: statusSuccess 0.5s ease-in-out;
}

.status-error {
    border-left: 4px solid #dc3545;
    background: linear-gradient(135deg, #f8d7da 0%, #ffffff 100%);
    animation: statusError 0.5s ease-in-out;
}

.status-warning {
    border-left: 4px solid #ffc107;
    background: linear-gradient(135deg, #fff3cd 0%, #ffffff 100%);
    animation: statusWarning 0.5s ease-in-out;
}

@keyframes statusSuccess {
    0% { transform: scale(0.95); opacity: 0; }
    100% { transform: scale(1); opacity: 1; }
}

@keyframes statusError {
    0% { transform: scale(0.95); opacity: 0; }
    100% { transform: scale(1); opacity: 1; }
}

@keyframes statusWarning {
    0% { transform: scale(0.95); opacity: 0; }
    100% { transform: scale(1); opacity: 1; }
}

/* ===================================================================
   BOXES D'INFORMATION
   ================================================================== */

.info-box {
    background: linear-gradient(135deg, #d1ecf1 0%, #f8fffe 100%);
    border: 1px solid #bee5eb;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.success-box {
    background: linear-gradient(135deg, #d4edda 0%, #f8fff9 100%);
    border: 1px solid #c3e6cb;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.warning-box {
    background: linear-gradient(135deg, #fff3cd 0%, #fffef8 100%);
    border: 1px solid #ffeaa7;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.error-box {
    background: linear-gradient(135deg, #f8d7da 0%, #fffafa 100%);
    border: 1px solid #f5c6cb;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

/* ===================================================================
   ZONE DE SAISIE
   ================================================================== */

.input-section {
    background: white;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 2rem;
    border: 1px solid #e9ecef;
}

.input-group {
    margin-bottom: 1.5rem;
}

.input-group label {
    display: block;
    margin-bottom: 0.75rem;
    font-weight: 600;
    color: #495057;
    font-size: 1.1rem;
}

.input-group input {
    width: 100%;
    padding: 1rem;
    border: 2px solid #e9ecef;
    border-radius: 10px;
    font-size: 1.1rem;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    letter-spacing: 1px;
    transition: all 0.3s ease;
    background: #fafafa;
}

.input-group input:focus {
    outline: none;
    border-color: #667eea;
    background: white;
    box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
    transform: translateY(-1px);
}

.input-group input.valid {
    border-color: #28a745;
    background: #f8fff9;
    box-shadow: 0 0 0 4px rgba(40, 167, 69, 0.1);
}

.input-group input.invalid {
    border-color: #dc3545;
    background: #fffafa;
    box-shadow: 0 0 0 4px rgba(220, 53, 69, 0.1);
}

.input-group small {
    color: #6c757d;
    font-size: 0.9rem;
    margin-top: 0.5rem;
    display: block;
}

/* ===================================================================
   BOUTONS
   ================================================================== */

.button-group {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    align-items: center;
}

.btn-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 1rem 2rem;
    border-radius: 10px;
    font-size: 1.1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    display: inline-flex;
    align-items: center;
    gap: 0.75rem;
    box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    position: relative;
    overflow: hidden;
}

.btn-primary::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s;
}

.btn-primary:hover::before {
    left: 100%;
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
}

.btn-primary:active {
    transform: translateY(0);
}

.btn-secondary {
    background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
    color: white;
    border: none;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    box-shadow: 0 2px 4px rgba(108, 117, 125, 0.3);
}

.btn-secondary:hover {
    background: linear-gradient(135deg, #5a6268 0%, #343a40 100%);
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(108, 117, 125, 0.4);
}

.btn-tab {
    background: #f8f9fa;
    border: 2px solid #dee2e6;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    margin: 0.25rem;
    cursor: pointer;
    transition: all 0.3s ease;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}

.btn-tab:hover {
    background: #e9ecef;
    transform: translateY(-1px);
}

.btn-tab.active {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-color: #667eea;
    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
}

/* ===================================================================
   AIDE CONTEXTUELLE
   ================================================================== */

.help-section {
    margin-top: 1rem;
}

.help-section details {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 0;
    transition: all 0.3s ease;
}

.help-section details[open] {
    background: white;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.help-section summary {
    padding: 0.75rem 1rem;
    cursor: pointer;
    font-weight: 500;
    color: #495057;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.help-section summary::-webkit-details-marker {
    display: none;
}

.help-section summary::after {
    content: '▼';
    margin-left: auto;
    transition: transform 0.3s ease;
}

.help-section details[open] summary::after {
    transform: rotate(180deg);
}

/* ===================================================================
   CARTES DE RÉSULTATS
   ================================================================== */

.risk-card {
    background: linear-gradient(135deg, white 0%, #f8f9fa 100%);
    padding: 3rem 2rem;
    border-radius: 20px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    border-left: 8px solid #667eea;
    margin: 2rem 0;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.risk-card::before {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 100px;
    height: 100px;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), transparent);
    border-radius: 0 0 0 100px;
}

.risk-score {
    font-size: 4rem;
    font-weight: 300;
    color: #667eea;
    margin: 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.risk-level {
    font-size: 1.8rem;
    margin: 1rem 0;
    font-weight: 600;
    color: #495057;
}

.risk-interpretation {
    font-style: italic;
    color: #6c757d;
    margin: 0;
    font-size: 1.1rem;
}

/* ===================================================================
   MÉTRIQUES
   ================================================================== */

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}

.metric-card {
    background: linear-gradient(135deg, #f8f9fa 0%, white 100%);
    padding: 2rem 1.5rem;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #e9ecef;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #667eea, #764ba2);
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #667eea;
    margin: 0;
    text-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

.metric-title {
    color: #495057;
    margin: 0.75rem 0 0 0;
    font-weight: 600;
    font-size: 1.1rem;
}

.metric-subtitle {
    color: #6c757d;
    font-size: 0.9rem;
    margin: 0.5rem 0 0 0;
}

/* ===================================================================
   DÉTAILS EN COLONNES
   ================================================================== */

.details-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 3rem;
    margin: 3rem 0;
}

.details-section h4 {
    color: #495057;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-size: 1.3rem;
    font-weight: 600;
}

.features-list {
    background: linear-gradient(135deg, #f8f9fa 0%, white 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border: 1px solid #e9ecef;
}

.features-list p {
    margin: 1rem 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid #f1f3f4;
}

.features-list p:last-child {
    border-bottom: none;
}

.features-list strong {
    color: #495057;
    font-weight: 600;
}

.features-list span {
    font-family: 'Consolas', 'Monaco', monospace;
    background: #e9ecef;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.9rem;
}

/* ===================================================================
   CHANGEMENTS SIMILAIRES
   ================================================================== */

.similar-change {
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border: 1px solid #dee2e6;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.similar-change::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    background: linear-gradient(180deg, #28a745, #20c997);
}

.similar-change.success {
    background: linear-gradient(135deg, #d4edda 0%, #f8fff9 100%);
    border-color: #c3e6cb;
}

.similar-change.success::before {
    background: linear-gradient(180deg, #28a745, #20c997);
}

.similar-change.error {
    background: linear-gradient(135deg, #f8d7da 0%, #fffafa 100%);
    border-color: #f5c6cb;
}

.similar-change.error::before {
    background: linear-gradient(180deg, #dc3545, #e74c3c);
}

.similar-change.warning {
    background: linear-gradient(135deg, #fff3cd 0%, #fffef8 100%);
    border-color: #ffeaa7;
}

.similar-change.warning::before {
    background: linear-gradient(180deg, #ffc107, #f39c12);
}

.similar-change:hover {
    transform: translateX(4px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

/* ===================================================================
   ONGLETS
   ================================================================== */

.tabs-container {
    margin: 3rem 0;
}

.tabs-buttons {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 2rem;
    flex-wrap: wrap;
    border-bottom: 2px solid #e9ecef;
    padding-bottom: 1rem;
}

.tab-content {
    padding: 2rem;
    background: white;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    min-height: 300px;
    border: 1px solid #e9ecef;
}

/* ===================================================================
   MODALS
   ================================================================== */

.modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.6);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 10000;
    backdrop-filter: blur(4px);
    animation: modalFadeIn 0.3s ease;
}

@keyframes modalFadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.modal-content {
    background: white;
    border-radius: 20px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    max-width: 90vw;
    max-height: 90vh;
    overflow: hidden;
    animation: modalSlideIn 0.3s ease;
}

@keyframes modalSlideIn {
    from { transform: scale(0.8) translateY(-20px); opacity: 0; }
    to { transform: scale(1) translateY(0); opacity: 1; }
}

.modal-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.modal-header h2 {
    margin: 0;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.modal-close {
    background: none;
    border: none;
    color: white;
    font-size: 2rem;
    cursor: pointer;
    padding: 0.5rem;
    border-radius: 50%;
    transition: background 0.3s ease;
}

.modal-close:hover {
    background: rgba(255,255,255,0.2);
}

.modal-body {
    padding: 2rem;
    max-height: 70vh;
    overflow-y: auto;
}

/* Styles pour le diagnostic */
.diagnostic-results {
    max-width: 600px;
}

.diagnostic-section {
    margin-bottom: 2rem;
}

.diagnostic-section h4 {
    color: #495057;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ===================================================================
   LOADING OVERLAY
   ================================================================== */

.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.7);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 9999;
    backdrop-filter: blur(4px);
}

.loading-spinner {
    background: white;
    padding: 3rem 2rem;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    max-width: 400px;
}

.loading-spinner i {
    color: #667eea;
    margin-bottom: 1.5rem;
}

.loading-spinner p {
    margin: 1rem 0;
    font-size: 1.1rem;
    color: #495057;
}

.loading-progress {
    width: 100%;
    height: 4px;
    background: #e9ecef;
    border-radius: 2px;
    overflow: hidden;
    margin-top: 1rem;
}

.progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #667eea, #764ba2);
    width: 0;
    animation: progressAnimation 2s ease-in-out infinite;
}

@keyframes progressAnimation {
    0% { width: 0; transform: translateX(-100%); }
    50% { width: 100%; transform: translateX(0); }
    100% { width: 100%; transform: translateX(100%); }
}

/* ===================================================================
   TOAST NOTIFICATIONS
   ================================================================== */

#toast-container {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 10000;
}

.toast {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    margin-bottom: 1rem;
    min-width: 320px;
    border-left: 4px solid #28a745;
    animation: toastSlideIn 0.4s ease;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.2);
}

.toast.error {
    border-left-color: #dc3545;
}

.toast.warning {
    border-left-color: #ffc107;
}

.toast.success {
    border-left-color: #28a745;
}

@keyframes toastSlideIn {
    from { 
        transform: translateX(100%); 
        opacity: 0; 
    }
    to { 
        transform: translateX(0); 
        opacity: 1; 
    }
}

/* ===================================================================
   FOOTER
   ================================================================== */

footer {
    margin-top: 4rem;
    padding-top: 3rem;
    border-top: 2px solid #e9ecef;
}

.footer-content {
    text-align: center;
    color: #6c757d;
    padding: 2rem;
    background: linear-gradient(135deg, #f8f9fa 0%, white 100%);
    border-radius: 15px;
    border: 1px solid #e9ecef;
}

.footer-content p {
    margin: 0.5rem 0;
}

.footer-links {
    margin-top: 1rem;
}

.footer-links a {
    color: #667eea;
    text-decoration: none;
    margin: 0 0.5rem;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    transition: all 0.3s ease;
}

.footer-links a:hover {
    background: #667eea;
    color: white;
}

/* ===================================================================
   RESPONSIVE DESIGN
   ================================================================== */

@media (max-width: 768px) {
    .main-row {
        flex-direction: column;
        gap: 1rem;
    }
    
    .sidebar {
        width: 100%;
        position: static;
    }
    
    .details-grid {
        grid-template-columns: 1fr;
        gap: 2rem;
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
    
    .tabs-buttons {
        justify-content: center;
    }
    
    .modal-content {
        margin: 1rem;
        max-width: calc(100vw - 2rem);
    }
    
    .modal-header {
        padding: 1.5rem;
    }
    
    .modal-body {
        padding: 1.5rem;
    }
    
    .risk-score {
        font-size: 3rem;
    }
    
    .risk-card {
        padding: 2rem 1.5rem;
    }
}

@media (max-width: 480px) {
    .container {
        padding: 0.5rem;
    }
    
    .main-header {
        padding: 1.5rem;
        border-radius: 10px;
    }
    
    .input-section {
        padding: 1.5rem;
    }
    
    .button-group {
        flex-direction: column;
        align-items: stretch;
    }
    
    .btn-primary,
    .btn-secondary {
        justify-content: center;
    }
    
    .toast {
        min-width: 280px;
        margin-left: 1rem;
        margin-right: 1rem;
    }
    
    #toast-container {
        right: 0;
        left: 0;
    }
}

/* ===================================================================
   UTILITIES ET ANIMATIONS
   ================================================================== */

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

/* Animations pour les éléments qui apparaissent */
.fade-in {
    animation: fadeIn 0.5s ease-in-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Effet de pulsation pour les éléments en chargement */
.pulse {
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

/* Effet de shake pour les erreurs */
.shake {
    animation: shake 0.5s ease-in-out;
}

@keyframes shake {
    0%, 100% { transform: translateX(0); }
    25% { transform: translateX(-5px); }
    75% { transform: translateX(5px); }
}

/* Scrollbar personnalisée */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #667eea, #764ba2);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #5a6fd8, #6a4c93);
}

/* Focus visible pour l'accessibilité */
*:focus-visible {
    outline: 2px solid #667eea;
    outline-offset: 2px;
}

/* Transitions fluides pour tous les éléments interactifs */
button, input, select, textarea, a {
    transition: all 0.3s ease;
}

/* ===================================================================
   DARK MODE SUPPORT (OPTIONNEL)
   ================================================================== */

@media (prefers-color-scheme: dark) {
    /* Variables pour le dark mode si souhaité dans le futur */
    :root {
        --bg-primary: #1a1a1a;
        --bg-secondary: #2d2d2d;
        --text-primary: #ffffff;
        --text-secondary: #cccccc;
    }
}

/* ===================================================================
   PRINT STYLES
   ================================================================== */

@media print {
    .loading-overlay,
    #toast-container,
    .modal,
    .button-group,
    .sidebar {
        display: none !important;
    }
    
    .main-row {
        flex-direction: column;
    }
    
    .main-content {
        width: 100%;
    }
    
    .main-header {
        background: #667eea !important;
        -webkit-print-color-adjust: exact;
        color-adjust: exact;
    }
}
```

## 🎨 **AMÉLIORATIONS CSS AJOUTÉES :**

### **✨ NOUVELLES FONCTIONNALITÉS :**
- **Modals** avec animations et backdrop blur
- **Toast notifications** améliorées
- **Loading overlay** avec progress bar animée
- **Diagnostic** avec styles dédiés
- **Aide contextuelle** dépliable

### **🎯 ANIMATIONS ET TRANSITIONS :**
- **Animations de statut** (success, error, warning)
- **Effet de survol** sur toutes les cartes
- **Transitions fluides** partout
- **Animations d'apparition** pour les éléments

### **📱 RESPONSIVE AMÉLIORÉ :**
- **Mobile-first** approche
- **Breakpoints** optimisés
- **Touch-friendly** boutons
- **Modals responsives**

### **🎨 DESIGN MODERNE :**
- **Gradients** subtils partout
- **Box-shadows** améliorées
- **Border-radius** cohérents
- **Typography** améliorée

**Maintenant on passe au fichier PYTHON ?** 🐍L'utilisateur veut maintenant le fichier CSS complet avec les corrections pour le format CAGIPCHG0290677 et toutes les améliorations que j'ai ajoutées dans le HTML (modals, diagnostic, boutons supplémentaires, etc.).

Je dois fournir un CSS complet qui inclut :
1. Tous les styles de base existants
2. Les styles pour les nouveaux éléments (modals, diagnostic, boutons, etc.)
3. Les améliorations visuelles
4. Les styles responsive
5. Les animations et transitions
6. Les styles pour les nouvelles fonctionnalités

## 🎨 **ONGLET CSS - VERSION COMPLÈTE AVEC TOUTES LES AMÉLIORATIONS :**

```css
/* ===================================================================
   CHANGE RISK ANALYZER - STYLES MODERNES COMPLETS
   Version avec modals, diagnostic et améliorations UX
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
    min-height: 100vh;
}

/* ===================================================================
   HEADER PRINCIPAL
   ================================================================== */

.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='60' height='60' viewBox='0 0 60 60'%3E%3Cg fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.1'%3E%3Cpath d='m36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E") repeat;
    opacity: 0.1;
    z-index: 0;
}

.main-header h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 300;
    position: relative;
    z-index: 1;
}

.main-header p {
    margin: 0.5rem 0 0 0;
    font-size: 1.1rem;
    opacity: 0.9;
    position: relative;
    z-index: 1;
}

/* ===================================================================
   CONTAINER ET LAYOUT
   ================================================================== */

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 1rem;
}

.main-row {
    display: flex;
    gap: 2rem;
    align-items: flex-start;
    min-height: 600px;
}

.sidebar {
    width: 30%;
    min-width: 300px;
    position: sticky;
    top: 1rem;
}

.main-content {
    flex: 1;
    min-width: 0;
}

/* ===================================================================
   STATUS CARDS
   ================================================================== */

.status-card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border: 1px solid #e9ecef;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    transition: all 0.3s ease;
    position: relative;
}

.status-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
}

.status-card h6 {
    margin: 0 0 0.5rem 0;
    font-weight: 600;
    color: #495057;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.status-card p {
    margin: 0;
    font-weight: 500;
    font-size: 1.1rem;
}

.status-card small {
    color: #6c757d;
    font-size: 0.875rem;
    display: block;
    margin-top: 0.5rem;
}

/* Status couleurs avec animations */
.status-success {
    border-left: 4px solid #28a745;
    background: linear-gradient(135deg, #d4edda 0%, #ffffff 100%);
    animation: statusSuccess 0.5s ease-in-out;
}

.status-error {
    border-left: 4px solid #dc3545;
    background: linear-gradient(135deg, #f8d7da 0%, #ffffff 100%);
    animation: statusError 0.5s ease-in-out;
}

.status-warning {
    border-left: 4px solid #ffc107;
    background: linear-gradient(135deg, #fff3cd 0%, #ffffff 100%);
    animation: statusWarning 0.5s ease-in-out;
}

@keyframes statusSuccess {
    0% { transform: scale(0.95); opacity: 0; }
    100% { transform: scale(1); opacity: 1; }
}

@keyframes statusError {
    0% { transform: scale(0.95); opacity: 0; }
    100% { transform: scale(1); opacity: 1; }
}

@keyframes statusWarning {
    0% { transform: scale(0.95); opacity: 0; }
    100% { transform: scale(1); opacity: 1; }
}

/* ===================================================================
   BOXES D'INFORMATION
   ================================================================== */

.info-box {
    background: linear-gradient(135deg, #d1ecf1 0%, #f8fffe 100%);
    border: 1px solid #bee5eb;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.success-box {
    background: linear-gradient(135deg, #d4edda 0%, #f8fff9 100%);
    border: 1px solid #c3e6cb;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.warning-box {
    background: linear-gradient(135deg, #fff3cd 0%, #fffef8 100%);
    border: 1px solid #ffeaa7;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.error-box {
    background: linear-gradient(135deg, #f8d7da 0%, #fffafa 100%);
    border: 1px solid #f5c6cb;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

/* ===================================================================
   ZONE DE SAISIE
   ================================================================== */

.input-section {
    background: white;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 2rem;
    border: 1px solid #e9ecef;
}

.input-group {
    margin-bottom: 1.5rem;
}

.input-group label {
    display: block;
    margin-bottom: 0.75rem;
    font-weight: 600;
    color: #495057;
    font-size: 1.1rem;
}

.input-group input {
    width: 100%;
    padding: 1rem;
    border: 2px solid #e9ecef;
    border-radius: 10px;
    font-size: 1.1rem;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    letter-spacing: 1px;
    transition: all 0.3s ease;
    background: #fafafa;
}

.input-group input:focus {
    outline: none;
    border-color: #667eea;
    background: white;
    box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
    transform: translateY(-1px);
}

.input-group input.valid {
    border-color: #28a745;
    background: #f8fff9;
    box-shadow: 0 0 0 4px rgba(40, 167, 69, 0.1);
}

.input-group input.invalid {
    border-color: #dc3545;
    background: #fffafa;
    box-shadow: 0 0 0 4px rgba(220, 53, 69, 0.1);
}

.input-group small {
    color: #6c757d;
    font-size: 0.9rem;
    margin-top: 0.5rem;
    display: block;
}

/* ===================================================================
   BOUTONS
   ================================================================== */

.button-group {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    align-items: center;
}

.btn-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 1rem 2rem;
    border-radius: 10px;
    font-size: 1.1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    display: inline-flex;
    align-items: center;
    gap: 0.75rem;
    box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    position: relative;
    overflow: hidden;
}

.btn-primary::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s;
}

.btn-primary:hover::before {
    left: 100%;
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
}

.btn-primary:active {
    transform: translateY(0);
}

.btn-secondary {
    background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
    color: white;
    border: none;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    box-shadow: 0 2px 4px rgba(108, 117, 125, 0.3);
}

.btn-secondary:hover {
    background: linear-gradient(135deg, #5a6268 0%, #343a40 100%);
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(108, 117, 125, 0.4);
}

.btn-tab {
    background: #f8f9fa;
    border: 2px solid #dee2e6;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    margin: 0.25rem;
    cursor: pointer;
    transition: all 0.3s ease;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}

.btn-tab:hover {
    background: #e9ecef;
    transform: translateY(-1px);
}

.btn-tab.active {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-color: #667eea;
    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
}

/* ===================================================================
   AIDE CONTEXTUELLE
   ================================================================== */

.help-section {
    margin-top: 1rem;
}

.help-section details {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 0;
    transition: all 0.3s ease;
}

.help-section details[open] {
    background: white;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.help-section summary {
    padding: 0.75rem 1rem;
    cursor: pointer;
    font-weight: 500;
    color: #495057;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.help-section summary::-webkit-details-marker {
    display: none;
}

.help-section summary::after {
    content: '▼';
    margin-left: auto;
    transition: transform 0.3s ease;
}

.help-section details[open] summary::after {
    transform: rotate(180deg);
}

/* ===================================================================
   CARTES DE RÉSULTATS
   ================================================================== */

.risk-card {
    background: linear-gradient(135deg, white 0%, #f8f9fa 100%);
    padding: 3rem 2rem;
    border-radius: 20px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    border-left: 8px solid #667eea;
    margin: 2rem 0;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.risk-card::before {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 100px;
    height: 100px;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), transparent);
    border-radius: 0 0 0 100px;
}

.risk-score {
    font-size: 4rem;
    font-weight: 300;
    color: #667eea;
    margin: 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.risk-level {
    font-size: 1.8rem;
    margin: 1rem 0;
    font-weight: 600;
    color: #495057;
}

.risk-interpretation {
    font-style: italic;
    color: #6c757d;
    margin: 0;
    font-size: 1.1rem;
}

/* ===================================================================
   MÉTRIQUES
   ================================================================== */

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}

.metric-card {
    background: linear-gradient(135deg, #f8f9fa 0%, white 100%);
    padding: 2rem 1.5rem;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #e9ecef;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #667eea, #764ba2);
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #667eea;
    margin: 0;
    text-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

.metric-title {
    color: #495057;
    margin: 0.75rem 0 0 0;
    font-weight: 600;
    font-size: 1.1rem;
}

.metric-subtitle {
    color: #6c757d;
    font-size: 0.9rem;
    margin: 0.5rem 0 0 0;
}

/* ===================================================================
   DÉTAILS EN COLONNES
   ================================================================== */

.details-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 3rem;
    margin: 3rem 0;
}

.details-section h4 {
    color: #495057;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-size: 1.3rem;
    font-weight: 600;
}

.features-list {
    background: linear-gradient(135deg, #f8f9fa 0%, white 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border: 1px solid #e9ecef;
}

.features-list p {
    margin: 1rem 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid #f1f3f4;
}

.features-list p:last-child {
    border-bottom: none;
}

.features-list strong {
    color: #495057;
    font-weight: 600;
}

.features-list span {
    font-family: 'Consolas', 'Monaco', monospace;
    background: #e9ecef;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.9rem;
}

/* ===================================================================
   CHANGEMENTS SIMILAIRES
   ================================================================== */

.similar-change {
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border: 1px solid #dee2e6;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.similar-change::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    background: linear-gradient(180deg, #28a745, #20c997);
}

.similar-change.success {
    background: linear-gradient(135deg, #d4edda 0%, #f8fff9 100%);
    border-color: #c3e6cb;
}

.similar-change.success::before {
    background: linear-gradient(180deg, #28a745, #20c997);
}

.similar-change.error {
    background: linear-gradient(135deg, #f8d7da 0%, #fffafa 100%);
    border-color: #f5c6cb;
}

.similar-change.error::before {
    background: linear-gradient(180deg, #dc3545, #e74c3c);
}

.similar-change.warning {
    background: linear-gradient(135deg, #fff3cd 0%, #fffef8 100%);
    border-color: #ffeaa7;
}

.similar-change.warning::before {
    background: linear-gradient(180deg, #ffc107, #f39c12);
}

.similar-change:hover {
    transform: translateX(4px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

/* ===================================================================
   ONGLETS
   ================================================================== */

.tabs-container {
    margin: 3rem 0;
}

.tabs-buttons {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 2rem;
    flex-wrap: wrap;
    border-bottom: 2px solid #e9ecef;
    padding-bottom: 1rem;
}

.tab-content {
    padding: 2rem;
    background: white;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    min-height: 300px;
    border: 1px solid #e9ecef;
}

/* ===================================================================
   MODALS
   ================================================================== */

.modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.6);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 10000;
    backdrop-filter: blur(4px);
    animation: modalFadeIn 0.3s ease;
}

@keyframes modalFadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.modal-content {
    background: white;
    border-radius: 20px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    max-width: 90vw;
    max-height: 90vh;
    overflow: hidden;
    animation: modalSlideIn 0.3s ease;
}

@keyframes modalSlideIn {
    from { transform: scale(0.8) translateY(-20px); opacity: 0; }
    to { transform: scale(1) translateY(0); opacity: 1; }
}

.modal-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.modal-header h2 {
    margin: 0;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.modal-close {
    background: none;
    border: none;
    color: white;
    font-size: 2rem;
    cursor: pointer;
    padding: 0.5rem;
    border-radius: 50%;
    transition: background 0.3s ease;
}

.modal-close:hover {
    background: rgba(255,255,255,0.2);
}

.modal-body {
    padding: 2rem;
    max-height: 70vh;
    overflow-y: auto;
}

/* Styles pour le diagnostic */
.diagnostic-results {
    max-width: 600px;
}

.diagnostic-section {
    margin-bottom: 2rem;
}

.diagnostic-section h4 {
    color: #495057;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ===================================================================
   LOADING OVERLAY
   ================================================================== */

.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.7);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 9999;
    backdrop-filter: blur(4px);
}

.loading-spinner {
    background: white;
    padding: 3rem 2rem;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    max-width: 400px;
}

.loading-spinner i {
    color: #667eea;
    margin-bottom: 1.5rem;
}

.loading-spinner p {
    margin: 1rem 0;
    font-size: 1.1rem;
    color: #495057;
}

.loading-progress {
    width: 100%;
    height: 4px;
    background: #e9ecef;
    border-radius: 2px;
    overflow: hidden;
    margin-top: 1rem;
}

.progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #667eea, #764ba2);
    width: 0;
    animation: progressAnimation 2s ease-in-out infinite;
}

@keyframes progressAnimation {
    0% { width: 0; transform: translateX(-100%); }
    50% { width: 100%; transform: translateX(0); }
    100% { width: 100%; transform: translateX(100%); }
}

/* ===================================================================
   TOAST NOTIFICATIONS
   ================================================================== */

#toast-container {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 10000;
}

.toast {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    margin-bottom: 1rem;
    min-width: 320px;
    border-left: 4px solid #28a745;
    animation: toastSlideIn 0.4s ease;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.2);
}

.toast.error {
    border-left-color: #dc3545;
}

.toast.warning {
    border-left-color: #ffc107;
}

.toast.success {
    border-left-color: #28a745;
}

@keyframes toastSlideIn {
    from { 
        transform: translateX(100%); 
        opacity: 0; 
    }
    to { 
        transform: translateX(0); 
        opacity: 1; 
    }
}

/* ===================================================================
   FOOTER
   ================================================================== */

footer {
    margin-top: 4rem;
    padding-top: 3rem;
    border-top: 2px solid #e9ecef;
}

.footer-content {
    text-align: center;
    color: #6c757d;
    padding: 2rem;
    background: linear-gradient(135deg, #f8f9fa 0%, white 100%);
    border-radius: 15px;
    border: 1px solid #e9ecef;
}

.footer-content p {
    margin: 0.5rem 0;
}

.footer-links {
    margin-top: 1rem;
}

.footer-links a {
    color: #667eea;
    text-decoration: none;
    margin: 0 0.5rem;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    transition: all 0.3s ease;
}

.footer-links a:hover {
    background: #667eea;
    color: white;
}

/* ===================================================================
   RESPONSIVE DESIGN
   ================================================================== */

@media (max-width: 768px) {
    .main-row {
        flex-direction: column;
        gap: 1rem;
    }
    
    .sidebar {
        width: 100%;
        position: static;
    }
    
    .details-grid {
        grid-template-columns: 1fr;
        gap: 2rem;
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
    
    .tabs-buttons {
        justify-content: center;
    }
    
    .modal-content {
        margin: 1rem;
        max-width: calc(100vw - 2rem);
    }
    
    .modal-header {
        padding: 1.5rem;
    }
    
    .modal-body {
        padding: 1.5rem;
    }
    
    .risk-score {
        font-size: 3rem;
    }
    
    .risk-card {
        padding: 2rem 1.5rem;
    }
}

@media (max-width: 480px) {
    .container {
        padding: 0.5rem;
    }
    
    .main-header {
        padding: 1.5rem;
        border-radius: 10px;
    }
    
    .input-section {
        padding: 1.5rem;
    }
    
    .button-group {
        flex-direction: column;
        align-items: stretch;
    }
    
    .btn-primary,
    .btn-secondary {
        justify-content: center;
    }
    
    .toast {
        min-width: 280px;
        margin-left: 1rem;
        margin-right: 1rem;
    }
    
    #toast-container {
        right: 0;
        left: 0;
    }
}

/* ===================================================================
   UTILITIES ET ANIMATIONS
   ================================================================== */

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

/* Animations pour les éléments qui apparaissent */
.fade-in {
    animation: fadeIn 0.5s ease-in-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Effet de pulsation pour les éléments en chargement */
.pulse {
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

/* Effet de shake pour les erreurs */
.shake {
    animation: shake 0.5s ease-in-out;
}

@keyframes shake {
    0%, 100% { transform: translateX(0); }
    25% { transform: translateX(-5px); }
    75% { transform: translateX(5px); }
}

/* Scrollbar personnalisée */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #667eea, #764ba2);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #5a6fd8, #6a4c93);
}

/* Focus visible pour l'accessibilité */
*:focus-visible {
    outline: 2px solid #667eea;
    outline-offset: 2px;
}

/* Transitions fluides pour tous les éléments interactifs */
button, input, select, textarea, a {
    transition: all 0.3s ease;
}

/* ===================================================================
   DARK MODE SUPPORT (OPTIONNEL)
   ================================================================== */

@media (prefers-color-scheme: dark) {
    /* Variables pour le dark mode si souhaité dans le futur */
    :root {
        --bg-primary: #1a1a1a;
        --bg-secondary: #2d2d2d;
        --text-primary: #ffffff;
        --text-secondary: #cccccc;
    }
}

/* ===================================================================
   PRINT STYLES
   ================================================================== */

@media print {
    .loading-overlay,
    #toast-container,
    .modal,
    .button-group,
    .sidebar {
        display: none !important;
    }
    
    .main-row {
        flex-direction: column;
    }
    
    .main-content {
        width: 100%;
    }
    
    .main-header {
        background: #667eea !important;
        -webkit-print-color-adjust: exact;
        color-adjust: exact;
    }
}
```

## 🎨 **AMÉLIORATIONS CSS AJOUTÉES :**

### **✨ NOUVELLES FONCTIONNALITÉS :**
- **Modals** avec animations et backdrop blur
- **Toast notifications** améliorées
- **Loading overlay** avec progress bar animée
- **Diagnostic** avec styles dédiés
- **Aide contextuelle** dépliable

### **🎯 ANIMATIONS ET TRANSITIONS :**
- **Animations de statut** (success, error, warning)
- **Effet de survol** sur toutes les cartes
- **Transitions fluides** partout
- **Animations d'apparition** pour les éléments

### **📱 RESPONSIVE AMÉLIORÉ :**
- **Mobile-first** approche
- **Breakpoints** optimisés
- **Touch-friendly** boutons
- **Modals responsives**

### **🎨 DESIGN MODERNE :**
- **Gradients** subtils partout
- **Box-shadows** améliorées
- **Border-radius** cohérents
- **Typography** améliorée

**Maintenant on passe au fichier PYTHON ?** 🐍

L'utilisateur me dit qu'il n'y a que Dash disponible sur Dataiku, pas Streamlit. Je dois donc convertir tout le code Streamlit que j'ai fait en Dash. 

Le code initial qu'il voit :
```python
import dash_html_components as html 

# build your Dash app

app.layout = html.Div()
```

Je dois donc créer une webapp Dash complète avec :
1. Les imports Dash nécessaires
2. Le layout avec HTML components
3. Les callbacks pour l'interactivité
4. Le CSS pour le design
5. La logique d'analyse

C'est un changement important car Dash fonctionne très différemment de Streamlit :
- Dash utilise des callbacks pour l'interactivité
- Le CSS doit être défini différemment
- Les composants sont différents
- La structure est plus complexe

Je vais créer une webapp Dash complète qui reproduit les fonctionnalités de la webapp Streamlit.

## 🎯 **WEBAPP DASH COMPLÈTE - ÉTAPE PAR ÉTAPE**

### **ÉTAPE 1 : Remplacer tout le contenu par ce code complet :**

```python
"""
WEBAPP DASH - ANALYSE DES RISQUES DE CHANGEMENT SERVICENOW
100% DONNÉES RÉELLES - Interface Dash pour Dataiku
"""

import dash
from dash import dcc, html, Input, Output, State, callback_table
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime
import json

# Import de nos modules
from change_risk_predictor import ChangeRiskPredictor
from servicenow_connector import ServiceNowConnector

# ===================================================================
# INITIALISATION DE L'APP DASH
# ===================================================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# ===================================================================
# INITIALISATION DES COMPOSANTS MÉTIER
# ===================================================================

# Initialisation globale
predictor = ChangeRiskPredictor()
connector = ServiceNowConnector()

# Vérification du statut
model_info = predictor.get_model_info()
connection_status = connector.get_connection_status()

# ===================================================================
# CSS CUSTOM
# ===================================================================

custom_css = """
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .risk-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        text-align: center;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        border: 1px solid #e9ecef;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
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
    
    .similar-change {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #ddd;
    }
    
    .feature-display {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
"""

# ===================================================================
# FUNCTIONS UTILITAIRES
# ===================================================================

def create_status_card(title, status, details=None):
    """Créer une carte de statut"""
    color = "success" if "✅" in status else "danger" if "❌" in status else "warning"
    
    card_content = [
        html.H6(title, className="card-title"),
        html.P(status, className="card-text")
    ]
    
    if details:
        card_content.append(html.Small(details, className="text-muted"))
    
    return dbc.Card(
        dbc.CardBody(card_content),
        color=color,
        outline=True,
        className="mb-2"
    )

def create_metric_card(title, value, subtitle=None):
    """Créer une carte métrique"""
    card_content = [
        html.H3(value, className="text-primary mb-0"),
        html.P(title, className="mb-0"),
    ]
    
    if subtitle:
        card_content.append(html.Small(subtitle, className="text-muted"))
    
    return html.Div(card_content, className="metric-card")

def format_similar_change(change):
    """Formater un changement similaire"""
    close_code = change['dv_close_code']
    
    if close_code == 'Succès':
        icon = "✅"
        bg_color = "#d4edda"
    elif 'Échec' in str(close_code):
        icon = "❌"
        bg_color = "#f8d7da"
    else:
        icon = "⚠️"
        bg_color = "#fff3cd"
    
    duration_text = ""
    if change.get('duration_hours') is not None:
        duration_text = f" • Durée: {change['duration_hours']}h"
    
    return html.Div([
        html.P([
            html.Strong(f"{icon} {change['number']} - {close_code}"),
            html.Br(),
            html.Small(change['short_description'][:100] + "..."),
            html.Br(),
            html.Small(f"Similarité: {change['similarity_score']}%{duration_text}", 
                      className="text-muted")
        ])
    ], style={"background": bg_color, "padding": "1rem", "border-radius": "8px", "margin": "0.5rem 0"})

# ===================================================================
# LAYOUT PRINCIPAL
# ===================================================================

app.layout = html.Div([
    
    # CSS Custom
    html.Div([
        html.Style(custom_css)
    ]),
    
    # Header principal
    html.Div([
        html.H1("🔍 Change Risk Analyzer", style={"margin": "0", "font-size": "2.5rem"}),
        html.P("Analyseur de risques pour changements ServiceNow • 100% Données Réelles", 
               style={"margin": "0.5rem 0 0 0", "font-size": "1.1rem"})
    ], className="main-header"),
    
    # Container principal
    dbc.Container([
        
        # Row pour le statut et la saisie
        dbc.Row([
            
            # Colonne gauche - Statuts
            dbc.Col([
                html.H4("🤖 Statut du Système"),
                
                # Statut modèle
                create_status_card(
                    "Modèle ML",
                    "✅ Opérationnel" if model_info.get("status") == "Modèle chargé" else "❌ Non disponible",
                    f"Algorithme: {model_info.get('algorithm', 'N/A')}" if model_info.get("status") == "Modèle chargé" else None
                ),
                
                # Statut connexions
                create_status_card(
                    "Connexions ServiceNow",
                    "✅ Connecté" if connection_status.get('status') == 'Connecté' else "❌ Erreur",
                    "Tables: change_request & incident_filtree" if connection_status.get('status') == 'Connecté' else connection_status.get('error', '')
                ),
                
                # Informations modèle
                html.H5("📊 Performance Modèle", className="mt-3"),
                html.Div(id="model-performance-info")
                
            ], width=4),
            
            # Colonne droite - Interface principale
            dbc.Col([
                html.H4("📝 Analyse de Changement"),
                
                # Zone de saisie
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Référence du changement:"),
                        dbc.Input(
                            id="change-reference-input",
                            placeholder="CHG0012345",
                            type="text",
                            className="mb-2"
                        ),
                        html.Small("Format: CHG + 7 chiffres", className="text-muted")
                    ], width=8),
                    
                    dbc.Col([
                        html.Br(),
                        dbc.Button(
                            "🔍 Analyser",
                            id="analyze-button",
                            color="primary",
                            className="me-2",
                            n_clicks=0
                        ),
                        dbc.Button(
                            "ℹ️ Test",
                            id="test-button",
                            color="secondary",
                            n_clicks=0
                        )
                    ], width=4)
                ]),
                
                # Zone de résultats
                html.Hr(),
                html.Div(id="analysis-results")
                
            ], width=8)
        ]),
        
        # Zone pour les résultats détaillés
        html.Div(id="detailed-results", className="mt-4")
        
    ], fluid=True)
])

# ===================================================================
# CALLBACKS
# ===================================================================

@app.callback(
    Output("model-performance-info", "children"),
    Input("analyze-button", "n_clicks")  # Trigger sur le premier chargement
)
def update_model_info(n_clicks):
    """Afficher les informations de performance du modèle"""
    
    if model_info.get("status") != "Modèle chargé":
        return html.Div("❌ Modèle non disponible", className="error-box")
    
    training_info = model_info.get('training_info', {})
    perf = training_info.get('performance', {})
    
    if perf:
        return html.Div([
            html.P([
                html.Strong("Recall: "), perf.get('recall', 'N/A'), html.Br(),
                html.Strong("Precision: "), perf.get('precision', 'N/A'), html.Br(),
                html.Strong("Features: "), str(model_info.get('features', {}).get('count', 'N/A'))
            ])
        ], className="info-box")
    else:
        return html.Div("Informations de performance non disponibles", className="warning-box")

@app.callback(
    [Output("analysis-results", "children"),
     Output("detailed-results", "children")],
    [Input("analyze-button", "n_clicks"),
     Input("test-button", "n_clicks")],
    [State("change-reference-input", "value")]
)
def perform_analysis(analyze_clicks, test_clicks, change_ref):
    """Effectuer l'analyse du changement"""
    
    # Déterminer quel bouton a été cliqué
    ctx = dash.callback_context
    if not ctx.triggered:
        return "", ""
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Test de connexion
    if button_id == "test-button":
        status = connector.get_connection_status()
        if status.get('status') == 'Connecté':
            return html.Div("✅ Test de connexion réussi", className="success-box"), ""
        else:
            return html.Div(f"❌ Test échoué: {status.get('error', 'Erreur inconnue')}", className="error-box"), ""
    
    # Analyse du changement
    if button_id == "analyze-button":
        
        if not change_ref:
            return html.Div("⚠️ Veuillez saisir une référence de changement", className="warning-box"), ""
        
        # Validation format
        if not connector.validate_change_reference(change_ref):
            return html.Div("❌ Format invalide. Utilisez CHG + 7 chiffres (ex: CHG0012345)", className="error-box"), ""
        
        # Récupération des données
        change_data = connector.get_change_data(change_ref)
        
        if not change_data:
            return html.Div(f"❌ Changement {change_ref} non trouvé dans ServiceNow", className="error-box"), ""
        
        # Analyse ML
        try:
            detailed_analysis = predictor.get_detailed_analysis(change_data)
        except Exception as e:
            return html.Div(f"❌ Erreur analyse ML: {str(e)}", className="error-box"), ""
        
        # === RÉSULTATS PRINCIPAUX ===
        risk_score = detailed_analysis['risk_score']
        risk_level = detailed_analysis['risk_level']
        risk_color = detailed_analysis['risk_color']
        
        main_results = html.Div([
            html.H3(f"📊 Analyse de {change_ref}"),
            
            # Score principal
            html.Div([
                html.H1(f"{risk_color} {risk_score}%", 
                        style={"color": "#667eea", "margin": "0", "text-align": "center"}),
                html.H4("Risque d'échec", style={"margin": "0.5rem 0", "text-align": "center"}),
                html.P(f"Niveau: {risk_level}", style={"margin": "0", "text-align": "center", "font-weight": "bold"}),
                html.P(detailed_analysis['interpretation'], 
                       style={"margin": "0", "text-align": "center", "font-style": "italic"})
            ], className="risk-card")
        ])
        
        # === RÉSULTATS DÉTAILLÉS ===
        detailed_results = html.Div([
            
            dbc.Row([
                # Colonne gauche - Facteurs et recommandations
                dbc.Col([
                    html.H4("🚨 Facteurs de risque"),
                    html.Div([
                        html.Ul([
                            html.Li(factor) for factor in detailed_analysis['risk_factors']
                        ]) if detailed_analysis['risk_factors'] else html.P("Aucun facteur spécifique détecté")
                    ], className="info-box"),
                    
                    html.H4("💡 Recommandations"),
                    html.Div([
                        html.Ul([
                            html.Li(f"✅ {rec}") for rec in detailed_analysis['recommendations']
                        ])
                    ], className="success-box")
                    
                ], width=6),
                
                # Colonne droite - Caractéristiques techniques
                dbc.Col([
                    html.H4("🔧 Caractéristiques techniques"),
                    html.Div([
                        html.P([
                            html.Strong("Type SILCA: "), change_data.get('dv_u_type_change_silca', 'N/A'), html.Br(),
                            html.Strong("Type de changement: "), change_data.get('dv_type', 'N/A'), html.Br(),
                            html.Strong("Nombre de CAB: "), str(change_data.get('u_cab_count', 'N/A')), html.Br(),
                            html.Strong("Périmètre BCR: "), '✅' if change_data.get('u_bcr') else '❌', html.Br(),
                            html.Strong("Périmètre BPC: "), '✅' if change_data.get('u_bpc') else '❌'
                        ])
                    ], className="feature-display"),
                    
                    html.H4("📋 Métadonnées"),
                    html.Div([
                        html.P([
                            html.Strong("Équipe: "), change_data.get('dv_assignment_group', 'N/A'), html.Br(),
                            html.Strong("CI/Solution: "), change_data.get('dv_cmdb_ci', 'N/A'), html.Br(),
                            html.Strong("Catégorie: "), change_data.get('dv_category', 'N/A'), html.Br(),
                            html.Strong("État: "), change_data.get('dv_state', 'N/A')
                        ])
                    ], className="info-box")
                    
                ], width=6)
            ]),
            
            # Onglets pour informations contextuelles
            html.Hr(),
            html.H3("📈 Informations contextuelles"),
            
            dcc.Tabs(id="context-tabs", value="team-stats", children=[
                dcc.Tab(label="👥 Statistiques équipe", value="team-stats"),
                dcc.Tab(label="🛠️ Incidents liés", value="incidents"),
                dcc.Tab(label="📋 Changements similaires", value="similar-changes")
            ]),
            
            html.Div(id="context-content", style={"padding": "1rem"})
        ])
        
        # Stocker les données pour les onglets
        global current_change_data
        current_change_data = change_data
        
        return main_results, detailed_results
    
    return "", ""

@app.callback(
    Output("context-content", "children"),
    [Input("context-tabs", "value")]
)
def update_context_content(active_tab):
    """Mettre à jour le contenu des onglets contextuels"""
    
    try:
        change_data = current_change_data
    except:
        return html.Div("Aucune donnée de changement disponible")
    
    if active_tab == "team-stats":
        # Statistiques équipe
        team_stats = connector.get_team_statistics(change_data.get('dv_assignment_group'))
        
        if team_stats and 'error' not in team_stats:
            return dbc.Row([
                dbc.Col([
                    create_metric_card("Total changements", team_stats['total_changes'], "6 derniers mois")
                ], width=3),
                dbc.Col([
                    create_metric_card("Taux de succès", f"{team_stats['success_rate']}%")
                ], width=3),
                dbc.Col([
                    create_metric_card("Échecs", team_stats['failures'])
                ], width=3),
                dbc.Col([
                    last_failure = team_stats.get('last_failure_date')
                    if last_failure and pd.notna(last_failure):
                        days_ago = (datetime.now() - pd.to_datetime(last_failure)).days
                        create_metric_card("Dernier échec", f"Il y a {days_ago}j")
                    else:
                        create_metric_card("Dernier échec", "Aucun récent")
                ], width=3)
            ])
        else:
            return html.Div("⚠️ Statistiques équipe non disponibles", className="warning-box")
    
    elif active_tab == "incidents":
        # Incidents liés
        incidents_data = connector.get_solution_incidents(change_data.get('dv_cmdb_ci'))
        
        if incidents_data:
            return html.Div([
                dbc.Row([
                    dbc.Col([
                        create_metric_card("Total incidents", incidents_data['total_incidents'], "3 derniers mois")
                    ], width=4),
                    dbc.Col([
                        create_metric_card("Incidents critiques", incidents_data['critical_incidents'])
                    ], width=4),
                    dbc.Col([
                        if incidents_data['avg_resolution_hours'] > 0:
                            create_metric_card("Résolution moyenne", f"{incidents_data['avg_resolution_hours']}h")
                        else:
                            create_metric_card("Résolution moyenne", "N/A")
                    ], width=4)
                ]),
                
                html.P("🔍 Données extraites de la table incident_filtree", className="text-muted mt-2")
            ])
        else:
            return html.Div("⚠️ Aucun incident trouvé pour cette solution/CI", className="warning-box")
    
    elif active_tab == "similar-changes":
        # Changements similaires
        similar_changes = connector.find_similar_changes(change_data)
        
        if similar_changes:
            return html.Div([
                html.P("📋 Historique réel des changements similaires:"),
                html.Div([
                    format_similar_change(change) for change in similar_changes[:5]
                ])
            ])
        else:
            return html.Div("ℹ️ Aucun changement similaire trouvé", className="info-box")
    
    return html.Div("Contenu non disponible")

# ===================================================================
# FOOTER ET INFORMATIONS
# ===================================================================

app.layout.children.append(
    html.Footer([
        html.Hr(),
        html.Div([
            html.P("🤖 Change Risk Analyzer • 100% Données Réelles ServiceNow", 
                   style={"text-align": "center", "margin": "0"}),
            html.P("Tables connectées: change_request_without_empty_columns & incident_filtree", 
                   style={"text-align": "center", "margin": "0", "font-style": "italic"}),
            html.P("ZERO SIMULATION • ZERO DONNÉES FACTICES", 
                   style={"text-align": "center", "margin": "0", "font-weight": "bold"})
        ], style={"padding": "2rem", "color": "#666"})
    ])
)

# ===================================================================
# POINT D'ENTRÉE
# ===================================================================

if __name__ == '__main__':
    app.run_server(debug=True)
```

## 🎯 **ÉTAPES POUR IMPLÉMENTER :**

### **ÉTAPE 2 : Sauvegarder et tester**
1. **Remplacer** tout le contenu de votre webapp par ce code
2. **Sauvegarder** le fichier
3. **Démarrer** la webapp depuis Dataiku

### **ÉTAPE 3 : Vérifier les imports**
Si erreur d'import, ajoutez en haut :
```python
# Si dash_bootstrap_components pas disponible
# Remplacer les dbc. par html. équivalents
```

### **ÉTAPE 4 : Test**
- Interface devrait charger avec le header gradienté
- Statuts des connexions affichés à gauche
- Zone de saisie à droite

**Cette webapp Dash reproduit toutes les fonctionnalités de la version Streamlit !** 🚀

**Testez et dites-moi ce qui se passe !**

L'utilisateur m'a dit qu'il n'y a que Dash disponible sur Dataiku, pas Streamlit. J'ai donc converti toute la webapp Streamlit en webapp Dash.

Voici les principales différences que j'ai dû gérer :

1. **Structure** : Dash utilise un layout déclaratif avec des composants HTML, tandis que Streamlit est plus procédural
2. **Interactivité** : Dash utilise des callbacks pour gérer l'interactivité, tandis que Streamlit utilise des boutons et des widgets directs
3. **CSS** : J'ai gardé le même CSS mais adapté pour Dash
4. **Composants** : J'ai utilisé dash_bootstrap_components pour avoir des composants plus jolis
5. **Gestion d'état** : Dash gère l'état différemment, j'ai dû utiliser des variables globales pour partager les données entre callbacks

Points clés de la conversion :

- **Layout principal** : Structure HTML avec des divs et des containers Bootstrap
- **Callbacks** : Un callback principal pour l'analyse et un pour les onglets contextuels
- **Gestion des données** : Variable globale `current_change_data` pour partager entre callbacks
- **Affichage des résultats** : Utilisation de composants HTML pour reproduire l'apparence
- **Onglets** : Utilisation de dcc.Tabs au lieu des tabs Streamlit

La webapp devrait maintenant fonctionner avec Dash sur Dataiku et offrir les mêmes fonctionnalités que la version Streamlit.

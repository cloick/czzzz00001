Je vois l'erreur ! Le problème est que le layout Dash est défini comme `None` au démarrage. L'erreur dit:

```
dash.exceptions.NoLayoutException: The `layout` was "None" at the time that `run` was called. Make sure to set the layout attribute of your application before running the server.
```

Le problème vient du fait que dans mon code, j'ai défini `app.layout` mais ensuite j'ai essayé d'ajouter des éléments avec `app.layout.children.append()`, ce qui ne marche pas correctement.

Je dois restructurer le code pour que le layout soit complet dès le début, sans essayer de le modifier après coup.

## 🚨 **ERREUR IDENTIFIÉE : Layout Dash non défini correctement**

**Le problème:** Le layout est `None` au démarrage. Je dois corriger la structure.

## 🔧 **CODE CORRIGÉ COMPLET :**

```python
"""
WEBAPP DASH - ANALYSE DES RISQUES DE CHANGEMENT SERVICENOW
100% DONNÉES RÉELLES - Version corrigée layout
"""

import dash
from dash import dcc, html, Input, Output, State
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

app = dash.Dash(__name__)

# ===================================================================
# INITIALISATION DES COMPOSANTS MÉTIER
# ===================================================================

# Initialisation globale
try:
    predictor = ChangeRiskPredictor()
    connector = ServiceNowConnector()
    
    # Variable globale pour stocker les données du changement actuel
    current_change_data = {}
    
    # Vérification du statut
    model_info = predictor.get_model_info()
    connection_status = connector.get_connection_status()
    
    init_success = True
    init_error = ""
    
except Exception as e:
    init_success = False
    init_error = str(e)
    model_info = {"status": "Erreur"}
    connection_status = {"status": "Erreur", "error": str(e)}

# ===================================================================
# FONCTIONS UTILITAIRES
# ===================================================================

def create_status_card(title, status, details=None):
    """Créer une carte de statut"""
    color = "#d4edda" if "✅" in status else "#f8d7da" if "❌" in status else "#fff3cd"
    
    card_content = [
        html.H6(title, style={"margin": "0 0 0.5rem 0"}),
        html.P(status, style={"margin": "0"})
    ]
    
    if details:
        card_content.append(html.Small(details, style={"color": "#666"}))
    
    return html.Div(
        card_content,
        style={
            "background": color,
            "padding": "1rem",
            "border-radius": "8px",
            "margin": "0.5rem 0",
            "border": "1px solid #ddd"
        }
    )

def create_metric_card(title, value, subtitle=None):
    """Créer une carte métrique"""
    card_content = [
        html.H3(str(value), style={"color": "#667eea", "margin": "0"}),
        html.P(title, style={"margin": "0.5rem 0"}),
    ]
    
    if subtitle:
        card_content.append(html.Small(subtitle, style={"color": "#666"}))
    
    return html.Div(
        card_content,
        style={
            "background": "#f8f9fa",
            "padding": "1.5rem",
            "border-radius": "10px",
            "text-align": "center",
            "margin": "0.5rem",
            "border": "1px solid #e9ecef"
        }
    )

# ===================================================================
# LAYOUT PRINCIPAL COMPLET
# ===================================================================

app.layout = html.Div([
    
    # Header principal
    html.Div([
        html.H1("🔍 Change Risk Analyzer", 
                style={"margin": "0", "font-size": "2.5rem"}),
        html.P("Analyseur de risques pour changements ServiceNow • 100% Données Réelles", 
               style={"margin": "0.5rem 0 0 0", "font-size": "1.1rem"})
    ], style={
        "background": "linear-gradient(90deg, #667eea 0%, #764ba2 100%)",
        "padding": "2rem",
        "border-radius": "15px",
        "color": "white",
        "text-align": "center",
        "margin-bottom": "2rem"
    }),
    
    # Container principal
    html.Div([
        
        # Affichage d'erreur d'initialisation si nécessaire
        html.Div(
            id="init-error",
            children=[
                html.Div(f"❌ Erreur d'initialisation: {init_error}", 
                        style={"background": "#f8d7da", "padding": "1rem", "border-radius": "8px", "margin": "1rem 0"})
            ] if not init_success else [],
            style={"display": "block" if not init_success else "none"}
        ),
        
        # Row pour le statut et la saisie
        html.Div([
            
            # Colonne gauche - Statuts
            html.Div([
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
                    "Tables: change_request & incident_filtree" if connection_status.get('status') == 'Connecté' else str(connection_status.get('error', ''))
                ),
                
                # Informations modèle
                html.H5("📊 Performance Modèle", style={"margin-top": "1rem"}),
                html.Div(id="model-performance-info")
                
            ], style={"width": "30%", "display": "inline-block", "vertical-align": "top", "padding": "1rem"}),
            
            # Colonne droite - Interface principale
            html.Div([
                html.H4("📝 Analyse de Changement"),
                
                # Zone de saisie
                html.Div([
                    html.Div([
                        html.Label("Référence du changement:"),
                        dcc.Input(
                            id="change-reference-input",
                            placeholder="CHG0012345",
                            type="text",
                            style={"width": "100%", "padding": "0.5rem", "margin": "0.5rem 0"}
                        ),
                        html.Small("Format: CHG + 7 chiffres", style={"color": "#666"})
                    ], style={"width": "60%", "display": "inline-block", "vertical-align": "top"}),
                    
                    html.Div([
                        html.Button(
                            "🔍 Analyser",
                            id="analyze-button",
                            n_clicks=0,
                            style={
                                "background": "#667eea", 
                                "color": "white", 
                                "border": "none", 
                                "padding": "0.5rem 1rem", 
                                "border-radius": "5px",
                                "margin": "0.5rem"
                            }
                        ),
                        html.Button(
                            "ℹ️ Test",
                            id="test-button",
                            n_clicks=0,
                            style={
                                "background": "#6c757d", 
                                "color": "white", 
                                "border": "none", 
                                "padding": "0.5rem 1rem", 
                                "border-radius": "5px",
                                "margin": "0.5rem"
                            }
                        )
                    ], style={"width": "35%", "display": "inline-block", "vertical-align": "top", "text-align": "center"})
                ], style={"margin": "1rem 0"}),
                
                # Zone de résultats
                html.Hr(),
                html.Div(id="analysis-results")
                
            ], style={"width": "65%", "display": "inline-block", "vertical-align": "top", "padding": "1rem"})
        ], style={"display": "block" if init_success else "none"}),
        
        # Zone pour les résultats détaillés
        html.Div(id="detailed-results", style={"margin-top": "2rem"})
        
    ], style={"max-width": "1200px", "margin": "0 auto", "padding": "1rem"}),
    
    # Footer intégré dans le layout principal
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
])

# ===================================================================
# CALLBACKS
# ===================================================================

@app.callback(
    Output("model-performance-info", "children"),
    Input("analyze-button", "n_clicks")
)
def update_model_info(n_clicks):
    """Afficher les informations de performance du modèle"""
    
    if not init_success:
        return html.Div("❌ Erreur d'initialisation", 
                       style={"background": "#f8d7da", "padding": "1rem", "border-radius": "8px"})
    
    if model_info.get("status") != "Modèle chargé":
        return html.Div("❌ Modèle non disponible", 
                       style={"background": "#f8d7da", "padding": "1rem", "border-radius": "8px"})
    
    training_info = model_info.get('training_info', {})
    perf = training_info.get('performance', {})
    
    if perf:
        return html.Div([
            html.P([
                html.Strong("Recall: "), str(perf.get('recall', 'N/A')), html.Br(),
                html.Strong("Precision: "), str(perf.get('precision', 'N/A')), html.Br(),
                html.Strong("Features: "), str(model_info.get('features', {}).get('count', 'N/A'))
            ])
        ], style={"background": "#d1ecf1", "padding": "1rem", "border-radius": "8px"})
    else:
        return html.Div("Informations de performance non disponibles", 
                       style={"background": "#fff3cd", "padding": "1rem", "border-radius": "8px"})

@app.callback(
    [Output("analysis-results", "children"),
     Output("detailed-results", "children")],
    [Input("analyze-button", "n_clicks"),
     Input("test-button", "n_clicks")],
    [State("change-reference-input", "value")]
)
def perform_analysis(analyze_clicks, test_clicks, change_ref):
    """Effectuer l'analyse du changement"""
    
    if not init_success:
        return html.Div("❌ Système non initialisé", 
                       style={"background": "#f8d7da", "padding": "1rem", "border-radius": "8px"}), ""
    
    # Déterminer quel bouton a été cliqué
    ctx = dash.callback_context
    if not ctx.triggered:
        return "", ""
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Test de connexion
    if button_id == "test-button":
        status = connector.get_connection_status()
        if status.get('status') == 'Connecté':
            return html.Div("✅ Test de connexion réussi", 
                           style={"background": "#d4edda", "padding": "1rem", "border-radius": "8px"}), ""
        else:
            return html.Div(f"❌ Test échoué: {status.get('error', 'Erreur inconnue')}", 
                           style={"background": "#f8d7da", "padding": "1rem", "border-radius": "8px"}), ""
    
    # Analyse du changement
    if button_id == "analyze-button":
        
        if not change_ref:
            return html.Div("⚠️ Veuillez saisir une référence de changement", 
                           style={"background": "#fff3cd", "padding": "1rem", "border-radius": "8px"}), ""
        
        # Validation format
        if not connector.validate_change_reference(change_ref):
            return html.Div("❌ Format invalide. Utilisez CHG + 7 chiffres (ex: CHG0012345)", 
                           style={"background": "#f8d7da", "padding": "1rem", "border-radius": "8px"}), ""
        
        # Récupération des données
        change_data = connector.get_change_data(change_ref)
        
        if not change_data:
            return html.Div(f"❌ Changement {change_ref} non trouvé dans ServiceNow", 
                           style={"background": "#f8d7da", "padding": "1rem", "border-radius": "8px"}), ""
        
        # Analyse ML
        try:
            detailed_analysis = predictor.get_detailed_analysis(change_data)
        except Exception as e:
            return html.Div(f"❌ Erreur analyse ML: {str(e)}", 
                           style={"background": "#f8d7da", "padding": "1rem", "border-radius": "8px"}), ""
        
        # Stocker les données globalement
        global current_change_data
        current_change_data = change_data
        
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
            ], style={
                "background": "white",
                "padding": "2rem",
                "border-radius": "15px",
                "box-shadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
                "border-left": "5px solid #667eea",
                "margin": "1rem 0",
                "text-align": "center"
            })
        ])
        
        # === RÉSULTATS DÉTAILLÉS ===
        detailed_results = html.Div([
            
            html.Div([
                # Colonne gauche - Facteurs et recommandations
                html.Div([
                    html.H4("🚨 Facteurs de risque"),
                    html.Div([
                        html.Ul([
                            html.Li(factor) for factor in detailed_analysis['risk_factors']
                        ]) if detailed_analysis['risk_factors'] else html.P("Aucun facteur spécifique détecté")
                    ], style={"background": "#d1ecf1", "padding": "1rem", "border-radius": "8px"}),
                    
                    html.H4("💡 Recommandations"),
                    html.Div([
                        html.Ul([
                            html.Li(f"✅ {rec}") for rec in detailed_analysis['recommendations']
                        ])
                    ], style={"background": "#d4edda", "padding": "1rem", "border-radius": "8px"})
                    
                ], style={"width": "48%", "display": "inline-block", "vertical-align": "top", "margin": "1%"}),
                
                # Colonne droite - Caractéristiques techniques
                html.Div([
                    html.H4("🔧 Caractéristiques techniques"),
                    html.Div([
                        html.P([
                            html.Strong("Type SILCA: "), str(change_data.get('dv_u_type_change_silca', 'N/A')), html.Br(),
                            html.Strong("Type de changement: "), str(change_data.get('dv_type', 'N/A')), html.Br(),
                            html.Strong("Nombre de CAB: "), str(change_data.get('u_cab_count', 'N/A')), html.Br(),
                            html.Strong("Périmètre BCR: "), '✅' if change_data.get('u_bcr') else '❌', html.Br(),
                            html.Strong("Périmètre BPC: "), '✅' if change_data.get('u_bpc') else '❌'
                        ])
                    ], style={"background": "#f8f9fa", "padding": "1rem", "border-radius": "8px"}),
                    
                    html.H4("📋 Métadonnées"),
                    html.Div([
                        html.P([
                            html.Strong("Équipe: "), str(change_data.get('dv_assignment_group', 'N/A')), html.Br(),
                            html.Strong("CI/Solution: "), str(change_data.get('dv_cmdb_ci', 'N/A')), html.Br(),
                            html.Strong("Catégorie: "), str(change_data.get('dv_category', 'N/A')), html.Br(),
                            html.Strong("État: "), str(change_data.get('dv_state', 'N/A'))
                        ])
                    ], style={"background": "#d1ecf1", "padding": "1rem", "border-radius": "8px"})
                    
                ], style={"width": "48%", "display": "inline-block", "vertical-align": "top", "margin": "1%"})
            ]),
            
            # Informations contextuelles
            html.Hr(),
            html.H3("📈 Informations contextuelles"),
            html.P("🔍 Données extraites des tables ServiceNow réelles", style={"color": "#666", "font-style": "italic"}),
            
            # Onglets simulés avec boutons
            html.Div([
                html.Button("👥 Statistiques équipe", id="btn-team", n_clicks=0, 
                           style={"margin": "0.5rem", "padding": "0.5rem 1rem", "border": "1px solid #ddd", "background": "#f8f9fa", "border-radius": "5px"}),
                html.Button("🛠️ Incidents liés", id="btn-incidents", n_clicks=0,
                           style={"margin": "0.5rem", "padding": "0.5rem 1rem", "border": "1px solid #ddd", "background": "#f8f9fa", "border-radius": "5px"}),
                html.Button("📋 Changements similaires", id="btn-similar", n_clicks=0,
                           style={"margin": "0.5rem", "padding": "0.5rem 1rem", "border": "1px solid #ddd", "background": "#f8f9fa", "border-radius": "5px"})
            ]),
            
            html.Div(id="context-content", style={"padding": "1rem"})
        ])
        
        return main_results, detailed_results
    
    return "", ""

@app.callback(
    Output("context-content", "children"),
    [Input("btn-team", "n_clicks"),
     Input("btn-incidents", "n_clicks"),
     Input("btn-similar", "n_clicks")]
)
def update_context_content(team_clicks, incidents_clicks, similar_clicks):
    """Mettre à jour le contenu des onglets contextuels"""
    
    if not init_success:
        return html.Div("❌ Système non initialisé")
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return html.Div("Cliquez sur un onglet pour voir les informations contextuelles", 
                       style={"color": "#666", "font-style": "italic", "text-align": "center", "padding": "2rem"})
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    try:
        change_data = current_change_data
        if not change_data:
            return html.Div("Aucune donnée de changement disponible")
    except:
        return html.Div("Aucune donnée de changement disponible")
    
    if button_id == "btn-team":
        # Statistiques équipe
        team_stats = connector.get_team_statistics(change_data.get('dv_assignment_group'))
        
        if team_stats and 'error' not in team_stats:
            # Calculer dernier échec
            last_failure = team_stats.get('last_failure_date')
            if last_failure and pd.notna(last_failure):
                days_ago = (datetime.now() - pd.to_datetime(last_failure)).days
                last_failure_text = f"Il y a {days_ago}j"
            else:
                last_failure_text = "Aucun récent"
            
            return html.Div([
                html.Div([
                    create_metric_card("Total changements", team_stats['total_changes'], "6 derniers mois"),
                    create_metric_card("Taux de succès", f"{team_stats['success_rate']}%"),
                    create_metric_card("Échecs", team_stats['failures']),
                    create_metric_card("Dernier échec", last_failure_text)
                ], style={"display": "flex", "justify-content": "space-around", "flex-wrap": "wrap"})
            ])
        else:
            return html.Div("⚠️ Statistiques équipe non disponibles", 
                           style={"background": "#fff3cd", "padding": "1rem", "border-radius": "8px"})
    
    elif button_id == "btn-incidents":
        # Incidents liés
        incidents_data = connector.get_solution_incidents(change_data.get('dv_cmdb_ci'))
        
        if incidents_data:
            resolution_text = f"{incidents_data['avg_resolution_hours']}h" if incidents_data['avg_resolution_hours'] > 0 else "N/A"
            
            return html.Div([
                html.Div([
                    create_metric_card("Total incidents", incidents_data['total_incidents'], "3 derniers mois"),
                    create_metric_card("Incidents critiques", incidents_data['critical_incidents']),
                    create_metric_card("Résolution moyenne", resolution_text)
                ], style={"display": "flex", "justify-content": "space-around", "flex-wrap": "wrap"}),
                
                html.P("🔍 Données extraites de la table incident_filtree", 
                       style={"text-align": "center", "color": "#666", "margin-top": "1rem"})
            ])
        else:
            return html.Div("⚠️ Aucun incident trouvé pour cette solution/CI", 
                           style={"background": "#fff3cd", "padding": "1rem", "border-radius": "8px"})
    
    elif button_id == "btn-similar":
        # Changements similaires
        similar_changes = connector.find_similar_changes(change_data)
        
        if similar_changes:
            similar_divs = []
            for change in similar_changes[:5]:
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
                
                similar_divs.append(html.Div([
                    html.P([
                        html.Strong(f"{icon} {change['number']} - {close_code}"),
                        html.Br(),
                        html.Small(change['short_description'][:100] + "..."),
                        html.Br(),
                        html.Small(f"Similarité: {change['similarity_score']}%{duration_text}")
                    ])
                ], style={
                    "background": bg_color, 
                    "padding": "1rem", 
                    "border-radius": "8px", 
                    "margin": "0.5rem 0"
                }))
            
            return html.Div([
                html.P("📋 Historique réel des changements similaires:"),
                html.Div(similar_divs)
            ])
        else:
            return html.Div("ℹ️ Aucun changement similaire trouvé", 
                           style={"background": "#d1ecf1", "padding": "1rem", "border-radius": "8px"})
    
    return html.Div("Contenu non disponible")
```

## 🔧 **PROBLÈMES CORRIGÉS :**

1. **Layout complet défini d'un coup** - Plus de modification après coup
2. **Footer intégré** dans le layout principal
3. **Gestion d'erreurs d'initialisation** avec try/catch
4. **Layout conditionnel** si erreur d'initialisation
5. **Variables globales** properly définies

**Maintenant le layout devrait être correctement défini dès le démarrage !** 🚀

**Remplacez tout votre code et testez !**

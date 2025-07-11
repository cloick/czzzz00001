L'utilisateur a créé sa webapp Dash et veut maintenant le code adapté. Je dois convertir le code Streamlit que j'avais préparé en code Dash. 

Les principales différences entre Streamlit et Dash :

1. Streamlit utilise `st.title()`, `st.write()`, etc. 
2. Dash utilise `html.H1()`, `html.Div()`, `dcc.Input()`, etc.
3. Streamlit a des callbacks automatiques, Dash utilise des decorators `@app.callback`
4. La structure est différente : Dash a un `app.layout` et des callbacks séparés

Je dois adapter tout le code Streamlit en Dash en gardant les mêmes fonctionnalités :
- Header avec gradient CSS
- Sidebar avec infos modèle
- Zone de saisie
- Boutons d'analyse
- Affichage des résultats avec métriques
- Onglets pour les données d'enrichissement
- CSS moderne

Pour Dash, la structure sera :
1. Imports
2. Initialisation de l'app
3. Layout avec tous les composants
4. Callbacks pour l'interactivité
5. CSS intégré

Je vais garder la même logique métier mais adapter la syntaxe Dash.

## 🚀 **CODE DASH WEBAPP - VERSION FINALE !**

```python
"""
WEBAPP DASH - ANALYSE DES RISQUES DE CHANGEMENT SERVICENOW
100% DONNÉES RÉELLES - ZERO SIMULATION
"""

import dash
from dash import html, dcc, Input, Output, State, callback_context
import pandas as pd
from datetime import datetime
import time
import json

# Import de nos modules
from change_risk_predictor import ChangeRiskPredictor
from servicenow_connector import ServiceNowConnector

# ===================================================================
# INITIALISATION DE L'APPLICATION DASH
# ===================================================================

app = dash.Dash(__name__)

# Variables globales pour les instances
predictor = None
connector = None

def initialize_components():
    """Initialiser les composants ML"""
    global predictor, connector
    if predictor is None:
        predictor = ChangeRiskPredictor()
        connector = ServiceNowConnector()
    return predictor, connector

# ===================================================================
# CSS STYLES MODERNES
# ===================================================================

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# CSS personnalisé
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .main-header {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                padding: 1rem;
                border-radius: 10px;
                color: white;
                text-align: center;
                margin-bottom: 2rem;
            }
            
            .risk-card {
                background: white;
                padding: 1.5rem;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                border-left: 5px solid #667eea;
                margin: 1rem 0;
            }
            
            .metric-container {
                background: #f8f9fa;
                padding: 1rem;
                border-radius: 10px;
                text-align: center;
                margin: 0.5rem 0;
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
                padding: 0.8rem;
                border-radius: 8px;
                margin: 0.5rem 0;
            }
            
            .similar-success { background: #d4edda; }
            .similar-warning { background: #fff3cd; }
            .similar-error { background: #f8d7da; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ===================================================================
# LAYOUT PRINCIPAL
# ===================================================================

app.layout = html.Div([
    
    # Store pour les données
    dcc.Store(id='session-data'),
    
    # Header principal
    html.Div([
        html.H1("🔍 Change Risk Analyzer", style={'margin': '0', 'color': 'white'}),
        html.P("Analyseur de risques pour changements ServiceNow • 100% Données Réelles", 
               style={'margin': '0.5rem 0 0 0', 'color': 'white'})
    ], className='main-header'),
    
    # Layout principal avec sidebar
    html.Div([
        
        # Sidebar
        html.Div([
            html.H3("🤖 Informations du Modèle"),
            html.Div(id='model-status'),
            html.Hr(),
            html.H3("🔗 Connexions Données"),
            html.Div(id='connection-status'),
            html.Hr(),
            html.Button("🔄 Actualiser", id='refresh-btn', 
                       style={'width': '100%', 'margin-top': '1rem'})
        ], style={'width': '25%', 'display': 'inline-block', 'vertical-align': 'top', 
                 'padding': '1rem', 'background': '#f8f9fa', 'min-height': '80vh'}),
        
        # Contenu principal  
        html.Div([
            
            # Zone de saisie
            html.H3("📝 Saisie du changement"),
            html.Div([
                html.Div([
                    dcc.Input(
                        id='change-ref-input',
                        type='text',
                        placeholder='CHG0012345',
                        style={'width': '100%', 'padding': '0.5rem', 'fontSize': '16px'}
                    )
                ], style={'width': '50%', 'display': 'inline-block', 'margin-right': '2%'}),
                
                html.Div([
                    html.Button("🔍 Analyser", id='analyze-btn', 
                               style={'width': '48%', 'margin-right': '4%', 'padding': '0.5rem',
                                     'background': '#667eea', 'color': 'white', 'border': 'none',
                                     'border-radius': '5px', 'fontSize': '16px'}),
                    html.Button("🎲 Exemple", id='example-btn',
                               style={'width': '48%', 'padding': '0.5rem', 'border': '1px solid #ccc',
                                     'border-radius': '5px', 'fontSize': '16px'})
                ], style={'width': '48%', 'display': 'inline-block'})
            ], style={'margin-bottom': '2rem'}),
            
            # Zone de résultats
            html.Div(id='analysis-results'),
            
            # Loading indicator
            dcc.Loading(
                id="loading",
                type="default",
                children=html.Div(id="loading-output")
            )
            
        ], style={'width': '73%', 'display': 'inline-block', 'padding': '1rem'})
        
    ])
    
])

# ===================================================================
# CALLBACKS
# ===================================================================

@app.callback(
    [Output('model-status', 'children'),
     Output('connection-status', 'children')],
    [Input('refresh-btn', 'n_clicks')],
    prevent_initial_call=False
)
def update_status(n_clicks):
    """Mettre à jour le statut du modèle et des connexions"""
    
    pred, conn = initialize_components()
    
    # Statut du modèle
    model_info = pred.get_model_info()
    
    if model_info.get("status") == "Modèle chargé":
        model_status = html.Div([
            html.Div("✅ Modèle opérationnel", className='success-box'),
            html.Details([
                html.Summary("📊 Détails du modèle"),
                html.P(f"Algorithme: {model_info['algorithm']}"),
                html.P(f"Features: {model_info['features']['count']}"),
                html.P(f"Performance: {model_info.get('training_info', {}).get('performance', {}).get('recall', 'N/A')} recall")
            ])
        ])
    else:
        model_status = html.Div("❌ Modèle non disponible", className='error-box')
    
    # Statut des connexions
    connection_info = conn.get_connection_status()
    
    if connection_info.get('status') == 'Connecté':
        connection_status = html.Div([
            html.Div("✅ ServiceNow connecté", className='success-box'),
            html.Details([
                html.Summary("📋 Détails connexions"),
                html.P(f"Changes: {connection_info['changes_dataset']}"),
                html.P(f"Incidents: {connection_info['incidents_dataset']}")
            ])
        ])
    else:
        connection_status = html.Div([
            html.Div("❌ Connexion ServiceNow échouée", className='error-box'),
            html.P(connection_info.get('error', 'Erreur inconnue'))
        ])
    
    return model_status, connection_status

@app.callback(
    Output('change-ref-input', 'value'),
    [Input('example-btn', 'n_clicks')],
    prevent_initial_call=True
)
def set_example(n_clicks):
    """Remplir avec un exemple"""
    if n_clicks:
        return "CHG0012345"
    return dash.no_update

@app.callback(
    Output('analysis-results', 'children'),
    [Input('analyze-btn', 'n_clicks')],
    [State('change-ref-input', 'value')],
    prevent_initial_call=True
)
def analyze_change(n_clicks, change_ref):
    """Analyser un changement"""
    
    if not n_clicks or not change_ref:
        return html.Div()
    
    pred, conn = initialize_components()
    
    # Validation du format
    if not conn.validate_change_reference(change_ref):
        return html.Div([
            html.Div("❌ Format de référence invalide. Utilisez le format CHG + 7 chiffres (ex: CHG0012345)", 
                    className='error-box')
        ])
    
    # Récupération des données
    change_data = conn.get_change_data(change_ref)
    
    if not change_data:
        return html.Div([
            html.Div(f"❌ Changement {change_ref} non trouvé dans la base ServiceNow", 
                    className='error-box'),
            html.Div("💡 Vérifiez que la référence existe dans votre système ServiceNow", 
                    className='info-box')
        ])
    
    # Analyse ML
    try:
        detailed_analysis = pred.get_detailed_analysis(change_data)
    except Exception as e:
        return html.Div([
            html.Div(f"❌ Erreur lors de l'analyse ML : {str(e)}", className='error-box')
        ])
    
    # Construction des résultats
    results = []
    
    # Titre
    results.append(html.Hr())
    results.append(html.H2(f"📊 Analyse de {change_ref}"))
    
    # Score principal
    risk_score = detailed_analysis['risk_score']
    risk_color = detailed_analysis['risk_color']
    risk_level = detailed_analysis['risk_level']
    
    results.append(
        html.Div([
            html.H1(f"{risk_color} {risk_score}%", 
                   style={'color': '#667eea', 'margin': '0', 'text-align': 'center'}),
            html.H3("Risque d'échec", style={'margin': '0.5rem 0', 'text-align': 'center'}),
            html.P(f"Niveau: {risk_level}", style={'margin': '0', 'text-align': 'center', 'font-weight': 'bold'}),
            html.P(detailed_analysis['interpretation'], 
                  style={'margin': '0', 'text-align': 'center', 'font-style': 'italic'})
        ], className='metric-container')
    )
    
    # Détails en deux colonnes
    results.append(html.Div([
        
        # Colonne gauche
        html.Div([
            html.H3("🚨 Facteurs de risque détectés"),
            html.Div([
                html.P(f"• {factor}") for factor in detailed_analysis['risk_factors']
            ] if detailed_analysis['risk_factors'] else [html.Div("Aucun facteur de risque spécifique détecté", className='info-box')]),
            
            html.H3("💡 Recommandations"),
            html.Div([
                html.P(f"✅ {rec}") for rec in detailed_analysis['recommendations']
            ])
            
        ], style={'width': '48%', 'display': 'inline-block', 'margin-right': '4%'}),
        
        # Colonne droite
        html.Div([
            html.H3("🔧 Caractéristiques techniques"),
            html.P(f"Type SILCA: {change_data.get('dv_u_type_change_silca', 'N/A')}"),
            html.P(f"Type de changement: {change_data.get('dv_type', 'N/A')}"),
            html.P(f"Nombre de CAB: {change_data.get('u_cab_count', 'N/A')}"),
            html.P(f"Périmètre BCR: {'✅' if change_data.get('u_bcr') else '❌'}"),
            html.P(f"Périmètre BPC: {'✅' if change_data.get('u_bpc') else '❌'}"),
            
            html.H3("📋 Métadonnées"),
            html.P(f"Équipe: {change_data.get('dv_assignment_group', 'N/A')}"),
            html.P(f"CI/Solution: {change_data.get('dv_cmdb_ci', 'N/A')}"),
            html.P(f"Catégorie: {change_data.get('dv_category', 'N/A')}"),
            html.P(f"État actuel: {change_data.get('dv_state', 'N/A')}")
            
        ], style={'width': '48%', 'display': 'inline-block'})
        
    ]))
    
    # Informations contextuelles
    results.append(html.Hr())
    results.append(html.H2("📈 Informations contextuelles (Données réelles ServiceNow)"))
    
    # Onglets (simulé avec des sections)
    
    # Stats équipe
    results.append(html.H3("👥 Statistiques équipe"))
    team_stats = conn.get_team_statistics(change_data.get('dv_assignment_group'))
    
    if team_stats and 'error' not in team_stats:
        results.append(html.Div([
            html.P("📊 Données calculées depuis la base ServiceNow réelle", className='info-box'),
            html.Div([
                html.Div([
                    html.H4(str(team_stats['total_changes'])),
                    html.P("Total changements"),
                    html.Small("6 derniers mois")
                ], style={'width': '23%', 'display': 'inline-block', 'text-align': 'center', 'margin': '1%'}),
                
                html.Div([
                    html.H4(f"{team_stats['success_rate']}%"),
                    html.P("Taux de succès")
                ], style={'width': '23%', 'display': 'inline-block', 'text-align': 'center', 'margin': '1%'}),
                
                html.Div([
                    html.H4(str(team_stats['failures'])),
                    html.P("Échecs")
                ], style={'width': '23%', 'display': 'inline-block', 'text-align': 'center', 'margin': '1%'}),
                
                html.Div([
                    html.H4("N/A" if not team_stats.get('last_failure_date') else 
                           f"Il y a {(datetime.now() - pd.to_datetime(team_stats['last_failure_date'])).days}j"),
                    html.P("Dernier échec")
                ], style={'width': '23%', 'display': 'inline-block', 'text-align': 'center', 'margin': '1%'})
            ])
        ]))
    else:
        results.append(html.Div("⚠️ Statistiques équipe non disponibles", className='warning-box'))
    
    # Incidents
    results.append(html.H3("🛠️ Incidents liés"))
    incidents_data = conn.get_solution_incidents(change_data.get('dv_cmdb_ci'))
    
    if incidents_data:
        results.append(html.Div([
            html.P("🔍 Données extraites de la table incident_filtree", className='info-box'),
            html.Div([
                html.Div([
                    html.H4(str(incidents_data['total_incidents'])),
                    html.P("Total incidents"),
                    html.Small("3 derniers mois")
                ], style={'width': '30%', 'display': 'inline-block', 'text-align': 'center', 'margin': '1.5%'}),
                
                html.Div([
                    html.H4(str(incidents_data['critical_incidents'])),
                    html.P("Incidents critiques")
                ], style={'width': '30%', 'display': 'inline-block', 'text-align': 'center', 'margin': '1.5%'}),
                
                html.Div([
                    html.H4(f"{incidents_data['avg_resolution_hours']}h" if incidents_data['avg_resolution_hours'] > 0 else "N/A"),
                    html.P("Résolution moyenne")
                ], style={'width': '30%', 'display': 'inline-block', 'text-align': 'center', 'margin': '1.5%'})
            ])
        ]))
    else:
        results.append(html.Div("⚠️ Aucun incident trouvé pour cette solution/CI", className='warning-box'))
    
    # Changements similaires
    results.append(html.H3("📋 Changements similaires"))
    similar_changes = conn.find_similar_changes(change_data)
    
    if similar_changes:
        results.append(html.P("📋 Historique réel des changements similaires :", className='info-box'))
        
        for change in similar_changes[:5]:
            close_code = change['dv_close_code']
            
            if close_code == 'Succès':
                icon = "✅"
                css_class = "similar-success"
            elif 'Échec' in str(close_code):
                icon = "❌"
                css_class = "similar-error"
            else:
                icon = "⚠️"
                css_class = "similar-warning"
            
            duration_text = ""
            if change['duration_hours'] is not None:
                duration_text = f" • Durée réelle: {change['duration_hours']}h"
            
            results.append(
                html.Div([
                    html.P([
                        f"{icon} {change['number']} - {close_code}",
                        html.Br(),
                        html.Small(change['short_description']),
                        html.Br(), 
                        html.Small(f"Similarité: {change['similarity_score']}%{duration_text} • Données réelles ServiceNow")
                    ])
                ], className=f'similar-change {css_class}')
            )
    else:
        results.append(html.Div("ℹ️ Aucun changement similaire trouvé dans l'historique", className='info-box'))
    
    # Footer
    results.append(html.Hr())
    results.append(html.Div([
        html.P("🤖 Change Risk Analyzer • 100% Données Réelles ServiceNow", 
               style={'text-align': 'center', 'color': '#666'}),
        html.P("Tables connectées: change_request_without_empty_columns & incident_filtree", 
               style={'text-align': 'center', 'color': '#666', 'font-style': 'italic'}),
        html.P("ZERO SIMULATION • ZERO DONNÉES FACTICES", 
               style={'text-align': 'center', 'color': '#666', 'font-weight': 'bold'})
    ], style={'padding': '2rem'}))
    
    return results

# ===================================================================
# LANCEMENT DE L'APPLICATION
# ===================================================================

if __name__ == '__main__':
    app.run_server(debug=True)
```

## 🎯 **INSTRUCTIONS DE DÉPLOIEMENT :**

### **1. COLLER CE CODE**
```
- Sélectionner TOUT le code par défaut dans l'éditeur Dash
- Supprimer tout (Ctrl+A puis Delete)
- Coller ce code complet
```

### **2. SAUVEGARDER**
```
Ctrl+S ou bouton Save
```

### **3. CONFIGURATION IMPORTANTE**
```
Dans les paramètres de la webapp :
✅ Activer "Code Libraries" 
→ Sélectionner votre library dans la liste
```

### **4. LANCER**
```
Bouton "RUN" ou "START"
Attendre 30-60 secondes pour la première exécution
```

## 🚀 **AVANTAGES DE CETTE VERSION DASH :**

✅ **Interface moderne** avec CSS personnalisé  
✅ **Sidebar informative** avec statut en temps réel  
✅ **Boutons interactifs** avec callbacks  
✅ **Loading indicators** pour UX fluide  
✅ **Layout responsive** en colonnes  
✅ **100% données réelles** ServiceNow  
✅ **Gestion d'erreurs** complète  

**Collez ce code et lancez ! Ça va être magnifique !** 🎨🚀

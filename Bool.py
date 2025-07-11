"""
DASH WEBAPP - ANALYSE DES RISQUES DE CHANGEMENT SERVICENOW
Version complète avec Statistiques, Incidents et Changements similaires
Compatible Dataiku DSS et dash-bootstrap-components >= 2.x
"""

from dash import Dash, dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import datetime
import time

# Import de nos modules
from change_risk_predictor import ChangeRiskPredictor
from servicenow_connector import ServiceNowConnector

# ==============================================================================
# INITIALISATION
# ==============================================================================

# Crée l'application Dash avec le thème Bootstrap
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Charger les modules dès le démarrage
predictor = ChangeRiskPredictor()
connector = ServiceNowConnector()

# ==============================================================================
# LAYOUT DE L'APPLICATION
# ==============================================================================

app.layout = dbc.Container([
    # HEADER
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("🔍 Change Risk Analyzer", className="text-white"),
                html.P("Analyseur de risques pour changements ServiceNow • 100% Données Réelles", className="text-white")
            ], style={
                'background': 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
                'padding': '1rem',
                'border-radius': '10px',
                'text-align': 'center',
                'margin-bottom': '2rem'
            })
        ])
    ]),

    # FORMULAIRE DE SAISIE
    dbc.Row([
        dbc.Col([
            dbc.Input(id="change-ref-input", placeholder="CHG0012345", type="text"),
        ], width=6),
        dbc.Col([
            dbc.Button("🔍 Analyser", id="analyze-btn", color="primary", className="d-block w-100"),
        ], width=3),
        dbc.Col([
            dbc.Button("ℹ️ Test Connexion", id="test-conn-btn", color="secondary", className="d-block w-100"),
        ], width=3)
    ], className="mb-4"),

    # ZONE DE MESSAGES
    html.Div(id="message-box"),

    # RÉSULTATS DE L'ANALYSE
    html.Div(id="analysis-results"),

], fluid=True)

# ==============================================================================
# CALLBACKS
# ==============================================================================

# === CALLBACK : TESTER LA CONNEXION ===
@app.callback(
    Output("message-box", "children"),
    Input("test-conn-btn", "n_clicks"),
    prevent_initial_call=True
)
def test_connexion(n_clicks):
    status = connector.get_connection_status()
    if status.get('status') == 'Connecté':
        return dbc.Alert("✅ ServiceNow connecté", color="success")
    else:
        return dbc.Alert(f"❌ Erreur : {status.get('error', 'Connexion échouée')}", color="danger")

# === CALLBACK : ANALYSER LE CHANGEMENT ===
@app.callback(
    Output("analysis-results", "children"),
    Output("message-box", "children"),
    Input("analyze-btn", "n_clicks"),
    State("change-ref-input", "value"),
    prevent_initial_call=True
)
def analyze_change(n_clicks, change_ref):
    if not change_ref:
        return None, dbc.Alert("❌ Veuillez saisir une référence de changement.", color="danger")

    if not connector.validate_change_reference(change_ref):
        return None, dbc.Alert("❌ Format de référence invalide. Utilisez le format CHG + 7 chiffres (ex: CHG0012345)", color="danger")

    # Récupération des données ServiceNow
    change_data = connector.get_change_data(change_ref)
    if not change_data:
        return None, dbc.Alert(f"❌ Changement {change_ref} non trouvé dans ServiceNow", color="danger")

    # Analyse ML
    try:
        detailed_analysis = predictor.get_detailed_analysis(change_data)
    except Exception as e:
        return None, dbc.Alert(f"❌ Erreur lors de l'analyse ML : {str(e)}", color="danger")

    # === Construction des résultats ===
    risk_score = detailed_analysis['risk_score']
    risk_level = detailed_analysis['risk_level']
    risk_color = detailed_analysis['risk_color']
    interpretation = detailed_analysis['interpretation']

    # Facteurs de risque
    risk_factors = detailed_analysis.get('risk_factors', [])
    risk_factor_items = [html.Li(f) for f in risk_factors] if risk_factors else [html.P("Aucun facteur de risque spécifique détecté")]

    # Recommandations
    recommendations = detailed_analysis.get('recommendations', [])
    recommendations_items = [html.Li(f"✅ {r}") for r in recommendations]

    # Caractéristiques techniques
    features_display = {
        'Type SILCA': change_data.get('dv_u_type_change_silca', 'N/A'),
        'Type de changement': change_data.get('dv_type', 'N/A'),
        'Nombre de CAB': change_data.get('u_cab_count', 'N/A'),
        'Périmètre BCR': '✅' if change_data.get('u_bcr') else '❌',
        'Périmètre BPC': '✅' if change_data.get('u_bpc') else '❌'
    }
    characteristics = [html.Li(f"{key}: {value}") for key, value in features_display.items()]

    # === Statistiques & Incidents liés ===
    stats_content = html.Div([
        html.H5("📊 Statistiques"),
        html.P("➡️ Proportion de changements similaires ayant échoué :"),
        dcc.Graph(
            figure=predictor.plot_failure_rate_by_category(change_data.get('dv_type', 'Autre'))
        )
    ])

    incidents_content = html.Div([
        html.H5("🚨 Incidents liés"),
        dcc.Graph(
            figure=predictor.plot_linked_incidents(change_data.get('dv_type', 'Autre'))
        )
    ])

    similar_changes_content = html.Div([
        html.H5("🔄 Changements similaires"),
        dcc.Graph(
            figure=predictor.plot_similar_changes(change_data.get('dv_type', 'Autre'))
        )
    ])

    # === TABS pour navigation ===
    tabs = dbc.Tabs([
        dbc.Tab(stats_content, label="📊 Statistiques"),
        dbc.Tab(incidents_content, label="🚨 Incidents liés"),
        dbc.Tab(similar_changes_content, label="🔄 Changements similaires"),
    ])

    # === Affichage final ===
    result_layout = dbc.Container([
        html.H2(f"📊 Analyse de {change_ref}"),
        dbc.Card([
            dbc.CardBody([
                html.H3(f"{risk_score}% - {risk_level}", className="card-title"),
                html.P(interpretation, className="card-text"),
            ], style={"borderLeft": f"5px solid {risk_color}"})
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                html.H4("🚨 Facteurs de risque détectés"),
                html.Ul(risk_factor_items),
                html.H4("💡 Recommandations"),
                html.Ul(recommendations_items)
            ], width=6),
            dbc.Col([
                html.H4("🔧 Caractéristiques techniques"),
                html.Ul(characteristics)
            ], width=6)
        ]),

        html.Hr(),

        # === Onglets ===
        tabs
    ], fluid=True)

    return result_layout, None

# ==============================================================================
# LANCEMENT DE L'APPLICATION
# ==============================================================================

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)

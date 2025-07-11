import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import datetime

# === Import de nos modules ===
from change_risk_predictor import ChangeRiskPredictor
from servicenow_connector import ServiceNowConnector

# === Initialisation du modèle et du connecteur ===
predictor = ChangeRiskPredictor()
connector = ServiceNowConnector()

# === App Dash avec Bootstrap ===
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

# === Layout principal ===
app.layout = dbc.Container([
    # === HEADER ===
    dbc.Row([
        dbc.Col(html.Div([
            html.H1("🔍 Change Risk Analyzer", className="text-white"),
            html.P("Analyseur de risques pour changements ServiceNow • 100% Données Réelles", className="text-white"),
        ], className="p-3 mb-4 rounded", style={"background": "linear-gradient(90deg, #667eea 0%, #764ba2 100%)"}))
    ]),

    # === SIDEBAR & MAIN ===
    dbc.Row([
        # === SIDEBAR ===
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🤖 Informations du Modèle"),
                dbc.CardBody([
                    html.Div(id="model-status"),
                    dbc.Button("ℹ️ Test Connexion", id="test-conn-btn", color="info", block=True, className="mt-3"),
                    html.Div(id="connection-status", className="mt-2")
                ])
            ], className="mb-4")
        ], width=3),

        # === MAIN CONTENT ===
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📝 Saisie du changement"),
                dbc.CardBody([
                    dbc.Input(id="change-ref", placeholder="CHG0012345", type="text", debounce=True),
                    dbc.Button("🔍 Analyser", id="analyze-btn", color="primary", block=True, className="mt-2"),
                    html.Div(id="analysis-results", className="mt-4"),

                    # === TABS POUR LES AUTRES SECTIONS ===
                    dbc.Tabs(id="details-tabs", active_tab="stats-tab", className="mt-4", children=[
                        dbc.Tab(label="📈 Statistiques équipe", tab_id="stats-tab"),
                        dbc.Tab(label="📋 Incidents liés", tab_id="incidents-tab"),
                        dbc.Tab(label="📂 Changements similaires", tab_id="changes-tab"),
                    ]),
                    html.Div(id="tab-content", className="p-3 border")
                ])
            ])
        ], width=9)
    ]),

    # === FOOTER ===
    dbc.Row([
        dbc.Col(html.Div([
            html.Hr(),
            html.P("🤖 Change Risk Analyzer • 100% Données Réelles ServiceNow", className="text-center text-muted"),
            html.P("Tables connectées: change_request_without_empty_columns & incident_filtree", className="text-center text-muted"),
            html.P("ZERO SIMULATION • ZERO DONNÉES FACTICES", className="text-center text-muted")
        ]))
    ])
], fluid=True)

# === CALLBACKS ===

# 📡 Charger le statut du modèle
@app.callback(
    Output("model-status", "children"),
    Input("analyze-btn", "n_clicks"),
    prevent_initial_call=True
)
def update_model_info(_):
    model_info = predictor.get_model_info()
    if model_info.get("status") == "Modèle chargé":
        perf = model_info.get("training_info", {}).get("performance", {})
        return html.Div([
            dbc.Alert("✅ Modèle opérationnel", color="success"),
            html.P(f"**Algorithme:** {model_info['algorithm']}"),
            html.P(f"**Features:** {model_info['features']['count']}"),
            html.P(f"**Recall:** {perf.get('recall', 'N/A')} • **Precision:** {perf.get('precision', 'N/A')}")
        ])
    else:
        return dbc.Alert("❌ Modèle non disponible", color="danger")

# 📡 Tester la connexion ServiceNow
@app.callback(
    Output("connection-status", "children"),
    Input("test-conn-btn", "n_clicks"),
    prevent_initial_call=True
)
def test_connection(_):
    status = connector.get_connection_status()
    if status.get("status") == "Connecté":
        return dbc.Alert("✅ ServiceNow connecté", color="success")
    else:
        return dbc.Alert(f"❌ Connexion ServiceNow échouée : {status.get('error')}", color="danger")

# 📊 Analyse d’un changement + remplissage des tabs
@app.callback(
    [Output("analysis-results", "children"),
     Output("tab-content", "children")],
    [Input("analyze-btn", "n_clicks"),
     State("change-ref", "value"),
     State("details-tabs", "active_tab")],
    prevent_initial_call=True
)
def analyze_change(_, change_ref, active_tab):
    if not change_ref:
        return dbc.Alert("❌ Veuillez saisir une référence de changement", color="danger"), ""

    if not connector.validate_change_reference(change_ref):
        return dbc.Alert("❌ Référence invalide. Format attendu: CHG0012345", color="danger"), ""

    # Récupération des données réelles
    change_data = connector.get_change_data(change_ref)
    if not change_data:
        return dbc.Alert(f"❌ Changement {change_ref} non trouvé dans ServiceNow", color="warning"), ""

    # Analyse ML
    try:
        detailed_analysis = predictor.get_detailed_analysis(change_data)
    except Exception as e:
        return dbc.Alert(f"❌ Erreur lors de l'analyse ML : {str(e)}", color="danger"), ""

    # Résultats principaux
    risk_score = detailed_analysis['risk_score']
    risk_level = detailed_analysis['risk_level']
    interpretation = detailed_analysis['interpretation']

    analysis_card = html.Div([
        dbc.Card([
            dbc.CardHeader(f"📊 Analyse de {change_ref}"),
            dbc.CardBody([
                html.H2(f"{risk_score}% - {risk_level}", className="text-primary"),
                html.P(interpretation),

                html.H4("🚨 Facteurs de risque"),
                html.Ul([html.Li(f) for f in detailed_analysis['risk_factors']]),

                html.H4("💡 Recommandations"),
                html.Ul([html.Li(rec) for rec in detailed_analysis['recommendations']])
            ])
        ])
    ])

    # === Contenu du Tab sélectionné ===
    if active_tab == "stats-tab":
        stats = detailed_analysis.get("team_statistics", {})
        tab_content = html.Div([
            html.H5("📈 Statistiques équipe"),
            html.P(f"Nombre total de changements: {stats.get('total_changes')}"),
            html.P(f"Taux de succès: {stats.get('success_rate')}%"),
            html.P(f"Taux d’échec: {stats.get('failure_rate')}%")
        ])
    elif active_tab == "incidents-tab":
        incidents = detailed_analysis.get("related_incidents", [])
        tab_content = html.Div([
            html.H5("📋 Incidents liés"),
            dbc.Table.from_dataframe(pd.DataFrame(incidents), striped=True, bordered=True, hover=True)
        ]) if incidents else html.P("Aucun incident lié trouvé.")
    elif active_tab == "changes-tab":
        changes = detailed_analysis.get("similar_changes", [])
        tab_content = html.Div([
            html.H5("📂 Changements similaires"),
            dbc.Table.from_dataframe(pd.DataFrame(changes), striped=True, bordered=True, hover=True)
        ]) if changes else html.P("Aucun changement similaire trouvé.")
    else:
        tab_content = html.P("Sélectionnez un onglet pour voir les détails.")

    return analysis_card, tab_content

# === Callback pour changer de Tab ===
@app.callback(
    Output("tab-content", "children"),
    [Input("details-tabs", "active_tab"),
     State("change-ref", "value")],
    prevent_initial_call=True
)
def switch_tab(active_tab, change_ref):
    if not change_ref:
        return html.P("Analysez un changement pour voir les détails.")
    return analyze_change(None, change_ref, active_tab)[1]

# === Exécution ===
app.title = "Change Risk Analyzer"
server = app.server

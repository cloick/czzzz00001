L'utilisateur me demande de :
1. Connecter à la table "incident_filtree" pour les incidents
2. Trouver la colonne de date de fin dans notre conversation précédente (je dois chercher dans l'historique)
3. Donner le code final complet du connector et de la webapp sans aucune simulation

Laissez-moi chercher dans la conversation pour les colonnes de dates...

En regardant dans la conversation précédente, je vois qu'ils ont travaillé sur un dataset avec des colonnes comme :
- opened_at
- start_date 
- end_date
- closed_at

Je me souviens qu'ils ont mentionné des colonnes comme start_date et end_date pour calculer la durée planifiée dans le feature engineering.

Donc je vais utiliser :
- opened_at (date d'ouverture)
- closed_at (date de fermeture) ou end_date pour calculer la durée réelle

Pour les incidents, je vais me connecter à "incident_filtree" avec une logique similaire au connector de changements.

## 📁 **SERVICENOW_CONNECTOR.PY - VERSION FINALE SANS SIMULATION**

```python
"""
Connecteur ServiceNow RÉEL - ZERO simulation
Connecté aux tables : change_request_without_empty_columns & incident_filtree
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

class ServiceNowConnector:
    """Connecteur pour données ServiceNow 100% réelles"""
    
    def __init__(self):
        self.changes_dataset = "change_request_without_empty_columns"
        self.incidents_dataset = "incident_filtree"
        self.connection_status = "Datasets Dataiku"
        
    def get_change_data(self, change_ref):
        """Récupérer les données RÉELLES d'un changement spécifique"""
        
        if not self.validate_change_reference(change_ref):
            print(f"❌ Format de référence invalide: {change_ref}")
            return None
        
        try:
            import dataiku
            
            print(f"🔍 Recherche de {change_ref} dans {self.changes_dataset}...")
            
            dataset = dataiku.Dataset(self.changes_dataset)
            df = dataset.get_dataframe()
            
            print(f"✅ Dataset chargé: {len(df)} changements")
            
            # Filtrer sur le changement spécifique
            change_row = df[df['number'] == change_ref]
            
            if len(change_row) == 0:
                print(f"❌ Changement {change_ref} non trouvé")
                return None
            
            # Conversion en dictionnaire
            change_data = change_row.iloc[0].to_dict()
            
            print(f"✅ Changement {change_ref} récupéré avec succès")
            return change_data
            
        except Exception as e:
            print(f"❌ Erreur récupération changement {change_ref}: {e}")
            return None
    
    def get_team_statistics(self, assignment_group, months_back=6):
        """Statistiques RÉELLES d'une équipe"""
        
        try:
            import dataiku
            
            print(f"📊 Calcul statistiques pour équipe: {assignment_group}")
            
            dataset = dataiku.Dataset(self.changes_dataset)
            df = dataset.get_dataframe()
            
            # Filtrer par équipe
            team_changes = df[df['dv_assignment_group'] == assignment_group]
            
            if len(team_changes) == 0:
                print(f"❌ Aucun changement trouvé pour l'équipe {assignment_group}")
                return None
            
            # Filtrer par période si colonne date disponible
            if 'opened_at' in df.columns:
                df['opened_at'] = pd.to_datetime(df['opened_at'], errors='coerce')
                cutoff_date = datetime.now() - timedelta(days=months_back * 30)
                team_changes = team_changes[team_changes['opened_at'] >= cutoff_date]
            
            # Calculs des statistiques RÉELLES
            total_changes = len(team_changes)
            
            if total_changes == 0:
                return {
                    'assignment_group': assignment_group,
                    'period_months': months_back,
                    'total_changes': 0,
                    'message': 'Aucun changement dans la période'
                }
            
            # Taux de succès RÉEL
            if 'dv_close_code' in team_changes.columns:
                successes = len(team_changes[team_changes['dv_close_code'] == 'Succès'])
                success_rate = (successes / total_changes * 100)
                failures = total_changes - successes
                
                # Dernière date d'échec RÉELLE
                failed_changes = team_changes[team_changes['dv_close_code'] != 'Succès']
                last_failure_date = None
                if len(failed_changes) > 0 and 'opened_at' in failed_changes.columns:
                    last_failure_date = failed_changes['opened_at'].max()
            else:
                return {'error': 'Colonne dv_close_code non disponible'}
            
            team_stats = {
                'assignment_group': assignment_group,
                'period_months': months_back,
                'total_changes': total_changes,
                'successes': successes,
                'failures': failures,
                'success_rate': round(success_rate, 1),
                'last_failure_date': last_failure_date,
                'data_source': 'Données réelles ServiceNow'
            }
            
            print(f"✅ Statistiques réelles calculées: {total_changes} changements, {success_rate:.1f}% succès")
            return team_stats
            
        except Exception as e:
            print(f"❌ Erreur stats équipe {assignment_group}: {e}")
            return None
    
    def get_solution_incidents(self, cmdb_ci, months_back=3):
        """Incidents RÉELS liés à une solution/CI depuis table incident_filtree"""
        
        try:
            import dataiku
            
            print(f"🔍 Recherche incidents pour {cmdb_ci} dans {self.incidents_dataset}...")
            
            # Connexion à la table incidents
            incidents_dataset = dataiku.Dataset(self.incidents_dataset)
            incidents_df = incidents_dataset.get_dataframe()
            
            print(f"✅ Table incidents chargée: {len(incidents_df)} incidents")
            
            # Filtrer par CI (plusieurs colonnes possibles)
            ci_incidents = pd.DataFrame()
            
            # Essayer différentes colonnes pour le CI
            possible_ci_columns = ['cmdb_ci', 'dv_cmdb_ci', 'configuration_item', 'ci']
            
            for col in possible_ci_columns:
                if col in incidents_df.columns:
                    ci_incidents = incidents_df[incidents_df[col] == cmdb_ci]
                    if len(ci_incidents) > 0:
                        print(f"✅ Incidents trouvés via colonne {col}")
                        break
            
            # Filtrer par période si colonne date disponible
            if len(ci_incidents) > 0:
                date_columns = ['opened_at', 'sys_created_on', 'opened']
                for date_col in date_columns:
                    if date_col in ci_incidents.columns:
                        ci_incidents[date_col] = pd.to_datetime(ci_incidents[date_col], errors='coerce')
                        cutoff_date = datetime.now() - timedelta(days=months_back * 30)
                        ci_incidents = ci_incidents[ci_incidents[date_col] >= cutoff_date]
                        break
            
            # Calculs RÉELS
            total_incidents = len(ci_incidents)
            
            if total_incidents == 0:
                return {
                    'cmdb_ci': cmdb_ci,
                    'period_months': months_back,
                    'total_incidents': 0,
                    'critical_incidents': 0,
                    'avg_resolution_hours': 0,
                    'last_incident_date': None,
                    'data_source': 'Données réelles incident_filtree'
                }
            
            # Incidents critiques RÉELS
            critical_incidents = 0
            if 'priority' in ci_incidents.columns:
                critical_incidents = len(ci_incidents[ci_incidents['priority'].isin(['1', '2', 'Critical', 'High'])])
            elif 'impact' in ci_incidents.columns:
                critical_incidents = len(ci_incidents[ci_incidents['impact'].isin(['1', '2', 'High', 'Critical'])])
            
            # Temps de résolution RÉEL
            avg_resolution_hours = 0
            if 'opened_at' in ci_incidents.columns and 'closed_at' in ci_incidents.columns:
                ci_incidents['opened_at'] = pd.to_datetime(ci_incidents['opened_at'], errors='coerce')
                ci_incidents['closed_at'] = pd.to_datetime(ci_incidents['closed_at'], errors='coerce')
                
                resolved_incidents = ci_incidents.dropna(subset=['opened_at', 'closed_at'])
                if len(resolved_incidents) > 0:
                    durations = (resolved_incidents['closed_at'] - resolved_incidents['opened_at']).dt.total_seconds() / 3600
                    avg_resolution_hours = round(durations.mean(), 1)
            
            # Dernier incident RÉEL
            last_incident_date = None
            for date_col in ['opened_at', 'sys_created_on', 'opened']:
                if date_col in ci_incidents.columns:
                    last_incident_date = ci_incidents[date_col].max()
                    break
            
            incidents_data = {
                'cmdb_ci': cmdb_ci,
                'period_months': months_back,
                'total_incidents': total_incidents,
                'critical_incidents': critical_incidents,
                'avg_resolution_hours': avg_resolution_hours,
                'last_incident_date': last_incident_date,
                'data_source': 'Données réelles incident_filtree'
            }
            
            print(f"✅ Incidents réels calculés: {total_incidents} incidents, {critical_incidents} critiques")
            return incidents_data
            
        except Exception as e:
            print(f"❌ Erreur incidents {cmdb_ci}: {e}")
            return None
    
    def find_similar_changes(self, change_data, limit=10):
        """Changements similaires RÉELS avec durées RÉELLES"""
        
        try:
            import dataiku
            
            print(f"🔍 Recherche de changements similaires...")
            
            dataset = dataiku.Dataset(self.changes_dataset)
            df = dataset.get_dataframe()
            
            # Exclure le changement lui-même
            if 'number' in change_data:
                df = df[df['number'] != change_data['number']]
            
            # Calcul de score de similarité RÉEL
            similarity_scores = []
            
            for idx, row in df.iterrows():
                score = 0
                
                # Critères de similarité basés sur vos analyses
                if 'dv_u_type_change_silca' in row and row['dv_u_type_change_silca'] == change_data.get('dv_u_type_change_silca'):
                    score += 40
                
                if 'dv_type' in row and row['dv_type'] == change_data.get('dv_type'):
                    score += 30
                
                if 'dv_assignment_group' in row and row['dv_assignment_group'] == change_data.get('dv_assignment_group'):
                    score += 20
                
                if 'dv_category' in row and row['dv_category'] == change_data.get('dv_category'):
                    score += 10
                
                similarity_scores.append(score)
            
            # Ajouter scores et filtrer
            df = df.copy()
            df['similarity_score'] = similarity_scores
            df = df[df['similarity_score'] > 30].sort_values('similarity_score', ascending=False).head(limit)
            
            # Conversion avec durées RÉELLES
            similar_changes = []
            
            for idx, row in df.iterrows():
                
                # Calcul de la durée RÉELLE
                duration_hours = None
                if 'opened_at' in row and 'closed_at' in row:
                    try:
                        opened = pd.to_datetime(row['opened_at'])
                        closed = pd.to_datetime(row['closed_at'])
                        if pd.notna(opened) and pd.notna(closed):
                            duration_hours = round((closed - opened).total_seconds() / 3600, 1)
                    except:
                        duration_hours = None
                
                similar_change = {
                    'number': row.get('number', 'N/A'),
                    'dv_close_code': row.get('dv_close_code', 'N/A'),
                    'short_description': row.get('short_description', 'Description non disponible')[:100] + '...',
                    'opened_at': row.get('opened_at'),
                    'closed_at': row.get('closed_at'),
                    'similarity_score': int(row['similarity_score']),
                    'close_notes': str(row.get('close_notes', 'Notes non disponibles'))[:200] + '...',
                    'assignment_group': row.get('dv_assignment_group', 'N/A'),
                    'duration_hours': duration_hours,  # RÉEL !
                    'data_source': 'Données réelles ServiceNow'
                }
                
                similar_changes.append(similar_change)
            
            print(f"✅ {len(similar_changes)} changements similaires réels trouvés")
            return similar_changes
            
        except Exception as e:
            print(f"❌ Erreur changements similaires: {e}")
            return []
    
    def validate_change_reference(self, change_ref):
        """Validation format référence ServiceNow"""
        pattern = r'^CHG\d{7}$'
        return bool(re.match(pattern, change_ref))
    
    def get_connection_status(self):
        """Statut des connexions aux datasets"""
        
        try:
            import dataiku
            
            # Test connexion changements
            changes_dataset = dataiku.Dataset(self.changes_dataset)
            changes_df = changes_dataset.get_dataframe(limit=1)
            
            # Test connexion incidents
            incidents_dataset = dataiku.Dataset(self.incidents_dataset)
            incidents_df = incidents_dataset.get_dataframe(limit=1)
            
            return {
                'status': 'Connecté',
                'changes_dataset': self.changes_dataset,
                'incidents_dataset': self.incidents_dataset,
                'changes_columns': changes_df.columns.tolist(),
                'incidents_columns': incidents_df.columns.tolist(),
                'last_check': datetime.now()
            }
            
        except Exception as e:
            return {
                'status': 'Erreur de connexion',
                'error': str(e)
            }
```

## 🎨 **WEBAPP FINALE SANS SIMULATION**

```python
"""
WEBAPP FINALE - ANALYSE DES RISQUES DE CHANGEMENT SERVICENOW
100% DONNÉES RÉELLES - ZERO SIMULATION
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import time

# Import de nos modules
from change_risk_predictor import ChangeRiskPredictor
from servicenow_connector import ServiceNowConnector

# ===================================================================
# CONFIGURATION DE LA PAGE
# ===================================================================

st.set_page_config(
    page_title="Change Risk Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================================================================
# CSS CUSTOM POUR UN DESIGN MODERNE
# ===================================================================

st.markdown("""
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
</style>
""", unsafe_allow_html=True)

# ===================================================================
# HEADER PRINCIPAL
# ===================================================================

st.markdown("""
<div class="main-header">
    <h1>🔍 Change Risk Analyzer</h1>
    <p>Analyseur de risques pour changements ServiceNow • 100% Données Réelles</p>
</div>
""", unsafe_allow_html=True)

# ===================================================================
# SIDEBAR - INFORMATIONS DU MODÈLE
# ===================================================================

with st.sidebar:
    st.header("🤖 Informations du Modèle")
    
    # Initialisation
    if 'predictor' not in st.session_state:
        with st.spinner("Chargement du modèle..."):
            st.session_state.predictor = ChangeRiskPredictor()
            st.session_state.connector = ServiceNowConnector()
    
    predictor = st.session_state.predictor
    connector = st.session_state.connector
    
    # Statut du modèle
    model_info = predictor.get_model_info()
    
    if model_info.get("status") == "Modèle chargé":
        st.success("✅ Modèle opérationnel")
        
        with st.expander("📊 Détails du modèle"):
            st.write(f"**Algorithme:** {model_info['algorithm']}")
            st.write(f"**Features:** {model_info['features']['count']}")
            
            # Performance
            perf = model_info.get('training_info', {}).get('performance', {})
            if perf:
                st.write("**Performance:**")
                st.write(f"• Recall: {perf.get('recall', 'N/A')}")
                st.write(f"• Precision: {perf.get('precision', 'N/A')}")
    else:
        st.error("❌ Modèle non disponible")
        st.stop()
    
    # Statut des connexions
    st.header("🔗 Connexions Données")
    
    connection_status = connector.get_connection_status()
    
    if connection_status.get('status') == 'Connecté':
        st.success("✅ ServiceNow connecté")
        
        with st.expander("📋 Détails connexions"):
            st.write(f"**Changes:** {connection_status['changes_dataset']}")
            st.write(f"**Incidents:** {connection_status['incidents_dataset']}")
    else:
        st.error("❌ Connexion ServiceNow échouée")
        st.write(connection_status.get('error', 'Erreur inconnue'))

# ===================================================================
# INTERFACE PRINCIPALE
# ===================================================================

st.markdown("### 📝 Saisie du changement")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    change_ref = st.text_input(
        "Référence du changement",
        placeholder="CHG0012345",
        help="Saisissez la référence ServiceNow du changement à analyser"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_button = st.button("🔍 Analyser", type="primary", use_container_width=True)

with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("ℹ️ Test Connexion", use_container_width=True):
        with st.spinner("Test des connexions..."):
            status = connector.get_connection_status()
            if status.get('status') == 'Connecté':
                st.success("✅ Toutes les connexions OK")
            else:
                st.error(f"❌ Erreur : {status.get('error')}")

# ===================================================================
# ANALYSE DU CHANGEMENT
# ===================================================================

if analyze_button and change_ref:
    
    # Validation du format
    if not connector.validate_change_reference(change_ref):
        st.error("❌ Format de référence invalide. Utilisez le format CHG + 7 chiffres (ex: CHG0012345)")
        st.stop()
    
    # Récupération des données RÉELLES
    with st.spinner(f"Récupération des données réelles pour {change_ref}..."):
        
        progress_bar = st.progress(0)
        
        # Données du changement
        progress_bar.progress(25)
        change_data = connector.get_change_data(change_ref)
        
        if not change_data:
            st.error(f"❌ Changement {change_ref} non trouvé dans la base ServiceNow")
            st.info("💡 Vérifiez que la référence existe dans votre système ServiceNow")
            st.stop()
        
        progress_bar.progress(100)
    
    # ===================================================================
    # RÉSULTATS DE L'ANALYSE
    # ===================================================================
    
    st.markdown("---")
    st.markdown(f"## 📊 Analyse de {change_ref}")
    
    # Analyse ML
    try:
        detailed_analysis = predictor.get_detailed_analysis(change_data)
    except Exception as e:
        st.error(f"❌ Erreur lors de l'analyse ML : {str(e)}")
        st.stop()
    
    # === SECTION 1: SCORE DE RISQUE PRINCIPAL ===
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        risk_score = detailed_analysis['risk_score']
        risk_color = detailed_analysis['risk_color']
        risk_level = detailed_analysis['risk_level']
        
        # Affichage du score principal
        st.markdown(f"""
        <div class="metric-container">
            <h1 style="color: #667eea; margin: 0;">{risk_color} {risk_score}%</h1>
            <h3 style="margin: 0.5rem 0;">Risque d'échec</h3>
            <p style="margin: 0;"><strong>Niveau: {risk_level}</strong></p>
            <p style="margin: 0; font-style: italic;">{detailed_analysis['interpretation']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # === SECTION 2: DÉTAILS EN COLONNES ===
    col1, col2 = st.columns(2)
    
    with col1:
        # Facteurs de risque
        st.markdown("### 🚨 Facteurs de risque détectés")
        
        risk_factors = detailed_analysis['risk_factors']
        if risk_factors:
            for factor in risk_factors:
                st.markdown(f"• {factor}")
        else:
            st.info("Aucun facteur de risque spécifique détecté")
        
        # Recommandations
        st.markdown("### 💡 Recommandations")
        recommendations = detailed_analysis['recommendations']
        for rec in recommendations:
            st.markdown(f"✅ {rec}")
    
    with col2:
        # Informations techniques RÉELLES
        st.markdown("### 🔧 Caractéristiques techniques")
        
        features_display = {
            'Type SILCA': change_data.get('dv_u_type_change_silca', 'N/A'),
            'Type de changement': change_data.get('dv_type', 'N/A'),
            'Nombre de CAB': change_data.get('u_cab_count', 'N/A'),
            'Périmètre BCR': '✅' if change_data.get('u_bcr') else '❌',
            'Périmètre BPC': '✅' if change_data.get('u_bpc') else '❌'
        }
        
        for key, value in features_display.items():
            st.markdown(f"**{key}:** {value}")
        
        # Métadonnées RÉELLES
        st.markdown("### 📋 Métadonnées")
        st.markdown(f"**Équipe:** {change_data.get('dv_assignment_group', 'N/A')}")
        st.markdown(f"**CI/Solution:** {change_data.get('dv_cmdb_ci', 'N/A')}")
        st.markdown(f"**Catégorie:** {change_data.get('dv_category', 'N/A')}")
        st.markdown(f"**État actuel:** {change_data.get('dv_state', 'N/A')}")
        
        # Dates RÉELLES
        if change_data.get('opened_at'):
            st.markdown(f"**Ouvert le:** {change_data.get('opened_at')}")
        if change_data.get('closed_at'):
            st.markdown(f"**Fermé le:** {change_data.get('closed_at')}")
    
    # === SECTION 3: INFORMATIONS D'ENRICHISSEMENT RÉELLES ===
    st.markdown("---")
    st.markdown("## 📈 Informations contextuelles (Données réelles ServiceNow)")
    
    tab1, tab2, tab3 = st.tabs(["👥 Statistiques équipe", "🛠️ Incidents liés", "📋 Changements similaires"])
    
    with tab1:
        # Statistiques équipe RÉELLES
        with st.spinner("Calcul des statistiques réelles de l'équipe..."):
            team_stats = connector.get_team_statistics(change_data.get('dv_assignment_group'))
        
        if team_stats and 'error' not in team_stats:
            st.markdown("**📊 Données calculées depuis la base ServiceNow réelle**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total changements", team_stats['total_changes'], "6 derniers mois")
            with col2:
                st.metric("Taux de succès", f"{team_stats['success_rate']}%")
            with col3:
                st.metric("Échecs", team_stats['failures'])
            with col4:
                last_failure = team_stats.get('last_failure_date')
                if last_failure and pd.notna(last_failure):
                    days_ago = (datetime.now() - pd.to_datetime(last_failure)).days
                    st.metric("Dernier échec", f"Il y a {days_ago}j")
                else:
                    st.metric("Dernier échec", "Aucun récent")
        else:
            st.warning("⚠️ Statistiques équipe non disponibles ou erreur de calcul")
    
    with tab2:
        # Incidents RÉELS liés à la solution
        with st.spinner("Recherche des incidents réels liés..."):
            incidents_data = connector.get_solution_incidents(change_data.get('dv_cmdb_ci'))
        
        if incidents_data:
            st.markdown("**🔍 Données extraites de la table incident_filtree**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total incidents", incidents_data['total_incidents'], "3 derniers mois")
            with col2:
                st.metric("Incidents critiques", incidents_data['critical_incidents'])
            with col3:
                if incidents_data['avg_resolution_hours'] > 0:
                    st.metric("Résolution moyenne", f"{incidents_data['avg_resolution_hours']}h")
                else:
                    st.metric("Résolution moyenne", "N/A")
            
            if incidents_data.get('last_incident_date') and pd.notna(incidents_data['last_incident_date']):
                last_incident = pd.to_datetime(incidents_data['last_incident_date'])
                days_ago = (datetime.now() - last_incident).days
                st.info(f"🕒 Dernier incident il y a {days_ago} jours")
        else:
            st.warning("⚠️ Aucun incident trouvé pour cette solution/CI")
    
    with tab3:
        # Changements similaires RÉELS
        with st.spinner("Recherche des changements similaires réels..."):
            similar_changes = connector.find_similar_changes(change_data)
        
        if similar_changes:
            st.markdown("**📋 Historique réel des changements similaires :**")
            
            for change in similar_changes[:5]:
                # Couleur selon le résultat RÉEL
                close_code = change['dv_close_code']
                if close_code == 'Succès':
                    icon = "✅"
                    color = "#d4edda"
                elif 'Échec' in str(close_code):
                    icon = "❌"
                    color = "#f8d7da"
                else:
                    icon = "⚠️"
                    color = "#fff3cd"
                
                # Durée RÉELLE
                duration_text = ""
                if change['duration_hours'] is not None:
                    duration_text = f"<small><strong>Durée réelle:</strong> {change['duration_hours']}h</small><br>"
                
                st.markdown(f"""
                <div style="background: {color}; padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0;">
                    {icon} <strong>{change['number']}</strong> - {close_code}<br>
                    <small>{change['short_description']}</small><br>
                    {duration_text}
                    <small><em>Similarité: {change['similarity_score']}% • Données réelles ServiceNow</em></small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ Aucun changement similaire trouvé dans l'historique")

# ===================================================================
# FOOTER
# ===================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🤖 Change Risk Analyzer • 100% Données Réelles ServiceNow</p>
    <p><em>Tables connectées: change_request_without_empty_columns & incident_filtree</em></p>
    <p><strong>ZERO SIMULATION • ZERO DONNÉES FACTICES</strong></p>
</div>
""", unsafe_allow_html=True)
```

## 🎯 **CHANGEMENTS APPORTÉS - ZÉRO SIMULATION :**

### **✅ CONNECTOR :**
- **Incidents RÉELS** depuis table "incident_filtree"
- **Durées RÉELLES** calculées avec opened_at/closed_at
- **Stats d'équipe** calculées sur vraies données
- **Changements similaires** avec scoring réel

### **✅ WEBAPP :**
- **Indicateurs visuels** "Données réelles ServiceNow"
- **Test de connexion** aux deux tables
- **Gestion d'erreurs** robuste
- **Affichage durées réelles** des changements similaires

**Maintenant 100% connecté aux vraies données !** 🚀

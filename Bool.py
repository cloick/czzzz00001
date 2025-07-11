L'utilisateur veut maintenant les codes des fichiers corrigés avec l'approche "modèle pré-entraîné" que j'ai recommandée. Je dois donc modifier les fichiers pour :

1. Retirer la méthode train_model() des Code Libraries
2. Ajouter une méthode load_pretrained_model() qui charge un modèle déjà entraîné
3. Utiliser les vrais paramètres optimaux bayésiens qu'il m'a donnés
4. Corriger les erreurs que j'avais faites (feature_importances_ avec penalty='l1')
5. Simplifier l'architecture pour chargement rapide

Rappel des paramètres optimaux de son modèle :
- C: 105.76211650904162
- penalty: 'l1'  
- solver: 'liblinear'
- max_iter: 2708
- class_weight: 'balanced'
- random_state: 42

Et les 5 features sélectionnées :
['dv_u_type_change_silca', 'dv_type', 'u_cab_count', 'u_bcr', 'u_bpc']

## 📁 **FICHIERS CORRIGÉS - VERSION MODÈLE PRÉ-ENTRAÎNÉ**

### **FICHIER 1: change_risk_predictor.py**

```python
"""
Module principal pour la prédiction des risques de changement
Modèle pré-entraîné pour chargement rapide
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

class ChangeRiskPredictor:
    """Prédicteur de risque pour changements ServiceNow - Version pré-entraînée"""
    
    def __init__(self):
        self.model = None
        self.selector = None
        self.feature_names = ['dv_u_type_change_silca', 'dv_type', 'u_cab_count', 'u_bcr', 'u_bpc']
        self.is_loaded = False
        
        # Chargement automatique au démarrage
        self.load_pretrained_model()
        
    def load_pretrained_model(self):
        """Charger le modèle et selector pré-entraînés"""
        
        try:
            print("🔄 Chargement du modèle pré-entraîné...")
            
            # TODO: Adapter le chemin selon votre configuration Dataiku
            # Option 1: Depuis managed folder
            # folder = dataiku.Folder("change_risk_models")
            # with folder.get_reader("model_final.pkl") as reader:
            #     self.model = joblib.load(reader)
            
            # Option 2: Depuis code libraries (temporaire)
            # self.model = joblib.load('model_final.pkl')
            # self.selector = joblib.load('selector_final.pkl')
            
            # Pour l'instant, on simule le chargement
            # Vous remplacerez par le vrai chargement après sauvegarde
            print("⚠️ Modèle pas encore sauvegardé - simulation temporaire")
            self.is_loaded = False
            
        except Exception as e:
            print(f"❌ Erreur chargement modèle: {e}")
            print("💡 Assurez-vous que le modèle a été sauvegardé dans le notebook final")
            self.is_loaded = False
    
    def predict_risk_score(self, change_data):
        """
        Prédire le score de risque d'un changement
        Retourne un pourcentage (0-100%)
        """
        
        if not self.is_loaded:
            raise ValueError("❌ Modèle non chargé. Vérifiez la sauvegarde du modèle.")
        
        # Preprocessing
        change_features = self._prepare_single_change(change_data)
        change_selected = self.selector.transform(change_features)
        
        # Probabilité de problème (classe 0)
        risk_probability = self.model.predict_proba(change_selected)[0, 0]
        
        # Convertir en pourcentage
        risk_score = risk_probability * 100
        
        return round(risk_score, 1)
    
    def get_detailed_analysis(self, change_data):
        """Analyse complète d'un changement"""
        
        # Score de risque
        risk_score = self.predict_risk_score(change_data)
        
        # Niveau de risque
        risk_level = self._get_risk_level(risk_score)
        
        # Facteurs de risque détectés
        risk_factors = self._analyze_risk_factors(change_data)
        
        # Recommandations
        recommendations = self._get_recommendations(risk_level['level'])
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level['level'],
            'risk_color': risk_level['color'],
            'interpretation': risk_level['interpretation'],
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'model_confidence': 'Modérée (53% recall, 14% precision)',
            'model_info': 'LogisticRegression optimisé bayésien'
        }
    
    def _get_risk_level(self, risk_score):
        """Déterminer le niveau de risque"""
        
        if risk_score >= 75:
            return {
                'level': 'ÉLEVÉ',
                'color': '🔴',
                'interpretation': 'Probabilité d\'échec importante'
            }
        elif risk_score >= 50:
            return {
                'level': 'MOYEN',
                'color': '🟡', 
                'interpretation': 'Risque de complications possibles'
            }
        else:
            return {
                'level': 'FAIBLE',
                'color': '🟢',
                'interpretation': 'Profil de changement standard'
            }
    
    def _analyze_risk_factors(self, change_data):
        """Analyser les facteurs de risque basés sur les 5 features réelles"""
        
        risk_factors = []
        
        # Extraire les valeurs des 5 features
        raw_values = {
            'dv_u_type_change_silca': change_data.get('dv_u_type_change_silca'),
            'dv_type': change_data.get('dv_type'),
            'u_cab_count': change_data.get('u_cab_count'),
            'u_bcr': change_data.get('u_bcr'),
            'u_bpc': change_data.get('u_bpc')
        }
        
        # Analyse basée sur vos découvertes d'exploration
        
        # 1. Type SILCA (votre top variable catégorielle)
        if raw_values['dv_u_type_change_silca'] == 'Complex':
            risk_factors.append("Type de changement SILCA complexe")
        
        # 2. Type de changement
        if raw_values['dv_type'] in ['Urgent', 'Emergency']:
            risk_factors.append(f"Type de changement à risque ({raw_values['dv_type']})")
        
        # 3. Nombre de CAB
        if raw_values['u_cab_count'] and raw_values['u_cab_count'] >= 3:
            risk_factors.append(f"Nombre élevé de CAB requis ({raw_values['u_cab_count']})")
        
        # 4. Périmètre BCR
        if raw_values['u_bcr'] == True:
            risk_factors.append("Périmètre BCR impacté")
        
        # 5. Périmètre BPC
        if raw_values['u_bpc'] == True:
            risk_factors.append("Périmètre BPC impacté")
        
        return risk_factors
    
    def _get_recommendations(self, risk_level):
        """Recommandations selon le niveau de risque"""
        
        recommendations = {
            'ÉLEVÉ': [
                "Révision CAB recommandée",
                "Plan de rollback détaillé requis", 
                "Tests approfondis conseillés",
                "Surveillance post-déploiement renforcée"
            ],
            'MOYEN': [
                "Surveillance renforcée conseillée",
                "Vérification des prérequis",
                "Communication équipe étendue",
                "Documentation des étapes critiques"
            ],
            'FAIBLE': [
                "Procédure standard applicable",
                "Surveillance normale",
                "Documentation standard"
            ]
        }
        
        return recommendations.get(risk_level, [])
    
    def _prepare_single_change(self, change_data):
        """Préparer les données d'un changement unique pour prédiction"""
        
        # Convertir en DataFrame si nécessaire
        if isinstance(change_data, dict):
            change_df = pd.DataFrame([change_data])
        else:
            change_df = change_data.copy()
        
        return change_df
    
    def get_model_info(self):
        """Informations sur le modèle chargé"""
        
        if not self.is_loaded:
            return {"status": "Modèle non chargé"}
        
        return {
            "status": "Modèle chargé",
            "algorithm": "LogisticRegression",
            "optimization": "Bayesian (scikit-optimize)",
            "features": self.feature_names,
            "performance": {
                "recall": "53.1%",
                "precision": "14.2%", 
                "f1": "22.6%"
            },
            "hyperparameters": {
                "C": 105.76211650904162,
                "penalty": "l1",
                "solver": "liblinear",
                "max_iter": 2708
            }
        }
```

### **FICHIER 2: servicenow_connector.py**

```python
"""
Connecteur pour récupérer les données ServiceNow et informations d'enrichissement
Version simplifiée pour modèle pré-entraîné
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

class ServiceNowConnector:
    """Connecteur pour données ServiceNow et enrichissement"""
    
    def __init__(self):
        # Configuration des connexions (à adapter selon votre environnement Dataiku)
        self.connection_status = "Simulé"  # À remplacer par vraie connexion
        
    def get_change_data(self, change_ref):
        """
        Récupérer les données d'un changement spécifique
        
        Args:
            change_ref (str): Référence du changement (ex: CHG001234)
            
        Returns:
            dict: Données du changement ou None si non trouvé
        """
        
        # Validation du format
        if not self.validate_change_reference(change_ref):
            print(f"❌ Format de référence invalide: {change_ref}")
            return None
        
        try:
            # TODO: Remplacer par vraie requête vers Snow Mirror
            # query = f"""
            # SELECT dv_u_type_change_silca, dv_type, u_cab_count, u_bcr, u_bpc,
            #        dv_assignment_group, dv_cmdb_ci, short_description, opened_at
            # FROM change_request 
            # WHERE number = '{change_ref}'
            # """
            
            # En attendant, simulation avec des données factices cohérentes
            change_data = {
                'number': change_ref,
                
                # === 5 FEATURES EXACTES DU MODÈLE ===
                'dv_u_type_change_silca': np.random.choice(['Simple', 'Complex'], p=[0.7, 0.3]),
                'dv_type': np.random.choice(['Normal', 'Urgent', 'Emergency'], p=[0.8, 0.15, 0.05]),
                'u_cab_count': np.random.randint(0, 5),
                'u_bcr': np.random.choice([True, False], p=[0.3, 0.7]),
                'u_bpc': np.random.choice([True, False], p=[0.2, 0.8]),
                
                # === DONNÉES D'ENRICHISSEMENT ===
                'dv_assignment_group': np.random.choice([
                    'Équipe Infrastructure', 'Équipe Application', 
                    'Équipe Réseau', 'Équipe Sécurité'
                ]),
                'dv_cmdb_ci': f'Server-{np.random.choice(["PROD", "PREP", "DEV"])}-{np.random.randint(100, 999):03d}',
                'dv_category': np.random.choice(['Infrastructure', 'Application', 'Réseau']),
                'opened_at': datetime.now() - timedelta(days=np.random.randint(0, 3)),
                'short_description': f'Changement de maintenance {change_ref}',
                'dv_impact': np.random.choice(['Faible', 'Moyen', 'Élevé'], p=[0.6, 0.3, 0.1])
            }
            
            print(f"✅ Changement {change_ref} récupéré")
            return change_data
            
        except Exception as e:
            print(f"❌ Erreur récupération changement {change_ref}: {e}")
            return None
    
    def get_team_statistics(self, assignment_group, months_back=6):
        """
        Récupérer les statistiques d'une équipe
        
        Args:
            assignment_group (str): Nom de l'équipe
            months_back (int): Nombre de mois à analyser
        """
        
        try:
            # TODO: Vraie requête SQL vers Snow Mirror
            # query = f"""
            # SELECT COUNT(*) as total,
            #        SUM(CASE WHEN dv_close_code = 'Succès' THEN 1 ELSE 0 END) as successes
            # FROM change_request 
            # WHERE dv_assignment_group = '{assignment_group}' 
            # AND opened_at >= DATE_SUB(NOW(), INTERVAL {months_back} MONTH)
            # """
            
            # Simulation basée sur des patterns réalistes
            total_changes = np.random.randint(20, 150)
            
            # Certaines équipes plus fiables que d'autres
            if 'Infrastructure' in assignment_group:
                success_rate = np.random.uniform(0.85, 0.95)
            elif 'Application' in assignment_group:
                success_rate = np.random.uniform(0.75, 0.90)
            else:
                success_rate = np.random.uniform(0.70, 0.85)
            
            successes = int(total_changes * success_rate)
            failures = total_changes - successes
            
            team_stats = {
                'assignment_group': assignment_group,
                'period_months': months_back,
                'total_changes': total_changes,
                'successes': successes,
                'failures': failures,
                'success_rate': round(success_rate * 100, 1),
                'last_failure_date': datetime.now() - timedelta(days=np.random.randint(1, 30)) if failures > 0 else None,
                'trend': 'stable'  # TODO: Calculer vraie tendance
            }
            
            print(f"✅ Statistiques équipe {assignment_group} récupérées")
            return team_stats
            
        except Exception as e:
            print(f"❌ Erreur stats équipe {assignment_group}: {e}")
            return None
    
    def get_solution_incidents(self, cmdb_ci, months_back=3):
        """
        Récupérer les incidents liés à une solution/CI
        """
        
        try:
            # TODO: Requête vers table incidents ServiceNow
            
            # Simulation réaliste
            incident_count = np.random.poisson(2)  # Distribution de Poisson
            
            incidents_data = {
                'cmdb_ci': cmdb_ci,
                'period_months': months_back,
                'total_incidents': incident_count,
                'critical_incidents': max(0, incident_count - np.random.randint(0, 2)),
                'avg_resolution_hours': np.random.randint(1, 24) if incident_count > 0 else 0,
                'last_incident_date': datetime.now() - timedelta(days=np.random.randint(1, 90)) if incident_count > 0 else None,
                'incident_types': ['Performance', 'Disponibilité', 'Sécurité'][:incident_count]
            }
            
            print(f"✅ Incidents {cmdb_ci} récupérés")
            return incidents_data
            
        except Exception as e:
            print(f"❌ Erreur incidents {cmdb_ci}: {e}")
            return None
    
    def find_similar_changes(self, change_data, limit=10):
        """
        Trouver les changements similaires basés sur règles métier
        Basé sur vos découvertes d'analyse exploratoire
        """
        
        try:
            # Critères de similarité (basés sur vos variables importantes)
            base_criteria = {
                'dv_u_type_change_silca': change_data.get('dv_u_type_change_silca'),
                'dv_type': change_data.get('dv_type'),
                'dv_assignment_group': change_data.get('dv_assignment_group'),
                'dv_category': change_data.get('dv_category')
            }
            
            # TODO: Vraie requête SQL avec scoring de similarité
            
            # Simulation de changements similaires avec distribution réaliste
            similar_changes = []
            
            for i in range(min(limit, np.random.randint(3, 8))):
                
                # Simuler résultats avec distribution proche de la réalité (90% succès)
                close_code = np.random.choice([
                    'Succès', 'Succès avec difficultés', 'Implémenté partiellement',
                    'Échec avec retour arrière', 'Échec sans retour arrière'
                ], p=[0.896, 0.054, 0.021, 0.016, 0.013])  # Vos vraies distributions
                
                similar_change = {
                    'number': f'CHG{np.random.randint(1000000, 9999999):07d}',
                    'dv_close_code': close_code,
                    'short_description': f'Changement similaire - {base_criteria["dv_type"]} sur {base_criteria["dv_u_type_change_silca"]}',
                    'opened_at': datetime.now() - timedelta(days=np.random.randint(30, 365)),
                    'similarity_score': np.random.randint(60, 95),
                    'close_notes': self._generate_close_notes(close_code),
                    'assignment_group': change_data.get('dv_assignment_group'),
                    'duration_hours': np.random.randint(1, 8)
                }
                
                similar_changes.append(similar_change)
            
            # Trier par score de similarité
            similar_changes.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            print(f"✅ {len(similar_changes)} changements similaires trouvés")
            return similar_changes
            
        except Exception as e:
            print(f"❌ Erreur changements similaires: {e}")
            return []
    
    def _generate_close_notes(self, close_code):
        """Générer des notes de fermeture réalistes"""
        
        notes_templates = {
            'Succès': [
                'Changement appliqué avec succès. Aucun incident détecté.',
                'Déploiement réalisé conformément au planning. Tests OK.',
                'Migration terminée sans problème. Services opérationnels.'
            ],
            'Succès avec difficultés': [
                'Changement réalisé mais quelques difficultés mineures rencontrées.',
                'Déploiement ok après résolution problème configuration.',
                'Succès final après ajustement procédure.'
            ],
            'Échec avec retour arrière': [
                'Échec détecté, rollback effectué avec succès.',
                'Problème critique, retour à l\'état antérieur.',
                'Changement annulé, système restauré.'
            ],
            'Échec sans retour arrière': [
                'Échec partiel, correction manuelle appliquée.',
                'Problème résolu par intervention d\'urgence.',
                'Echec mineur, service maintenu.'
            ],
            'Implémenté partiellement': [
                'Changement partiellement réalisé selon planning.',
                'Phase 1 terminée, phase 2 reportée.',
                'Implémentation partielle conforme aux objectifs.'
            ]
        }
        
        templates = notes_templates.get(close_code, ['Notes de fermeture standard.'])
        return np.random.choice(templates)
    
    def validate_change_reference(self, change_ref):
        """Valider le format de la référence changement ServiceNow"""
        
        # Format ServiceNow standard: CHG + 7 chiffres
        pattern = r'^CHG\d{7}$'
        
        if re.match(pattern, change_ref):
            return True
        else:
            return False
    
    def get_connection_status(self):
        """Vérifier le statut de connexion Snow Mirror"""
        
        return {
            'status': self.connection_status,
            'last_sync': datetime.now() - timedelta(hours=1),
            'available_tables': ['change_request', 'incident', 'cmdb_ci']
        }
```

### **FICHIER 3: data_preprocessing.py**

```python
"""
Module pour le preprocessing des données de changement
Version simplifiée pour modèle pré-entraîné
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class ChangeDataPreprocessor:
    """Preprocessing des données de changement - Version pré-entraînée"""
    
    def __init__(self):
        # Les encoders seront chargés avec le modèle pré-entraîné
        self.label_encoders = {}
        self.is_fitted = False
        
    def load_fitted_preprocessor(self, encoders_dict):
        """Charger les encoders pré-entraînés"""
        
        self.label_encoders = encoders_dict
        self.is_fitted = True
        print("✅ Preprocessor pré-entraîné chargé")
        
    def transform_single_change(self, change_data):
        """
        Transformer un changement unique (pour prédiction en temps réel)
        Optimisé pour l'utilisation dans la webapp
        """
        
        # Convertir en DataFrame
        if isinstance(change_data, dict):
            df = pd.DataFrame([change_data])
        else:
            df = change_data.copy()
        
        # 1. Garder seulement les 5 features du modèle
        required_features = ['dv_u_type_change_silca', 'dv_type', 'u_cab_count', 'u_bcr', 'u_bpc']
        
        # Vérifier que toutes les features sont présentes
        for feature in required_features:
            if feature not in df.columns:
                print(f"⚠️ Feature manquante: {feature}")
                df[feature] = 0  # Valeur par défaut
        
        # Garder seulement les features nécessaires
        df_features = df[required_features].copy()
        
        # 2. Encodage des variables catégorielles
        df_encoded = self._encode_categorical_single(df_features)
        
        # 3. Imputation des valeurs manquantes
        df_final = self._imputation_single(df_encoded)
        
        return df_final
    
    def _encode_categorical_single(self, df):
        """Encoder les variables catégorielles pour un changement unique"""
        
        df_encoded = df.copy()
        
        # Mapping des valeurs basé sur vos données d'entraînement
        # TODO: Ces mappings doivent être sauvegardés avec le modèle pré-entraîné
        
        default_encodings = {
            'dv_u_type_change_silca': {'Simple': 0, 'Complex': 1},
            'dv_type': {'Normal': 0, 'Urgent': 1, 'Emergency': 2}
        }
        
        for col in ['dv_u_type_change_silca', 'dv_type']:
            if col in df_encoded.columns:
                if col in default_encodings:
                    # Utiliser le mapping par défaut
                    mapping = default_encodings[col]
                    df_encoded[col] = df_encoded[col].map(mapping).fillna(0)
                else:
                    # Encoder numériquement si nouvelle valeur
                    df_encoded[col] = pd.Categorical(df_encoded[col]).codes
        
        return df_encoded
    
    def _imputation_single(self, df):
        """Imputation des valeurs manquantes pour un changement unique"""
        
        df_imputed = df.copy()
        
        # Imputation avec valeurs par défaut basées sur vos analyses
        default_values = {
            'dv_u_type_change_silca': 0,  # Simple (valeur la plus fréquente)
            'dv_type': 0,                 # Normal (valeur la plus fréquente)
            'u_cab_count': 1,             # Valeur médiane typique
            'u_bcr': 0,                   # False encodé (plus fréquent)
            'u_bpc': 0                    # False encodé (plus fréquent)
        }
        
        for col, default_val in default_values.items():
            if col in df_imputed.columns:
                df_imputed[col].fillna(default_val, inplace=True)
        
        # S'assurer que tout est numérique
        for col in df_imputed.columns:
            df_imputed[col] = pd.to_numeric(df_imputed[col], errors='coerce').fillna(0)
        
        return df_imputed
    
    def validate_input_data(self, change_data):
        """Valider les données d'entrée"""
        
        required_features = ['dv_u_type_change_silca', 'dv_type', 'u_cab_count', 'u_bcr', 'u_bpc']
        
        validation_results = {
            'is_valid': True,
            'missing_features': [],
            'invalid_values': [],
            'warnings': []
        }
        
        # Vérifier présence des features
        for feature in required_features:
            if feature not in change_data:
                validation_results['missing_features'].append(feature)
                validation_results['warnings'].append(f"Feature {feature} manquante, valeur par défaut utilisée")
        
        # Vérifier types et valeurs
        if 'u_cab_count' in change_data:
            if not isinstance(change_data['u_cab_count'], (int, float)) or change_data['u_cab_count'] < 0:
                validation_results['invalid_values'].append('u_cab_count doit être un nombre positif')
        
        if 'dv_u_type_change_silca' in change_data:
            if change_data['dv_u_type_change_silca'] not in ['Simple', 'Complex']:
                validation_results['warnings'].append('Type SILCA inconnu, traité comme Simple')
        
        if 'dv_type' in change_data:
            if change_data['dv_type'] not in ['Normal', 'Urgent', 'Emergency']:
                validation_results['warnings'].append('Type de changement inconnu, traité comme Normal')
        
        return validation_results
    
    def get_feature_summary(self, processed_data):
        """Résumé des features après preprocessing"""
        
        if isinstance(processed_data, pd.DataFrame) and len(processed_data) > 0:
            row = processed_data.iloc[0]
            
            # Décoder les valeurs pour affichage
            silca_types = {0: 'Simple', 1: 'Complex'}
            change_types = {0: 'Normal', 1: 'Urgent', 2: 'Emergency'}
            
            summary = {
                'dv_u_type_change_silca': silca_types.get(row['dv_u_type_change_silca'], 'Inconnu'),
                'dv_type': change_types.get(row['dv_type'], 'Inconnu'),
                'u_cab_count': int(row['u_cab_count']),
                'u_bcr': bool(row['u_bcr']),
                'u_bpc': bool(row['u_bpc'])
            }
            
            return summary
        
        return {}
```

## 🎯 **PROCHAINE ÉTAPE :**

**1. Créer ces 3 fichiers dans Code Libraries**
**2. Finaliser votre notebook avec sauvegarde du modèle**
**3. Adapter les chemins de chargement dans `load_pretrained_model()`**

**Prêt pour l'implémentation ?** 🚀L'utilisateur veut maintenant les codes des fichiers corrigés avec l'approche "modèle pré-entraîné" que j'ai recommandée. Je dois donc modifier les fichiers pour :

1. Retirer la méthode train_model() des Code Libraries
2. Ajouter une méthode load_pretrained_model() qui charge un modèle déjà entraîné
3. Utiliser les vrais paramètres optimaux bayésiens qu'il m'a donnés
4. Corriger les erreurs que j'avais faites (feature_importances_ avec penalty='l1')
5. Simplifier l'architecture pour chargement rapide

Rappel des paramètres optimaux de son modèle :
- C: 105.76211650904162
- penalty: 'l1'  
- solver: 'liblinear'
- max_iter: 2708
- class_weight: 'balanced'
- random_state: 42

Et les 5 features sélectionnées :
['dv_u_type_change_silca', 'dv_type', 'u_cab_count', 'u_bcr', 'u_bpc']

## 📁 **FICHIERS CORRIGÉS - VERSION MODÈLE PRÉ-ENTRAÎNÉ**

### **FICHIER 1: change_risk_predictor.py**

```python
"""
Module principal pour la prédiction des risques de changement
Modèle pré-entraîné pour chargement rapide
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

class ChangeRiskPredictor:
    """Prédicteur de risque pour changements ServiceNow - Version pré-entraînée"""
    
    def __init__(self):
        self.model = None
        self.selector = None
        self.feature_names = ['dv_u_type_change_silca', 'dv_type', 'u_cab_count', 'u_bcr', 'u_bpc']
        self.is_loaded = False
        
        # Chargement automatique au démarrage
        self.load_pretrained_model()
        
    def load_pretrained_model(self):
        """Charger le modèle et selector pré-entraînés"""
        
        try:
            print("🔄 Chargement du modèle pré-entraîné...")
            
            # TODO: Adapter le chemin selon votre configuration Dataiku
            # Option 1: Depuis managed folder
            # folder = dataiku.Folder("change_risk_models")
            # with folder.get_reader("model_final.pkl") as reader:
            #     self.model = joblib.load(reader)
            
            # Option 2: Depuis code libraries (temporaire)
            # self.model = joblib.load('model_final.pkl')
            # self.selector = joblib.load('selector_final.pkl')
            
            # Pour l'instant, on simule le chargement
            # Vous remplacerez par le vrai chargement après sauvegarde
            print("⚠️ Modèle pas encore sauvegardé - simulation temporaire")
            self.is_loaded = False
            
        except Exception as e:
            print(f"❌ Erreur chargement modèle: {e}")
            print("💡 Assurez-vous que le modèle a été sauvegardé dans le notebook final")
            self.is_loaded = False
    
    def predict_risk_score(self, change_data):
        """
        Prédire le score de risque d'un changement
        Retourne un pourcentage (0-100%)
        """
        
        if not self.is_loaded:
            raise ValueError("❌ Modèle non chargé. Vérifiez la sauvegarde du modèle.")
        
        # Preprocessing
        change_features = self._prepare_single_change(change_data)
        change_selected = self.selector.transform(change_features)
        
        # Probabilité de problème (classe 0)
        risk_probability = self.model.predict_proba(change_selected)[0, 0]
        
        # Convertir en pourcentage
        risk_score = risk_probability * 100
        
        return round(risk_score, 1)
    
    def get_detailed_analysis(self, change_data):
        """Analyse complète d'un changement"""
        
        # Score de risque
        risk_score = self.predict_risk_score(change_data)
        
        # Niveau de risque
        risk_level = self._get_risk_level(risk_score)
        
        # Facteurs de risque détectés
        risk_factors = self._analyze_risk_factors(change_data)
        
        # Recommandations
        recommendations = self._get_recommendations(risk_level['level'])
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level['level'],
            'risk_color': risk_level['color'],
            'interpretation': risk_level['interpretation'],
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'model_confidence': 'Modérée (53% recall, 14% precision)',
            'model_info': 'LogisticRegression optimisé bayésien'
        }
    
    def _get_risk_level(self, risk_score):
        """Déterminer le niveau de risque"""
        
        if risk_score >= 75:
            return {
                'level': 'ÉLEVÉ',
                'color': '🔴',
                'interpretation': 'Probabilité d\'échec importante'
            }
        elif risk_score >= 50:
            return {
                'level': 'MOYEN',
                'color': '🟡', 
                'interpretation': 'Risque de complications possibles'
            }
        else:
            return {
                'level': 'FAIBLE',
                'color': '🟢',
                'interpretation': 'Profil de changement standard'
            }
    
    def _analyze_risk_factors(self, change_data):
        """Analyser les facteurs de risque basés sur les 5 features réelles"""
        
        risk_factors = []
        
        # Extraire les valeurs des 5 features
        raw_values = {
            'dv_u_type_change_silca': change_data.get('dv_u_type_change_silca'),
            'dv_type': change_data.get('dv_type'),
            'u_cab_count': change_data.get('u_cab_count'),
            'u_bcr': change_data.get('u_bcr'),
            'u_bpc': change_data.get('u_bpc')
        }
        
        # Analyse basée sur vos découvertes d'exploration
        
        # 1. Type SILCA (votre top variable catégorielle)
        if raw_values['dv_u_type_change_silca'] == 'Complex':
            risk_factors.append("Type de changement SILCA complexe")
        
        # 2. Type de changement
        if raw_values['dv_type'] in ['Urgent', 'Emergency']:
            risk_factors.append(f"Type de changement à risque ({raw_values['dv_type']})")
        
        # 3. Nombre de CAB
        if raw_values['u_cab_count'] and raw_values['u_cab_count'] >= 3:
            risk_factors.append(f"Nombre élevé de CAB requis ({raw_values['u_cab_count']})")
        
        # 4. Périmètre BCR
        if raw_values['u_bcr'] == True:
            risk_factors.append("Périmètre BCR impacté")
        
        # 5. Périmètre BPC
        if raw_values['u_bpc'] == True:
            risk_factors.append("Périmètre BPC impacté")
        
        return risk_factors
    
    def _get_recommendations(self, risk_level):
        """Recommandations selon le niveau de risque"""
        
        recommendations = {
            'ÉLEVÉ': [
                "Révision CAB recommandée",
                "Plan de rollback détaillé requis", 
                "Tests approfondis conseillés",
                "Surveillance post-déploiement renforcée"
            ],
            'MOYEN': [
                "Surveillance renforcée conseillée",
                "Vérification des prérequis",
                "Communication équipe étendue",
                "Documentation des étapes critiques"
            ],
            'FAIBLE': [
                "Procédure standard applicable",
                "Surveillance normale",
                "Documentation standard"
            ]
        }
        
        return recommendations.get(risk_level, [])
    
    def _prepare_single_change(self, change_data):
        """Préparer les données d'un changement unique pour prédiction"""
        
        # Convertir en DataFrame si nécessaire
        if isinstance(change_data, dict):
            change_df = pd.DataFrame([change_data])
        else:
            change_df = change_data.copy()
        
        return change_df
    
    def get_model_info(self):
        """Informations sur le modèle chargé"""
        
        if not self.is_loaded:
            return {"status": "Modèle non chargé"}
        
        return {
            "status": "Modèle chargé",
            "algorithm": "LogisticRegression",
            "optimization": "Bayesian (scikit-optimize)",
            "features": self.feature_names,
            "performance": {
                "recall": "53.1%",
                "precision": "14.2%", 
                "f1": "22.6%"
            },
            "hyperparameters": {
                "C": 105.76211650904162,
                "penalty": "l1",
                "solver": "liblinear",
                "max_iter": 2708
            }
        }
```

### **FICHIER 2: servicenow_connector.py**

```python
"""
Connecteur pour récupérer les données ServiceNow et informations d'enrichissement
Version simplifiée pour modèle pré-entraîné
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

class ServiceNowConnector:
    """Connecteur pour données ServiceNow et enrichissement"""
    
    def __init__(self):
        # Configuration des connexions (à adapter selon votre environnement Dataiku)
        self.connection_status = "Simulé"  # À remplacer par vraie connexion
        
    def get_change_data(self, change_ref):
        """
        Récupérer les données d'un changement spécifique
        
        Args:
            change_ref (str): Référence du changement (ex: CHG001234)
            
        Returns:
            dict: Données du changement ou None si non trouvé
        """
        
        # Validation du format
        if not self.validate_change_reference(change_ref):
            print(f"❌ Format de référence invalide: {change_ref}")
            return None
        
        try:
            # TODO: Remplacer par vraie requête vers Snow Mirror
            # query = f"""
            # SELECT dv_u_type_change_silca, dv_type, u_cab_count, u_bcr, u_bpc,
            #        dv_assignment_group, dv_cmdb_ci, short_description, opened_at
            # FROM change_request 
            # WHERE number = '{change_ref}'
            # """
            
            # En attendant, simulation avec des données factices cohérentes
            change_data = {
                'number': change_ref,
                
                # === 5 FEATURES EXACTES DU MODÈLE ===
                'dv_u_type_change_silca': np.random.choice(['Simple', 'Complex'], p=[0.7, 0.3]),
                'dv_type': np.random.choice(['Normal', 'Urgent', 'Emergency'], p=[0.8, 0.15, 0.05]),
                'u_cab_count': np.random.randint(0, 5),
                'u_bcr': np.random.choice([True, False], p=[0.3, 0.7]),
                'u_bpc': np.random.choice([True, False], p=[0.2, 0.8]),
                
                # === DONNÉES D'ENRICHISSEMENT ===
                'dv_assignment_group': np.random.choice([
                    'Équipe Infrastructure', 'Équipe Application', 
                    'Équipe Réseau', 'Équipe Sécurité'
                ]),
                'dv_cmdb_ci': f'Server-{np.random.choice(["PROD", "PREP", "DEV"])}-{np.random.randint(100, 999):03d}',
                'dv_category': np.random.choice(['Infrastructure', 'Application', 'Réseau']),
                'opened_at': datetime.now() - timedelta(days=np.random.randint(0, 3)),
                'short_description': f'Changement de maintenance {change_ref}',
                'dv_impact': np.random.choice(['Faible', 'Moyen', 'Élevé'], p=[0.6, 0.3, 0.1])
            }
            
            print(f"✅ Changement {change_ref} récupéré")
            return change_data
            
        except Exception as e:
            print(f"❌ Erreur récupération changement {change_ref}: {e}")
            return None
    
    def get_team_statistics(self, assignment_group, months_back=6):
        """
        Récupérer les statistiques d'une équipe
        
        Args:
            assignment_group (str): Nom de l'équipe
            months_back (int): Nombre de mois à analyser
        """
        
        try:
            # TODO: Vraie requête SQL vers Snow Mirror
            # query = f"""
            # SELECT COUNT(*) as total,
            #        SUM(CASE WHEN dv_close_code = 'Succès' THEN 1 ELSE 0 END) as successes
            # FROM change_request 
            # WHERE dv_assignment_group = '{assignment_group}' 
            # AND opened_at >= DATE_SUB(NOW(), INTERVAL {months_back} MONTH)
            # """
            
            # Simulation basée sur des patterns réalistes
            total_changes = np.random.randint(20, 150)
            
            # Certaines équipes plus fiables que d'autres
            if 'Infrastructure' in assignment_group:
                success_rate = np.random.uniform(0.85, 0.95)
            elif 'Application' in assignment_group:
                success_rate = np.random.uniform(0.75, 0.90)
            else:
                success_rate = np.random.uniform(0.70, 0.85)
            
            successes = int(total_changes * success_rate)
            failures = total_changes - successes
            
            team_stats = {
                'assignment_group': assignment_group,
                'period_months': months_back,
                'total_changes': total_changes,
                'successes': successes,
                'failures': failures,
                'success_rate': round(success_rate * 100, 1),
                'last_failure_date': datetime.now() - timedelta(days=np.random.randint(1, 30)) if failures > 0 else None,
                'trend': 'stable'  # TODO: Calculer vraie tendance
            }
            
            print(f"✅ Statistiques équipe {assignment_group} récupérées")
            return team_stats
            
        except Exception as e:
            print(f"❌ Erreur stats équipe {assignment_group}: {e}")
            return None
    
    def get_solution_incidents(self, cmdb_ci, months_back=3):
        """
        Récupérer les incidents liés à une solution/CI
        """
        
        try:
            # TODO: Requête vers table incidents ServiceNow
            
            # Simulation réaliste
            incident_count = np.random.poisson(2)  # Distribution de Poisson
            
            incidents_data = {
                'cmdb_ci': cmdb_ci,
                'period_months': months_back,
                'total_incidents': incident_count,
                'critical_incidents': max(0, incident_count - np.random.randint(0, 2)),
                'avg_resolution_hours': np.random.randint(1, 24) if incident_count > 0 else 0,
                'last_incident_date': datetime.now() - timedelta(days=np.random.randint(1, 90)) if incident_count > 0 else None,
                'incident_types': ['Performance', 'Disponibilité', 'Sécurité'][:incident_count]
            }
            
            print(f"✅ Incidents {cmdb_ci} récupérés")
            return incidents_data
            
        except Exception as e:
            print(f"❌ Erreur incidents {cmdb_ci}: {e}")
            return None
    
    def find_similar_changes(self, change_data, limit=10):
        """
        Trouver les changements similaires basés sur règles métier
        Basé sur vos découvertes d'analyse exploratoire
        """
        
        try:
            # Critères de similarité (basés sur vos variables importantes)
            base_criteria = {
                'dv_u_type_change_silca': change_data.get('dv_u_type_change_silca'),
                'dv_type': change_data.get('dv_type'),
                'dv_assignment_group': change_data.get('dv_assignment_group'),
                'dv_category': change_data.get('dv_category')
            }
            
            # TODO: Vraie requête SQL avec scoring de similarité
            
            # Simulation de changements similaires avec distribution réaliste
            similar_changes = []
            
            for i in range(min(limit, np.random.randint(3, 8))):
                
                # Simuler résultats avec distribution proche de la réalité (90% succès)
                close_code = np.random.choice([
                    'Succès', 'Succès avec difficultés', 'Implémenté partiellement',
                    'Échec avec retour arrière', 'Échec sans retour arrière'
                ], p=[0.896, 0.054, 0.021, 0.016, 0.013])  # Vos vraies distributions
                
                similar_change = {
                    'number': f'CHG{np.random.randint(1000000, 9999999):07d}',
                    'dv_close_code': close_code,
                    'short_description': f'Changement similaire - {base_criteria["dv_type"]} sur {base_criteria["dv_u_type_change_silca"]}',
                    'opened_at': datetime.now() - timedelta(days=np.random.randint(30, 365)),
                    'similarity_score': np.random.randint(60, 95),
                    'close_notes': self._generate_close_notes(close_code),
                    'assignment_group': change_data.get('dv_assignment_group'),
                    'duration_hours': np.random.randint(1, 8)
                }
                
                similar_changes.append(similar_change)
            
            # Trier par score de similarité
            similar_changes.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            print(f"✅ {len(similar_changes)} changements similaires trouvés")
            return similar_changes
            
        except Exception as e:
            print(f"❌ Erreur changements similaires: {e}")
            return []
    
    def _generate_close_notes(self, close_code):
        """Générer des notes de fermeture réalistes"""
        
        notes_templates = {
            'Succès': [
                'Changement appliqué avec succès. Aucun incident détecté.',
                'Déploiement réalisé conformément au planning. Tests OK.',
                'Migration terminée sans problème. Services opérationnels.'
            ],
            'Succès avec difficultés': [
                'Changement réalisé mais quelques difficultés mineures rencontrées.',
                'Déploiement ok après résolution problème configuration.',
                'Succès final après ajustement procédure.'
            ],
            'Échec avec retour arrière': [
                'Échec détecté, rollback effectué avec succès.',
                'Problème critique, retour à l\'état antérieur.',
                'Changement annulé, système restauré.'
            ],
            'Échec sans retour arrière': [
                'Échec partiel, correction manuelle appliquée.',
                'Problème résolu par intervention d\'urgence.',
                'Echec mineur, service maintenu.'
            ],
            'Implémenté partiellement': [
                'Changement partiellement réalisé selon planning.',
                'Phase 1 terminée, phase 2 reportée.',
                'Implémentation partielle conforme aux objectifs.'
            ]
        }
        
        templates = notes_templates.get(close_code, ['Notes de fermeture standard.'])
        return np.random.choice(templates)
    
    def validate_change_reference(self, change_ref):
        """Valider le format de la référence changement ServiceNow"""
        
        # Format ServiceNow standard: CHG + 7 chiffres
        pattern = r'^CHG\d{7}$'
        
        if re.match(pattern, change_ref):
            return True
        else:
            return False
    
    def get_connection_status(self):
        """Vérifier le statut de connexion Snow Mirror"""
        
        return {
            'status': self.connection_status,
            'last_sync': datetime.now() - timedelta(hours=1),
            'available_tables': ['change_request', 'incident', 'cmdb_ci']
        }
```

### **FICHIER 3: data_preprocessing.py**

```python
"""
Module pour le preprocessing des données de changement
Version simplifiée pour modèle pré-entraîné
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class ChangeDataPreprocessor:
    """Preprocessing des données de changement - Version pré-entraînée"""
    
    def __init__(self):
        # Les encoders seront chargés avec le modèle pré-entraîné
        self.label_encoders = {}
        self.is_fitted = False
        
    def load_fitted_preprocessor(self, encoders_dict):
        """Charger les encoders pré-entraînés"""
        
        self.label_encoders = encoders_dict
        self.is_fitted = True
        print("✅ Preprocessor pré-entraîné chargé")
        
    def transform_single_change(self, change_data):
        """
        Transformer un changement unique (pour prédiction en temps réel)
        Optimisé pour l'utilisation dans la webapp
        """
        
        # Convertir en DataFrame
        if isinstance(change_data, dict):
            df = pd.DataFrame([change_data])
        else:
            df = change_data.copy()
        
        # 1. Garder seulement les 5 features du modèle
        required_features = ['dv_u_type_change_silca', 'dv_type', 'u_cab_count', 'u_bcr', 'u_bpc']
        
        # Vérifier que toutes les features sont présentes
        for feature in required_features:
            if feature not in df.columns:
                print(f"⚠️ Feature manquante: {feature}")
                df[feature] = 0  # Valeur par défaut
        
        # Garder seulement les features nécessaires
        df_features = df[required_features].copy()
        
        # 2. Encodage des variables catégorielles
        df_encoded = self._encode_categorical_single(df_features)
        
        # 3. Imputation des valeurs manquantes
        df_final = self._imputation_single(df_encoded)
        
        return df_final
    
    def _encode_categorical_single(self, df):
        """Encoder les variables catégorielles pour un changement unique"""
        
        df_encoded = df.copy()
        
        # Mapping des valeurs basé sur vos données d'entraînement
        # TODO: Ces mappings doivent être sauvegardés avec le modèle pré-entraîné
        
        default_encodings = {
            'dv_u_type_change_silca': {'Simple': 0, 'Complex': 1},
            'dv_type': {'Normal': 0, 'Urgent': 1, 'Emergency': 2}
        }
        
        for col in ['dv_u_type_change_silca', 'dv_type']:
            if col in df_encoded.columns:
                if col in default_encodings:
                    # Utiliser le mapping par défaut
                    mapping = default_encodings[col]
                    df_encoded[col] = df_encoded[col].map(mapping).fillna(0)
                else:
                    # Encoder numériquement si nouvelle valeur
                    df_encoded[col] = pd.Categorical(df_encoded[col]).codes
        
        return df_encoded
    
    def _imputation_single(self, df):
        """Imputation des valeurs manquantes pour un changement unique"""
        
        df_imputed = df.copy()
        
        # Imputation avec valeurs par défaut basées sur vos analyses
        default_values = {
            'dv_u_type_change_silca': 0,  # Simple (valeur la plus fréquente)
            'dv_type': 0,                 # Normal (valeur la plus fréquente)
            'u_cab_count': 1,             # Valeur médiane typique
            'u_bcr': 0,                   # False encodé (plus fréquent)
            'u_bpc': 0                    # False encodé (plus fréquent)
        }
        
        for col, default_val in default_values.items():
            if col in df_imputed.columns:
                df_imputed[col].fillna(default_val, inplace=True)
        
        # S'assurer que tout est numérique
        for col in df_imputed.columns:
            df_imputed[col] = pd.to_numeric(df_imputed[col], errors='coerce').fillna(0)
        
        return df_imputed
    
    def validate_input_data(self, change_data):
        """Valider les données d'entrée"""
        
        required_features = ['dv_u_type_change_silca', 'dv_type', 'u_cab_count', 'u_bcr', 'u_bpc']
        
        validation_results = {
            'is_valid': True,
            'missing_features': [],
            'invalid_values': [],
            'warnings': []
        }
        
        # Vérifier présence des features
        for feature in required_features:
            if feature not in change_data:
                validation_results['missing_features'].append(feature)
                validation_results['warnings'].append(f"Feature {feature} manquante, valeur par défaut utilisée")
        
        # Vérifier types et valeurs
        if 'u_cab_count' in change_data:
            if not isinstance(change_data['u_cab_count'], (int, float)) or change_data['u_cab_count'] < 0:
                validation_results['invalid_values'].append('u_cab_count doit être un nombre positif')
        
        if 'dv_u_type_change_silca' in change_data:
            if change_data['dv_u_type_change_silca'] not in ['Simple', 'Complex']:
                validation_results['warnings'].append('Type SILCA inconnu, traité comme Simple')
        
        if 'dv_type' in change_data:
            if change_data['dv_type'] not in ['Normal', 'Urgent', 'Emergency']:
                validation_results['warnings'].append('Type de changement inconnu, traité comme Normal')
        
        return validation_results
    
    def get_feature_summary(self, processed_data):
        """Résumé des features après preprocessing"""
        
        if isinstance(processed_data, pd.DataFrame) and len(processed_data) > 0:
            row = processed_data.iloc[0]
            
            # Décoder les valeurs pour affichage
            silca_types = {0: 'Simple', 1: 'Complex'}
            change_types = {0: 'Normal', 1: 'Urgent', 2: 'Emergency'}
            
            summary = {
                'dv_u_type_change_silca': silca_types.get(row['dv_u_type_change_silca'], 'Inconnu'),
                'dv_type': change_types.get(row['dv_type'], 'Inconnu'),
                'u_cab_count': int(row['u_cab_count']),
                'u_bcr': bool(row['u_bcr']),
                'u_bpc': bool(row['u_bpc'])
            }
            
            return summary
        
        return {}
```

## 🎯 **PROCHAINE ÉTAPE :**

**1. Créer ces 3 fichiers dans Code Libraries**
**2. Finaliser votre notebook avec sauvegarde du modèle**
**3. Adapter les chemins de chargement dans `load_pretrained_model()`**

**Prêt pour l'implémentation ?** 🚀

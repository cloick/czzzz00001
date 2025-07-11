L'utilisateur me demande le code complet du fichier change_risk_predictor.py avec toutes les corrections qu'on a apportées, notamment :

1. Modèle pré-entraîné avec chargement depuis managed folder
2. Utilisation de pickle au lieu de joblib
3. get_download_stream au lieu de get_reader
4. get_model_info() qui utilise les vraies données du modèle au lieu de valeurs en dur
5. Les vrais paramètres optimaux bayésiens qu'il a trouvés
6. Stockage des métadonnées lors du chargement

Je vais donner le fichier complet et corrigé.

## 📁 **FICHIER COMPLET CORRIGÉ : change_risk_predictor.py**

```python
"""
Module principal pour la prédiction des risques de changement
Modèle pré-entraîné pour chargement rapide
Version corrigée avec métadonnées dynamiques
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime

class ChangeRiskPredictor:
    """Prédicteur de risque pour changements ServiceNow - Version pré-entraînée"""
    
    def __init__(self):
        self.model = None
        self.selector = None
        self.feature_names = ['dv_u_type_change_silca', 'dv_type', 'u_cab_count', 'u_bcr', 'u_bpc']
        self.training_metadata = {}
        self.is_loaded = False
        
        # Chargement automatique au démarrage
        self.load_pretrained_model()
        
    def load_pretrained_model(self):
        """Charger le modèle depuis le managed folder"""
        
        try:
            import dataiku
            
            print("🔄 Chargement du modèle pré-entraîné...")
            
            folder = dataiku.Folder("change_risk_models")
            
            # Utilisation de get_download_stream pour Dataiku
            with folder.get_download_stream("model_final.pkl") as reader:
                model_package = pickle.load(reader)
            
            # Chargement des composants
            self.model = model_package['model']
            self.selector = model_package['selector']
            self.feature_names = model_package['feature_names']
            
            # Stockage des métadonnées
            self.training_metadata = {
                'training_date': model_package.get('training_date'),
                'performance': model_package.get('performance', {}),
                'hyperparameters': model_package.get('hyperparameters', {})
            }
            
            self.is_loaded = True
            print("✅ Modèle pré-entraîné chargé depuis managed folder")
            
        except Exception as e:
            print(f"❌ Erreur chargement modèle: {e}")
            print("💡 Assurez-vous que le modèle a été sauvegardé dans le managed folder")
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
            'model_confidence': f"Modérée ({self.training_metadata.get('performance', {}).get('recall', 'N/A')} recall)",
            'model_info': f'{type(self.model).__name__} optimisé bayésien'
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
        """Informations sur le modèle chargé - Version dynamique"""
        
        if not self.is_loaded:
            return {"status": "Modèle non chargé"}
        
        try:
            # Récupérer les hyperparamètres du modèle réel
            model_params = self.model.get_params()
            
            return {
                "status": "Modèle chargé",
                "algorithm": str(type(self.model).__name__),
                "features": {
                    "count": len(self.feature_names),
                    "names": self.feature_names
                },
                "hyperparameters": {
                    "C": model_params.get('C'),
                    "penalty": model_params.get('penalty'),
                    "solver": model_params.get('solver'),
                    "max_iter": model_params.get('max_iter'),
                    "class_weight": model_params.get('class_weight'),
                    "random_state": model_params.get('random_state')
                },
                "model_coefficients": {
                    "count": len(self.model.coef_[0]) if hasattr(self.model, 'coef_') else 0,
                    "intercept": float(self.model.intercept_[0]) if hasattr(self.model, 'intercept_') else None
                },
                "training_info": {
                    "training_date": self.training_metadata.get('training_date'),
                    "performance": self.training_metadata.get('performance', {}),
                    "saved_hyperparameters": self.training_metadata.get('hyperparameters', {})
                }
            }
            
        except Exception as e:
            return {
                "status": "Erreur récupération infos",
                "error": str(e)
            }
    
    def test_prediction(self, sample_data=None):
        """Tester la prédiction avec des données d'exemple"""
        
        if not self.is_loaded:
            return {"error": "Modèle non chargé"}
        
        # Données d'exemple si non fournies
        if sample_data is None:
            sample_data = {
                'dv_u_type_change_silca': 'Complex',
                'dv_type': 'Urgent',
                'u_cab_count': 4,
                'u_bcr': True,
                'u_bpc': False
            }
        
        try:
            # Test de prédiction
            risk_score = self.predict_risk_score(sample_data)
            detailed_analysis = self.get_detailed_analysis(sample_data)
            
            return {
                "test_status": "✅ Succès",
                "sample_data": sample_data,
                "risk_score": risk_score,
                "detailed_analysis": detailed_analysis
            }
            
        except Exception as e:
            return {
                "test_status": "❌ Échec",
                "error": str(e)
            }
    
    def get_feature_importance(self):
        """Récupérer l'importance des features (coefficients pour LogisticRegression)"""
        
        if not self.is_loaded:
            return {"error": "Modèle non chargé"}
        
        try:
            if hasattr(self.model, 'coef_'):
                # Pour LogisticRegression, utiliser les coefficients absolus
                importance_values = np.abs(self.model.coef_[0])
                
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'coefficient_abs': importance_values,
                    'coefficient': self.model.coef_[0]
                }).sort_values('coefficient_abs', ascending=False)
                
                return {
                    "status": "✅ Succès",
                    "importance": importance_df.to_dict('records')
                }
            else:
                return {"error": "Coefficients non disponibles pour ce modèle"}
                
        except Exception as e:
            return {"error": str(e)}
```

## 🎯 **AMÉLIORATIONS APPORTÉES :**

✅ **get_download_stream** pour Dataiku  
✅ **Métadonnées dynamiques** au lieu de valeurs en dur  
✅ **Stockage des infos** lors du chargement  
✅ **Méthodes de test** intégrées  
✅ **Gestion d'erreurs** robuste  
✅ **Feature importance** basée sur les vrais coefficients  

**Prêt pour les tests et la webapp !** 🚀

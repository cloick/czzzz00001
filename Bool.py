## 🔍 **TROUVONS OÙ EST LA LIGNE `change_selected` !**

**Il y a encore une référence à `change_selected` quelque part dans votre code !**

---

## 🔧 **DANS `change_risk_predictor.py`, cherchez ces lignes et supprimez-les :**

### **Dans la fonction `predict_risk_score()` :**

```python
def predict_risk_score(self, change_data):
    # ... code existant ...
    
    # Preprocessing
    change_features = self._prepare_single_change(change_data)
    
    # 🚨 SUPPRIMER CETTE LIGNE si elle existe encore :
    # change_selected = self.selector.transform(change_features)  ← SUPPRIMER !
    
    # 🚨 ET SUPPRIMER les références à change_selected :
    # risk_probability = self.model.predict_proba(change_selected)[0, 0]  ← MAUVAIS
    
    # ✅ UTILISER DIRECTEMENT change_features :
    risk_probability = self.model.predict_proba(change_features)[0, 0]  # ← BON
```

### **Dans la fonction `get_detailed_analysis()` :**

```python
def get_detailed_analysis(self, change_data):
    # ... code existant ...
    
    # 🚨 CHERCHER ET SUPPRIMER toute ligne avec change_selected
    # Comme :
    # feature_importance = self._calculate_feature_importance(change_selected)  ← SUPPRIMER
    
    # ✅ REMPLACER par :
    # feature_importance = self._calculate_feature_importance(change_features)  ← BON
```

---

## 🔧 **VERSION COMPLÈTE CORRIGÉE de `predict_risk_score()` :**

```python
def predict_risk_score(self, change_data):
    """Prédire le score de risque sans SelectKBest"""
    
    if not self.is_loaded:
        raise ValueError("❌ Modèle non chargé")
    
    try:
        # Preprocessing des données
        change_features = self._prepare_single_change(change_data)
        
        print(f"📊 Features préparées: {change_features.columns.tolist()}")
        print(f"📊 Shape: {change_features.shape}")
        print(f"📊 Données: {change_features.iloc[0].to_dict()}")
        
        # Vérification des colonnes
        expected_features = ['dv_u_type_change_silca', 'dv_type', 'u_cab_count', 'u_bcr', 'u_bpc']
        
        if list(change_features.columns) != expected_features:
            raise ValueError(f"Features incorrectes. Attendu: {expected_features}, Reçu: {list(change_features.columns)}")
        
        # ✅ PRÉDICTION DIRECTE
        risk_probability = self.model.predict_proba(change_features)[0, 0]
        risk_score = risk_probability * 100
        
        print(f"🎯 Probabilité de risque: {risk_probability}")
        print(f"🎯 Score final: {risk_score}%")
        
        return round(risk_score, 1)
        
    except Exception as e:
        print(f"❌ Erreur dans predict_risk_score: {str(e)}")
        raise
```

---

## 🔧 **VERSION COMPLÈTE CORRIGÉE de `get_detailed_analysis()` :**

```python
def get_detailed_analysis(self, change_data):
    """Analyse détaillée d'un changement"""
    
    try:
        # Calculer le score de risque
        risk_score = self.predict_risk_score(change_data)
        
        # Déterminer le niveau de risque
        if risk_score >= 70:
            risk_level = "Très Élevé"
            risk_color = "danger"
        elif risk_score >= 50:
            risk_level = "Élevé" 
            risk_color = "warning"
        elif risk_score >= 30:
            risk_level = "Modéré"
            risk_color = "info"
        else:
            risk_level = "Faible"
            risk_color = "success"
        
        # Préparer les features pour l'analyse
        change_features = self._prepare_single_change(change_data)
        
        # Facteurs de risque (analyse des features)
        risk_factors = []
        
        # Analyser chaque feature
        if 'dv_u_type_change_silca' in change_features.columns:
            silca_type = change_features['dv_u_type_change_silca'].iloc[0]
            if silca_type == 'Complex':
                risk_factors.append("Type de changement complexe")
            elif silca_type == 'Simple':
                risk_factors.append("Type de changement simple (favorable)")
        
        if 'dv_type' in change_features.columns:
            change_type = change_features['dv_type'].iloc[0]
            if change_type == 'Emergency':
                risk_factors.append("Changement d'urgence")
            elif change_type == 'Normal':
                risk_factors.append("Changement normal")
        
        if 'u_cab_count' in change_features.columns:
            cab_count = change_features['u_cab_count'].iloc[0]
            if cab_count > 3:
                risk_factors.append(f"Nombre élevé de CAB ({cab_count})")
        
        if 'u_bcr' in change_features.columns and change_features['u_bcr'].iloc[0]:
            risk_factors.append("Business Case Review requis")
            
        if 'u_bpc' in change_features.columns and change_features['u_bpc'].iloc[0]:
            risk_factors.append("Business Process Change")
        
        # Assemblage du résultat
        result = {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_factors': risk_factors,
            'features_analyzed': {
                'dv_u_type_change_silca': change_features.get('dv_u_type_change_silca', {}).iloc[0] if 'dv_u_type_change_silca' in change_features.columns else 'N/A',
                'dv_type': change_features.get('dv_type', {}).iloc[0] if 'dv_type' in change_features.columns else 'N/A',
                'u_cab_count': change_features.get('u_cab_count', {}).iloc[0] if 'u_cab_count' in change_features.columns else 'N/A',
                'u_bcr': change_features.get('u_bcr', {}).iloc[0] if 'u_bcr' in change_features.columns else 'N/A',
                'u_bpc': change_features.get('u_bpc', {}).iloc[0] if 'u_bpc' in change_features.columns else 'N/A'
            },
            'model_info': {
                'algorithm': 'LogisticRegression',
                'features_count': len(change_features.columns),
                'confidence': min(100, max(50, 100 - abs(50 - risk_score)))
            }
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Erreur dans get_detailed_analysis: {str(e)}")
        raise
```

---

## 🚀 **ACTION :**

1. **✅ Remplacez** les deux fonctions ci-dessus dans `change_risk_predictor.py`
2. **🔄 Rebuild** les Code Libraries
3. **🧪 Testez** à nouveau

**Ça devrait éliminer toutes les références à `change_selected` !** 🎯

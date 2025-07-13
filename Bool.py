## 🔍 **PROBLÈMES IDENTIFIÉS DANS VOTRE CODE :**

### **🚨 PROBLÈMES MAJEURS :**

1. **Ligne avec `#'short_description'`** : Ligne commentée qui casse la syntaxe du dictionnaire
2. **Valeurs `row.get()` peuvent retourner `NaN`** sans protection
3. **`duration_hours`** peut être `NaN` au lieu de `None`
4. **Pas de validation JSON** avant le retour

---

## 🔧 **VERSION CORRIGÉE COMPLÈTE :**

```python
def find_similar_changes(self, change_data, limit=10):
    """Changements similaires RÉELS avec durées RÉELLES - VERSION CORRIGÉE"""
    
    try:
        import dataiku
        import pandas as pd
        
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
        
        # Conversion avec durées RÉELLES et protection NaN
        similar_changes = []
        
        for idx, row in df.iterrows():
            
            # ✅ CALCUL DURÉE AVEC PROTECTION COMPLÈTE
            duration_hours = self._calculate_safe_duration(row)
            
            # ✅ CRÉATION DICTIONNAIRE AVEC PROTECTION UNIVERSELLE
            similar_change = {
                'number': self._safe_str(row.get('number', 'N/A')),
                'dv_close_code': self._safe_str(row.get('dv_close_code', 'N/A')),
                'justification': self._safe_str(row.get('justification', 'Justification non disponible')),
                'risk_impact_analysis': self._safe_str(row.get('risk_impact_analysis', 'Analyse non disponible')),
                'close_notes': self._safe_str(row.get('close_notes', 'Notes non disponibles')),
                'opened_at': self._safe_str(row.get('opened_at', '')),
                'closed_at': self._safe_str(row.get('closed_at', '')),
                'similarity_score': int(row['similarity_score']),
                'assignment_group': self._safe_str(row.get('dv_assignment_group', 'N/A')),
                'duration_hours': duration_hours,
                'data_source': 'Données réelles ServiceNow'
            }
            
            # ✅ VALIDATION FINALE JSON
            similar_change = self._ensure_json_safe(similar_change)
            similar_changes.append(similar_change)
        
        print(f"✅ {len(similar_changes)} changements similaires réels trouvés")
        return similar_changes
        
    except Exception as e:
        print(f"❌ Erreur changements similaires: {e}")
        return []

def _calculate_safe_duration(self, row):
    """Calculer la durée en heures sans risque de NaN"""
    try:
        if 'opened_at' not in row or 'closed_at' not in row:
            return None
            
        opened_val = row['opened_at']
        closed_val = row['closed_at']
        
        # Vérifier que les valeurs ne sont pas NaN
        if pd.isna(opened_val) or pd.isna(closed_val):
            return None
            
        opened = pd.to_datetime(opened_val)
        closed = pd.to_datetime(closed_val)
        
        # Vérifier que les conversions ont réussi
        if pd.isna(opened) or pd.isna(closed):
            return None
            
        # Calculer la durée
        duration_seconds = (closed - opened).total_seconds()
        
        if duration_seconds <= 0:
            return None
            
        duration_hours = round(duration_seconds / 3600, 1)
        
        # Vérification finale contre NaN/inf
        if pd.isna(duration_hours) or not pd.isfinite(duration_hours):
            return None
            
        return duration_hours
        
    except Exception as e:
        print(f"⚠️ Erreur calcul durée: {e}")
        return None

def _safe_str(self, value, default='Non disponible'):
    """Convertir toute valeur en string JSON-safe"""
    if value is None or pd.isna(value):
        return default
        
    # Vérifier les float problématiques
    if isinstance(value, float):
        if pd.isna(value) or not pd.isfinite(value):
            return default
    
    try:
        result = str(value).strip()
        return result if result else default
    except:
        return default

def _ensure_json_safe(self, data_dict):
    """S'assurer que TOUT le dictionnaire est JSON-safe"""
    import json
    
    cleaned = {}
    
    for key, value in data_dict.items():
        try:
            # Test si la valeur est sérialisable en JSON
            json.dumps(value)
            cleaned[key] = value
        except (TypeError, ValueError):
            # Si ça plante, nettoyer selon le type
            if key == 'duration_hours':
                cleaned[key] = None
            elif key == 'similarity_score':
                cleaned[key] = 0
            else:
                cleaned[key] = 'Non disponible'
    
    return cleaned
```

---

## 🔧 **FONCTIONS SUPPLÉMENTAIRES À AJOUTER :**

**Ajoutez ces 3 fonctions dans votre classe `ServiceNowConnector` :**

```python
def _calculate_safe_duration(self, row):
    """Calculer la durée en heures sans risque de NaN"""
    try:
        if 'opened_at' not in row or 'closed_at' not in row:
            return None
            
        opened_val = row['opened_at']
        closed_val = row['closed_at']
        
        if pd.isna(opened_val) or pd.isna(closed_val):
            return None
            
        opened = pd.to_datetime(opened_val)
        closed = pd.to_datetime(closed_val)
        
        if pd.isna(opened) or pd.isna(closed):
            return None
            
        duration_seconds = (closed - opened).total_seconds()
        
        if duration_seconds <= 0:
            return None
            
        duration_hours = round(duration_seconds / 3600, 1)
        
        if pd.isna(duration_hours) or not pd.isfinite(duration_hours):
            return None
            
        return duration_hours
        
    except Exception as e:
        print(f"⚠️ Erreur calcul durée: {e}")
        return None

def _safe_str(self, value, default='Non disponible'):
    """Convertir toute valeur en string JSON-safe"""
    if value is None or pd.isna(value):
        return default
        
    if isinstance(value, float):
        if pd.isna(value) or not pd.isfinite(value):
            return default
    
    try:
        result = str(value).strip()
        return result if result else default
    except:
        return default

def _ensure_json_safe(self, data_dict):
    """S'assurer que TOUT le dictionnaire est JSON-safe"""
    import json
    
    cleaned = {}
    
    for key, value in data_dict.items():
        try:
            json.dumps(value)
            cleaned[key] = value
        except (TypeError, ValueError):
            if key == 'duration_hours':
                cleaned[key] = None
            elif key == 'similarity_score':
                cleaned[key] = 0
            else:
                cleaned[key] = 'Non disponible'
    
    return cleaned
```

---

## 🎯 **CORRECTIONS APPORTÉES :**

1. **✅ Supprimé la ligne `#'short_description'` problématique**
2. **✅ Protégé TOUTES les valeurs avec `_safe_str()`**
3. **✅ Calcul durée ultra-sécurisé avec `_calculate_safe_duration()`**
4. **✅ Validation JSON finale avec `_ensure_json_safe()`**
5. **✅ Gestion complète des `NaN`, `None`, `inf`**

**Cette version élimine définitivement les erreurs JSON !** 🚀

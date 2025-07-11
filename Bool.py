L'utilisateur a raison ! Il faut d'abord tester le connector dans un notebook avant de le mettre dans les Code Libraries et dans la webapp. C'est une bonne pratique pour s'assurer que tout fonctionne correctement.

Je dois lui donner un code de test simple pour un notebook qui va :
1. Tester les imports
2. Tester la connexion aux datasets
3. Tester chaque méthode du connector
4. Vérifier que les colonnes existent bien
5. Tester avec de vraies données

Je vais faire un notebook de test complet mais simple.

## 🧪 **NOTEBOOK DE TEST DU CONNECTOR - À CRÉER AVANT TOUT**

### **CRÉER NOUVEAU NOTEBOOK : "07_Test_ServiceNow_Connector"**

```python
# ===================================================================
# TEST DU SERVICENOW CONNECTOR - AVANT MISE EN PRODUCTION
# ===================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dataiku

print("🧪 TEST DU SERVICENOW CONNECTOR")
print("=" * 50)

# ===================================================================
# ÉTAPE 1 : VÉRIFICATION DES DATASETS
# ===================================================================

print("\n📋 ÉTAPE 1 : Vérification des datasets")

# Test dataset changements
try:
    changes_dataset = dataiku.Dataset("change_request_without_empty_columns")
    changes_df = changes_dataset.get_dataframe(limit=5)
    print(f"✅ Dataset changements OK : {len(changes_df)} lignes échantillon")
    print(f"📊 Colonnes disponibles : {changes_df.columns.tolist()}")
except Exception as e:
    print(f"❌ Erreur dataset changements : {e}")

# Test dataset incidents
try:
    incidents_dataset = dataiku.Dataset("incident_filtree")
    incidents_df = incidents_dataset.get_dataframe(limit=5)
    print(f"✅ Dataset incidents OK : {len(incidents_df)} lignes échantillon")
    print(f"📊 Colonnes disponibles : {incidents_df.columns.tolist()}")
except Exception as e:
    print(f"❌ Erreur dataset incidents : {e}")

# ===================================================================
# ÉTAPE 2 : VÉRIFICATION DES COLONNES CRITIQUES
# ===================================================================

print("\n🔍 ÉTAPE 2 : Vérification des colonnes critiques")

# Colonnes nécessaires pour le modèle ML
required_columns = ['dv_u_type_change_silca', 'dv_type', 'u_cab_count', 'u_bcr', 'u_bpc']

print("📌 Colonnes requises pour le ML :")
for col in required_columns:
    if col in changes_df.columns:
        print(f"✅ {col} : présent")
        # Échantillon de valeurs
        unique_values = changes_df[col].unique()[:5]
        print(f"   Exemples : {unique_values}")
    else:
        print(f"❌ {col} : MANQUANT !")

# Autres colonnes importantes
other_columns = ['number', 'dv_assignment_group', 'dv_cmdb_ci', 'opened_at', 'closed_at', 'dv_close_code']

print("\n📌 Autres colonnes importantes :")
for col in other_columns:
    if col in changes_df.columns:
        print(f"✅ {col} : présent")
    else:
        print(f"⚠️ {col} : manquant")

# ===================================================================
# ÉTAPE 3 : TEST BASIQUE DU CONNECTOR
# ===================================================================

print("\n🔧 ÉTAPE 3 : Test basique du connector")

# Import du connector (copier-coller temporaire pour test)
import re

class ServiceNowConnectorTest:
    """Version test du connector"""
    
    def __init__(self):
        self.changes_dataset = "change_request_without_empty_columns"
        self.incidents_dataset = "incident_filtree"
        
    def validate_change_reference(self, change_ref):
        pattern = r'^CHG\d{7}$'
        return bool(re.match(pattern, change_ref))
    
    def get_change_data(self, change_ref):
        try:
            dataset = dataiku.Dataset(self.changes_dataset)
            df = dataset.get_dataframe()
            
            change_row = df[df['number'] == change_ref]
            
            if len(change_row) == 0:
                return None
            
            return change_row.iloc[0].to_dict()
            
        except Exception as e:
            print(f"Erreur : {e}")
            return None

# Initialisation du connector test
connector = ServiceNowConnectorTest()

# ===================================================================
# ÉTAPE 4 : TEST AVEC DONNÉES RÉELLES
# ===================================================================

print("\n📝 ÉTAPE 4 : Test avec données réelles")

# Prendre un vrai numéro de changement du dataset
real_change_numbers = changes_df['number'].head(3).tolist()
print(f"🎯 Numéros de test trouvés : {real_change_numbers}")

# Test de récupération
for change_ref in real_change_numbers:
    print(f"\n🔍 Test avec {change_ref} :")
    
    # Validation format
    is_valid = connector.validate_change_reference(change_ref)
    print(f"   Format valide : {is_valid}")
    
    if is_valid:
        # Récupération des données
        change_data = connector.get_change_data(change_ref)
        
        if change_data:
            print(f"   ✅ Changement récupéré")
            print(f"   📊 Features ML disponibles :")
            
            for feature in required_columns:
                value = change_data.get(feature, 'MANQUANT')
                print(f"      {feature}: {value}")
                
            # Infos supplémentaires
            print(f"   📋 Équipe : {change_data.get('dv_assignment_group', 'N/A')}")
            print(f"   📋 CI : {change_data.get('dv_cmdb_ci', 'N/A')}")
            print(f"   📋 État : {change_data.get('dv_close_code', 'N/A')}")
        else:
            print(f"   ❌ Changement non trouvé")
    else:
        print(f"   ❌ Format invalide")

# ===================================================================
# ÉTAPE 5 : TEST STATISTIQUES ÉQUIPE
# ===================================================================

print("\n👥 ÉTAPE 5 : Test statistiques équipe")

# Prendre une vraie équipe
if 'dv_assignment_group' in changes_df.columns:
    teams = changes_df['dv_assignment_group'].value_counts().head(3)
    print(f"🎯 Équipes les plus actives : {teams.index.tolist()}")
    
    test_team = teams.index[0]
    print(f"\n📊 Test avec équipe : {test_team}")
    
    # Calcul manuel pour vérification
    team_changes = changes_df[changes_df['dv_assignment_group'] == test_team]
    total_team_changes = len(team_changes)
    
    print(f"   Total changements équipe : {total_team_changes}")
    
    if 'dv_close_code' in changes_df.columns:
        success_count = len(team_changes[team_changes['dv_close_code'] == 'Succès'])
        success_rate = (success_count / total_team_changes * 100) if total_team_changes > 0 else 0
        print(f"   Taux de succès : {success_rate:.1f}%")
else:
    print("❌ Colonne dv_assignment_group manquante")

# ===================================================================
# ÉTAPE 6 : TEST INCIDENTS
# ===================================================================

print("\n🛠️ ÉTAPE 6 : Test incidents")

# Vérifier les colonnes incidents
print("📋 Colonnes table incidents :")
print(incidents_df.columns.tolist())

# Chercher colonnes CI possibles
ci_columns = [col for col in incidents_df.columns if 'ci' in col.lower() or 'config' in col.lower()]
print(f"🎯 Colonnes CI potentielles : {ci_columns}")

# Test avec un CI réel
if 'dv_cmdb_ci' in changes_df.columns:
    test_ci = changes_df['dv_cmdb_ci'].dropna().iloc[0] if len(changes_df['dv_cmdb_ci'].dropna()) > 0 else None
    if test_ci:
        print(f"🔍 Test avec CI : {test_ci}")
        
        # Recherche dans incidents
        for ci_col in ci_columns:
            if ci_col in incidents_df.columns:
                matching_incidents = incidents_df[incidents_df[ci_col] == test_ci]
                print(f"   Incidents trouvés via {ci_col} : {len(matching_incidents)}")

# ===================================================================
# ÉTAPE 7 : RÉSUMÉ DES TESTS
# ===================================================================

print("\n📊 ÉTAPE 7 : Résumé des tests")
print("=" * 50)

print("✅ CONNEXIONS :")
print(f"   - Dataset changements : {'OK' if 'changes_df' in locals() else 'ERREUR'}")
print(f"   - Dataset incidents : {'OK' if 'incidents_df' in locals() else 'ERREUR'}")

print("\n🔧 COLONNES ML :")
missing_features = [col for col in required_columns if col not in changes_df.columns]
if missing_features:
    print(f"   ❌ Colonnes manquantes : {missing_features}")
else:
    print("   ✅ Toutes les colonnes ML présentes")

print("\n📋 DONNÉES TEST :")
print(f"   - Changements testés : {len(real_change_numbers) if 'real_change_numbers' in locals() else 0}")
print(f"   - Format CHG valide : {'Vérifié' if 'connector' in locals() else 'Non testé'}")

print("\n🎯 RECOMMANDATIONS :")
if missing_features:
    print("   ⚠️ Vérifier les noms des colonnes manquantes")
    print("   ⚠️ Adapter le preprocessing si nécessaire")

print("   ✅ Si tous les tests passent → Intégrer dans Code Libraries")
print("   ✅ Si erreurs → Corriger avant intégration")

print("\n🚀 Prêt pour l'intégration !")
```

## 🎯 **ÉTAPES D'EXÉCUTION :**

### **1. CRÉER CE NOTEBOOK**
### **2. EXÉCUTER TOUTES LES CELLULES**
### **3. ANALYSER LES RÉSULTATS :**

#### **Si ✅ TOUT OK :**
- Intégrer le connector dans Code Libraries
- Tester la webapp

#### **Si ❌ ERREURS :**
- Corriger les noms de colonnes
- Adapter le code du connector
- Retester

### **4. ME MONTRER LES RÉSULTATS**

**Lancez ce test et montrez-moi ce qui sort !** 📊

**Cela nous permettra d'adapter le connector aux vrais noms de colonnes de vos datasets.** 🎯

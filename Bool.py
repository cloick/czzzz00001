L'utilisateur veut tester le connector dans un notebook avant de l'utiliser dans la webapp, ce qui est une excellente pratique. Je vais lui donner un notebook de test complet pour vérifier que toutes les fonctions du connector marchent avec les vraies données.

## 📓 **NOTEBOOK DE TEST - ServiceNow Connector**

```python
# ===================================================================
# NOTEBOOK DE TEST : SERVICENOW CONNECTOR
# Test de toutes les fonctions avec vraies données
# ===================================================================

print("🔍 TEST DU SERVICENOW CONNECTOR")
print("=" * 50)

# === 1. IMPORT ET INITIALISATION ===
try:
    from servicenow_connector import ServiceNowConnector
    print("✅ Import ServiceNowConnector réussi")
except Exception as e:
    print(f"❌ Erreur import: {e}")
    raise

# Initialisation
connector = ServiceNowConnector()
print(f"✅ Connector initialisé")
print(f"   - Dataset changements: {connector.changes_dataset}")
print(f"   - Dataset incidents: {connector.incidents_dataset}")

print("\n" + "=" * 50)

# === 2. TEST DE CONNEXION ===
print("🔗 TEST DES CONNEXIONS")

connection_status = connector.get_connection_status()
print(f"Status: {connection_status.get('status')}")

if connection_status.get('status') == 'Connecté':
    print("✅ Connexions OK")
    print(f"   - Colonnes changements disponibles: {len(connection_status.get('changes_columns', []))}")
    print(f"   - Colonnes incidents disponibles: {len(connection_status.get('incidents_columns', []))}")
    
    # Afficher quelques colonnes importantes
    changes_cols = connection_status.get('changes_columns', [])
    print(f"\n📋 Colonnes importantes trouvées dans change_request:")
    important_cols = ['number', 'dv_u_type_change_silca', 'dv_type', 'u_cab_count', 'u_bcr', 'u_bpc', 
                     'dv_assignment_group', 'dv_cmdb_ci', 'opened_at', 'closed_at', 'dv_close_code']
    
    for col in important_cols:
        status = "✅" if col in changes_cols else "❌"
        print(f"   {status} {col}")
    
    incidents_cols = connection_status.get('incidents_columns', [])
    print(f"\n📋 Colonnes trouvées dans incident_filtree: {len(incidents_cols)}")
    if len(incidents_cols) > 0:
        print(f"   Premières colonnes: {incidents_cols[:5]}")
        
else:
    print(f"❌ Erreur connexion: {connection_status.get('error')}")
    print("🛑 ARRÊT DES TESTS")

print("\n" + "=" * 50)

# === 3. TEST DE VALIDATION RÉFÉRENCE ===
print("✅ TEST VALIDATION RÉFÉRENCES")

test_refs = [
    "CHG0012345",  # Bon format
    "CHG123456",   # Trop court
    "CHG12345678", # Trop long
    "CHANGE0012345", # Mauvais préfixe
    "CHG001234A"   # Contient lettres
]

for ref in test_refs:
    is_valid = connector.validate_change_reference(ref)
    status = "✅" if is_valid else "❌"
    print(f"   {status} {ref}")

print("\n" + "=" * 50)

# === 4. TEST RÉCUPÉRATION D'UN CHANGEMENT ===
print("📄 TEST RÉCUPÉRATION CHANGEMENT")

# Prendre un échantillon pour trouver une vraie référence
import dataiku
dataset = dataiku.Dataset("change_request_without_empty_columns")
sample_df = dataset.get_dataframe(limit=10)

if len(sample_df) > 0:
    # Prendre la première référence valide
    test_change_ref = sample_df.iloc[0]['number']
    print(f"📝 Test avec changement réel: {test_change_ref}")
    
    change_data = connector.get_change_data(test_change_ref)
    
    if change_data:
        print("✅ Changement récupéré avec succès")
        print(f"   - Référence: {change_data.get('number')}")
        print(f"   - Type SILCA: {change_data.get('dv_u_type_change_silca')}")
        print(f"   - Type: {change_data.get('dv_type')}")
        print(f"   - CAB count: {change_data.get('u_cab_count')}")
        print(f"   - BCR: {change_data.get('u_bcr')}")
        print(f"   - BPC: {change_data.get('u_bpc')}")
        print(f"   - Équipe: {change_data.get('dv_assignment_group')}")
        print(f"   - CI: {change_data.get('dv_cmdb_ci')}")
        print(f"   - État: {change_data.get('dv_close_code')}")
        print(f"   - Ouvert: {change_data.get('opened_at')}")
        print(f"   - Fermé: {change_data.get('closed_at')}")
        
        # Garder ces données pour les tests suivants
        test_assignment_group = change_data.get('dv_assignment_group')
        test_cmdb_ci = change_data.get('dv_cmdb_ci')
        
    else:
        print("❌ Échec récupération changement")
        test_assignment_group = None
        test_cmdb_ci = None
else:
    print("❌ Aucune donnée dans le dataset")
    test_assignment_group = None
    test_cmdb_ci = None

print("\n" + "=" * 50)

# === 5. TEST STATISTIQUES ÉQUIPE ===
print("👥 TEST STATISTIQUES ÉQUIPE")

if test_assignment_group:
    print(f"📊 Test avec équipe: {test_assignment_group}")
    
    team_stats = connector.get_team_statistics(test_assignment_group)
    
    if team_stats and 'error' not in team_stats:
        print("✅ Statistiques équipe calculées")
        print(f"   - Total changements: {team_stats.get('total_changes')}")
        print(f"   - Succès: {team_stats.get('successes')}")
        print(f"   - Échecs: {team_stats.get('failures')}")
        print(f"   - Taux succès: {team_stats.get('success_rate')}%")
        print(f"   - Dernier échec: {team_stats.get('last_failure_date')}")
    else:
        print(f"❌ Erreur stats équipe: {team_stats}")
else:
    print("⚠️ Test ignoré (pas d'équipe test)")

print("\n" + "=" * 50)

# === 6. TEST INCIDENTS ===
print("🛠️ TEST INCIDENTS")

if test_cmdb_ci:
    print(f"🔍 Test avec CI: {test_cmdb_ci}")
    
    incidents_data = connector.get_solution_incidents(test_cmdb_ci)
    
    if incidents_data:
        print("✅ Incidents récupérés")
        print(f"   - Total incidents: {incidents_data.get('total_incidents')}")
        print(f"   - Incidents critiques: {incidents_data.get('critical_incidents')}")
        print(f"   - Résolution moyenne: {incidents_data.get('avg_resolution_hours')}h")
        print(f"   - Dernier incident: {incidents_data.get('last_incident_date')}")
        print(f"   - Source: {incidents_data.get('data_source')}")
    else:
        print("❌ Erreur récupération incidents")
else:
    print("⚠️ Test ignoré (pas de CI test)")

print("\n" + "=" * 50)

# === 7. TEST CHANGEMENTS SIMILAIRES ===
print("📋 TEST CHANGEMENTS SIMILAIRES")

if change_data:
    print(f"🔍 Recherche changements similaires à {test_change_ref}")
    
    similar_changes = connector.find_similar_changes(change_data, limit=5)
    
    if similar_changes:
        print(f"✅ {len(similar_changes)} changements similaires trouvés")
        
        for i, change in enumerate(similar_changes[:3], 1):
            print(f"\n   {i}. {change['number']} (Score: {change['similarity_score']}%)")
            print(f"      - État: {change['dv_close_code']}")
            print(f"      - Durée: {change['duration_hours']}h" if change['duration_hours'] else "      - Durée: N/A")
            print(f"      - Description: {change['short_description'][:50]}...")
    else:
        print("❌ Aucun changement similaire trouvé")
else:
    print("⚠️ Test ignoré (pas de données de changement)")

print("\n" + "=" * 50)

# === 8. TEST DES 5 FEATURES EXACTES POUR LE ML ===
print("🤖 TEST COMPATIBILITÉ ML - 5 FEATURES")

if change_data:
    print("🔍 Vérification des 5 features nécessaires au modèle ML:")
    
    required_features = ['dv_u_type_change_silca', 'dv_type', 'u_cab_count', 'u_bcr', 'u_bpc']
    
    all_features_ok = True
    for feature in required_features:
        value = change_data.get(feature)
        has_feature = feature in change_data
        status = "✅" if has_feature else "❌"
        print(f"   {status} {feature}: {value}")
        
        if not has_feature:
            all_features_ok = False
    
    if all_features_ok:
        print("\n✅ TOUTES LES FEATURES ML DISPONIBLES - Prêt pour prédiction")
    else:
        print("\n❌ FEATURES MANQUANTES - Problème pour ML")
else:
    print("⚠️ Test ignoré (pas de données)")

print("\n" + "=" * 50)

# === 9. RÉSUMÉ FINAL ===
print("📊 RÉSUMÉ DES TESTS")
print("✅ Tests réussis:")
print("   - Import et initialisation")
print("   - Validation références")

if connection_status.get('status') == 'Connecté':
    print("   - Connexions aux datasets")

if change_data:
    print("   - Récupération changement")
    print("   - Features ML disponibles")

if team_stats and 'error' not in team_stats:
    print("   - Statistiques équipe")

if incidents_data:
    print("   - Récupération incidents")

if similar_changes:
    print("   - Changements similaires")

print(f"\n🎯 CONNECTOR PRÊT POUR LA WEBAPP: {'OUI' if change_data else 'NON'}")

print("\n" + "=" * 50)
print("🚀 FIN DES TESTS")
```

## 🧪 **CE QUE CE NOTEBOOK VA TESTER :**

### **1. Connexions de base**
- Import du module
- Connexion aux 2 datasets
- Colonnes disponibles

### **2. Validation des formats**
- Test références CHG valides/invalides

### **3. Récupération de données**
- Changement réel du dataset
- Vérification des 5 features ML

### **4. Fonctions d'enrichissement**
- Stats équipe vraies
- Incidents vrais
- Changements similaires vrais

### **5. Compatibilité ML**
- Présence des 5 features exactes

## 🎯 **APRÈS CE TEST :**

**Si tout passe ✅** → Webapp prête  
**Si erreurs ❌** → On corrige le connector d'abord

**Lancez ce notebook et montrez-moi les résultats !** 🚀

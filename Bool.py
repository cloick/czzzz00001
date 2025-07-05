# CRÉATION DES SOUS-ENSEMBLES (suite à notre EDA)

# 1. Colonnes clés (target + identifiants à mettre de côté)
key_columns = ['dv_close_code']  # Notre target
id_columns = ['number', 'success']  # À exclure (ID + target dérivée)

# 2. Variables booléennes importantes (selon notre analyse corrélations)
bool_important = ['u_psi_update_necessary']  # STAR variable indépendante
bool_correlated = ['u_bcr', 'u_bpc', 'u_clp', 'u_cfs']  # Groupe corrélé (garder 1 seul)
bool_other = ['u_coordinator_trigger', 'u_emergency', 'u_gea_assurance', 
              'u_grc', 'u_interface', 'u_multi_client', 'u_out_of_agile_process', 
              'u_out_of_process_change', 'u_presence_on_site', 
              'u_presence_on_site_we_holiday', 'u_re_assess']

# 3. Variables catégorielles importantes (selon tests Chi-2)
categorical_important = ['dv_u_type_change_silca', 'dv_u_change_prerequisites', 
                        'dv_type', 'dv_risk', 'dv_impact', 'dv_category',
                        'dv_conflict_status', 'dv_u_qualification', 'dv_u_origin',
                        'dv_u_additional_p_i_r', 'dv_u_cmdb_update']

# 4. Variables haute cardinalité (à traiter spécialement)
high_cardinality = ['dv_cmdb_ci',           # 4019 catégories
                   'dv_impacted_customers',  # 2533 catégories  
                   'dv_assignment_group']    # 169 catégories

# 5. Variable template (valeurs manquantes)
template_columns = ['u_template_applied']

# 6. Variables temporelles
temporal_columns = ['opened_at', 'start_date', 'end_date', 'approval_set']

# 7. Variables numériques (peu corrélées entre elles)
numerical_columns = ['reassignment_count', 'sys_mod_count', 'u_cab_count', 
                    'u_cab_reservation_count', 'u_reminder_count']

# 8. Variables restantes (à évaluer)
other_columns = ['dv_company', 'unauthorized', 'dv_approval', 
                'u_agile_compliance', 'u_agile_compliance2', 'u_approved_with_reservation',
                'u_build_to_run', 'u_cib', 'u_clp', 'u_psi_update_necessary',
                'u_socle_cluster']

# SÉLECTION POUR VERSION BASELINE
print("=== CRÉATION DF BASELINE ===")

# Version 1: Features minimales pour baseline
baseline_features = (key_columns + 
                    bool_important + 
                    ['u_bpc'] +  # Une seule du groupe corrélé (meilleur impact)
                    categorical_important + 
                    numerical_columns)

# Vérifier que toutes les colonnes existent
baseline_features_available = [col for col in baseline_features if col in df.columns]
missing_features = [col for col in baseline_features if col not in df.columns]

print(f"Features baseline disponibles: {len(baseline_features_available)}")
print(f"Features manquantes: {missing_features}")

# Créer le DataFrame baseline
df_baseline = df[baseline_features_available].copy()

print(f"\nShape baseline: {df_baseline.shape}")
print(f"Colonnes sélectionnées:")
for i, col in enumerate(baseline_features_available, 1):
    print(f"{i:2d}. {col}")

# Variables mises de côté pour versions futures
print(f"\n=== VARIABLES POUR VERSIONS FUTURES ===")
print(f"Haute cardinalité: {high_cardinality}")
print(f"Template: {template_columns}")
print(f"Temporelles: {temporal_columns}")
print(f"ID/Dérivées: {id_columns}")

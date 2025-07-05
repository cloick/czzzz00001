# CRÉATION DES SOUS-ENSEMBLES (basée sur notre VRAIE analyse EDA)

# 1. Colonnes clés et à exclure
target_column = ['dv_close_code']  # Notre target
exclude_columns = ['number', 'success']  # ID + target dérivée créée

# 2. Variables booléennes importantes (selon notre analyse corrélations)
bool_star = ['u_psi_update_necessary']  # STAR variable (impact 39.6%)
bool_correlated_group = ['u_bcr', 'u_bpc', 'u_clp', 'u_cfs']  # Groupe corrélé (0.7-0.8)
bool_independent = ['u_coordinator_trigger']  # Indépendante
bool_low_impact = ['cab_required', 'u_approved_with_reservation', 'u_gea_assurance', 
                   'u_grc', 'u_interface', 'u_multi_client', 'u_out_of_agile_process', 
                   'u_out_of_process_change']  # Impact faible dans nos analyses

# 3. Variables catégorielles importantes (selon nos tests Chi-2)
categorical_top = ['dv_u_type_change_silca',      # Cramér's V: 0.084
                   'dv_u_change_prerequisites',   # Cramér's V: 0.080  
                   'dv_type',                     # Cramér's V: 0.075
                   'dv_impact',                   # Cramér's V: 0.041
                   'dv_category']                 # Cramér's V: 0.037

categorical_other = ['dv_risk', 'dv_conflict_status', 'dv_u_qualification', 
                    'dv_u_origin', 'dv_u_additional_p_i_r', 'dv_u_cmdb_update',
                    'dv_company', 'dv_approval']

# 4. Variables haute cardinalité (identifiées dans notre analyse)
high_cardinality = ['dv_cmdb_ci',              # 4019 catégories
                   'dv_u_impacted_customers',  # 2533 catégories  
                   'dv_assignment_group']      # 169 catégories

# 5. Variable template (11% manquants, faible impact)
template_columns = ['u_template_applied']

# 6. Variables temporelles
temporal_columns = ['opened_at', 'start_date', 'end_date', 'approval_set']

# 7. Variables numériques (corrélations faibles entre elles)
numerical_columns = ['reassignment_count', 'sys_mod_count', 'u_cab_count', 
                    'u_cab_reservation_count', 'u_reminder_count']

# 8. Variables restantes
other_columns = ['u_cxi', 'u_indice_gea_epargne', 'u_cib', 'u_socle_cluster']

# CRÉATION DF BASELINE (version minimale)
print("=== CRÉATION DF BASELINE ===")

baseline_features = (target_column + 
                    bool_star + 
                    ['u_bpc'] +  # Une seule du groupe corrélé (meilleur dans analyse)
                    categorical_top + 
                    numerical_columns)

print(f"Features baseline sélectionnées:")
for i, col in enumerate(baseline_features, 1):
    print(f"{i:2d}. {col}")

# Créer le DataFrame baseline
df_baseline = df[baseline_features].copy()
print(f"\nShape baseline: {df_baseline.shape}")

print(f"\n=== VARIABLES POUR VERSIONS FUTURES ===")
print(f"Haute cardinalité: {high_cardinality}")
print(f"Template: {template_columns}")  
print(f"Temporelles: {temporal_columns}")
print(f"Groupe corrélé restant: {bool_correlated_group[:-1]}")  # sans u_bpc

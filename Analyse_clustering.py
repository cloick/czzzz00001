# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import numpy as np
from collections import Counter
import re

print("🔍 ANALYSE DÉTAILLÉE DES CLUSTERS POUR VALIDATION MÉTIER")
print("="*70)

# Read recipe inputs
dataset = dataiku.Dataset("incident_with_clusters_intensive")
df = dataset.get_dataframe()

print(f"Dataset chargé : {len(df)} lignes")
print(f"Clusters analysés : {df['cluster'].nunique() - (1 if -1 in df['cluster'].values else 0)}")

# Statistiques générales
clusters = df['cluster'].unique()
n_clusters = len([c for c in clusters if c != -1])
n_noise = (df['cluster'] == -1).sum()

print(f"Points de bruit : {n_noise} ({n_noise/len(df)*100:.1f}%)")

# Analyse complète par cluster
print(f"\n📊 ANALYSE DÉTAILLÉE PAR CLUSTER")
print("="*70)

cluster_analysis = []

for cluster in sorted([c for c in clusters if c != -1]):
    cluster_data = df[df['cluster'] == cluster]
    
    print(f"\n🎯 CLUSTER {cluster} - {len(cluster_data)} tickets ({len(cluster_data)/len(df)*100:.1f}%)")
    print("-" * 50)
    
    analysis = {
        'cluster_id': cluster,
        'n_tickets': len(cluster_data),
        'pourcentage': len(cluster_data)/len(df)*100,
        'n_fiables': 0,
        'cause_dominante': 'À déterminer',
        'causes_repartition': {},
        'groupe_dominant': '',
        'service_dominant': '',
        'cat1_dominant': '',
        'cat2_dominant': '',
        'priorite_dominante': '',
        'mots_cles_frequents': [],
        'exemples_tickets': [],
        'coherence_score': 0
    }
    
    # 1. ANALYSE DES TICKETS FIABLES
    if 'est_fiable' in cluster_data.columns:
        fiables = cluster_data[cluster_data['est_fiable']]
        analysis['n_fiables'] = len(fiables)
        
        print(f"📋 Tickets fiables : {len(fiables)}/{len(cluster_data)} ({len(fiables)/len(cluster_data)*100:.1f}%)")
        
        if len(fiables) > 0 and 'cause' in fiables.columns:
            # Distribution des causes
            cause_counts = fiables['cause'].value_counts()
            analysis['causes_repartition'] = cause_counts.to_dict()
            
            if len(cause_counts) > 0:
                analysis['cause_dominante'] = cause_counts.index[0]
                confidence = cause_counts.iloc[0] / len(fiables)
                
                print(f"🎯 Cause dominante : {analysis['cause_dominante']} ({confidence:.1%} des tickets fiables)")
                
                # Afficher toutes les causes présentes
                print(f"📈 Répartition des causes :")
                for cause, count in cause_counts.items():
                    pct = count / len(fiables) * 100
                    print(f"   • {cause}: {count} tickets ({pct:.1f}%)")
    
    # 2. ANALYSE DES VARIABLES CATÉGORIELLES
    print(f"\n🏢 Variables dominantes :")
    
    cat_vars = {
        'Groupe affecté': 'groupe_dominant',
        'Service métier': 'service_dominant', 
        'Cat1': 'cat1_dominant',
        'Cat2': 'cat2_dominant',
        'Priorité': 'priorite_dominante'
    }
    
    for col, key in cat_vars.items():
        if col in cluster_data.columns:
            top_values = cluster_data[col].value_counts().head(3)
            if len(top_values) > 0:
                analysis[key] = top_values.index[0]
                print(f"   • {col}: {top_values.index[0]} ({top_values.iloc[0]} tickets - {top_values.iloc[0]/len(cluster_data)*100:.1f}%)")
                
                # Afficher top 3 si pertinent
                if len(top_values) > 1:
                    others = [f"{val} ({count})" for val, count in top_values.iloc[1:3].items()]
                    print(f"     Autres: {', '.join(others)}")
    
    # 3. ANALYSE TEXTUELLE DES NOTES DE RÉSOLUTION
    if 'Notes de résolution' in cluster_data.columns:
        print(f"\n📝 Analyse textuelle :")
        
        # Nettoyer et extraire les mots-clés
        all_text = ' '.join(cluster_data['Notes de résolution'].fillna('').astype(str))
        
        # Nettoyer le texte
        text_clean = re.sub(r'[^\w\s]', ' ', all_text.lower())
        text_clean = re.sub(r'\s+', ' ', text_clean)
        
        # Mots vides à ignorer
        stop_words = {
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'mais', 'pour', 'par', 'avec', 'sur', 'dans', 'en', 'à', 'il', 'elle', 'ce', 'cette', 'qui', 'que', 'dont', 'où', 'est', 'sont', 'était', 'étaient', 'sera', 'seront', 'avoir', 'être', 'faire', 'dire', 'aller', 'voir', 'savoir', 'pouvoir', 'falloir', 'vouloir', 'venir', 'prendre', 'donner', 'mettre', 'tenir', 'partir', 'porter', 'montrer', 'demander', 'passer', 'suivre', 'sortir', 'entrer', 'rester', 'tomber', 'arriver', 'répondre', 'ouvrir', 'fermer', 'commencer', 'finir', 'continuer', 'arrêter', 'changer', 'utiliser', 'travailler', 'jouer', 'gagner', 'perdre', 'acheter', 'vendre', 'payer', 'coûter', 'valoir', 'compter', 'mesurer', 'peser', 'couper', 'casser', 'réparer', 'construire', 'détruire', 'créer', 'produire', 'fabriquer', 'publier', 'écrire', 'lire', 'écouter', 'regarder', 'chercher', 'trouver', 'découvrir', 'apprendre', 'enseigner', 'expliquer', 'comprendre', 'connaître', 'reconnaître', 'se', 'me', 'te', 'nous', 'vous', 'lui', 'leur', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses', 'notre', 'nos', 'votre', 'vos', 'leur', 'leurs', 'ticket', 'incident', 'problème', 'issue', 'erreur', 'bug', 'souci', 'pb'
        }
        
        # Extraire mots significatifs
        words = [word for word in text_clean.split() if len(word) > 3 and word not in stop_words]
        word_freq = Counter(words)
        
        # Top 10 mots-clés
        top_words = word_freq.most_common(10)
        analysis['mots_cles_frequents'] = [f"{word} ({count})" for word, count in top_words]
        
        print(f"   🔑 Mots-clés fréquents :")
        for word, count in top_words[:8]:  # Limiter l'affichage
            print(f"      • {word}: {count} occurrences")
    
    # 4. TOUS LES TICKETS DU CLUSTER
    print(f"\n📄 Tous les tickets du cluster :")
    
    # TOUS les numéros de tickets du cluster
    if 'N° INC' in cluster_data.columns:
        tous_tickets = cluster_data['N° INC'].tolist()
        analysis['exemples_tickets'] = tous_tickets
        print(f"   📋 {len(tous_tickets)} tickets : {', '.join(map(str, tous_tickets[:10]))}")
        if len(tous_tickets) > 10:
            print(f"   📋 ... et {len(tous_tickets)-10} autres tickets")
    else:
        # Si pas de colonne N° INC, utiliser les index
        tous_tickets = cluster_data.index.tolist()
        analysis['exemples_tickets'] = tous_tickets
        print(f"   📋 {len(tous_tickets)} tickets (index) : {', '.join(map(str, tous_tickets[:10]))}")
    
    # Échantillon représentatif pour affichage détaillé
    exemples_display = []
    if 'est_fiable' in cluster_data.columns:
        # D'abord les tickets fiables
        fiables_sample = cluster_data[cluster_data['est_fiable']].head(3)
        exemples_display.extend(fiables_sample.index.tolist())
        
        # Puis des tickets non fiables
        non_fiables = cluster_data[~cluster_data['est_fiable']]
        if len(non_fiables) > 0:
            non_fiables_sample = non_fiables.sample(min(2, len(non_fiables)), random_state=42)
            exemples_display.extend(non_fiables_sample.index.tolist())
    else:
        # Échantillon aléatoire
        sample_size = min(5, len(cluster_data))
        sample_tickets = cluster_data.sample(sample_size, random_state=42)
        exemples_display.extend(sample_tickets.index.tolist())
    
    print(f"\n📄 Échantillon détaillé (5 premiers) :")
    
    for i, idx in enumerate(exemples_display[:5], 1):
        ticket = df.loc[idx]
        fiable_str = " (FIABLE)" if ticket.get('est_fiable', False) else ""
        cause_str = f" | Cause: {ticket.get('cause', 'N/A')}" if 'cause' in ticket else ""
        
        print(f"   {i}. Ticket {ticket.get('N° INC', idx)}{fiable_str}{cause_str}")
        print(f"      Groupe: {ticket.get('Groupe affecté', 'N/A')} | Service: {ticket.get('Service métier', 'N/A')}")
        
        if 'Notes de résolution' in ticket:
            note = str(ticket['Notes de résolution'])[:150]
            print(f"      Note: {note}{'...' if len(str(ticket['Notes de résolution'])) > 150 else ''}")
        print()
    
    # 5. SCORE DE COHÉRENCE DU CLUSTER (explication détaillée)
    coherence_factors = []
    coherence_details = []
    
    # Cohérence des causes (si tickets fiables)
    if analysis['n_fiables'] > 0 and analysis['causes_repartition']:
        max_cause_pct = max(analysis['causes_repartition'].values()) / analysis['n_fiables']
        coherence_factors.append(max_cause_pct)
        coherence_details.append(f"Cohérence causes: {max_cause_pct:.2f}")
    
    # Cohérence du groupe affecté
    if analysis['groupe_dominant']:
        groupe_pct = cluster_data[cluster_data['Groupe affecté'] == analysis['groupe_dominant']].shape[0] / len(cluster_data)
        coherence_factors.append(groupe_pct)
        coherence_details.append(f"Cohérence groupe: {groupe_pct:.2f}")
    
    # Cohérence du service
    if analysis['service_dominant']:
        service_pct = cluster_data[cluster_data['Service métier'] == analysis['service_dominant']].shape[0] / len(cluster_data)
        coherence_factors.append(service_pct)
        coherence_details.append(f"Cohérence service: {service_pct:.2f}")
    
    if coherence_factors:
        analysis['coherence_score'] = np.mean(coherence_factors)
        
        print(f"📊 Score de cohérence : {analysis['coherence_score']:.2f}")
        print(f"   📝 Détail : {' | '.join(coherence_details)}")
        print(f"   📖 Signification :")
        print(f"      • >0.70 = Cluster très cohérent (tickets très similaires)")
        print(f"      • 0.50-0.70 = Cluster moyennement cohérent") 
        print(f"      • <0.50 = Cluster peu cohérent (tickets disparates)")
        
        if analysis['coherence_score'] > 0.7:
            print(f"   ✅ Cluster très cohérent - Validation recommandée")
        elif analysis['coherence_score'] > 0.5:
            print(f"   🟡 Cluster moyennement cohérent - Validation conseillée")
        else:
            print(f"   ⚠️  Cluster peu cohérent - Validation OBLIGATOIRE")
    
    cluster_analysis.append(analysis)

# RÉSUMÉ GLOBAL ET RECOMMANDATIONS
print(f"\n" + "="*70)
print(f"📈 RÉSUMÉ GLOBAL DE L'ANALYSE")
print("="*70)

# DataFrame pour analyse
df_analysis = pd.DataFrame(cluster_analysis)

# Statistiques globales
total_fiables = df_analysis['n_fiables'].sum()
clusters_avec_cause = len(df_analysis[df_analysis['cause_dominante'] != 'À déterminer'])
clusters_coherents = len(df_analysis[df_analysis['coherence_score'] > 0.7])

print(f"🎯 Couverture des tickets fiables : {total_fiables}/{df['est_fiable'].sum() if 'est_fiable' in df.columns else 'N/A'}")
print(f"🎯 Clusters avec cause identifiée : {clusters_avec_cause}/{len(df_analysis)} ({clusters_avec_cause/len(df_analysis)*100:.1f}%)")
print(f"🎯 Clusters cohérents (score > 0.7) : {clusters_coherents}/{len(df_analysis)} ({clusters_coherents/len(df_analysis)*100:.1f}%)")

# Distribution des causes
if 'est_fiable' in df.columns:
    print(f"\n📊 MAPPING CLUSTERS → CAUSES :")
    
    causes_mapping = {}
    for _, row in df_analysis.iterrows():
        cause = row['cause_dominante']
        if cause != 'À déterminer':
            if cause not in causes_mapping:
                causes_mapping[cause] = []
            causes_mapping[cause].append(row['cluster_id'])
    
    for cause, clusters_list in causes_mapping.items():
        clusters_str = ', '.join(map(str, sorted(clusters_list)))
        print(f"   📋 {cause}: Clusters {clusters_str} ({len(clusters_list)} clusters)")
    
    clusters_a_determiner = df_analysis[df_analysis['cause_dominante'] == 'À déterminer']['cluster_id'].tolist()
    if clusters_a_determiner:
        clusters_str = ', '.join(map(str, sorted(clusters_a_determiner)))
        print(f"   ❓ À déterminer: Clusters {clusters_str} ({len(clusters_a_determiner)} clusters)")

# RECOMMANDATIONS POUR VALIDATION MÉTIER
print(f"\n💡 RECOMMANDATIONS POUR VALIDATION MÉTIER :")
print("-" * 40)

# Clusters prioritaires à valider
clusters_prioritaires = df_analysis.nlargest(5, 'n_tickets')
print(f"🔍 Clusters prioritaires (plus gros volumes) :")
for _, cluster in clusters_prioritaires.iterrows():
    print(f"   • Cluster {cluster['cluster_id']}: {cluster['n_tickets']} tickets - {cluster['cause_dominante']}")

# Clusters incohérents à examiner
clusters_incoherents = df_analysis[df_analysis['coherence_score'] < 0.5]
if len(clusters_incoherents) > 0:
    print(f"\n⚠️  Clusters peu cohérents à examiner :")
    for _, cluster in clusters_incoherents.iterrows():
        print(f"   • Cluster {cluster['cluster_id']}: Score {cluster['coherence_score']:.2f} - {cluster['n_tickets']} tickets")

# Clusters sans cause à identifier
clusters_sans_cause = df_analysis[df_analysis['cause_dominante'] == 'À déterminer']
if len(clusters_sans_cause) > 0:
    print(f"\n❓ Clusters sans cause identifiée :")
    for _, cluster in clusters_sans_cause.iterrows():
        print(f"   • Cluster {cluster['cluster_id']}: {cluster['n_tickets']} tickets - Mots-clés: {', '.join(cluster['mots_cles_frequents'][:3])}")

# Sauvegarder l'analyse
print(f"\n💾 SAUVEGARDE DE L'ANALYSE...")

# Dataset détaillé pour validation métier
df_validation = df.copy()
df_validation = df_validation.merge(
    df_analysis[['cluster_id', 'cause_dominante', 'coherence_score']].rename(columns={'cluster_id': 'cluster'}),
    on='cluster',
    how='left'
)

output_validation = dataiku.Dataset("cluster_analysis_validation")
output_validation.write_with_schema(df_validation)

# Rapport de synthèse
output_rapport = dataiku.Dataset("cluster_synthesis_report")
output_rapport.write_with_schema(df_analysis)

print(f"✅ Datasets créés :")
print(f"   📊 cluster_analysis_validation - Dataset enrichi pour validation")
print(f"   📋 cluster_synthesis_report - Rapport de synthèse")

print(f"\n" + "="*70)
print(f"🎉 ANALYSE TERMINÉE ! Prêt pour la validation métier")
print("="*70)

## 🔧 **MODIFICATIONS POUR AFFICHER JUSTIFICATION + RISK_IMPACT_ANALYSIS + CLOSE_NOTES**

**Il faut modifier à 2 endroits : Backend Python + Frontend JavaScript**

---

## 📝 **1. MODIFICATION BACKEND - Dans `servicenow_connector.py` :**

### **Lignes 305-315, remplacez :**

```python
# ❌ ANCIEN (avec description tronquée)
'short_description': row.get('short_description', 'Description non disponible')[:100] + '...',

# ✅ NOUVEAU (avec les 3 champs complets)
'justification': row.get('justification', 'Justification non disponible'),
'risk_impact_analysis': row.get('risk_impact_analysis', 'Analyse non disponible'),
'close_notes': row.get('close_notes', 'Notes non disponibles'),
```

### **Modification complète de la section :**

```python
similar_change = {
    'number': row.get('number', 'N/A'),
    'dv_close_code': row.get('dv_close_code', 'N/A'),
    # ✅ NOUVEAUX CHAMPS (complets, non tronqués)
    'justification': row.get('justification', 'Justification non disponible'),
    'risk_impact_analysis': row.get('risk_impact_analysis', 'Analyse non disponible'),
    'close_notes': row.get('close_notes', 'Notes non disponibles'),
    'opened_at': row.get('opened_at'),
    'closed_at': row.get('closed_at'),
    'similarity_score': int(row['similarity_score']),
    'assignment_group': row.get('dv_assignment_group', 'N/A'),
    'duration_hours': duration_hours,
    'data_source': 'Données réelles ServiceNow'
}
```

---

## 🎨 **2. MODIFICATION FRONTEND - Dans le JavaScript :**

### **Dans la fonction `displaySimilarChanges()`, remplacez :**

```javascript
// ❌ ANCIEN AFFICHAGE
changesHtml += 
    '<div class="' + className + '">' +
        '<p>' +
            '<strong>' + icon + ' ' + (change.number || 'N/A') + ' - ' + closeCode + '</strong><br>' +
            '<small>' + (change.short_description || 'Pas de description').substring(0, 100) + '...</small><br>' +
            '<small style="color: #666;">' +
                'Similarité: ' + (change.similarity_score || 0) + '%' + durationText +
            '</small>' +
        '</p>' +
    '</div>';

// ✅ NOUVEL AFFICHAGE COMPLET
changesHtml += 
    '<div class="' + className + '">' +
        '<div style="margin-bottom: 0.5rem;">' +
            '<strong>' + icon + ' ' + (change.number || 'N/A') + ' - ' + closeCode + '</strong>' +
            '<span style="float: right; color: #666; font-size: 0.9rem;">' +
                'Similarité: ' + (change.similarity_score || 0) + '%' + durationText +
            '</span>' +
        '</div>' +
        
        '<div style="margin-bottom: 0.5rem;">' +
            '<strong style="color: #0066cc;">📋 Justification:</strong><br>' +
            '<div style="background: #f8f9fa; padding: 0.5rem; border-radius: 4px; margin-bottom: 0.5rem;">' +
                (change.justification || 'Non disponible') +
            '</div>' +
        '</div>' +
        
        '<div style="margin-bottom: 0.5rem;">' +
            '<strong style="color: #ff6600;">⚠️ Analyse Risque/Impact:</strong><br>' +
            '<div style="background: #fff3cd; padding: 0.5rem; border-radius: 4px; margin-bottom: 0.5rem;">' +
                (change.risk_impact_analysis || 'Non disponible') +
            '</div>' +
        '</div>' +
        
        '<div>' +
            '<strong style="color: #28a745;">📝 Notes de fermeture:</strong><br>' +
            '<div style="background: #d4edda; padding: 0.5rem; border-radius: 4px;">' +
                (change.close_notes || 'Non disponibles') +
            '</div>' +
        '</div>' +
    '</div>';
```

---

## 🎨 **VERSION PLUS LISIBLE AVEC ACCORDÉON (OPTIONNEL) :**

### **Si vous voulez un affichage pliable/dépliable :**

```javascript
changesHtml += 
    '<div class="' + className + '" style="margin-bottom: 1rem;">' +
        // En-tête toujours visible
        '<div style="cursor: pointer; background: #f8f9fa; padding: 0.75rem; border-radius: 6px; border-left: 4px solid #007bff;" onclick="toggleDetails(this)">' +
            '<strong>' + icon + ' ' + (change.number || 'N/A') + ' - ' + closeCode + '</strong>' +
            '<span style="float: right; color: #666;">' +
                'Similarité: ' + (change.similarity_score || 0) + '%' + durationText + ' <i class="fas fa-chevron-down"></i>' +
            '</span>' +
        '</div>' +
        
        // Détails pliables
        '<div class="change-details" style="display: none; padding: 1rem; border: 1px solid #dee2e6; border-top: none; border-radius: 0 0 6px 6px;">' +
            '<div style="margin-bottom: 1rem;">' +
                '<h6 style="color: #0066cc; margin: 0 0 0.5rem 0;"><i class="fas fa-clipboard-list"></i> Justification</h6>' +
                '<div style="background: #f8f9fa; padding: 0.75rem; border-radius: 4px; border-left: 3px solid #0066cc;">' +
                    (change.justification || 'Non disponible') +
                '</div>' +
            '</div>' +
            
            '<div style="margin-bottom: 1rem;">' +
                '<h6 style="color: #ff6600; margin: 0 0 0.5rem 0;"><i class="fas fa-exclamation-triangle"></i> Analyse Risque/Impact</h6>' +
                '<div style="background: #fff3cd; padding: 0.75rem; border-radius: 4px; border-left: 3px solid #ff6600;">' +
                    (change.risk_impact_analysis || 'Non disponible') +
                '</div>' +
            '</div>' +
            
            '<div>' +
                '<h6 style="color: #28a745; margin: 0 0 0.5rem 0;"><i class="fas fa-check-circle"></i> Notes de fermeture</h6>' +
                '<div style="background: #d4edda; padding: 0.75rem; border-radius: 4px; border-left: 3px solid #28a745;">' +
                    (change.close_notes || 'Non disponibles') +
                '</div>' +
            '</div>' +
        '</div>' +
    '</div>';
```

### **Et ajoutez cette fonction JavaScript :**

```javascript
function toggleDetails(element) {
    const details = element.nextElementSibling;
    const icon = element.querySelector('i.fa-chevron-down, i.fa-chevron-up');
    
    if (details.style.display === 'none') {
        details.style.display = 'block';
        if (icon) icon.className = 'fas fa-chevron-up';
    } else {
        details.style.display = 'none';
        if (icon) icon.className = 'fas fa-chevron-down';
    }
}
```

---

## 🚀 **ÉTAPES D'APPLICATION :**

1. **✅ Modifiez** `servicenow_connector.py` (lignes 305-315)
2. **✅ Modifiez** la fonction `displaySimilarChanges()` dans le JavaScript
3. **🔄 Rebuild** les Code Libraries
4. **🧪 Testez** l'onglet "Changements similaires"

**Maintenant vous verrez les 3 champs complets et non tronqués !** 📋

Je vois les erreurs dans les images :

1. **Image 1** : Erreur de diagnostic : "apiCall is not defined"
2. **Image 2** : Plusieurs erreurs :
   - Modèle ML : Erreur "failed to construct 'URL': Invalid URL"
   - Connexions ServiceNow : Erreur "failed to construct 'URL': Invalid URL" 
   - Performance Modèle : "Erreur de chargement"
   - Quand on clique "Analyser" : "Erreur d'analyse: Failed to construct 'URL': Invalid URL"

Le problème principal semble être que la fonction `apiCall` n'arrive pas à construire les URLs pour les appels API. Cela vient probablement de la fonction `getWebAppBackendUrl()` qui n'est pas définie ou qui ne fonctionne pas correctement.

Dans une webapp Dataiku standard, il faut utiliser une méthode différente pour faire les appels API vers le backend Python. Je vais corriger le JavaScript pour résoudre ce problème.

## 🚨 **PROBLÈME IDENTIFIÉ : Erreur d'URL dans apiCall**

**L'erreur "Failed to construct 'URL': Invalid URL" vient de la fonction `getWebAppBackendUrl()` qui n'existe pas dans l'environnement Dataiku standard.**

## 🔧 **CORRECTION JAVASCRIPT - Fonction apiCall à remplacer :**

```javascript
// ===================================================================
// API CALLS - VERSION CORRIGÉE POUR DATAIKU
// ===================================================================

async function apiCall(endpoint, params = {}) {
    try {
        console.log(`🔗 Appel API: ${endpoint}`, params);
        
        // Construction de l'URL pour webapp Dataiku standard
        let url = `./${endpoint}`;
        
        // Ajouter les paramètres GET
        if (Object.keys(params).length > 0) {
            const searchParams = new URLSearchParams();
            Object.keys(params).forEach(key => {
                if (params[key] !== null && params[key] !== undefined) {
                    searchParams.append(key, params[key]);
                }
            });
            url += `?${searchParams.toString()}`;
        }
        
        console.log(`📡 URL: ${url}`);
        
        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            const text = await response.text();
            throw new Error(`Réponse non-JSON reçue: ${text.substring(0, 200)}`);
        }
        
        const data = await response.json();
        console.log(`📥 Réponse ${endpoint}:`, data);
        
        if (data.status === 'error') {
            throw new Error(data.message || 'Erreur inconnue');
        }
        
        return data;
        
    } catch (error) {
        console.error(`❌ Erreur API ${endpoint}:`, error);
        throw error;
    }
}
```

## 🔧 **ALTERNATIVE - Si ça ne marche toujours pas :**

```javascript
// ===================================================================
// API CALLS - VERSION ALTERNATIVE AVEC CHEMINS ABSOLUS
// ===================================================================

async function apiCall(endpoint, params = {}) {
    try {
        console.log(`🔗 Appel API: ${endpoint}`, params);
        
        // Récupérer l'URL de base de la webapp
        const baseUrl = window.location.origin + window.location.pathname;
        
        // Construire l'URL complète
        let url = baseUrl;
        if (!url.endsWith('/')) {
            url += '/';
        }
        url += endpoint;
        
        // Ajouter les paramètres GET
        if (Object.keys(params).length > 0) {
            const searchParams = new URLSearchParams();
            Object.keys(params).forEach(key => {
                if (params[key] !== null && params[key] !== undefined) {
                    searchParams.append(key, String(params[key]));
                }
            });
            url += `?${searchParams.toString()}`;
        }
        
        console.log(`📡 URL complète: ${url}`);
        
        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            credentials: 'same-origin'  // Important pour Dataiku
        });
        
        console.log(`📡 Status: ${response.status}`);
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }
        
        const data = await response.json();
        console.log(`📥 Réponse ${endpoint}:`, data);
        
        if (data.status === 'error') {
            throw new Error(data.message || 'Erreur inconnue');
        }
        
        return data;
        
    } catch (error) {
        console.error(`❌ Erreur API ${endpoint}:`, error);
        
        // Debug supplémentaire
        console.error(`🔍 URL actuelle: ${window.location.href}`);
        console.error(`🔍 Origin: ${window.location.origin}`);
        console.error(`🔍 Pathname: ${window.location.pathname}`);
        
        throw error;
    }
}
```

## 🔧 **AUSSI : Fonction de debug à ajouter :**

```javascript
// ===================================================================
// FONCTION DE DEBUG POUR TESTER LES URLS
// ===================================================================

async function debugURLs() {
    console.log("🔍 DEBUG URLs:");
    console.log("Current URL:", window.location.href);
    console.log("Origin:", window.location.origin);
    console.log("Pathname:", window.location.pathname);
    console.log("Base URL:", window.location.origin + window.location.pathname);
    
    // Test des endpoints
    const endpoints = ['health', 'get_model_status', 'get_connection_status'];
    
    for (const endpoint of endpoints) {
        try {
            const url = `./${endpoint}`;
            console.log(`Testing: ${url}`);
            
            const response = await fetch(url, { method: 'HEAD' });
            console.log(`${endpoint}: ${response.status}`);
            
        } catch (error) {
            console.error(`${endpoint}: ERROR - ${error.message}`);
        }
    }
}

// Lancer le debug au chargement
window.debugURLs = debugURLs;
```

## 🎯 **INSTRUCTIONS DE CORRECTION :**

### **1. Remplacer la fonction apiCall**
Remplacez complètement la fonction `apiCall` dans votre JavaScript par une des versions ci-dessus.

### **2. Tester avec la console**
Ouvrez la console du navigateur (F12) et tapez :
```javascript
debugURLs()
```

### **3. Si ça ne marche toujours pas**
Ajoutez cette fonction temporaire pour tester :
```javascript
async function testSimpleAPI() {
    try {
        const response = await fetch('./health');
        const text = await response.text();
        console.log("Response:", text);
        return text;
    } catch (error) {
        console.error("Error:", error);
        return error.message;
    }
}
```

## 🚀 **REMPLACEZ la fonction apiCall dans votre JavaScript et testez !**

**Le problème principal est que `getWebAppBackendUrl()` n'existe pas dans l'environnement Dataiku standard.**

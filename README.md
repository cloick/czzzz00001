graph TB
    subgraph External["🌐 SERVICES EXTERNES"]
        Confluence[Confluence API<br/>REST v2<br/>Rate: 100 req/min]
        OpenAI[OpenAI API<br/>text-embedding-ada-002<br/>gpt-4-turbo]
        Cohere[Cohere API<br/>rerank-english-v3.0]
    end
    
    subgraph DataikuDSS["🔷 DATAIKU DSS 12.x"]
        
        subgraph WebApps["📱 WEB APPS Dash/Bokeh"]
            WA_Chat[Web App: Chat RAG<br/>---<br/>Framework: Dash<br/>Components: Input, Chat, Feedback<br/>Backend: Python Function]
            WA_Curation[Web App: Qualité & Curation<br/>---<br/>Framework: Dash + DataTables<br/>Tabs: Signalés, Problèmes, Suggestions<br/>Actions: Supprimer, Modifier, Étiqueter]
            WA_Monitor[Web App: Monitoring<br/>---<br/>Framework: Dash + Plotly<br/>Charts: Precision, Recall, Latency<br/>Refresh: Auto 5 min]
        end
        
        subgraph Scenarios["⏰ SCENARIOS Orchestration"]
            SC_Nightly[Scenario: Nightly Analysis<br/>---<br/>Trigger: Time-based 2:00 AM daily<br/>Duration: 1-2h<br/>Cost: ~$30/nuit<br/>Steps: 7 recipes séquentiels<br/>Retry: 3 attempts<br/>Notification: Email on error]
            SC_Weekly[Scenario: Weekly Suggestions<br/>---<br/>Trigger: Time-based Sunday 11 PM<br/>Duration: 2-3h<br/>Cost: ~$100<br/>Steps: 4 recipes LLM intensifs]
            SC_Metrics[Scenario: Metrics Update<br/>---<br/>Trigger: Time-based every 5 min<br/>Duration: 10s<br/>Cost: $0<br/>Step: 1 recipe aggregation]
        end
        
        subgraph Recipes["🔧 RECIPES Python 3.11"]
            R_RAG[Recipe: RAG Query Pipeline<br/>---<br/>Type: Python<br/>Input: User query<br/>Steps:<br/>1. Query preprocessing<br/>2. OpenAI embedding<br/>3. pgvector search<br/>4. Cohere reranking<br/>5. Context building<br/>6. GPT-4 generation<br/>7. Log to query_logs<br/>Output: Response + Sources<br/>Latency: ~2s]
            
            R_Sync[Recipe: Confluence Sync<br/>---<br/>Type: Python<br/>API: Confluence REST v2<br/>Actions:<br/>• Fetch all pages<br/>• Detect new/modified/deleted<br/>• Update documents table

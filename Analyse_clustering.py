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
            
            R_Sync[Recipe: Confluence Sync<br/>---<br/>Type: Python<br/>API: Confluence REST v2<br/>Actions:<br/>• Fetch all pages<br/>• Detect new/modified/deleted<br/>• Update documents table<br/>Duration: ~30 min]
            
            R_Embed[Recipe: Generate Embeddings<br/>---<br/>Type: Python<br/>API: OpenAI ada-002<br/>Batch size: 100 docs<br/>Input: documents sans embeddings<br/>Output: document_embeddings<br/>Duration: ~1h<br/>Cost: ~$10]
            
            R_Detect[Recipe: Detect Issues<br/>---<br/>Type: Python<br/>Analyses:<br/>• Obsolete: dates + regex tech<br/>• Orphans: graphe liens<br/>• Duplicates: cosine > 0.85<br/>• Contradictions: GPT-4<br/>• Gaps: NER + keywords<br/>Duration: ~1h<br/>Cost: ~$20]
            
            R_Suggest[Recipe: Generate Suggestions<br/>---<br/>Type: Python<br/>API: GPT-4 pour génération<br/>Génère:<br/>• Fusion proposals<br/>• Creation proposals<br/>• Tag suggestions<br/>Duration: ~2h<br/>Cost: ~$100]
            
            R_Metrics[Recipe: Aggregate Metrics<br/>---<br/>Type: Python<br/>Input: query_logs last 5 min<br/>Compute:<br/>• P50, P95, P99 latency<br/>• Precision, Recall<br/>• Faithfulness<br/>• Cost per query<br/>Output: monitoring_metrics]
        end
        
        subgraph Datasets["💾 DATASETS PostgreSQL"]
            DS_Docs[(documents<br/>---<br/>Table: documents<br/>Rows: ~1,000<br/>Columns: id, title, content,<br/>created_at, updated_at)]
            
            DS_Embeddings[(document_embeddings<br/>---<br/>Table: document_embeddings<br/>Type: vector1536<br/>Index: HNSW<br/>Rows: ~5,000 chunks)]
            
            DS_Logs[(query_logs<br/>---<br/>Table: query_logs<br/>Rows: ~10K/jour<br/>Retention: 90 jours)]
            
            DS_Reports[(user_reports<br/>---<br/>Table: user_reports<br/>Status: pending/resolved)]
            
            DS_Issues[(Tables Issues<br/>---<br/>• obsolete_pages<br/>• orphan_pages<br/>• duplicate_groups<br/>• contradictions<br/>• missing_documentation)]
            
            DS_Suggestions[(Tables Suggestions<br/>---<br/>• fusion_proposals<br/>• creation_proposals<br/>• tag_suggestions)]
            
            DS_Metrics[(monitoring_metrics<br/>---<br/>Table: monitoring_metrics<br/>Granularité: 5 min<br/>Retention: 6 mois)]
        end
        
        subgraph Connections["🔌 CONNECTIONS"]
            CONN_PG[PostgreSQL Connection<br/>---<br/>Host: your-host:5432<br/>Database: rag_wiki_db<br/>User: dataiku_user<br/>Extensions: pgvector<br/>SSL: Enabled]
            
            CONN_APIs[API Connections<br/>---<br/>• OpenAI: HTTP Preset<br/>• Cohere: HTTP Preset<br/>• Confluence: HTTP Preset<br/>Auth: API Keys stored<br/>in Dataiku secrets]
        end
        
        subgraph CodeEnv["🐍 CODE ENVIRONMENT"]
            PythonEnv[Python 3.11 Managed Env<br/>---<br/>Packages:<br/>• openai==1.12.0<br/>• cohere==4.47<br/>• pgvector==0.2.4<br/>• psycopg2-binary==2.9.9<br/>• sqlalchemy==2.0.25<br/>• ragas==0.1.4<br/>• pandas==2.2.0<br/>• sentence-transformers==2.3.1]
        end
    end
    
    subgraph Database["💾 PostgreSQL 16 + pgvector"]
        PG[(PostgreSQL Database<br/>---<br/>Version: 16+<br/>Extension: pgvector 0.5.x<br/>Size: ~250 MB + 50 MB/jour<br/>Connections: Pool max 20)]
    end
    
    subgraph Users["👥 UTILISATEURS"]
        EndUsers[Utilisateurs Finaux<br/>---<br/>Accès: Web Apps<br/>Auth: Dataiku SSO]
        Admins[Administrateurs<br/>---<br/>Accès: Dataiku Console<br/>+ Web Apps]
    end
    
    %% Flux Utilisateurs
    EndUsers -->|HTTPS| WA_Chat
    EndUsers -->|HTTPS| WA_Curation
    Admins -->|HTTPS| WA_Monitor
    
    %% Web Apps -> Recipes
    WA_Chat -->|trigger| R_RAG
    WA_Chat -->|feedback| DS_Reports
    WA_Curation -->|display| DS_Reports
    WA_Curation -->|display| DS_Issues
    WA_Curation -->|display| DS_Suggestions
    WA_Monitor -->|display| DS_Metrics
    
    %% Scenarios -> Recipes
    SC_Nightly -->|step 1| R_Sync
    SC_Nightly -->|step 2| R_Embed
    SC_Nightly -->|step 3-7| R_Detect
    
    SC_Weekly -->|step 1-4| R_Suggest
    
    SC_Metrics -->|step 1| R_Metrics
    
    %% Recipes -> Datasets
    R_RAG -->|write| DS_Logs
    R_Sync -->|write| DS_Docs
    R_Embed -->|write| DS_Embeddings
    R_Detect -->|write| DS_Issues
    R_Suggest -->|write| DS_Suggestions
    R_Metrics -->|read| DS_Logs
    R_Metrics -->|write| DS_Metrics
    
    %% Recipes -> External APIs
    R_RAG -->|embed + generate| OpenAI
    R_RAG -->|rerank| Cohere
    R_RAG -->|search| DS_Embeddings
    R_Sync -->|fetch pages| Confluence
    R_Embed -->|embed| OpenAI
    R_Detect -->|detect contradictions| OpenAI
    R_Suggest -->|generate content| OpenAI
    
    %% Datasets -> Database
    DS_Docs -.->|SQL| PG
    DS_Embeddings -.->|SQL + vector ops| PG
    DS_Logs -.->|SQL| PG
    DS_Reports -.->|SQL| PG
    DS_Issues -.->|SQL| PG
    DS_Suggestions -.->|SQL| PG
    DS_Metrics -.->|SQL| PG
    
    %% Connections
    CONN_PG -.->|connect| PG
    CONN_APIs -.->|authenticate| OpenAI
    CONN_APIs -.->|authenticate| Cohere
    CONN_APIs -.->|authenticate| Confluence
    
    %% Code Environment
    PythonEnv -.->|used by| R_RAG
    PythonEnv -.->|used by| R_Sync
    PythonEnv -.->|used by| R_Embed
    PythonEnv -.->|used by| R_Detect
    PythonEnv -.->|used by| R_Suggest
    PythonEnv -.->|used by| R_Metrics
    
    %% Styles
    classDef webapp fill:#667eea,stroke:#5568d3,color:#fff,stroke-width:2px
    classDef scenario fill:#ed8936,stroke:#dd6b20,color:#fff,stroke-width:2px
    classDef recipe fill:#48bb78,stroke:#38a169,color:#fff,stroke-width:2px
    classDef dataset fill:#4299e1,stroke:#3182ce,color:#fff,stroke-width:2px
    classDef connection fill:#9f7aea,stroke:#805ad5,color:#fff,stroke-width:2px
    classDef external fill:#f6ad55,stroke:#ed8936,color:#000,stroke-width:2px
    classDef database fill:#2c5282,stroke:#2a4365,color:#fff,stroke-width:2px
    
    class WA_Chat,WA_Curation,WA_Monitor webapp
    class SC_Nightly,SC_Weekly,SC_Metrics scenario
    class R_RAG,R_Sync,R_Embed,R_Detect,R_Suggest,R_Metrics recipe
    class DS_Docs,DS_Embeddings,DS_Logs,DS_Reports,DS_Issues,DS_Suggestions,DS_Metrics dataset
    class CONN_PG,CONN_APIs connection
    class PythonEnv connection
    class Confluence,OpenAI,Cohere external
    class PG database

```mermaid
flowchart LR
    A[Raw Support Tickets<br/>(JSON / Stream)] --> B[Bronze Delta Table]
    B --> C[Silver Table<br/>Cleaned + Normalized]
    C --> D[Gold Feature Table<br/>Text + Labels]
    D --> E[ML Training<br/>(PySpark + MLflow)]
    D --> F[Embeddings Generation<br/>(Embedding API)]
    F --> G[Embeddings Delta Table]
    Q[New Incoming Ticket] --> H[Embed Query Text]
    H --> I[Similarity Search<br/>Against Embeddings]
    I --> J[Build RAG Prompt<br/>Context + Query]
    J --> K[LLM Endpoint<br/>(Databricks Model Serving / Other)]
    K --> L[AI Response Suggestion]
```

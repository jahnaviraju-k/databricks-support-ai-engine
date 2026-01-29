# Real-Time Customer Support AI Engine on Databricks

This project showcases an **advanced end-to-end Data + AI pipeline on Databricks**:

- 🧱 Lakehouse architecture (Bronze → Silver → Gold)
- ⚙️ Streaming ETL with PySpark
- 💾 Delta Lake for reliable storage
- 🧠 ML model training & tracking with MLflow
- 🔍 Embeddings + retrieval (RAG-style) for smarter responses
- 🤖 LLM-based response generation (placeholder for Databricks Model Serving or OpenAI)

> This repo is designed as a portfolio project to demonstrate skills in **Databricks, PySpark, Delta, MLflow, and modern AI pipelines**.

## Quick Start (Databricks)

1. Import this repo into **Databricks Repos** or upload the files.
2. Create/attach a cluster.
3. Run `notebooks/01_dlt_bronze_silver_gold.py` to create Delta tables.
4. Run `notebooks/03_train_ticket_classifier_mlflow.py` to train the model.
5. Run `notebooks/02_generate_embeddings_and_vector_index.py` to build the embeddings table.
6. Use `notebooks/04_rag_inference_notebook.py` or `src/rag_service.py` as the basis for an inference / serving endpoint.

See `diagrams/architecture_mermaid.md` for the architecture diagram.

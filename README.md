# Real-Time Customer Support AI Engine on Databricks

This project showcases an **advanced end-to-end Data + AI pipeline on Databricks**:

- Lakehouse architecture (Bronze → Silver → Gold)
- Streaming ETL with PySpark
- Delta Lake for reliable storage
- ML model training & tracking with MLflow
- Embeddings + retrieval (RAG-style) for smarter responses
- LLM-based response generation (placeholder for Databricks Model Serving or OpenAI)

## Architecture

See [architecture_mermaid.md](architecture_mermaid.md) for the full pipeline diagram.

## Project Structure

| File | Description |
|------|-------------|
| `config.py` | Central configuration (table names, paths, experiment settings) |
| `etl_streaming.py` | Streaming ingestion of raw JSON tickets into the Bronze Delta table |
| `01_dlt_bronze_silver_gold.py` | Notebook: Bronze → Silver (clean/normalize) → Gold (feature table) |
| `02_generate_embeddings_and_vector_index.py` | Notebook: Generate embeddings from the Gold table |
| `03_train_ticket_classifier_mlflow.py` | Notebook: Train a ticket classifier with MLflow tracking |
| `04_rag_inference_notebook.py` | Notebook: RAG-based inference — retrieve similar tickets and generate a response |
| `embedding_utils.py` | Text embedding utility (placeholder — replace with real API) |
| `vector_index_utils.py` | Cosine similarity search over embedding vectors |
| `model_training.py` | Spark ML pipeline: Tokenizer → StopWords → TF-IDF → LogisticRegression |
| `rag_service.py` | RAG orchestration: embed query → retrieve → prompt → LLM |
| `sample_tickets.jsonl` | Sample support ticket data for testing |

## Prerequisites

- **Databricks workspace** (Community Edition or higher)
- **Databricks Runtime 13.x+** with MLflow pre-installed
- Upload `sample_tickets.jsonl` to `dbfs:/FileStore/support_ai/raw/`

## How to Run

1. Upload all `.py` files to your Databricks workspace
2. Upload `sample_tickets.jsonl` to `dbfs:/FileStore/support_ai/raw/`
3. Run notebooks in order:
   - `etl_streaming.py` — starts the streaming ingestion into Bronze
   - `01_dlt_bronze_silver_gold.py` — transforms Bronze → Silver → Gold
   - `02_generate_embeddings_and_vector_index.py` — generates embeddings
   - `03_train_ticket_classifier_mlflow.py` — trains the classifier
   - `04_rag_inference_notebook.py` — runs RAG inference on a sample query

## Configuration

All table names, paths, and MLflow settings are centralized in `config.py`.

LLM credentials should be set via **environment variables** or **Databricks secrets**:
- `LLM_ENDPOINT` — your LLM serving endpoint URL
- `LLM_TOKEN` — your API token

## Important Notes

- **Embeddings are placeholder**: `embedding_utils.py` currently returns a fixed zero vector for demo purposes. Replace `get_text_embedding()` with a real embedding API (OpenAI, Databricks Foundation Models, Sentence Transformers, etc.) for meaningful similarity search.
- **LLM call is placeholder**: `rag_service.py` includes a stub `call_llm()` function. Integrate with Databricks Model Serving or an external LLM provider for real responses.

## Dependencies

See `requirements.txt` for the Python dependencies used in this project. On Databricks, PySpark and MLflow are pre-installed on the cluster.



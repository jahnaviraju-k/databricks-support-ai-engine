# config.py

BRONZE_TABLE = "support_tickets_bronze"
SILVER_TABLE = "support_tickets_silver"
GOLD_FEATURE_TABLE = "support_tickets_gold_features"
EMBEDDING_TABLE = "support_tickets_embeddings"

CAT_TARGET_COL = "category"
PRIORITY_TARGET_COL = "priority"

RAW_DATA_PATH = "dbfs:/FileStore/support_ai/raw/"
CHECKPOINT_BASE_PATH = "dbfs:/FileStore/support_ai/checkpoints/"

EXPERIMENT_NAME = "/Shared/experiments/support_ticket_classifier"
MODEL_NAME = "support_ticket_classifier"

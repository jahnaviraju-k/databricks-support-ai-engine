# Databricks notebook: 02_generate_embeddings_and_vector_index

from pyspark.sql import functions as F, types as T

import config
from embedding_utils import get_text_embedding

gold_df = spark.table(config.GOLD_FEATURE_TABLE).select("ticket_id", "text", "category", "priority")

def embed_udf_py(text: str):
    if text is None:
        return None
    return get_text_embedding(text)

embed_udf = F.udf(embed_udf_py, T.ArrayType(T.FloatType()))

emb_df = (
    gold_df
    .withColumn("embedding", embed_udf(F.col("text")))
)

emb_df.write.mode("overwrite").format("delta").saveAsTable(config.EMBEDDING_TABLE)

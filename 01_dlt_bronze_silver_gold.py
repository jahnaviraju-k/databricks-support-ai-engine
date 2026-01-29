# Databricks notebook: 01_dlt_bronze_silver_gold

from pyspark.sql import functions as F
import src.config as config

# Ensure bronze table exists (schema comes from streaming job)
spark.sql(f"CREATE TABLE IF NOT EXISTS {config.BRONZE_TABLE} USING delta AS SELECT * FROM VALUES (1) WHERE 1=0")

# SILVER: clean and normalize
bronze_df = spark.table(config.BRONZE_TABLE)

silver_df = (
    bronze_df
    .withColumn("subject_clean", F.trim(F.col("subject")))
    .withColumn("body_clean", F.trim(F.col("body")))
    .withColumn("body_clean", F.regexp_replace("body_clean", "\\s+", " "))
    .dropDuplicates(["ticket_id"])
)

silver_df.write.mode("overwrite").format("delta").saveAsTable(config.SILVER_TABLE)

# GOLD: feature table for ML
silver_df = spark.table(config.SILVER_TABLE)

gold_df = (
    silver_df
    .withColumn("text", F.concat_ws(" ", "subject_clean", "body_clean"))
    .select(
        "ticket_id", "customer_id", "channel", "created_at",
        "text",
        "category",
        "priority",
        "status"
    )
)

gold_df.write.mode("overwrite").format("delta").saveAsTable(config.GOLD_FEATURE_TABLE)

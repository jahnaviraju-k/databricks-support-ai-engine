# etl_streaming.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, input_file_name
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
import config

def main():
    spark = (
        SparkSession.builder
        .appName("support_tickets_streaming_etl")
        .getOrCreate()
    )

    schema = StructType([
        StructField("ticket_id", StringType(), True),
        StructField("created_at", TimestampType(), True),
        StructField("channel", StringType(), True),
        StructField("customer_id", StringType(), True),
        StructField("subject", StringType(), True),
        StructField("body", StringType(), True),
        StructField("category", StringType(), True),
        StructField("priority", StringType(), True),
        StructField("status", StringType(), True),
    ])

    raw_path = config.RAW_DATA_PATH

    df_stream = (
        spark.readStream
        .schema(schema)
        .option("maxFilesPerTrigger", 1)
        .json(raw_path)
        .withColumn("ingest_ts", current_timestamp())
        .withColumn("source_file", input_file_name())
    )

    # Write to bronze Delta table
    (
        df_stream.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", config.CHECKPOINT_BASE_PATH + "bronze")
        .table(config.BRONZE_TABLE)
    )

    spark.streams.awaitAnyTermination()

if __name__ == "__main__":
    main()

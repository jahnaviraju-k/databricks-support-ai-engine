# model_training.py

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import mlflow
import mlflow.spark

import config

def main():
    spark = (
        SparkSession.builder
        .appName("support_ticket_classifier_training")
        .getOrCreate()
    )

    mlflow.set_experiment(config.EXPERIMENT_NAME)

    df = spark.table(config.GOLD_FEATURE_TABLE).dropna(subset=["text", config.CAT_TARGET_COL])

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    label_indexer = StringIndexer(
        inputCol=config.CAT_TARGET_COL,
        outputCol="label",
        handleInvalid="keep"
    )

    tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    hashing_tf = HashingTF(inputCol="filtered_tokens", outputCol="raw_features", numFeatures=1 << 18)
    idf = IDF(inputCol="raw_features", outputCol="features")

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=30,
        regParam=0.1
    )

    pipeline = Pipeline(stages=[label_indexer, tokenizer, remover, hashing_tf, idf, lr])

    with mlflow.start_run():
        model = pipeline.fit(train_df)

        preds = model.transform(test_df)

        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="f1"
        )

        f1 = evaluator.evaluate(preds)
        mlflow.log_metric("f1", f1)

        print(f"Validation F1: {f1}")

        mlflow.spark.log_model(
            spark_model=model,
            artifact_path="support_ticket_classifier",
            registered_model_name=config.MODEL_NAME
        )

if __name__ == "__main__":
    main()

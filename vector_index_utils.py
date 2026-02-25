# vector_index_utils.py

from typing import List
from pyspark.sql import DataFrame, functions as F, types as T

def dot_product(a: List[float], b: List[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))

def norm(a: List[float]) -> float:
    return float(sum(x * x for x in a) ** 0.5)

def cosine_similarity(a: List[float], b: List[float]) -> float:
    na = norm(a)
    nb = norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot_product(a, b) / (na * nb)

def add_similarity_column(emb_df: DataFrame, query_vec: List[float]) -> DataFrame:
    """Add cosine similarity column between `embedding` and query_vec."""
    def sim_func(vec):
        if vec is None:
            return 0.0
        return cosine_similarity(vec, query_vec)

    sim_udf = F.udf(sim_func, T.FloatType())
    return emb_df.withColumn("similarity", sim_udf(F.col("embedding")))

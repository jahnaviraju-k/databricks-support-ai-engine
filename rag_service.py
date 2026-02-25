# rag_service.py

import os
from typing import List, Dict
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from embedding_utils import get_text_embedding
from vector_index_utils import add_similarity_column
import config

# LLM endpoint — use Databricks secrets or environment variables
LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "https://your-llm-endpoint")
LLM_TOKEN = os.environ.get("LLM_TOKEN", "")

def build_prompt(query: str, similar_docs: List[Dict]) -> str:
    context_block = "\n\n".join(
        [f"[Ticket {d['ticket_id']}] {d['text']}" for d in similar_docs]
    )
    prompt = f"""You are an AI support assistant.

Context from similar tickets:
{context_block}

User query:
{query}

Based on the context, draft a helpful, concise support response."""
    return prompt

def call_llm(prompt: str) -> str:
    """Placeholder LLM call.

    Replace with Databricks Model Serving or any REST LLM provider.
    """
    return f"[DEMO ONLY] This is where the LLM response would go.\nPrompt was:\n{prompt}"

def answer_ticket(query_text: str, top_k: int = 5) -> Dict:
    spark = (
        SparkSession.builder
        .appName("rag_inference")
        .getOrCreate()
    )

    emb_df = spark.table(config.EMBEDDING_TABLE)

    query_vec = get_text_embedding(query_text)
    scored_df = add_similarity_column(emb_df, query_vec)

    top_docs = (
        scored_df
        .orderBy(col("similarity").desc())
        .limit(top_k)
        .select("ticket_id", "text", "similarity")
        .collect()
    )

    top_docs_list = [
        {
            "ticket_id": row["ticket_id"],
            "text": row["text"],
            "similarity": float(row["similarity"])
        }
        for row in top_docs
    ]

    prompt = build_prompt(query_text, top_docs_list)
    llm_response = call_llm(prompt)

    return {
        "query": query_text,
        "retrieved_docs": top_docs_list,
        "answer": llm_response
    }

if __name__ == "__main__":
    example_query = "My credit card keeps getting declined when I try to renew my subscription."
    result = answer_ticket(example_query)
    print(result["answer"])

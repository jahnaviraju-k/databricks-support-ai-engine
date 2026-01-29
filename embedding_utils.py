# src/embedding_utils.py

from typing import List

def get_text_embedding(text: str) -> List[float]:
    """Placeholder embedding generator.

    Replace this with a real embedding API (OpenAI, Databricks, etc.).
    For demo / portfolio, we just return a fixed-length zero vector.
    """
    if text is None:
        return None
    # 10-dim dummy vector for demo
    return [0.0] * 10

# Databricks notebook: 04_rag_inference_notebook

from rag_service import answer_ticket

query_text = "I am being charged twice for my monthly subscription, please help."
result = answer_ticket(query_text)

print("Query:")
print(result["query"])
print("\nTop retrieved tickets:")
for doc in result["retrieved_docs"]:
    print(f"- {doc['ticket_id']} (similarity={doc['similarity']:.3f}) -> {doc['text'][:80]}...")
print("\nProposed AI response:")
print(result["answer"])

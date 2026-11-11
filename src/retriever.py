from typing import Any

from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


TOP_K = 5
NEIGHBOR_TOP_N = 2
NEIGHBOR_WINDOW = 1


class Retriever:
    def __init__(
        self,
        top_k: int = TOP_K,
    ):
        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        self.top_k = top_k
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()

    def retrieve(
        self,
        query: str,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        query_embedding = (
            self.embedding_model.embed_query(query)
        )

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]
        scores = results.get("scores", [[]])[0]

        retrieved_chunks = []

        for index, document in enumerate(documents):
            retrieved_chunks.append(
                {
                    "id": ids[index],
                    "text": document,
                    "metadata": metadatas[index],
                    "distance": distances[index],
                    "score": (
                        scores[index]
                        if index < len(scores)
                        else None
                    ),
                    "is_adjacent": False,
                }
            )

        

        return retrieved_chunks


if __name__ == "__main__":
    retriever = Retriever()

    query = input(
        "Enter your question: "
    ).strip()

    results = retriever.retrieve(query)

    print(
        f"\nRetrieved {len(results)} chunks:\n"
    )

    for index, result in enumerate(
        results,
        start=1,
    ):
        print("=" * 80)
        print(f"Rank: {index}")
        print(f"ID: {result['id']}")
        print(
            f"Page: "
            f"{result['metadata'].get('page')}"
        )
        print(
            f"Distance: "
            f"{result['distance']}"
        )
        print(
            f"Adjacent: "
            f"{result.get('is_adjacent', False)}"
        )
        print("\nText:")
        print(result["text"])
        print()
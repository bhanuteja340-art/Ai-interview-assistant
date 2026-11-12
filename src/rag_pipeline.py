
import logging
import sys
import time
from typing import Any

from src.cache import ChatCache
from src.generator import Generator
from src.retriever import Retriever


# New cache version for answers generated from one chunk.
# Old cached answers remain stored but will not be reused.
CACHE_VERSION = "single_chunk_v3"


logger = logging.getLogger("chatbot.rag_pipeline")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)

logger.propagate = False


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()
        self.cache = ChatCache()
        logger.info("RAG pipeline initialized")

    @staticmethod
    def get_selected_chunk(
        retrieved_chunks: list[dict[str, Any]],
    ) -> dict[str, Any] | None:

        # Same selection logic used in generator.py.
        # The retriever returns results ranked by Qdrant.
        return next(
            (
                chunk
                for chunk in retrieved_chunks
                if str(chunk.get("text") or "").strip()
            ),
            None,
        )

    def answer(self, question: str) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        start_time = time.perf_counter()
        logger.info("Question processing started")

        # Versioned cache key prevents old five-chunk
        # answers from being returned.
        cache_question = f"{CACHE_VERSION}:{question}"

        try:
            cached_result = self.cache.get(cache_question)
        except Exception as exc:
            logger.error(
                "Cache lookup failed: %s",
                type(exc).__name__,
            )
            raise

        if cached_result is not None:
            total_latency = time.perf_counter() - start_time

            retrieved_chunks = cached_result.get(
                "retrieved_chunks", []
            )

            selected_chunk = self.get_selected_chunk(
                retrieved_chunks
            )

            logger.info("Cache hit")
            logger.info(
                "Request completed | status=cache_hit | total=%.3fs",
                total_latency,
            )

            return {
                "question": question,
                "answer": cached_result["answer"],
                "grounded_content": cached_result.get(
                    "grounded_content", ""
                ),
                "retrieved_chunks": retrieved_chunks,
                "selected_chunk": selected_chunk,
                "retrieval_latency": 0.0,
                "generation_latency": 0.0,
                "total_latency": total_latency,
                "status": "cache_hit",
                "error": None,
            }

        logger.info("Cache miss")

        retrieval_start = time.perf_counter()

        try:
            retrieved_chunks = self.retriever.retrieve(
                question
            )
        except Exception as exc:
            logger.error(
                "Retrieval failed: %s",
                type(exc).__name__,
            )
            raise

        retrieval_latency = (
            time.perf_counter() - retrieval_start
        )

        logger.info(
            "Retrieval completed | chunks=%d | duration=%.3fs",
            len(retrieved_chunks),
            retrieval_latency,
        )

        generation_start = time.perf_counter()

        try:
            generation_result = (
                self.generator.generate_with_grounding(
                    question,
                    retrieved_chunks,
                )
            )
        except Exception as exc:
            logger.error(
                "Generation failed: %s",
                type(exc).__name__,
            )
            raise

        grounded_content = generation_result[
            "grounded_content"
        ]

        answer = generation_result["answer"]

        selected_chunk = generation_result.get(
            "selected_chunk"
        )

        generation_latency = (
            time.perf_counter() - generation_start
        )

        logger.info(
            "Generation completed | duration=%.3fs",
            generation_latency,
        )

        if selected_chunk is not None:
            metadata = selected_chunk.get("metadata") or {}

            logger.info(
                "Selected source | chunk_id=%s | page=%s",
                metadata.get("chunk_id"),
                metadata.get("page"),
            )

        try:
            self.cache.set(
                question=cache_question,
                answer=answer,
                grounded_content=grounded_content,
                retrieved_chunks=retrieved_chunks,
                finish_reason=generation_result.get(
                    "finish_reason"
                ),
            )
        except Exception as exc:
            logger.error(
                "Cache save failed: %s",
                type(exc).__name__,
            )
            raise

        total_latency = time.perf_counter() - start_time

        logger.info(
            "Request completed | status=success | total=%.3fs",
            total_latency,
        )

        return {
            "question": question,
            "answer": answer,
            "grounded_content": grounded_content,
            "retrieved_chunks": retrieved_chunks,
            "selected_chunk": selected_chunk,
            "retrieval_latency": retrieval_latency,
            "generation_latency": generation_latency,
            "total_latency": total_latency,
            "status": "success",
            "error": None,
        }


if __name__ == "__main__":
    pipeline = RAGPipeline()

    question = input("Enter your question: ").strip()
    result = pipeline.answer(question)

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(result["question"])

    print("\n" + "=" * 80)
    print("GROUNDED CONTENT")
    print("=" * 80)
    print(result["grounded_content"])

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(result["answer"])

    print("\n" + "=" * 80)
    print("PERFORMANCE")
    print("=" * 80)
    print(
        f"Retrieval latency: "
        f"{result['retrieval_latency']:.3f} seconds"
    )
    print(
        f"Generation latency: "
        f"{result['generation_latency']:.3f} seconds"
    )
    print(
        f"Total latency: "
        f"{result['total_latency']:.3f} seconds"
    )
    print(f"Status: {result['status']}")

    print("\n" + "=" * 80)
    print("SELECTED SOURCE CHUNK")
    print("=" * 80)

    selected_chunk = result.get("selected_chunk")

    if selected_chunk:
        metadata = selected_chunk.get("metadata") or {}

        print(
            f"Chunk ID: {metadata.get('chunk_id')}"
        )
        print(
            f"Page: {metadata.get('page')}"
        )
        print(
            f"Distance: {selected_chunk.get('distance')}"
        )
        print("\nText:")
        print(selected_chunk["text"])
    else:
        print("No source chunk selected.")

    print("\n" + "=" * 80)
    print("ALL RETRIEVED CHUNKS")
    print("=" * 80)

    for index, chunk in enumerate(
        result["retrieved_chunks"],
        start=1,
    ):
        metadata = chunk.get("metadata") or {}

        print(f"\nRank: {index}")
        print(
            f"Page: {metadata.get('page')}"
        )
        print(
            f"Distance: {chunk.get('distance')}"
        )
        print(chunk["text"])

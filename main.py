import logging

from src.rag_pipeline import RAGPipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    try:
        pipeline = RAGPipeline()
    except Exception:
        logger.exception("Failed to initialize the RAG pipeline.")
        print(
            "Could not start: the knowledge base or AI service isn't "
            "available. Check the logs above for details."
        )
        return

    print("Knowledge base ready. Type 'exit' to quit.")

    while True:
        try:
            question = input(
                "\nEnter your question (or type 'exit' to quit): "
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if question.lower() == "exit":
            print("Goodbye.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            result = pipeline.answer(question)
        except RuntimeError as exc:
            logger.warning("Query failed: %s", exc)
            print(f"\nCouldn't answer that: {exc}")
            continue
        except Exception as exc:
            logger.exception("Unexpected error answering question.")
            print(
                f"\nSomething went wrong answering that question "
                f"({type(exc).__name__}). Please try again."
            )
            continue

        if not isinstance(result, dict) or "answer" not in result:
            logger.error(
                "Unexpected result shape from pipeline.answer(): %r",
                result,
            )
            print(
                "\nGot an unexpected response from the pipeline. "
                "Please try again."
            )
            continue

        print("\n" + "=" * 80)
        print("QUESTION")
        print("=" * 80)
        print(question)

        print("\n" + "=" * 80)
        print("PERFORMANCE")
        print("=" * 80)
        print(
            f"Retrieval latency: "
            f"{result.get('retrieval_latency', 0):.3f} seconds"
        )
        print(
            f"Generation latency: "
            f"{result.get('generation_latency', 0):.3f} seconds"
        )
        print(
            f"Total latency: "
            f"{result.get('total_latency', 0):.3f} seconds"
        )
        print(f"Status: {result.get('status', 'unknown')}")

        retrieved_chunks = result.get("retrieved_chunks", [])

        print("\n" + "=" * 80)
        print("RETRIEVED CHUNKS")
        print("=" * 80)

        for index, chunk in enumerate(retrieved_chunks, start=1):
            metadata = chunk.get("metadata", {})

            print(f"\nRank: {index}")
            print(f"Source: {metadata.get('source')}")
            print(f"Page: {metadata.get('page')}")
            print(f"Distance: {chunk.get('distance')}")
            print("\nText:")
            print(chunk.get("text", ""))

        print("\n" + "=" * 80)
        print("GENERATED ANSWER")
        print("=" * 80)
        print(result["answer"])


if __name__ == "__main__":
    main()
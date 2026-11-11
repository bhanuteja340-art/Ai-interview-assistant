import os
import time
from typing import Any, Iterator

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types


load_dotenv()


EMBEDDING_MODEL = "gemini-embedding-001"
EMBED_BATCH_SIZE = 50

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

RETRYABLE_STATUS_CODES = {
    408,
    500,
    502,
    503,
    504,
}


class EmbeddingModel:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured."
            )

        self.client = genai.Client(
            api_key=api_key
        )

    def embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            raise ValueError(
                "Embedding batch cannot be empty."
            )

        for text in texts:
            if not text.strip():
                raise ValueError(
                    "Document text cannot be empty."
                )

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT"
                    ),
                )

                embeddings = [
                    embedding.values
                    for embedding in response.embeddings
                ]

                if len(embeddings) != len(texts):
                    raise RuntimeError(
                        "Embedding response count does not "
                        "match the input batch size."
                    )

                return embeddings

            except errors.APIError as exc:

                if exc.code == 429:
                    raise RuntimeError(
                        "Gemini embedding quota has been "
                        "exceeded. Wait for the quota window "
                        "to reset and run ingestion again."
                    ) from exc

                if (
                    exc.code not in RETRYABLE_STATUS_CODES
                    or attempt == MAX_RETRIES - 1
                ):
                    raise

                delay = RETRY_DELAY_SECONDS * (2**attempt)

                print(
                    f"Embedding batch failed with status "
                    f"{exc.code}. Retrying in {delay} seconds..."
                )

                time.sleep(delay)

        raise RuntimeError(
            "Embedding batch failed."
        )

    def embed_batches(
        self,
        chunks: list[dict[str, Any]],
        batch_size: int = EMBED_BATCH_SIZE,
        delay_seconds: float = 0,
        label: str = "chunks",
    ) -> Iterator[
        tuple[
            list[dict[str, Any]],
            list[list[float]],
        ]
    ]:
        if not chunks:
            raise ValueError(
                "No chunks provided."
            )

        if batch_size <= 0:
            raise ValueError(
                "Batch size must be greater than zero."
            )

        total = len(chunks)

        for start in range(
            0,
            total,
            batch_size,
        ):
            batch_chunks = chunks[
                start:start + batch_size
            ]

            print(
                f"Embedding {label} "
                f"{start + 1}-{start + len(batch_chunks)} "
                f"of {total}..."
            )

            batch_texts = [
                chunk["text"]
                for chunk in batch_chunks
            ]

            batch_embeddings = self.embed_batch(
                batch_texts
            )

            if len(batch_embeddings) != len(batch_chunks):
                raise RuntimeError(
                    "Embedding count does not match "
                    "batch size."
                )

            yield batch_chunks, batch_embeddings

            is_last_batch = (
                start + batch_size >= total
            )

            if delay_seconds and not is_last_batch:
                print(
                    f"Waiting {delay_seconds} seconds "
                    f"before the next batch..."
                )

                time.sleep(delay_seconds)

    def embed_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> list[list[float]]:
        if not documents:
            raise ValueError(
                "No documents provided."
            )

        embeddings: list[list[float]] = []

        for _, batch_embeddings in self.embed_batches(
            documents,
            label="documents",
        ):
            embeddings.extend(batch_embeddings)

            print(
                f"Embedded "
                f"{len(embeddings)}/{len(documents)} "
                f"documents."
            )

        return embeddings

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        response = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY"
            ),
        )

        return response.embeddings[0].values
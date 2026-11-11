import hashlib
import os
import uuid
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()

COLLECTION_NAME = "interview_knowledge_base"
VECTOR_SIZE = 3072
DISTANCE_METRIC = models.Distance.COSINE


class VectorStore:
    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url:
            raise RuntimeError("QDRANT_URL is not configured.")

        if not qdrant_api_key:
            raise RuntimeError("QDRANT_API_KEY is not configured.")

        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60,
        )

        self.collection_name = COLLECTION_NAME
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=DISTANCE_METRIC,
            ),
        )

    def clear(self) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(
                collection_name=self.collection_name
            )

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=DISTANCE_METRIC,
            ),
        )

    def document_id(self, chunk: dict[str, Any]) -> str:
        return self._build_id(chunk)

    @staticmethod
    def _build_id(chunk: dict[str, Any]) -> str:
        metadata = chunk["metadata"]
        chunk_id = metadata["chunk_id"]

        source = (
            metadata.get("source")
            or metadata.get("document_id")
            or metadata.get("file_name")
            or metadata.get("pdf_hash")
        )

        if source is not None:
            raw_id = f"{source}_{chunk_id}"
        else:
            text_hash = hashlib.sha256(
                chunk["text"].encode("utf-8")
            ).hexdigest()[:12]

            raw_id = f"{text_hash}_{chunk_id}"

        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                raw_id,
            )
        )

    def existing_ids(self, ids: list[str]) -> set[str]:
        if not ids:
            return set()

        existing_points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_payload=False,
            with_vectors=False,
        )

        return {str(point.id) for point in existing_points}

    def add_documents(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        if not chunks:
            raise ValueError("No chunks were provided.")

        points = []

        for chunk, embedding in zip(chunks, embeddings):
            if not embedding:
                raise ValueError("An empty embedding was provided.")

            if len(embedding) != VECTOR_SIZE:
                raise ValueError(
                    f"Embedding dimension mismatch. "
                    f"Expected {VECTOR_SIZE}, got {len(embedding)}."
                )

            points.append(
                models.PointStruct(
                    id=self._build_id(chunk),
                    vector=embedding,
                    payload={
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 8,
    ) -> dict[str, list[list[Any]]]:

        if not query_embedding:
            raise ValueError("Query embedding cannot be empty.")

        if len(query_embedding) != VECTOR_SIZE:
            raise ValueError(
                f"Query embedding dimension mismatch. "
                f"Expected {VECTOR_SIZE}, got {len(query_embedding)}."
            )

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not self.client.collection_exists(self.collection_name):
            raise RuntimeError("Vector store is empty.")

        collection_info = self.client.get_collection(
            self.collection_name
        )

        collection_count = collection_info.points_count or 0

        if collection_count == 0:
            raise RuntimeError("Vector store is empty.")

        top_k = min(top_k, collection_count)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        documents = []
        metadatas = []
        distances = []
        ids = []
        scores = []

        for point in response.points:
            payload = point.payload or {}

            text = payload.get("text", "")
            metadata = payload.get("metadata", {})
            score = float(point.score)

            documents.append(text)
            metadatas.append(metadata)
            distances.append(1.0 - score)
            ids.append(str(point.id))
            scores.append(score)

        return {
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
            "ids": [ids],
            "scores": [scores],
        }

    def get_adjacent_chunks(
        self,
        chunks: list[dict[str, Any]],
        top_n: int = 2,
        window: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Retrieve neighboring chunks for the highest-ranked semantic hits.

        This is useful when one answer spans multiple consecutive chunks.
        Only chunks from the same source are considered.
        """

        if not chunks:
            return []

        if top_n <= 0 or window <= 0:
            return []

        candidates = []
        seen_ids = {
            str(chunk.get("id"))
            for chunk in chunks
            if chunk.get("id") is not None
        }

        for chunk in chunks[:top_n]:
            metadata = chunk.get("metadata") or {}

            source = (
                metadata.get("source")
                or metadata.get("document_id")
                or metadata.get("file_name")
                or metadata.get("pdf_hash")
            )

            chunk_id = metadata.get("chunk_id")

            if source is None or chunk_id is None:
                continue

            try:
                current_chunk_id = int(chunk_id)
            except (TypeError, ValueError):
                continue

            for offset in range(-window, window + 1):
                if offset == 0:
                    continue

                adjacent_chunk_id = current_chunk_id + offset

                if adjacent_chunk_id < 0:
                    continue

                adjacent_metadata = {
                    "source": source,
                    "chunk_id": adjacent_chunk_id,
                }

                adjacent_id = self._build_id(
                    {
                        "text": "",
                        "metadata": adjacent_metadata,
                    }
                )

                if adjacent_id in seen_ids:
                    continue

                candidates.append(
                    {
                        "id": adjacent_id,
                        "_source": source,
                        "_chunk_id": adjacent_chunk_id,
                    }
                )

                seen_ids.add(adjacent_id)

        if not candidates:
            return []

        retrieved = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[candidate["id"] for candidate in candidates],
            with_payload=True,
            with_vectors=False,
        )

        candidate_lookup = {
            candidate["id"]: candidate
            for candidate in candidates
        }

        adjacent_chunks = []

        for point in retrieved:
            payload = point.payload or {}

            adjacent_chunks.append(
                {
                    "id": str(point.id),
                    "text": payload.get("text", ""),
                    "metadata": payload.get("metadata", {}),
                    "distance": None,
                    "score": None,
                    "is_adjacent": True,
                }
            )

        adjacent_chunks.sort(
            key=lambda item: (
                str(
                    (item.get("metadata") or {}).get("source", "")
                ),
                int(
                    (item.get("metadata") or {}).get(
                        "chunk_id",
                        0,
                    )
                ),
            )
        )

        return adjacent_chunks

    def count(self) -> int:
        if not self.client.collection_exists(self.collection_name):
            return 0

        collection_info = self.client.get_collection(
            self.collection_name
        )

        return collection_info.points_count or 0
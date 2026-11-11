import hashlib
import json
from pathlib import Path

from src.chunking import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    create_chunks,
)
from src.embeddings import (
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL,
    EmbeddingModel,
)
from src.pdf_loader import load_pdf
from src.vector_store import VectorStore


PDF_PATH = (
    Path(__file__).resolve().parent.parent
    / "knowledge_base"
    / "top 100 common interview questions and best answers.pdf"
)

METADATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "vector_store"
    / "ingestion_metadata.json"
)

EMBED_BATCH_DELAY_SECONDS = 65

CHUNKING_VERSION = "section-aware-v4"


def calculate_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            hasher.update(block)

    return hasher.hexdigest()


def build_fingerprint(
    pdf_hash: str,
) -> dict:
    return {
        "pdf_hash": pdf_hash,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunking_version": CHUNKING_VERSION,
        "embedding_model": EMBEDDING_MODEL,
    }


def load_existing_fingerprint() -> dict | None:
    if not METADATA_PATH.exists():
        return None

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_fingerprint(
    fingerprint: dict,
    chunk_count: int,
) -> None:
    METADATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        **fingerprint,
        "chunk_count": chunk_count,
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )


def fingerprint_matches(
    existing_fingerprint: dict | None,
    fingerprint: dict,
) -> bool:
    if existing_fingerprint is None:
        return False

    return all(
        existing_fingerprint.get(key) == value
        for key, value in fingerprint.items()
    )


def full_rebuild(
    vector_store: VectorStore,
    embedding_model: EmbeddingModel,
    chunks: list[dict],
    fingerprint: dict,
) -> None:
    print(
        "Knowledge base configuration changed. "
        "A full rebuild is required."
    )

    vector_store.clear()

    stored_count = 0

    for (
        batch_chunks,
        batch_embeddings,
    ) in embedding_model.embed_batches(
        chunks,
        batch_size=EMBED_BATCH_SIZE,
        delay_seconds=EMBED_BATCH_DELAY_SECONDS,
        label="chunks",
    ):
        vector_store.add_documents(
            chunks=batch_chunks,
            embeddings=batch_embeddings,
        )

        stored_count += len(batch_chunks)

        print(
            f"Stored {stored_count}/{len(chunks)} chunks."
        )

    if stored_count != len(chunks):
        raise RuntimeError(
            "Stored chunk count does not match chunk count."
        )

    save_fingerprint(
        fingerprint=fingerprint,
        chunk_count=len(chunks),
    )

    print("Knowledge base ingestion completed.")
    print(
        f"Stored chunks: {vector_store.count()}"
    )


def incremental_update(
    vector_store: VectorStore,
    embedding_model: EmbeddingModel,
    chunks: list[dict],
    fingerprint: dict,
) -> None:
    ids = [
        vector_store.document_id(chunk)
        for chunk in chunks
    ]

    existing_ids = vector_store.existing_ids(ids)

    missing_chunks = [
        chunk
        for chunk, chunk_id in zip(
            chunks,
            ids,
        )
        if chunk_id not in existing_ids
    ]

    print(
        f"Already stored: {len(existing_ids)}"
    )

    print(
        f"Remaining to embed: {len(missing_chunks)}"
    )

    if not missing_chunks:
        save_fingerprint(
            fingerprint=fingerprint,
            chunk_count=len(chunks),
        )

        print(
            "All chunks are already stored."
        )

        print(
            f"Stored chunks: {vector_store.count()}"
        )

        return

    stored_so_far = 0

    for (
        batch_chunks,
        batch_embeddings,
    ) in embedding_model.embed_batches(
        missing_chunks,
        batch_size=EMBED_BATCH_SIZE,
        delay_seconds=EMBED_BATCH_DELAY_SECONDS,
        label="chunks",
    ):
        vector_store.add_documents(
            chunks=batch_chunks,
            embeddings=batch_embeddings,
        )

        stored_so_far += len(batch_chunks)

        print(
            f"Stored "
            f"{stored_so_far}/"
            f"{len(missing_chunks)} "
            f"missing chunks."
        )

    save_fingerprint(
        fingerprint=fingerprint,
        chunk_count=len(chunks),
    )

    print(
        "Knowledge base ingestion completed."
    )

    print(
        f"Stored chunks: "
        f"{vector_store.count()}"
    )


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base PDF not found: "
            f"{PDF_PATH}"
        )

    pdf_hash = calculate_file_hash(
        PDF_PATH
    )

    fingerprint = build_fingerprint(
        pdf_hash
    )

    existing_fingerprint = (
        load_existing_fingerprint()
    )

    vector_store = VectorStore()

    documents = load_pdf(PDF_PATH)

    chunks = create_chunks(documents)

    print(
        f"Pages loaded: {len(documents)}"
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    current_count = vector_store.count()

    if (
        fingerprint_matches(
            existing_fingerprint,
            fingerprint,
        )
        and current_count == len(chunks)
    ):
        print(
            "Knowledge base is already up to date."
        )

        print(
            f"Stored chunks: {current_count}"
        )

        print(
            "No document embeddings were generated."
        )

        return

    fingerprint_changed = (
        existing_fingerprint is not None
        and not fingerprint_matches(
            existing_fingerprint,
            fingerprint,
        )
    )

    embedding_model = EmbeddingModel()

    if fingerprint_changed:
        full_rebuild(
            vector_store=vector_store,
            embedding_model=embedding_model,
            chunks=chunks,
            fingerprint=fingerprint,
        )
    else:
        incremental_update(
            vector_store=vector_store,
            embedding_model=embedding_model,
            chunks=chunks,
            fingerprint=fingerprint,
        )


if __name__ == "__main__":
    main()
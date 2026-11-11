import re
from typing import Any

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def _split_into_sections(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sections = []
    current_section = None

    for document in documents:
        text = document["text"].strip()

        if not text:
            continue

        lines = text.splitlines()

        for line in lines:
            line = line.strip()

            if not line:
                continue

            match = re.match(r"^(\d+)[\.\)]\s+(.+)", line)

            if match:
                if current_section is not None:
                    sections.append(current_section)

                current_section = {
                    "question_number": int(match.group(1)),
                    "text": line,
                    "metadata": {
                        **document["metadata"],
                    },
                }
            elif current_section is not None:
                current_section["text"] += " " + line

    if current_section is not None:
        sections.append(current_section)

    return sections


def _split_into_units(text: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n+", text.strip())
    units = []

    for paragraph in paragraphs:
        paragraph = re.sub(r"\s+", " ", paragraph).strip()

        if not paragraph:
            continue

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)

        for sentence in sentences:
            sentence = sentence.strip()

            if sentence:
                units.append(sentence)

    return units


def _split_long_text(text: str) -> list[str]:
    words = text.split()
    pieces = []
    current = []

    for word in words:
        if len(word) > CHUNK_SIZE:
            if current:
                pieces.append(" ".join(current))
                current = []

            for start in range(0, len(word), CHUNK_SIZE):
                pieces.append(word[start:start + CHUNK_SIZE])

            continue

        candidate = " ".join(current + [word])

        if len(candidate) <= CHUNK_SIZE:
            current.append(word)
        else:
            if current:
                pieces.append(" ".join(current))

            current = [word]

    if current:
        pieces.append(" ".join(current))

    return pieces


def _get_overlap_units(units: list[str]) -> list[str]:
    overlap_units = []
    overlap_length = 0

    for unit in reversed(units):
        additional_length = len(unit)

        if overlap_units:
            additional_length += 1

        if overlap_length + additional_length > CHUNK_OVERLAP:
            break

        overlap_units.insert(0, unit)
        overlap_length += additional_length

    return overlap_units


def _build_chunks_from_units(
    units: list[str],
    metadata: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> None:
    expanded_units = []

    for unit in units:
        if len(unit) <= CHUNK_SIZE:
            expanded_units.append(unit)
        else:
            expanded_units.extend(_split_long_text(unit))

    current_units = []
    current_length = 0

    for unit in expanded_units:
        unit_length = len(unit)
        separator_length = 1 if current_units else 0

        if (
            current_units
            and current_length + separator_length + unit_length > CHUNK_SIZE
        ):
            chunks.append(
                {
                    "text": " ".join(current_units).strip(),
                    "metadata": {
                        **metadata,
                        "chunk_id": len(chunks),
                    },
                }
            )

            overlap_units = _get_overlap_units(current_units)
            overlap_text = " ".join(overlap_units)

            if len(overlap_text) + 1 + unit_length <= CHUNK_SIZE:
                current_units = overlap_units
                current_length = len(overlap_text)
            else:
                current_units = []
                current_length = 0

        separator_length = 1 if current_units else 0

        if current_length + separator_length + unit_length > CHUNK_SIZE:
            current_units = [unit]
            current_length = unit_length
        else:
            if current_units:
                current_length += 1

            current_units.append(unit)
            current_length += unit_length

    if current_units:
        chunks.append(
            {
                "text": " ".join(current_units).strip(),
                "metadata": {
                    **metadata,
                    "chunk_id": len(chunks),
                },
            }
        )


def create_chunks(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    sections = _split_into_sections(documents)
    chunks = []

    for section in sections:
        units = _split_into_units(section["text"])

        _build_chunks_from_units(
            units=units,
            metadata=section["metadata"],
            chunks=chunks,
        )

    return chunks


if __name__ == "__main__":
    from pathlib import Path

    from src.pdf_loader import load_pdf

    pdf_path = (
        Path(__file__).resolve().parent.parent
        / "knowledge_base"
        / "top 100 common interview questions and best answers.pdf"
    )

    documents = load_pdf(pdf_path)
    chunks = create_chunks(documents)

    print(f"Pages: {len(documents)}")
    print(f"Chunks: {len(chunks)}")

    target_questions = [
        "2. What are your strengths?",
        "5. Why should we hire you",
        "6. Where do you see yourself in 5 years",
        "9. What do you like the least about your current job",
    ]

    for chunk in chunks:
        text = chunk["text"]

        if any(
            question.lower() in text.lower()
            for question in target_questions
        ):
            print("\n" + "=" * 100)
            print(f"Chunk ID: {chunk['metadata']['chunk_id']}")
            print(f"Page: {chunk['metadata']['page']}")
            print(f"Length: {len(text)}")
            print(text)
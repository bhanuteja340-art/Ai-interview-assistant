from pathlib import Path

from pypdf import PdfReader


def load_pdf(pdf_path: str | Path) -> list[dict]:
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {path}")

    reader = PdfReader(str(path))
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()

        if not text:
            continue

        documents.append(
            {
                "text": text,
                "metadata": {
                    "source": path.name,
                    "page": page_number,
                },
            }
        )

    if not documents:
        raise ValueError(f"No readable text found in PDF: {path}")

    return documents


if __name__ == "__main__":
    pdf_path = Path(__file__).resolve().parent.parent / "knowledge_base" / "top 100 common interview questions and best answers.pdf"

    documents = load_pdf(pdf_path)

    print(f"Pages with text: {len(documents)}")
    print(f"Total characters: {sum(len(document['text']) for document in documents)}")

    for document in documents[:2]:
        print(f"\nPage: {document['metadata']['page']}")
        print(document["text"][:1000])
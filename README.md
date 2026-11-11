# AI Interview Assistant

This POC is a RAG-based interview assistant.

It uses an interview questions and answers PDF as the knowledge source. When a user asks a question, the application retrieves relevant content from the knowledge base and uses Gemini to generate a grounded answer.

## RAG Pipeline

The application flow is:

Knowledge Base
↓
PDF Loader
↓
Section-Aware Chunking
↓
Embeddings
↓
Qdrant Vector Store
↓
User Question
↓
Query Embedding
↓
Retriever
↓
Relevant Contexts
↓
RAG Pipeline
↓
Gemini Generator
↓
Grounded Answer
↓
SQLite Cache

### How it works

1. The interview PDF is loaded from the `knowledge_base` directory.
2. The PDF loader extracts the document text.
3. Section-aware chunking creates retrieval units while preserving useful section context.
4. Embeddings are generated for the document chunks and stored in Qdrant with metadata.
5. A user question is converted into a query embedding and searched against Qdrant.
6. Relevant chunks are passed to the Gemini generator.
7. Gemini generates a grounded answer using the retrieved content.
8. The result is stored in SQLite.
9. If the same question is asked again, the cached result can be returned without repeating retrieval and generation.

## Cache

The cache is implemented using SQLite.

For a new question:
`Question → Cache Miss → Retrieval → Gemini Generation → SQLite Store → Answer`

For the same question again:
`Question → Cache Hit → Stored Answer / Context / Retrieved Chunks`

The cache stores the answer, grounded content, and retrieved chunks.

## Project Structure

```text
Chatbot/
│
├── knowledge_base/
│   └── top 100 common interview questions and best answers.pdf
│
├── src/
│   ├── cache.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── generator.py
│   ├── ingest.py
│   ├── pdf_loader.py
│   ├── retriever.py
│   ├── vector_store.py
│   └── rag_pipeline.py
│
├── evaluation/
│   ├── eval_dataset.py
│   ├── eval_dataset.json
│   ├── neweval_dataset.json
│   └── run_evaluation.py
│
├── TruLens/
│   ├── evaluation.py
│   ├── trulens_evaluation.py
│   ├── trulens_results.json
│   └── trulens_results.csv
│
├── cache/
│   └── chatbot_cache.db
│
├── vector_store/
│   └── ingestion_metadata.json
│
├── app.py
├── main.py
├── requirements.txt
├── .env
├── .env.example
├── README.md
└── .gitignore
```

## Configuration

Create a `.env` file in the project root and configure the Gemini API key:

```text
GEMINI_API_KEY=<your Gemini API key>
```

Do not commit the `.env` file or expose the API key in source code, screenshots, or documentation.

## Setup

The following sequence can be used to set up the POC on another Windows machine.

### Step 1 — Extract the project

Extract the POC ZIP into the required working directory.

```powershell
Expand-Archive -Path "C:\path\to\Chatbot.zip" -DestinationPath "C:\Users\<user>\Desktop\Chatbot"
```

### Step 2 — Open the project directory

```powershell
cd C:\Users\<user>\Desktop\Chatbot
```

### Step 3 — Create the virtual environment

If `.venv` is not already available:

```powershell
python -m venv .venv
```

### Step 4 — Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 5 — Check Python

```powershell
python --version
```

The verified environment used Python 3.11.9.

### Step 6 — Update pip

```powershell
python -m pip install --upgrade pip
```

### Step 7 — Install project dependencies

```powershell
pip install -r requirements.txt
```

### Step 8 — Check dependencies

```powershell
pip check
```

### Step 9 — Verify Gemini configuration

```powershell
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('GEMINI_API_KEY configured:', bool(os.getenv('GEMINI_API_KEY')))"
```

The command checks whether the key is configured without displaying the key itself.

## Prepare the Evaluation Dataset

The controlled evaluation dataset used by the current evaluation flow is:

```text
evaluation/neweval_dataset.json
```

Generate or refresh the dataset with:

```powershell
python -m evaluation.eval_dataset
```

The dataset generation flow uses the interview PDF and produces the controlled evaluation dataset.

## Run the RAG Application

### Ingest the knowledge base

Run ingestion before using a new or changed knowledge base:

```powershell
python -m src.ingest
```

### Run the RAG pipeline directly

```powershell
python -m src.rag_pipeline
```

Enter an interview question when prompted.

### Run the command-line application

```powershell
python main.py
```

### Run the Streamlit application

```powershell
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## RAGAS

RAGAS is used for dataset-oriented RAG evaluation.

Run:

```powershell
python -m evaluation.run_evaluation
```

The current controlled dataset is `evaluation/neweval_dataset.json`.

The RAGAS evaluation uses a Gemini-backed evaluator through the OpenAI-compatible client configuration in `evaluation/run_evaluation.py`.

## TruLens

TruLens is used to evaluate the RAG application using deterministic checks and Gemini-based judging.

The evaluation covers:

- Context Precision
- Context Recall
- Faithfulness
- Answer Relevancy
- Answer Correctness

Run:

```powershell
python .\TruLens\trulens_evaluation.py
```

The TruLens evaluation includes deterministic checks, case selection, Gemini-based evidence judging, metric calculation, feedback records, and the TruLens dashboard.

LiteLLM is used only by the TruLens evaluation component to communicate with Gemini. It is not part of the normal RAG application flow.

## Evaluation Flow

The two evaluation paths are separate from the normal RAG runtime:

```text
RAG Pipeline
├── RAGAS
│   └── Controlled Dataset → RAGAS Metrics → JSON / CSV
│
└── TruLens
    └── Deterministic Checks
        → Case Selection
        → Gemini Evaluation
        → Metric Calculation
        → TruLens Feedback
        → Dashboard
```

## Main Run Order

For a complete setup and execution:

```text
1. Extract the ZIP
2. Open the project directory
3. Create the virtual environment
4. Activate the virtual environment
5. Check Python
6. Update pip
7. Install requirements.txt
8. Run pip check
9. Configure .env
10. Verify GEMINI_API_KEY
11. Generate the evaluation dataset
12. Run knowledge-base ingestion
13. Run the RAG pipeline
14. Run the CLI application
15. Run the Streamlit application
16. Verify cache behaviour using the same question twice
17. Run RAGAS
18. Run TruLens


## Application Entry Points

- app.py — Streamlit web application
- main.py— command-line application
- src/rag_pipeline.py — main RAG pipeline
- src/ingest.py — knowledge-base ingestion
- evaluation/run_evaluation.py — RAGAS evaluation
- TruLens/trulens_evaluation.py — TruLens evaluation


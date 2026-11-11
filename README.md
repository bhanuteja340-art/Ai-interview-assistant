# AI Interview Assistant

A Retrieval-Augmented Generation (RAG) application that answers interview-related questions using an interview preparation PDF as its knowledge base.

The application uses **Google Gemini**, **Qdrant Vector Database**, **Streamlit**, and **SQLite** to provide relevant, source-grounded answers with efficient caching.

## Key Features

- **PDF-based Knowledge Base:** Extracts interview questions and answers from a PDF document.
- **Section-Aware Chunking:** Splits document content into manageable chunks while preserving useful context.
- **Semantic Search:** Uses embeddings and Qdrant to retrieve the top 5 relevant chunks.
- **AI Answer Generation:** Uses Google Gemini to generate natural-language answers grounded in retrieved information.
- **SQLite Caching:** Stores generated answers, supporting content, and retrieved chunks for repeated questions.
- **Source References:** Displays retrieved PDF chunks, page numbers, and distance values.
- **PDF Viewer:** Allows users to view the source document within the Streamlit application.
- **Performance Monitoring:** Displays retrieval, generation, and total response times.
- **Application Logging:** Records request progress, cache behavior, performance, and processing failures.
- **Error Handling:** Handles application and API failures with user-friendly messages.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Web interface | Streamlit |
| Generative AI | Google Gemini |
| Vector database | Qdrant |
| Embeddings | Gemini Embedding API |
| Cache database | SQLite |
| PDF processing | PyPDF |
| Configuration | Python Dotenv |
| Version control | Git and GitHub |

## System Architecture

```text
                    PDF Knowledge Base
                            |
                         PDF Loader
                            |
                    Section-Aware Chunking
                            |
                      Gemini Embeddings
                            |
                       Qdrant Cloud
                            |
                 Stored Document Embeddings


                       User Question
                            |
                       SQLite Cache
                       /          \
                  Cache Hit      Cache Miss
                      |              |
                 Cached Answer   Query Embedding
                      |              |
                      |         Qdrant Retrieval
                      |              |
                      |         Top 5 PDF Chunks
                      |              |
                      |         Gemini Generator
                      |              |
                      |         Grounded Answer
                      |              |
                      |         Store in SQLite
                      |              |
                      +------+-------+
                             |
                     Streamlit Response
                             |
                  Answer + Source References
```

## How It Works

1. The PDF document is loaded from the `knowledge_base` directory.
2. Text is extracted and divided into meaningful chunks.
3. Embeddings are generated and stored in Qdrant with metadata.
4. The user submits an interview question.
5. SQLite checks whether a valid cached response exists.
6. On a cache miss, the question is converted into an embedding.
7. Qdrant retrieves the top 5 matching document chunks.
8. Gemini generates an answer using the retrieved context.
9. The answer, grounded content, and source chunks are stored in SQLite.
10. Streamlit displays the answer, response timings, and PDF references.

## Project Structure

```text
Chatbot/
|
├── knowledge_base/
│   └── top 100 common interview questions and best answers.pdf
|
├── src/
│   ├── __init__.py
│   ├── cache.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── generator.py
│   ├── ingest.py
│   ├── pdf_loader.py
│   ├── rag_pipeline.py
│   ├── retriever.py
│   └── vector_store.py
|
├── app.py
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

Local-only directories, virtual environments, credentials, and cache databases are excluded from GitHub.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/bhanuteja340-art/Ai-interview-assistant.git
cd Ai-interview-assistant
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

Verify dependency compatibility:

```powershell
python -m pip check
```

### 5. Configure environment variables

Create a `.env` file in the project root.

```env
GEMINI_API_KEY=your_gemini_api_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
```

Replace the example values with your own credentials.

Never commit real API keys to GitHub.

### 6. Prepare the knowledge base

The application requires a Qdrant collection containing document embeddings.

To ingest the configured knowledge-base PDF:

```powershell
python -m src.ingest
```

If the existing Qdrant collection already contains the correct embeddings, ingestion does not need to be repeated.

### 7. Run the application

```powershell
python -m streamlit run app.py
```

Open:

`http://localhost:8501`

## SQLite Answer Caching

The application uses SQLite to avoid unnecessary repeated retrieval and generation.

**First request:**

```text
Question → Cache Miss → Qdrant → Gemini → SQLite → Answer
```

**Repeated request:**

```text
Question → Cache Hit → Stored Answer + Source Chunks
```

A cached response can avoid additional Qdrant and Gemini requests, improving response time.

Cache contents are local to the application environment and should not be treated as permanent storage on ephemeral cloud hosting.

## Performance Monitoring

The Streamlit interface displays:

- Retrieval latency
- Generation latency
- Total request latency
- Request status (`success` or `cache_hit`)

The application also records request-processing events using Python's built-in logging module.

Example application logs:

```text
INFO | RAG pipeline initialized
INFO | Question processing started
INFO | Cache miss
INFO | Retrieval completed | chunks=5 | duration=1.892s
INFO | Generation completed | duration=2.004s
INFO | Request completed | status=success | total=3.924s
```

These logs help diagnose slow requests, cache behavior, and processing failures.

Application logs should not include API keys or complete document contents.

## Cloud Deployment

The application is designed to be deployed using Streamlit Community Cloud.

Deployment steps:

1. Push the application code to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Select the GitHub repository.
4. Set the main application file to `app.py`.
5. Configure the required environment credentials privately.
6. Deploy the application.
7. Verify retrieval, generation, PDF viewing, caching, and logs.

A Streamlit secrets configuration can look like:

```toml
GEMINI_API_KEY = "your_gemini_api_key"
QDRANT_URL = "your_qdrant_cluster_url"
QDRANT_API_KEY = "your_qdrant_api_key"
```

The application must be configured to read these values
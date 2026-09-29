# Python RAG with detailed LangSmith retriever traces

A compact reference application that indexes local PDF, Markdown, and text files,
retrieves relevant chunks from a persistent Chroma vector store, and answers with
an open-source model through Hugging Face Inference Providers.

Every question creates one `rag_pipeline` trace with two child spans:

- `vector_similarity_search` (`retriever`): query, top-k, score threshold,
  collection size, embedding model, distance metric, returned text, source,
  page, chunk, rank, cosine distance, and relevance score.
- `hugging_face_chat_completion` (`llm`): messages, model settings, response,
  finish reason, and token usage when the provider returns it.

LangSmith adds timing, status, errors, and the parent-child execution view.

## Setup

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies and create local configuration:

```bash
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then add:

- `HF_TOKEN`: a Hugging Face token with permission to call Inference Providers.
- `LANGSMITH_API_KEY`: a LangSmith API key.

The default LLM is `Qwen/Qwen2.5-7B-Instruct`. Change `HF_MODEL` to another
open-source chat model available through your Hugging Face provider if needed.
The local Chroma embedding model is `all-MiniLM-L6-v2` and requires no API key.

## Run

The repository includes Microsoft's official *Cloud Design Patterns* book as a
substantial system-design sample. It covers 24 patterns and related guidance
for cloud and distributed-system architecture. Index it with:

```bash
python app.py ingest ./documents/cloud-design-patterns.pdf
```

The document is published by Microsoft and is available from the official
[Microsoft Download Center](https://www.microsoft.com/en-us/download/details.aspx?id=42026).

You can instead index any supported file or every supported file below a
directory:

```bash
python app.py ingest ./documents
```

This replaces the existing collection. Use `--append` to retain it:

```bash
python app.py ingest ./more-documents --append
```

Ask a system-design question:

```bash
python app.py ask "When should I use the circuit breaker pattern?"
```

Inspect the exact retrieved text and scores in the terminal as well as LangSmith:

```bash
python app.py ask "When should I use the circuit breaker pattern?" --show-context
```

Start an interactive session:

```bash
python app.py chat
```

Open the project named by `LANGSMITH_PROJECT` in LangSmith and select the
`vector_similarity_search` child run to inspect retrieval. Set
`LANGSMITH_TRACING=false` to run without sending traces.

## Retrieval score

The collection uses cosine distance. Chroma returns distance, where lower is
better; the application records `relevance_score = 1 - distance`, where higher
is better. `MIN_RELEVANCE_SCORE` filters weak results after retrieval.

The local index is written to `.rag_index/` and is intentionally excluded from
Git.

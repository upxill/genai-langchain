# URL Document Q&A (LangChain + FAISS)

A Streamlit app that turns a list of URLs into a queryable knowledge base: it scrapes the pages, chunks and embeds the text, indexes it in FAISS, and answers questions with cited sources using an OpenAI LLM. It's a hands-on implementation of the classic "load → split → embed → retrieve → generate" RAG pattern, built as a compact way to exercise LangChain's core retrieval primitives end to end.

## How this differs from my other RAG/data repos

This is the baseline/reference implementation in the cluster — a single-file, single-pass RAG pipeline over live web pages, with no agent framework, no orchestration graph, and no structured-data component. The other repos build on this foundation: `hybrid-rag-agent` and `apachespark-rag-agent` wrap a similar retrieval step in a LangGraph tool-routing agent alongside Spark DataFrame querying, `cockroachdb-agent` replaces the vector-store-plus-LLM pattern entirely with a CockroachDB-only memory layer, and `csv-spark-kafka` has no LLM/RAG component at all — it's a pure data pipeline.

## Tech Stack

- Python, Streamlit (UI)
- LangChain + `langchain_community` (document loading, chains)
- `UnstructuredURLLoader` (web page ingestion)
- OpenAI (`OpenAI` LLM, `OpenAIEmbeddings`)
- FAISS (`faiss-cpu`) for the vector index
- `tiktoken`, `python-dotenv`

## How it works

The Streamlit UI in `main.py` drives a four-step flow, with intermediate state kept in `st.session_state` between button clicks:

1. **Load Documents** — `UnstructuredURLLoader` fetches and parses the URLs the user pastes into a text area.
2. **Split Documents** — `RecursiveCharacterTextSplitter` chunks the loaded documents (1000 chars, 200 overlap).
3. **Create Vector Store** — `OpenAIEmbeddings` + `FAISS.from_documents` build an in-memory FAISS index over the chunks.
4. **Create QA Chain** — `RetrievalQAWithSourcesChain.from_llm` wraps an `OpenAI(temperature=0)` LLM around the FAISS retriever.

Once the chain exists, typed questions are answered with both an answer and the source documents used to produce it.

## Getting Started

### Prerequisites

- Python 3.8+
- An [OpenAI API key](https://platform.openai.com/account/api-keys)

### Setup

```bash
git clone <this-repo-url>
cd genai-langchain
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key
```

### Run

```bash
streamlit run main.py
```

## Project Structure

This is intentionally a single-file app:

- `main.py` — Streamlit UI + the load/split/embed/retrieve pipeline
- `requirements.txt` — Python dependencies
- `.env` — environment variables (not committed)

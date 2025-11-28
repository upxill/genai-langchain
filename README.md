# GenAI LangChain FAISS

A Streamlit-powered Document Question Answering System using LangChain, OpenAI, and FAISS for vector search.

## Features

- Load documents from URLs
- Split documents into manageable chunks
- Create vector stores using FAISS
- Ask questions and get answers with sources using OpenAI LLM

## Getting Started

### Prerequisites

- Python 3.8+
- [OpenAI API Key](https://platform.openai.com/account/api-keys) 

### Installation

```sh
git clone https://github.com/polaganiearch/AI.git
cd genai_langchain_faiss
pip install -r requirements.txt
```

### Setup

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key
```

### Running the App

```sh
streamlit run main.py
```

## Project Structure

- `main.py` - Streamlit app entry point
- `requirements.txt` - Python dependencies
- `.env` - Environment variables (not committed)
- `README.md` - Project documentation



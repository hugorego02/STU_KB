# STU KB RAG

Semantic search and question-answering system for STU IT knowledge base documents.

## What It Does

- Reads `.docx` knowledge base files from `data_raw/docs_reais`
- Splits documents into chunks
- Generates OpenAI embeddings
- Stores vectors in FAISS
- Answers questions through a Streamlit app or terminal interface

## Main Files

- `vectorize.py`: builds the FAISS index from the KB documents
- `app_streamlit.py`: web chat interface
- `05_rag_terminal.py`: terminal RAG assistant
- `search.py`: terminal retrieval inspection tool
- `kb_config.py`: shared project configuration
- `kb_retrieval.py`: shared embedding and retrieval helpers

## Important Note

Do not commit or share a real `.env` file. Use `.env.example` as the template and set a fresh `OPENAI_API_KEY` locally.

## Run Locally

1. Create a virtual environment and install dependencies from `requirements.txt`
2. Copy `.env.example` to `.env`
3. Add your OpenAI API key to `.env`
4. Build the index:

```powershell
.venv\Scripts\python.exe vectorize.py
```

5. Start the app:

```powershell
.venv\Scripts\python.exe -m streamlit run app_streamlit.py
```

## Current Priorities

- Clean and standardize KB documents before indexing
- Add evaluation questions for retrieval quality
- Improve metadata and source attribution

---
title: RAG QA
emoji: 📓
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.6.0
app_file: app.py
pinned: false
license: mit
short_description: Chat with your documents — RAG with Groq + FAISS
---

# RAG QA — Chat with Your Documents

Retrieval-Augmented Generation (RAG) question answering system using:

- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (local, free)
- **Vector store:** FAISS (persistent for CLI, in-memory for web)
- **LLM:** Groq API — free-tier models (Llama 3.3 / 3.1, Gemma 2, Mixtral)
- **UI:** Gradio (NotebookLM-style document chat)

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
cp .env.example .env              # then edit and add your GROQ_API_KEY
```

Get a free Groq API key at <https://console.groq.com/keys>.

### Tesseract OCR (required for scanned PDFs and image files)

- **Scanned PDFs** — pages with no text layer are auto-detected and OCR'd.
- **Image files** (JPG/PNG/BMP/TIFF/WEBP) — OCR'd directly. Tesseract is **required**, the app will raise a clear error if it's missing.

If Tesseract is not installed, text-layer PDFs / TXT / DOCX still work as normal.

**Windows:**

1. Download installer from <https://github.com/UB-Mannheim/tesseract/wiki>
2. During install, tick the language packs you need (e.g. `English`, `Thai`)
3. Make sure `tesseract.exe` is on `PATH` (default: `C:\Program Files\Tesseract-OCR\`)
4. Verify: `tesseract --version`

**macOS:** `brew install tesseract tesseract-lang`

**Ubuntu/Debian:** `sudo apt install tesseract-ocr tesseract-ocr-tha tesseract-ocr-eng`

OCR language packs used by this project: `eng+tha` (see [src/ingestion.py:13](src/ingestion.py#L13) to change).

## Web App (recommended — NotebookLM-style)

```bash
python src/web_app.py
```

Open <http://localhost:7860>. Then:

1. Paste your **Groq API key**
2. Pick a **model** (default: `llama-3.3-70b-versatile`)
3. **Upload documents** (PDF / TXT / MD / DOCX / **JPG / PNG / TIFF** — multiple OK)
4. Click **Build Index** — creates a fresh in-memory FAISS index for this session
5. Chat away. Sources are shown under each answer.

Click **Clear Session** to drop the index and start over with new documents.

## CLI

Place documents into `data/`, then:

```bash
# 1. Build the persistent index
python src/app.py --build

# 2. Ask questions
python src/app.py --query "What does the document say about X?"
python src/app.py --query "..." --model llama-3.1-8b-instant
```

## Models

| Model ID | Notes |
| -------- | ----- |
| `llama-3.3-70b-versatile` | Best quality (default) |
| `llama-3.1-8b-instant` | Fastest, lightweight |
| `gemma2-9b-it` | Good reasoning |
| `mixtral-8x7b-32768` | 32k context for long docs |

## Configuration

Edit `config.yaml` to change chunk size, top-k retrieval, embedding model, default LLM model, temperature, etc.

## Project Layout

```text
RAG_QA/
├── data/               # Put your source documents here (CLI mode)
├── vectorstore/        # Persisted FAISS index (CLI mode)
├── src/
│   ├── ingestion.py    # Loaders + chunking
│   ├── embeddings.py   # HuggingFace embedding factory
│   ├── retriever.py    # FAISS build/load + retriever
│   ├── qa_chain.py     # Groq LLM + ConversationalRetrievalChain
│   ├── app.py          # CLI
│   └── web_app.py      # Gradio NotebookLM-style UI
├── config.yaml
└── requirements.txt
```

## Notes

- API key is read from `GROQ_API_KEY` env var (CLI) or the web form (web app); it is never written to disk.
- For Thai documents, swap the embedding model in `config.yaml` to `intfloat/multilingual-e5-base`.

# ARIA — Academic Research and Information Assistant

A fully local, **multimodal RAG (Retrieval-Augmented Generation)** system that lets you chat with your documents, images, and audio files. Everything runs on your own machine — no API keys, no data leaving your computer.

## ✨ Features

- **Multimodal ingestion — one unified index**
  - 📄 **Text** (`pdf`, `docx`, `txt`, `odt`) — extracted and chunked
  - 🖼️ **Images** (`png`, `jpg`, `jpeg`, `webp`, `bmp`) — captioned by **Qwen2.5-VL** via Ollama; captions are indexed
  - 🎞️ **PDF figures** — large embedded images inside PDFs are optionally extracted and captioned too
  - 🎙️ **Audio** (`mp3`, `wav`, `m4a`, `flac`, `ogg`) — transcribed by **faster-whisper** with `[mm:ss]` timestamps, so answers can cite the exact moment
- **Single FAISS index** (cosine similarity, MiniLM embeddings) — retrieval and chat stay identical regardless of the original modality
- **Web UI (Gradio)** — two-panel layout: chat on the left, file upload and model controls on the right
- **Smart caching** — the index is fingerprinted against your `documents/` folder and only rebuilt when files change
- **Grounded answers with sources** — the model answers *only* from retrieved context and lists the source files (including figure page numbers and audio timestamps)
- **100% local** — LLM runs in Ollama, embeddings and Whisper run locally

## 🖥️ UI Overview

| Area | What it does |
|------|--------------|
| **Chat panel (left)** | Ask questions; ARIA answers from your knowledge base with cited sources |
| **Upload Files Here** | Drag-and-drop or click to add documents, images, or audio |
| **Upload Data** | Processes uploads (caption / transcribe / chunk) and adds them to the index |
| **Clear RAG Database** | Wipes the knowledge base and cache |
| **Knowledge base status** | Live chunk and document count (e.g. *509 chunks · 18 documents*) |
| **Model Temperature** | 0–1 slider (default 0.1) for answer creativity |
| **Max Output Tokens** | 128–4096 slider (default 512) for answer length |

## 🏗️ Architecture

```
documents/  (pdf, docx, txt, odt, images, audio)
     │
     ├── text extraction ──────────────┐
     ├── image → Qwen2.5-VL caption ───┤
     ├── PDF figures → caption ────────┼──► text chunks (300 words, 50 overlap)
     └── audio → faster-whisper ───────┘
                                       │
                     all-MiniLM-L6-v2 embeddings (normalized)
                                       │
                             FAISS IndexFlatIP (cosine)
                                       │
        query ──► retrieve top-k ──► prompt ──► Qwen2.5-VL (Ollama) ──► answer + sources
```

Two processes share the work:

- **`rag_multimodal.py`** — GPU index builder + CLI chat. Uses CUDA for embeddings and Whisper.
- **`rag_gradio_multimodal.py`** — CUDA-free Gradio web app. Loads the prebuilt index from `rag_cache/` and encodes queries/uploads on CPU. Qwen runs on GPU inside Ollama's separate process, so the two never fight over VRAM.

## 📋 Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (for the index builder; the web UI itself runs on CPU)
- [Ollama](https://ollama.com) installed and running

## 🚀 Installation

```bash
# 1. Pull the vision-language model
ollama pull qwen2.5vl:3b

# 2. Install Python dependencies
pip install torch faiss-cpu sentence-transformers gradio ollama \
            pymupdf python-docx odfpy faster-whisper numpy
```

> **Note (Windows):** import order matters — `torch` must be imported before `faiss`. The scripts already handle this.

## ▶️ Usage

**1. Build the index** (first run, or after bulk changes to `documents/`):

```bash
python rag_multimodal.py
```

Drop your files into the `documents/` folder first. This also gives you a terminal chat loop for quick testing.

**2. Launch the web UI:**

```bash
python rag_gradio_multimodal.py
```

Then open **http://127.0.0.1:7860**.

**3. Chat!** You can also upload new files directly from the UI — they're captioned/transcribed, chunked, embedded, and appended to the live index.

## ⚙️ Configuration

Key settings at the top of the scripts:

| Setting | Default | Description |
|---------|---------|-------------|
| `VL_MODEL` / `CHAT_MODEL` | `qwen2.5vl:3b` | Ollama model used for captions and chat |
| `WHISPER_MODEL` | `large-v3` | faster-whisper model for audio transcription |
| `EXTRACT_PDF_IMAGES` | `True` | Also caption figures embedded in PDFs |
| `MIN_FIG_SIDE` | `300` | Skip decorative images smaller than this (px) |
| Chunk size / overlap | `300` / `50` words | Text chunking parameters |

## 📁 Project Structure

```
.
├── rag_multimodal.py           # GPU index builder + CLI chat
├── rag_gradio_multimodal.py    # Gradio web UI (ARIA)
├── documents/                  # your source files go here
├── rag_cache/                  # generated: kb.index, chunks.pkl, fingerprint.json
└── docs/
    └── UI.jpg                  # UI screenshot
```

## 🧠 How answers stay grounded

Every query retrieves the top-5 most similar chunks. The prompt instructs the model to answer **only** from that context, and to reply *"I could not find this information in the provided documents."* when the answer isn't there — reducing hallucination. Sources (filenames, figure pages, audio timestamps) are returned alongside each answer.

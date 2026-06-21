# 🔬 Research RAG System

A GPU-accelerated Retrieval-Augmented Generation (RAG) system for querying research papers using **Qwen 2.5 7B** and **FAISS**.

---

## 📌 Overview

This system allows you to ask natural language questions against a collection of research papers and get accurate, context-grounded answers. It reads your documents, breaks them into chunks, stores them as vector embeddings, and retrieves the most relevant passages to answer your query using a local LLM — entirely offline, no API keys needed.

---

## 🏗️ How It Works

```
Documents (PDF / DOCX / TXT / ODT)
        ↓
  Text Extraction
        ↓
  Preprocessing & Chunking
  (300 words per chunk, 50 word overlap)
        ↓
  Embeddings — SentenceTransformer
  (all-MiniLM-L6-v2) on GPU
        ↓
  FAISS Vector Index (IndexFlatL2)
        ↓
  User Question → Query Embedding
        ↓
  Top-5 Relevant Chunks Retrieved
        ↓
  Prompt + Context → Qwen 2.5 7B via Ollama (GPU)
        ↓
  Final Answer + Source Documents
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Qwen 2.5 7B (Q4 quantized) via Ollama |
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Vector Store | FAISS (IndexFlatL2) |
| GPU Acceleration | CUDA via PyTorch |
| Document Parsing | PyMuPDF, python-docx, odfpy |

---

## 📄 Supported File Formats

| Format | Library Used |
|---|---|
| `.pdf` | PyMuPDF (fitz) |
| `.docx` | python-docx |
| `.txt` | Built-in Python |
| `.odt` | odfpy |

---

## ⚙️ RAG Pipeline — Detailed

### 1. Text Extraction
Each document in the `documents/` folder is read and raw text is extracted based on its file type using dedicated parsers for PDF, DOCX, TXT, and ODT formats.

### 2. Preprocessing
Extracted text is cleaned by collapsing all extra whitespace into single spaces, ensuring consistent token boundaries for chunking.

### 3. Chunking
Text is split into overlapping word chunks:
- **Chunk size:** 300 words
- **Overlap:** 50 words

The overlap ensures that context is not lost at chunk boundaries — a sentence split across two chunks will still be retrievable.

### 4. Embedding
Each chunk is converted into a 384-dimensional dense vector using `all-MiniLM-L6-v2`, a lightweight but highly accurate sentence embedding model. Embeddings are generated in batches of 64 on GPU for speed.

### 5. FAISS Indexing
All chunk embeddings are stored in a FAISS `IndexFlatL2` index, which performs exact nearest-neighbour search using L2 (Euclidean) distance. For research papers (~300–500 chunks), this is extremely fast — under 1ms per query.

### 6. Retrieval
When a question is asked, it is embedded using the same model. FAISS searches the index and returns the top-5 most semantically similar chunks along with their source filenames.

### 7. Generation
The retrieved chunks are assembled into a structured prompt with source labels and passed to **Qwen 2.5 7B** running locally via Ollama. The model is instructed to answer strictly from the provided context and to say "I could not find this information" if the answer is not present — preventing hallucination.

---

## 🔧 Model Parameters (Qwen 2.5 7B)

| Parameter | Value | Purpose |
|---|---|---|
| `temperature` | `0.1` | Low randomness → factual, deterministic answers |
| `num_predict` | `300` | Max output length (~225 words) |
| `top_p` | `0.9` | Filters bottom 10% unlikely word choices |
| `repeat_penalty` | `1.1` | Reduces repetitive output |
| `num_gpu` | `99` | Forces all 32 model layers onto GPU |

---

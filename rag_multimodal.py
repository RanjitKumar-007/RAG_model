"""
Multimodal RAG index builder — text, images, and audio.

  Text  (pdf/docx/txt/odt) : extracted as before
  Images (png/jpg/...)     : captioned by Qwen2.5-VL 7B via Ollama, caption is indexed
  PDF figures              : optionally extracted and captioned too (EXTRACT_PDF_IMAGES)
  Audio (mp3/wav/...)      : transcribed by faster-whisper, transcript is indexed
                             with [mm:ss] markers so answers can cite timestamps

Everything ends up as text chunks in ONE FAISS index (MiniLM embeddings),
so retrieval and the chat pipeline stay unchanged.

Requirements:
  ollama pull qwen2.5vl:3b
  pip install faster-whisper
"""

import os
import io
import json
import pickle
import hashlib

# --- Import order matters on Windows: torch first, then faiss.
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import fitz
import docx
from odf.opendocument import load
from odf import teletype
from odf.text import P
import ollama

# ---------------- Config ----------------
VL_MODEL           = "qwen2.5vl:3b"   # captions images; also your chat model now
WHISPER_MODEL      = "large-v3"          # Systran/faster-whisper-large-v3 (already in your HF cache)
EXTRACT_PDF_IMAGES = True             # also caption figures embedded in PDFs
MIN_FIG_SIDE       = 300              # skip tiny decorative images (px)

TEXT_EXT  = {"pdf", "docx", "txt", "odt"}
IMAGE_EXT = {"png", "jpg", "jpeg", "webp", "bmp"}
AUDIO_EXT = {"mp3", "wav", "m4a", "flac", "ogg"}
SUPPORTED_EXT = TEXT_EXT | IMAGE_EXT | AUDIO_EXT

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not found! Make sure your GPU drivers and CUDA are installed.")

device = "cuda"
print("=" * 50)
print(f"  GPU : {torch.cuda.get_device_name(0)}")
print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 50)

embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Lazy-loaded whisper (only if audio files exist)
_whisper = None
def get_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        print(f"Loading faster-whisper '{WHISPER_MODEL}' on GPU...")
        try:
            _whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
        except Exception:
            print("GPU whisper failed → falling back to CPU (int8)")
            _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _whisper

CACHE_DIR   = "rag_cache"
INDEX_PATH  = os.path.join(CACHE_DIR, "kb.index")
CHUNKS_PATH = os.path.join(CACHE_DIR, "chunks.pkl")
FPRINT_PATH = os.path.join(CACHE_DIR, "fingerprint.json")

# ---------------- Text extraction ----------------
def read_pdf(path):
    text = ""
    doc = fitz.open(path)
    for page in doc:
        text += page.get_text()
    return text

def read_docx(path):
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)

def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_odt(path):
    d = load(path)
    return "\n".join(teletype.extractText(p) for p in d.getElementsByType(P))

def extract_text(file_path):
    ext = file_path.lower().split(".")[-1]
    if ext == "pdf":
        return read_pdf(file_path)
    if ext == "docx":
        return read_docx(file_path)
    if ext == "txt":
        return read_txt(file_path)
    if ext == "odt":
        return read_odt(file_path)
    return ""

# ---------------- Image captioning (Qwen2.5-VL) ----------------
CAPTION_PROMPT = (
    "Describe this image in detail for a research document search system. "
    "Include: what the image shows, any text/labels/axis names you can read, "
    "chart or diagram type, key numbers or trends, and its likely purpose. "
    "Be factual and thorough."
)

def caption_image_bytes(img_bytes):
    resp = ollama.chat(
        model=VL_MODEL,
        messages=[{"role": "user", "content": CAPTION_PROMPT, "images": [img_bytes]}],
        options={"temperature": 0.1, "num_predict": 400},
    )
    return resp["message"]["content"].strip()

def caption_image_file(path):
    with open(path, "rb") as f:
        return caption_image_bytes(f.read())

def extract_pdf_figures(path, filename):
    """Caption large embedded images inside a PDF. Returns list of (chunk, source)."""
    out = []
    doc = fitz.open(path)
    for page_num, page in enumerate(doc, start=1):
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.width < MIN_FIG_SIDE or pix.height < MIN_FIG_SIDE:
                    continue
                if pix.n - pix.alpha >= 4:          # CMYK → RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                caption = caption_image_bytes(pix.tobytes("png"))
                chunk = f"[Figure from {filename}, page {page_num}] {caption}"
                out.append((chunk, f"{filename} (figure p.{page_num})"))
                print(f"    figure p.{page_num} captioned")
            except Exception as e:
                print(f"    figure p.{page_num} skipped ({e})")
    return out

# ---------------- Audio transcription (faster-whisper) ----------------
def transcribe_audio(path, filename):
    """Return transcript text with [mm:ss] markers per segment."""
    model = get_whisper()
    segments, info = model.transcribe(path, vad_filter=True)
    parts = []
    for seg in segments:
        m, s = divmod(int(seg.start), 60)
        parts.append(f"[{m:02d}:{s:02d}] {seg.text.strip()}")
    print(f"    transcribed ({info.duration:.0f}s of audio, lang={info.language})")
    return "\n".join(parts)

# ---------------- Chunking / fingerprint / cache ----------------
def preprocess(text):
    return " ".join(text.split())

def chunk_text(text, size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        if len(chunk) > 100:
            chunks.append(chunk)
    return chunks

def folder_fingerprint(data_path, supported_ext):
    entries = []
    for f in sorted(os.listdir(data_path)):
        fp = os.path.join(data_path, f)
        if os.path.isfile(fp) and f.lower().split(".")[-1] in supported_ext:
            st = os.stat(fp)
            entries.append(f"{f}|{st.st_size}|{st.st_mtime_ns}")
    return hashlib.md5("\n".join(entries).encode()).hexdigest()

def save_knowledge_base(index, chunks, metadata, fingerprint):
    os.makedirs(CACHE_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "metadata": metadata}, f)
    with open(FPRINT_PATH, "w") as f:
        json.dump({"fingerprint": fingerprint}, f)
    print(f"Knowledge base saved to '{CACHE_DIR}/'")

def load_knowledge_base(fingerprint):
    if not (os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH) and os.path.exists(FPRINT_PATH)):
        return None
    try:
        with open(FPRINT_PATH) as f:
            saved = json.load(f).get("fingerprint")
        if saved != fingerprint:
            print("Documents changed since last run → rebuilding index...")
            return None
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded cached index: {index.ntotal} vectors, {len(data['chunks'])} chunks")
        return index, data["chunks"], data["metadata"]
    except Exception as e:
        print(f"Cache load failed ({e}) → rebuilding index...")
        return None

# ---------------- Per-file processing ----------------
def process_file(file_path, filename):
    """Return (chunks, metadata) lists for one file of any supported type."""
    ext = filename.lower().split(".")[-1]

    if ext in TEXT_EXT:
        text = preprocess(extract_text(file_path))
        chunks = chunk_text(text)
        meta = [filename] * len(chunks)
        if ext == "pdf" and EXTRACT_PDF_IMAGES:
            for chunk, src in extract_pdf_figures(file_path, filename):
                chunks.append(chunk)
                meta.append(src)
        return chunks, meta

    if ext in IMAGE_EXT:
        caption = caption_image_file(file_path)
        chunk = f"[Image: {filename}] {caption}"
        return [chunk], [f"{filename} (image)"]

    if ext in AUDIO_EXT:
        transcript = preprocess(transcribe_audio(file_path, filename))
        chunks = chunk_text(transcript)
        return chunks, [f"{filename} (audio)"] * len(chunks)

    return [], []

# ---------------- Build ----------------
def build_knowledge_base(data_path):
    fingerprint = folder_fingerprint(data_path, SUPPORTED_EXT)
    cached = load_knowledge_base(fingerprint)
    if cached is not None:
        return cached

    files = [f for f in os.listdir(data_path)
             if os.path.isfile(os.path.join(data_path, f))
             and f.lower().split(".")[-1] in SUPPORTED_EXT]
    if not files:
        print("No supported files found in the folder!")
        return None, [], []
    print(f"Found {len(files)} file(s): {files}\n")

    all_chunks, metadata = [], []
    for file in files:
        file_path = os.path.join(data_path, file)
        try:
            print(f"{file} ...")
            chunks, meta = process_file(file_path, file)
            all_chunks.extend(chunks)
            metadata.extend(meta)
            print(f"{file} → {len(chunks)} chunks")
        except Exception as e:
            print(f"{file} → Error: {e}")

    if not all_chunks:
        print("No chunks generated. Check your documents.")
        return None, [], []

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Generating embeddings...")
    embeddings = embed_model.encode(
        all_chunks, batch_size=64, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    print(f"FAISS index built with {index.ntotal} vectors (dim={embeddings.shape[1]}, cosine)")

    save_knowledge_base(index, all_chunks, metadata, fingerprint)
    return index, all_chunks, metadata

# ---------------- Retrieval + answer (VL model) ----------------
def retrieve(question, index, chunks, metadata, k=5):
    q_emb = embed_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(q_emb.astype(np.float32), k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({"chunk": chunks[idx], "source": metadata[idx], "score": float(score)})
    return results

def answer_question(question, index, chunks, metadata):
    results = retrieve(question, index, chunks, metadata, k=5)
    context = "\n\n".join(f"[Source {i+1}: {r['source']}]\n{r['chunk']}"
                          for i, r in enumerate(results))
    sources = list(dict.fromkeys(r["source"] for r in results))
    prompt = f"""You are a research assistant. Answer the question using ONLY the provided context.
The context may include text from papers, captions of images/figures, and audio transcripts with [mm:ss] timestamps.
If the answer is not found in the context, say "I could not find this information in the provided documents."
Do not make up information. Be concise and precise.
Context:
{context}
Question: {question}
Answer:"""
    response = ollama.chat(
        model=VL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1, "num_predict": 512, "top_p": 0.9,
                 "repeat_penalty": 1.1, "num_gpu": 99},
    )
    return response["message"]["content"], sources

# ---------------- Main ----------------
if __name__ == "__main__":
    data_path = "documents"
    print("=" * 50)
    print("  Multimodal RAG — Qwen2.5-VL 7B + Whisper + FAISS")
    print("=" * 50)
    print(f"\nLoading documents from: '{data_path}/'")
    index, chunks, metadata = build_knowledge_base(data_path)
    if index is None:
        exit(1)
    print("\nRAG Ready! Type your question (or 'exit' to quit)\n" + "-" * 50)
    while True:
        q = input("\n Question: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("Bye!")
            break
        ans, src = answer_question(q, index, chunks, metadata)
        print(f"\nAnswer:\n{ans}")
        print(f"\nSources: {', '.join(src)}")
        print("-" * 50)

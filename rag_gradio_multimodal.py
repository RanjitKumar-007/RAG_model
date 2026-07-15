"""
ARIA — Academic Research and Information Assistant
Gradio UI (two-panel layout: chat on the left, controls on the right).

CUDA-free web process: loads the prebuilt index from rag_cache/, encodes
queries and uploaded documents on CPU. Qwen runs on GPU inside Ollama's
separate process.

Workflow:
  1) First time (or bulk changes): build the index with  python rag_2.py
  2) Run the UI:  python rag_gradio.py
  3) Open http://127.0.0.1:7860
"""

import os

# Keep CUDA out of this process — CPU encoding is fine here.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import pickle
import shutil
import hashlib
import threading

import numpy as np
import faiss
import ollama
import gradio as gr
from sentence_transformers import SentenceTransformer

import fitz                     # PyMuPDF
import docx
from odf.opendocument import load as odf_load
from odf import teletype
from odf.text import P

CACHE_DIR   = "rag_cache"
DOCS_DIR    = "documents"
INDEX_PATH  = os.path.join(CACHE_DIR, "kb.index")
CHUNKS_PATH = os.path.join(CACHE_DIR, "chunks.pkl")
FPRINT_PATH = os.path.join(CACHE_DIR, "fingerprint.json")

TEXT_EXT      = {"pdf", "docx", "txt", "odt"}
IMAGE_EXT     = {"png", "jpg", "jpeg", "webp", "bmp"}
AUDIO_EXT     = {"mp3", "wav", "m4a", "flac", "ogg"}
SUPPORTED_EXT = TEXT_EXT | IMAGE_EXT | AUDIO_EXT
EMBED_DIM     = 384              # all-MiniLM-L6-v2
CHAT_MODEL    = "qwen2.5vl:3b"   # Qwen2.5-VL: text + images (audio handled by Whisper)
WHISPER_MODEL = "large-v3"

kb_lock = threading.Lock()

# ---------------------------------------------------------------
# Text extraction / chunking (same logic as rag_2.py)
# ---------------------------------------------------------------
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
    d = odf_load(path)
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

def preprocess(text):
    return " ".join(text.split())

# ---------------------------------------------------------------
# Image captioning (Qwen2.5-VL via Ollama) + audio transcription
# ---------------------------------------------------------------
CAPTION_PROMPT = (
    "Describe this image in detail for a research document search system. "
    "Include: what the image shows, any text/labels/axis names you can read, "
    "chart or diagram type, key numbers or trends, and its likely purpose. "
    "Be factual and thorough."
)

def caption_image_file(path):
    with open(path, "rb") as f:
        img_bytes = f.read()
    resp = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": CAPTION_PROMPT, "images": [img_bytes]}],
        options={"temperature": 0.1, "num_predict": 400},
    )
    return resp["message"]["content"].strip()

_whisper = None
def get_whisper():
    """Lazy-load faster-whisper on CPU (this process is CUDA-free)."""
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _whisper

def transcribe_audio(path):
    segments, info = get_whisper().transcribe(path, vad_filter=True)
    parts = []
    for seg in segments:
        m, s = divmod(int(seg.start), 60)
        parts.append(f"[{m:02d}:{s:02d}] {seg.text.strip()}")
    return "\n".join(parts)

def process_file(file_path, filename):
    """Return (chunks, metadata) for one file of any supported type."""
    ext = filename.lower().split(".")[-1]
    if ext in TEXT_EXT:
        text = preprocess(extract_text(file_path))
        file_chunks = chunk_text(text)
        return file_chunks, [filename] * len(file_chunks)
    if ext in IMAGE_EXT:
        caption = caption_image_file(file_path)
        return [f"[Image: {filename}] {caption}"], [f"{filename} (image)"]
    if ext in AUDIO_EXT:
        transcript = preprocess(transcribe_audio(file_path))
        file_chunks = chunk_text(transcript)
        return file_chunks, [f"{filename} (audio)"] * len(file_chunks)
    return [], []

def chunk_text(text, size=300, overlap=50):
    words = text.split()
    out = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        if len(chunk) > 100:
            out.append(chunk)
    return out

def folder_fingerprint(data_path):
    """Must match rag_2.py exactly so the CLI script trusts our cache."""
    entries = []
    if os.path.isdir(data_path):
        for f in sorted(os.listdir(data_path)):
            fp = os.path.join(data_path, f)
            if os.path.isfile(fp) and f.lower().split(".")[-1] in SUPPORTED_EXT:
                st = os.stat(fp)
                entries.append(f"{f}|{st.st_size}|{st.st_mtime_ns}")
    return hashlib.md5("\n".join(entries).encode()).hexdigest()

def save_knowledge_base():
    os.makedirs(CACHE_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "metadata": metadata}, f)
    with open(FPRINT_PATH, "w") as f:
        json.dump({"fingerprint": folder_fingerprint(DOCS_DIR)}, f)

# ---------------------------------------------------------------
# Load prebuilt index + chunks (or start empty)
# ---------------------------------------------------------------
if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        _data = pickle.load(f)
    chunks, metadata = _data["chunks"], _data["metadata"]
    print(f"Loaded index: {index.ntotal} vectors, {len(set(metadata))} documents")
else:
    index = faiss.IndexFlatIP(EMBED_DIM)
    chunks, metadata = [], []
    print("No prebuilt index found — starting with an empty knowledge base. "
          "Upload documents in the UI or run 'python rag_2.py'.")

print("Loading encoder (CPU)...")
encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
print("Ready.")


def kb_status_md():
    n_docs = len(set(metadata))
    return f"**Knowledge base:** {index.ntotal} chunks · {n_docs} documents"

# ---------------------------------------------------------------
# Upload / clear database handlers
# ---------------------------------------------------------------
def upload_data(files, progress=gr.Progress()):
    if not files:
        return "No files selected. Drop files above, then click **Upload Data**.", kb_status_md()

    os.makedirs(DOCS_DIR, exist_ok=True)
    # metadata entries can be "file.pdf", "img.png (image)", "talk.mp3 (audio)"
    existing = {m.split(" (")[0] for m in metadata}
    report = []

    new_chunks, new_meta = [], []
    for f in progress.tqdm(files, desc="Processing files"):
        src = f.name if hasattr(f, "name") else f
        name = os.path.basename(src)
        ext = name.lower().split(".")[-1]
        if ext not in SUPPORTED_EXT:
            report.append(f"⚠️ {name} — unsupported type, skipped")
            continue
        if name in existing:
            report.append(f"⏭️ {name} — already indexed, skipped")
            continue
        try:
            dest = os.path.join(DOCS_DIR, name)
            shutil.copy(src, dest)
            file_chunks, file_meta = process_file(dest, name)
            if not file_chunks:
                report.append(f"⚠️ {name} — no usable content, skipped")
                continue
            new_chunks.extend(file_chunks)
            new_meta.extend(file_meta)
            kind = ("image" if ext in IMAGE_EXT else
                    "audio" if ext in AUDIO_EXT else "text")
            report.append(f"✅ {name} ({kind}) — {len(file_chunks)} chunks")
        except Exception as e:
            report.append(f"❌ {name} — error: {e}")

    if not new_chunks:
        return "\n\n".join(report) or "Nothing to add.", kb_status_md()

    progress(0.8, desc="Embedding new chunks (CPU)...")
    embeddings = encoder.encode(
        new_chunks,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    with kb_lock:
        index.add(embeddings)
        chunks.extend(new_chunks)
        metadata.extend(new_meta)
        save_knowledge_base()

    report.append(f"**Index updated:** +{len(new_chunks)} chunks, cache saved.")
    return "\n\n".join(report), kb_status_md()


def clear_rag_database():
    global index, chunks, metadata
    with kb_lock:
        index = faiss.IndexFlatIP(EMBED_DIM)
        chunks, metadata = [], []
        for p in (INDEX_PATH, CHUNKS_PATH, FPRINT_PATH):
            if os.path.exists(p):
                os.remove(p)
    return "🗑️ RAG database cleared. Upload documents to start again.", kb_status_md()

# ---------------------------------------------------------------
# Retrieval + chat
# ---------------------------------------------------------------
def retrieve(question, k=5):
    if index.ntotal == 0:
        return []
    q_emb = encoder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    with kb_lock:
        scores, indices = index.search(q_emb.astype(np.float32), min(k, index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "chunk": chunks[idx],
                "source": metadata[idx],
                "score": float(score),
            })
    return results


def chat(message, chat_display, chat_memory, temperature, max_tokens):
    """Streaming handler. chat_display drives the Chatbot UI; chat_memory is
    the conversation history sent to the model (cleared independently)."""
    message = (message or "").strip()
    if not message:
        yield chat_display, chat_memory, ""
        return

    chat_display = chat_display + [{"role": "user", "content": message}]

    results = retrieve(message, k=5)
    if not results:
        reply = ("The knowledge base is empty. Upload documents on the right "
                 "(**Chat With Files → Upload Data**) and ask again.")
        chat_display = chat_display + [{"role": "assistant", "content": reply}]
        yield chat_display, chat_memory, ""
        return

    sources = list(dict.fromkeys(r["source"] for r in results))
    context = "\n\n".join(
        f"[Source {i+1}: {r['source']}]\n{r['chunk']}" for i, r in enumerate(results)
    )
    prompt = f"""You are ARIA, an academic research assistant. Answer the question using ONLY the provided context.
The context may include text from papers, captions of images, and audio transcripts with [mm:ss] timestamps.
If the answer is not found in the context, say "I could not find this information in the provided documents."
Do not make up information. Be concise and precise.
Context:
{context}
Question: {message}
Answer:"""

    # conversation memory + current RAG-augmented question
    messages = chat_memory + [{"role": "user", "content": prompt}]

    chat_display = chat_display + [{"role": "assistant", "content": ""}]
    partial = ""
    try:
        stream = ollama.chat(
            model=CHAT_MODEL,
            messages=messages,
            stream=True,
            options={
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_gpu": 99,
            },
        )
        for part in stream:
            partial += part["message"]["content"]
            chat_display[-1]["content"] = partial
            yield chat_display, chat_memory, ""
    except Exception as e:
        chat_display[-1]["content"] = partial + f"\n\n[Error from Ollama: {e}]"
        yield chat_display, chat_memory, ""
        return

    final = partial + "\n\n*Sources: " + ", ".join(sources) + "*"
    chat_display[-1]["content"] = final

    # store the plain question/answer (not the huge RAG prompt) in memory
    chat_memory = chat_memory + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": partial},
    ]
    yield chat_display, chat_memory, ""


def clear_chat_window(chat_memory):
    return [], chat_memory

def clear_chat_and_memory():
    return [], []

# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
# Force dark mode regardless of the browser/OS preference.
FORCE_DARK_JS = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'dark') {
    url.searchParams.set('__theme', 'dark');
    window.location.href = url.href;
  }
}
"""

# Keep the two knowledge-base buttons the same size on one line.
CUSTOM_CSS = """
.kb-btn {
  white-space: nowrap !important;
  height: 44px !important;
  font-size: 14px !important;
}
"""

with gr.Blocks(
    title="ARIA — Academic Research and Information Assistant",
) as demo:
    gr.Markdown("# ARIA: Academic Research and Information Assistant")

    chat_memory = gr.State([])

    with gr.Row():
        # ---------------- Left: chat panel ----------------
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(label="ARIA", height=620)  # Gradio 6: messages format is the default (and only) format
            msg_box = gr.Textbox(
                placeholder="Enter your message here and hit return when you're ready...",
                show_label=False,
            )
            with gr.Row():
                clear_win_btn = gr.Button("Clear Chat Window")
                clear_all_btn = gr.Button("Clear Chat Window and Chat Memory")

        # ---------------- Right: controls ----------------
        with gr.Column(scale=3):
            with gr.Tab("Chat With Files"):
                file_input = gr.File(
                    label="Upload Files Here (documents, images, audio)",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".txt", ".odt",
                                ".png", ".jpg", ".jpeg", ".webp", ".bmp",
                                ".mp3", ".wav", ".m4a", ".flac", ".ogg"],
                )
                with gr.Row():
                    upload_btn = gr.Button(
                        "Upload Data", variant="primary",
                        scale=1, min_width=140, elem_classes="kb-btn",
                    )
                    clear_db_btn = gr.Button(
                        "Clear RAG Database",
                        scale=1, min_width=140, elem_classes="kb-btn",
                    )
                upload_status = gr.Markdown()
                kb_status = gr.Markdown(kb_status_md())

            with gr.Group():
                gr.Markdown("**Chat Model**\n\nQwen 2.5 7B (via Ollama)")
                temperature = gr.Slider(
                    0, 1, value=0.1, step=0.05,
                    label="Model Temperature",
                    info="Select a temperature between 0 and 1 for the model.",
                )
                max_tokens = gr.Slider(
                    128, 4096, value=512, step=64,
                    label="Max Output Tokens",
                    info="Set the maximum number of tokens the model can respond with.",
                )

    # ---------------- Events ----------------
    msg_box.submit(
        fn=chat,
        inputs=[msg_box, chatbot, chat_memory, temperature, max_tokens],
        outputs=[chatbot, chat_memory, msg_box],
    )
    upload_btn.click(
        fn=upload_data,
        inputs=[file_input],
        outputs=[upload_status, kb_status],
    )
    clear_db_btn.click(
        fn=clear_rag_database,
        outputs=[upload_status, kb_status],
    )
    clear_win_btn.click(
        fn=clear_chat_window,
        inputs=[chat_memory],
        outputs=[chatbot, chat_memory],
    )
    clear_all_btn.click(
        fn=clear_chat_and_memory,
        outputs=[chatbot, chat_memory],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        js=FORCE_DARK_JS,      # Gradio 6: js/css moved from Blocks() to launch()
        css=CUSTOM_CSS,
    )

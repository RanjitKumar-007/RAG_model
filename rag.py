import os
import fitz
import docx
from odf.opendocument import load
from odf.text import P
import faiss
import numpy as np
import ollama
import torch
from sentence_transformers import SentenceTransformer

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not found! Make sure your GPU drivers and CUDA are installed.")

device = "cuda"
gpu_name = torch.cuda.get_device_name(0)
vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9

print("=" * 50)
print(f"  GPU : {gpu_name}")
print(f"  VRAM: {vram_gb:.1f} GB")
print("=" * 50)

#Embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

#File reading
def read_pdf(path):
    text = ""
    doc = fitz.open(path)
    for page in doc:
        text += page.get_text()
    return text

def read_docx(path):
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_odt(path):
    doc = load(path)
    paragraphs = doc.getElementsByType(P)
    return "\n".join([p.firstChild.data if p.firstChild else "" for p in paragraphs])

#Text extraction
def extract_text(file_path):
    ext = file_path.lower().split('.')[-1]
    if ext == "pdf":
        return read_pdf(file_path)
    elif ext == "docx":
        return read_docx(file_path)
    elif ext == "txt":
        return read_txt(file_path)
    elif ext == "odt":
        return read_odt(file_path)
    return ""


# Preprocessing
def preprocess(text):
    return " ".join(text.split())

#Chunking
def chunk_text(text, size=300, overlap=50):
    """
    Chunk by word count.
    size=300  → slightly larger chunks give Qwen more context per retrieval.
    overlap=50 → prevents losing context at chunk boundaries.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        if len(chunk) > 100:
            chunks.append(chunk)
    return chunks

def build_knowledge_base(data_path):
    all_chunks = []
    metadata = []
    supported_ext = {"pdf", "docx", "txt", "odt"}
    files = [f for f in os.listdir(data_path)
             if f.lower().split('.')[-1] in supported_ext]
    if not files:
        print("No supported files found in the folder!")
        return None, [], []
    print(f"Found {len(files)} file(s): {files}\n")
    for file in files:
        file_path = os.path.join(data_path, file)
        try:
            text = preprocess(extract_text(file_path))
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            metadata.extend([file] * len(chunks))
            print(f"{file} → {len(chunks)} chunks")
        except Exception as e:
            print(f"{file} → Error: {e}")
    if not all_chunks:
        print("No chunks generated. Check your documents.")
        return None, [], []
    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Generating embeddings...")
    embeddings = embed_model.encode(
        all_chunks,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # FAISS 
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors (dim={dim})")
    return index, all_chunks, metadata


# RETRIEVAL
def retrieve(question, index, chunks, metadata, k=5):
    """
    Retrieve top-k relevant chunks for the question.
    k=5 gives Qwen 2.5 enough context from research papers.
    """
    q_emb = embed_model.encode([question], convert_to_numpy=True)
    distances, indices = index.search(q_emb, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "chunk": chunks[idx],
            "source": metadata[idx],
            "distance": float(dist)
        })
    return results

# ANSWER WITH QWEN 2.5
def answer_question(question, index, chunks, metadata):
    results = retrieve(question, index, chunks, metadata, k=5)
    # Build context with source labels
    context_parts = []
    for i, r in enumerate(results):
        context_parts.append(f"[Source {i+1}: {r['source']}]\n{r['chunk']}")
    context = "\n\n".join(context_parts)
    sources = list(dict.fromkeys([r['source'] for r in results]))  # preserve order, deduplicate
    prompt = f"""You are a research assistant. Answer the question using ONLY the provided context from research papers.
If the answer is not found in the context, say "I could not find this information in the provided documents."
Do not make up information. Be concise and precise.
Context:
{context}
Question: {question}
Answer:"""
    response = ollama.chat(
        model="qwen2.5:7b",          # ← Qwen 2.5 7B Q4
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.1,      # low temp → factual, less hallucination
            "num_predict": 300,      # enough for detailed research answers
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_gpu": 99            # force ALL layers onto GPU
        }
    )
    return response['message']['content'], sources

# MAIN
if __name__ == "__main__":
    data_path = "documents"          # put research papers here
    print("=" * 50)
    print("  RAG System — Qwen 2.5 7B + FAISS")
    print("=" * 50)
    print(f"\nLoading documents from: '{data_path}/'")
    index, chunks, metadata = build_knowledge_base(data_path)
    if index is None:
        print("Failed to build knowledge base. Exiting.")
        exit(1)
    print("\nRAG Ready! Type your question (or 'exit' to quit)\n")
    print("-" * 50)
    while True:
        q = input("\n❓ Question: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("Bye!")
            break
        print("\nRetrieving relevant chunks...")
        ans, src = answer_question(q, index, chunks, metadata)
        print(f"\nAnswer:\n{ans}")
        print(f"\nSources: {', '.join(src)}")
        print("-" * 50)

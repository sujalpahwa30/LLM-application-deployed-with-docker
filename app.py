"""
app.py — Advanced RAG Document Q&A (with local-model fallback via Ollama)

How it behaves:
1. At startup, checks for local Ollama API availability and whether LOCAL_MODEL is present.
2. rag_chat_fn tries:
   a) Local Ollama (if available)
   b) Primary Groq LLM via llama_index (streaming)
   c) Hugging Face Inference API fallback
   d) OpenRouter fallback
3. If providers are unreachable, they're skipped automatically.
"""

import os
import tempfile
import traceback
import time
from typing import List, Optional
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv

# Llama Index imports (your environment may have slightly different import paths depending on version)
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

# Content extraction
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# Mindmap (graphviz)
import graphviz

# Load environment
load_dotenv()

# ---------------------------
# Config + API keys (env)
# ---------------------------
GROQ_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile")

HF_API_KEY = os.environ.get("HF_API_KEY")
HF_MODEL = os.environ.get("HF_MODEL", "google/flan-t5-large")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-120b")

# Local Ollama settings
LOCAL_URL = os.environ.get("LOCAL_URL", "http://localhost:11434")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "llama2")  # change to the model you pulled locally

EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
INDEX_PATH = os.environ.get("INDEX_PATH", None)

# ---------------------------
# Provider availability flags (updated at startup)
# ---------------------------
LOCAL_AVAILABLE = False
LOCAL_HAS_MODEL = False
HF_AVAILABLE = False
OPENROUTER_AVAILABLE = False
GROQ_AVAILABLE = False

# ---------------------------
# Initialize embedding + LLM (Groq) if available
# ---------------------------
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

llm = None
if GROQ_KEY:
    try:
        llm = Groq(model=GROQ_MODEL, api_key=GROQ_KEY)
        GROQ_AVAILABLE = True
    except Exception as e:
        print("Warning: could not initialize Groq LLM:", e)
        llm = None
        GROQ_AVAILABLE = False

# apply to Settings if available
try:
    Settings.llm = llm
    Settings.embed_model = embed_model
except Exception:
    pass

# Global index & diagnostics
vector_index: Optional[VectorStoreIndex] = None
processed_sources: List[str] = []
PRIMARY_FAILURE_COUNT = 0
FALLBACK_USE_COUNT = 0

# ---------------------------
# Utilities / extraction
# ---------------------------
def extract_text_from_file(file_path: str) -> tuple[str, str]:
    filename = os.path.basename(file_path)
    try:
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()
        text = "\n\n".join([doc.text for doc in documents if getattr(doc, "text", None)])
        return text, filename
    except Exception:
        try:
            with open(file_path, "rb") as f:
                raw = f.read().decode("utf-8", errors="ignore")
                return raw, filename
        except Exception:
            return "", filename

def extract_youtube_id(url: str) -> Optional[str]:
    if "youtube.com" in url:
        if "v=" in url:
            return url.split("v=")[1].split("&")[0]
    elif "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    return None

def extract_text_from_url(url: str) -> tuple[str, str]:
    try:
        video_id = extract_youtube_id(url)
        if video_id:
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                text = "\n".join([item["text"] for item in transcript])
                return text, f"YouTube: {video_id}"
            except (TranscriptsDisabled, NoTranscriptFound):
                pass
        article = Article(url)
        article.download()
        article.parse()
        if article.text:
            title = article.title or url
            return article.text, f"Article: {title}"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for s in soup(["script", "style", "aside", "nav", "header", "footer"]):
            s.extract()
        text = "\n".join([p.get_text(strip=True) for p in soup.find_all("p")])
        return text, f"Web: {url[:60]}..."
    except Exception as e:
        print("extract_text_from_url error:", e)
        return "", ""

def create_documents_from_sources(file, pasted_text: str, url: str) -> tuple[List[Document], List[str]]:
    docs: List[Document] = []
    sources: List[str] = []
    if file is not None:
        file_path = file.name if hasattr(file, "name") else str(file)
        txt, name = extract_text_from_file(file_path)
        if txt.strip():
            docs.append(Document(text=txt, metadata={"source": name, "type": "file", "timestamp": datetime.now().isoformat()}))
            sources.append(f"📄 {name}")
    if pasted_text and pasted_text.strip():
        docs.append(Document(text=pasted_text, metadata={"source": "Pasted Text", "type": "text", "timestamp": datetime.now().isoformat()}))
        sources.append("📝 Pasted Text")
    if url and url.strip():
        txt, src_name = extract_text_from_url(url)
        if txt.strip():
            docs.append(Document(text=txt, metadata={"source": src_name, "type": "url", "timestamp": datetime.now().isoformat()}))
            sources.append(f"🔗 {src_name}")
    return docs, sources

# ---------------------------
# Persistence helpers (optional)
# ---------------------------
def save_index_to_disk(index: VectorStoreIndex, path: str):
    try:
        index.save_to_disk(path)
    except Exception as e:
        print("Warning: failed to save index:", e)

def load_index_from_disk(path: str) -> Optional[VectorStoreIndex]:
    try:
        if os.path.exists(path):
            return VectorStoreIndex.load_from_disk(path)
    except Exception as e:
        print("Warning: failed to load index:", e)
    return None

if INDEX_PATH:
    idx = load_index_from_disk(INDEX_PATH)
    if idx:
        vector_index = idx
        print("Loaded vector index from", INDEX_PATH)

# ---------------------------
# Local (Ollama) helper
# ---------------------------
def test_local_ollama(local_url: str, model: str, timeout: float = 5.0) -> tuple[bool, bool]:
    """
    Check if Ollama HTTP API is reachable and whether the local model exists.
    Returns (api_reachable, model_present).
    """
    try:
        # quick DNS/connectivity check
        health = requests.get(local_url + "/api/models", timeout=timeout)
        if health.status_code not in (200, 201):
            # some Ollama versions may not respond 200; still consider reachable
            pass
        j = None
        try:
            j = health.json()
        except Exception:
            j = None
        model_present = False
        if j and isinstance(j, list):
            # Ollama returns a list of models
            model_names = [m.get("name") or m.get("model") or "" for m in j]
            model_present = model in model_names or any(model in s for s in model_names)
        else:
            # fallback: try to create a tiny generation to see if model exists (non-blocking)
            pass
        return True, model_present
    except Exception:
        return False, False

def generate_text_local(prompt: str, max_tokens: int = 512) -> str:
    """
    Call local Ollama API. Returns a textual answer.
    Expected Ollama API: POST {LOCAL_URL}/api/generate with JSON {model, prompt, ...}
    This function uses a robust parsing strategy for common result shapes.
    """
    url = LOCAL_URL.rstrip("/") + "/api/generate"
    payload = {"model": LOCAL_MODEL, "prompt": prompt, "max_tokens": max_tokens}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    # response formats vary; try to parse common shapes
    try:
        j = r.json()
    except Exception:
        return r.text
    # common shapes: {"id":..., "result":{"output":"..."} } or {"output":"..."} or {"text":"..."}
    if isinstance(j, dict):
        if "result" in j and isinstance(j["result"], dict):
            # some versions nest output
            for key in ("output", "text", "content"):
                if key in j["result"]:
                    return j["result"][key]
            # if there is a list of outputs
            if "outputs" in j["result"] and isinstance(j["result"]["outputs"], list):
                # join textual outputs
                outs = []
                for out in j["result"]["outputs"]:
                    if isinstance(out, dict):
                        for k in ("output", "text", "content"):
                            if k in out:
                                outs.append(out[k])
                                break
                    elif isinstance(out, str):
                        outs.append(out)
                return "\n".join(outs)
        # top-level
        for key in ("output", "text", "content"):
            if key in j:
                return j[key]
    # fallback
    return str(j)

# ---------------------------
# Remote fallback helpers
# ---------------------------
def generate_text_hf(prompt: str, max_tokens: int = 512) -> str:
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY not configured.")
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    out = r.json()
    if isinstance(out, list) and len(out) and isinstance(out[0], dict) and "generated_text" in out[0]:
        return out[0]["generated_text"]
    # try other shapes
    if isinstance(out, dict):
        for k in ("generated_text", "text", "output"):
            if k in out:
                return out[k]
    return str(out)

def generate_text_openrouter(prompt: str, max_tokens: int = 512) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not configured.")
    url = "https://api.openrouter.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    payload = {"model": OPENROUTER_MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    j = r.json()
    try:
        return j["choices"][0]["message"]["content"]
    except Exception:
        return str(j)

# ---------------------------
# Startup health checks
# ---------------------------
def run_startup_checks():
    global LOCAL_AVAILABLE, LOCAL_HAS_MODEL, HF_AVAILABLE, OPENROUTER_AVAILABLE, GROQ_AVAILABLE
    # Local Ollama
    try:
        reachable, has_model = test_local_ollama(LOCAL_URL, LOCAL_MODEL)
        LOCAL_AVAILABLE = reachable
        LOCAL_HAS_MODEL = has_model
    except Exception:
        LOCAL_AVAILABLE = False
        LOCAL_HAS_MODEL = False

    # HF quick check (model reachable)
    try:
        if HF_API_KEY:
            resp = requests.post(f"https://api-inference.huggingface.co/models/{HF_MODEL}", headers={"Authorization": f"Bearer {HF_API_KEY}"}, json={"inputs":"health check"}, timeout=8)
            HF_AVAILABLE = resp.status_code < 400
        else:
            HF_AVAILABLE = False
    except Exception:
        HF_AVAILABLE = False

    # OpenRouter DNS/health
    try:
        if OPENROUTER_API_KEY:
            # check DNS first
            try:
                requests.head("https://api.openrouter.ai/v1/chat/completions", timeout=6)
                OPENROUTER_AVAILABLE = True
            except Exception:
                OPENROUTER_AVAILABLE = False
        else:
            OPENROUTER_AVAILABLE = False
    except Exception:
        OPENROUTER_AVAILABLE = False

    # Groq already attempted init earlier; keep GROQ_AVAILABLE as-is

run_startup_checks()

print("=== Startup provider availability ===")
print("LOCAL_AVAILABLE:", LOCAL_AVAILABLE, "LOCAL_HAS_MODEL:", LOCAL_HAS_MODEL, "LOCAL_MODEL:", LOCAL_MODEL)
print("GROQ_AVAILABLE:", GROQ_AVAILABLE, "GROQ_MODEL:", GROQ_MODEL)
print("HF_AVAILABLE:", HF_AVAILABLE, "HF_MODEL:", HF_MODEL)
print("OPENROUTER_AVAILABLE:", OPENROUTER_AVAILABLE, "OPENROUTER_MODEL:", OPENROUTER_MODEL)
print("====================================")

# ---------------------------
# Core processing (same as prior)
# ---------------------------
def process_sources(file, pasted_text: str, url: str) -> str:
    global vector_index, processed_sources
    try:
        if not any([file, (pasted_text and pasted_text.strip()), (url and url.strip())]):
            return "⚠️ Please provide at least one input source."
        docs, sources = create_documents_from_sources(file, pasted_text, url)
        if not docs:
            return "❌ No text extracted. Check inputs."
        vector_index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
        processed_sources = sources
        if INDEX_PATH:
            try:
                save_index_to_disk(vector_index, INDEX_PATH)
            except Exception:
                pass
        return f"✅ Processed {len(sources)} sources. Ready to ask questions."
    except Exception as e:
        traceback.print_exc()
        return f"❌ Error processing sources: {e}"

# ---------------------------
# RAG chat function (tries local first)
# ---------------------------
def rag_chat_fn(message: str, history):
    global vector_index, llm, PRIMARY_FAILURE_COUNT, FALLBACK_USE_COUNT

    if vector_index is None:
        yield "⚠️ Please process at least one source first."
        return
    if not message.strip():
        yield "⚠️ Please enter a question."
        return

    def compose_retrieval_prompt(question: str, top_k: int = 3) -> str:
        try:
            retriever = vector_index.as_retriever(similarity_top_k=top_k)
            retrieved = []
            if hasattr(retriever, "retrieve"):
                retrieved = retriever.retrieve(question)
            else:
                q_engine = vector_index.as_query_engine(streaming=False, llm=None, similarity_top_k=top_k)
                res = q_engine.query(question)
                retrieved = [res] if res else []
            snippets = []
            for r in retrieved[:top_k]:
                try:
                    snippet = getattr(r, "text", None) or str(r)
                except Exception:
                    snippet = str(r)
                snippets.append(snippet[:1500])
            context = "\n\n---\n\n".join(snippets).strip()
            prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely and cite relevant parts from the context."
            return prompt
        except Exception:
            return f"Question: {question}\n\nAnswer concisely."

    # 1) Try local Ollama if available and model present
    if LOCAL_AVAILABLE and LOCAL_HAS_MODEL:
        try:
            prompt = compose_retrieval_prompt(message, top_k=3)
            FALLBACK_USE_COUNT += 1
            ans = generate_text_local(prompt, max_tokens=512)
            yield ans
            return
        except Exception as e:
            print("Local (Ollama) fallback failed:", traceback.format_exc())

    # 2) Try primary Groq with streaming
    primary_exception = None
    if llm is not None:
        for attempt in range(2):
            try:
                query_engine = vector_index.as_query_engine(streaming=True, similarity_top_k=3, response_mode="compact", llm=llm)
                streaming_response = query_engine.query(message)
                partial = ""
                if hasattr(streaming_response, "response_gen") and streaming_response.response_gen:
                    for token in streaming_response.response_gen:
                        partial += str(token)
                        yield partial
                    return
                else:
                    final_text = getattr(streaming_response, "response", None) or str(streaming_response)
                    yield final_text
                    return
            except Exception as e:
                primary_exception = e
                time.sleep(0.8 * (attempt + 1))
                continue

    PRIMARY_FAILURE_COUNT += 1
    yield "⚠️ Primary LLM unavailable or returned an error. Attempting remote fallback provider..."

    # 3) Remote fallbacks: HF then OpenRouter (if reachable)
    prompt = compose_retrieval_prompt(message, top_k=3)
    if HF_AVAILABLE:
        try:
            FALLBACK_USE_COUNT += 1
            ans = generate_text_hf(prompt, max_tokens=512)
            yield ans
            return
        except Exception as e:
            print("HF fallback failed at runtime:", traceback.format_exc())
    if OPENROUTER_AVAILABLE:
        try:
            FALLBACK_USE_COUNT += 1
            ans = generate_text_openrouter(prompt, max_tokens=512)
            yield ans
            return
        except Exception as e:
            print("OpenRouter fallback failed at runtime:", traceback.format_exc())

    # All failed
    pmsg = str(primary_exception)[:400] if primary_exception else "No primary error recorded."
    yield f"❌ All LLM attempts failed. Primary error (truncated): {pmsg}\nCheck local Ollama/Groq/HF/OpenRouter availability."

# ---------------------------
# Mindmap / Quiz / Summary (unchanged)
# ---------------------------
def generate_mindmap(focus_hint: str = "") -> tuple[Optional[str], str]:
    global vector_index
    if vector_index is None:
        return None, "⚠️ Please process sources first."
    try:
        query = ("Create a hierarchical outline (3 levels max) of main topics and subtopics. "
                 f"Focus on: {focus_hint if focus_hint else 'all content'}")
        query_engine = vector_index.as_query_engine(similarity_top_k=5)
        response = query_engine.query(query)
        outline_text = str(response)
        dot = graphviz.Digraph(format="png")
        dot.attr(rankdir="LR", bgcolor="white")
        dot.attr("node", shape="box", style="rounded,filled", fillcolor="lightblue", fontname="Arial")
        dot.attr("edge", color="gray")
        root = "Document Overview"
        dot.node("root", root, fillcolor="lightcoral", fontsize="14")
        lines = [l.strip() for l in outline_text.split("\n") if l.strip()]
        node_count = 0
        for line in lines[:30]:
            if line and len(line) < 200:
                node_id = f"n{node_count}"
                label = line[:90] + "..." if len(line) > 90 else line
                dot.node(node_id, label, fontsize="10")
                dot.edge("root", node_id)
                node_count += 1
        output_path = tempfile.mktemp(suffix=".png")
        dot.render(filename=output_path, cleanup=True)
        return output_path + ".png", f"✅ Mindmap generated with {node_count} nodes."
    except Exception as e:
        traceback.print_exc()
        if "failed to execute PosixPath('dot')" in str(e) or "executable file not found" in str(e):
            return None, "❌ Graphviz 'dot' not found on PATH. Install Graphviz or use networkx fallback."
        return None, f"❌ Error generating mindmap: {e}"

def generate_quiz(num_questions: int = 5) -> str:
    global vector_index
    if vector_index is None:
        return "⚠️ Please process sources first."
    try:
        prompt = (f"Generate {num_questions} multiple-choice questions from the content. "
                  "Format: Question / A) / B) / C) / D) / Answer / Explanation.")
        q_engine = vector_index.as_query_engine(similarity_top_k=5)
        resp = q_engine.query(prompt)
        return str(resp)
    except Exception as e:
        traceback.print_exc()
        return f"❌ Error generating quiz: {e}"

def summarize_content() -> str:
    global vector_index
    if vector_index is None:
        return "⚠️ Please process sources first."
    try:
        prompt = "Provide a concise summary (4-7 bullets) of the indexed documents."
        q_engine = vector_index.as_query_engine(similarity_top_k=5)
        resp = q_engine.query(prompt)
        return str(resp)
    except Exception as e:
        traceback.print_exc()
        return f"❌ Error summarizing: {e}"

# ---------------------------
# Utility & UI functions
# ---------------------------
def clear_all():
    global vector_index, processed_sources
    vector_index = None
    processed_sources = []
    return None, "", "", "🔄 All data cleared. Ready for new sources."

def show_sources() -> str:
    global processed_sources
    if not processed_sources:
        return "No sources processed yet."
    return "**Processed Sources:**\n\n" + "\n".join(processed_sources)

def show_diagnostics() -> str:
    return (f"Diagnostics:\n"
            f"  • LOCAL_AVAILABLE: {LOCAL_AVAILABLE}, LOCAL_HAS_MODEL: {LOCAL_HAS_MODEL}, LOCAL_MODEL: {LOCAL_MODEL}\n"
            f"  • GROQ_AVAILABLE: {GROQ_AVAILABLE}, GROQ_MODEL: {GROQ_MODEL}\n"
            f"  • HF_AVAILABLE: {HF_AVAILABLE}, HF_MODEL: {HF_MODEL}\n"
            f"  • OPENROUTER_AVAILABLE: {OPENROUTER_AVAILABLE}, OPENROUTER_MODEL: {OPENROUTER_MODEL}\n"
            f"  • PRIMARY_FAILURE_COUNT: {PRIMARY_FAILURE_COUNT}\n"
            f"  • FALLBACK_USE_COUNT: {FALLBACK_USE_COUNT}\n")

# ---------------------------
# Gradio UI (type='messages' fix)
# ---------------------------
CUSTOM_CSS = """
.gradio-container { max-width: 1300px !important; margin: auto; font-family: 'Inter', sans-serif; }
.main-header { text-align: center; padding: 18px; background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); color: white; border-radius: 10px; margin-bottom: 18px; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), css=CUSTOM_CSS, title="RAG Document Q&A (Local-first)") as demo:
    gr.Markdown("""<div class="main-header"><h2>Advanced Document Q&A — Local-first (Ollama)</h2></div>""")

    with gr.Row():
        with gr.Column(scale=1, min_width=340):
            gr.Markdown("### Inputs")
            file_input = gr.File(label="Upload Document", file_types=[".txt", ".pdf", ".docx", ".md", ".csv"], file_count="single")
            paste_text = gr.Textbox(label="Paste Text", placeholder="Paste content here...", lines=5)
            url_text = gr.Textbox(label="Article / YouTube URL", placeholder="https://...")
            process_btn = gr.Button("Process Sources", variant="primary")
            status_box = gr.Textbox(label="Status", interactive=False, lines=4)
            gr.Markdown("---")
            with gr.Row():
                clear_btn = gr.Button("Clear All", variant="secondary")
                sources_btn = gr.Button("View Sources", variant="secondary")
                diag_btn = gr.Button("Diagnostics", variant="secondary")
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("Chat"):
                    gr.Markdown("Ask questions about the processed documents")
                    chatbot = gr.ChatInterface(
                        fn=rag_chat_fn,
                        chatbot=gr.Chatbot(height=520, show_label=False, type="messages"),
                        textbox=gr.Textbox(placeholder="Ask anything...", container=False),
                        submit_btn="Send",
                        type="messages"
                    )
                with gr.TabItem("Mindmap"):
                    mindmap_btn = gr.Button("Generate Mindmap", variant="primary")
                    mindmap_status = gr.Textbox(interactive=False)
                    mindmap_img = gr.Image(type="filepath")
                with gr.TabItem("Quiz"):
                    quiz_btn = gr.Button("Generate Quiz", variant="primary")
                    quiz_output = gr.Textbox(lines=18, show_copy_button=True)
                with gr.TabItem("Summary"):
                    summary_btn = gr.Button("Generate Summary", variant="primary")
                    summary_output = gr.Textbox(lines=12, show_copy_button=True)

    gr.Markdown("<div style='text-align:center; color:#666; margin-top:12px;'>Local-first: Ollama -> Groq -> HF -> OpenRouter</div>")

    # events
    process_btn.click(fn=process_sources, inputs=[file_input, paste_text, url_text], outputs=status_box)
    clear_btn.click(fn=clear_all, outputs=[file_input, paste_text, url_text, status_box])
    sources_btn.click(fn=show_sources, outputs=status_box)
    diag_btn.click(fn=show_diagnostics, outputs=status_box)
    mindmap_btn.click(fn=generate_mindmap, inputs=[], outputs=[mindmap_img, mindmap_status])
    quiz_btn.click(fn=lambda n: generate_quiz(int(n)) if isinstance(n, (int, float)) else generate_quiz(5), inputs=None, outputs=quiz_output)
    summary_btn.click(fn=summarize_content, outputs=summary_output)

# ---------------------------
# Launch
# ---------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Starting RAG Document Q&A — Local-first fallback")
    print("LOCAL_AVAILABLE:", LOCAL_AVAILABLE, "LOCAL_HAS_MODEL:", LOCAL_HAS_MODEL, "LOCAL_MODEL:", LOCAL_MODEL)
    print("GROQ_AVAILABLE:", GROQ_AVAILABLE, "GROQ_MODEL:", GROQ_MODEL)
    print("HF_AVAILABLE:", HF_AVAILABLE, "HF_MODEL:", HF_MODEL)
    print("OPENROUTER_AVAILABLE:", OPENROUTER_AVAILABLE, "OPENROUTER_MODEL:", OPENROUTER_MODEL)
    print("=" * 60)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)

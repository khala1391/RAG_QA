"""NotebookLM-style Gradio web app.

Run:
    python src/web_app.py

Opens at http://localhost:7860 — each browser session keeps its own
in-memory FAISS index built from the user's uploaded files.
"""
import base64
import shutil
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.embeddings import get_embeddings
from src.ingestion import doc_summary, format_doc_inventory, ingest, ocr_stats
from src.qa_chain import ask, build_chain, build_llm
from src.retriever import as_retriever, build_index

CFG = load_config()
AVAILABLE_MODELS = CFG["llm"]["available_models"]
DEFAULT_MODEL = CFG["llm"]["default_model"]

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

# ── Bot avatar — falls back to None if file missing ─────────────
_bot_avatar_path = ASSETS / "bot.svg"
BOT_AVATAR = str(_bot_avatar_path) if _bot_avatar_path.exists() else None

# ── Profile assets ──────────────────────────────────────────────
_profile_b64 = ""
_profile_candidates = [
    ASSETS / "profile.jpg",
    ASSETS / "profile.png",
    ASSETS / "Yuttapong M.jpg",
]
for _p in _profile_candidates:
    if _p.exists():
        ext = "png" if _p.suffix.lower() == ".png" else "jpeg"
        _profile_b64 = (
            f"data:image/{ext};base64,"
            + base64.b64encode(_p.read_bytes()).decode()
        )
        break

LINKEDIN_URL = "https://www.linkedin.com/in/yuttapong-m/"
LINKEDIN_HANDLE = "yuttapong-m"

_LI_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" '
    'fill="white" viewBox="0 0 24 24" style="flex-shrink:0">'
    '<path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14'
    "c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11z"
    "m-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764"
    " 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604"
    "c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777"
    ' 7 2.476v6.759z"/></svg>'
)

_img_html = (
    f'<img src="{_profile_b64}" alt="profile" />' if _profile_b64
    else '<div class="profile-placeholder">YM</div>'
)

PROFILE_BADGE_HTML = f"""
<div id="profile-badge">
  {_img_html}
  <a class="li-btn" href="{LINKEDIN_URL}" target="_blank" rel="noopener">
    {_LI_SVG} {LINKEDIN_HANDLE}
  </a>
</div>
"""

# ── Custom CSS — light theme + profile badge ────────────────────
CUSTOM_CSS = """
/* Force light theme — outer page in soft slate */
:root, body, .gradio-container {
    color-scheme: light !important;
    background: #e6ecf3 !important;
}
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
.dark { color-scheme: light !important; }

/* Chat box — pure white, lifted with shadow for contrast */
#main-chatbot, #main-chatbot > div {
    background: #ffffff !important;
    border: 1px solid #c7d2e0 !important;
    border-radius: 14px !important;
    box-shadow: 0 6px 20px rgba(45, 87, 168, 0.10) !important;
}
#main-chatbot .message-wrap { background: transparent !important; }
#main-chatbot .message.user {
    background: #e8f0fb !important;
    border-radius: 12px !important;
}
#main-chatbot .message.bot {
    background: #f7f9fc !important;
    border-radius: 12px !important;
}

/* Sidebar panels — also lifted from page background */
.gradio-container .form, .gradio-container .block {
    background: #ffffff !important;
    border-radius: 10px !important;
}

/* Profile badge — fixed top right */
#profile-badge {
    position: fixed; top: 1rem; right: 1.5rem;
    z-index: 2147483647;
    display: flex; align-items: center; gap: 0.5rem;
    pointer-events: auto;
}
#profile-badge img {
    width: 40px; height: 40px; border-radius: 50%;
    object-fit: cover; border: 2px solid #4a7fc1; flex-shrink: 0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}
#profile-badge .profile-placeholder {
    width: 40px; height: 40px; border-radius: 50%;
    background: #4a7fc1; color: white; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem; border: 2px solid #4a7fc1;
}
#profile-badge a.li-btn {
    display: inline-flex; align-items: center; gap: 0.35rem;
    background: #3b6fc4; color: white !important; text-decoration: none;
    padding: 0.32rem 0.8rem; border-radius: 8px; font-size: 0.82rem;
    font-weight: 600; line-height: 1; font-family: sans-serif;
    box-shadow: 0 2px 6px rgba(59,111,196,0.25);
    transition: background 0.15s ease;
}
#profile-badge a.li-btn:hover { background: #2d57a8; }

/* Section headers */
.section-title {
    font-size: 1.05rem; font-weight: 600; color: #2c3e50;
    margin: 0.4rem 0 0.6rem 0;
}

/* Status pill */
#status-md { font-size: 0.9rem; padding: 0.5rem 0.75rem;
    background: #f0f4fa; border-radius: 8px; border-left: 3px solid #4a7fc1; }

/* Header row — title left, How-it-works trigger right */
#header-row { align-items: center; }
#header-info-col {
    padding-top: 1rem;
    padding-right: 5rem;     /* leave room for the fixed profile badge */
    display: flex; justify-content: flex-end; align-items: flex-start;
    overflow: visible !important;
}
/* Kill any inner scroll wrapper Gradio adds to the column */
#header-info-col > .form,
#header-info-col > div,
#header-info-col .block {
    overflow: visible !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Trigger chip-button */
#how-it-works-trigger {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.5rem 0.9rem;
    background: #ffffff;
    color: #2c3e50;
    border: 1px solid #c7d2e0;
    border-radius: 999px;
    cursor: pointer;
    font-size: 0.85rem;
    font-weight: 600;
    font-family: inherit;
    box-shadow: 0 2px 6px rgba(45,87,168,0.08);
    transition: background 0.15s ease, transform 0.15s ease;
    white-space: nowrap;
}
#how-it-works-trigger:hover {
    background: #f0f4fa;
    transform: translateY(-1px);
}

/* Balloon popover */
#how-it-works-balloon {
    position: fixed;
    top: 5.2rem;
    right: 5.5rem;
    width: 460px;
    max-width: calc(100vw - 4rem);
    max-height: 70vh;
    background: #ffffff;
    border: 1px solid #c7d2e0;
    border-radius: 14px;
    box-shadow: 0 14px 44px rgba(45,87,168,0.22);
    padding: 1.2rem 1.4rem 1.1rem 1.4rem;
    z-index: 9998;
    overflow-y: auto;
    color: #2c3e50;
    font-size: 0.88rem;
    opacity: 0;
    transform: translateY(-8px) scale(0.97);
    pointer-events: none;
    transition: opacity 0.18s ease, transform 0.18s ease;
}
#how-it-works-balloon.open {
    opacity: 1;
    transform: translateY(0) scale(1);
    pointer-events: auto;
}
/* Arrow on top pointing up to the trigger */
#how-it-works-balloon::before {
    content: '';
    position: absolute;
    top: -8px;
    right: 38px;
    width: 14px; height: 14px;
    background: #ffffff;
    border-left: 1px solid #c7d2e0;
    border-top: 1px solid #c7d2e0;
    transform: rotate(45deg);
}

/* Close × button */
#balloon-close {
    position: absolute;
    top: 0.55rem;
    right: 0.75rem;
    width: 28px; height: 28px;
    border-radius: 50%;
    border: none;
    background: #f0f4fa;
    color: #2c3e50;
    cursor: pointer;
    font-size: 1.2rem;
    line-height: 1;
    display: flex; align-items: center; justify-content: center;
    transition: background 0.15s ease;
}
#balloon-close:hover { background: #e3e9f3; }

/* Scroll-to-latest floating arrow */
#scroll-to-latest {
    position: fixed;
    bottom: 6.5rem;
    right: 2rem;
    z-index: 9999;
    width: 46px; height: 46px; border-radius: 50%;
    background: #3b6fc4; color: white !important;
    border: none; cursor: pointer;
    font-size: 1.4rem; line-height: 1;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 14px rgba(45,87,168,0.35);
    transition: transform 0.15s ease, background 0.15s ease, opacity 0.2s ease;
    opacity: 0; pointer-events: none;
}
#scroll-to-latest.visible { opacity: 1; pointer-events: auto; }
#scroll-to-latest:hover { background: #2d57a8; transform: translateY(-2px); }
#scroll-to-latest::after { content: "↓"; font-weight: 700; }
"""

# ── Force light mode regardless of system preference ────────────
FORCE_LIGHT_JS = """
() => {
    const url = new URL(window.location.href);
    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.replace(url.toString());
    }
}
"""

# ── How-it-works trigger + balloon popover ──────────────────────
HOW_IT_WORKS_TRIGGER_HTML = """
<button id="how-it-works-trigger" aria-label="Show how it works"
        onclick="event.stopPropagation();
                 document.getElementById('how-it-works-balloon')
                         .classList.toggle('open');">
  <span style="font-size:1rem">ℹ️</span>
  <span>How it works &amp; limitations</span>
</button>
"""

HOW_IT_WORKS_BALLOON_HTML = """
<div id="how-it-works-balloon" role="dialog" aria-label="How it works">
  <button id="balloon-close" aria-label="Close"
          onclick="document.getElementById('how-it-works-balloon')
                           .classList.remove('open');">×</button>
  <h3 style="margin:0 0 0.6rem 0; color:#2c3e50; font-size:1.05rem;">
    ℹ️ How it works &amp; limitations
  </h3>

  <p style="margin:0.4rem 0 0.3rem 0;"><strong>ขั้นตอน:</strong></p>
  <ol style="margin:0; padding-left:1.2rem; line-height:1.55;">
    <li>Upload เอกสาร (PDF / DOCX / TXT / JPG / PNG ...)</li>
    <li>กด <strong>Build Index</strong> — ระบบสร้าง FAISS index แบบ in-memory (per-session)</li>
    <li>ถามคำถามได้ — bot ตอบจาก context ในเอกสารเท่านั้น</li>
    <li>ถามต่อได้ — chat เก็บ history เพื่อให้สนทนาต่อเนื่อง</li>
    <li>Refresh page = เริ่ม session ใหม่</li>
  </ol>

  <p style="margin:0.9rem 0 0.3rem 0;">
    <strong>⚠️ ข้อจำกัด เทียบกับ LLM ทั่วไป (ChatGPT / Claude):</strong>
  </p>
  <ul style="margin:0; padding-left:1.2rem; line-height:1.55;">
    <li>🔒 <strong>ตอบเฉพาะจากเอกสารที่อัพโหลด</strong> — ไม่ใช้ความรู้ภายนอก</li>
    <li>🚫 <strong>ไม่แต่งคำตอบเกินกว่าเอกสาร</strong> — ถ้าไม่พบจะตอบว่า
        <em>"ฉันไม่พบข้อมูลนี้ในเอกสารที่ให้มา"</em> แทนการเดา</li>
    <li>📚 <strong>คำถามทั่วไปอาจตอบไม่ได้</strong> — เช่น "1+1 = ?" หรือเรื่องนอกขอบเขต</li>
    <li>🎯 <strong>คุณภาพคำตอบขึ้นกับเอกสาร</strong> — คลุมเครือ → ตอบคลุมเครือ</li>
    <li>🌐 <strong>ภาษา</strong> — ตอบภาษาเดียวกับที่ถาม (ไทย/อังกฤษ/อื่นๆ)</li>
  </ul>

  <p style="margin:0.9rem 0 0.3rem 0;"><strong>ข้อดี:</strong></p>
  <ul style="margin:0 0 0.2rem 0; padding-left:1.2rem; line-height:1.55;">
    <li>✅ ตอบจากข้อมูลล่าสุด/ภายในองค์กร โดยไม่ต้อง fine-tune</li>
    <li>✅ ตรวจสอบที่มาได้จาก <em>Sources</em> ใต้แต่ละคำตอบ</li>
    <li>✅ ลด hallucination เพราะตอบจาก context จริง</li>
  </ul>
</div>
"""

BALLOON_HELPER_JS = """
() => {
    const balloon = document.getElementById('how-it-works-balloon');
    const trigger = document.getElementById('how-it-works-trigger');
    if (!balloon || !trigger || balloon.__ragWired) return;

    // Close on click outside
    document.addEventListener('click', (e) => {
        if (!balloon.classList.contains('open')) return;
        if (balloon.contains(e.target) || trigger.contains(e.target)) return;
        balloon.classList.remove('open');
    });
    // Close on Escape
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') balloon.classList.remove('open');
    });
    balloon.__ragWired = true;
}
"""

# ── Scroll-to-latest button + auto show/hide on scroll ──────────
SCROLL_BUTTON_HTML = """
<button id="scroll-to-latest" aria-label="Scroll to latest message"
        onclick="window.__rag_scrollChat && window.__rag_scrollChat()"></button>
"""

SCROLL_HELPER_JS = """
() => {
    const findScrollWrap = () => {
        const chat = document.querySelector('#main-chatbot');
        if (!chat) return null;
        const cands = chat.querySelectorAll('div');
        for (const el of cands) {
            const s = getComputedStyle(el);
            if ((s.overflowY === 'auto' || s.overflowY === 'scroll')
                && el.scrollHeight > el.clientHeight) return el;
        }
        return chat;
    };

    window.__rag_scrollChat = () => {
        const wrap = findScrollWrap();
        if (wrap) wrap.scrollTo({ top: wrap.scrollHeight, behavior: 'smooth' });
    };

    const btn = document.getElementById('scroll-to-latest');
    if (!btn) return;

    const update = () => {
        const wrap = findScrollWrap();
        if (!wrap) return;
        const nearBottom = wrap.scrollHeight - wrap.scrollTop - wrap.clientHeight < 80;
        btn.classList.toggle('visible', !nearBottom && wrap.scrollHeight > wrap.clientHeight + 40);
    };

    // Watch scroll
    const attachScroll = () => {
        const wrap = findScrollWrap();
        if (wrap && !wrap.__ragHooked) {
            wrap.addEventListener('scroll', update, { passive: true });
            wrap.__ragHooked = true;
        }
    };

    // Watch DOM for new messages (auto-show button when answer arrives if user scrolled up)
    const observer = new MutationObserver(() => { attachScroll(); update(); });
    const chat = document.querySelector('#main-chatbot');
    if (chat) observer.observe(chat, { childList: true, subtree: true });

    attachScroll();
    update();
}
"""


def _initial_state() -> dict:
    return {"store": None, "chain": None, "model": None, "api_key": None, "files": []}


WELCOME_TEMPLATE = (
    "✅ **พร้อมแล้ว!** ผมได้อ่านเอกสาร **{n_files} ไฟล์** ({n_chunks} chunks{ocr_note}) ของคุณเรียบร้อย\n\n"
    "ลองถามอะไรก็ได้เกี่ยวกับเอกสารเหล่านี้ — ตัวอย่างเช่น:\n"
    "- _สรุปเนื้อหาหลักให้หน่อย_\n"
    "- _เอกสารพูดถึงเรื่อง X ไว้ว่าอย่างไร?_\n"
    "- _มีตัวเลข/สถิติอะไรที่น่าสนใจบ้าง?_\n\n"
    "🤖 Model: `{model}`"
)


def build_session_index(api_key: str, model: str, files: list, state: dict):
    """Generator — yields (state, status, chatbot) so the UI shows progress."""
    state = state or _initial_state()
    empty_chat: list = []

    if not files:
        yield state, "⚠️ Please upload at least one document first.", empty_chat
        return
    if not api_key:
        yield state, "⚠️ Please enter your Groq API key.", empty_chat
        return

    yield state, "🔄 Reading documents...", empty_chat
    time.sleep(0.05)

    tmp_dir = Path(tempfile.mkdtemp(prefix="rag_qa_"))
    file_paths: list[Path] = []
    for f in files:
        src_path = Path(f.name if hasattr(f, "name") else f)
        dst = tmp_dir / src_path.name
        shutil.copy(src_path, dst)
        file_paths.append(dst)

    try:
        yield (
            state,
            f"🔄 Extracting text from {len(file_paths)} file(s) — OCR will run on scanned pages...",
            empty_chat,
        )
        docs = ingest(
            paths=file_paths,
            chunk_size=CFG["ingestion"]["chunk_size"],
            chunk_overlap=CFG["ingestion"]["chunk_overlap"],
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not docs:
        yield (
            state,
            "❌ ไม่สามารถดึงข้อความจากไฟล์ได้ — ถ้าเป็น scanned PDF "
            "ลองติดตั้ง Tesseract OCR (ดู README)",
            empty_chat,
        )
        return

    n_pages, n_ocr = ocr_stats(docs)
    ocr_note = f", {n_ocr}/{n_pages} pages via OCR" if n_ocr else ""

    yield (
        state,
        f"🔄 Generating embeddings for {len(docs)} chunks{ocr_note}...",
        empty_chat,
    )
    embeddings = get_embeddings(
        model_name=CFG["embedding"]["model_name"],
        device=CFG["embedding"]["device"],
    )

    yield state, "🔄 Building FAISS index...", empty_chat
    store = build_index(docs, embeddings, persist=False)
    r_cfg = CFG["retriever"]
    retriever = as_retriever(
        store,
        top_k=r_cfg["top_k"],
        search_type=r_cfg.get("search_type", "mmr"),
        fetch_k=r_cfg.get("fetch_k", 24),
        lambda_mult=r_cfg.get("lambda_mult", 0.5),
    )

    yield state, f"🔄 Initializing Groq model `{model}`...", empty_chat
    try:
        llm = build_llm(
            api_key=api_key,
            model=model,
            temperature=CFG["llm"]["temperature"],
            max_tokens=CFG["llm"]["max_tokens"],
        )
    except Exception as e:
        yield state, f"❌ Failed to initialize Groq model: {e}", empty_chat
        return

    inventory = format_doc_inventory(doc_summary(docs))
    chain = build_chain(llm, retriever, doc_inventory=inventory)
    state = {
        "store": store,
        "chain": chain,
        "model": model,
        "api_key": api_key,
        "files": [p.name for p in file_paths],
    }

    welcome_msg = WELCOME_TEMPLATE.format(
        n_files=len(file_paths),
        n_chunks=len(docs),
        ocr_note=ocr_note,
        model=model,
    )
    final_status = (
        f"✅ Index ready — **{len(docs)} chunks** from **{len(file_paths)} file(s)**"
        f"{ocr_note}. Model: `{model}`"
    )
    yield state, final_status, [(None, welcome_msg)]


def chat_fn(message: str, history: list, state: dict):
    state = state or _initial_state()

    if state.get("chain") is None:
        return "⚠️ Please upload documents and click **Build Index** first."
    if not message or not message.strip():
        return ""

    chat_history: list[tuple[str, str]] = []
    for turn in history or []:
        if isinstance(turn, dict):
            continue
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            user_msg, bot_msg = turn
            # Skip system messages (e.g. the welcome bubble where user side is None)
            if user_msg is None or bot_msg is None:
                continue
            chat_history.append((user_msg, bot_msg))

    try:
        result = ask(state["chain"], message, chat_history=chat_history)
    except Exception as e:
        return f"❌ Error from Groq: {e}"

    answer = result["answer"].strip()
    sources = result["sources"]
    if sources:
        seen: list[str] = []
        bullets: list[str] = []
        for d in sources:
            src = d.metadata.get("source", "unknown")
            snippet = d.page_content.strip().replace("\n", " ")
            if len(snippet) > 220:
                snippet = snippet[:220] + "..."
            key = f"{src}::{snippet[:80]}"
            if key in seen:
                continue
            seen.append(key)
            bullets.append(f"- **{src}** — {snippet}")
        sources_block = "\n".join(bullets)
        answer += (
            f"\n\n<details><summary>📚 Sources ({len(bullets)})</summary>\n\n"
            f"{sources_block}\n\n</details>"
        )

    return answer


def clear_session():
    return (
        _initial_state(),
        "_Session cleared. Upload new documents to start again._",
        [],
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="RAG QA — NotebookLM-style",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CUSTOM_CSS,
    ) as demo:
        # Profile badge (fixed, top-right)
        gr.HTML(PROFILE_BADGE_HTML)
        # Floating scroll-to-latest arrow
        gr.HTML(SCROLL_BUTTON_HTML)

        # ── Header row: title left, How-it-works trigger right ──
        with gr.Row(elem_id="header-row"):
            with gr.Column(scale=5):
                gr.Markdown(
                    "# 📓 RAG QA — Chat with Your Documents\n"
                    "Upload files, build a temporary index, and ask questions powered by **Groq**.",
                    elem_id="header-title",
                )
            with gr.Column(scale=2, min_width=300, elem_id="header-info-col"):
                gr.HTML(HOW_IT_WORKS_TRIGGER_HTML)

        # Balloon popover (fixed position, hidden by default)
        gr.HTML(HOW_IT_WORKS_BALLOON_HTML)

        state = gr.State(_initial_state())

        with gr.Row():
            # ─── Sidebar ───
            with gr.Column(scale=1, min_width=340):
                gr.Markdown('<div class="section-title">⚙️ Settings</div>')

                api_key_in = gr.Textbox(
                    label="Groq API Key",
                    placeholder="gsk_...",
                    type="password",
                    info="ใส่ API key จาก https://console.groq.com/keys (เก็บเฉพาะใน session นี้ ไม่ถูกบันทึก)",
                )
                model_in = gr.Dropdown(
                    label="Model",
                    choices=AVAILABLE_MODELS,
                    value=DEFAULT_MODEL,
                    info="เลือก LLM model: llama-3.3-70b คุณภาพดีสุด, 8b-instant เร็ว, mixtral context 32k",
                )
                files_in = gr.File(
                    label="Documents",
                    file_count="multiple",
                    file_types=[
                        ".pdf", ".txt", ".md", ".docx",
                        ".jpg", ".jpeg", ".png", ".bmp",
                        ".tiff", ".tif", ".webp",
                    ],
                )
                gr.Markdown(
                    "Supported: PDF, TXT, MD, DOCX, **JPG/PNG/TIFF** "
                    " ",
                    elem_classes=["file-hint"],
                )

                build_btn = gr.Button(
                    "🔨 Build Index",
                    variant="primary",
                    size="lg",
                )
                clear_btn = gr.Button("🧹 Clear Session", size="sm")

                gr.Markdown('<div class="section-title">📊 Status</div>')
                status = gr.Markdown(
                    "_No index yet — upload files and click **Build Index**._",
                    elem_id="status-md",
                )

            # ─── Main chat ───
            with gr.Column(scale=2):
                gr.Markdown('<div class="section-title">💬 Chat</div>')
                chatbot = gr.Chatbot(
                    elem_id="main-chatbot",
                    height=560,
                    show_copy_button=True,
                    render_markdown=True,
                    avatar_images=(None, BOT_AVATAR),
                    bubble_full_width=False,
                )
                gr.ChatInterface(
                    fn=chat_fn,
                    chatbot=chatbot,
                    additional_inputs=[state],
                    type="tuples",
                    textbox=gr.Textbox(
                        placeholder="ถามอะไรเกี่ยวกับเอกสารของคุณ...",
                        container=False,
                        scale=7,
                    ),
                )

        build_btn.click(
            build_session_index,
            inputs=[api_key_in, model_in, files_in, state],
            outputs=[state, status, chatbot],
        )
        clear_btn.click(
            clear_session,
            inputs=None,
            outputs=[state, status, chatbot],
        )

        # Force light theme on load
        demo.load(None, None, None, js=FORCE_LIGHT_JS)
        # Initialise scroll-to-latest button (hook scroll listeners & DOM observer)
        demo.load(None, None, None, js=SCROLL_HELPER_JS)
        # Wire click-outside / Escape handlers for the How-it-works balloon
        demo.load(None, None, None, js=BALLOON_HELPER_JS)

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)

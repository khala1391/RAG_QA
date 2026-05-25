import os
import shutil
from pathlib import Path
from typing import Iterable

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".docx",
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# OCR settings — fallback when a PDF page has too little text
OCR_TEXT_THRESHOLD = 30        # chars; below this we treat the page as scanned
OCR_LANG = "eng+tha"           # preferred; falls back to "eng" if `tha` is missing
OCR_DPI = 300                  # higher = better OCR but slower

# Common Windows install paths for tesseract — checked if not on PATH
_WINDOWS_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    os.path.expandvars(r"%USERPROFILE%\AppData\Local\Tesseract-OCR\tesseract.exe"),
]


def _configure_tesseract() -> bool:
    """Ensure pytesseract knows where tesseract.exe lives. Return True if found."""
    try:
        import pytesseract
    except ImportError:
        return False
    # Already on PATH?
    if shutil.which("tesseract"):
        return True
    # Windows fallback: check common install paths
    for candidate in _WINDOWS_TESSERACT_PATHS:
        if candidate and Path(candidate).is_file():
            pytesseract.pytesseract.tesseract_cmd = candidate
            return True
    return False


def _ocr_available() -> bool:
    """Check if pytesseract + tesseract binary are usable."""
    try:
        import pytesseract
        from PIL import Image  # noqa: F401
        if not _configure_tesseract():
            return False
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _ocr_languages() -> str:
    """Return the best language string available — prefer `eng+tha`, else `eng`."""
    try:
        import pytesseract
        available = set(pytesseract.get_languages(config=""))
        wanted = [lang for lang in OCR_LANG.split("+") if lang in available]
        if wanted:
            return "+".join(wanted)
        # Last resort — let tesseract default to eng if even eng is missing,
        # this will surface a clearer downstream error.
        return "eng"
    except Exception:
        return "eng"


def _ocr_page(page, lang: str | None = None, dpi: int = OCR_DPI) -> str:
    """Rasterize a PyMuPDF page and run tesseract OCR on it."""
    try:
        import pytesseract
        from PIL import Image

        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return pytesseract.image_to_string(img, lang=lang or _ocr_languages()).strip()
    except Exception:
        return ""


def _load_pdf(path: Path) -> list[Document]:
    """Load a PDF using PyMuPDF. For pages with little/no text layer
    (typical of scanned PDFs), automatically fall back to OCR."""
    import fitz  # PyMuPDF

    try:
        pdf = fitz.open(str(path))
    except Exception as e:
        raise RuntimeError(f"Cannot open PDF '{path.name}': {e}") from e

    docs: list[Document] = []
    ocr_ok = _ocr_available()

    for i, page in enumerate(pdf):
        text = page.get_text("text").strip()
        ocr_used = False

        if len(text) < OCR_TEXT_THRESHOLD and ocr_ok:
            ocr_text = _ocr_page(page)
            if ocr_text:
                text = ocr_text
                ocr_used = True

        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "page": i + 1,
                        "ocr": ocr_used,
                    },
                )
            )

    pdf.close()
    return docs


def _load_image(path: Path) -> list[Document]:
    """OCR an image file (JPG/PNG/...) using tesseract."""
    if not _ocr_available():
        raise RuntimeError(
            f"Cannot read image '{path.name}': Tesseract OCR not found.\n"
            "→ Windows: install from https://github.com/UB-Mannheim/tesseract/wiki\n"
            "  (ทำเครื่องหมายเลือก Thai language pack ตอนติดตั้งด้วย ถ้าจะอ่านภาษาไทย)\n"
            "→ macOS: brew install tesseract tesseract-lang\n"
            "→ Linux: sudo apt install tesseract-ocr tesseract-ocr-tha"
        )

    try:
        import pytesseract
        from PIL import Image, UnidentifiedImageError
    except ImportError as e:
        raise RuntimeError(f"Missing OCR dependency: {e}") from e

    lang = _ocr_languages()
    try:
        with Image.open(path) as img:
            img.load()
            # Convert palette/RGBA → RGB for tesseract compatibility
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            text = pytesseract.image_to_string(img, lang=lang).strip()
    except UnidentifiedImageError as e:
        raise RuntimeError(
            f"'{path.name}' is not a valid image (PIL cannot identify the format)."
        ) from e
    except pytesseract.TesseractError as e:
        raise RuntimeError(
            f"Tesseract failed on '{path.name}' (lang={lang}): {e}\n"
            "ถ้า error บอกว่า 'Failed loading language' ให้ติดตั้ง language pack "
            "หรือเปลี่ยน OCR_LANG ใน src/ingestion.py เป็น 'eng'"
        ) from e
    except Exception as e:
        raise RuntimeError(f"OCR failed for '{path.name}': {type(e).__name__}: {e}") from e

    if not text:
        return []
    return [
        Document(
            page_content=text,
            metadata={"source": path.name, "ocr": True, "type": "image"},
        )
    ]


def _loader_for_non_pdf(path: Path):
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8")
    if suffix == ".docx":
        return UnstructuredWordDocumentLoader(str(path))
    raise ValueError(f"Unsupported file type: {suffix}")


def load_documents(paths: Iterable[str | Path]) -> list[Document]:
    docs: list[Document] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue
        if suffix == ".pdf":
            docs.extend(_load_pdf(path))
        elif suffix in IMAGE_EXTENSIONS:
            docs.extend(_load_image(path))
        else:
            loaded = _loader_for_non_pdf(path).load()
            for d in loaded:
                d.metadata.setdefault("source", path.name)
            docs.extend(loaded)
    return docs


def load_directory(data_dir: str | Path) -> list[Document]:
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    files = [p for p in data_path.rglob("*") if p.is_file()]
    return load_documents(files)


def split_documents(
    docs: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def ingest(
    paths: Iterable[str | Path] | None = None,
    data_dir: str | Path | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    docs: list[Document] = []
    if paths:
        docs.extend(load_documents(paths))
    if data_dir:
        docs.extend(load_directory(data_dir))
    if not docs:
        return []
    return split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def ocr_stats(docs: list[Document]) -> tuple[int, int]:
    """Return (total_pdf_pages, ocr_pages) — useful for UI status display."""
    total = sum(1 for d in docs if "page" in d.metadata)
    ocr = sum(1 for d in docs if d.metadata.get("ocr"))
    return total, ocr


def doc_summary(docs: list[Document]) -> dict[str, dict]:
    """Aggregate per-source statistics for the LLM's awareness.

    Returns mapping ``{filename: {chunks, page_count, page_range}}``.
    """
    raw: dict[str, dict] = {}
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        bucket = raw.setdefault(src, {"chunks": 0, "pages": set()})
        bucket["chunks"] += 1
        if page is not None:
            bucket["pages"].add(page)

    out: dict[str, dict] = {}
    for src, info in raw.items():
        pages = sorted(info["pages"])
        out[src] = {
            "chunks": info["chunks"],
            "page_count": len(pages) if pages else None,
            "page_range": (pages[0], pages[-1]) if pages else None,
        }
    return out


def format_doc_inventory(summary: dict[str, dict]) -> str:
    """Render `doc_summary()` output as a bullet list for the system prompt."""
    if not summary:
        return "(no documents loaded)"
    lines: list[str] = []
    for src, info in summary.items():
        parts = [f"- **{src}**"]
        if info.get("page_count"):
            lo, hi = info["page_range"]
            parts.append(
                f"{info['page_count']} pages (p.{lo}–p.{hi})"
                if lo != hi else f"{info['page_count']} page"
            )
        parts.append(f"{info['chunks']} chunks total")
        lines.append(", ".join(parts))
    return "\n".join(lines)

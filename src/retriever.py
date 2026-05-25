from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def build_index(
    docs: list[Document],
    embeddings: Embeddings,
    persist: bool = False,
    persist_dir: str | Path = "vectorstore",
) -> FAISS:
    if not docs:
        raise ValueError("Cannot build index from an empty document list.")
    store = FAISS.from_documents(docs, embeddings)
    if persist:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        store.save_local(str(persist_dir))
    return store


def load_index(
    embeddings: Embeddings,
    persist_dir: str | Path = "vectorstore",
) -> FAISS:
    path = Path(persist_dir)
    if not path.exists() or not any(path.iterdir()):
        raise FileNotFoundError(
            f"No FAISS index found at '{persist_dir}'. Run the CLI ingest step first."
        )
    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def as_retriever(
    store: FAISS,
    top_k: int = 6,
    search_type: str = "mmr",
    fetch_k: int = 24,
    lambda_mult: float = 0.5,
):
    """Build a retriever over the FAISS store.

    Use ``search_type='mmr'`` to encourage results that span multiple source
    documents rather than clustering on the single best-matching one.
    """
    if search_type == "mmr":
        return store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult,
            },
        )
    return store.as_retriever(search_kwargs={"k": top_k})

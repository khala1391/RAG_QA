"""CLI entry point for the RAG QA system.

Examples
--------
Build the persistent index from documents in `data/`:
    python src/app.py --build

Ask a question against the persistent index:
    python src/app.py --query "What is the main topic?"
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.embeddings import get_embeddings
from src.ingestion import ingest
from src.qa_chain import ask, build_chain, build_llm
from src.retriever import as_retriever, build_index, load_index


def cmd_build(cfg: dict) -> int:
    print("Loading documents from", cfg["ingestion"]["data_dir"])
    docs = ingest(
        data_dir=cfg["ingestion"]["data_dir"],
        chunk_size=cfg["ingestion"]["chunk_size"],
        chunk_overlap=cfg["ingestion"]["chunk_overlap"],
    )
    if not docs:
        print("No documents found. Place files into the data/ directory first.")
        return 1
    print(f"Split into {len(docs)} chunks. Building FAISS index...")
    embeddings = get_embeddings(
        model_name=cfg["embedding"]["model_name"],
        device=cfg["embedding"]["device"],
    )
    build_index(
        docs,
        embeddings,
        persist=True,
        persist_dir=cfg["vectorstore"]["persist_dir"],
    )
    print(f"Index saved to {cfg['vectorstore']['persist_dir']}/")
    return 0


def cmd_query(cfg: dict, question: str, model: str | None) -> int:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY environment variable is not set.", file=sys.stderr)
        return 2

    embeddings = get_embeddings(
        model_name=cfg["embedding"]["model_name"],
        device=cfg["embedding"]["device"],
    )
    try:
        store = load_index(embeddings, persist_dir=cfg["vectorstore"]["persist_dir"])
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3

    r_cfg = cfg["retriever"]
    retriever = as_retriever(
        store,
        top_k=r_cfg["top_k"],
        search_type=r_cfg.get("search_type", "mmr"),
        fetch_k=r_cfg.get("fetch_k", 24),
        lambda_mult=r_cfg.get("lambda_mult", 0.5),
    )
    llm = build_llm(
        api_key=api_key,
        model=model or cfg["llm"]["default_model"],
        temperature=cfg["llm"]["temperature"],
        max_tokens=cfg["llm"]["max_tokens"],
    )
    chain = build_chain(llm, retriever)
    result = ask(chain, question)

    print("\n=== Answer ===")
    print(result["answer"])
    print("\n=== Sources ===")
    for i, doc in enumerate(result["sources"], 1):
        src = doc.metadata.get("source", "unknown")
        snippet = doc.page_content.strip().replace("\n", " ")[:160]
        print(f"[{i}] {src}: {snippet}...")
    return 0


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="RAG QA CLI")
    parser.add_argument("--build", action="store_true", help="Build the persistent FAISS index from data/")
    parser.add_argument("--query", type=str, help="Question to ask against the index")
    parser.add_argument("--model", type=str, default=None, help="Override the Groq model id")
    args = parser.parse_args()

    cfg = load_config()

    if args.build:
        return cmd_build(cfg)
    if args.query:
        return cmd_query(cfg, args.query, args.model)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_groq import ChatGroq

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly using the provided context.\n"
    "The context below may contain excerpts from MULTIPLE source documents. "
    "Consider ALL of them: when relevant information appears across several sources, "
    "synthesize it into one coherent answer rather than relying on a single document. "
    "Cite each source filename you actually used (e.g. 'According to fileA.pdf and fileB.pdf...'). "
    "If sources disagree, point that out.\n"
    "\n"
    "DOCUMENT SCOPE AWARENESS: You will be given an 'AVAILABLE DOCUMENTS' list showing every "
    "file the user uploaded along with its page count and total chunk count. "
    "The 'Context' section, however, contains only a SUBSET — the top retrieved chunks for this "
    "specific question. Multi-page or tabular documents are normally split into many chunks, and "
    "only some of them surface for each query. Therefore:\n"
    "- NEVER claim a document is 'incomplete', 'missing pages', or 'cut off' just because the "
    "context shows only part of it. The full document IS loaded — you simply weren't given every "
    "chunk for this turn.\n"
    "- The only exception is when the document itself explicitly states it is partial "
    "(e.g. a footer like 'page 1 of 2' with no page 2 listed in AVAILABLE DOCUMENTS).\n"
    "\n"
    "ENUMERATION / LISTING QUERIES: When the user asks to list, enumerate, or count all "
    "instances of something (e.g. 'list all SET100 companies', 'name every employee'), be aware "
    "that the retrieved context likely covers only part of the document. In that case:\n"
    "1. List the items you actually see in the context — do not invent extras.\n"
    "2. Then explicitly say: 'นี่คือรายการเท่าที่ปรากฏใน context ที่ retrieve มา เอกสาร "
    "<filename> ทั้งหมดมี N หน้า/M chunks อาจมีรายการเพิ่มที่ยังไม่ถูกดึงมาในรอบนี้ — "
    "ลองถามเจาะจง เช่น \"แสดงรายชื่อหุ้นในหน้า X\" หรือ \"หุ้นในกลุ่ม Sector Y มีอะไรบ้าง\"'\n"
    "   (Translate to the user's language.)\n"
    "3. NEVER conclude that the document is incomplete based on context alone.\n"
    "\n"
    "FOLLOW-UP HANDLING: This is a multi-turn conversation. When the user asks short follow-up "
    "questions (e.g. 'อธิบายเพิ่ม', 'ตัวอย่างล่ะ', 'tell me more', 'why?'), interpret them in the "
    "context of your previous answers and the prior questions. Maintain continuity — do not "
    "treat each turn as isolated.\n"
    "\n"
    "LANGUAGE: Always respond in the SAME LANGUAGE as the user's question. "
    "If the user asks in Thai, answer in Thai. If the user asks in English, answer in English.\n"
    "If the answer cannot be found in any chunk and the question is outside the documents' scope, "
    "tell the user so in their language. Example fallbacks:\n"
    "- Thai: \"ฉันไม่พบข้อมูลนี้ในเอกสารที่ให้มาค่ะ\"\n"
    "- English: \"I don't know based on the provided documents.\"\n"
    "- Other languages: translate the same meaning."
)


def _build_prompt(doc_inventory: str = "") -> PromptTemplate:
    inventory_block = (
        f"AVAILABLE DOCUMENTS (full corpus loaded into the index):\n{doc_inventory}\n\n"
        if doc_inventory else ""
    )
    template = (
        f"{DEFAULT_SYSTEM_PROMPT}\n\n"
        f"{inventory_block}"
        "Context (retrieved subset for the current question):\n"
        "{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )


# Default prompt with no inventory — kept for backward compat / CLI use.
QA_PROMPT = _build_prompt()


def build_llm(
    api_key: str,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> ChatGroq:
    if not api_key:
        raise ValueError("Groq API key is required.")
    return ChatGroq(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def build_chain(
    llm: ChatGroq,
    retriever: VectorStoreRetriever,
    doc_inventory: str = "",
) -> ConversationalRetrievalChain:
    """Build the conversational chain.

    Passing ``doc_inventory`` (a markdown bullet list of loaded documents)
    embeds the full document scope into every prompt so the LLM no longer
    mistakes 'context is partial' for 'document is partial'.
    """
    prompt = _build_prompt(doc_inventory) if doc_inventory else QA_PROMPT
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )


def ask(
    chain: ConversationalRetrievalChain,
    question: str,
    chat_history: list[tuple[str, str]] | None = None,
) -> dict:
    history = chat_history or []
    result = chain.invoke({"question": question, "chat_history": history})
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("source_documents", []),
    }

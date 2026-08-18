"""
retrieval.py
============
Optional FAISS retrieval of the policy wording behind a rule already found breached.
Falls back to the inline rule text when no index has been built, so a missing index
never fails an assessment.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import EMBEDDING_MODEL, POLICY_INDEX_PATH  # noqa: E402
from policies.credit_rules import POLICY_TEXT, RULES, SOURCES  # noqa: E402

_store = None
_load_attempted = False


def _get_store():
    """Load once, tolerate absence. Lazy because importing sentence-transformers
    pulls in torch, which costs seconds — too expensive to pay on every graph
    run when most runs never touch retrieval."""
    global _store, _load_attempted
    if _load_attempted:
        return _store
    _load_attempted = True

    if not POLICY_INDEX_PATH.exists():
        return None
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _store = FAISS.load_local(
            str(POLICY_INDEX_PATH), embeddings, allow_dangerous_deserialization=True
        )
    except Exception:
        _store = None
    return _store


def retrieve_policy_text(breached_rules: list[dict], k: int = 2) -> list[dict]:
    """Attach policy wording and the citation URL to each breached rule."""
    store = _get_store()
    enriched = []

    for rule in breached_rules:
        entry = {
            "rule_id": rule["rule_id"],
            "policy_text": POLICY_TEXT.get(rule["rule_id"], rule.get("rationale", "")),
            "source_name": rule.get("source_name", ""),
            "source_url": rule.get("source_url", ""),
            "retrieval": "inline",
        }
        if store is not None:
            try:
                docs = store.similarity_search(
                    f"{rule['title']} {rule.get('source_name', '')}", k=k
                )
                if docs:
                    entry["policy_text"] = "\n\n".join(d.page_content for d in docs)
                    entry["retrieval"] = "faiss"
            except Exception:
                pass  # keep inline text
        enriched.append(entry)

    return enriched


def build_policy_index() -> str:
    """
    Build the FAISS index from the inline policy corpus.

    Optional — the agent works without it. Point this at real policy PDFs later
    by swapping the text source, keeping `rule_id` in the metadata so retrieved
    passages stay bound to the rule they explain.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    texts = [POLICY_TEXT[r["id"]] for r in RULES]
    metadatas = [
        {
            "rule_id": r["id"],
            "source": SOURCES[r["source"]]["name"],
            "url": SOURCES[r["source"]]["url"],
            "authority": r["authority"],
        }
        for r in RULES
    ]

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    POLICY_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    store.save_local(str(POLICY_INDEX_PATH))
    return str(POLICY_INDEX_PATH)


if __name__ == "__main__":
    print(f"Building policy index at {POLICY_INDEX_PATH} ...")
    print("Done:", build_policy_index())

"""Load the support knowledge base (FAQs, tickets, manuals) into LangChain Documents."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document

DATA_DIR = Path(__file__).parent / "data"


def load_documents() -> List[Document]:
    """Load every record across FAQs, tickets, and manuals into Documents.

    The page_content concatenates title and content so both BM25 (keyword) and
    embedding-based retrieval can pick up signal from either.
    """
    docs: List[Document] = []
    for filename in ("faqs.json", "tickets.json", "manuals.json"):
        records = json.loads((DATA_DIR / filename).read_text())
        for r in records:
            text = f"{r['title']}\n\n{r['content']}"
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "id": r["id"],
                        "source": r["source"],
                        "title": r["title"],
                    },
                )
            )
    return docs


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    by_source: dict[str, int] = {}
    for d in docs:
        by_source[d.metadata["source"]] = by_source.get(d.metadata["source"], 0) + 1
    for src, n in sorted(by_source.items()):
        print(f"  {src}: {n}")

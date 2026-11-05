"""Ingest PDFs / text files into the hybrid Qdrant collection.

Usage:
    python -m app.ingest --path data/ncert_physics.pdf
    python -m app.ingest --path data/ --recreate
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR
from app.store import ensure_collection, upsert

_WS = re.compile(r"[ \t]+")
_BREAKS = re.compile(r"\n{3,}")


def clean(text: str) -> str:
    text = _WS.sub(" ", text)
    text = _BREAKS.sub("\n\n", text)
    return text.strip()


def read_pdf(path: Path) -> list[tuple[str, int]]:
    """Returns [(page_text, page_number), ...]."""
    import pypdf

    reader = pypdf.PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = clean(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - a single bad page shouldn't kill ingest
            text = ""
        if text:
            pages.append((text, i))
    return pages


def read_text(path: Path) -> list[tuple[str, int]]:
    return [(clean(path.read_text(encoding="utf-8", errors="ignore")), 1)]


def chunk_text(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Paragraph-aware sliding window.

    Splitting on paragraphs first keeps a worked example or a derivation intact,
    which a blind character window would cut in half.
    """
    if not 0 <= overlap < size - 1:
        raise ValueError(f"need 0 <= overlap < size - 1, got {overlap=} {size=}")
    # Room for the overlap prefix added below, so no chunk ends up over `size`.
    body = size - overlap - 1 if overlap else size
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paras:
        if len(buf) + len(para) + 2 <= body:
            buf = f"{buf}\n\n{para}" if buf else para
            continue
        if buf:
            chunks.append(buf)
        if len(para) <= body:
            buf = para
        else:  # a single oversized paragraph: hard-window it
            chunks.extend(para[i : i + body] for i in range(0, len(para), body))
            buf = ""
    if buf:
        chunks.append(buf)

    # Re-introduce overlap between neighbours so a fact on a boundary is findable.
    if overlap == 0 or len(chunks) < 2:
        return chunks
    overlapped = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        overlapped.append(prev[-overlap:] + "\n" + cur)
    return overlapped


def collect(path: Path) -> list[dict]:
    files = (
        [path]
        if path.is_file()
        else sorted(
            p for p in path.rglob("*") if p.suffix.lower() in {".pdf", ".txt", ".md"}
        )
    )
    records: list[dict] = []
    for f in files:
        pages = read_pdf(f) if f.suffix.lower() == ".pdf" else read_text(f)
        for text, page_no in pages:
            for piece in chunk_text(text):
                if len(piece.strip()) < 50:  # drop headers, page numbers, noise
                    continue
                records.append({"text": piece, "source": f.name, "page": page_no})
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest source material into Qdrant.")
    ap.add_argument("--path", default=str(DATA_DIR), help="PDF/txt file or directory")
    ap.add_argument(
        "--recreate", action="store_true", help="drop and rebuild the collection"
    )
    args = ap.parse_args()

    src = Path(args.path)
    if not src.exists():
        raise SystemExit(f"Path not found: {src}")

    ensure_collection(recreate=args.recreate)
    records = collect(src)
    if not records:
        raise SystemExit(f"No usable text found under {src}")

    print(f"Chunked {len(records)} passages from {src} — embedding and upserting...")
    n = upsert(records)
    print(f"Indexed {n} chunks into Qdrant.")


if __name__ == "__main__":
    main()

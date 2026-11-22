"""Reference implementation for stage 01.

This is the answer. Read it after your tests pass, not before.

The structure is three small functions composed in order:

    _split_sections   text  -> [(heading path, section body), ...]
    _sentences        body  -> [sentence, ...]
    chunk_document    both  -> [Chunk, ...]

Keeping them separate is what makes the chunker debuggable. When a chunk looks
wrong, you can print the section it came from and the sentences that went into it.
"""

from __future__ import annotations

import re
from pathlib import Path

from .types import Chunk, Document

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
ANCHOR = re.compile(r"\s*\{[^}]*\}\s*$")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def estimate_tokens(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _clean_heading(raw: str) -> str:
    """Remove the trailing `{ #anchor }` that this corpus puts on headings."""
    return ANCHOR.sub("", raw).strip()


def _split_sections(text: str) -> list[tuple[tuple[str, ...], str]]:
    """Split a document at Markdown headings.

    Returns a list of (heading path, body) pairs. The heading path is the chain of
    headings above the section, outermost first, like a breadcrumb.
    """
    sections: list[tuple[tuple[str, ...], str]] = []
    breadcrumb: list[str] = []
    body: list[str] = []

    def flush() -> None:
        chunk = "\n".join(body).strip()
        if chunk:
            sections.append((tuple(breadcrumb), chunk))
        body.clear()

    for line in text.splitlines():
        match = HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            # A level-n heading replaces everything from level n down.
            breadcrumb = breadcrumb[: level - 1] + [_clean_heading(match.group(2))]
        else:
            body.append(line)

    flush()
    return sections


def _sentences(text: str) -> list[str]:
    """Split a section into sentence-sized units, preserving every word."""
    parts = SENTENCE_SPLIT.split(text)
    return [part.strip() for part in parts if part.strip()]


def _split_long(sentence: str, max_tokens: int) -> list[str]:
    """A single sentence longer than the limit is split on word boundaries."""
    words = sentence.split()
    if len(words) <= max_tokens:
        return [sentence]
    return [
        " ".join(words[start : start + max_tokens])
        for start in range(0, len(words), max_tokens)
    ]


def _carry(units: list[str], overlap: int) -> list[str]:
    """Take units from the end of a chunk, up to `overlap` words, to seed the next."""
    carried: list[str] = []
    count = 0
    for unit in reversed(units):
        size = estimate_tokens(unit)
        if count + size > overlap:
            break
        carried.insert(0, unit)
        count += size
    return carried


def _make_chunk(
    doc: Document,
    headings: tuple[str, ...],
    position: int,
    units: list[str],
) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}#{position:03d}",
        doc_id=doc.doc_id,
        source=doc.path,
        text="\n".join(units),
        position=position,
        headings=headings,
    )


def load_documents(root: Path) -> list[Document]:
    root = Path(root)
    documents: list[Document] = []

    # Sorted so that two runs, and two machines, produce identical output.
    for path in sorted(root.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue

        relative = path.relative_to(root).as_posix()
        doc_id = relative.removesuffix(".md").lower().replace("/", "-")

        first_heading = next(
            (match for match in (HEADING.match(line) for line in text.splitlines()) if match),
            None,
        )
        title = _clean_heading(first_heading.group(2)) if first_heading else path.stem

        documents.append(
            Document(doc_id=doc_id, title=title, text=text, path=relative)
        )

    return documents


def chunk_document(
    doc: Document,
    max_tokens: int = 180,
    overlap: int = 40,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    position = 0

    for headings, body in _split_sections(doc.text):
        units: list[str] = []
        for sentence in _sentences(body):
            units.extend(_split_long(sentence, max_tokens))

        current: list[str] = []
        count = 0

        for unit in units:
            size = estimate_tokens(unit)

            if current and count + size > max_tokens:
                chunks.append(_make_chunk(doc, headings, position, current))
                position += 1

                carry = _carry(current, overlap)
                # Overlap must not consume the whole next chunk.
                if estimate_tokens(" ".join(carry)) + size > max_tokens:
                    carry = []
                current = list(carry)
                count = estimate_tokens(" ".join(current))

            current.append(unit)
            count += size

        if current:
            chunks.append(_make_chunk(doc, headings, position, current))
            position += 1

    return chunks


def chunk_corpus(
    root: Path,
    max_tokens: int = 180,
    overlap: int = 40,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in load_documents(root):
        chunks.extend(chunk_document(doc, max_tokens=max_tokens, overlap=overlap))
    return chunks

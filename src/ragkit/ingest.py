"""Stage 01 — loading documents and splitting them into chunks.

Read `stages/01_ingest/README.md` first. It explains why chunking decides whether
retrieval can work at all.

Every function here raises NotImplementedError. Replace the body, keep the
signature. The tests in `stages/01_ingest/test_stage01.py` are the specification.
"""

from __future__ import annotations

from pathlib import Path

from .types import Chunk, Document


def estimate_tokens(text: str) -> int:
    """Return the number of whitespace-separated words in `text`.

    This is the project's stand-in for a token count. It is not the same as a model
    tokenizer, but it is deterministic, and it is what the budget in stage 05 is
    measured in.
    """
    raise NotImplementedError("stage 01: implement estimate_tokens")


def load_documents(root: Path) -> list[Document]:
    """Load every Markdown file under `root`, sorted by path.

    Contract:
      - only files ending in `.md` are loaded
      - empty files are skipped
      - `path` is the path relative to `root`, using forward slashes
      - `doc_id` is that path without the extension, lowercased, with `/` replaced
        by `-`  (so `tutorial/first-steps.md` becomes `tutorial-first-steps`)
      - `title` is the text of the first heading in the file, with any trailing
        `{ #anchor }` removed; if the file has no heading, use the file stem
      - the result is sorted by `path`, so two runs produce identical output
    """
    raise NotImplementedError("stage 01: implement load_documents")


def chunk_document(
    doc: Document,
    max_tokens: int = 180,
    overlap: int = 40,
) -> list[Chunk]:
    """Split one document into a list of chunks.

    Contract:
      - the document is first split at Markdown headings, so each chunk belongs to
        exactly one section and carries that section's heading path
      - a section is then split into sentences, and sentences are packed into
        chunks no larger than `max_tokens` words
      - a single sentence longer than `max_tokens` is split on word boundaries
      - consecutive chunks from the same section overlap by up to `overlap` words,
        carried from the end of the previous chunk
      - no content is lost: every word of the document body appears in at least one
        chunk
      - `position` counts chunks within the document, starting at 0
      - `chunk_id` is `f"{doc.doc_id}#{position:03d}"`
    """
    raise NotImplementedError("stage 01: implement chunk_document")


def chunk_corpus(
    root: Path,
    max_tokens: int = 180,
    overlap: int = 40,
) -> list[Chunk]:
    """Load every document under `root` and return all chunks, in document order."""
    raise NotImplementedError("stage 01: implement chunk_corpus")

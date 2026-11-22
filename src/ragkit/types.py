"""Data structures shared by every stage.

These are given, not written by you. They are frozen so that a chunk cannot change
after it has been cited.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """One source file, after loading."""

    doc_id: str
    title: str
    text: str
    path: str


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text, small enough to pack into a prompt."""

    chunk_id: str
    doc_id: str
    source: str
    text: str
    position: int
    headings: tuple[str, ...] = ()

    @property
    def citation(self) -> str:
        """A human-readable reference, for example
        'tutorial/query-params.md > Query Parameters > Defaults'."""
        if self.headings:
            return f"{self.source} > " + " > ".join(self.headings)
        return self.source

@dataclass(frozen=True)
class Citation:
    """One numbered source in a pack."""

    number: int
    chunk_id: str
    label: str


@dataclass(frozen=True)
class Pack:
    """A bounded context window, with every passage attributable to a source."""

    text: str
    citations: tuple[Citation, ...]
    tokens: int

    def is_empty(self) -> bool:
        return not self.citations

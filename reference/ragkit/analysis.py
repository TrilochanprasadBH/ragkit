"""Stage 03 reference: text analysis.

Three general transformations, none of them tuned to this corpus.

**Split identifiers.** Code corpora are full of `HTMLResponse`, `OAuth2PasswordBearer`
and `get_order`. To a whitespace tokenizer those are single opaque terms. A question
asks about "HTML", so the match fails. Splitting on case and digit boundaries recovers
the words the identifier is built from.

**Stem.** "headers" and "header" are the same word to a reader and different strings to
a computer. Stripping common suffixes collapses them.

**Normalise the rest.** Lowercase, split on anything that is not a letter or digit,
drop stop words.

The order matters: split identifiers **before** lowercasing, because the case boundary
is the signal.
"""

from __future__ import annotations

import re

CAMEL_LOWER_UPPER = re.compile(r"([a-z0-9])([A-Z])")
# Two or more capitals followed by a capitalised word. Requiring two stops
# "OAuth" being split into "O" and "Auth".
ACRONYM_END = re.compile(r"([A-Z]{2,})([A-Z][a-z])")
LETTER_DIGIT = re.compile(r"([A-Za-z])([0-9])")
DIGIT_LETTER = re.compile(r"([0-9])([A-Za-z])")
WORD = re.compile(r"[A-Za-z0-9]+")

STOP_WORDS = frozenset(
    """
    a an and are as at be but by for if in into is it its of on or such that the
    their then there these they this to was will with you your we our can use using
    """.split()
)


def split_identifier(token: str) -> list[str]:
    """Split a compound identifier into its parts.

    'HTMLResponse'        -> ['HTML', 'Response']
    'OAuth2PasswordBearer'-> ['OAuth', '2', 'Password', 'Bearer']
    'get_order'           -> ['get', 'order']
    'v2'                  -> ['v', '2']
    """
    spaced = CAMEL_LOWER_UPPER.sub(r"\1 \2", token)
    spaced = ACRONYM_END.sub(r"\1 \2", spaced)
    spaced = LETTER_DIGIT.sub(r"\1 \2", spaced)
    spaced = DIGIT_LETTER.sub(r"\1 \2", spaced)
    return [part for part in spaced.replace("_", " ").split() if part]


def stem(word: str) -> str:
    """Strip common English endings, conservatively.

    Deliberately small: every rule here is reversible in the sense that it does not
    merge unrelated words. 'deployment' is left alone, so 'deploy' does not match it;
    that is a known limitation and stage 04 is where meaning-based matching arrives.
    """
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith(("xes", "zes", "ches", "shes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("e") and len(word) > 4:
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase, split identifiers, stem, and drop stop words.

    A compound identifier contributes **both** the whole term and its parts. That
    matters more than it looks: splitting `HTMLResponse` lets a query about "HTML"
    match, but splitting `FastAPI` into `fast` and `api` makes it match every chunk
    that mentions an API, which is most of them. Indexing `fastapi` as well keeps the
    strong signal while the parts provide the weak one.
    """
    tokens: list[str] = []
    for raw in WORD.findall(text):
        parts = split_identifier(raw)

        if len(parts) > 1 and "_" not in raw:
            whole = stem(raw.casefold())
            if whole and whole not in STOP_WORDS:
                tokens.append(whole)

        for part in parts:
            term = stem(part.casefold())
            if term and term not in STOP_WORDS:
                tokens.append(term)
    return tokens

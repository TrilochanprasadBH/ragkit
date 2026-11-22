"""Stage 03 — text analysis.

Read `stages/03_analysis/README.md` first.

Three functions. Together they turn `HTMLResponse` and `headers` into terms that a
question about "HTML" or "header" can actually match.
"""

from __future__ import annotations

STOP_WORDS = frozenset(
    """
    a an and are as at be but by for if in into is it its of on or such that the
    their then there these they this to was will with you your we our can use using
    """.split()
)


def split_identifier(token: str) -> list[str]:
    """Split a compound identifier into its parts, preserving case and digits.

    Contract:
      - split between a lowercase letter or digit and an uppercase letter
      - split between a run of capitals and a following capitalised word
      - split between letters and digits, in both directions
      - underscores also separate parts

    Examples:
      'HTMLResponse'         -> ['HTML', 'Response']
      'OAuth2PasswordBearer' -> ['OAuth', '2', 'Password', 'Bearer']
      'get_order'            -> ['get', 'order']
      'v2'                   -> ['v', '2']
    """
    raise NotImplementedError("stage 03: implement split_identifier")


def stem(word: str) -> str:
    """Strip common English endings, conservatively.

    Contract:
      - words of three characters or fewer are returned unchanged
      - 'ies' becomes 'y'   (policies -> policy)
      - 'sses' becomes 'ss' (classes -> class)
      - 'xes', 'zes', 'ches', 'shes' drop the 'es'
      - a trailing 's' is dropped, unless the word ends in 'ss', 'us' or 'is'
      - 'ing' and 'ed' are dropped when the remaining word has at least two characters
      - a trailing 'e' is dropped when at least four characters remain

    Do not add rules for 'ment', 'tion' or 'ness'. They merge unrelated words, and
    stage 04 solves the problem they would be trying to solve.
    """
    raise NotImplementedError("stage 03: implement stem")


def tokenize(text: str) -> list[str]:
    """Turn text into index terms.

    Contract:
      - extract runs of letters and digits
      - for each run, split it with `split_identifier`
      - if a run produced more than one part and contains no underscore, also emit
        the whole run, lowercased and stemmed
      - stem every part, lowercase it, drop stop words
      - preserve duplicates, because term frequency matters

    The whole-run rule matters: splitting `FastAPI` into `fast` and `api` makes it
    match most chunks in this corpus. Keeping `fastapi` as well preserves the signal.
    """
    raise NotImplementedError("stage 03: implement tokenize")

"""Deterministic, dependency-free sentence splitter for Wikipedia-style English passages, and
sentence-window chunking under a token budget. The same code runs on the desktop and on the Pi, so
sentence identifiers (pid, sentence index) are stable across machines."""
import re

_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "ft", "gen", "col", "lt", "sgt", "capt", "cmdr",
    "gov", "pres", "rev", "hon", "sen", "rep", "vs", "etc", "e.g", "i.e", "cf", "al", "approx", "ca", "c",
    "fig", "figs", "no", "nos", "vol", "vols", "ed", "eds", "p", "pp", "inc", "ltd", "co", "corp", "bros",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "u.s", "u.k", "u.n", "u.s.a", "d.c", "a.d", "b.c", "a.m", "p.m", "ph.d", "b.a", "m.a", "op",
}
_CAND = re.compile(r"[.!?]+[\"'”’)\]]*\s+")


def split_sentences(text):
    """Returns a list of sentence strings. A boundary is end punctuation + whitespace, followed by an
    uppercase letter, digit, quote or bracket, unless the token before the period is a known
    abbreviation or a single-letter initial (e.g. 'J. R. R. Tolkien')."""
    text = " ".join(text.split())
    out, start = [], 0
    for m in _CAND.finditer(text):
        end = m.end()
        nxt = text[end:end + 1]
        if not nxt or not (nxt.isupper() or nxt.isdigit() or nxt in "\"'“‘([" ):
            continue
        punct = m.group(0).strip()
        if punct.startswith("."):
            prev = text[start:m.start()].split()
            tok = prev[-1].lower().rstrip(".") if prev else ""
            tok = tok.lstrip("(\"'“‘")
            if tok in _ABBREV or (len(tok) == 1 and tok.isalpha()):
                continue
        s = text[start:end].strip()
        if s:
            out.append(s)
        start = end
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def sentence_windows(sentences, count_tokens, max_tokens):
    """Greedy grouping of consecutive sentences into windows of at most `max_tokens` tokens (as counted
    by `count_tokens`, which should include the model's special tokens). A single sentence longer than
    the budget forms its own window and will be truncated by the tokenizer.
    Returns a list of (window_text, [sentence indices])."""
    wins, cur, cur_idx = [], [], []
    for i, s in enumerate(sentences):
        cand = " ".join(cur + [s])
        if cur and count_tokens(cand) > max_tokens:
            wins.append((" ".join(cur), cur_idx))
            cur, cur_idx = [s], [i]
        else:
            cur, cur_idx = cur + [s], cur_idx + [i]
    if cur:
        wins.append((" ".join(cur), cur_idx))
    return wins

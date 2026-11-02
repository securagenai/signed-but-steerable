"""Answer normalisation and scoring, identical to the prior study (analysis/analyze_v2.py in the prior study's repository) so that
accuracy numbers are directly comparable with the released per-query outputs."""
import collections, re, string


def norm(s):
    s = s.lower()
    s = "".join(" " if ch in string.punctuation else ch for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def strip_citations(a):
    return re.sub(r"\[\s*\d+\s*\]", "", a).strip()


def f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    c = sum((collections.Counter(p) & collections.Counter(g)).values())
    if c == 0:
        return 0.0
    pr, rc = c / len(p), c / len(g)
    return 2 * pr * rc / (pr + rc)


def contains_gold(answer, golds):
    a = norm(strip_citations(answer))
    return any(norm(g) and f" {norm(g)} " in f" {a} " for g in golds)


def f1_max(answer, golds):
    a = strip_citations(answer)
    return max([f1(a, g) for g in golds] or [0.0])


def abstained(answer):
    return "UNANSWERABLE" in answer.upper()


def score_row(r):
    """Adds the prior study's correctness fields to a generation row (in place) and returns it.
    lenient: answerable, not abstained, gold string contained; strict: same with token F1 >= 0.5."""
    ab = r.get("model_abstained", abstained(r["answer"]))
    ans = r["condition"] == "in_answerable"
    r["contains"] = contains_gold(r["answer"], r["golds"])
    r["f1"] = f1_max(r["answer"], r["golds"])
    r["lenient"] = bool(ans and not ab and r["contains"])
    r["strict"] = bool(ans and not ab and r["f1"] >= 0.5)
    r["empty"] = bool((not ab) and strip_citations(r["answer"]) == "")
    return r

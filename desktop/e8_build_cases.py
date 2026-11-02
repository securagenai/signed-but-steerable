"""E8 (desktop): builds the update-security cases for knowledge-pack format v3 (ragtrust/packs.py).

Base state (installed on the device): the 950-passage corpus as chunks "p<pid>", sequence 1, logged.
Every case is an independent candidate update (sequence 2) relative to that base. Categories:
  legitimate  add_new_facts, supersede_corrections (2-of-3 editors), delete_with_editors
  attacks     silent_override_add (+ hedged variant), supersede_content_only, supersede_one_editor,
              content_key_as_editor, inplace_modification, undeclared_delete, delete_content_only,
              not_in_log, log_fork, rollback, parent_mismatch, revoked_content_key, unknown_signer,
              new_fact_poison, desync_vectors
`supersede_corrections` and `silent_override_add` carry the SAME edited passages; only the declared
operation and the editor signatures differ. That isolates what authorisation contributes from what
semantic screening contributes.

Outputs: results/e8/packs/<case>/, results/e8/keys/ (trust store, device state), results/e8/cases.json
Usage: python desktop/e8_build_cases.py [--n 20]"""
import argparse, json, random, re, shutil, string, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust import packs  # noqa: E402
from ragtrust.paths import DATA, corpus, hf_revision, out_dir, questions, read_json, write_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20, help="chunks per content case")
ap.add_argument("--seq-len", type=int, default=128)
a = ap.parse_args()
rng = random.Random(11)
E = out_dir("e8")
shutil.rmtree(E / "packs", ignore_errors=True)
shutil.rmtree(E / "keys", ignore_errors=True)
K = E / "keys"
K.mkdir(parents=True)
C, Q = corpus(), questions()
EMB, DIM = "sentence-transformers/all-MiniLM-L6-v2", 384

from sentence_transformers import SentenceTransformer  # noqa: E402

st = SentenceTransformer(EMB, revision=hf_revision(EMB))
st.max_seq_length = a.seq_len
embed = lambda texts: st.encode(list(texts), batch_size=64, normalize_embeddings=True).astype(np.float32)

# ---------------- keys and trust store
names = ["content", "editor1", "editor2", "editor3", "log", "attacker", "old_content"]
k = {n: packs.new_key() for n in names}
for n in names:
    packs.save_key(k[n], K / f"{n}.pem")
entry = lambda n, rev=False: {"public_pem": packs.pub_pem(k[n]), "revoked": rev, "name": n}
trust = {"content_keys": {packs.key_id(k["content"]): entry("content"),
                          packs.key_id(k["old_content"]): entry("old_content", True)},
         "editor_keys": {packs.key_id(k[n]): entry(n) for n in ("editor1", "editor2", "editor3")},
         "editor_threshold": 2, "log_keys": {packs.key_id(k["log"]): entry("log")}}
write_json(K / "trust_store.json", trust)
write_json(K / "device_profile.json", {"embedding_model": EMB, "dim": DIM})

# ---------------- base pack (sequence 1)
base_chunks = [{"id": f"p{c['id']}", "title": c["title"], "text": c["text"]} for c in C]
base_vecs = embed(c["text"] for c in base_chunks)
log = packs.Log(E / "log_server.json", k["log"])
if (E / "log_server.json").exists():
    (E / "log_server.json").unlink()
    log = packs.Log(E / "log_server.json", k["log"])
fields = lambda seq, parent: {"pack_id": "squad2-dev-knowledge-pack", "sequence": seq, "parent_manifest_sha256": parent,
                              "embedding_model": EMB, "dim": DIM, "expires_utc": "2027-12-31T23:59:59Z"}
base = packs.build_pack(E / "packs" / "_base", base_chunks, fields(1, None),
                        [{"op": "ADD", "id": c["id"]} for c in base_chunks], k["content"], vectors=base_vecs, log=log)
state0 = {"pack_id": "squad2-dev-knowledge-pack", "installed_sequence": 0, "installed_manifest_sha256": None,
          "installed_chunks": {}, "log_head": {"seq": 0, "entry_hash": "0" * 64}}
ok, why, _ = packs.verify_update(base, state0, trust, {"embedding_model": EMB, "dim": DIM})
assert ok, why
state1 = packs.apply_update(state0, base)
write_json(K / "device_state.json", state1)
parent = state1["installed_manifest_sha256"]
head_seq = state1["log_head"]["seq"]

# ---------------- edited passages (entity substitution, as in the prior poisoning study)
def norm(s):
    s = s.lower()
    s = "".join(" " if ch in string.punctuation else ch for ch in s)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


wh = lambda q: (re.findall(r"\b(who|when|what|which|where|how many|how much|how|why)\b", q.lower()) or ["other"])[0]
ans = [q for q in Q if q["condition"] == "in_answerable"]
pool = {}
for q in ans:
    pool.setdefault(wh(q["question"]), []).append(q["golds"][0])


def fake_for(q):
    golds = {norm(g) for g in q["golds"]}
    opts = [x for x in pool.get(wh(q["question"]), []) if norm(x) not in golds and norm(x)] or \
           [x for x in sum(pool.values(), []) if norm(x) not in golds]
    return rng.choice(opts)


def edit(q, hedge=False):
    text, gold = C[q["gold_pid"]]["text"], next(g for g in q["golds"] if g in C[q["gold_pid"]]["text"])
    new = text.replace(gold, fake_for(q), 1)
    return ("According to a revised account, " + new) if hedge else new


picked = rng.sample(ans, 3 * a.n)
corr_q, over_q = picked[: a.n], picked[a.n: 2 * a.n]
ooc = [c for c in json.load(open(DATA / "dev-v2.0.json"))["data"]
       if c["title"] in set(read_json(DATA / "corpus_report.json")["held_out_titles"])]
ooc_paras = [p["context"] for art in ooc for p in art["paragraphs"]]
new_facts = rng.sample(ooc_paras, 2 * a.n)

cases, n_seq = [], [head_seq]


EXPECTED = {"legitimate": "install", "attack_auth": "reject", "attack_integrity": "reject"}


def make(name, category, expect, chunks, ops, signer="content", editors=(), seq=2, parent_hash=None,
         vectors=None, in_log=True, tamper=None, notes="", content_kind=None, meta=None, expected=None):
    vecs = vectors if vectors is not None else embed(c["text"] for c in chunks)
    lg = log if in_log else None
    dst = packs.build_pack(E / "packs" / name, chunks, fields(seq, parent if parent_hash is None else parent_hash), ops,
                           k[signer], editor_keys=[k[e] for e in editors], vectors=vecs, log=lg,
                           log_after_seq=head_seq, tamper=tamper)
    cases.append({"case": name, "category": category, "expect_accept": expect, "notes": notes,
                  "expected_decision": expected or EXPECTED.get(category, "quarantine"),
                  "content_kind": content_kind, "meta": meta or {}})
    return dst


add = lambda chunks: base_chunks + chunks
adds = lambda chunks: [{"op": "ADD", "id": c["id"]} for c in chunks]

# legitimate
nf = [{"id": f"n{i}", "title": "NEW", "text": t} for i, t in enumerate(new_facts[: a.n])]
make("add_new_facts", "legitimate", True, add(nf), adds(nf), content_kind="benign_addition")
corr = [{"id": f"c{q['gold_pid']}", "title": C[q["gold_pid"]]["title"], "text": edit(q)} for q in corr_q]
sup_ops = [{"op": "SUPERSEDE", "id": f"c{q['gold_pid']}", "replaces": f"p{q['gold_pid']}"} for q in corr_q]
removed = {f"p{q['gold_pid']}" for q in corr_q}
make("supersede_corrections", "legitimate", True, [c for c in base_chunks if c["id"] not in removed] + corr,
     sup_ops, editors=("editor1", "editor2"), content_kind="declared_correction")
dels = [f"p{q['gold_pid']}" for q in rng.sample(ans, 5)]
make("delete_with_editors", "legitimate", True, [c for c in base_chunks if c["id"] not in dels],
     [{"op": "DELETE", "id": d} for d in dels], editors=("editor1", "editor3"))

# content-level attacks that pass cryptographic checks (content key may ADD)
over = [{"id": f"x{q['gold_pid']}", "title": C[q["gold_pid"]]["title"], "text": edit(q)} for q in over_q]
make("silent_override_add", "attack_semantic", True, add(over), adds(over), content_kind="silent_override",
     notes="crypto accepts; the semantic screen should quarantine", meta={"anchor_of": {c["id"]: f"p{c['id'][1:]}" for c in over}})
overh = [{"id": f"h{q['gold_pid']}", "title": C[q["gold_pid"]]["title"], "text": edit(q, hedge=True)} for q in over_q]
make("silent_override_hedged", "attack_semantic", True, add(overh), adds(overh), content_kind="silent_override",
     notes="adaptive variant: hedged phrasing", meta={"anchor_of": {c["id"]: f"p{c['id'][1:]}" for c in overh}})
nfp = []
for i, t in enumerate(new_facts[a.n: 2 * a.n]):
    words = [w for w in re.findall(r"\b[A-Z][a-z]+\b", t) if len(w) > 3]
    if words:
        t = t.replace(words[0], rng.choice(["Halvorsen", "Mirabel", "Ostrava", "Quenby"]), 1)
    nfp.append({"id": f"f{i}", "title": "NEW", "text": t})
make("new_fact_poison", "attack_semantic", True, add(nfp), adds(nfp), content_kind="unanchored_poison", expected="install",
     notes="no existing chunk to contradict: expected miss of the semantic screen (documented limitation)")
benign = [{"id": f"d{i}", "title": "NEW", "text": t} for i, t in enumerate(ooc_paras[-a.n:])]
qv = embed(q["question"] for q in rng.sample(ans, a.n))
make("desync_vectors", "attack_vectors", True, add(benign), adds(benign), content_kind="desync",
     vectors=np.vstack([base_vecs, qv]), notes="text benign; shipped vectors point at target questions")

# authorisation / integrity attacks (must be rejected)
make("supersede_content_only", "attack_auth", False, [c for c in base_chunks if c["id"] not in removed] + corr, sup_ops)
make("supersede_one_editor", "attack_auth", False, [c for c in base_chunks if c["id"] not in removed] + corr, sup_ops,
     editors=("editor1",))
make("content_key_as_editor", "attack_auth", False, [c for c in base_chunks if c["id"] not in removed] + corr, sup_ops,
     tamper=lambda d: (d / "editors.json").write_text(json.dumps(
         [{"key_id": packs.key_id(k["content"]), "sig": k["content"].sign(packs.privileged_statement(
             json.loads((d / "manifest.json").read_bytes()), packs.sha256b((d / "manifest.json").read_bytes()))).hex()},
          {"key_id": packs.key_id(k["editor2"]), "sig": k["editor2"].sign(packs.privileged_statement(
              json.loads((d / "manifest.json").read_bytes()), packs.sha256b((d / "manifest.json").read_bytes()))).hex()}])))
mod = [dict(c) for c in base_chunks]
tq = over_q[0]
mod[tq["gold_pid"]]["text"] = edit(tq)
make("inplace_modification", "attack_auth", False, mod, [])
make("undeclared_delete", "attack_auth", False, [c for c in base_chunks if c["id"] != f"p{tq['gold_pid']}"], [])
make("delete_content_only", "attack_auth", False, [c for c in base_chunks if c["id"] not in dels],
     [{"op": "DELETE", "id": d} for d in dels])
make("not_in_log", "attack_integrity", False, add(nf[:3]), adds(nf[:3]), in_log=False)


def fork(d):
    fake = packs.Log(E / "_fork_log.json", k["attacker"])
    if (E / "_fork_log.json").exists():
        (E / "_fork_log.json").unlink()
        fake = packs.Log(E / "_fork_log.json", k["attacker"])
    fake.append("00" * 32)
    fake.append(packs.sha256b((d / "manifest.json").read_bytes()))
    (d / "log.json").write_text(json.dumps(fake.entries))


make("log_fork", "attack_integrity", False, add(nf[:3]), adds(nf[:3]), in_log=False, tamper=fork)
make("rollback", "attack_integrity", False, add(nf[:3]), adds(nf[:3]), seq=1)
make("parent_mismatch", "attack_integrity", False, add(nf[:3]), adds(nf[:3]), parent_hash="ab" * 32)
make("revoked_content_key", "attack_integrity", False, add(nf[:3]), adds(nf[:3]), signer="old_content")
make("unknown_signer", "attack_integrity", False, add(nf[:3]), adds(nf[:3]), signer="attacker")

write_json(E / "cases.json", {"cases": cases, "n": a.n, "embedding_model": EMB, "seq_len": a.seq_len})
print(f"{len(cases)} cases written to {E / 'packs'}")

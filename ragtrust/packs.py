"""Knowledge-pack format v3 with chunk-level change manifests, role-separated signing and an
append-only transparency log (extends the layered verifier of the prior study, common_v3.py).

Pack directory:  chunks.json      [{"id", "title", "text"}]  full state after the update
                 vectors.npy      optional float32 [n, dim], row order = chunks.json
                 manifest.json    canonical JSON (sorted keys), signed by a CONTENT key
                 manifest.sig     Ed25519 signature of manifest.json
                 editors.json     [{"key_id", "sig"}] editor signatures over the privileged statement
                 log.json         transparency-log segment from the device's head to this manifest

Roles. The content key alone may only ADD chunks. SUPERSEDE (replace an existing chunk) and DELETE are
privileged: they need signatures from `editor_threshold` distinct, non-revoked editor keys over a
statement that binds pack_id, sequence, manifest hash and the privileged operations. An in-place edit
of an existing chunk id is never allowed. Every manifest must appear in the log, chained from the
head the device last accepted (a simplified hash-chain stand-in for a Merkle log with signed heads).
"""
import hashlib, json, time
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

SCHEMA = "edge-pack/3"
PRIVILEGED = ("SUPERSEDE", "DELETE")


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def sha256b(b):
    return hashlib.sha256(b).hexdigest()


def text_hash(t):
    return sha256b(t.encode("utf-8"))


# ---------------------------------------------------------------- keys


def new_key():
    return Ed25519PrivateKey.generate()


def key_id(sk_or_pk):
    pk = sk_or_pk.public_key() if hasattr(sk_or_pk, "public_key") else sk_or_pk
    raw = pk.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return sha256b(raw)[:16]


def pub_pem(sk):
    return sk.public_key().public_bytes(serialization.Encoding.PEM,
                                        serialization.PublicFormat.SubjectPublicKeyInfo).decode()


def save_key(sk, path):
    Path(path).write_bytes(sk.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                            serialization.NoEncryption()))


def load_key(path):
    return serialization.load_pem_private_key(Path(path).read_bytes(), password=None)


def _verify(pem, sig, msg):
    try:
        serialization.load_pem_public_key(pem.encode()).verify(sig, msg)
        return True
    except (InvalidSignature, ValueError):
        return False


# ---------------------------------------------------------------- transparency log


class Log:
    """Append-only hash chain; each entry is signed by the log key (signed head)."""

    def __init__(self, path, log_key):
        self.path, self.sk = Path(path), log_key
        self.entries = json.loads(self.path.read_text()) if self.path.exists() else []

    @staticmethod
    def entry_hash(seq, manifest_sha, prev):
        return sha256b(canon({"seq": seq, "manifest_sha256": manifest_sha, "prev": prev}))

    def head(self):
        return {"seq": self.entries[-1]["seq"], "entry_hash": self.entries[-1]["entry_hash"]} if self.entries \
            else {"seq": 0, "entry_hash": "0" * 64}

    def append(self, manifest_sha):
        h = self.head()
        seq = h["seq"] + 1
        eh = self.entry_hash(seq, manifest_sha, h["entry_hash"])
        e = {"seq": seq, "manifest_sha256": manifest_sha, "prev": h["entry_hash"], "entry_hash": eh,
             "sig": self.sk.sign(bytes.fromhex(eh)).hex()}
        self.entries.append(e)
        self.path.write_text(json.dumps(self.entries, indent=1))
        return e

    def segment(self, after_seq, upto_manifest_sha):
        seg = [e for e in self.entries if e["seq"] > after_seq]
        for i, e in enumerate(seg):
            if e["manifest_sha256"] == upto_manifest_sha:
                return seg[: i + 1]
        return seg


# ---------------------------------------------------------------- publisher side


def build_pack(dst, chunks, manifest_fields, ops, content_key, editor_keys=(), vectors=None, log=None,
               log_after_seq=0, tamper=None):
    """Writes a pack. `ops` lists ADD/SUPERSEDE/DELETE operations relative to the parent state.
    `tamper` is a test hook called with the directory after writing (to simulate attacks)."""
    import numpy as np
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "chunks.json").write_text(json.dumps(chunks))
    files = {"chunks.json": sha256b((dst / "chunks.json").read_bytes())}
    if vectors is not None:
        np.save(dst / "vectors.npy", np.asarray(vectors, np.float32))
        files["vectors.npy"] = sha256b((dst / "vectors.npy").read_bytes())
    man = dict(manifest_fields)
    man.update({"schema": SCHEMA, "chunks": {c["id"]: text_hash(c["text"]) for c in chunks}, "ops": ops,
                "files": files, "content_key_id": key_id(content_key), "vectors_shipped": vectors is not None})
    mb = canon(man)
    (dst / "manifest.json").write_bytes(mb)
    (dst / "manifest.sig").write_bytes(content_key.sign(mb))
    stmt = privileged_statement(man, sha256b(mb))
    eds = [{"key_id": key_id(k), "sig": k.sign(stmt).hex()} for k in editor_keys] if stmt else []
    (dst / "editors.json").write_text(json.dumps(eds))
    seg = []
    if log is not None:
        log.append(sha256b(mb))
        seg = log.segment(log_after_seq, sha256b(mb))
    (dst / "log.json").write_text(json.dumps(seg))
    if tamper:
        tamper(dst)
    return dst


def privileged_statement(man, manifest_sha):
    priv = [o for o in man["ops"] if o["op"] in PRIVILEGED]
    if not priv:
        return None
    return canon({"pack_id": man["pack_id"], "sequence": man["sequence"], "manifest_sha256": manifest_sha,
                  "privileged_ops": priv})


# ---------------------------------------------------------------- device side


def verify_update(pack, state, trust, device):
    """Returns (accepted: bool, reason: str, info: dict). Checks, in order: schema, content signature
    (declared key, known, not revoked), file hashes, pack id, sequence, expiry, compatibility, parent
    hash, chunk hashes, declared vs actual change set, editor threshold for privileged operations,
    transparency-log chain from the device's head."""
    pack = Path(pack)
    try:
        mb = (pack / "manifest.json").read_bytes()
        man = json.loads(mb)
        sig = (pack / "manifest.sig").read_bytes()
    except Exception as e:
        return False, f"unreadable pack: {e}", {}
    msha = sha256b(mb)
    if man.get("schema") != SCHEMA:
        return False, "unsupported schema", {}
    ck = trust["content_keys"].get(man.get("content_key_id"))
    if ck is None:
        return False, "content key not in trust store", {}
    if ck.get("revoked"):
        return False, "content key revoked", {}
    if not _verify(ck["public_pem"], sig, mb):
        return False, "content signature does not verify with declared key", {}
    for f, h in man["files"].items():
        if not (pack / f).exists() or sha256b((pack / f).read_bytes()) != h:
            return False, f"hash mismatch: {f}", {}
    if man.get("pack_id") != state["pack_id"]:
        return False, "wrong pack_id", {}
    if int(man.get("sequence", -1)) <= int(state["installed_sequence"]):
        return False, "rollback/replay: sequence not newer", {}
    exp = man.get("expires_utc")
    if exp and time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) > exp:
        return False, "expired", {}
    if man.get("embedding_model") != device["embedding_model"] or int(man.get("dim", -1)) != int(device["dim"]):
        return False, "incompatible embedding model/dimension", {}
    if man.get("parent_manifest_sha256") != state["installed_manifest_sha256"]:
        return False, "parent mismatch (fork or skipped version)", {}
    chunks = json.loads((pack / "chunks.json").read_text())
    if {c["id"]: text_hash(c["text"]) for c in chunks} != man["chunks"]:
        return False, "chunk text hashes differ from manifest", {}
    old, new = state["installed_chunks"], man["chunks"]
    modified = sorted(i for i in set(old) & set(new) if old[i] != new[i])
    if modified:
        return False, f"in-place modification of existing chunk(s): {modified[:3]}", {}
    added, removed = set(new) - set(old), set(old) - set(new)
    ops = man["ops"]
    d_add = {o["id"] for o in ops if o["op"] == "ADD"}
    d_sup = {o["id"]: o["replaces"] for o in ops if o["op"] == "SUPERSEDE"}
    d_del = {o["id"] for o in ops if o["op"] == "DELETE"}
    if any(o["op"] not in ("ADD",) + PRIVILEGED for o in ops):
        return False, "unknown operation", {}
    if added != d_add | set(d_sup) or removed != set(d_sup.values()) | d_del or (d_add & set(d_sup)):
        return False, "declared operations do not match the change set", {}
    priv = [o for o in ops if o["op"] in PRIVILEGED]
    n_ed = 0
    if priv:
        stmt = privileged_statement(man, msha)
        eds = json.loads((pack / "editors.json").read_text()) if (pack / "editors.json").exists() else []
        seen = set()
        for e in eds:
            k = trust["editor_keys"].get(e.get("key_id"))
            if k is None or k.get("revoked") or e["key_id"] in seen or e["key_id"] == man["content_key_id"]:
                continue
            if _verify(k["public_pem"], bytes.fromhex(e["sig"]), stmt):
                seen.add(e["key_id"])
        n_ed = len(seen)
        if n_ed < int(trust["editor_threshold"]):
            return False, f"privileged operations need {trust['editor_threshold']} editor signatures, got {n_ed}", {}
    seg = json.loads((pack / "log.json").read_text()) if (pack / "log.json").exists() else []
    prev, seq = state["log_head"]["entry_hash"], int(state["log_head"]["seq"])
    for e in seg:
        if e["prev"] != prev or e["seq"] != seq + 1 or Log.entry_hash(e["seq"], e["manifest_sha256"], e["prev"]) != e["entry_hash"]:
            return False, "transparency log does not extend the device's head", {}
        prev, seq = e["entry_hash"], e["seq"]
    if not seg or seg[-1]["manifest_sha256"] != msha:
        return False, "manifest not in transparency log", {}
    if not any(_verify(k["public_pem"], bytes.fromhex(seg[-1]["sig"]), bytes.fromhex(seg[-1]["entry_hash"]))
               for k in trust["log_keys"].values()):
        return False, "log head signature invalid", {}
    info = {"n_add": len(d_add), "n_supersede": len(d_sup), "n_delete": len(d_del), "editor_sigs": n_ed,
            "manifest_sha256": msha, "log_head": {"seq": seq, "entry_hash": prev}}
    return True, "accepted", info


def apply_update(state, pack):
    """Device state after accepting `pack` (call only after verify_update returned True)."""
    pack = Path(pack)
    mb = (pack / "manifest.json").read_bytes()
    man = json.loads(mb)
    seg = json.loads((pack / "log.json").read_text())
    return {"pack_id": state["pack_id"], "installed_sequence": man["sequence"],
            "installed_manifest_sha256": sha256b(mb), "installed_chunks": dict(man["chunks"]),
            "log_head": {"seq": seg[-1]["seq"], "entry_hash": seg[-1]["entry_hash"]}}

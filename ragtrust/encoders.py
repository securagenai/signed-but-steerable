"""Small-encoder inference with interchangeable back ends, on ONNX Runtime:

  tokenisation (numpy) -> full graph on ONNX Runtime -> task head (numpy)
  backends:  ort-fp32 / ort-int8  full graph on the CPU (ONNX Runtime)

Every model folder (artifacts/models/<name>/, produced by desktop/export_onnx.py) holds
model_card.json, tokenizer.json, host_embed.npz, head.npz and the ONNX graphs. Timing is accumulated per stage.
"""
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .paths import ARTIFACTS, model_dir, read_json

# ----------------------------------------------------------------------------------------------
# numpy pieces
# ----------------------------------------------------------------------------------------------


def layer_norm(x, g, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


class HostEmbedder:
    """BERT/RoBERTa input embeddings (word + position + token type, then LayerNorm) in FP32."""

    def __init__(self, npz_path, model_type, pad_id, eps):
        z = np.load(npz_path)
        self.word, self.pos, self.type = z["word"], z["pos"], z["type"]
        self.g, self.b = z["ln_g"], z["ln_b"]
        self.model_type, self.pad_id, self.eps = model_type, int(pad_id), float(eps)

    def positions(self, ids):
        if self.model_type == "roberta":  # HF create_position_ids_from_input_ids
            m = (ids != self.pad_id).astype(np.int64)
            return np.cumsum(m, axis=1) * m + self.pad_id
        return np.broadcast_to(np.arange(ids.shape[1]), ids.shape)

    def __call__(self, ids, type_ids, mode="full"):
        """mode 'full': word + position + type, then LayerNorm; 'no_ln': without the
        LayerNorm; 'word': word embeddings only. The non-default modes are kept for completeness."""
        if mode == "word":
            return self.word[ids].astype(np.float32)
        x = self.word[ids] + self.pos[self.positions(ids)] + self.type[type_ids]
        if mode == "no_ln":
            return x.astype(np.float32)
        return layer_norm(x, self.g, self.b, self.eps).astype(np.float32)


def mean_pool(h, mask, normalize=True):
    m = mask[..., None].astype(np.float32)
    v = (h * m).sum(1) / np.maximum(m.sum(1), 1e-9)
    if normalize:
        v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    return v.astype(np.float32)


class Head:
    """Task heads evaluated on the host in FP32, identical for every backend."""

    def __init__(self, npz_path, head_type):
        self.t = head_type
        self.z = dict(np.load(npz_path)) if head_type != "none" else {}

    def __call__(self, h, mask):
        z = self.z
        if self.t == "none":
            return h
        if self.t == "bert_pooler_classifier":
            pooled = np.tanh(h[:, 0] @ z["pool_w"].T + z["pool_b"])
            return pooled @ z["cls_w"].T + z["cls_b"]
        if self.t == "roberta_classifier":
            x = np.tanh(h[:, 0] @ z["dense_w"].T + z["dense_b"])
            return x @ z["out_w"].T + z["out_b"]
        if self.t == "qa_outputs":
            return h @ z["qa_w"].T + z["qa_b"]  # [N, L, 2]
        raise ValueError(self.t)


class TextTokenizer:
    """HF fast tokenizer (tokenizer.json) for single texts and text pairs, fixed or dynamic padding."""

    def __init__(self, path, max_len, pad_id, pad_token, fixed_length):
        from tokenizers import Tokenizer
        self.max_len, self.fixed = max_len, fixed_length
        self._t = {}
        for strategy in ("longest_first", "only_first", "only_second"):
            t = Tokenizer.from_file(str(path))
            t.enable_truncation(max_length=max_len, strategy=strategy)
            if fixed_length:
                t.enable_padding(length=max_len, pad_id=pad_id, pad_token=pad_token)
            else:
                t.enable_padding(pad_id=pad_id, pad_token=pad_token)
            self._t[strategy] = t
        self._count = Tokenizer.from_file(str(path))
        self._count.no_truncation()
        self._count.no_padding()

    def __call__(self, a, b=None, truncate="only_second"):
        if b is None:
            enc = self._t["longest_first"].encode_batch(list(a))
        else:
            pairs = list(zip(a, b))
            try:
                enc = self._t[truncate].encode_batch(pairs)
            except Exception:  # the kept side alone exceeds max_len: fall back to longest_first for those pairs
                enc = []
                for x, y in pairs:
                    try:
                        enc.append(self._t[truncate].encode(x, y))
                    except Exception:
                        enc.append(self._t["longest_first"].encode(x, y))
        ids = np.asarray([e.ids for e in enc], dtype=np.int64)
        typ = np.asarray([e.type_ids for e in enc], dtype=np.int64)
        msk = np.asarray([e.attention_mask for e in enc], dtype=np.int64)
        return ids, typ, msk

    def count(self, text):
        """Token count including special tokens (no truncation)."""
        return len(self._count.encode(text).ids)


# ----------------------------------------------------------------------------------------------
# backends: forward(ids, type_ids, mask) -> last hidden state [N, L, H] (float32)
# ----------------------------------------------------------------------------------------------


class OrtBackend:
    def __init__(self, onnx_path, threads=4):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.inputs = [i.name for i in self.sess.get_inputs()]

    def forward(self, ids, typ, msk, host_embedder=None):
        feeds = {"input_ids": ids, "token_type_ids": typ, "attention_mask": msk}
        return self.sess.run(None, {k: feeds[k] for k in self.inputs})[0]

    def close(self):
        pass


class Encoder:
    """Encoder(name, backend='ort-fp32'|'ort-int8', ...).

    embed(texts)            -> L2-normalised sentence embeddings [N, H]   (task 'embed')
    score_pairs(a, b)       -> logits [N] (rerank) or [N, C] (nli)
    Timing per stage is accumulated in self.timing (milliseconds) and self.counts.
    Batches are padded to the longest sequence by default (cpu_padding='longest').
    """

    def __init__(self, name, backend="ort-fp32", threads=4, batch=32, seq_len=None, cpu_padding="longest"):
        self.dir = model_dir(name)
        self.card = read_json(self.dir / "model_card.json")
        c = self.card
        self.name, self.backend_name, self.batch = name, backend, int(batch)
        self.L = int(seq_len or c["max_len"])
        fixed = cpu_padding == "max"
        self.tok = TextTokenizer(self.dir / c["files"]["tokenizer"], self.L, c["pad_id"], c["pad_token"], fixed)
        self.host = HostEmbedder(self.dir / c["files"]["host_embed"], c["model_type"], c["pad_id"], c["eps"])
        self.head = Head(self.dir / c["files"]["head"], c["head_type"])
        if backend == "ort-fp32":
            self.be = OrtBackend(self.dir / c["files"]["full_fp32"], threads)
        elif backend == "ort-int8":
            self.be = OrtBackend(self.dir / c["files"]["full_int8"], threads)
        else:
            raise ValueError(backend)
        self.reset_timing()

    def reset_timing(self):
        self.timing = defaultdict(float)
        self.counts = defaultdict(int)

    def _forward(self, ids, typ, msk):
        t0 = time.perf_counter()
        h = self.be.forward(ids, typ, msk)
        self.timing["backend_ms"] += (time.perf_counter() - t0) * 1e3
        return h

    def _timed_host(self, ids, typ, mode="full"):
        t0 = time.perf_counter()
        x = self.host(ids, typ, mode)
        dt = (time.perf_counter() - t0) * 1e3
        self.timing["host_embed_ms"] += dt
        self.timing["backend_ms"] -= dt  # reported separately
        return x

    def _batches(self, n):
        for i in range(0, n, self.batch):
            yield slice(i, min(n, i + self.batch))

    def embed(self, texts):
        texts = list(texts)
        out = []
        for s in self._batches(len(texts)):
            t0 = time.perf_counter()
            ids, typ, msk = self.tok(texts[s])
            self.timing["tokenize_ms"] += (time.perf_counter() - t0) * 1e3
            h = self._forward(ids, typ, msk)
            t0 = time.perf_counter()
            v = h if h.ndim == 2 else mean_pool(h, msk, normalize=False)
            v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
            self.timing["head_ms"] += (time.perf_counter() - t0) * 1e3
            out.append(v.astype(np.float32))
            self.counts["items"] += s.stop - s.start
            self.counts["calls"] += 1
        return np.concatenate(out) if out else np.zeros((0, self.card["hidden"]), np.float32)

    def score_pairs(self, a, b, truncate="only_second"):
        a, b = list(a), list(b)
        out = []
        for s in self._batches(len(a)):
            t0 = time.perf_counter()
            ids, typ, msk = self.tok(a[s], b[s], truncate=truncate)
            self.timing["tokenize_ms"] += (time.perf_counter() - t0) * 1e3
            h = self._forward(ids, typ, msk)
            t0 = time.perf_counter()
            logits = self.head(h, msk)
            self.timing["head_ms"] += (time.perf_counter() - t0) * 1e3
            out.append(np.asarray(logits, np.float32))
            self.counts["items"] += s.stop - s.start
            self.counts["calls"] += 1
        y = np.concatenate(out) if out else np.zeros((0, 1), np.float32)
        return y[:, 0] if y.ndim == 2 and y.shape[1] == 1 else y

    def entail_prob(self, premises, hypotheses):
        """NLI cross-encoders: softmax probability of the entailment class (premise first)."""
        logits = self.score_pairs(premises, hypotheses, truncate="only_first")
        e = np.exp(logits - logits.max(1, keepdims=True))
        p = e / e.sum(1, keepdims=True)
        return p[:, int(self.card["entail_index"])], p[:, int(self.card["contradiction_index"])]

    def close(self):
        self.be.close()

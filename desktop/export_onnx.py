"""Exports the small encoders for the Pi (desktop, PyTorch + ONNX):

  artifacts/models/<name>/
    full.onnx            input_ids, token_type_ids, attention_mask -> last_hidden_state (dynamic B, L)
    full_int8.onnx       ONNX Runtime dynamic INT8 quantisation of full.onnx
    host_embed.npz       embedding tables + LayerNorm
    head.npz             task head (pooler/classifier, RoBERTa head, or QA outputs)
    tokenizer.json       fast tokenizer used on the Pi (no transformers needed there)
    model_card.json      configuration, file names, parity and INT8-fidelity measurements

The graphs come from ragtrust.minibert (explicit eager attention), and each export is checked
against the original Hugging Face model; the script aborts if parity is not within tolerance.

Usage: python desktop/export_onnx.py --models all-minilm-l6-v2,ms-marco-minilm-l6,nli-minilm2-l6-h768,minilm-squad2 [--gelu erf]"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.paths import ARTIFACTS, DATA, hf_revision, questions, read_json, write_json  # noqa: E402

MODELS = {
    "all-minilm-l6-v2": dict(hf="sentence-transformers/all-MiniLM-L6-v2", task="embed", max_len=128),
    "ms-marco-minilm-l6": dict(hf="cross-encoder/ms-marco-MiniLM-L-6-v2", task="rerank", max_len=128),
    "nli-minilm2-l6-h768": dict(hf="cross-encoder/nli-MiniLM2-L6-H768", task="nli", max_len=128),
    "minilm-squad2": dict(hf="deepset/minilm-uncased-squad2", task="qa", max_len=384),
}
ap = argparse.ArgumentParser()
ap.add_argument("--models", default=",".join(MODELS))
ap.add_argument("--gelu", default="erf", choices=["erf", "tanh"])
ap.add_argument("--opset", type=int, default=17)
ap.add_argument("--tol", type=float, default=2e-3, help="max |diff| allowed between HF and rebuilt graphs")
a = ap.parse_args()

import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: E402
from transformers import (AutoModel, AutoModelForQuestionAnswering, AutoModelForSequenceClassification,  # noqa: E402
                          AutoTokenizer)

from ragtrust.encoders import Head, mean_pool  # noqa: E402
from ragtrust.minibert import Encoder, FullModel, check_parity, export_onnx  # noqa: E402

torch.set_grad_enabled(False)
sents = [s for row in read_json(DATA / "corpus_sentences.json")[:40] for s in row["sentences"]][:48]
qs = [q["question"] for q in questions()[:48]]


def ort_run(path, feeds):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    names = [i.name for i in s.get_inputs()]
    return s.run(None, {k: feeds[k] for k in names})[0]


def pin_ir(path, ir=8):
    """Keeps the ONNX IR version at the one of opset 17 (IR 8), so the graphs load in the ONNX Runtime
    build on the Pi even when the desktop has a newer onnx package."""
    m = onnx.load(str(path))
    if m.ir_version > ir:
        m.ir_version = ir
        onnx.save(m, str(path))


def maybe_simplify(path):
    try:
        import onnxsim
        m, ok = onnxsim.simplify(onnx.load(str(path)))
        if ok:
            onnx.save(m, str(path))
            return True
    except Exception:
        pass
    return False


for name in a.models.split(","):
    spec = MODELS[name]
    rev = hf_revision(spec["hf"])
    D = ARTIFACTS / "models" / name
    D.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    tok = AutoTokenizer.from_pretrained(spec["hf"], revision=rev)
    cls = {"embed": AutoModel, "rerank": AutoModelForSequenceClassification, "nli": AutoModelForSequenceClassification,
           "qa": AutoModelForQuestionAnswering}[spec["task"]]
    model = cls.from_pretrained(spec["hf"], revision=rev).eval()
    cfg = model.config
    base = model if spec["task"] == "embed" else model.base_model
    sd = {k: v for k, v in base.state_dict().items()}
    full_sd = model.state_dict()
    roberta = cfg.model_type in ("roberta", "xlm-roberta")
    L0 = spec["max_len"]

    # ---------------- head (host, FP32)
    head, head_type = {}, "none"
    if spec["task"] in ("rerank", "nli"):
        if roberta:
            head_type = "roberta_classifier"
            head = {"dense_w": full_sd["classifier.dense.weight"], "dense_b": full_sd["classifier.dense.bias"],
                    "out_w": full_sd["classifier.out_proj.weight"], "out_b": full_sd["classifier.out_proj.bias"]}
        else:
            head_type = "bert_pooler_classifier"
            head = {"pool_w": sd["pooler.dense.weight"], "pool_b": sd["pooler.dense.bias"],
                    "cls_w": full_sd["classifier.weight"], "cls_b": full_sd["classifier.bias"]}
    elif spec["task"] == "qa":
        head_type = "qa_outputs"
        head = {"qa_w": full_sd["qa_outputs.weight"], "qa_b": full_sd["qa_outputs.bias"]}
    np.savez(D / "head.npz", **{k: v.float().numpy() for k, v in head.items()})
    np.savez(D / "host_embed.npz", word=sd["embeddings.word_embeddings.weight"].float().numpy(),
             pos=sd["embeddings.position_embeddings.weight"].float().numpy(),
             type=sd["embeddings.token_type_embeddings.weight"].float().numpy(),
             ln_g=sd["embeddings.LayerNorm.weight"].float().numpy(), ln_b=sd["embeddings.LayerNorm.bias"].float().numpy())
    tok.backend_tokenizer.save(str(D / "tokenizer.json"))

    # ---------------- parity batch (real text, padded to the model length)
    if spec["task"] == "embed":
        batch = tok(sents[:16], truncation=True, max_length=L0, padding="max_length", return_tensors="pt")
    elif spec["task"] == "nli":
        batch = tok(sents[:16], qs[:16], truncation="only_first", max_length=L0, padding="max_length", return_tensors="pt")
    else:
        batch = tok(qs[:16], sents[:16], truncation="only_second", max_length=L0, padding="max_length", return_tensors="pt")
    if "token_type_ids" not in batch:
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
    full = FullModel(sd, cfg, act=a.gelu).eval()
    par = check_parity(base, full, None, batch, "vector")
    if par["max_abs_diff_hf_vs_rebuilt"] > a.tol:
        raise SystemExit(f"{name}: rebuilt graph differs from Hugging Face by {par['max_abs_diff_hf_vs_rebuilt']:.2e}")

    # ---------------- CPU graphs
    feeds = {k: batch[k].numpy().astype(np.int64) for k in ("input_ids", "token_type_ids", "attention_mask")}
    export_onnx(full, (batch["input_ids"], batch["token_type_ids"], batch["attention_mask"]), D / "full.onnx",
                ["input_ids", "token_type_ids", "attention_mask"], ["last_hidden_state"],
                {"input_ids": {0: "B", 1: "L"}, "token_type_ids": {0: "B", 1: "L"},
                 "attention_mask": {0: "B", 1: "L"}, "last_hidden_state": {0: "B", 1: "L"}}, a.opset)
    pin_ir(D / "full.onnx")
    onnx.checker.check_model(str(D / "full.onnx"))
    ref = full(batch["input_ids"], batch["token_type_ids"], batch["attention_mask"]).numpy()
    y32 = ort_run(D / "full.onnx", feeds)
    keep = feeds["attention_mask"].astype(bool)
    par["max_abs_diff_torch_vs_ort_fp32"] = float(np.abs(y32 - ref)[keep].max())
    quantize_dynamic(str(D / "full.onnx"), str(D / "full_int8.onnx"), weight_type=QuantType.QInt8)
    pin_ir(D / "full_int8.onnx")
    y8 = ort_run(D / "full_int8.onnx", feeds)
    hd = Head(D / "head.npz", head_type)
    msk = feeds["attention_mask"]
    if spec["task"] == "embed":
        v32, v8 = mean_pool(y32, msk), mean_pool(y8, msk)
        cos = (v32 * v8).sum(1)
        fid = {"int8_cosine_min": float(cos.min()), "int8_cosine_median": float(np.median(cos))}
    elif spec["task"] in ("rerank", "nli"):
        l32, l8 = hd(y32, msk), hd(y8, msk)
        fid = {"int8_logit_max_abs_diff": float(np.abs(l32 - l8).max()),
               "int8_argmax_agreement": float(np.mean(l32.argmax(1) == l8.argmax(1))) if l32.shape[1] > 1 else None,
               "int8_rank_corr": float(np.corrcoef(np.argsort(np.argsort(l32[:, 0])), np.argsort(np.argsort(l8[:, 0])))[0, 1])}
    else:
        fid = {"int8_max_abs_diff_hidden": float(np.abs(y32 - y8)[keep].max())}

    labels = {int(k): v.lower() for k, v in (cfg.id2label or {}).items()}
    card = {"name": name, "hf": spec["hf"], "revision": rev, "task": spec["task"], "model_type": cfg.model_type,
            "hidden": cfg.hidden_size, "layers": cfg.num_hidden_layers, "heads": cfg.num_attention_heads,
            "eps": cfg.layer_norm_eps, "max_len": L0, "pad_id": tok.pad_token_id, "pad_token": tok.pad_token,
            "head_type": head_type, "pooling": "mean" if spec["task"] == "embed" else None, "labels": labels,
            "entail_index": next((k for k, v in labels.items() if v.startswith("entail")), None),
            "contradiction_index": next((k for k, v in labels.items() if v.startswith("contradict")), None),
            "files": {"tokenizer": "tokenizer.json", "full_fp32": "full.onnx", "full_int8": "full_int8.onnx",
                      "host_embed": "host_embed.npz", "head": "head.npz"},
            "gelu": a.gelu, "opset": a.opset, "parity": par, "int8_fidelity": fid,
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                         "onnx": onnx.__version__, "onnxruntime": ort.__version__},
            "exported_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": time.time() - t_start}
    write_json(D / "model_card.json", card)
    print(f"{name}: parity {par}  int8 {fid}")

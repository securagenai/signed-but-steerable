"""Do floating-point pipelines give
bit-identical embeddings for the same text when only the batch size, the padding or the run changes?

Same model (all-MiniLM-L6-v2, pinned), same 256 corpus windows, sequence length 128. Each family is
compared with its own reference (batch 32, padding to the longest sequence, first run):
  gpu_fp32       PyTorch on the GPU, FP32 (default kernels), batch 1/8/32/256, max-length padding, 3 repeats
  gpu_fp16       same with FP16 autocast (common in production), batch 1 vs 32
  cpu_torch_fp32 PyTorch on the CPU, FP32, batch 1 vs 32
  cpu_ort_fp32   ONNX Runtime FP32 (our full.onnx), batch 1 vs 32, longest vs max-length padding
  cpu_ort_int8   ONNX Runtime INT8 (our full_int8.onnx), batch 1 vs 32
Writes results/e5/determinism_float.json.
Usage: python desktop/determinism_float.py [--n 256] [--device cuda]"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragtrust.paths import DATA, RESULTS, hf_revision, out_dir, read_json, write_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()
import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

HF, L = "sentence-transformers/all-MiniLM-L6-v2", 128
texts = [w["text"] for w in read_json(DATA / f"windows_minilm_L{L}.json")[: a.n]]
tok = AutoTokenizer.from_pretrained(HF, revision=hf_revision(HF))


def cmp(x, y):
    return {"bit_identical": bool(np.array_equal(x, y)),
            "rows_identical": float(np.mean([np.array_equal(u, v) for u, v in zip(x, y)])),
            "max_abs_diff": float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max()),
            "min_cosine": float((x * y).sum(1).min())}


def torch_embed(model, device, batch, padding="longest", fp16=False):
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=padding, truncation=True, max_length=L, return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=fp16):
            h = model(**enc).last_hidden_state.float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        out.append(torch.nn.functional.normalize(v, dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


res = {"n": len(texts), "model": HF, "seq_len": L, "torch": torch.__version__,
       "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
       "cudnn_deterministic": bool(torch.backends.cudnn.deterministic), "families": {}}
if a.device.startswith("cuda") and torch.cuda.is_available():
    res["gpu"] = torch.cuda.get_device_name(0)
    m = AutoModel.from_pretrained(HF, revision=hf_revision(HF)).to(a.device).eval()
    ref = torch_embed(m, a.device, 32)
    fam = {f"batch{b}_vs_32": cmp(ref, torch_embed(m, a.device, b)) for b in (1, 8, 256)}
    fam["maxlen_padding_vs_longest"] = cmp(ref, torch_embed(m, a.device, 32, padding="max_length"))
    reps = [torch_embed(m, a.device, 32) for _ in range(3)]
    fam["repeat_same_config"] = cmp(ref, reps[-1])
    res["families"]["gpu_fp32"] = fam
    r16 = torch_embed(m, a.device, 32, fp16=True)
    res["families"]["gpu_fp16"] = {"batch1_vs_32": cmp(r16, torch_embed(m, a.device, 1, fp16=True)),
                                   "repeat_same_config": cmp(r16, torch_embed(m, a.device, 32, fp16=True))}
    del m
    torch.cuda.empty_cache()
mc = AutoModel.from_pretrained(HF, revision=hf_revision(HF)).eval()
rc = torch_embed(mc, "cpu", 32)
res["families"]["cpu_torch_fp32"] = {"batch1_vs_32": cmp(rc, torch_embed(mc, "cpu", 1)),
                                     "repeat_same_config": cmp(rc, torch_embed(mc, "cpu", 32))}
from ragtrust.encoders import Encoder  # noqa: E402

for be in ("ort-fp32", "ort-int8"):
    r32 = Encoder("all-minilm-l6-v2", be, threads=4, batch=32, seq_len=L, cpu_padding="longest").embed(texts)
    r1 = Encoder("all-minilm-l6-v2", be, threads=4, batch=1, seq_len=L, cpu_padding="longest").embed(texts)
    rm = Encoder("all-minilm-l6-v2", be, threads=4, batch=32, seq_len=L, cpu_padding="max").embed(texts)
    res["families"][f"cpu_{be.replace('-', '_')}"] = {"batch1_vs_32": cmp(r32, r1), "maxlen_padding_vs_longest": cmp(r32, rm)}
write_json(out_dir("e5") / "determinism_float.json", res)
print(f"{'family':16s} {'check':28s} {'bit-identical':>13s} {'rows same':>9s} {'max |diff|':>11s}")
for f, checks in res["families"].items():
    for c, v in checks.items():
        print(f"{f:16s} {c:28s} {str(v['bit_identical']):>13s} {v['rows_identical']:9.3f} {v['max_abs_diff']:11.2e}")

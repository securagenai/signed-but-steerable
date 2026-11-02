"""Minimal BERT/RoBERTa encoder rebuilt from a Hugging Face state dict (desktop only, PyTorch), used
for ONNX export. Attention is written out explicitly (matmul, additive mask, softmax), so the exported
graph uses only plain operators and every export is the same computation.
`check_parity` compares against the original Hugging Face model before anything is exported."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _p(t):
    return nn.Parameter(t.detach().clone().float(), requires_grad=False)


class Linear(nn.Module):
    def __init__(self, sd, name):
        super().__init__()
        self.w, self.b = _p(sd[f"{name}.weight"]), _p(sd[f"{name}.bias"])

    def forward(self, x):
        return torch.matmul(x, self.w.t()) + self.b


class LayerNorm(nn.Module):
    def __init__(self, sd, name, eps):
        super().__init__()
        self.g, self.b, self.eps = _p(sd[f"{name}.weight"]), _p(sd[f"{name}.bias"]), float(eps)
        self.n = int(self.g.shape[0])  # static normalized_shape: a traced x.shape[-1] breaks ONNX export

    def forward(self, x):
        return F.layer_norm(x, (self.n,), self.g, self.b, self.eps)


def gelu(x, kind):
    if kind == "tanh":
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))
    return F.gelu(x)  # exact (erf) GELU, as in BERT/MiniLM/RoBERTa ("gelu")


class Block(nn.Module):
    def __init__(self, sd, p, hidden, heads, eps, act):
        super().__init__()
        self.q, self.k, self.v = (Linear(sd, f"{p}attention.self.{n}") for n in ("query", "key", "value"))
        self.o = Linear(sd, f"{p}attention.output.dense")
        self.ln1 = LayerNorm(sd, f"{p}attention.output.LayerNorm", eps)
        self.i = Linear(sd, f"{p}intermediate.dense")
        self.out = Linear(sd, f"{p}output.dense")
        self.ln2 = LayerNorm(sd, f"{p}output.LayerNorm", eps)
        self.heads, self.dh, self.act = heads, hidden // heads, act

    def forward(self, x, bias, B, L):
        def split(t):
            return t.reshape(B, L, self.heads, self.dh).permute(0, 2, 1, 3)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        s = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(self.dh))
        if bias is not None:
            s = s + bias
        c = torch.matmul(torch.softmax(s, dim=-1), v).permute(0, 2, 1, 3).reshape(B, L, self.heads * self.dh)
        x = self.ln1(self.o(c) + x)
        return self.ln2(self.out(gelu(self.i(x), self.act)) + x)


class Encoder(nn.Module):
    """hidden_states [B, L, H] (float) + mask -> last hidden state [B, L, H].
    mask_mode 'vector': mask [B, L] (1 = token, 0 = padding), broadcast over heads and query rows;
    mask_mode 'matrix': mask [B, L, L], broadcast over heads only; mask_mode 'none': no mask input at all (padding is attended; the host applies
    the mask in pooling). static=(B, L) freezes shapes for a static export."""

    def __init__(self, sd, cfg, act="erf", mask_mode="vector", static=None):
        super().__init__()
        H, nh, eps = cfg.hidden_size, cfg.num_attention_heads, cfg.layer_norm_eps
        self.blocks = nn.ModuleList(Block(sd, f"encoder.layer.{i}.", H, nh, eps, act)
                                    for i in range(cfg.num_hidden_layers))
        self.mask_mode, self.static = mask_mode, static

    def forward(self, h, mask=None):
        B, L = self.static if self.static else (h.shape[0], h.shape[1])
        if self.mask_mode == "none":  # single-input encoder: attends to every position
            bias = None
        elif self.mask_mode == "vector":
            bias = ((1.0 - mask) * -10000.0).reshape(B, 1, 1, L)
        else:
            bias = ((1.0 - mask) * -10000.0).reshape(B, 1, L, L)
        for blk in self.blocks:
            h = blk(h, bias, B, L)
        return h


class Embeddings(nn.Module):
    def __init__(self, sd, cfg):
        super().__init__()
        self.word = _p(sd["embeddings.word_embeddings.weight"])
        self.pos = _p(sd["embeddings.position_embeddings.weight"])
        self.typ = _p(sd["embeddings.token_type_embeddings.weight"])
        self.ln = LayerNorm(sd, "embeddings.LayerNorm", cfg.layer_norm_eps)
        self.roberta = cfg.model_type in ("roberta", "xlm-roberta")
        self.pad = int(cfg.pad_token_id or 0)

    def forward(self, ids, typ):
        if self.roberta:
            m = (ids != self.pad).long()
            pos = torch.cumsum(m, dim=1) * m + self.pad
        else:
            pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0).expand_as(ids)
        return self.ln(F.embedding(ids, self.word) + F.embedding(pos, self.pos) + F.embedding(typ, self.typ))


class FullModel(nn.Module):
    """input_ids, token_type_ids, attention_mask (int64) -> last hidden state; the CPU graph."""

    def __init__(self, sd, cfg, act="erf"):
        super().__init__()
        self.emb = Embeddings(sd, cfg)
        self.enc = Encoder(sd, cfg, act=act)

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.enc(self.emb(input_ids, token_type_ids), attention_mask.to(torch.float32))


@torch.no_grad()
def check_parity(hf_base, full, enc_static, batch, mask_mode):
    """Max |difference| on non-padding positions between (a) HF and the rebuilt full model and (b) the
    rebuilt full model and host-embeddings + static encoder (the split model)."""
    ids, typ, msk = batch["input_ids"], batch["token_type_ids"], batch["attention_mask"]
    ref = hf_base(input_ids=ids, token_type_ids=typ, attention_mask=msk).last_hidden_state
    out = full(ids, typ, msk)
    keep = msk.bool()
    d_full = (ref - out).abs()[keep].max().item()
    d_split = 0.0
    if enc_static is not None:
        h0 = full.emb(ids, typ)
        for i in range(ids.shape[0]):
            m = msk[i:i + 1].float()
            if mask_mode == "matrix":
                m = m[:, None, :].expand(1, m.shape[1], m.shape[1])
            y = enc_static(h0[i:i + 1], m)
            d_split = max(d_split, (y - out[i:i + 1]).abs()[keep[i:i + 1]].max().item())
    return {"max_abs_diff_hf_vs_rebuilt": d_full, "max_abs_diff_full_vs_split": d_split}


def export_onnx(module, args, path, input_names, output_names, dynamic_axes=None, opset=17):
    kw = dict(input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,
              opset_version=opset, do_constant_folding=True)
    try:
        torch.onnx.export(module, args, str(path), dynamo=False, **kw)  # TorchScript exporter: plain ops
    except TypeError:
        torch.onnx.export(module, args, str(path), **kw)

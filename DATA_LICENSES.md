# Data licences

- Code and our own results (per-query outputs, statistics, figures): MIT (see LICENSE).
- Corpus, windows and questions derived from SQuAD 2.0 (data/): CC BY-SA 4.0, as the original SQuAD 2.0
  (Rajpurkar et al., 2018). Redistribution keeps that licence.
- Results files that contain SQuAD text keep CC BY-SA 4.0 as well, for example results/e8/packs/*/chunks.json,
  results/e2/contexts/* and results/e1/sentences_top10.jsonl.
- BEIR corpora and MS MARCO are NOT redistributed; the scripts download them from their official sources under
  their own terms. Vectors and per-document outputs derived from them are not included.
- Model weights (BGE-M3, MiniLM models, Qwen2.5-3B) are NOT redistributed; the scripts
  download the pinned revisions (pins.json).

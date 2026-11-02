Created by the desktop scripts:
- `dev-v2.0.json` SQuAD 2.0 dev (downloaded by `desktop/setup_desktop.sh`)
- `prior/` question set, released per-query outputs and build statistics of the prior study
- `corpus.json`, `corpus_sentences.json`, `corpus_report.json` (`desktop/rebuild_corpus.py`)
- `retrieval_top20.json`, `vec_bge_m3.npz`, `windows_minilm_L128.json`, `vec_minilm_L128.npz` (`desktop/embed_retrieve.py`)
- `prior_pack_manifest.json` manifest of the prior study's knowledge pack, used to check that the rebuilt corpus is byte-identical

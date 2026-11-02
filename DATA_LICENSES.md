# Data and third-party licences

The code and the results we produced are released under the MIT licence (`LICENSE`), except the files listed
below, which contain third-party material and keep its terms.

## SQuAD 2.0 (CC BY-SA 4.0)
SQuAD 2.0 (Rajpurkar, Jia and Liang, 2018; https://rajpurkar.github.io/SQuAD-explorer/) is licensed under the Creative
Commons Attribution-ShareAlike 4.0 International licence (https://creativecommons.org/licenses/by-sa/4.0/).
- `data/dev-v2.0.json` is the unmodified SQuAD 2.0 development set.
- Modified material derived from it: the corpus, sentence, window and question files in `data/` and `data/prior/`, and
  the results files that contain SQuAD text: `results/e1/sentences_top10.jsonl`, `results/e2/contexts/`,
  `results/e2/gen_*.jsonl` (including the end-to-end E7 configurations), `results/e8/packs/*/chunks.json`,
  `results/e8/cases.json` and `results/e8/semantic_variants.json`.
These files are released under CC BY-SA 4.0. Changes: passages were selected and segmented into sentences and
windows, questions were sampled, and texts were combined with generated answers and scores.

## Generated answers (Qwen2.5-3B-Instruct)
The answers in `results/e2/gen_*.jsonl` were generated with Qwen2.5-3B-Instruct
(https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), which we used for research under the Qwen Research License
Agreement. The model itself is not included.

## BEIR datasets and MS MARCO (no text included)
For these datasets the repository contains only numbers derived from them and their identifiers, no text:
`results/e7/margins/*.npz` and `results/e7/beir_margins.json` (per-query retrieval margins) and
`results/e1/ood_msmarco_rows.jsonl` and `results/e1/ood_msmarco.json` (MS MARCO query IDs and scores).
Each dataset keeps its own terms: SciFact (claims CC BY 4.0, abstracts ODC-By 1.0), ArguAna (CC BY 4.0), NFCorpus
(free for academic use; other uses follow NutritionFacts.org's terms), FiQA-2018 (non-commercial use) and MS MARCO
(non-commercial research purposes). BEIR (Thakur et al., 2021) is the benchmark through which the four corpora were
obtained; it does not license them itself.

## Models
The models are downloaded by the scripts and not redistributed: BAAI/bge-m3 (MIT),
sentence-transformers/all-MiniLM-L6-v2 (Apache-2.0), cross-encoder/ms-marco-MiniLM-L-6-v2 (Apache-2.0),
cross-encoder/nli-MiniLM2-L6-H768 (Apache-2.0), cross-encoder/nli-deberta-v3-base (Apache-2.0),
deepset/minilm-uncased-squad2 (CC BY 4.0) and Qwen2.5-3B-Instruct (Qwen Research License).

## Other files
- `data/prior/` comes from the authors' earlier public release (https://github.com/securagenai/edge-rag-trust-mechanisms)
  and keeps the terms stated there; SQuAD-derived parts are CC BY-SA 4.0 as above.
- The keys and signatures in `results/e8/` are throw-away test material generated for the experiments (public keys
  only) and are released under the MIT licence.

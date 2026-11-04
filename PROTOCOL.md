# Protocol (fixed before data collection)

This file fixes the hypotheses, primary metrics, statistical procedures and decision rules before any
new measurement is taken. Changes after data collection go to the deviations log at the end, with date
and reason. The analysis scripts implement the decision rules below; where a rule is evaluated
automatically, the output field is named.

_Public version._ This release contains the protocol for the analyses reported in the paper. Hypotheses
H3, H4 and H5 and Gates B and C concern measurements that the paper does not evaluate as hypotheses
(E4 and E5 are reported only descriptively); they are omitted, and the few words of the remaining text
that refer to those measurements are omitted or replaced by [...]. The text is otherwise as fixed before
data collection; it was dated and kept with the code, not deposited in a public registry.

## Data and fixed settings

- Questions: the prior study's 300 SQuAD 2.0 dev questions (`questions_v2.json`), 100 per condition:
  `in_answerable`, `in_unanswerable` (in-topic, no answer in the corpus), `out_of_corpus` (7 held-out
  articles).
- Corpus: the prior study's 950 passages, rebuilt deterministically (`desktop/rebuild_corpus.py`; the
  SHA-256 must match the prior pack manifest when it is available).
- Retrieval: BGE-M3 at the pinned revision. New configurations use desktop exact top-20 retrieval; its
  agreement with the prior on-device top-5 is reported (`results/e1/retrieval_agreement.json`).
- Generation: Qwen2.5-3B-Instruct Q4_K_M, the pinned llama.cpp build, the prior system prompt,
  `max_tokens = 64`, temperature 0, seed 42, one slot.
- Scoring of answers: identical to the prior study. Lenient = not abstained and the normalised gold
  string is contained in the answer; strict = not abstained and token F1 >= 0.5.

## Hypotheses and primary metrics

**H1 (E1, primary).** At k = 5, the best [...] scorer (sentence-level bi-encoder cosine
`bi_max` or cross-encoder logit `ce_max`) separates answerable from in-topic unanswerable questions with
AUROC >= 0.80, and is better than the retrieval top-1 score (prior: 0.68).
- Test: paired DeLong against `top1`, Holm-adjusted over the three candidates (`bi_max`, `ce_max`,
  `qa_margin`). The extractive-QA no-answer margin `qa_margin` (12 layers, CPU only) is a reference
  ceiling, not a [...] candidate.
- Rule (`results/e1/summary.json -> H1_decision.GO`): point AUROC >= 0.80 and Holm-adjusted p < 0.05
  for the best candidate.
- Secondary: the same at k = 3 and k = 10; AUROC against out-of-corpus and against all should-abstain
  questions; the split-conformal gate operating point (alpha = 0.1, prior procedure) with retained and
  abstained rates on held-out halves.

**H2 (E2).** Sentence pruning (`k5_ce3`, `k10_ce3`) is non-inferior in lenient answerable accuracy to the
prior main configuration (`q3_k3`, 0.75), with a margin of 5 percentage points, and reduces median
end-to-end latency by at least 40 %.
- Test: paired bootstrap over questions (B = 4,000).
- Rule (`results/e2/summary.json -> H2_decision`): the lower 95 % bound of (pruned - prior k3) is above
  -0.05, and the upper 95 % bound of the median-latency ratio is below 0.60.
- Secondary: pruned vs prior `q3_k5` (0.52); abstention on unanswerables; the bi-encoder selector
  `k5_bi3`; the pre-generation gate policy M+V+N with energy per query, where gated questions cost only
  the scorer.

**H6 (E7).** For packs that ship vectors, the following are expected:

| Defence | Expected attack success |
|---|---|
| Vector check with [...] honest tolerance (tau from E5) | A1 hijack >= 50 % of targets in at least one embedding space |
| Vector check against an FP32 recomputation (tau = 0.999) | <= 5 % |
| Full on-device re-indexing | 0 by construction (its cost comes from E5) |
| Random FP32 spot-checks | Detection follows the hypergeometric formula |

End-to-end, the payload attack raises abstention on targeted answerable questions above the prior
`q3_k3` rate on the same questions.

**H7 (E8).** The following outcomes are expected:
- **Deterministic cases:** every authorisation and integrity attack is rejected, and every legitimate
  update is installed (`all_as_expected`).
- **Semantic screen:** AUROC for silent overrides vs benign additions is reported, with TPR and FPR at
  theta = 0.5. The hedged variant measures robustness to rephrasing.
- **Unanchored new-fact poison:** expected to pass (a documented limitation). Its flag rate is
  reported.
- **Desynchronised vectors:** caught by the vector check.

## Statistical procedures

- AUROC: Mann-Whitney with ties (identical to the prior study). 95 % CIs from 2,000 bootstrap resamples,
  positives and negatives resampled separately. Paired comparisons: DeLong (Sun & Xu fast algorithm).
- Proportions: Wilson intervals. Paired differences between systems: bootstrap over questions,
  B = 4,000, seed 1.
- Thresholds: split-conformal on answerable scores, alpha = 0.1, B random halves, evaluated on the
  held-out half (the prior procedure).
- Latency: medians and p95 over questions. Inter-token latency p50/p99 over all tokens of all repeats.
  Warm-up calls are excluded.
- Seeds are fixed in every script. All per-query and per-call outputs are kept, so every summary can be
  regenerated (`analysis/reproduce_all.sh`).

## Decision gates

- **Gate A (after E1, day 3).** If H1 fails, the abstention claim is dropped and reported as a negative
  result. Pruning (E2), integrity (E7, E8) and [...] continue.

## Threats to validity

- **Narrow setup.** One board, one LLM, one corpus domain, and 300 questions. Intervals are reported
  everywhere; claims are limited to this setting.
- **In-distribution reference.** `deepset/minilm-uncased-squad2` was trained on the SQuAD 2.0 train
  split and evaluated here on dev. It is a reference ceiling, not a candidate.
- **Synthetic update cases.** E8 cases come from entity substitution (as in the prior poisoning
  study). Legitimate corrections and silent overrides carry identical text, on purpose, so the
  contribution of authorisation is isolated from that of the semantic screen.
- **Retrieval differences.** Desktop exact search and Pi HNSW retrieval can differ slightly. Agreement
  is reported, and the reproducibility run checks the pipeline end to end.

## Deviations log

| Date | Change | Reason |
|---|---|---|
| 2026-10-31 | Exploratory, not pre-registered: the extractive-QA no-answer margin (deepset/minilm-uncased-squad2, 12 layers, CPU) is evaluated as a pre-generation gate on the Pi (cost, INT8 fidelity, M+V+N policy in E2), plus an out-of-distribution check because the model was trained on SQuAD 2.0. Reported separately from the confirmatory results. | Gate A showed relevance scorers cannot separate in-topic unanswerables (0.63-0.69) while qa_margin reaches 0.857. |
| 2026-10-31 | Exploratory, not pre-registered: E2 configuration k5_ce3w (top-3 cross-encoder sentences, each with its neighbouring sentences, prefixed by the article title). Preliminary k5_ce3 (241/300): lenient accuracy 0.46 vs 0.75 for the prior k3, although the answer string is present in 96% of the pruned contexts (0.98 for full passages). | Tests whether the loss comes from missing context around the selected sentences (coreference, topic) rather than from sentence selection. |
| 2026-10-31 | Exploratory: margin law for the E7 attack (analysis/margin_law.py, margin_law_squad.py, desktop/beir_margins.py): success at tolerance tau = empirical CDF of angular retrieval margins at arccos(tau); validated against the E7 simulation (max diff 0.0), on 4,690 SQuAD questions and four BEIR corpora (SciFact, NFCorpus, ArguAna, FiQA) with BGE-M3 and MiniLM; transfer attack using other questions about the same passage. | Generalises E7 beyond the 100-question study set. |
| 2026-10-31 | Exploratory: storage formats of vector databases (analysis/vecdb_quant.py: FP16, SQ8, SQ4, PQ, binary): tolerance = largest honest reconstruction angle; hijack rate from the margin law. | Integrity audits against stored compressed vectors. |
| 2026-11-01 | H2 rejected as pre-registered (k5_ce3, k10_ce3 not non-inferior). Exploratory diagnostics: answer string present in 96% of pruned contexts; k5_ce3w (neighbouring sentences + title) and k3_ce3/k3_ce3w (top-3 passages only) added; error decomposition shows the loss is mostly added abstention (wrong-answer rate 0.12-0.14 vs 0.11 [superseded by the 2026-11-02 E2 decomposition entry]). | Explains the H2 failure. |
| 2026-11-01 | Latency caveat: llama-server reuses cached prompts across configurations; rows with cache hits are identified from prompt_ms x prompt_per_second and latency is reported without cache hits (or corrected with the measured prefill rate). Answers and accuracy are unaffected (deterministic decoding). | Configurations that repeat earlier contexts (E7 variants, online selectors) were faster than possible without the cache. |
| 2026-11-01 | E8 semantic screen as pre-registered: AUROC 0.18 (fails). Cause: most-similar sentence pairs are unchanged copies, and small NLI models judge near-identical edited sentences as entailed. Exploratory offline re-analysis (analysis/e8_semantic_variants.py): V1 excluding verbatim sentences 0.39; V2 near-duplicate edit signal 0.998 (hedged 0.995). V2 was designed after seeing the result and is evaded by full paraphrases. | Reports both the pre-registered failure and the post-hoc alternative. |
| 2026-11-01 | E8 runs repeated after fixing a tokenizer crash (pair truncation when the hypothesis alone exceeds the NLI max length; fallback to longest_first). No change to cases, thresholds or models. | Bug fix. |
| 2026-11-01 | Exploratory out-of-distribution check of E1 (desktop/e1_ood_msmarco.py): MS MARCO v2.1 validation, 500 answerable + 500 'No Answer Present' queries, same models and settings. AUROC: bi_max 0.757 [0.726, 0.787], ce_max 0.829 [0.803, 0.853], qa_margin 0.700 [0.668, 0.731]. The E1 ranking reverses: each scorer is best on its training distribution; SQuAD 2.0 unanswerables are adversarially similar to answerable ones, MS MARCO ones are naturally less relevant. Conclusion: no pre-generation scorer transfers; an abstention gate must be calibrated in domain. | Tests whether the E1 qa_margin result is in-distribution only. |
| 2026-11-01 | Exploratory: k3_ce3 and k3_ce3w (pruning restricted to the top-3 passages). k3_ce3 equals k5_ce3 (correct 0.44, abstained 0.43, wrong 0.13; 70/100 identical contexts): distractor passages 4-5 do not explain the H2 failure. Context windows (k*_ce3w) lower abstention but double wrong answers (0.24). | Closes the H2 diagnosis. |
| 2026-11-01 | Exploratory: exact verification of quantisation CODES across platforms (analysis/vecdb_code_verify.py; 1,745 MiniLM windows, x86 CPU / ARM CPU / GPU FP32, codebooks trained on x86). Identical codes: PQ 100% and binary 100% for all pairs; SQ4 99.94-100%; SQ8 99.08-99.26%; SQ with a 1e-6 guard band at bin edges 100%. | Tests a defence for compressed vector stores. |
| 2026-11-01 | Exploratory: code-level verification at scale (desktop/vecdb_code_verify_scale.py; BEIR FiQA, MiniLM on all 57,638 documents, BGE-M3 on a seeded 5,000; GPU vs CPU FP32, codebooks trained on GPU). Exact identical codes: MiniLM PQ 99.995%, binary 99.991%, SQ4 99.93%, SQ8 98.98%; BGE-M3 PQ 99.96%, binary 99.94%, SQ4 99.54%, SQ8 91.78%. With a guard band of 1e-6 at decision boundaries: 100% except BGE-M3 SQ8 (99.98%); cross-platform float noise 4.3e-7 (MiniLM) and 1.0e-5 (BGE-M3), so the guard band must be sized to the measured platform noise. | Scale and generality of the code-level defence. |
| 2026-11-01 | Exploratory: residual attack budget of the guard band (analysis/guard_budget.py; SQuAD, ~3,000-3,650 questions with gold ranked first; publisher picks, for every ambiguous entry, the admissible code that best raises a carrier and lowers the gold). Hijack rate with eps 1e-6: BGE-M3 SQ8 0.05%, SQ4 0.03%, PQ 0.09%; MiniLM 0%. With eps 1e-5: BGE-M3 SQ8 0.41%, SQ4 0.22%, PQ 0.31%; MiniLM SQ8 0.22%, SQ4 0.13%, PQ 0.07% (tolerance-based checks: SQ8 2-15%, SQ4 55-89%, PQ ~100%). Design rule adopted: verify and canonicalise, i.e. the verifier replaces ambiguous entries with its own codes, which removes the publisher's choice. | Quantifies the residual budget of the proposed defence. |
| 2026-11-01 | Exploratory: real vector database (desktop/faiss_integrity.py; FAISS IndexFlatIP, IndexScalarQuantizer SQ8, IndexPQ d/8x8, IndexBinaryFlat over all 57,638 BEIR FiQA documents, MiniLM; publisher GPU, verifier CPU; 230 targets with gold at rank 1). Tolerance = largest honest reconstruction angle; margin-law hijack: Flat 0.0%, SQ8 8.3%, PQ 100%, binary 100%. Executed attack on IndexPQ (60 targets, greedy codebook choice, lowest-gain changes reverted until the decoded vector passes the verifier's global tolerance check): 100% of tampered vectors pass the check, rank 1 hijacked 25% [16, 37]; code-level verification with canonicalisation blocks 100% [94, 100]. A first version checked the tolerance per sub-vector only; it was corrected before use. | Demonstrates the gap in tolerance-based audits, and the defence, on FAISS indexes (FAISS behaves as designed; the audit is ours). |
| 2026-11-01 | Methods change after external review: single reference verifier ragtrust/codeverify.py for all code-level measurements. Ambiguity is decided ONLY from the verifier's recomputation; a neighbouring code (SQ) or second-nearest centroid (PQ) is accepted only at entries the verifier marks ambiguous; accepted vectors are canonicalised to the verifier's codes; constant dimensions are never ambiguous; eps_SQ/eps_bin (value units) and eps_PQ (squared-distance units) are defined separately. Earlier scale results used publisher-side ambiguity and are superseded. Apple M2 (Metal GPU <-> ARM CPU, both directions): MiniLM 57,638 docs exact 99.12% (SQ8), 99.95% (SQ4), 99.997% (PQ), 99.991% (binary), all accepted from 3e-7 (SQ8) and 1e-7 (others); BGE-M3 5,000 docs SQ8 91.0% -> 100% at 3e-6, PQ 100% without guard. The first M2 run of the reference verifier gave flat SQ curves and contradictory counters; they disappeared after handling constant dimensions and upgrading NumPy 2.2.6 (source build on Python 3.14) to 2.5.4; a NumPy sanity check passed on both versions, so the root cause was not isolated; only the corrected run is reported. | Aligns the measured verifier with the described one. |
| 2026-11-01 | Re-measurement with the reference verifier on the desktop (desktop/code_verify_ref.py; both directions). FiQA MiniLM 57,638 docs (max dev 4.3e-7): exact SQ8 98.99-99.03%, smallest sufficient eps 3e-7; SQ4 1e-7-3e-7; PQ 1e-7; binary 1e-7-3e-7. BGE-M3 5,000 docs (max dev 1.0e-5): SQ8 91.3-91.5% -> 3e-6; SQ4 1e-6-3e-6; PQ none; binary 1e-7. x86<->ARM on the Raspberry Pi 5 (max dev 1.9e-7): SQ8 99.26-99.43% -> 1e-7; PQ and binary 100% without guard. Same smallest sufficient widths as on the M2. The M2 model revisions (MiniLM 1110a243, BGE-M3 5617a9f6) equal the pinned ones. | Cross-platform calibration of the guard band. |
| 2026-11-01 | Residual budget without canonicalisation recomputed with verifier-side ambiguity (analysis/guard_budget.py, SQuAD): MiniLM 0% at the calibrated widths for SQ8, SQ4 and PQ; BGE-M3 SQ8 0% at 3e-7, 0.05% at 1e-6, 0.16% at 3e-6, 0.41% at 1e-5; SQ4 at most 0.22%; PQ 0% up to 1e-6, 0.14% at 1e-5. With canonicalisation the budget is zero by construction. Supersedes the earlier guard-budget entry. | Residual exposure of the defence at calibrated widths. |
| 2026-11-01 | Numerical precision check of the margins after external review (analysis/margin_precision_check.py): all margins recomputed in float64 with the stable formula 2*asin(||q-d||/2) from the stored float32 document vectors and re-embedded queries. Stored float32 values reproduced exactly; max difference < 1e-4 deg; every reported rate unchanged (fp32_noise, 0.99999, 0.999) for all 10 corpus-model pairs. Exact-zero margins are duplicate documents: ArguAna 18/420 (BGE-M3) and 14/373 (MiniLM), NFCorpus 2/141 and 2/136; of the ArguAna queries exposed at the fp32-noise tolerance (23 and 17), 18 and 14 are such duplicates; excluding them, exposure at that tolerance is at most 1.2%. SQuAD and FiQA exposures at that tolerance contain no exact duplicates. | Addresses a numerical-precision concern and separates displacement by a duplicate from harmful hijacking. |
| 2026-11-01 | Extended precision check and unit reconciliation after external review (analysis/margin_precision_check.py, analysis/squad_unit_reconcile.py). (a) Table 2 uses BGE-M3 on the earlier study's passage index (n=3,648) and MiniLM on 128-token windows (n=3,301); Fig. 1 and the BEIR comparison encode all corpora at passage level, truncated at 512 tokens (BGE-M3) and 256 (MiniLM), n=3,646 and 3,148. The two BGE-M3 encodings are identical (cosine 1.00000000) for the 940 passages of at most 512 tokens and differ only for the 10 longer ones (cosine 0.950-0.997); the two questions selected only by the Table 2 analysis both concern passage 228 (701 tokens). MiniLM differs by unit (19.7% passage-level vs 16.8% window-level at tau=0.999). Both units reproduced in float64 to within 4e-5 deg, every rate unchanged. (b) Text identity checked for ALL queries exposed at the fp32-noise tolerance (normalisation: whitespace collapsed, lower-cased): identical competitor text in ArguAna 18/23 (BGE-M3) and 14/17 (MiniLM), NFCorpus 2/2 and 2/3, none in SQuAD (0/18, 0/7), FiQA (0/2) or SciFact; all identical-text cases have zero margin. Excluding them: ArguAna 5/420 and 3/373, NFCorpus 0/141 and 1/136. | Reconciles the SQuAD counts and documents the duplicates. |
| 2026-11-02 | Checks of the E7 end-to-end experiment after external review (analysis/e7_e2e_checks.py): every configuration (no check, tau = 0.999, CPU-INT8 reference threshold 0.904) uses the same 100 answerable questions in contexts and generations; the shipped payload vector never exceeds the angular budget; payload in the top-3 context for 100%, 2% and 94% of questions. Exposure at the CPU-INT8 threshold for every corpus and the SQuAD table recorded in results/e7/cpu_int8_exposure.json (analysis/cpu_int8_exposure.py). | Separates delivery of the payload from its effect on the answer. |
| 2026-11-02 | E2 error decomposition recomputed from the generation files with one scoring function for every configuration (analysis/e2_decomposition.py, results/e2/decomposition.json): prior k3 0.75 correct / 0.13 abstained / 0.12 wrong on answerable questions, k5_ce3 0.44 / 0.43 / 0.13, k5_ce3w 0.45 / 0.30 / 0.25; explicit abstention on in-topic unanswerable 0.66 (prior k3) vs 0.80 (k5_ce3), out of corpus 0.99 for both. An earlier summary had reported 0.14 / 0.11 for the prior configuration and the mean over unanswerable and out-of-corpus questions (0.82 vs 0.90) as abstention on unanswerable questions; the paper now uses these values. | Corrects the reported E2 decomposition. |

_12 of the 33 logged deviations concern experiments that are not reported in the paper and are omitted from this release._

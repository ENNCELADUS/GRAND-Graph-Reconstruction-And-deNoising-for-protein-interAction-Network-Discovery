# Replace TCCIG with graph-prior retrieval

Accepted. TCCIG is replaced in place rather than introduced as a parallel model because the old dense all-pairs graph generator preserved the public PRING wiring but optimized the wrong localization problem. Keeping `model_config.model: tccig`, `tccig_train`, and the existing graph-assembly output path avoids duplicating pipeline stages while making retrieval, reranking, and train-only graph-prior distillation the canonical implementation.

## Considered Options

- Add a new `gpr_ppi` model and stage, leaving old TCCIG as the default.
- Replace TCCIG in place and keep old `m_hat`/dense-decoder behavior only as diagnostics.

## Consequences

Future TCCIG changes should treat `all_test_ppi.txt` reconstruction, retrieval recall, candidate AUPRC, and hybrid graph assembly as the primary contract. Learned `m_hat` and fixed `0.5` threshold outputs are diagnostics, not the official decision rule.

The first implementation boundary is exact retrieval-gated inference: graph assembly encodes the PRING test proteins once, generates exact Top-R retrieval candidates, reranks only retrieved candidates that intersect `all_test_ppi.txt`, and emits background `0` labels for non-retrieved records. ANN retrieval backends and training-side PPR/random-walk distillation remain follow-up work behind this contract.

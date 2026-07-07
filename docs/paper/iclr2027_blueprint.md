# ICLR 2027 Paper Blueprint

**Working titles (pick later):**
- *Interactomes by Construction: Conditional Local-Graph Generation for Inductive PPI Reconstruction*
- *From Pairs to Interactomes: Context-Conditioned Graph Generation for Protein Interaction Networks*
- *Growing Interactomes: Why PPI Prediction Should Be Local-Graph Generation, Not Independent Edge Scoring*

**Status:** research plan / blueprint. Altitude = research problem + methodology (deliberately not implementation detail). RQs confirmed 2026-07-06; method route = generative local-network growth; contribution = a new method that must win; eval = human intra-species (rigorous) + PRING function tasks.

---

## 0. Configuration

| Field | Value |
|---|---|
| Venue | ICLR 2027 (submission ~Sep 2026) |
| Paper type | Conference paper — empirical ML method (ML for computational biology / graph representation learning) |
| Length | ~9 pages main + unlimited appendix |
| Structure pattern | Conference / ML-IMRaD hybrid |
| Citation style | ICLR (author-year, natbib) |
| Benchmark | PRING (NeurIPS 2025 D&B) — human intra-species primary; function-oriented tasks secondary |
| Backbone | Frozen ESM-3 (`esm3_sm_open_v1`, 1536-d) as intrinsic node features |

---

## 1. Thesis

> **[INSIGHT: thesis_statement]**
> PPI interactome reconstruction should be modeled as **conditional local-graph generation from intrinsic protein features**, not independent pairwise edge scoring. Starting from two seed proteins and a *potential-neighbor set*, a model grows a topologically coherent local network conditioned on node features, so every edge decision is made **in context** rather than in isolation. This produces realistic interactomes that pairwise-independent scoring cannot — whether the training objective is standard BCE or augmented with topology losses — because the fundamental limitation is the pairwise-independent structure of the model, not the loss function alone. The approach stays strictly inductive and leakage-free.

- **Thesis type:** constructive/methodological (new formulation + method), supported by a limitation argument against the incumbent formulation.
- **Scope & boundary:** inductive reconstruction of an interactome over *unseen* proteins from sequence-intrinsic features; evaluated on PRING (human intra-species topology + function tasks). Out of scope: transductive graph-completion settings that require an existing network as input.
- **What the reader should believe after reading:** topological realism is a *joint, generative* property of an interactome; you cannot reach it by scoring edges independently and thresholding, no matter how strong the pairwise model.

---

## 2. The Research Problem

**Two community formulations, both flawed.**

1. **Graph completion / transductive link prediction.** Treats the known interactome as input and predicts held-out edges on the *same* graph. Strong topology signal, but (a) requires a pre-existing network, so it cannot serve genuinely unseen proteins, and (b) leaks — test proteins/edges are visible to the model through the input graph. This violates the inductive, unseen-protein setting PRING is built to enforce.
2. **Protein-centric / intrinsic pairwise scoring.** Encodes each protein from sequence (PLM) and scores each candidate edge independently. Fully inductive, but topology-blind: it optimizes per-edge correctness, and independent scoring compounds into an unrealistic assembled graph.

**The gap.** A PPI signal is *both* local (a specific pairwise interaction) *and* structural (each edge lives inside a neighborhood with characteristic density, degree, and clustering). Neither incumbent formulation captures both under a clean inductive setting. Graph completion has the structure but leaks; pairwise scoring is clean but structure-blind.

**The empirical hook (already in hand).** Our frozen pairwise scorer (v3.1) beats *published* PRING SOTA on pairwise AUPR (**0.792** vs PPITrans 0.769, PLM-interact-650M 0.707, TUnA 0.720), yet the interactome it assembles is badly unrealistic: graph-similarity **0.337**, relative density **3.13** (target ≈ 1). The root cause is a **training objective mismatch**: BCE over independent pairs gives the model no incentive to produce realistic topology, so it over-predicts density globally without any graph-structural constraint.

The natural fix is therefore **topology-aware training** (`src/topology/` finetune stage): augment the training objective with differentiable topology losses (graph similarity, relative density, degree-distribution MMD, clustering-coefficient MMD) computed over sampled subgraphs. The classifier architecture is unchanged — only the training signal changes. Results for this approach are pending and will constitute a key internal baseline.

A separate, weaker "fix" — **post-hoc denoising** of the assembled pairwise graph (our S2GAE refiner, Route A) — treats the output as a broken artifact to clean up rather than correcting what the model is optimized for. It does not work: at threshold 0.5 it is a no-op (density stays 3.16); at a calibrated threshold it over-corrects into an over-sparse graph (graph_sim 0.190, density 0.16). The gap is not a knob you can turn. Both fixes — topology-aware training and post-hoc denoising — remain strictly pairwise; neither introduces the joint local-graph context that motivates our method.

**Why now / timeliness.** PRING (2025) is the first benchmark to score PPI models as *graph reconstructors* under a strict inductive split, and it exposes exactly this failure across the field. PLMs give strong intrinsic node features for the first time. The tools to attack the problem generatively (feature-conditioned graph generation) now exist. The question is open and freshly measurable.

---

## 3. Research Questions → Contributions

| RQ | Question | Contribution | Current strength |
|---|---|---|---|
| **RQ1** | What formally makes an interactome representation "structured and realistic," and why can't a pairwise-accurate, independent-edge model achieve it — even with topology-aware training? | **C1 — Formal motivating characterization.** Define topological realism (joint density / degree / clustering / modularity) and argue that independent per-edge scoring compounds error into O(N²) false edges regardless of the training objective, because no per-pair loss can constrain joint graph statistics. | **Strong** — evidence already owned (AUPR 0.792 vs density 3.13; exp02 calibration cliff; topology finetune baseline pending). |
| **RQ2** | Can edge prediction be recast as conditional local-graph generation from two seed nodes + a potential-neighbor set? | **C2 — The method.** A feature-conditioned generator that grows a local network around a seed pair and reads edges off generated topology. | **Pending** — method unbuilt; core novelty. |
| **RQ3** | Does generative, context-conditioned prediction beat (a) independent scoring and (b) post-hoc denoising on PRING topology, under a strict leakage-free inductive setting? | **C3 — The "wins" result + necessity of context.** Beat both baselines on graph_sim / density while preserving pairwise quality; ablate context to show it is necessary. | **Pending / high-risk** — the load-bearing empirical claim. |
| **RQ4** | Can one model serve single-pair queries *and* full graph reconstruction with a self-consistent operating point (no hand-tuned threshold)? | **C4 — Dual operation.** Generation emits a graph, not a score to threshold, dissolving the balanced-1:1 vs all-vs-all (~1.6% density) mismatch that sank the denoiser. | **Pending.** |
| **RQ5** | Do generated interactomes improve *function-oriented* outcomes, not just topology distributions? | **C5 — Downstream biological utility.** Gains on PRING GO-enrichment / complex-pathway / essential-protein. | **Pending.** |

C2 + C3 are load-bearing; C1 motivates; C4 and C5 broaden the claim. Be honest in the paper: C1 is a demonstrated limitation, C3 is the bet.

---

## 4. Methodology (high level)

The method reframes reconstruction from "score every candidate edge, then threshold" to "generate the local network, then the edges are whatever was generated." Five conceptual stages:

1. **Intrinsic node features.** Frozen ESM-3 embeddings per protein. The target topology is never an input — this is what keeps the setting inductive and distinguishes us from graph completion.
2. **Potential-neighbor set.** For a seed protein (or seed pair), define a tractable candidate set of possible neighbors from feature space (retrieval / coarse prefilter), constructed *without* the test graph. This bounds the O(N²) candidate universe (2M+ pairs over 2,018 test proteins) into local generation problems and operationalizes "context."
3. **Conditional local-graph generation.** Starting from the seed(s), generate the local subgraph structure conditioned on node features and the partially-generated context. Because edges are emitted jointly within a generated neighborhood, topological priors (sparsity, degree profile, clustering) are respected *by construction* rather than imposed as a post-hoc loss.
4. **Assembly + self-consistent operating point.** Compose local generations into the full interactome. The generative process yields its own density — no per-dataset threshold tuning — which is the mechanism behind C4 and the intended cure for the denoiser's calibration cliff.
5. **Training objective.** Likelihood of observed local graph structure (generative), optionally regularized toward matching global topology distributions. Trained strictly on train-node subgraphs; leakage-free by construction.

**Relationship to prior in-repo work.** This *subsumes* the current S2GAE denoiser (Route A) as a special case (denoising = one-step refinement of a fixed candidate graph) and pivots to generation (Route B). Keep the frozen ESM-3 + pairwise-scorer machinery as feature/candidate infrastructure; the new object is the generator.

**Design questions deferred to prototyping (not decided here):** the generative family (autoregressive edge/node addition vs. one-shot latent subgraph decoder vs. diffusion over local adjacency); whether the frozen pairwise scorer seeds the generator as a prior or is discarded; how the potential-neighbor set is retrieved. These are §9 decisions, resolved by a minimal prototype, not by argument.

---

## 5. Evaluation Plan

**Primary — topology reconstruction (PRING human intra-species).** graph_sim, relative density (→1), degree-MMD, clustering-MMD, spectral-MMD, over BFS/DFS/RandomWalk splits. **Reconcile our MMD with the official PRING `eval.py`** (we currently report a normalized ratio; PRING reports raw) so numbers are directly comparable to their Table 2.

**Secondary — function-oriented (PRING).** GO-enrichment, complex-pathway, essential-protein. This is the added scope you chose; it argues the generated graph is biologically useful, not just distribution-matched.

**Pairwise sanity.** AUROC / AUPRC on the balanced test set, to demonstrate C4 (pairwise quality preserved while topology improves).

**Baselines.**
- Independent pairwise scoring: our frozen v3.1 scorer (the strong incumbent) + published PRING baselines (PLM-interact 35M/650M, TUnA, PPITrans, D-SCRIPT, Topsy-Turvy, SPRINT, Struct2Graph, TAGPPI).
- Topology-aware training (`src/topology/` finetune): same v3.1 classifier, augmented training objective with GS / RD / degree-MMD / clustering-MMD losses — the *natural* internal baseline; same architecture as the incumbent but with topology-aware training signal.
- Post-hoc denoising: our S2GAE refiner (Route A) — applied after the frozen pairwise scorer; treats the graph as an artifact to clean up rather than fixing the training objective.
- (Recommended) a transductive graph-completion method run under PRING's inductive split, to make the leakage/inductive argument concrete rather than rhetorical.

**Ablations (define C3's necessity claim).** Remove context (collapses to independent scoring); vary the potential-neighbor set size/construction; generation vs. denoising with matched features; self-consistent operating point vs. tuned threshold.

**Integrity gates (must clear before any headline number).**
- Fix the ratio-5 negative-pool leak (test proteins currently enter refiner training/threshold calibration unfiltered) → certify strict inductive.
- Confirm no target-graph information reaches the model at train or inference time.

---

## 6. Paper Structure & Chapter Plan

Conference / ML-IMRaD, ~9 pages. Each chapter below carries its core argument, key evidence, the counter-argument a reviewer will raise, and our response.

### 1. Introduction (~1.25 pp)
- **Core argument:** the pair-to-graph gap is real and unfixable by either thresholding or changing the training objective alone; interactomes must be generated, not assembled from independent scores.
- **Evidence:** v3.1 beats SOTA AUPR yet density 3.13; topology-aware training (same classifier, new loss) is the natural fix but remains pairwise-independent in structure; post-hoc denoiser (S2GAE) is a no-op at thr 0.5 and over-sparse at calibrated thr 0.96.
- **Counter → response:** *"Just a calibration problem."* → the exp02 calibration cliff shows no single threshold reaches realistic density without collapsing recall; and the training-objective fix (`src/topology/`) does not change the pairwise-independent structure of the model.

### 2. Related Work (~0.75 pp)
- **Core argument:** the two incumbent formulations each miss half the problem; graph-native context methods leak, pairwise/PLM methods are structure-blind.
- **Evidence/positioning:** PRING (gap benchmark); PLM-interact / TUnA / MINT (pair-aware, structure-blind); DNE (graph-native, needs a network → leakage); MAPE-PPI (microenvironment "context" but transductive); GSL / masked-graph-autoencoder lineage.
- **Counter → response:** *"GNN-PPI already uses graph context."* → those are transductive or link-prediction-on-the-same-graph; they violate the unseen-protein constraint we hold.

### 3. Problem Formulation & the Pair-to-Graph Gap (~1 pp) — houses C1
- **Core argument:** define "structured/realistic" interactome representation; show independent per-edge prediction cannot satisfy it.
- **Evidence:** formal density/degree/clustering characterization + the compounding-error argument, instantiated on our own numbers.
- **Counter → response:** *"Topology metrics are arbitrary."* → they are PRING's published, biology-motivated metrics; we adopt them unchanged.

### 4. Method: Conditional Local-Graph Generation (~2 pp) — houses C2
- **Core argument:** generation conditioned on features + potential-neighbor set respects topology by construction and yields a self-consistent operating point.
- **Evidence:** the five-stage formulation (§4 above); how it subsumes denoising.
- **Counter → response:** *"Generation over 2M pairs is intractable."* → the potential-neighbor set localizes generation; complexity is bounded and inductive.

### 5. Experiments (~2.5 pp) — houses C3, C4, C5
- **Core argument:** the method beats independent scoring and denoising on topology (C3), preserves pairwise quality with no tuned threshold (C4), and improves function tasks (C5); ablations show context is necessary.
- **Evidence:** §5 tables; leakage-audited inductive protocol; PRING-aligned metrics.
- **Counter → response:** *"The win is from the frozen scorer / from leakage."* → context-ablation degrades to baseline; strict inductive certificate; matched-feature comparison.

### 6. Discussion & Limitations (~0.5 pp)
- Honest limitations: single benchmark (PRING), human-primary, generative-model cost, cross-species left as future work.

### 7. Conclusion (~0.25 pp)
- One idea to remember: **realistic interactomes are grown, not thresholded.**

**Appendix:** formal argument details, method/architecture specifics, full metric tables, leakage audit, reproducibility.

---

## 7. Positioning vs Related Work (one-line map)

- **PRING** — the benchmark that exposes the gap; our anchor and our metric source.
- **PLM-interact / TUnA / MINT / PPITrans** — strong *pairwise* incumbents; our "independent-scoring" baseline family; structure-blind.
- **DNE, transductive GNN-PPI** — graph-native context but require an existing network → the leakage counterpoint we run under the inductive split.
- **MAPE-PPI** — nearest "context" precedent (residue microenvironment), but transductive network reasoning; we are inductive and generate topology.
- **Graph generation / masked-graph autoencoders / GSL** — our methodological toolbox; the underexplored cluster (feature-conditioned generators optimizing graph-level topology) is exactly our niche.

---

## 8. Risks & Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Generative method unbuilt and *must* win by Sep 2026 | High | De-risk with a minimal generative prototype in the first milestone; keep the diagnosis + denoiser as a fallback narrative (a real negative result). |
| Potential-neighbor set intractable / leaky | High | Feature-space retrieval or coarse prefilter, constructed without the test graph; bound generation locally. |
| Leakage in our own pipeline undercuts the inductive claim | High | Fix ratio-5 negative pool + certify strict inductive **before** any headline number. |
| Metric non-comparability vs official PRING | Medium | Reconcile MMD to raw PRING `eval.py`; report both if needed. |
| Generative training instability / scale | Medium | Start local/small; validate on train-node subgraphs first. |
| Cross-species not implemented (data-only) | Low | Explicitly future work; not claimed. |

---

## 9. Open Decisions to Lock Next (recommended defaults)

1. **Generative family** — *recommend* starting with autoregressive local-subgraph growth (most directly matches the "grow from two seeds" thesis and gives a natural self-consistent stopping/density rule); revisit one-shot/diffusion if training is unstable.
2. **Frozen scorer role** — *recommend* using it as a prior/candidate generator seeding the potential-neighbor set (reuses your proven AUPR-0.792 signal), not discarding it — this also makes the "subsumes denoising" story clean.
3. **Transductive baseline** — *recommend* including one graph-completion method under the inductive split, to convert the leakage critique into a measured result.
4. **Function-task depth** — *recommend* all three PRING function tasks if time allows; GO-enrichment is the minimum.

---

## 10. Timeline (back-solved from ~Sep 2026)

| Window | Milestone |
|---|---|
| Jul 2026 | Lock §9 decisions; fix leakage + align metrics; re-baseline v3.1 scorer & S2GAE denoiser on the clean inductive protocol; write C1 formalism. |
| Aug 2026 | Minimal generative prototype (local growth on train-node subgraphs); first topology numbers vs. baselines; go/no-go on the generative bet. |
| Aug–early Sep 2026 | Scale prototype; ablations (context necessity, neighbor-set, generation-vs-denoise); function-task evaluation. |
| Sep 2026 | Draft full paper (`academic-paper full`), internal review, freeze results, submit. |

**Go/no-go checkpoint (end Aug):** if the generative prototype does not clearly beat the denoiser on graph_sim + density under the clean protocol, pivot the framing to "diagnosis + why fixes fail" (still a defensible paper) rather than forcing a weak "wins" claim.

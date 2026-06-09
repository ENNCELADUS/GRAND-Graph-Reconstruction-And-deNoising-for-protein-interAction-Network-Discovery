## 1. 总结

这套方案应该定义为 **Pairwise-generated graph → S2GAE-style topology denoiser/refiner → refined graph**，而不是“直接拿 vanilla S2GAE 做 inductive test”。训练时必须在 train/internal-val proteins 上先用 frozen 或 out-of-fold pairwise classifier 生成 `G_pairwise`，再让一个 **S2GAE-style inductive denoiser** 学习从 noisy predicted graph 恢复真实 train topology；测试时 held-out proteins 的图输入只能来自 pairwise predictions，`human_test_graph.pkl` 只能作为 topology metric 的 ground truth，不能进入模型。你的附件里 `all_test_ppi.txt` 被定义为 topology reconstruction 的 candidate universe，`human_test_graph.pkl` 是 topology metric ground truth，这正好支持这个严格 test protocol。

## 2. Stage-by-stage 方案

### Stage 0 — Pairwise classifier 准备

第一版具体 pairwise 模块是 checkpoint-backed v3.1
`pair_context_gated_abba_no_cross` scorer。它通过
`pairwise_scorer.target: tccig.train:score_pairs_with_v3_1` 接入
`tccig/train.py`，从配置加载 v3.1 architecture config、single checkpoint 和
ESM3 embedding cache，对 TCCIG scaffold 传入的 label-free candidate pairs 输出：

```text
s_ij = sigmoid(Cφ(x_i, x_j))
l_ij = logit(s_ij)
```

这个 scorer 的固定 architecture contract 是：

```yaml
model_config:
  model: v3.1
  pair_readout:
    mode: pair_context_gated
    order_aggregation: abba_max
  interaction:
    mode: none
```

当前实现保留 frozen pairwise scorer，并通过 `refiner.train_target:
tccig.s2gae:train_refiner` / `refiner.predict_target:
tccig.s2gae:predict_refined` 接入 S2GAE residual denoiser。TCCIG run 内不训练
pairwise model；pairwise scorer 只负责生成 `G_pairwise` 和 residual baseline。

训练 denoiser 时不要直接用“在同一批 train labels 上训练过的 pairwise model”给 train graph 打分，否则 noisy graph 会过于干净。建议用 **K-fold out-of-fold pairwise predictions**：

```text
for fold k:
    train Cφ_k on train proteins except fold k
    score candidate pairs inside fold k / validation subgraph
merge OOF scores -> G_pairwise_train
```

测试时只用最终 pairwise model `Cφ_all` 对 held-out candidate pairs 打分。

---

### Stage 1 — 构造 noisy graph `G_pairwise`

对每个 split 的 candidate universe `Ω`：

```text
score all (i, j) in Ω with Cφ
build E_pairwise = { (i, j): s_ij >= τ_pair }
G_pairwise = (V, E_pairwise, edge_weight=s_ij)
```

`G_pairwise` 是 refiner 的 noisy input graph；它可以用 pairwise scorer
概率阈值构造，但不能使用真实 topology。当前实现把这个阈值定义为
**pairwise input graph threshold**：epoch 0 在 scorer-only validation scores
上选择达到 `precision >= 0.8` 的最低 threshold，然后冻结并用于所有 split 的
`G_pairwise` 构造。

```text
τ_pair = min τ such that precision(s_ij >= τ on validation) >= 0.8
build E_pairwise = { (i, j): s_ij >= τ_pair }
```

refined hard graph 的最终决策不复用 `τ_pair`。Stage 6 使用独立的
**refined output threshold**，当前固定为 `p_refined >= 0.5`。per-node top-k
和 global top-M 仍然禁用，因为这些规则分别强制非生物的固定出边数和固定全图
边数。

---

### Stage 2 — Denoiser / refiner 模型

核心模型是 **S2GAE-style inductive GNN encoder + S2GAE cross-correlation decoder + pairwise residual connection**。

S2GAE 原文的关键 formulation 是：

```text
G_perb = (V, E_remain), E_remain = E - E_mask
h_v^k = COM(h_v^{k-1}, AGG({h_u^{k-1}: u in N_v}))
h_e(v,u) = concat_{k,j=1..K}(h_v^k * h_u^j)
g(v,u) = MLP(h_e(v,u))
```

这里不直接照搬 vanilla S2GAE 的 `true graph -> mask -> reconstruct masked true edges`。对 TCCIG，`G_pairwise` 已经是 classifier 生成的 noisy / perturbed graph，因此 first implementation 可以把它视作 S2GAE 的 `G_perb`，再训练 decoder 去恢复真实 train topology。可选地，在训练时对 `G_pairwise` 做轻量 edge dropout 作为额外 perturbation，但目标仍然是 `A_true`，不是重建 `G_pairwise` 本身。

#### Forward

输入：

```text
X = protein node features
G_pairwise = predicted graph from pairwise classifier
Ω_batch = candidate pairs to score
```

Encoder 采用 S2GAE-style GNN backbone。它必须是 **feature-based inductive encoder**，不能是 transductive node embedding table。v1 实现使用 PyG `GraphConv`，并把 `G_pairwise` 中的 pairwise score confidence 作为 `edge_weight=s_ij` 输入 weighted message passing。

```text
G_input = G_pairwise
# optional training-only perturbation:
# G_input = drop_edges(G_pairwise, drop_rate)

h_i^0 = MLP_x(x_i)
for k = 1..K:
    h_i^k =
        W_self h_i^{k-1}
        + W_neigh AGG_k({edge_weight_ij * h_j^{k-1}: j in N_input(i)})
```

这里的 `G_input` 对应 S2GAE 的 perturbed graph input；区别是 perturbation 主来源不是从真实图里人工 mask，而是 pairwise classifier 的 imperfect predictions。当前 `GraphConv` 路径不显式加入 self-loop；node self-information 来自 root/self transform `W_self h_i`。这样 test 时也能用同一套流程：`X_test + G_pairwise_test -> H_test`，不需要也不允许访问 `human_test_graph.pkl`。

Decoder 用 S2GAE 的 cross-correlation 思路，显式利用多层 node representations 的交叉粒度信息：

```text
z_ij = concat_{a,b=1..K}(h_i^a * h_j^b)
z_ij = concat(z_ij, |h_i^K - h_j^K|, h_i^K * h_j^K)
Δ_ij = MLP_dec(z_ij)
p_refined_ij = sigmoid(l_pairwise_ij + Δ_ij)
```

`p_refined_ij = sigmoid(l_pairwise_ij + Δ_ij)` 是硬性设计：denoiser 学的是 residual refinement，而不是从零替代 pairwise classifier。训练时可以在 edge-score level 上计算 BCE；validation/test 使用独立配置的 refined output threshold，当前固定为 `0.5`。

S2GAE 适合作为主模板，因为它不是只重建 node feature，而是用 perturbed graph input、multi-layer GNN hidden states、cross-correlation decoder 来预测 missing edges。TCCIG 的改动是把 “masked edges from true graph” 换成 “missing / false edges relative to true topology under a classifier-generated noisy graph”。

---

### Stage 3 — Training loss

训练 target 是真实 train topology，不是 pairwise graph 本身。

对 train proteins：

```text
G_input  = G_pairwise_train
A_true   = adjacency(train positive PPI graph)
p_refine = Dθ(X_train, G_input, Ω_train)
```

v1 总 loss 已接入 `tccig.s2gae:train_refiner`。当前实现包含 supervised link
denoising、residual anchor，以及只来自 train topology buckets 的可微 topology
surrogate：

```text
L = L_bce
  + λ_resid * L_residual_anchor
  + λ_topo * (α * L_graph_similarity + β * L_relative_density)
```

#### 1. BCE denoise loss

```text
L_bce = BCEWithLogits(l_ij_refined, y_ij; pos_weight, label_smoothing)
```

其中 `y_ij` 来自 train candidate rows 的真实 PRING labels。权重通过
`refiner.loss` 显式配置，默认保持 ratio5 采样语义不变：

```yaml
refiner:
  loss:
    type: bce_with_logits
    pos_weight: 1.0
    label_smoothing: 0.0
```

#### 2. Residual anchor loss

防止 denoiser 过度 hallucinate。该项对 train batch 里的全部 candidate
pairs 计算，不只约束 negatives：

```text
L_residual_anchor = mean_{(i,j) in Ω_batch}(Δ_ij^2)
```

权重要小，只是约束 refined score 不要脱离 pairwise evidence。

#### 3. Train soft topology loss

训练侧 topology loss 只从 train true topology target 采样 PRING-style
bucket all-pairs；validation/test truth 不进入 backward。loss 使用已有可微
surrogate，在第一版只启用 graph similarity 和 relative density：

```text
L_topology =
    α * graph_similarity_loss
  + β * relative_density_loss

γ = 0, δ = 0
```

hard NetworkX topology metrics 仍然只用于 validation/test reporting 和 checkpoint
monitor，不作为 test-time threshold search。后续如需更强正则，可以再评估
degree/clustering MMD 或 soft distillation：

```text
L_distill = BCE(p_ij_refined, stopgrad(s_ij_pairwise))
```

#### Future: Optional Bandana-style weighted message objective

如果 `G_pairwise` 的 edge weights 很有信息，不建议立刻二值化丢掉。Bandana 的优势正是把 Bernoulli edge mask 换成 continuous bandwidth mask，并做 layer-wise bandwidth prediction；它适合作为 **weighted message-passing / edge-confidence refinement variant**，但比 S2GAE/MGAE 更 experimental。

---

### Stage 4 — Backward / optimization

第一版保持最稳，已经按 standalone `tccig.s2gae:train_refiner`
实现为 refiner-only optimization：

```text
freeze Cφ
train θ only
```

`Cφ` 是 pairwise scorer hook，只在进入 refiner 前生成 pairwise probabilities
和 `G_pairwise`。它不进入 autograd graph，也不会被 optimizer 持有。

当前 standalone S2GAE 路径使用 **full-batch epoch update**，避免在同一
epoch 内对同一个 `G_pairwise` 重复运行 GraphConv encoder。每个 epoch：

```text
1. construct/load G_pairwise subgraph
2. encode full graph once: H = Encoderθ(X, G_pairwise)
3. shard Ω by distributed rank with original pair indices
4. decode rank-local Ω chunks from cached H
5. reduce rank-local BCE + residual-anchor losses to the global mean objective
6. accelerator.backward(loss) on θ
7. optionally clip θ gradients
8. one AdamW update on θ
```

Validation and test refined prediction follow the same operational rule: encode the
split graph once per eval pass, decode only rank-local candidate pairs, gather
scores back into original candidate-file order, then apply the fixed refined output
threshold globally. Initial pairwise scoring is also rank-sharded by original
candidate row index, with progress evidence under
`data/tccig/score_cache/<run_id>/`.

Training progress is written after every completed epoch to
`logs/tccig/<run_id>/tccig_train_step.csv`, alongside
`training_summary.json`. Per-epoch details are retained inside
`training_summary.json.history`; standalone per-epoch manifest files are not
written.

当前 config 明确使用 fixed-LR AdamW，不启用 scheduler：

```yaml
refiner:
  optimizer:
    type: adamw
    lr: 0.001
    weight_decay: 0.0
    beta1: 0.9
    beta2: 0.999
    eps: 1.0e-8
  scheduler:
    type: none
  optimization:
    gradient_clip_norm: 1.0
```

不要 end-to-end 更新 pairwise classifier；否则 refinement gain 很难解释。只有在
denoiser 稳定提升后，才做 optional joint fine-tune，而且要用 very small LR，
并保留 frozen-pairwise baseline。

---

### Stage 5 — Validation

Validation 必须模拟 test：

```text
1. 从 human_val_ppi_ratio5_exclusive.txt 的正例构造 validation true topology
2. 在 validation true topology 上采样 PRING-style node buckets: 20, 40, ..., 200
3. 对每个 validation bucket materialize bucket 内所有 non-self protein pairs
4. 用 pairwise classifier 对这些 label-free bucket all-pairs 打分
5. 构造 G_pairwise_val_topology
6. 输入 G_pairwise_val_topology + X_val_topology 到 denoiser
7. 输出 refined scores
8. 用 fixed refined output threshold 重建 hard refined bucket graphs
9. 和 validation true topology bucket subgraphs 比较，计算 graph_sim / relative_density / MMD metrics
```

注意这里的 validation topology candidate universe 不是
`human_val_ppi_ratio5_exclusive.txt` 本身。ratio5 文件只定义 validation true
topology target；真正用于 topology validation 的候选边是 bucket 内 all-pairs，
这样 validation 才和 test-time `all_test_ppi.txt` graph reconstruction 语义对齐。

v1 选择 checkpoint 和 hard graph rule 的主指标由 config 控制：

```text
refiner.monitor_metric =
    val_topology_loss
  | internal_val_graph_sim
  | val_graph_sim
  | internal_val_relative_density
  | val_relative_density
  | val_auprc
```

当 monitor 是 topology 指标时，validation 只决定 best checkpoint。refined
output rule 由 config 固定为 `threshold=0.5`，validation/test 都不再 sweep
logit bias 或重新选择 threshold；test 也不能根据 `human_test_graph.pkl` 选
threshold、edge budget、或 degree budget。

`val_topology_loss` 使用和 topology fine-tune validation 一致的 hard-metric
penalty：

```text
alpha * (1 - graph_sim)
+ beta * (relative_density - 1)^2
+ gamma * deg_dist_mmd
+ delta * cc_mmd
```

---

### Stage 6 — Test time

Test 输入只有：

```text
held-out protein features X_test
candidate pairs Ω_test = all_test_ppi.txt
```

流程：

```text
1. s_ij_pairwise = Cφ_all(x_i, x_j), for all (i,j) in all_test_ppi.txt
2. build G_pairwise_test from pairwise scores only
3. H_test = Encθ(X_test, G_pairwise_test)
4. p_ij_refined = sigmoid(l_ij_pairwise + Decθ(H_i, H_j))
5. apply fixed refined output threshold: p_ij_refined >= 0.5
6. write refined positive pairs as predicted graph
7. evaluate against human_test_graph.pkl
```

For the standalone `tccig/` pipeline, binary pairwise test metrics follow the
same full-model scoring boundary: `logs/tccig/{run_id}/pairwise_test` uses
refined probabilities from v3.1 + refiner on `human_test_ppi.txt`. The frozen
v3.1-only result is retained as the pinned
`logs/tccig/pairwise_baseline` artifact and is not regenerated by the
pipeline.

硬性约束：

```text
G_pairwise_test must not use human_test_graph.pkl
G_pairwise_test must not use true labels from all_test_ppi.txt
refined output threshold must not be selected on test
global top-M and per-node top-k are forbidden graph decision rules
```

附件里 topology test 明确说 `all_test_ppi.txt` 是 candidate universe，预测 positive rows 后重建 predicted graph；`all_test_ppi.txt` 的 labels 不是 topology metric source of truth，`human_test_graph.pkl` 才是 ground truth。

---

## 模型选择结论

**主方案：S2GAE-style residual denoiser。**
S2GAE 最适合作为核心，因为它把 perturbed graph input、GNN encoder、multi-layer cross-correlation decoder、masked-edge reconstruction 连成了一个清晰 formulation；但必须改成“noisy pairwise graph 输入 → true topology target”的 supervised denoising，而不是直接 vanilla SSL pretrain。

**强 baseline：MGAE-style denoiser。**
MGAE 更简单：high-ratio edge masking、partial graph encoder、cross-correlation decoder。适合作为 minimal strong topology denoiser baseline。它的论文也显示 mask ratio around `0.5–0.7` 相对稳，`0.7` 附近表现强。

**实验升级：Bandana-style weighted denoiser。**
Bandana 适合处理你的 `G_pairwise`，因为 pairwise graph 天然有 soft confidence；continuous bandwidth 比 binary edge deletion 更贴合 “predicted noisy graph refinement”。但它应作为 ablation/upgrade，而不是第一版主方法。

最终推荐的 paper claim 是：

```text
We convert a mature pairwise PPI classifier into an inductive graph reconstruction system by using its predictions to construct an input graph, then training an S2GAE-style residual topology denoiser to refine that noisy graph under train soft-topology objectives and validation-monitored checkpointing.
```

v1 的实现性 claim 应收窄为：

```text
We convert a mature pairwise PPI classifier into an inductive graph reconstruction system by using its predictions to construct an input graph, then training an S2GAE-style residual denoiser with supervised link reconstruction, train GS/RD topology loss, and fixed refined-output thresholding.
```

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
# 或者：per-node top-k / global top-M / target-density-calibrated top-M
G_pairwise = (V, E_pairwise, edge_weight=s_ij)
```

推荐优先用 **validation-calibrated top-M/top-k**，而不是固定 `0.5`，因为 pairwise classifier 的 probability calibration 未必能给出合理 graph density。可以在 internal validation 上选择：

```text
τ_pair or k* = argmin topology_val_loss
```

如果当前 `topology_evaluate` 固定用 `sigmoid(logit) >= 0.5`，可以把阈值吸收到 logit：

```text
l'_ij = l_ij - logit(τ*)
```

这样 evaluator 仍然用 `0.5`，但实际 operating point 是 validation 选出来的 `τ*`。

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

Encoder 采用 S2GAE-style GNN backbone。它必须是 **feature-based inductive encoder**，不能是 transductive node embedding table；backbone 可以是 GraphSAGE、GIN、GATv2，或者显式使用 `edge_weight=s_ij` 的 weighted message passing。

```text
G_input = G_pairwise
# optional training-only perturbation:
# G_input = drop_edges(G_pairwise, drop_rate)

h_i^0 = MLP_x(x_i)
for k = 1..K:
    h_i^k = COM_k(
        h_i^{k-1},
        AGG_k({edge_weight_ij * h_j^{k-1}: j in N_input(i)})
    )
```

这里的 `G_input` 对应 S2GAE 的 perturbed graph input；区别是 perturbation 主来源不是从真实图里人工 mask，而是 pairwise classifier 的 imperfect predictions。这样 test 时也能用同一套流程：`X_test + G_pairwise_test -> H_test`，不需要也不允许访问 `human_test_graph.pkl`。

Decoder 用 S2GAE 的 cross-correlation 思路，显式利用多层 node representations 的交叉粒度信息：

```text
z_ij = concat_{a,b=1..K}(h_i^a * h_j^b)
z_ij = concat(z_ij, |h_i^K - h_j^K|, h_i^K * h_j^K)
Δ_ij = MLP_dec(z_ij)
p_refined_ij = sigmoid(l_pairwise_ij + Δ_ij)
```

`p_refined_ij = sigmoid(l_pairwise_ij + Δ_ij)` 是硬性设计：denoiser 学的是 residual refinement，而不是从零替代 pairwise classifier。训练时可以在 edge-score level 上计算 BCE；validation/test 再用 validation-selected threshold/top-k/top-M 把 refined scores 组装成 hard graph。

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

v1 总 loss 已接入 `tccig.s2gae:train_refiner`。当前实现只做 supervised
link denoising，不把 topology metric loss 放进 backward：

```text
L = L_bce
  + λ_resid * L_residual_anchor
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

#### Future: Topology loss

当前 v1 不训练这个 loss；topology metrics 只用于 validation/test 侧的 graph
decision rule 和结果解释。后续如果要加入训练目标，可以再用 soft adjacency
`P_refined` 对齐 `A_true`：

```text
L_topology =
    α * graph_similarity_loss
  + β * relative_density_loss
  + γ * degree_mmd
  + δ * clustering_mmd
```

或后续加入 soft distillation：

```text
L_distill = BCE(p_ij_refined, stopgrad(s_ij_pairwise))
```

#### Future: Optional Bandana-style weighted message objective

如果 `G_pairwise` 的 edge weights 很有信息，不建议立刻二值化丢掉。Bandana 的优势正是把 Bernoulli edge mask 换成 continuous bandwidth mask，并做 layer-wise bandwidth prediction；它适合作为 **weighted message-passing / edge-confidence refinement variant**，但比 S2GAE/MGAE 更 experimental。

---

### Stage 4 — Backward / optimization

第一版保持最稳：

```text
freeze Cφ
train θ only
```

每个 batch：

```text
1. construct/load G_pairwise subgraph
2. forward S2GAE-style encoder on G_pairwise
3. decode Ω_batch candidate pairs with cross-correlation decoder
4. compute BCE + residual anchor losses
5. backward on θ
6. update θ
```

不要 end-to-end 更新 pairwise classifier；否则 refinement gain 很难解释。只有在 denoiser 稳定提升后，才做 optional joint fine-tune，而且要用 very small LR，并保留 frozen-pairwise baseline。

---

### Stage 5 — Validation

Validation 必须模拟 test：

```text
1. 用 pairwise classifier 对 validation candidate pairs 打分
2. 构造 G_pairwise_val
3. 输入 G_pairwise_val + X_val 到 denoiser
4. 输出 refined scores
5. 用 validation 选 τ_refine / top-k / top-M
6. 重建 hard refined graph
7. 和 validation true topology 比较
```

v1 选择 checkpoint 的主指标：

```text
primary = val_auprc
secondary = validation rule metrics
```

附件里 internal topology validation 本身就是用 validation target graph、固定 node buckets、hard predicted subgraphs，并计算 graph metrics；monitor metric 支持 `val_topology_loss`、`internal_val_graph_sim`、`internal_val_relative_density`、`val_auprc` 等。

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
5. apply validation-selected τ_refine/top-k/top-M
6. write refined positive pairs as predicted graph
7. evaluate against human_test_graph.pkl
```

硬性约束：

```text
G_pairwise_test must not use human_test_graph.pkl
G_pairwise_test must not use true labels from all_test_ppi.txt
threshold/top-k must be selected on validation, not test
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
We convert a mature pairwise PPI classifier into an inductive graph reconstruction system by using its predictions to construct an input graph, then training an S2GAE-style residual topology denoiser to refine that noisy graph under validation-calibrated graph-level objectives.
```

v1 的实现性 claim 应收窄为：

```text
We convert a mature pairwise PPI classifier into an inductive graph reconstruction system by using its predictions to construct an input graph, then training an S2GAE-style residual denoiser with supervised link reconstruction and validation-calibrated graph decision rules.
```

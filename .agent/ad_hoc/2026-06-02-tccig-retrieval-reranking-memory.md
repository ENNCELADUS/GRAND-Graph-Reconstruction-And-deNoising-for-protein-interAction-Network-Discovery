**当前 TCCIG 的核心问题不是 saturation，而是任务建模错位**。它试图在 transcriptome/proteome 级别的 dense candidate universe 上直接学习全局 edge localization，但训练时真正有 BCE 监督的 pair 只有 `6.3%`，导致模型可以在少量 supervised pairs 上拟合 BCE，同时在全局 candidate ranking 上完全失位。P4 把概率从 exact 1.000 拉到 0.971，但 top-K 反而比随机还差，这说明“logit calibration / saturation”不是主因；主因是 **retrieval / localization objective 没学到**。

我建议现在把方案从：

```text
feature-only all-pairs graph generator
```

改成：

```text
train-graph-prior-conditioned retrieval + candidate re-ranking model
```

也就是：**充分利用 training PPI graph 学结构先验；test 时不在全枚举空间上做 heavy ranking，而是先用 factorizable retrieval 产生 candidate graph，再只对 candidate pairs 做强 re-ranking，最后按 PRING 格式输出 reconstructed graph。**

---

# 1. PRING 对任务的真实定义：必须重新对齐

PRING 的 topology-oriented task 不是普通 binary PPI prediction。它的 intra-species 任务明确要求：用训练好的 PPI predictor 去重建完整 human test PPI network；文件上区分了 `human_test_ppi.txt` 作为 binary classification test pairs，和 `all_test_ppi.txt` 作为 graph reconstruction test pairs。PRING 文档直接写明 `all_test_ppi.txt` 是 “Test pairs (graph reconstruction)”，而 `human_test_ppi.txt` 是 “Test pairs (binary classification)”；模型推理阶段要求对 `all_test_ppi.txt` 中的 pairs 预测，然后用这些预测重建完整 test graph。([GitHub][1])

PRING 的 cross-species 设定也类似：用 human-trained model 去 non-human species 上重建 PPI network；每个 species 提供 `*_all_test_ppi.txt` 作为 all-against-all test pairs，并用 BFS / DFS / Random Walk sampled subgraphs 评估 graph topology。([GitHub][2]) 数据结构文件也确认：human split 下每个 BFS/DFS/RANDOM_WALK 子目录都有 `all_test_ppi.txt`、`human_test_graph.pkl`、`human_test_ppi.txt`、`human_train_graph.pkl`、`human_train_ppi.txt` 和 `human_val_ppi.txt`；cross-species 的 arath/ecoli/yeast 也有 `*_all_test_ppi.txt`、`*_test_graph.pkl` 和 sampled node files。([GitHub][3])

所以更新后的任务定义应是：

```text
Train:
  input:
    human_train_ppi.txt
    human_val_ppi.txt
    human_train_graph.pkl
    protein sequences / ESM embeddings
  allowed:
    training graph topology as prior/supervision
  forbidden:
    test graph edges, test degrees, test neighborhoods, test communities

Validation:
  use validation pairs / validation-like reconstruction plan for threshold, retrieval depth, graph assembly calibration

Test:
  input:
    all_test_ppi.txt candidate universe
    test protein intrinsic features
  output:
    all_test_ppi_pred.txt:
      uniprot_id1 uniprot_id2 label
  graph reconstruction:
    positive labels form predicted graph
```

PRING 的 `eval.py` 只读取 predicted file 中 `label > 0` 的 pairs 来 reconstruct graph，然后评估 graph similarity、relative density、degree MMD、clustering coefficient MMD 和 Laplacian spectral MMD。([GitHub][4]) 其中 graph similarity 实际是：

[
\text{GS}=1-\frac{\lVert \hat A-A\rVert_1}{\sum \hat A+\sum A}
]

PRING 代码里就是把 predicted subgraph 和 ground-truth subgraph 的 adjacency matrix 做 absolute difference 后归一化。([GitHub][4])

这意味着你现在不应该再把 `human_test_ppi` 的 binary metrics 当主目标。`human_test_ppi` 只能是 auxiliary diagnostic；主任务是 `all_test_ppi → all_test_ppi_pred → reconstructed graph → PRING eval`。

---

# 2. 当前设计为什么从根上不适配 PRING

## 2.1 Dense all-pairs graph generator 的监督密度太低

你现在的训练日志显示 `bce_supervised_fraction=0.063`，这意味着在 sampled graph forward 里，绝大多数 candidate pairs 没有 BCE 监督。更严重的是，PPI 中 unobserved pairs 不是可靠 negatives；把它们隐式纳入 topology loss 或 score ranking，会产生大量 label ambiguity。

所以当前模型面临的是：

[
\text{few supervised pair labels}
+
\text{huge unlabeled pair universe}
+
\text{extreme sparsity}
+
\text{graph-level topology evaluation}
]

而当前 TCCIG 的 all-pairs decoder 实际在做：

[
P_{ij}=f_\theta(x_i,x_j,\text{weak set context})
]

它没有一个强 retrieval objective 来学习“哪些区域应该出现真实 edges”。因此 P4 即使修了 branch scaling，top-K localization 仍然坏掉。

## 2.2 Graph prior 只作为 weak teacher 不够

之前把 MGAE/S2GAE/MaskGAE/Bandana 类方法只作为 teacher distillation，信号太弱。masked-edge reconstruction literature 的意义不是“给 pair score 加一个 soft label”，而是它可以从 training PPI graph 中学习 **node structural roles、local connectivity patterns、diffusion neighborhoods、module priors、hard negatives**。你现在只把它作为在线 teacher 的 pair-level distillation，等价于把强图结构先验压缩成了很窄的一点 pair score，损失了大部分结构信息。

已有材料也明确指出，PPI 中 ESM/ESM3 已经给了强 node features，真正缺的是 topology signal；结构 masked variants，尤其 MGAE、MaskGAE、S2GAE、Bandana，正是通过 mask edges / paths / bandwidths 学 missing links、degrees 或 connectivity patterns。

## 2.3 你真正需要的是 retrieval，而不是 dense matrix generation

PRING 虽然提供 `all_test_ppi.txt`，但这不等于模型必须 heavy-score / heavy-sort 全枚举空间。RaftPPI 和 FlashPPI 都说明了一个关键方向：proteome-scale PPI prediction 应该重构成 **retrieval problem**。

RaftPPI 明确指出，传统 residue-level pair model 在 proteome scale 需要 (O(N^2L^2))，而它通过 Gaussian kernel + random Fourier features + low-rank attention，把 residue-aware PPI score factorize 成 per-protein embedding dot product，从而用 ANN / HNSW 替代 exhaustive pairwise scoring。 FlashPPI 也把 PPI prediction reframed as dense retrieval：先把 interacting proteins 映射到 shared latent space，再 top-k retrieval，只对 (N \times k) candidates 做 contact prediction；训练用 InfoNCE、false negative masking 和 online hard negative mining。

这与 P4 失败高度吻合：你不是缺一个更稳定的 sigmoid；你缺的是一个能在巨大 candidate universe 中把 true edges retrieve 到前面的 objective。

---

# 3. 新任务定义：PRING-aligned Graph-Prior Retrieval

我建议把项目任务定义更新为：

> Given a PRING test protein set and its `all_test_ppi.txt` candidate universe, learn a feature-only, graph-prior-conditioned retrieval and re-ranking model trained on the human training PPI graph. At inference, the model must not use target-test topology; it retrieves a sparse candidate graph from intrinsic protein features and learned graph priors, re-ranks only retrieved candidate pairs, and outputs PRING-compatible binary labels for graph reconstruction.

形式化：

训练数据：

[
G_{\text{train}}=(V_{\text{train}},E_{\text{train}})
]

[
X_{\text{train}}={x_i:i\in V_{\text{train}}}
]

测试输入：

[
X_{\text{test}},\quad C_{\text{test}}=\texttt{all_test_ppi.txt}
]

输出：

[
\hat y_{ij}\in{0,1},\quad (i,j)\in C_{\text{test}}
]

但模型内部不需要对所有 ((i,j)\in C_{\text{test}}) 做 expensive pair encoder。它只需要：

```text
1. encode each protein once
2. ANN retrieve top-K candidate partners per protein
3. intersect retrieved candidates with all_test_ppi
4. re-rank only retrieved candidate pairs
5. label selected pairs as 1, all others as 0
```

这样严格对齐 PRING，因为最后仍然输出完整 `all_test_ppi_pred.txt`；同时避免在 test 上对全枚举空间做 heavy pairwise ranking。

---

# 4. 主推新架构：GPR-PPI

我建议替换当前 TCCIG 为：

# **GPR-PPI: Graph-Prior Retrieval and Re-ranking for PRING**

整体 pipeline：

```text
Training graph G_train + protein intrinsic features
        ↓
Train-only graph prior teacher
        ↓
Feature-to-structure prior distillation
        ↓
Factorized retrieval encoder
        ↓
ANN candidate generator
        ↓
Candidate re-ranker with graph-prior features
        ↓
PRING graph assembly
```

核心变化是：**graph prior 不再只是 pair teacher，而是变成 train-time dense structural supervision；test-time 使用的是从 intrinsic features 预测出来的 graph-prior embedding，不使用 target topology。**

---

# 5. Module 1：Protein encoder

输入仍然是 ESM / PLM embeddings，但建议从 protein-level embedding 升级到 residue-aware factorized embedding。

最小版本：

[
h_i = \text{MLP}_{\theta}(\text{ESMMeanPool}(p_i))
]

推荐版本：

[
R_i = \text{ESMResidueEmbeddings}(p_i)
]

然后学习多个 per-protein vectors：

[
z_i^q,\ z_i^k,\ z_i^{struct},\ z_i^{res},\ z_i^{module},\ \hat d_i
]

含义：

```text
z_i^q / z_i^k       retrieval query/key embeddings
z_i^struct          predicted graph-structural prior embedding
z_i^res             residue-factorized interaction embedding
z_i^module          predicted module / complex prior
d_hat_i             predicted degree / hub propensity
```

---

# 6. Module 2：Train-only graph prior teacher

这里是对之前 MGAE/S2GAE teacher 的重大升级。

不要只训练一个 online MGAE teacher 给 pair score。改成离线训练一个 **Graph Prior Teacher**，从 `human_train_graph.pkl` 中提取多种 topology targets：

## 6.1 Direct edge teacher

用 S2GAE / MaskGAE / MGAE 在 training graph 上做 masked-edge reconstruction：

[
T_{ij}^{edge}=p_T(A_{ij}=1\mid G_{\text{train}}^{visible},X)
]

S2GAE 的思路是随机 mask 一部分 edges，然后只重建 missing edges，而不是重建完整输入结构；S2GAE 论文也把它解释成 edge-level contrastive learning framework。 MGAE 则强调高比例 edge masking，例如 70%，以及 cross-correlation decoder。 MaskGAE 进一步强调 masked graph modeling，即 masking edges and reconstructing missing parts from partially visible structure。

## 6.2 Structural embedding teacher

从 training graph 学 node-level structural embedding：

[
t_i = \text{GraphTeacherEmbed}(i;G_{\text{train}})
]

可以用：

```text
S2GAE encoder embedding
MGAE encoder embedding
node2vec / DeepWalk
PPR / diffusion embedding
Laplacian positional embedding
DNE / graph-native PPI embedding if available
```

Student 学：

[
z_i^{struct}=f_\theta(x_i)\approx t_i
]

这个 loss 非常关键，因为它把监督从少量 pair BCE 扩展为 **每个 train protein 都有 dense structural target**：

[
\mathcal L_{struct-distill}
===========================

\sum_i \lVert z_i^{struct}-t_i\rVert_2^2
+
\mathcal L_{\text{contrast}}(z_i^{struct},t_i)
]

这比 pair-level teacher 强很多。

## 6.3 Diffusion/context teacher

从 training graph 中采样 random walks / PPR neighbors：

[
\mathcal C_i = \text{TopPPRNeighbors}(i)
]

训练：

[
z_i^{struct} \cdot z_j^{struct}
]

对 graph context neighbors 高，对 far negatives 低。这样模型学到的不只是 direct edge，而是 network-locality prior。

这很重要，因为 PRING 的 graph-level metrics 评估 sampled subgraph 的 degree、clustering、spectrum，而这些性质不是单边 BCE 能学出来的。PRING 评估代码确实对每个 sampled subgraph 计算 graph similarity、relative density、degree histogram MMD、clustering histogram MMD 和 normalized Laplacian spectral MMD。([GitHub][4])

---

# 7. Module 3：Factorized retrieval encoder

这是新方案的主干，借鉴 RaftPPI / FlashPPI。

对每个 protein 编码一次：

[
e_i^q=f_q(x_i,z_i^{struct})
]

[
e_i^k=f_k(x_i,z_i^{struct})
]

初始 retrieval score：

[
s_{ij}^{retr}
=============

\langle e_i^q,e_j^k\rangle
+
\lambda_{res}\langle e_i^{res},e_j^{res}\rangle
+
\lambda_{struct}\langle z_i^{struct},z_j^{struct}\rangle
+
\lambda_{mod} z_i^{module\top}Bz_j^{module}
+
\lambda_d(\hat d_i+\hat d_j)
]

其中 (e_i^{res}) 可以用 RaftPPI-style residue factorization：

[
\ell(A,B)\approx \langle \hat h_A,\hat h_B\rangle
]

RaftPPI 的关键就是把 residue-level Gaussian kernel interactions 用 random Fourier features 近似，再用 low-rank attention pool 成 per-protein embedding，从而支持 ANN retrieval。

训练时使用 InfoNCE / sampled softmax，而不是只用 BCE：

[
\mathcal L_{\text{retrieval}}
=============================

-\log
\frac{\exp(s_{ij}/\tau)}
{\sum_{k\in\mathcal N_i\cup{j}}\exp(s_{ik}/\tau)}
]

其中 ((i,j)\in E_{\text{train}})。

这一步会把一个 positive edge 和 batch 内大量 negatives 对比，因此每个 batch 的监督密度从 “少量 BCE pairs” 变成 “dense similarity matrix”。FlashPPI 正是用 dual encoder + InfoNCE，把 PPI prediction 变成 dense retrieval，并用 false negative masking 避免把 batch 中潜在真实 interactions 当成负例。

---

# 8. Module 4：False-negative masking + adaptive hard negative weighting

PPI non-edge 不是可靠 negative。RaftPPI 和 FlashPPI 都非常强调 negative construction 问题。

RaftPPI 指出 experimentally confirmed non-interactions 很少，构造 negatives 的质量差异很大；它使用 adaptive negative weighting，让 harder negatives 获得更大权重。 FlashPPI 用 false negative masking 防止 batch 内真实 interaction 被当作 negative，并用 contrastive embedding space 做 online hard negative mining，把高相似但非交互 pairs 用来训练 contact head。

因此新训练策略应是：

```text
Positive:
  observed train edges

Do not treat all unobserved pairs as true negatives.

Negative pools:
  random negatives
  degree-matched negatives
  sequence-similarity-matched negatives
  same-module but unobserved hard negatives
  retrieval top false positives from previous epoch

False-negative masking:
  if pair is known edge, graph-context positive, same complex/pathway if allowed, or high teacher edge probability:
      mask out from negative denominator
```

Adaptive negative weighting：

[
w_{ij}^{neg}
============

\frac{\exp(\alpha s_{ij})}
{\sum_{(u,v)\in N}\exp(\alpha s_{uv})}
]

[
\mathcal L_{neg}
================

-\sum_{(i,j)\in N}w_{ij}^{neg}\log \sigma(-s_{ij})
]

这比当前 BCE supervised subset 更适合你的 `bce_supervised_fraction=0.063` 场景。

---

# 9. Module 5：Candidate re-ranker

Retrieval encoder 负责 high-recall localization；re-ranker 负责 precision。

测试时：

```text
for each protein i:
    retrieve top K partners by ANN using s_retr
candidate set C_retr = union of all retrieved pairs
C_eval = C_retr ∩ all_test_ppi
only re-rank C_eval
```

Re-ranker score：

[
s_{ij}^{rank}
=============

g_\phi([
h_i,h_j,
h_i\odot h_j,
|h_i-h_j|,
z_i^{struct},z_j^{struct},
z_i^{module},z_j^{module},
\hat d_i,\hat d_j,
s_{ij}^{retr}
])
]

如果 compute 允许，可以对 `C_eval` 的 top candidates 加一个更强的 pair-context model：

```text
Option A: MINT-style cross-chain attention teacher / re-ranker
Option B: PLM-Interact teacher / re-ranker
Option C: FlashPPI-style contact head
```

MINT 的价值在于它不是普通 single-chain PLM，而是在 STRING-DB 的大量 physical PPIs 上训练，通过 cross-chain attention 让 token representation 受 interacting sequences 影响；论文中说它从 STRING 过滤得到 96M high-quality PPIs，并在 PPI tasks 上优于通用 PLM。 但 MINT 是 pair-contextual，不适合作为全枚举 scorer；更适合作为 **retrieved candidate re-ranker 或 distillation teacher**。

---

# 10. Module 6：PRING graph assembly

PRING 需要 binary labels，不需要 calibrated probability 本身。因此 assembly 应从“fixed threshold=0.5”改为：

```text
retrieval top-K
    ↓
re-rank candidate pairs
    ↓
select graph edges by validation-calibrated policy
    ↓
write all_test_ppi_pred.txt
```

推荐三种 assembly policy：

## 10.1 Global validation-density budget

从 validation reconstruction plan 估计 density：

[
\rho_{\text{val}}=\frac{|E_{\text{val}}|}{|C_{\text{val}}|}
]

测试时选择：

[
\hat m = \rho_{\text{val}}\cdot |C_{\text{test}}|
]

然后取 top-(\hat m) candidates。

## 10.2 Per-node degree prior budget

Student 预测每个 protein 的 expected degree：

[
\hat d_i = f_d(x_i,z_i^{struct})
]

graph assembly 变成 constrained b-matching：

[
\max_{\hat A}
\sum_{(i,j)\in C_{\text{retr}}}s_{ij}^{rank}\hat A_{ij}
]

subject to:

[
\sum_j \hat A_{ij}\approx \hat d_i
]

这比 global top-K 更能控制 artificial hubs。

## 10.3 Hybrid

实际最推荐：

```text
global edge count budget
+ per-node soft degree cap
+ no self loops
+ symmetric graph
```

输出格式：

```text
uniprot_id1 uniprot_id2 label
```

对未被 retrieval/re-ranker 选中的 `all_test_ppi` pairs，label 写 0。PRING 的 `eval.py` 只会把 label > 0 的 pairs 加入 predicted graph。([GitHub][4])

---

# 11. 新 training algorithm

## Stage A：训练 graph prior teacher

```python
# train graph only, no test topology
G_train = load_graph("human_train_graph.pkl")
X_train = load_esm_embeddings(train_proteins)

teacher = S2GAE_or_MaskGAE()

for epoch in range(E_teacher):
    visible_edges, masked_edges = mask_edges_or_paths(G_train)

    z_T = teacher.encoder(X_train, visible_edges)

    pos = masked_edges
    neg = sample_hard_negatives(
        G_train,
        degree_matched=True,
        same_component=True,
        same_module=True
    )

    loss_edge = masked_edge_reconstruction_loss(z_T, pos, neg)
    loss_degree = degree_auxiliary_loss(z_T, degree(G_train))
    loss_context = graph_context_contrastive_loss(z_T, random_walk_contexts)

    loss = loss_edge + lambda_degree * loss_degree + lambda_context * loss_context
    loss.backward()
    optimizer.step()

save frozen_teacher
save teacher_structural_embeddings t_i
save teacher_edge_scores for selected train candidate pairs
```

这里的重点不是 teacher 本身，而是生成：

```text
t_i: node structural embedding target
T_ij: edge prior score
C_i: graph context positives
H_i: hard negatives
```

---

## Stage B：训练 student retrieval encoder

```python
for batch_edges in sample_positive_edges(E_train):
    # each item is an observed train PPI edge (i, j)
    proteins = unique_nodes(batch_edges)
    X = esm_embeddings[proteins]

    out = student.encode_proteins(X)
    z_q, z_k = out.query, out.key
    z_struct = out.struct
    z_res = out.residue_factorized
    z_mod = out.module
    d_hat = out.degree

    # dense batch similarity matrix
    S = score_all_queries_against_all_targets(
        z_q, z_k, z_res, z_struct, z_mod, d_hat
    )

    mask = false_negative_mask(
        known_train_edges=E_train,
        teacher_high_score_pairs=T_high,
        graph_context_pairs=C_context
    )

    loss_retrieval = info_nce_loss(S, positives=batch_edges, mask=mask)

    loss_struct = mse_or_contrastive(z_struct, teacher_struct_embeddings)
    loss_degree = mse(log1p(d_hat), log1p(train_degree))
    loss_teacher = distill_edge_scores(S, teacher_edge_scores_on_sampled_pairs)

    loss = (
        loss_retrieval
        + lambda_struct * loss_struct
        + lambda_degree * loss_degree
        + lambda_teacher * loss_teacher
    )

    loss.backward()
    optimizer.step()
```

这个阶段直接解决当前最大问题：**模型先学 retrieval localization，而不是在 6.3% BCE supervised pairs 上学 dense matrix reconstruction。**

---

## Stage C：online hard negative mining

```python
for epoch in range(E_hard):
    # Build ANN index over train proteins
    embeddings = student.encode_all(train_proteins)
    index = build_faiss_index(embeddings.key)

    hard_negatives = []
    for i in train_proteins:
        retrieved = index.search(embeddings.query[i], K=K_train)

        for j in retrieved:
            if (i, j) not in E_train and not false_negative(i, j):
                hard_negatives.append((i, j))

    train retrieval + reranker on:
        positives = E_train
        negatives = hard_negatives + random_negatives
```

FlashPPI 的 hard negative mining 正是这个思想：contrastive head 先找到 embedding-similar hard negatives，再把它们用于 contact head / pair head 训练。

---

## Stage D：训练 candidate re-ranker

```python
for batch in sampled_candidate_pairs:
    # positives: observed train edges
    # negatives: online hard negatives
    features = build_pair_features(batch)

    score = reranker(features)

    loss_pair = adaptive_weighted_bce(score, labels)
    loss_rank = pairwise_margin_loss(score_pos, score_hard_neg)

    # optional teacher distillation from MINT/PLM-Interact/FlashPPI contact head
    loss_mint = distill_pair_teacher(score, mint_teacher_score)

    loss = loss_pair + lambda_rank * loss_rank + lambda_mint * loss_mint
    loss.backward()
```

---

# 12. PRING inference algorithm

```python
def pring_inference(test_proteins, all_test_ppi):
    # 1. encode each protein once
    Z = student.encode_all(test_proteins)

    # 2. build ANN index
    index = build_ann_index(Z.key)

    # 3. retrieve top-K candidate partners per query
    C_retr = set()
    for i in test_proteins:
        nbrs = index.search(Z.query[i], K=K_test)
        for j in nbrs:
            if i != j:
                C_retr.add(canonical_pair(i, j))

    # 4. align with PRING candidate universe
    C_eval = C_retr.intersection(all_test_ppi)

    # 5. re-rank only retrieved candidates
    scores = {}
    for (i, j) in C_eval:
        scores[(i, j)] = reranker_score(i, j, Z)

    # 6. graph assembly
    selected_edges = assemble_graph(
        scores,
        edge_budget="validation_density_or_degree_prior",
        degree_cap=True
    )

    # 7. write PRING file
    with open("all_test_ppi_pred.txt", "w") as f:
        for (i, j) in all_test_ppi:
            label = 1 if (i, j) in selected_edges else 0
            f.write(f"{i} {j} {label}\n")
```

这完全符合 PRING：`all_test_ppi_pred.txt` 仍覆盖所有 pairs；只是模型只对 retrieved candidates 做 expensive scoring，其他 pairs 默认 background label 0。

---

# 13. 为什么这个方案比当前 TCCIG 更适合你的结果

你的 P4 表明：

```text
oracle-density graph_sim 也崩
top-K precision 比随机还差
AUPRC 0.010
AUROC 0.363
```

所以主故障是：

```text
candidate localization / global ranking wrong
```

新方案直接把训练主目标改成：

```text
retrieve true interactors into top-K
```

而不是：

```text
make full soft adjacency look graph-like
```

训练监控也要改。最先看：

```text
Retrieval Recall@K%
Precision@K
AUPRC over retrieved+hard-negative candidate universe
PRING graph_sim after assembly
relative_density
degree MMD
```

不要先看 fixed-threshold BCE metrics。RaftPPI 也是把 proteome-level evaluation 设成 Recall@K%，例如 Recall@1/3/5/10/20%，因为这才对应 large candidate screening。

---

# 14. 更新后的 validation / checkpoint selection

新的 checkpoint monitor 不应是 topology distribution loss，也不应是 BCE。建议：

[
\text{Score}
============

0.45\cdot \text{RetrievalRecall@K}
+
0.25\cdot \text{AUPRC}_{candidate}
+
0.20\cdot \text{GraphSim}
-------------------------

## 0.05|\log(\text{RelativeDensity})|

## 0.03\cdot \text{DegreeMMD}

0.02\cdot \text{ClusteringMMD}
]

其中：

```text
RetrievalRecall@K:
  on validation all-pair candidate universe or validation reconstruction plan

AUPRC_candidate:
  on retrieved + hard negatives, not only balanced test pairs

GraphSim:
  PRING-style reconstructed validation graph

RelativeDensity / DegreeMMD / ClusteringMMD:
  PRING topology diagnostics
```

如果 validation 没有 `all_val_ppi`，你需要构造一个 PRING-like validation candidate universe：

```text
all_val_ppi = all unordered pairs among validation proteins
or sampled all-pairs with realistic class imbalance
```

而不是继续用 balanced `human_val_ppi`。

---

# 15. 代码层面的重构方向

不是修 bug，而是换训练主干。

## 15.1 保留

```text
ESM embedding cache
PRING topology_evaluate wrapper
all_test_ppi_pred writer
topology metrics logging
```

## 15.2 弃用或降级

```text
TCCIG forward_graph all-pairs sampled subgraph training
online MGAE teacher inside student step
fixed threshold = 0.5
dense all-pairs graph assembly as primary path
```

## 15.3 新增

```text
model/graph_prior_retrieval.py
train/graph_prior_teacher.py
train/retrieval_train.py
train/hard_negative_mining.py
eval/pring_retrieval_eval.py
```

核心类：

```python
class GraphPriorRetrievalPPI(nn.Module):
    def encode_proteins(self, protein_embeddings, residue_embeddings=None):
        return {
            "query": q,
            "key": k,
            "struct": z_struct,
            "residue": z_res,
            "module": z_module,
            "degree": d_hat,
        }

    def retrieval_score(self, encoded_i, encoded_j):
        ...

    def rerank_score(self, encoded_i, encoded_j):
        ...
```

---

# 16. 实验队列

## Run R0：ESM dot-product retrieval baseline

目的：确认 PRING candidate universe 上的 retrieval difficulty。

```text
embedding = ESM mean pool
score = cosine / dot
retrieve top-K
assemble by validation density
```

成功标准：

```text
Recall@20% > random baseline
graph_sim > current P4
```

如果这个 baseline 都比 TCCIG 强，说明 TCCIG decoder 已经破坏了 ESM geometry。

## Run R1：RaftPPI-style factorized residue retriever

加入：

```text
RFF Gaussian kernel
low-rank residue attention
dot-product retrieval
adaptive negative weighting
```

成功标准：

```text
Retrieval Recall@K 明显高于 R0
AUPRC_candidate 提升
PRING graph_sim 至少回到 P1/P2 水平以上
```

## Run R2：Graph prior structural distillation

加入：

```text
offline S2GAE / MaskGAE teacher
z_struct regression / contrastive distillation
degree prior head
graph context contrastive loss
```

成功标准：

```text
Recall@K 提升
degree MMD 降低
relative_density 更接近 1
```

## Run R3：Online hard negative mining

加入：

```text
ANN top-K train hard negatives
false-negative masking
adaptive negative weighting
```

成功标准：

```text
Precision@K 提升
AUPRC_candidate 提升
graph_sim 提升
```

## Run R4：Candidate re-ranker

加入：

```text
pair MLP re-ranker
optional MINT / PLM-Interact teacher distillation on retrieved candidates only
```

成功标准：

```text
top-K precision 提升
graph_sim 提升
不牺牲 Recall@K 太多
```

## Run R5：Graph assembly calibration

比较：

```text
global validation density top-K
degree-prior constrained top-K
hybrid global + per-node degree cap
```

成功标准：

```text
relative_density 0.8–1.2
degree MMD 下降
graph_sim 不下降
```

---

# 17. 关键 stop rules

下一阶段不要等完整 PRING eval 才发现模型坏了。每轮必须先看 retrieval diagnostics：

```text
Stop if:
  Retrieval Recall@20% ≤ random baseline
  top-K precision below random-density expectation
  validation graph_sim < ESM dot-product baseline
  candidate recall is low even when K is large
  hard negatives dominate positives but loss keeps下降，AUPRC 不升
```

尤其是你现在 P4 这种情况：

```text
oracle-density graph_sim 也很差
```

这就说明 assembly/budget 不是主因，应该立即回到 retrieval localization，而不是继续调 density/threshold。

---

# 18. 最终建议

现在应停止继续强化当前 TCCIG 的 dense all-pairs graph generator。它和 PRING 的真实困难不匹配。

新的主线应该是：

```text
Graph-prior teacher on human_train_graph
        ↓
student learns feature → structural prior embedding
        ↓
dual-encoder / RaftPPI-style factorized retrieval
        ↓
InfoNCE + false-negative masking + adaptive hard negatives
        ↓
candidate-only re-ranking with optional MINT/FlashPPI-style pair/context teacher
        ↓
validation-calibrated sparse graph assembly
        ↓
PRING all_test_ppi_pred.txt
```

一句话总结：

> **把“重建 adjacency matrix”改成“学习 training graph prior 下的 proteome-scale interaction retrieval”。PRING 最终仍然评估 reconstructed graph，但模型优化的第一目标必须是把真实 edges retrieve 到 candidate set 里；否则 graph-level topology loss 和 threshold calibration 都是在错误边集上做后处理。**

[1]: https://github.com/SophieSarceau/PRING/blob/main/topology_task/intra_species.md "PRING/topology_task/intra_species.md at main · SophieSarceau/PRING · GitHub"
[2]: https://github.com/SophieSarceau/PRING/blob/main/topology_task/cross_species.md "PRING/topology_task/cross_species.md at main · SophieSarceau/PRING · GitHub"
[3]: https://github.com/SophieSarceau/PRING/blob/main/data_process/data_format.md "PRING/data_process/data_format.md at main · SophieSarceau/PRING · GitHub"
[4]: https://github.com/SophieSarceau/PRING/blob/main/topology_task/eval.py "PRING/topology_task/eval.py at main · SophieSarceau/PRING · GitHub"


你的判断是对的：现在这个方向的主要问题已经不是 code bug，也不是 sigmoid saturation，而是 **task formulation 和 architecture formulation 错了**。更准确地说，当前 TCCIG 把问题建模成：

[
X_S \rightarrow P_{|S|\times |S|} \rightarrow \text{global top-}m
]

但 PRING 的实际任务更像：

[
\text{train PPI graph prior} + \text{protein intrinsic features}
\rightarrow
\text{candidate retrieval / ranking}
\rightarrow
\text{binary reconstructed graph}
]

而不是“对 transcriptome/proteome 级别全 pair 空间做一个弱监督 adjacency generator”。

下面是我更新后的判断和方案。

---

# 1. 重新对齐 PRING 的任务定义

PRING 的 README 明确说它不是只做 isolated pair prediction，而是从 network-level perspective 评估 PPI prediction；任务分为 topology-oriented 和 function-oriented 两大类，其中 topology-oriented 包括 intra-species 和 cross-species PPI network generation，function-oriented 包括 protein complex pathway prediction、GO enrichment 和 essential protein justification。([GitHub][1])

## 1.1 Intra-species human setting

PRING 的 intra-species 文档定义得很清楚：

训练使用：

```text
human_train_ppi.txt
human_val_ppi.txt
human_simple.fasta
```

测试阶段 **不是只在 `human_test_ppi.txt` 上评 pairwise classifier**，而是要求模型对：

```text
all_test_ppi.txt
```

中的 pairs 做预测，然后用预测结果重建完整 test graph。`human_test_ppi.txt` 只是 binary classification diagnostic；真正 graph reconstruction 使用的是 `all_test_ppi.txt`。([GitHub][2])

PRING evaluator 要求输出：

```text
uniprot_id1 uniprot_id2 label
```

其中 `label=1` 表示把这条边放入 predicted graph，`label=0` 表示不放入 predicted graph。然后 evaluator 用 `human_test_graph.pkl` 和 `test_sampled_nodes.pkl` 计算 graph-level metrics。([GitHub][2])

## 1.2 Cross-species setting

Cross-species setting 更严格：模型用 human PPI network 训练，然后直接在 ARATH / YEAST / ECOLI 的 `*_all_test_ppi.txt` 上做 inference，重建目标物种 test graph；BFS / DFS / Random Walk sampling 只用于 evaluation subgraphs，不允许作为模型输入。([GitHub][3])

这意味着新的任务定义应是：

> **Given a PRING candidate-pair universe (C_{\text{test}}), predict a binary edge set (\hat E \subseteq C_{\text{test}}), using only protein intrinsic features and graph priors learned from training PPI graphs.**

不是：

> Generate or rank the full (O(n^2)) pair space from scratch.

也不是：

> Use target graph topology to build node embeddings.

## 1.3 PRING graph similarity 是 identity-aware edge-overlap，不是 generic topology score

PRING `eval.py` 的 `compute_graph_similarity` 本质是：

[
\text{GS}
=========

1-\frac{\lVert A_{\text{pred}}-A_{\text{gt}}\rVert_1}
{\sum A_{\text{pred}}+\sum A_{\text{gt}}}
]

对简单无向图近似等价于：

[
\text{GS}
=========

\frac{2TP}{2TP+FP+FN}
]

也就是 edge-level F1 / Dice-like overlap，只是在 sampled subgraph 上计算。PRING 还计算 relative density、degree distribution MMD、clustering coefficient MMD、spectral MMD。([GitHub][4])

这解释了 P4 的失败：P4 的 density / distribution 可能没有完全崩，但 **top-ranked edges 的 identity 错了**。一旦 top-K 选不到真实边，graph_sim 会直接掉。degree / clustering / spectral MMD 不能补救 edge localization 错误。

---

# 2. 当前 TCCIG 的核心设计问题

## 2.1 BCE supervised fraction 太低，full-candidate ranking 没有被训练

你现在的 `bce_supervised_fraction=0.063` 意味着大部分 candidate pairs 没有直接监督。模型可以把少量 supervised BCE pairs 拟合好，但完全学不到：

[
\text{对于 anchor protein } i,\quad \text{哪些 candidates 应排在它的真实 interactors 前面？}
]

这正是 P4 的症状：64,404 条边只命中约 451 条真边，比按测试图密度随机选还差。这个不是 calibration 问题，而是 **retrieval localization failure**。

## 2.2 topology teacher 作为 auxiliary loss 太弱

现在把 GSL / MGAE teacher 当成附加 teacher signal，本质还是：

[
\text{student pair scores} \leftarrow \text{teacher soft labels on sampled pairs}
]

这没有充分利用训练 PPI graph 的结构。训练图里真正有价值的是：

* 每个 protein 的 observed neighborhood；
* 多跳 random-walk / PPR proximity；
* functional module / community structure；
* topological role / hubness；
* graph diffusion prior；
* hard negatives：相似但不连接、同模块但无边、high-score non-edge。

如果 teacher 只给 sampled candidate pair 一个 soft probability，它没有把这些结构转成强监督。

## 2.3 full adjacency generator 不适合 PRING candidate setting

TCCIG 试图从 protein set 直接生成 dense / all-pairs probability matrix，然后 global top-(\hat m)。但 PRING 已经给了 `all_test_ppi.txt` candidate universe。严格对齐 PRING 时，模型应做的是：

```text
score / rank PRING candidate pairs
select positives
write all_test_ppi_pred.txt
```

不是在外部自由生成任意 graph，也不是必须 materialize full (n^2) matrix。

更重要的是，现代 proteome-scale PPI retrieval 论文都在避免 all-pairs explicit scoring。RaftPPI 的出发点就是 residue-level pair models 很准但 (O(N^2L^2)) 过慢，因此把 residue-level interaction factorize 成 per-protein indexable embeddings，用 ANN search 替代 exhaustive scoring；它在 human proteome 上从约 200M candidate pairs 中检索 top 20% 只需分钟级。  FlashPPI 也把 PPI prediction 改写成 dense retrieval，先 top-k retrieval，再对候选做 contact-based reranking。

所以新方案应该从 **graph generator** 改为 **graph-prior retrieval/reranking model**。

---

# 3. 更新后的任务定义：PRING-aligned graph-prior candidate ranking

我建议把任务重新定义为：

## 3.1 Training task

给定 human training graph：

[
G_{\text{train}}=(V_{\text{train}},E_{\text{train}})
]

和 intrinsic protein features：

[
X_{\text{train}}={x_i}*{i\in V*{\text{train}}}
]

学习一个 scorer：

[
s_\theta(i,j\mid x_i,x_j,\Pi_{\text{train}})
]

其中 (\Pi_{\text{train}}) 是从 training PPI graph 学到的 **graph prior parameters**，不是 test graph topology。

[
\Pi_{\text{train}}
==================

{\text{graph embedding prior},\text{module prior},\text{degree/role prior},\text{diffusion prior},\text{candidate retrieval prior}}
]

训练目标不是“对所有 missing pairs 做 BCE”，而是：

[
\text{observed train edges should be ranked above plausible hard non-edges}
]

也就是 retrieval / ranking objective。

## 3.2 Inference task

给定 PRING candidate universe：

[
C_{\text{test}}=\texttt{all_test_ppi.txt}
]

和 protein features：

[
X_{\text{test}}
]

输出：

[
\hat y_{ij}\in{0,1},\quad (i,j)\in C_{\text{test}}
]

写成：

```text
uniprot_id1 uniprot_id2 label
```

用于 PRING `eval.py`。

模型可以只显式 score 一个 retrieval shortlist：

[
R_{\text{test}}\subset C_{\text{test}}
]

对于未进入 shortlist 的 pairs：

[
\hat y_{ij}=0
]

这样既符合 PRING output format，又避免 full-candidate exhaustive ranking。

---

# 4. 主推新架构：Graph-Prior Retrieval and Reranking for PRING

我建议把模型改成：

# **GPR-PPI: Graph-Prior Protein Interaction Retriever**

整体结构：

```text
Training PPI graph only
        ↓
Graph-prior pretraining / structure learning
        ↓
Graph-prior distillation into feature encoder
        ↓
Dual-encoder candidate retrieval
        ↓
Optional residue/contact reranker on retrieved pairs
        ↓
PRING all_test_ppi_pred.txt
```

---

# 5. Module 1：Graph-prior pretraining，不再只是 teacher

这一部分要从“teacher auxiliary loss”升级为“训练图结构先验学习器”。

## 5.1 输入

只用 training graph：

[
G_{\text{train}}
]

不能用 validation/test graph。

## 5.2 学到的 graph prior

从 (G_{\text{train}}) 中学习四类 target：

### A. Graph-aware node embedding

用 S2GAE / MaskGAE / GSR / node2vec / PPR-SVD 在 train graph 上学：

[
z_i^G
]

这个 embedding 表示 protein 在 training interactome 中的 graph-context role。

Masked-edge family 仍然有用，但角色改变：不再只是给 sampled edges soft labels，而是学习 **graph prior representation**。MGAE / S2GAE 通过 masked-edge reconstruction 学结构；MaskGAE 进一步强调 masked graph modeling 对 GAE representation 的改进；Bandana 提醒 binary edge deletion 可能破坏 message flow，可以用 continuous bandwidth masking 增强 topology representation。

### B. Diffusion / random-walk neighborhood distribution

对每个 node (i)，构造 train graph 上的 soft neighbor distribution：

[
\pi_i^G(j)
==========

\lambda_1 A_{ij}
+
\lambda_2 \text{PPR}*{ij}
+
\lambda_3 \text{RW}*{ij}
]

这里 (\pi_i^G) 不是 test graph 信息，只是 training graph prior。

### C. Module / community prior

在 (G_{\text{train}}) 上做 Louvain / Leiden / NOCD / DMoN-like overlapping community detection：

[
c_i^G \quad \text{or} \quad q_i^G\in[0,1]^K
]

再训练 student 从 intrinsic features 预测 module membership：

[
\hat q_i=f_\theta(x_i)
]

你已有资料里也指出，GNN4DM / ModulePred 这类 biology-facing work 的价值在于 functional modules、overlap、graph augmentation，比普通 pairwise PPI 更接近 PRING 的 function-oriented goals。

### D. Topological role prior

预测：

[
r_i^G = [\log(1+d_i),\text{local clustering},\text{k-core},\text{PageRank},\text{betweenness proxy}]
]

这些只作为 training targets，不作为 test inputs。

---

# 6. Module 2：Protein feature encoder

输入 ESM / ESM2 / structure-token / MINT-like features：

[
x_i
]

输出两套 embeddings：

[
q_i=f_Q(x_i), \quad k_i=f_K(x_i)
]

用于 retrieval：

[
s_{\text{ret}}(i,j)=\frac{q_i^\top k_j}{\tau}
]

对于 undirected PPI，最终可以 symmetrize：

[
s_{\text{ret}}^{sym}(i,j)=
\frac{1}{2}
\left(
q_i^\top k_j+q_j^\top k_i
\right)
]

这里应采用 **dual encoder / factorized scorer**，而不是 pair MLP。原因是：

* pair MLP 需要显式 pair scoring；
* dual encoder 可以 ANN retrieval；
* in-batch contrastive training能提供密集监督；
* 更符合 RaftPPI / FlashPPI 的 proteome-scale retrieval范式。

RaftPPI 的核心启示是：保留 residue-level interaction inductive bias，但把它 factorize 成可索引的 per-protein embedding；FlashPPI 的核心启示是：先用 contrastive embedding space 做 retrieval，再用 contact head rerank hard candidates。 

---

# 7. Module 3：Graph-prior distillation heads

Student 不应只输出 pair score。它还要从 intrinsic features 预测 training graph prior：

## 7.1 Graph embedding alignment

[
\mathcal{L}_{embed}
===================

1-\cos(h_i,z_i^G)
]

或者 InfoNCE：

[
\mathcal{L}_{embed}
===================

-\log
\frac{\exp(\cos(h_i,z_i^G)/\tau)}
{\sum_j \exp(\cos(h_i,z_j^G)/\tau)}
]

## 7.2 Neighborhood distribution distillation

对 batch 中 nodes (B)，令 student scores：

[
p_\theta(j\mid i)
=================

\text{softmax}*{j\in B}(s*\theta(i,j))
]

teacher graph prior：

[
\pi_i^G(j)
]

训练：

[
\mathcal{L}_{neighbor}
======================

\sum_i
\text{KL}\left(
\pi_i^G(\cdot\mid B)
\parallel
p_\theta(\cdot\mid i,B)
\right)
]

这一步非常关键。它把 training graph 的多跳结构变成 dense listwise supervision，而不是 0.063 fraction 的 sparse BCE。

## 7.3 Module distillation

[
\mathcal{L}_{module}
====================

\text{BCE/KL}(\hat q_i,q_i^G)
]

然后 pair score 里加入 module compatibility：

[
s_{\text{module}}(i,j)=\hat q_i^\top B \hat q_j
]

但这个 term 要作为 retrieval score 的 small additive bias，而不是像 P4 那样把 logits 推成高常数带。

## 7.4 Role / degree prior

[
\mathcal{L}_{role}
==================

\left|
\hat r_i-r_i^G
\right|_2^2
]

这让模型从 intrinsic features 学会“哪些 protein 更像 hub / scaffold / peripheral node”，但不在 test 输入 target degree。

---

# 8. Module 4：Retrieval-first pair scoring

最终 score 不再是当前 TCCIG 的：

[
\text{pair MLP}+\text{hub}+\text{lowrank}+\text{module}+\text{density}
]

而是：

[
s(i,j)
======

s_{\text{ret}}(i,j)
+
\alpha s_{\text{module}}(i,j)
+
\beta s_{\text{role}}(i,j)
+
\gamma s_{\text{graph-prior}}(i,j)
]

其中主项必须是 retrieval score：

[
s_{\text{ret}}(i,j)=q_i^\top k_j
]

因为 PRING graph_sim 需要正确 edge localization，不能靠 density / clustering MMD 事后修。

如果想保留 residue-level signal，可以引入 Raft-style factorized residue score：

[
h_i^{raft}=\sum_r w_{ir}\psi(z_{ir})
]

[
s_{\text{raft}}(i,j)=
\langle h_i^{raft},h_j^{raft}\rangle
]

这比 pair MLP 更适合大规模 candidate retrieval。RaftPPI 显示这种结构可以近似 residue-level interaction，同时生成可 ANN 检索的单蛋白 embedding。

---

# 9. Module 5：Optional reranker，只对 retrieved candidates 运行

Stage 1 retrieval 生成：

[
R_i=\text{TopR}_{j}(q_i^\top k_j)
]

只对：

[
(i,j)\in R=\bigcup_i R_i
]

运行更强 reranker。

Reranker 可以是：

* MINT-like cross-chain attention；
* FlashPPI-like contact head；
* residue-level contact map scorer；
* small pairwise MLP using retrieval features + ESM features + module compatibility。

MINT 的启示是：普通 PLM 单独编码 protein 会丢失 interaction-specific context；它通过 cross-chain attention 和 STRING-derived PPI corpus 预训练来学习 PPI language。 但 MINT-style cross-attention 不适合跑全 (O(n^2)) pairs，所以应该作为 **retrieved-candidate reranker**，不是 primary all-pairs scorer。

FlashPPI 也采用类似思想：dense retrieval 后再做 contact-based verification，并用 online hard negative mining 让 contact head 学会区分 deceptive non-interacting pairs。

---

# 10. 训练目标：从 BCE 改成 listwise retrieval + graph-prior distillation

总 loss：

[
\mathcal{L}
===========

\mathcal{L}*{retrieval}
+
\lambda_1\mathcal{L}*{neighbor}
+
\lambda_2\mathcal{L}*{embed}
+
\lambda_3\mathcal{L}*{module}
+
\lambda_4\mathcal{L}*{role}
+
\lambda_5\mathcal{L}*{rerank}
+
\lambda_6\mathcal{L}_{hardneg}
]

## 10.1 Query-wise supervised contrastive retrieval

对 batch 中 positive train edges：

[
E_B^+={(u_i,v_i)}
]

构造 score matrix：

[
S_{ij}=q_{u_i}^\top k_{v_j}
]

训练：

[
\mathcal{L}_{retrieval}
=======================

-\frac{1}{|B|}
\sum_i
\log
\frac{\exp(S_{ii}/\tau)}
{\sum_j \exp(S_{ij}/\tau)}
]

如果 batch 中有多个 valid positives，则用 multi-positive supervised contrastive loss：

[
\mathcal{L}_{retrieval}
=======================

-\sum_i
\log
\frac{\sum_{j\in P(i)}\exp(S_{ij}/\tau)}
{\sum_{j\in B\setminus \text{masked false negs}}\exp(S_{ij}/\tau)}
]

这一步把每个 batch 变成 (B\times B) dense ranking supervision，而不是 6.3% BCE coverage。

## 10.2 False negative masking

PPI 中 unknown pair 不是可靠 negative。需要 mask known positives：

```text
if (i,j) in E_train:
    do not treat as negative
```

还可以 mask high-confidence graph-prior positives：

```text
if PPR(i,j) high or same known complex/module:
    downweight as negative
```

FlashPPI 在 contrastive training 里也使用 false negative masking，避免把 batch 中潜在 valid interactions 当成 negatives。

## 10.3 Adaptive negative weighting

RaftPPI 的 adaptive negative weighting 对当前问题很重要，因为你的模型显然学坏了 hard region。RaftPPI 用 model confidence 给 harder negatives 更高权重；当 (\tau=0) 时退化为 balanced BCE，(\tau\to\infty) 时聚焦 hardest negative。

你这里可以做：

[
w_{ij}^{neg}
============

\text{stopgrad}
\left[
\frac{\exp(s_{ij}/\tau)}
{\sum_{k\in N(i)}\exp(s_{ik}/\tau)}
\right]
]

然后：

[
\mathcal{L}_{hardneg}
=====================

-\sum_{(i,j)\in N}
w_{ij}^{neg}\log\sigma(-s_{ij})
]

## 10.4 Graph-prior neighbor distillation

这是真正解决 “GSL teacher 监督不够强” 的关键。

对每个 anchor (i)，不要只问 “pair (i,j) 是不是边”，而是问：

[
\text{模型给 batch 中所有 candidate proteins 的排名是否像 train graph diffusion prior？}
]

[
\mathcal{L}_{neighbor}
======================

\text{KL}
\left(
\pi_i^G(\cdot)
\parallel
\text{softmax}(s_\theta(i,\cdot))
\right)
]

这会把 PPI graph prior 变成 listwise dense signal。

## 10.5 Reranker loss

只在 retrieved / hard candidate pairs 上做：

[
\mathcal{L}_{rerank}
====================

\text{FocalBCE}(s_{\text{rerank}}, y)
]

负例来自：

* ANN high-score non-edges；
* same-module non-edges；
* PLM-similar non-edges；
* self-negatives；
* degree-matched negatives。

FlashPPI 的 online hard negative mining 正是用 contrastive embedding 找 high-similarity non-interacting pairs，再用 contact head 学会拒绝它们。

---

# 11. PRING-aligned inference

## 11.1 Intra-species inference

Input:

```text
human/BFS/all_test_ppi.txt
human_simple.fasta
trained model
```

Steps:

```python
# 1. Load unique proteins appearing in all_test_ppi
V_test = unique_nodes(all_test_ppi)

# 2. Encode each protein once
q, k, module, role = encoder(ESM(V_test))

# 3. ANN retrieval
R = {}
for protein i:
    R_i = ANN_top_R(q_i, K_index)
    R.update((i,j) for j in R_i)

# 4. PRING candidate filter
C_pring = set(all_test_ppi pairs)
R = R ∩ C_pring

# 5. Optional rerank only R
scores = reranker(R)

# 6. Select positives
E_hat = select_by_validation_rule(scores, R)

# 7. Write all_test_ppi_pred
for (i,j) in C_pring:
    label = 1 if (i,j) in E_hat else 0
    write(i,j,label)
```

这里没有对 all (n^2) 枚举空间排序。即使 `all_test_ppi.txt` 是完整 candidate universe，你也只真正 score retrieval shortlist；其他 pairs 直接 label 0。

## 11.2 Cross-species inference

训练仍然只用 human train graph。对 ARATH / YEAST / ECOLI：

```text
encode target species proteins
ANN retrieval within target species proteins
intersect with species all_test_ppi
apply human-validation-selected R / threshold / per-node budget
write *_all_test_ppi_pred.txt
```

不能用 target species graph 做 calibration。

---

# 12. Validation / model selection 必须改

当前 monitor 选 “density/distribution 更像”，但 P4 证明这会选出 edge localization 很差的 checkpoint。

新的 primary validation metrics：

## 12.1 Candidate recall@R

在 validation candidate universe 上：

[
\text{CandidateRecall@R}
========================

\frac{|E_{\text{val}}\cap R_{\text{val}}|}
{|E_{\text{val}}|}
]

这是第一关。如果 retrieval shortlist 覆盖不了真边，reranker 没救。

## 12.2 PRING graph_sim@selection

用 validation graph 构造 PRING-like eval：

[
\text{GraphSim@R/threshold}
]

而不是只看 density / degree / clustering / spectral。

因为 PRING graph similarity 本质是 edge F1-like overlap，必须把它作为 checkpoint monitor。([GitHub][4])

## 12.3 AUPRC on all validation candidates

这个要用 graph-mode / retrieval-mode scores，而不是 pairwise `forward()` scores。

## 12.4 Secondary metrics

* relative density；
* degree MMD；
* clustering MMD；
* spectral MMD。

这些是 secondary constraints，不是 primary monitor。

推荐 composite：

[
\text{Monitor}
==============

\text{GraphSim}*{val}
+
0.5\cdot \text{AUPRC}*{val}
+
0.2\cdot \text{CandidateRecall@R}
---------------------------------

## 0.05|\log(\text{RelDensity})|

0.01(\text{DegMMD}+\text{CCMMD}+\text{SpecMMD})
]

---

# 13. 为什么这个方案比当前 TCCIG 更适合你的失败模式

## 当前 TCCIG 的失败模式

P4 的结果说明：

```text
AUROC 0.363
AUPRC 0.010
precision 0.007
graph_sim 0.022
oracle-density GS 0.007
```

这不是“预算错”，也不是“exact saturation”。因为即使用 oracle density，top-ranked edges 也不在正确位置。问题是：

[
\text{rank localization is anti-informative}
]

## 新方案的改动点

| 当前设计                           | 问题                                        | 新设计                                                             |
| ------------------------------ | ----------------------------------------- | --------------------------------------------------------------- |
| pair MLP + topology losses     | pair scores 没有 retrieval rank supervision | dual-encoder InfoNCE retrieval                                  |
| sampled BCE only               | supervised fraction 0.063，信号太稀疏           | (B\times B) in-batch contrastive dense supervision              |
| GSL as soft teacher            | 只给局部 pair signal                          | graph embedding + PPR distribution + module + role distillation |
| full candidate top-m           | 需要在大候选空间找 needle                          | ANN top-R candidate retrieval                                   |
| topology MMD monitor           | 不保证边命中                                    | graph_sim / CandidateRecall@R monitor                           |
| rerank all pairs impossible    | 计算爆炸                                      | rerank retrieved candidates only                                |
| module head weakly regularized | 不会真正学 biological modules                  | train graph community / overlap module supervision              |

---

# 14. 具体模型公式

## 14.1 Encoder

[
x_i = \text{ESM}(p_i)
]

[
h_i = f_\theta(x_i)
]

[
q_i = W_Q h_i,\quad k_i=W_K h_i
]

[
m_i = \text{softmax}(W_Mh_i)
]

[
r_i = W_Rh_i
]

## 14.2 Retrieval score

[
s_{\text{ret}}(i,j)
===================

\frac{q_i^\top k_j+q_j^\top k_i}{2\tau}
]

## 14.3 Module compatibility

[
s_{\text{mod}}(i,j)
===================

m_i^\top Bm_j
]

## 14.4 Role compatibility

[
s_{\text{role}}(i,j)
====================

\text{MLP}([r_i,r_j,|r_i-r_j|,r_i\odot r_j])
]

## 14.5 Final retrieval score

[
s(i,j)
======

s_{\text{ret}}(i,j)
+
\alpha s_{\text{mod}}(i,j)
+
\beta s_{\text{role}}(i,j)
]

with small initialized (\alpha,\beta), e.g. 0.05–0.1.

Optional Raft-style residue factorization:

[
s(i,j)
======

s(i,j)
+
\eta
\langle h_i^{raft},h_j^{raft}\rangle
]

---

# 15. Training algorithm

```python
# Stage 0: Build train graph prior
G_train = build_graph_from_positive_edges(human_train_ppi)

z_graph = train_s2gae_or_gsr(G_train, node_features=ESM)
ppr_targets = compute_ppr_or_random_walk_distributions(G_train)
module_targets = compute_louvain_or_nocd_modules(G_train)
role_targets = compute_degree_core_pagerank_roles(G_train)

# Stage 1: Train feature-only retriever
for batch in sample_positive_edges(G_train):
    # batch edges: (u_i, v_i)
    U, V = batch.sources, batch.targets

    h_U = encoder(ESM[U])
    h_V = encoder(ESM[V])

    q_U, k_V = project_query_key(h_U, h_V)

    # Dense B x B ranking matrix
    S = q_U @ k_V.T / tau

    # Mask false negatives using known train edges
    mask = known_positive_mask(U, V, G_train)

    loss_retrieval = multi_positive_infonce(S, positives=diagonal_or_multi_pos, mask=mask)

    # Graph-prior dense distillation over batch nodes
    loss_neighbor = KL(
        ppr_targets[U][:, V],
        softmax(S)
    )

    # Node-level graph prior distillation
    loss_embed = contrastive_align(h_U, z_graph[U]) + contrastive_align(h_V, z_graph[V])
    loss_module = CE(module_head(h_U), module_targets[U])
    loss_role = MSE(role_head(h_U), role_targets[U])

    # Online hard negatives
    hard_neg_pairs = mine_hard_negatives(S, known_edges=G_train)
    loss_hardneg = adaptive_neg_loss(hard_neg_pairs)

    loss = (
        loss_retrieval
        + λ_neighbor * loss_neighbor
        + λ_embed * loss_embed
        + λ_module * loss_module
        + λ_role * loss_role
        + λ_hardneg * loss_hardneg
    )

    backward(loss)
```

Optional reranker:

```python
# Stage 2: Train reranker on retrieved/hard candidates
for anchor proteins:
    candidate_pairs = ANN_retrieve_top_R(anchor)
    candidate_pairs += known positives
    candidate_pairs += hard negatives

    score = reranker(pair)
    loss = focal_bce(score, label) + adaptive_hard_negative_loss
```

---

# 16. PRING output algorithm

```python
def predict_pring_all_test(model, all_test_ppi, fasta, R, threshold_or_budget):
    C = read_pairs(all_test_ppi)
    V = unique_nodes(C)

    embeddings = encode_once(V, fasta)
    ann_index = build_ann_index(embeddings.keys)

    retrieved = set()
    for i in V:
        nbrs = ann_index.search(embeddings.query[i], top_k=R)
        for j in nbrs:
            pair = canonical_pair(i, j)
            if pair in C:
                retrieved.add(pair)

    scores = {}
    for pair in retrieved:
        scores[pair] = model.score_pair(pair)

    E_hat = select_edges(scores, rule=threshold_or_budget)

    with open("all_test_ppi_pred.txt", "w") as f:
        for pair in C:
            label = 1 if pair in E_hat else 0
            f.write(f"{pair.i} {pair.j} {label}\n")
```

---

# 17. 实验计划

## Run 1：Dual-encoder retrieval baseline

目的：验证 retrieval formulation 是否比 TCCIG full-graph generator 更适合 PRING。

训练：

```text
ESM protein embedding
query/key projection
InfoNCE over train positives
false-negative masking
```

评估：

```text
CandidateRecall@R
AUPRC on all_test_ppi
PRING graph_sim
relative_density
```

成功标准：

```text
AUPRC > P2
graph_sim > P2
precision > random-density baseline
```

如果 Run 1 都不能超过 P2，说明 intrinsic features 与 PRING split 的 edge localization 非常弱，需要更强 protein encoder / external pretraining / graph prior。

## Run 2：+ graph embedding distillation

加入：

```text
S2GAE/GSR train graph embedding z_i^G
L_embed
```

成功标准：

```text
CandidateRecall@R 提升
graph_sim 提升
AUROC/AUPRC 不下降
```

## Run 3：+ PPR / random-walk neighbor distribution distillation

加入：

```text
L_neighbor = KL(PPR_train || softmax(scores))
```

这是最关键实验。它直接测试“training graph prior 是否能转化为 retrieval localization”。

成功标准：

```text
top-R retrieved true edges 增加
oracle-density graph_sim 增加
```

## Run 4：+ hard negative mining / adaptive negative weighting

加入：

```text
ANN hard negatives
same-module hard negatives
adaptive negative weighting
```

成功标准：

```text
precision@K 明显提升
AUPRC 提升
P4 那种比随机更差的 ranking 消失
```

## Run 5：+ residue/contact reranker

只对 top-R candidates rerank。

候选：

```text
Raft-style factorized residue scorer
FlashPPI-like contact head
MINT-like cross-chain attention adapter
```

成功标准：

```text
precision@selected_edges 提升
graph_sim 提升
不会降低 candidate recall
```

---

# 18. 关于 external data / MINT / STRING 的 PRING 公平性

MINT 使用 STRING-derived 96M high-quality PPIs 做 unsupervised PPI-specific training。 这非常有启发，但在 PRING benchmark 中要小心：如果 PRING 的 test edges 与 STRING-derived training corpus 有重叠，直接使用 MINT 或 STRING-pretrained graph prior 可能引入 leakage。

所以建议报告两套 setting：

## A. PRING-closed setting

只使用：

```text
PRING human_train_ppi
PRING human_val_ppi
protein sequences / ESM embeddings
```

这是主结果。

## B. External-prior setting

允许使用：

```text
MINT / STRING pretraining
PDB / AFDB / DDI contact data
RaftPPI / FlashPPI pretrained encoder
```

但必须标注为 external-pretrained，并做 overlap audit。

---

# 19. 最终建议

我建议停止继续强化当前 TCCIG 的 full adjacency generator 方向。现在的核心应改成：

> **A PRING-aligned graph-prior retrieval model: use the training PPI graph to learn graph priors, distill those priors into a feature-only dual encoder, retrieve candidate interactions without full all-pairs scoring, and optionally rerank only retrieved candidates with residue/contact-aware models.**

最小可执行新版是：

```text
ESM embeddings
    ↓
dual encoder q/k retrieval
    ↓
InfoNCE on train PPI edges
    ↓
S2GAE/GSR graph embedding distillation
    ↓
PPR/random-walk neighbor distribution distillation
    ↓
ANN top-R retrieval on PRING all_test_ppi proteins
    ↓
validation-selected threshold/top-R
    ↓
all_test_ppi_pred.txt
```

这个方向直接回应 P4 的失败：不是去调 sigmoid，而是把训练目标从 “少量 BCE + topology distribution” 改成 “query-wise edge localization / retrieval ranking”。PRING 的 graph_sim 本质是 edge-overlap F1-like metric，所以模型必须先学会 **把真实边排到前面**，再谈 density、degree、clustering、spectral realism。

[1]: https://github.com/SophieSarceau/PRING "GitHub - SophieSarceau/PRING: [NeurIPS 2025] PRING: Rethinking Protein-Protein Interaction Prediciton from Pairs to Graphs · GitHub"
[2]: https://raw.githubusercontent.com/SophieSarceau/PRING/main/topology_task/intra_species.md "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/SophieSarceau/PRING/main/topology_task/cross_species.md "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/SophieSarceau/PRING/main/topology_task/eval.py "raw.githubusercontent.com"

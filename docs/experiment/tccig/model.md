下面直接定义 proposed 完整版模型。我把它命名为：

# **TopoPPI-Gen: Topology-Constrained Conditional PPI Graph Generator**

目标不是训练一个 isolated pair classifier，而是学习：

$$
p_\theta(A_S \mid X_S)
$$

其中 $S={p_1,\dots,p_n}$ 是 unseen protein set，$X_S$ 是 protein-intrinsic features，例如 ESM/ESM-2/ESM3 embeddings，输出是整个 edge probability matrix：

$$
P_S \in [0,1]^{n\times n}, \quad P_{ij}=p_\theta(A_{ij}=1\mid X_S)
$$

再通过 learned edge budget / validation-calibrated threshold 生成 sparse predicted interactome：

$$
\hat{G}_S=(S,\hat{E}_S)
$$

核心原则：**training 可以从 training graphs/source interactomes 学 topology prior；test-time 对 target unseen protein set 只能输入 intrinsic features，不能输入 target edges、degrees、neighborhoods、communities、Laplacian 或任何 target topology-derived signal。** 这直接对应 research_problem.md 的边界：graph 是 scientific object，不是 pairwise predictions 的可视化；edge probabilities 可以局部合理，但 thresholded graph 仍可能 density、hub、clustering、module structure 错误。

## Implementation status: 2026-05-30

当前 `tccig-train-stage` 分支实现的是这个 full design 的 v1 子集：

- Public model: `src/model/tccig.py` registers `model_config.model: tccig` and exposes `forward_graph(...)` for feature-only graph scoring.
- Config and launcher: canonical config is `configs/tccig/tccig.yaml`; canonical HPC launcher is `scripts/tccig.sh`.
- Student path: cached protein embeddings → mean pooling/projection → set-summary conditioning → all-pairs candidate universe for sampled subgraphs, or PRING candidate records during topology evaluation.
- Edge decoder: symmetric pair features, hub propensity, low-rank affinity, overlapping module memberships, set-level density bias, and learned `m_hat`.
- Graph Assembly: topology evaluation encodes unique test proteins once, scores candidate records in chunks, and selects top-`m_hat` edges.
- Train-only teacher: optional online MGAE teacher masks positive training edges and distills candidate-edge probabilities into the student.
- Active losses: masked BCE edge loss, teacher distillation, budget, density, degree MMD, and optional clustering MMD.

Not implemented yet: feature kNN/anchor candidate proposer, offline S2GAE/MaskGAE/Bandana teacher pretraining, spectral/module/ranking/calibration/sparsity losses as active nonzero objectives, and validation-calibrated threshold selection beyond the current top-`m_hat` assembly path.


# 1. Overall pipeline

完整 pipeline 分成两条信息流：

```text
A. Student generator: test-time deployed, feature-only
   Protein sequences / ESM embeddings
        ↓
   Protein feature encoder
        ↓
   Set-conditioned node encoder
        ↓
   Feature-only candidate edge proposer
        ↓
   Degree-corrected module-aware edge generator
        ↓
   Soft self-refinement on predicted graph only
        ↓
   Edge probability matrix P
        ↓
   Edge-budget / threshold graph assembler
        ↓
   Predicted PPI graph

B. Train-only topology teacher: used only on training graph
   Train-only PPI graph + ESM node features
        ↓
   S2GAE / MaskGAE / Bandana-style mask-reconstruct pretraining
        ↓
   Teacher edge probabilities / topology-aware representations
        ↓
   Distillation signal for Student
```

关键是：**Student 是最终模型；Teacher 是训练阶段的 privileged topology pretraining module，不在 test graph 上运行。**

masked_edge_reconstruction 文件中也明确指出，PPI 场景中 ESM/ESM3 已经提供强 node features，缺失的是 topology signal；因此最相关的是 structure-masked variants，例如 MGAE、MaskGAE、S2GAE、Bandana，而不是 GraphMAE-style feature reconstruction。


# 2. Module-by-module architecture

## Module 0: Protein feature extractor

输入 protein sequence $s_i$，使用 frozen 或 lightly fine-tuned protein language model：

$$
x_i = \text{ESM}(s_i)
$$

其中：

$$
x_i \in \mathbb{R}^{d_{\text{ESM}}}
$$

推荐默认：

```text
ESM backbone: frozen
Representation: mean-pooled residue embedding, CLS token, or learned pooling over residue embeddings
Projection dimension: 512
Fine-tuning: initially off; later optional LoRA / adapter fine-tuning
```

然后投影：

$$
h_i^{(0)} = \text{LayerNorm}(\text{MLP}_{\text{proj}}(x_i))
$$

$$
h_i^{(0)} \in \mathbb{R}^{d}, \quad d=256\text{ or }512
$$

这个模块不使用任何 graph topology。


## Module 1: Set-conditioned node encoder

PPI reconstruction 是 set-level problem：同一个 protein 在不同 candidate protein universe 中可能有不同可连接对象。因此需要让每个 node representation 知道当前 protein set 的 composition，但不能知道 target graph topology。

使用 permutation-equivariant set encoder：

$$
H^{(1)} = \text{SetEncoder}(H^{(0)})
$$

可选实现：

```text
Option A: Set Transformer
Option B: Perceiver-style latent bottleneck
Option C: Linear attention Transformer
Option D: DeepSets + attention pooling
```

推荐实现：

$$
c_i = \text{CrossAttn}(h_i^{(0)}, Z_{\text{set}})
$$

其中 $Z_{\text{set}}$ 是从所有 proteins 的 embeddings 聚合得到的 latent set tokens。

最终：

$$
h_i = \text{MLP}([h_i^{(0)}, c_i])
$$

这个模块允许模型学习“这组 proteins 整体上可能形成怎样密度和模块结构的 graph”，但仍然没有读取 target graph。


## Module 2: Feature-only candidate edge proposer

全 pair scoring 是 $O(n^2)$。如果 $n=20{,}000$，候选边约 $2\times 10^8$，不可直接训练。因此需要 feature-only candidate proposer：

$$
C_S \subset {(i,j): i<j}
$$

只允许用 $H$ 构造候选边，不能用 target degree / observed edge / neighborhood。

推荐两级候选：

### 2.1 Biochemical candidate pool

学习 query/key：

$$
q_i = W_qh_i,\quad k_i=W_kh_i
$$

相似度：

$$
r_{ij} = \frac{q_i^\top k_j}{\sqrt{d}}
$$

对每个 node 取 top-(L)：

$$
C_i = \text{TopL}_j(r_{ij})
$$

默认：

```text
L = 64–256 per protein
candidate graph symmetrized
include random exploration negatives during training
```

### 2.2 Anchor-based scalable candidate pool

引入 $M$ 个 learnable anchors：

$$
a_1,\dots,a_M
$$

每个 protein 分配到若干 anchors：

$$
\pi_{im}=\text{softmax}(h_i^\top a_m)
$$

只在 shared / compatible anchor buckets 内生成 candidates。这个设计借鉴 GSL / NodeFormer / DGM 的 latent graph inference 思路：Route B 中最相关的是从 node features 构造 latent graph，并需要 explicit sparsification 和 inductive intent。

最终候选：

$$
C_S = C_{\text{kNN}} \cup C_{\text{anchor}} \cup C_{\text{explore}}
$$

测试时也只用 features 构造 $C_S$。


## Module 3: Degree-corrected module-aware edge generator

这是核心 decoder。不能只用：

$$
P_{ij}=\sigma(\text{MLP}([h_i,h_j]))
$$

因为这会退化为 pairwise classifier。我们使用 decomposed edge logit：

$$
\ell_{ij}
=
s_{\text{pair}}(h_i,h_j)
+
\alpha_i+\alpha_j
+
u_i^\top u_j
+
q_i^\top B q_j
+
b_S
$$

$$
P_{ij}=\sigma(\ell_{ij})
$$

每个项有明确功能。


## 3.1 Pairwise biochemical compatibility score

$$
s_{\text{pair}}(h_i,h_j)
=
\text{MLP}_{\text{pair}}([h_i,h_j,h_i\odot h_j,|h_i-h_j|])
$$

为了保证 symmetry，可以用：

$$
s_{\text{pair}}(i,j)=s_{\text{pair}}(j,i)
$$

输入特征：

```text
h_i
h_j
h_i ⊙ h_j
|h_i - h_j|
cos(h_i, h_j)
optional: residue-level cross-attention summary if available
```

这个分支负责 local biochemical evidence。


## 3.2 Node sociability / hub propensity head

PPI graph 有 hubs。不能在 test 输入 true degree，但可以从 intrinsic features 学一个 **predicted degree propensity**：

$$
\alpha_i = f_{\text{hub}}(h_i)
$$

它进入 logit：

$$
\ell_{ij} \leftarrow \ell_{ij}+\alpha_i+\alpha_j
$$

解释：某些 proteins intrinsically 更可能是 hubs，例如 multi-domain scaffolding proteins、chaperones、signaling adaptors。模型通过 training graphs 学习这种 feature-to-hubness mapping，但 test 时不读 target degree。


## 3.3 Low-rank latent affinity term

$$
u_i = W_uh_i
$$

$$
s_{\text{lowrank}}(i,j)=u_i^\top u_j
$$

这个项让模型捕捉 smooth latent affinity，不完全依赖 MLP pair score。


## 3.4 Overlapping module membership head

真实 PPI 网络不是 flat edge set，而是由 complexes、pathways、functional modules 组织起来。模型输出 soft overlapping module membership：

$$
q_i = \text{softplus}(W_q h_i)
$$

$$
q_i \in \mathbb{R}_{\ge 0}^{K}
$$

其中 $K$ 是 latent modules 数量，例如 64 或 128。

模块交互项：

$$
s_{\text{module}}(i,j)=q_i^\top B q_j
$$

其中：

$$
B\in \mathbb{R}^{K\times K}
$$

如果 $B$ 近似 diagonal，则鼓励 within-module edges；如果 $B$ full-rank，则允许 pathway-to-pathway interactions。

这个模块非常重要，因为 research_problem.md 强调 protein complexes、pathways、community detection、GO enrichment 等都是 graph-level biological tasks，不是单边分类问题。


## 3.5 Set-level density bias

PPI graph 极稀疏。固定 threshold 0.5 通常错误。因此模型预测 set-level density bias：

$$
g_S = \text{Pool}(\{h_i\}_{i\in S})
$$

$$
b_S = f_{\text{density}}(g_S)
$$

它进入所有 edge logits：

$$
\ell_{ij}\leftarrow \ell_{ij}+b_S
$$

这相当于让模型学习“这一组 proteins 的整体 interaction density 应该是多少”。


## Module 4: Edge budget head

除了 probability matrix，还预测 expected edge count：

$$
\hat{m}_S = f_{\text{budget}}(g_S)
$$

可用：

$$
\hat{\rho}_S = \sigma(f_{\rho}(g_S))
$$

$$
\hat{m}_S = \hat{\rho}_S \cdot \binom{|S|}{2}
$$

但为了稳定，建议预测 log-edge-count：

$$
\log(1+\hat{m}_S)=f_m(g_S)
$$

最终图 assembly：

```text
Option A: choose top-⌊m_hat⌋ edges by P_ij
Option B: choose all edges P_ij > t_val
Option C: constrained version: top-m with per-node soft max-degree regularization
```

默认用 **top-$\hat{m}_S$**，因为 density control 是 graph reconstruction 的关键。research_problem.md 中的例子显示，强 pairwise predictor 可以产生 consistently over-dense predicted graphs，因此 edge budget 不是后处理小细节，而是模型的一部分。


## Module 5: Soft self-refinement on predicted graph

为了让模型不是“一次 pair scoring”，加入 $R=1$ 或 $2$ 轮 self-refinement。

第 0 轮：

$$
P^{(0)} = \text{EdgeGenerator}(H^{(0)})
$$

然后只在 predicted soft graph 上 message passing：

$$
h_i^{(r+1)}
=
\text{GRU}
\left(
h_i^{(r)},
\sum_{j:(i,j)\in C_S}
P_{ij}^{(r)} W_m h_j^{(r)}
\right)
$$

或：

$$
H^{(r+1)}=\text{SoftGraphTransformer}(H^{(r)}, P^{(r)})
$$

再重新解码：

$$
P^{(r+1)}=\text{EdgeGenerator}(H^{(r+1)})
$$

关键：这里 message passing 的 graph 是模型自己从 features 生成的 $P^{(r)}$，不是 observed target graph。因此 test-time 合法。

推荐默认：

```text
R = 1 initially
R = 2 for full model
do not exceed 2 early on, otherwise topology hallucination risk increases
```


## Module 6: Topology statistics heads / differentiable topology losses

这些不是独立 inference 输入，而是 training objective。

模型从 $P$ 得到 soft topology statistics：

### 6.1 Soft degree

$$
\hat{d}_i=\sum_j P_{ij}
$$

### 6.2 Soft density

$$
\hat{\rho}=\frac{\sum_{i<j}P_{ij}}{\binom{n}{2}}
$$

### 6.3 Soft triangle count

$$
\hat{T}=\frac{1}{6}\text{tr}(P^3)
$$

### 6.4 Soft wedge count

$$
\hat{W}=\sum_i \frac{\hat{d}_i(\hat{d}_i-1)}{2}
$$

### 6.5 Soft clustering / transitivity

$$
\hat{C}=\frac{3\hat{T}}{\hat{W}+\epsilon}
$$

### 6.6 Spectral summaries

用 soft Laplacian：

$$
\hat{L}=\hat{D}-P
$$

不建议大图全 eigen-decomposition。使用：

```text
small induced subgraph eigenvalues
or Chebyshev trace estimates
or heat-kernel traces tr(exp(-tL))
```


# 3. Train-only topology teacher

## 3.1 是否做 MGAE / S2GAE / MaskGAE / Bandana mask-reconstruct?

答案：**做，但只在 training graphs 上做，作为 train-only topology teacher / auxiliary pretraining；test-time 完全不做 target graph mask-reconstruct。**

具体采用：

```text
Mandatory: S2GAE-style direction-aware masked-edge reconstruction
Auxiliary: MaskGAE-style path-wise masking + degree regression
Optional robustness branch: Bandana-style continuous bandwidth masking
Baseline / ablation: MGAE-style high-ratio random edge masking
```

不建议把四者简单 ensemble 成四个大模型。更可执行的完整版是一个 **multi-corruption topology teacher**：

每个 teacher training step 随机选择一种 corruption mode：

```text
mode ∈ {s2_edge_mask, path_mask, bandwidth_mask}
```

其中：

* `s2_edge_mask` 是主模式；
* `path_mask` 增强 motif / pathway / hub learning；
* `bandwidth_mask` 防止 binary edge deletion 太 harsh；
* `mgae_random_mask` 作为 ablation 或 warm-up，不是主模式。

masked_edge_reconstruction 文件给出的 ranking 也支持这个选择：S2GAE 是 top choice，MaskGAE 是 second choice，MGAE 是 minimal strong baseline，Bandana 是 experimental upgrade；GraphMAE 不应作为 primary self-supervised loss，因为它主要 reconstruct node features，而这里需要 focus on missing PPI topology。


## 3.2 Teacher architecture

Teacher 输入：

$$
G_{\text{train}}=(V_{\text{train}},E_{\text{train}})
$$

$$
X_{\text{train}}={x_i:i\in V_{\text{train}}}
$$

Teacher encoder 是 graph-aware GNN：

$$
Z_T = \text{GNN}_T(X_{\text{train}}, E_{\text{visible}})
$$

其中 $E_{\text{visible}}$ 是 masked 后剩下的 training edges。

Decoder 用 cross-correlation decoder：

$$
\ell_{ij}^{T}=\text{CCDecoder}(z_i^T,z_j^T,\{z_i^{T,l},z_j^{T,l}\}_{l=1}^L)
$$

S2GAE 和 MGAE 都强调 cross-correlation decoder 对 masked-edge reconstruction 很关键；S2GAE 的 ablation 显示 graph masking、masked graph reconstruction、cross-correlation decoder 都是核心设计。

Teacher 输出：

```text
teacher edge probability T_ij for training pairs
teacher topology-aware embedding z_i^T for train proteins
teacher degree prediction, if using MaskGAE auxiliary
```

但注意：**teacher embedding 不能直接用于 unseen test proteins，除非 teacher encoder 本身不需要 target graph。这里我们不把 teacher 部署到 test。**


# 4. Student training losses

对 training subset $S_b$，Student 只输入：

$$
X_{S_b}
$$

但 loss 可以用 training graph induced adjacency：

$$
A_{S_b}
$$

总 loss：

$$
\mathcal{L}
=
\mathcal{L}_{edge}
+
\lambda_{\text{rank}}\mathcal{L}_{rank}
+
\lambda_{\text{distill}}\mathcal{L}_{teacher}
+
\lambda_{\text{budget}}\mathcal{L}_{budget}
+
\lambda_{\text{degree}}\mathcal{L}_{degree}
+
\lambda_{\text{clust}}\mathcal{L}_{clust}
+
\lambda_{\text{spec}}\mathcal{L}_{spec}
+
\lambda_{\text{module}}\mathcal{L}_{module}
+
\lambda_{\text{cal}}\mathcal{L}_{cal}
+
\lambda_{\text{sparse}}\mathcal{L}_{sparse}
$$


## 4.1 Edge classification loss

Positive edges:

$$
E_b^+ = E_{\text{train}}\cap (S_b\times S_b)
$$

Negative edges:

$$
E_b^- \subset {(i,j):i,j\in S_b, i<j, (i,j)\notin E_{\text{train}}}
$$

Use class-balanced BCE or focal BCE:

$$
\mathcal{L}_{edge}
=
-\sum_{(i,j)\in E_b^+} w_+ \log P_{ij}
-\sum_{(i,j)\in E_b^-} w_- \log(1-P_{ij})
$$

Because PPI labels are incomplete, negative sampling should include:

```text
random negatives
hard negatives from high-scoring non-edges
degree-matched negatives
sequence-similarity matched negatives
```

Avoid sampling validation/test positives as train negatives.


## 4.2 Ranking loss

For each positive edge ((i,j)), sample hard negative ((i,k)) or ((u,v)):

$$
\mathcal{L}_{rank}
=
\sum \max(0,\gamma - \ell_{ij}+\ell_{uv})
$$

This improves Precision@K / AUPR.


## 4.3 Teacher distillation loss

Teacher provides soft edge probability $T_{ij}$ for training pairs.

$$
\mathcal{L}_{teacher}
=
\sum_{(i,j)\in \Omega_b}
\text{KL}(\text{Bern}(T_{ij})|\text{Bern}(P_{ij}))
$$

or soft BCE:

$$
\mathcal{L}_{teacher}
=
-\sum_{(i,j)}
T_{ij}\log P_{ij}+(1-T_{ij})\log(1-P_{ij})
$$

Important: this is only over training proteins / training graph subsets. It transfers topology-aware signal into Student without using target-test topology.


## 4.4 Edge budget / density loss

$$
\hat{m}_b=\sum_{i<j}P_{ij}
$$

$$
m_b=|E_b^+|
$$

$$
\mathcal{L}_{budget}
=
\text{Huber}(\log(1+\hat{m}_b)-\log(1+m_b))
$$

Density loss:

$$
\mathcal{L}_{density}
=
\left|
\frac{\hat{m}_b}{\binom{|S_b|}{2}}
-
\frac{m_b}{\binom{|S_b|}{2}}
\right|
$$


## 4.5 Degree distribution loss

Soft degree:

$$
\hat{d}_i=\sum_jP_{ij}
$$

True training-subgraph degree:

$$
d_i=\sum_jA_{ij}
$$

Use node-level and distribution-level terms:

$$
\mathcal{L}_{degree-node}
=
\frac{1}{|S_b|}\sum_i
\left(\log(1+\hat{d}_i)-\log(1+d_i)\right)^2
$$

$$
\mathcal{L}_{degree-dist}
=
\text{MMD}(\{\log(1+\hat{d}_i)\},\{\log(1+d_i)\})
$$

$$
\mathcal{L}_{degree}
=
\mathcal{L}_{degree-node}
+
\mathcal{L}_{degree-dist}
$$

这个 loss 不等于 test-time 输入 degree。它只是让模型在 training graphs 上学习 feature-to-degree-propensity mapping。


## 4.6 Clustering / triangle loss

$$
\hat{T}=\frac{1}{6}\text{tr}(P^3)
$$

$$
T=\frac{1}{6}\text{tr}(A^3)
$$

$$
\mathcal{L}_{triangle}
=
|\log(1+\hat{T})-\log(1+T)|
$$

Transitivity:

$$
\hat{C}=\frac{3\hat{T}}{\hat{W}+\epsilon}
$$

$$
C=\frac{3T}{W+\epsilon}
$$

$$
\mathcal{L}_{clust}=|\hat{C}-C|
$$

对于大图，使用 sampled induced subgraphs 近似。


## 4.7 Spectral topology loss

用 normalized Laplacian：

$$
\hat{L}_{norm}=I-\hat{D}^{-1/2}P\hat{D}^{-1/2}
$$

训练时对 small induced subgraph 计算 top-(k) eigenvalues：

$$
\mathcal{L}_{spec}
=
\left|
\lambda_{1:k}(\hat{L}_{norm})
-
\lambda_{1:k}(L_{norm})
\right|_2^2
$$

大图版本用 heat trace：

$$
h_t(L)=\text{tr}(\exp(-tL))
$$

$$
\mathcal{L}_{spec}
=
\sum_{t\in\mathcal{T}}
|h_t(\hat{L})-h_t(L)|
$$


## 4.8 Module / community loss

模型输出 $Q=[q_1,\dots,q_n]$。

使用 differentiable modularity-style objective：

$$
\mathcal{L}_{module}
=
-\frac{1}{2m}
\text{Tr}
\left[
Q^\top
\left(
A-\frac{dd^\top}{2m}
\right)
Q
\right]
$$

并加 entropy / balance regularization 防止所有 nodes collapse 到同一个 module：

$$
\mathcal{L}_{balance}
=
\left|
\frac{1}{n}\sum_i\text{softmax}(q_i)
-
\frac{1}{K}\mathbf{1}
\right|_2^2
$$

$$
\mathcal{L}_{entropy}
=
-\frac{1}{n}\sum_i H(\text{softmax}(q_i))
$$


## 4.9 Calibration loss

$$
\mathcal{L}_{cal}
=
\frac{1}{|\Omega_b|}
\sum_{(i,j)\in\Omega_b}
(P_{ij}-A_{ij})^2
$$

即 Brier score。validation 上再做 temperature scaling：

$$
P_{ij}^{calibrated} = \sigma(\ell_{ij}/T)
$$


## 4.10 Sparsity loss

$$
\mathcal{L}_{sparse}
=
\frac{1}{|C_b|}\sum_{(i,j)\in C_b}P_{ij}
$$

这个 loss 只作为轻量 regularizer，不能压过 edge recall。


# 5. Training behavior

## 5.1 Training 阶段允许的信息

允许：

```text
training proteins 的 intrinsic features
training graph edges
training graph induced degrees / clustering / spectrum / modules，用作 loss target
source species interactomes
validation graph only for model selection and threshold calibration
```

禁止：

```text
test target graph edges
test target degrees
test target neighborhoods
test target community labels
test graph Laplacian
message passing over observed test graph
test graph-derived node embeddings
```


## 5.2 Complete training schedule

推荐 4-stage training。

### Stage 1: ESM feature preparation

```text
Compute ESM embeddings for all train/val/test proteins.
Freeze ESM initially.
Store protein-level embeddings.
```

### Stage 2: Train-only topology teacher pretraining

Use train graph only.

```text
Teacher corruption mode:
  60% S2GAE direction-aware edge masking
  25% MaskGAE path-wise masking + degree regression
  15% Bandana continuous bandwidth masking
```

若训练图非常 sparse：

```text
increase directed masking probability
mask ratio around 0.6–0.7
avoid disconnecting too many small components
```

### Stage 3: Student graph generator training

Student never receives graph adjacency as input.

Input:

$$
X_{S_b}
$$

Output:

$$
P_{S_b}, \hat{m}_{S_b}, Q_{S_b}
$$

Loss:

$$
\mathcal{L}_{student}
$$

Teacher frozen. Use teacher logits as auxiliary soft labels.

### Stage 4: Validation calibration and model selection

On validation protein subsets:

```text
forward only from features
do not input validation graph to model
use validation graph only to compute metrics
select threshold / edge-budget calibration
select checkpoint by composite pairwise + topology score
```


# 6. Test behavior

For unseen protein set $S_{\text{test}}$:

```text
Input:
  protein sequences or pretrained ESM embeddings only

Not input:
  target-test edges
  target-test degrees
  target-test neighborhoods
  target-test communities
  target-test Laplacian
  target graph for message passing
```

Forward:

$$
X_{\text{test}}\rightarrow H\rightarrow C_{\text{test}}\rightarrow P_{\text{test}}\rightarrow \hat{m}_{\text{test}}\rightarrow \hat{E}_{\text{test}}
$$

Graph assembly:

$$
\hat{E}_{\text{test}}
=
\text{Top}_{\lfloor \hat{m}_{\text{test}}\rfloor}
{P_{ij}:(i,j)\in C_{\text{test}}}
$$

or:

$$
\hat{E}_{\text{test}}={(i,j):P_{ij}>t_{\text{val}}}
$$

Teacher is not called at test.

No mask-reconstruct is performed on target test graph.


# 7. Forward-prop pseudocode: Student model

```python
def forward_student(protein_set, esm_embeddings):
    """
    protein_set: list of protein ids in current set S
    esm_embeddings: tensor [n, d_esm]
    No graph adjacency is accepted here.
    """

    # ----- Module 0: feature projection -----
    X = esm_embeddings                              # [n, d_esm]
    H0 = LayerNorm(MLP_proj(X))                     # [n, d]

    # ----- Module 1: set-conditioned encoding -----
    Z_set = SetLatentPooling(H0)                    # [m_latent, d]
    Ctx = CrossAttention(query=H0, key=Z_set, value=Z_set)
    H = MLP_node(concat(H0, Ctx))                   # [n, d]

    # ----- Module 2: feature-only candidate proposal -----
    Q_key = W_key(H)                                # [n, d_k]
    Q_query = W_query(H)                            # [n, d_k]

    # Candidate construction uses only features.
    C_knn = topL_feature_neighbors(Q_query, Q_key, L=128)
    C_anchor = anchor_bucket_candidates(H)
    C = symmetrize(union(C_knn, C_anchor))          # sparse pair list

    # ----- Module 3-5: iterative edge generation/refinement -----
    for r in range(R + 1):
        alpha = hub_head(H)                         # [n, 1]
        U = lowrank_head(H)                         # [n, d_u]
        Qmod = softplus(module_head(H))             # [n, K]

        g = pool_set(H)                             # [d]
        b_set = density_bias_head(g)                # scalar
        m_hat = exp(edge_budget_head(g)) - 1.0      # scalar

        logits = {}
        for (i, j) in C:
            pair_feat = concat(
                H[i], H[j],
                H[i] * H[j],
                abs(H[i] - H[j])
            )

            s_pair = pair_mlp(pair_feat)
            s_hub = alpha[i] + alpha[j]
            s_lowrank = dot(U[i], U[j])
            s_module = Qmod[i].T @ B_module @ Qmod[j]

            logits[(i, j)] = (
                s_pair
                + s_hub
                + s_lowrank
                + s_module
                + b_set
            )

        P = sigmoid_sparse(logits, C)               # sparse edge probabilities

        if r < R:
            # Self-refinement uses predicted soft graph only.
            M = soft_message_passing(H, P, C)
            H = GRUCell(input=M, hidden=H)

    return {
        "P": P,               # probabilities on candidate edges
        "C": C,               # candidate edge set
        "m_hat": m_hat,       # predicted edge budget
        "Qmod": Qmod,         # latent module memberships
        "H": H,               # final protein embeddings
        "logits": logits
    }
```

重要 implementation rule：`forward_student()` 的函数签名中不允许出现 `A_target`、`degree_target`、`laplacian_target`、`community_target`。


# 8. Forward/backward pseudocode: train-only topology teacher

```python
def train_teacher_epoch(train_graph, esm_embeddings):
    """
    train_graph: graph over training proteins only.
    This module is never run on target test graph.
    """

    for S_b in sample_training_subgraphs(train_graph):
        X_b = esm_embeddings[S_b]
        E_b = induced_edges(train_graph, S_b)

        mode = sample_mode({
            "s2_edge_mask": 0.60,
            "path_mask": 0.25,
            "bandwidth_mask": 0.15
        })

        if mode == "s2_edge_mask":
            # S2GAE-style direction-aware masking
            M_pos, E_visible = direction_aware_edge_mask(
                E_b,
                mask_ratio=0.6_or_0.7
            )

            Z_layers = teacher_gnn(
                X_b,
                E_visible,
                return_all_layers=True
            )

            pos_logits = cc_decoder(Z_layers, M_pos)

            M_neg = sample_negatives(
                nodes=S_b,
                exclude_edges=E_b,
                num=len(M_pos)
            )

            neg_logits = cc_decoder(Z_layers, M_neg)

            loss_mask = bce_pos_neg(pos_logits, neg_logits)
            loss = loss_mask

        elif mode == "path_mask":
            # MaskGAE-style path-wise masking
            M_pos, E_visible = random_walk_path_mask(
                E_b,
                walk_length=2_to_4,
                num_paths=batch_dependent
            )

            Z = teacher_gnn(X_b, E_visible)

            pos_logits = structure_decoder(Z, M_pos)

            M_neg = sample_negatives(
                nodes=S_b,
                exclude_edges=E_b,
                num=len(M_pos)
            )

            neg_logits = structure_decoder(Z, M_neg)

            d_true = degree_vector(E_b, S_b)
            d_pred = degree_decoder(Z)

            loss_edge = bce_pos_neg(pos_logits, neg_logits)
            loss_degree = mse(log1p(d_pred), log1p(d_true))

            loss = loss_edge + lambda_deg_teacher * loss_degree

        elif mode == "bandwidth_mask":
            # Bandana-style continuous edge bandwidth masking
            bandwidth = sample_continuous_bandwidths(E_b)
            E_weighted = apply_bandwidth(E_b, bandwidth)

            Z_layers = teacher_gnn(
                X_b,
                E_weighted,
                return_all_layers=True
            )

            bandwidth_pred = bandwidth_decoder(Z_layers, E_b)

            loss = mse(bandwidth_pred, bandwidth)

        optimizer_teacher.zero_grad()
        loss.backward()
        clip_grad_norm_(teacher.parameters(), max_norm=1.0)
        optimizer_teacher.step()
```

Teacher 的目的不是最终推理，而是让 Student 学到 topology-aware inductive bias。masked_edge 文件给出的 transfer recipe 也是：在 train-only PPI graph 上 mask training edges、用 visible graph 编码、decode masked edges，并可加 degree auxiliary / path-wise masking。


# 9. Forward/backward pseudocode: Student training

```python
def train_student_epoch(train_graph, teacher, esm_embeddings):
    """
    Student input is feature-only.
    Training graph is used only to compute losses.
    Teacher is frozen.
    """

    teacher.eval()

    for S_b in sample_training_subgraphs(train_graph):
        X_b = esm_embeddings[S_b]

        # Student feature-only forward
        out = forward_student(S_b, X_b)
        P = out["P"]
        C = out["C"]
        m_hat = out["m_hat"]
        Qmod = out["Qmod"]

        # Ground-truth train subgraph, used as label only
        A_b = induced_adjacency(train_graph, S_b)
        E_pos = positive_edges(A_b)

        # Keep positives inside candidate set.
        E_pos_c = intersect(E_pos, C)

        # Add positive rescue: ensure some train positives are in C during training.
        # At test this rescue is disabled.
        C_train = union(C, sample_missing_positive_candidates(E_pos))
        P = rescore_if_needed(out, C_train)

        E_neg = sample_negatives(
            nodes=S_b,
            exclude_edges=positive_edges(A_b),
            candidates=C_train,
            ratio=neg_per_pos
        )

        # ----- Pairwise supervised losses -----
        loss_edge = balanced_bce(P, E_pos_c, E_neg)
        loss_rank = margin_ranking_loss(P, E_pos_c, E_neg)

        # ----- Teacher distillation -----
        with no_grad():
            T_logits = teacher_score_pairs(teacher, S_b, C_train)
            T_prob = sigmoid(T_logits)

        loss_teacher = soft_bce(P[C_train], T_prob)

        # ----- Topology losses from soft P -----
        d_hat = soft_degree(P, C_train, nodes=S_b)
        d_true = degree_vector(A_b)

        m_true = len(E_pos)

        loss_budget = huber(log1p(sum(P.values())), log1p(m_true))
        loss_degree = mmd(log1p(d_hat), log1p(d_true))

        C_hat = soft_clustering(P, C_train)
        C_true = clustering_coefficient(A_b)
        loss_clust = abs(C_hat - C_true)

        loss_spec = spectral_loss_sampled(P, A_b, C_train)

        loss_module = modularity_loss(Qmod, A_b) + balance_regularizer(Qmod)

        loss_cal = brier_score(P, A_b, C_train)

        loss_sparse = mean(P.values())

        # ----- Total -----
        loss = (
            loss_edge
            + lambda_rank * loss_rank
            + lambda_teacher * loss_teacher
            + lambda_budget * loss_budget
            + lambda_degree * loss_degree
            + lambda_clust * loss_clust
            + lambda_spec * loss_spec
            + lambda_module * loss_module
            + lambda_cal * loss_cal
            + lambda_sparse * loss_sparse
        )

        optimizer_student.zero_grad()
        loss.backward()
        clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer_student.step()
```

注意这里的 `sample_missing_positive_candidates(E_pos)` 只在训练阶段用于防止 candidate proposer 早期漏掉太多 positives；test 阶段不能使用，因为 test positives 不可见。


# 10. Validation algorithm

```python
def validate(val_graph, esm_embeddings, threshold_grid, beta):
    """
    Validation graph is used only for metrics and calibration.
    It is not passed into forward_student().
    """

    scores = []

    for S_v in sample_validation_subgraphs(val_graph):
        X_v = esm_embeddings[S_v]

        out = forward_student(S_v, X_v)

        P = out["P"]
        C = out["C"]
        m_hat = out["m_hat"]

        A_true = induced_adjacency(val_graph, S_v)

        for rule in ["top_m_hat", "fixed_threshold", "density_matched_val"]:
            A_pred = assemble_graph(P, C, m_hat, rule)

            pair_metrics = compute_pair_metrics(P, A_true, C)
            topo_metrics = compute_topology_metrics(A_pred, A_true)

            score = (
                pair_metrics["AUPR"]
                - beta["density"] * abs(log(topo_metrics["relative_density"]))
                - beta["degree"] * topo_metrics["degree_mmd"]
                - beta["clust"] * topo_metrics["clustering_mmd"]
                - beta["spec"] * topo_metrics["spectral_distance"]
            )

            scores.append((score, rule, pair_metrics, topo_metrics))

    return select_best_rule_and_checkpoint(scores)
```

Validation 可以使用 true graph 计算 metrics、选择 threshold、选择 checkpoint；但 validation graph 不能进入 model forward。


# 11. Test inference algorithm

```python
def test_inference(unseen_proteins, esm_embeddings, selected_rule):
    """
    No target graph object exists in this function.
    """

    X = esm_embeddings[unseen_proteins]

    out = forward_student(unseen_proteins, X)

    P = out["P"]
    C = out["C"]
    m_hat = out["m_hat"]

    if selected_rule == "top_m_hat":
        E_hat = top_k_edges(P, k=round(m_hat))

    elif selected_rule == "fixed_threshold":
        E_hat = [(i, j) for (i, j), p in P.items() if p > t_val]

    elif selected_rule == "hybrid":
        E_hat = constrained_top_k(
            P,
            k=round(m_hat),
            max_degree_soft_cap=True
        )

    return {
        "edge_probabilities": P,
        "predicted_edges": E_hat,
        "predicted_graph": Graph(nodes=unseen_proteins, edges=E_hat),
        "edge_budget": m_hat
    }
```

Test-time 不运行 teacher，不 mask target edges，不做 target graph reconstruction pretext，不读取 target topology。


# 12. Exact role of MGAE / S2GAE / MaskGAE / Bandana

| Method family | 在 proposed model 中的角色 | 是否 test-time 使用 | 原因 |
| ------------- | ------------------------- | ----------------- | ---- |
| MGAE | baseline / warm-up corruption mode | 否 | 高比例 random edge masking 是简单强 topology pretext |
| S2GAE | primary teacher pretraining objective | 否 | direction-aware masked-edge reconstruction 最直接对齐 link/topology recovery |
| MaskGAE | auxiliary teacher objective | 否 | path-wise masking + degree regression 帮助 motifs、hub、local pathway structure |
| Bandana | optional robustness corruption | 否 | continuous bandwidth mask 避免 binary deletion 破坏 connectivity |
| GraphMAE | only auxiliary feature denoising or contrast class | 否 | primary target 是 topology，不是 node feature reconstruction |

masked_edge 文件也把直接相关 shortlist 总结为 VGAE baseline + MGAE / MaskGAE / S2GAE / Bandana，而 GraphMAE 主要作为“从 topology pretext 切到 feature pretext 会怎样”的 contrast class。


# 13. Why this architecture is not just pairwise classification

普通 pairwise classifier：

$$
P_{ij}=\sigma(f(x_i,x_j))
$$

TopoPPI-Gen：

$$
P_{ij}
=
\sigma
\left(
s_{\text{pair}}
+
\alpha_i+\alpha_j
+
u_i^\top u_j
+
q_i^\top B q_j
+
b_S
\right)
$$

再加：

```text
learned edge budget
soft degree loss
density loss
clustering loss
spectral loss
module/community loss
teacher topology distillation
self-refinement over predicted graph
```

所以 edge decisions 不是 independent。它们被 shared global variables $b_S,\hat{m}_S$、node hub propensities $\alpha_i$、module memberships $q_i$、soft topology losses 共同耦合。

换句话说，模型不是“预测很多 pair 再画图”，而是直接学习：

$$
\text{protein set features} \rightarrow \text{biologically plausible sparse interactome}
$$

这正是 GSL-for-PPI 文档中 Route B 的方向：从 node features 做 latent graph inference / graph reconstruction，但需要显式 sparsification 和 graph-level topology objectives，否则容易退化回 pairwise scoring。


# 14. Recommended default hyperparameters

一个可执行的 full-model 默认配置：

```yaml
protein_encoder:
  esm_frozen: true
  projection_dim: 512
  dropout: 0.1

set_encoder:
  type: perceiver_or_set_transformer
  num_layers: 2
  num_latents: 64
  hidden_dim: 512

candidate_proposer:
  top_L_per_node: 128
  anchors: 256
  include_random_exploration_edges_train: true
  positive_rescue_train_only: true

edge_generator:
  pair_mlp_layers: 3
  lowrank_dim: 128
  num_modules: 64
  use_hub_head: true
  use_density_bias: true

self_refinement:
  rounds: 1  # increase to 2 after stable training
  message_passing: sparse_soft_graph_transformer_or_gru

teacher:
  main: s2gae
  mask_ratio: 0.6-0.7
  directed_masking_for_sparse_graphs: true
  path_mask_probability: 0.25
  bandwidth_mask_probability: 0.15
  degree_auxiliary: true

loss_weights_initial:
  edge: 1.0
  rank: 0.1
  teacher: 0.1
  budget: 0.1
  degree: 0.05
  clustering: 0.02
  spectral: 0.0   # warm up later
  module: 0.02
  calibration: 0.01
  sparse: 0.001

loss_schedule:
  epochs_0_10: edge + rank + teacher + budget
  epochs_10_30: add degree + clustering + module
  epochs_30_plus: add spectral with small weight
```


# 15. Practical implementation details

## 15.1 Candidate recall problem

Early in training, feature-only candidate proposer may miss many true positives. During training, use positive rescue:

```text
C_train = C_feature_only ∪ sampled_train_positives
```

But test uses:

```text
C_test = C_feature_only only
```

So candidate proposer must be evaluated separately:

```text
candidate recall@L
positive coverage
average candidates per node
memory/runtime
```

If candidate recall is too low, no decoder can recover missing edges.


## 15.2 Negative sampling after masking

For teacher mask-reconstruct:

```text
mask positives first
then sample negatives excluding all true train positives
```

This avoids overlap between masked positives and sampled negatives.


## 15.3 Topology loss scheduling

Do not activate all topology losses from epoch 1. Otherwise model may learn plausible-looking graph statistics while sacrificing edge evidence.

Recommended curriculum:

```text
Warm-up:
  edge BCE + ranking + teacher distillation + budget

Middle:
  add degree and clustering

Late:
  add spectral and module regularization

Final:
  validation threshold / edge-budget calibration
```


## 15.4 Large graph scaling

For (n>10{,}000):

```text
do not materialize full n x n P
score candidate edges in chunks
compute topology losses on sampled induced subgraphs
use ANN for candidate generation
use anchor buckets to reduce pair space
use sparse tensors for P
```


# 16. Minimal ablation set for this full architecture

Even though you asked for the complete model, these ablations are essential to prove the architecture works:

```text
A0: Pairwise ESM MLP only
A1: Pairwise ESM MLP + top-m density calibration
A2: Student without teacher
A3: Student + S2GAE teacher only
A4: Student + S2GAE + MaskGAE degree/path auxiliary
A5: A4 + Bandana bandwidth corruption
A6: A5 without edge budget
A7: A5 without module term q_i^T B q_j
A8: A5 without topology losses
A9: A5 without self-refinement
```

Success criterion should not be only AUPR. Use:

```text
AUPR / Precision@K
relative density
degree distribution MMD
clustering MMD
spectral distance
component size distribution
module / GO / complex recovery if available
```


# 17. Final concise definition

**TopoPPI-Gen 完整版模型：**

$$
X_S
\rightarrow
H_S
\rightarrow
C_S
\rightarrow
P_S
\rightarrow
\hat{m}_S
\rightarrow
\hat{G}_S
$$

with:

$$
\ell_{ij}
=
\text{MLP}([h_i,h_j,h_i\odot h_j,|h_i-h_j|])
+
\alpha_i+\alpha_j
+
u_i^\top u_j
+
q_i^\top B q_j
+
b_S
$$

Training uses:

$$
\mathcal{L}
=
\mathcal{L}_{edge}
+
\mathcal{L}_{rank}
+
\mathcal{L}_{teacher}
+
\mathcal{L}_{density/budget}
+
\mathcal{L}_{degree}
+
\mathcal{L}_{clustering}
+
\mathcal{L}_{spectral}
+
\mathcal{L}_{module}
+
\mathcal{L}_{calibration}
$$

Teacher uses train-only:

```text
S2GAE masked-edge reconstruction
+ MaskGAE path-wise masking / degree regression
+ optional Bandana bandwidth prediction
```

Test uses only:

```text
ESM / intrinsic protein features
feature-only candidate generation
student edge generator
learned edge budget / validation threshold
```

No target graph input. No target graph message passing. No target degree. No target community. No target Laplacian.

This gives you a model that is still grounded in pairwise biochemical evidence, but whose actual output object is a calibrated sparse PPI network with learned hubness, modularity, density, clustering, and spectral constraints.

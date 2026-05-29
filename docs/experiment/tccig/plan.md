## 0. Executive decision

我建议主推一个 **feature-conditioned, topology-constrained PPI graph generator**，而不是普通 pairwise classifier，也不是直接把 GAE/VGAE 套到 test graph 上。核心形式是：

$$
p_\theta(A_S \mid X_S)
$$

其中 $S$ 是一组 unseen proteins，$X_S$ 是它们的 ESM/PLM embeddings，模型输出整个 edge probability matrix $P_S \in [0,1]^{|S|\times |S|}$，再通过 learned edge budget / validation-calibrated threshold 组装成 sparse PPI graph。训练时可以使用 source/training interactomes 的边作为监督和 topology distribution prior；推理时 **绝不能输入 target-test edges、degrees、neighborhoods 或 community labels**。这正好对应 research_problem.md 中的边界：目标是“from intrinsic protein features”重建 unseen protein set 的 biologically realistic PPI graph，而 target topology 只能作为 evaluation signal，不能作为 input signal。

我给这个方案暂命名为 **TCCIG: Topology-Constrained Conditional Interactome Generator**。它的关键不是“把 pairwise score threshold 一下”，而是把 **edge evidence + graph realism** 同时放进训练目标：pairwise BCE / ranking / PR-AUC surrogate 保证边有生物化学依据，density / degree / clustering / spectral / modularity / functional-module losses 约束组装出来的网络“长得像”真实 PPI graph。


## 1. Research problem boundary

### 1.1 输入、输出、禁止项

给定 unseen protein set：

$$
S={p_1,\dots,p_n}, \quad X_S = [x_1,\dots,x_n]
$$

其中 $x_i$ 来自 ESM/ESM-2/ESM-C/ESM3 或结构衍生 embedding。模型必须输出：

$$
P_{ij}=p_\theta(A_{ij}=1 \mid X_S), \quad i<j
$$

以及一个 thresholded graph：

$$
\hat{G}_S=(S,\hat{E}_S)
$$

禁止输入包括：target-test graph 的 edges、degrees、neighbors、shortest paths、communities、centrality、complex labels。因为这些正是要恢复的对象。research_problem.md 明确把这个任务定义为 inductive graph reconstruction，而不是 observed graph 上的 transductive link prediction。

### 1.2 成功标准不能只看 AUROC/AUPR

已有项目材料给出的核心判断非常重要：PPI prediction 的 biological object 不是 isolated edge decisions，而是一个 sparse、modular、functionally organized interactome；一个 pair classifier 即使 edge metrics 好，也可能生成过密、模块错乱或 hub distorted 的网络。

PRING 进一步把这个 gap 量化了。其 arXiv 摘要说明，PRING 专门从 graph-level perspective 评估 PPI prediction，数据包含 21,484 proteins 和 186,818 interactions，并设置 topology-oriented 与 function-oriented 两类任务。([arXiv][1]) 其官方仓库也明确列出 topology-oriented tasks 包括 intra-species 和 cross-species PPI network generation，function-oriented tasks 包括 protein complex pathway prediction、GO enrichment analysis 和 essential protein justification。([GitHub][2])

因此本项目的模型选择标准应是：

$$
\text{good PPI model} \neq \max \text{AUPR only}
$$

而应是：

$$
\max \text{edge quality} \quad \text{subject to} \quad \text{realistic graph topology}
$$


## 2. 从现有相关工作得到的建模启示

### 2.1 GAE/VGAE 是祖先，但原始形式会违反 inductive boundary

VGAE/GAE 的基本形式是：

$$
Z = \text{GNNEncoder}(X,A), \quad \hat{A}_{ij}=\sigma(z_i^\top z_j)
$$

它天然适合 link reconstruction，但标准 encoder 需要输入 observed adjacency $A$。如果在 target unseen set 上把 target graph 或部分 target edges 输入给 GNN，就违反了你的问题边界。masked_edge 文档也指出，classical GAE/VGAE 的核心模板就是用 GCN encoder 产生 node embeddings，再重建 adjacency。

结论：**GAE/VGAE 可以作为训练思想和 baseline，但不能在 test graph 上使用 target topology 作为 encoder input。**

### 2.2 GraphMAE 不应作为主损失

GraphMAE 的贡献是 masked node-feature reconstruction，而不是结构重建。项目文档明确指出，GraphMAE-style feature reconstruction 对这个任务通常不是 primary pretext，因为 ESM/ESM3 已经提供了强 node features；缺失的是 topology signal。

结论：GraphMAE 可作为 auxiliary feature denoising 或 contrast class，但不应作为主推模型。

### 2.3 MGAE / S2GAE / MaskGAE / Bandana 是最相关的 pretraining family

masked_edge 文档把最相关家族归为 structure-masked variants：MGAE、MaskGAE、S2GAE、Bandana，它们通过 mask edges / paths / bandwidths 学习 missing links、degree 或 connectivity patterns。

其中：

* **S2GAE**：最接近 masked-edge recovery，使用 direction-aware graph masking 和 cross-correlation decoder。文档明确推荐其作为 train-only PPI graph 上的 top choice。
* **MaskGAE**：path-wise masking + degree regression，适合 shared neighbors、short motifs、local pathway structure、hub behavior 等 higher-order topology。
* **MGAE**：高比例 edge masking + cross-correlation decoder，是最小强 baseline。
* **Bandana**：用 continuous edge bandwidth masks 替代 Bernoulli edge deletion，适合 binary edge deletion 太 brittle 的情况。

但关键限制是：这些方法通常仍在 **training graph** 上用 visible graph message passing。对于你的 target unseen protein set，不能输入 target graph。因此它们最适合作为 **training objective / teacher / pretraining signal**，而不是直接作为 target inference architecture。

### 2.4 Latent graph inference / GSL 解决“无图输入时如何生成图”

GSL-for-PPI 文档把 Route B 定义为“inductive graph reconstruction / generation from protein node features, with no trusted graph”，并认为最有前景的是 scalable latent graph inference from node features，尤其 NodeFormer / DGM family。

NodeFormer 的官方摘要说明，它为 large graphs 设计 all-pair message passing，通过 kernelized Gumbel-Softmax 将 latent graph structure learning 的复杂度降到 linear w.r.t. node numbers，并可用于 input graphs missing 的场景。([NeurIPS Proceedings][3]) DGM 的 arXiv 摘要则直接指出，许多 GNN 假设 graph known/fixed，但现实中 graph may be noisy/partially/completely unknown；DGM 在 inductive settings 中从 data 推断 graph，学习 edge probabilities。([arXiv][4])

结论：**主推模型的 inference backbone 应更接近 DGM/NodeFormer/latent graph inference，而不是 transductive GAE。**

### 2.5 Graph-level metrics 应进入训练与模型选择

图生成评估文献指出，graph generative model evaluation 通常依赖 degree coefficients、clustering coefficients、orbit counts 的 MMD，但单个指标会带来模型排序困难，MMD 本身也有 pitfalls。 OpenReview 上的 ICLR 2022 graph generative metrics paper 也强调需要 principled comparison metric，并系统分析 MMD 的问题和 practical recommendations。([OpenReview][5])

OpenGSL 则提醒：GSL 方法在统一 benchmark 下并不总是优于 vanilla GNN，且 learned structure 的 homophily 与 task performance 不一定显著相关。([arXiv][6]) 这意味着你的方案必须避免“用了 GSL 所以一定更好”的叙述，而要用 PRING-style graph metrics 做严格验证。


## 3. 可行建模方向比较

| 方向                                             |                是否满足无 target topology 输入 | 优点                                                  | 致命问题                                                | 结论                                           |
| ---------------------------------------------- | --------------------------------------: | --------------------------------------------------- | --------------------------------------------------- | -------------------------------------------- |
| Pairwise ESM MLP classifier                    |                                       是 | 简单、强 baseline、AUPR 好                                | edge 独立，density/hub/community 无控制                   | 必须做 baseline，但不能主推                           |
| Standard GAE/VGAE                              |                   否，若 test 时输入 $A_{\text{target}}$ | adjacency reconstruction 目标直接                       | target graph input leakage                          | 只作 transductive upper bound 或 training idea  |
| S2GAE / MaskGAE / MGAE                         | 训练时可用 train graph；test 不能用 target graph | topology-first pretraining，masked-edge objective 对齐 | 若直接在 target 上跑需要 target graph                       | 作为 pretraining/teacher，不作为唯一 inference model |
| DGM / NodeFormer latent graph inference        |                                       是 | 从 features 学 latent adjacency                       | 原始目标多为 node classification，不保证 PPI topology realism | 作为主架构骨架                                      |
| Graph diffusion / full graph generator         |                                      可以 | 更像 graph generation                                 | 大图 $N^2$ 成本高，训练复杂，PPI edge sparsity 难               | 第二阶段探索，不作为 MVP                               |
| Route A: pairwise graph → denoising/refinement |    可以，如果初始图由 features-only predictor 生成 | 实用、可控制 topology                                     | 容易退化为 post-processing pairwise scores               | 可作为主方案的 refinement layer 或 ablation          |


## 4. 主推方案：TopoPPI-Gen

### 4.1 总体思想

TopoPPI-Gen 学习：

$$
p_\theta(A_S \mid X_S)
$$

它由四个部分组成：

1. **Protein feature encoder**：把 ESM embeddings 投影成 interaction-aware node states。
2. **Feature-only latent edge generator**：不输入 target graph，用 node features 和 set context 直接生成 $P_{ij}$。
3. **Topology-aware graph assembly layer**：学习 edge budget、degree propensity、module structure，使输出图不是无约束 edge list。
4. **Multi-objective training**：pairwise edge loss + masked-edge/distillation loss + graph topology distribution losses。


## 5. 模型结构

### 5.1 Node encoder：只吃 intrinsic protein features

$$
h_i^{(0)} = \phi(x_i)
$$

其中 $x_i$ 是 frozen 或 lightly fine-tuned ESM embedding。推荐初期 frozen，避免训练集 PPI 太小导致 PLM representation drift。

加入 set context：

$$
c_i = \text{SetEncoder}(h_i^{(0)}, \{h_j^{(0)}:j\in S\})
$$

SetEncoder 可以是 DeepSets、Perceiver、Set Transformer 或 linear attention Transformer。它的作用是让每个 protein 的 edge propensity 受当前 protein set composition 影响，但仍不使用任何 target topology。

最终 node representation：

$$
h_i = \text{MLP}([h_i^{(0)}, c_i])
$$

### 5.2 Edge logit decomposition：不要只有 pair MLP

普通 pairwise classifier 通常是：

$$
\ell_{ij} = f(h_i,h_j)
$$

这容易造成全局 density uncontrolled。TopoPPI-Gen 使用 degree-corrected overlapping block model + pairwise biochemical evidence：

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

其中：

* $s_{\text{pair}}(h_i,h_j)$：pairwise biochemical compatibility，可用 $[h_i,h_j,h_i\odot h_j, |h_i-h_j|]$。
* $\alpha_i$：predicted node sociability / hub propensity，由 intrinsic features 预测，不读取 degree。
* $u_i^\top u_j$：low-rank latent affinity。
* $q_i$：overlapping module membership，$q_i=\text{softplus/softmax}(W h_i)$。
* $B$：module interaction matrix，允许 within-module 和 between-module interaction。
* $b_S$：graph-level density bias，由 set-level representation 预测。

输出：

$$
P_{ij}=\sigma(\ell_{ij}),\quad P_{ij}=P_{ji},\quad P_{ii}=0
$$

这个 decomposition 的优点是：pairwise evidence、hubness、modules、global density 都有独立参数，不会把所有结构压力压在一个 MLP 上。

### 5.3 Sparsification / graph assembly

PPI graph 很稀疏，不能直接把所有高 sigmoid 边都保留。推荐两层机制：

**候选边池：**

训练和推理时只在 candidate set $C_S$ 上打分，避免 $O(n^2)$ 爆炸。候选池可由 learned embedding kNN、approximate nearest neighbor、low-rank top-k 或 block candidate generation 产生。注意：candidate generation 只能用 features，不能用 target graph。

**edge budget head：**

$$
\hat{m}_S = g_\theta(\text{pool}(\{h_i\}))
$$

预测当前 protein set 的 expected edge count。最终图可取 top-$\hat{m}_S$ edges，或在 validation 上选择 threshold $t^*$。这比固定 0.5 threshold 更合理，因为 PPI graph density 极低。

### 5.4 可选：self-refinement without observed graph

可以做一到两轮 self-conditioning：

$$
H^{(1)} = \text{SoftMPNN}(H^{(0)}, P^{(0)})
$$
$$
P^{(1)} = \text{EdgeDecoder}(H^{(1)})
$$

这里 $P^{(0)}$ 是模型自己从 features 生成的 soft adjacency，不是 target graph。因此它不违反 inductive boundary。这个模块相当于“自己生成候选 topology，再用它做一次 soft message passing”。


## 6. Training objectives

训练时从 source/training interactomes 采样 protein subsets $S_b$。模型输入只有 $X_{S_b}$，但 loss 可用训练图的 $A_{S_b}$。对 test/unseen set 不使用 $A$。

总损失：

$$
\mathcal{L}
=
\mathcal{L}_{edge}
+
\lambda_{mask}\mathcal{L}_{mask}
+
\lambda_{dens}\mathcal{L}_{density}
+
\lambda_{deg}\mathcal{L}_{degree}
+
\lambda_{clust}\mathcal{L}_{clustering}
+
\lambda_{spec}\mathcal{L}_{spectral}
+
\lambda_{comm}\mathcal{L}_{community}
+
\lambda_{cal}\mathcal{L}_{calibration}
$$

### 6.1 Pairwise edge loss

用 positive edges + sampled negatives：

$$
\mathcal{L}_{edge}
=
-\sum_{(i,j)\in E^+} w_+ \log P_{ij}
-\sum_{(i,j)\in E^-} w_- \log(1-P_{ij})
$$

建议：

* 使用 focal loss 或 class-balanced BCE；
* negative sampling 包含 random negatives + hard negatives；
* 对未知 negatives 保守处理，因为 PPI 数据有 missing positives；
* 额外加入 ranking loss：positive edge score 应高于 hard negative。

### 6.2 Masked-edge / S2GAE-style training signal

在 training graph 上做 S2GAE/MaskGAE-style mask-reconstruct，但要注意推理模型不能依赖 target graph。因此推荐 teacher-student：

**Teacher**：graph-aware S2GAE/MaskGAE，只在 training graph 上学习 topology-aware representation。

**Student**：TopoPPI-Gen feature-only generator，输入 $X_S$，输出 $P_S$。

蒸馏目标：

$$
\mathcal{L}_{distill}
=
\text{KL}(P^{teacher}_{ij} \mid P^{student}_{ij})
$$

同时 student 直接对 masked positive edges 做 reconstruction loss。这样 masked-edge literature 的 topology signal 会进入模型，但 target inference 仍是 features-only。

### 6.3 Density / edge-budget loss

$$
\hat{m}_S = \sum_{i<j} P_{ij}
$$

$$
\mathcal{L}_{density}
=
\left|
\frac{\hat{m}_S}{\binom{|S|}{2}}
-
\frac{m_S}{\binom{|S|}{2}}
\right|
$$

或者用 log-density loss。该项直接防止 PRING 中常见的 over-dense predicted graph 问题。research_problem.md 中给出的 PLM-interact baseline 在 Human intra-species reconstruction 上 relative density 为 1.64、2.03、1.76，说明过密是现实问题而不是假设。

### 6.4 Degree distribution loss

soft degree：

$$
\hat{d}_i = \sum_j P_{ij}
$$

训练两个目标：

1. node-level degree propensity head：

$$
\mathcal{L}_{degree-node}
=
\sum_i \text{NB/NLL}(d_i \mid \hat{d}_i)
$$

2. graph-level degree distribution matching：

$$
\mathcal{L}_{degree-dist}
=
\text{MMD}(\{\log(1+\hat{d}_i)\},\{\log(1+d_i)\})
$$

这不是输入 target degree，而是在 training graphs 上学习“哪些 protein features 可能对应 hubs”。推理时只由 features 预测 hub propensity。

### 6.5 Clustering / triangle / wedge closure loss

真实 PPI 网络有 complex/pathway-driven local clustering。用 soft triangle count：

$$
\hat{T}=\frac{1}{6}\text{tr}(P^3)
$$

soft wedge count：

$$
\hat{W}=\sum_i \binom{\hat{d}_i}{2}
$$

global transitivity：

$$
\hat{C}=\frac{3\hat{T}}{\hat{W}+\epsilon}
$$

loss：

$$
\mathcal{L}_{clust}
=
|\hat{C}-C|
$$

也可对 local clustering coefficients 做 MMD。这样避免模型只学 pair compatibility 而不学 complex-like closure。

### 6.6 Spectral topology loss

构造 predicted soft Laplacian：

$$
\hat{L}=\hat{D}-P
$$

匹配训练子图和预测子图的 spectral summaries：

$$
\mathcal{L}_{spectral}
=
\text{MMD}(\lambda_k(\hat{L}),\lambda_k(L))
$$

大图上不建议全 eigen-decomposition。可用：

* 小 batch induced subgraphs 上算 top-k eigenvalues；
* Chebyshev / Hutchinson trace estimates：
  $$
  \text{tr}(L^r), r=1,\dots,R
  $$
* heat kernel trace：
  $$
  \text{tr}(\exp(-tL))
  $$

spectral loss 不应一开始就开大权重，否则会牺牲 edge precision。建议 curriculum：edge loss → density/degree → clustering → spectral。

### 6.7 Community / module loss

由模型输出 overlapping module memberships $q_i$。两种可行做法：

**Differentiable modularity objective：**

$$
\mathcal{L}_{comm}
=
-\frac{1}{2m}
\text{Tr}(Q^\top (A-\frac{dd^\top}{2m})Q)
$$

训练时用 true training adjacency；推理时 $Q$ 由 features 预测。

**Edge decoder block term：**

$$
q_i^\top B q_j
$$

直接让 edge probability 受 latent module compatibility 影响。

这和 biology 需求吻合：protein complexes/pathways 是多节点结构，不是独立 pair。graph_structure 文档也强调 GNN4DM、ModulePred 这类 biology-facing work 的价值在于 functional modules、overlap 和 graph augmentation，更接近你的 claim space。

### 6.8 Calibration loss

最后输出需要 calibrated edge probabilities：

* validation temperature scaling；
* Brier score / ECE；
* graph-level threshold calibration；
* report selected threshold 或 selected edge budget。

对于 PPI，推荐主要 selection metric 不是 AUROC，而是 validation 上的 composite score：

$$
\text{Score}
=
\text{AUPR}
-
\beta_1|\log \text{relative density}|
-
\beta_2\text{MMD}_{degree}
-
\beta_3\text{MMD}_{clustering}
-
\beta_4\text{MMD}_{spectral}
+
\beta_5\text{functional module score}
$$


## 7. Inference procedure on unseen proteins

给定 unseen protein set $S$：

1. 计算每个 protein 的 ESM embedding $x_i$。
2. Node encoder 得到 $h_i$。
3. Candidate generator 产生 candidate edge set $C_S$。
4. Edge decoder 输出 $P_{ij}$。
5. Edge budget head 预测 $\hat{m}_S$。
6. 取 top-$\hat{m}_S$ edges，或使用 validation-selected threshold $t^*$：
   $$
   \hat{E}_S={(i,j):P_{ij}>t^*}
   $$
7. 输出：

   * edge probability matrix；
   * thresholded graph；
   * uncertainty intervals，若使用 ensemble / MC dropout / sampled graph distribution。

关键：整个过程没有读取 target-test $A$、degree、neighborhood、community labels。


## 8. 为什么这不是普通 pairwise classifier

普通 pairwise classifier：

$$
P_{ij}=\sigma(f(x_i,x_j))
$$

每条边独立，threshold 后 graph topology 是副产品。

TopoPPI-Gen：

$$
P_{ij}=\sigma(s_{\text{pair}}+\alpha_i+\alpha_j+u_i^\top u_j+q_i^\top B q_j+b_S)
$$

其中 $b_S$、$\alpha_i$、$q_i$、edge budget、density/degree/clustering/spectral losses 让模型直接学习“这批 proteins 应该形成什么样的 graph”。也就是说，graph 是 primary output，不是 pairwise predictions 的被动可视化。research_problem.md 中也强调 graph 不是 passive visualization，而是决定 biological interpretation 是否成立的对象。


## 9. 实验方案

### 9.1 数据划分

基于已有 PRING benchmark，至少做三种 split：

1. **Node-disjoint intra-species split**
   train proteins 与 test proteins 不重叠；test subgraph 的 edges 完全 withheld。

2. **Cross-species split**
   在 human/yeast/ecoli/arabidopsis 等物种间训练和测试，贴近 PRING cross-species PPI network generation。

3. **Cold-start subset reconstruction**
   随机采样 unseen protein subsets，评估局部 interactome reconstruction。

### 9.2 Baselines

必须包括：

1. **Pairwise ESM MLP / bilinear scorer**
   最小 baseline。

2. **Pairwise model + density-calibrated top-K**
   区分“模型改进”与“threshold/edge budget 改进”。

3. **kNN graph from ESM cosine similarity**
   无监督 feature graph baseline。

4. **VGAE/GAE train-only pretraining + pair classifier**
   不能在 target graph 上 message passing；只可用 training graph pretraining。

5. **S2GAE / MaskGAE teacher + pair classifier**
   检验 masked-edge pretraining 是否提升 pairwise 与 topology。

6. **DGM / NodeFormer-style latent graph inference**
   与主推 feature-only graph generator 比较。

7. **Route A variant：pairwise predicted graph → ProGNN/IDGL-like refinement**
   作为 practical denoising baseline。GSL-for-PPI 文档也把 Route A 的强候选列为 Pro-GNN、IDGL、GSR、GNNGUARD、STABLE。

8. **Random graph controls**
   Erdős–Rényi with matched density、configuration model with matched degree distribution、stochastic block model fitted on training graphs。

9. **Transductive GAE upper bound**
   明确标注为 leakage upper bound，不作为公平 baseline。

### 9.3 Pairwise metrics

* AUROC
* AUPR / average precision
* Precision@K
* Recall@K
* F1 at validation threshold
* Brier score / ECE
* calibration curve

由于 PPI class imbalance 严重，primary pairwise metric 应该是 **AUPR / Precision@K**，AUROC 只能辅助。

### 9.4 Graph-level metrics

建议按 PRING 逻辑分两类。

**Topology-oriented：**

* relative density
* edge count error
* degree distribution MMD / KS
* clustering coefficient distribution MMD
* triangle / wedge count error
* spectral MMD / Laplacian eigenvalue distance
* connected component size distribution
* modularity / community count / conductance
* graphlet / motif distribution

**Function-oriented：**

* protein complex recovery
* pathway module recovery
* GO enrichment consistency
* essential protein centrality separation
* predicted neighborhood functional coherence

PRING 官方说明其 tasks 覆盖 topology-oriented network generation 和 function-oriented biological plausibility，这可以直接作为你的评估框架。([GitHub][2])

### 9.5 Ablation studies

必须做：

| Ablation                            | 目的                               |
| ----------------------------------- | -------------------------------- |
| remove topology losses              | 看是否退化为 pairwise classifier       |
| remove density / edge-budget head   | 验证是否过密                           |
| remove degree propensity $\alpha_i$ | 验证 hub modeling                  |
| remove module term $q_i^\top B q_j$ | 验证 community/pathway structure   |
| remove clustering/spectral losses   | 验证 graph realism                 |
| remove S2GAE/MaskGAE teacher        | 验证 masked-edge pretraining 是否有用  |
| no set context                      | 验证 graph-level conditioning 是否必要 |
| threshold-only calibration baseline | 排除“只是 top-K 选得好”的可能              |


## 10. MVP implementation plan

### Phase 1：最小可运行主模型

实现：

$$
\ell_{ij}=s_{\text{pair}}(h_i,h_j)+\alpha_i+\alpha_j+q_i^\top B q_j+b_S
$$

训练 loss：

$$
\mathcal{L}
=
\mathcal{L}_{edge}
+
\lambda_{dens}\mathcal{L}_{density}
+
\lambda_{deg}\mathcal{L}_{degree}
+
\lambda_{clust}\mathcal{L}_{clustering}
$$

先不加 spectral loss，避免训练不稳定。

成功标准：

* AUPR 不低于 pairwise ESM MLP；
* relative density 更接近 1；
* degree/clustering MMD 明显下降；
* Precision@K 不显著下降。

### Phase 2：加入 S2GAE/MaskGAE teacher

在 train-only PPI graph 上训练 S2GAE/MaskGAE teacher，再蒸馏到 feature-only generator。masked_edge 文档明确推荐 S2GAE 作为 top choice，MaskGAE 作为 higher-order topology 重要时的 second choice。

成功标准：

* cold-start edges 的 AUPR/Precision@K 提升；
* graph topology losses 不靠牺牲 edge precision 实现；
* node embeddings 的 module separability 更好。

### Phase 3：加入 spectral / module objectives

加入：

$$
\mathcal{L}_{spectral}
$$

和 differentiable modularity / overlapping community objective。

成功标准：

* PRING-style topology similarity 提升；
* complex/pathway recovery 提升；
* 不出现过度 community collapse。

### Phase 4：scalability

如果 $n\sim 20k$，全 pair scoring 约 $2\times 10^8$ edges，训练不可直接全量。需要：

* ANN top-L candidate pool；
* low-rank/block candidate generation；
* mini-batch induced subgraph training；
* edge chunking at inference；
* optional NodeFormer-like linear latent graph module。NodeFormer 的线性复杂度 latent structure learning 正适合这一扩展方向。([OpenReview][7])


## 13. 最终建议

主推方案不是“GAE for PPI”，而是：

> **Use S2GAE/MaskGAE as topology-aware training signal, use DGM/NodeFormer-style feature-only latent graph inference as the inference backbone, and add explicit graph-level topology regularization so the model generates a calibrated sparse interactome rather than independent edge scores.**

最小强版本就是：

$$
\boxed{
\text{ESM features}
\rightarrow
\text{set-conditioned node encoder}
\rightarrow
\text{degree-corrected module-aware edge generator}
\rightarrow
\text{edge-budget calibrated sparse graph}
}
$$

训练目标：

$$
\boxed{
\text{edge BCE/ranking}
+
\text{density}
+
\text{degree distribution}
+
\text{clustering}
+
\text{masked-edge distillation}
}
$$

这个方案在可执行性、与现有 masked-edge/GSL 文献的连续性、以及对 PRING-style graph-level evaluation 的针对性之间平衡最好。

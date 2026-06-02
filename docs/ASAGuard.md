# ASAGuard: Association-Safe Aggregation for Federated Backdoor Defense
最终方案执行摘要

## 0. 一句话定位
ASAGuard 是一种面向视觉联邦学习后门防御的关联安全聚合框架。它不再把核心问题表述为“如何识别并剔除恶意客户端”，而是表述为“如何阻断全局更新中触发器—目标类别关联的形成”。服务器使用少量干净参考样本构造反事实触发探针，估计后门敏感的功能子空间，并在聚合前将每个客户端更新投影到该子空间的正交补上，从而移除后门关联分量，同时保留大部分正常任务学习信号。
核心公式：Delta_i^perp = (I - U U^T) Delta_i，其中 U 表示由反事实触发探针估计得到的后门敏感子空间基。

## 1. 核心设计动机
现有联邦后门防御大多遵循“检测—降权—剔除”的范式，即通过距离、聚类、符号统计、历史稳定性或验证集表现判断某个客户端更新是否可疑。这类方法在强 non-IID、分布式后门、低比例投毒、持久化后门或自适应攻击下容易失效，因为攻击者可以让恶意更新在参数空间中接近良性更新。
ASAGuard 的核心转向是：恶意更新是否离群并不是后门形成的本质；后门的本质是模型函数中形成了“triggered input -> target label”的条件关联。因此，防御不应只问哪个客户端是恶意的，而应直接问聚合更新中哪一部分功能方向正在增强触发器—目标类别关联。
这一视角将联邦后门防御从 client filtering 转化为 functional association editing。与传统鲁棒聚合相比，ASAGuard 不依赖恶意更新与良性更新在整体统计上的可分性；与触发器检测相比，它也不训练固定视觉模式分类器，而是通过反事实 clean/trigger pair 捕获会导致后门关联增长的一阶功能方向。

## 2. 威胁模型
ASAGuard 面向 targeted federated backdoor attack，尤其是视觉分类任务中的触发器后门。服务器是诚实验证方，维护一个小规模 clean reference buffer，并能在每轮聚合前对上传更新执行轻量级前向与反向计算。
攻击者能力包括：控制部分客户端；任意修改本地数据和标签；任意改变本地训练过程；上传任意模型更新；知道 ASAGuard 的整体算法；使用经典、分布式、持久化、隐蔽或自适应后门攻击。
防御者能力包括：维护少量干净参考样本；每轮秘密生成或采样反事实触发探针；枚举可能目标类别或在高风险类别上计算后门敏感方向；对客户端更新执行功能投影；在不改变客户端训练流程和通信协议的情况下完成聚合。
ASAGuard 不假设恶意客户端会执行任何本地防御逻辑，不要求客户端提交训练日志，不使用 TEE、ZKP 或区块链，也不要求服务器访问客户端原始数据。其安全边界是：如果一个后门主要通过触发器—目标类别关联方向影响全局模型，则该方向应能被反事实探针捕获并被投影移除；若攻击者能构造完全不经过服务器探针可观测关联方向的后门，则属于当前方法的极端失效情形。

## 3. 方法总览
ASAGuard 由三个核心组件构成，且只有一个主方法：Association-Safe Projection Aggregation。
- Counterfactual Probe Bank：服务器从 clean reference buffer 中生成 clean/trigger 成对探针，用于刻画触发器存在与不存在时的功能差异。
- Backdoor-Sensitive Subspace Estimator：服务器基于成对探针计算触发器—目标类别 margin 的梯度差，并得到低维后门敏感子空间 U。
- Association-Safe Projection Aggregator：服务器对每个客户端更新执行 Delta_i^perp = (I - U U^T) Delta_i，再聚合净化后的更新。
整体流程如下：

```text
Server maintains a clean reference buffer B and trigger-family transforms T.
For each FL round t:
  1. Server broadcasts current global model w_t.
  2. Clients train locally and submit updates Delta_i^t.
  3. Server constructs counterfactual probe pairs (x, T_f(x)) from B.
  4. Server computes backdoor-sensitive directions q_{x,f,y}.
  5. Server orthonormalizes these directions to obtain U_t.
  6. Server projects each update: Delta_i^perp = (I - U_t U_t^T) Delta_i.
  7. Server aggregates projected updates and obtains w_{t+1}.
```

## 4. 核心组件一：Counterfactual Probe Bank
Counterfactual Probe Bank 是 ASAGuard 的触发器反事实探针模块。服务器从 clean reference buffer 中采样干净样本 x，并对每个 x 构造多个触发器族变换 T_f(x)。这里的触发器不是为了匹配某个固定攻击模板，而是为了估计“触发器存在”相对于“触发器不存在”会诱导出的目标类别功能方向。

| Trigger Family | 覆盖的攻击类型 | 设计目的 |
| --- | --- | --- |
| Patch / Shape | BadNets, DBA, model replacement 中的局部贴片触发 | 覆盖显式局部触发模式 |
| Blend / Low-alpha | Blended, low-alpha backdoor | 覆盖弱可见或透明触发 |
| Frequency / Sine | SIG, frequency backdoor | 覆盖非局部频域触发 |
| Warping / Geometric | WaNet 与形变类触发 | 覆盖几何变换触发 |
| Model-inverted / Adaptive-like | 面向当前模型优化的触发 | 模拟更强的自适应触发方向 |

Probe Bank 的关键原则是 paired、secret、lightweight 和 family-level。每个触发探针都有对应的 clean probe，从而抵消正常类别学习和 non-IID 造成的共同变化；探针实例只在服务器端生成和使用；探针数量保持在几十到几百个量级，避免过高额外开销；探针族覆盖触发机制而不是枚举每一种具体图案。

## 5. 核心组件二：Backdoor-Sensitive Subspace Estimator
ASAGuard 的关键不是给客户端打风险分，而是估计一个会导致后门关联增长的功能子空间。设 z_c(x; w) 为模型 w 对类别 c 的 logit，目标类 margin 定义为：

```text
M_y(x; w) = z_y(x; w) - max_{c != y} z_c(x; w)
```

对于 clean probe x、触发变换 T_f 和目标类别 y，定义后门敏感方向：

```text
q_{x,f,y} = grad_w M_y(T_f(x); w_t) - grad_w M_y(x; w_t)
```

直观上，q_{x,f,y} 表示：如果模型参数沿该方向更新，则带触发器输入相对于干净输入会更强地提升目标类别 margin。因此，q 捕获的是“触发器—目标类别条件关联”的一阶功能方向。
服务器收集所有探针族、样本和目标类别得到的 q 向量，并通过 QR/SVD 正交化得到低维基矩阵 U。为避免过拟合探针噪声，只保留解释主要关联能量的前 r 个方向。这里的 r 是唯一核心超参数，表示后门敏感子空间维度。
为了可视化该机制，引入关联系数 Association Coefficient：

```text
AC(Delta) = || U^T Delta ||_2 / (|| Delta ||_2 + epsilon)
```

该系数量化某个更新有多少能量落在后门敏感子空间中。ASAGuard 的机制可视化应展示：投影前恶意或聚合更新的 AC 明显较高，投影后 AC 接近零；同时 clean accuracy 不显著下降。

## 6. 核心组件三：Association-Safe Projection Aggregator
服务器构造正交投影算子：

```text
P_t = I - U_t U_t^T
```

对每个客户端更新 Delta_i^t，执行：

```text
Delta_i^{t,perp} = P_t Delta_i^t = Delta_i^t - U_t (U_t^T Delta_i^t)
```

最终聚合为：

```text
w_{t+1} = w_t + (1 / |S_t|) * sum_{i in S_t} Delta_i^{t,perp}
```

该设计不依赖三类规则、多信号加权或复杂阈值。ASAGuard 的主方法就是一个清晰的功能投影聚合：先估计后门敏感子空间，再从每个上传更新中移除该子空间分量，最后聚合净化后的更新。
与简单拒绝可疑更新不同，ASAGuard 并不假设整个客户端更新都是恶意的。即使一个恶意客户端的更新同时包含对主任务有用的成分和后门成分，ASAGuard 也只剥离与触发器—目标类别关联对齐的分量。这一点使其更适合 non-IID 场景，因为 non-IID 良性更新可能在参数空间中表现异常，但未必沿后门敏感方向推动模型。

## 7. 为什么 ASAGuard 构成更明确的新视角
ASAGuard 的新颖性不在于“多做了一个触发器探针模块”，而在于重新定义了联邦后门防御的基本对象：从客户端身份和更新离群性，转向聚合更新中的条件功能关联。

| 已有范式 | 典型方法 | 核心问题 | ASAGuard 的区别 |
| --- | --- | --- | --- |
| 参数空间鲁棒聚合 | Krum, Trimmed Mean, Median, Huber aggregation | 假设恶意更新在距离、坐标或统计分布上可分 | 不依赖整体更新离群性，而是移除后门敏感功能方向 |
| 客户端过滤/选举 | FLAME, Snowball, FedInv | 寻找或选择更可能良性的客户端更新 | 不判断谁恶意，直接净化每个更新中的后门关联分量 |
| 触发器反演/检测 | FLIP, trigger inversion 系方法 | 寻找可能触发器或训练检测器 | 触发器只用于估计功能子空间，不作为固定模式检测器 |
| 协议或结构隔离 | DoBlock 等结构性方法 | 改变模型共享或传播方式以阻断关联 | 不改变客户端协议，通过聚合端投影阻断关联传播 |

因此，论文主线可以明确讲成：Existing defenses ask “which update should be trusted?” ASAGuard asks “which functional association should never be aggregated?” 这是更具有顶会论文辨识度的问题重塑。

## 8. 可视化叙事设计
为了让论文故事不只依赖表格，ASAGuard 应将“后门关联子空间”和“功能投影”具象化。建议主文至少包含以下四类图。

| 图号 | 图名 | 核心展示内容 | 希望传达的信息 |
| --- | --- | --- | --- |
| Figure 1 | Method Overview | 从 clean/trigger pair 到 q 向量、U 子空间、投影聚合的完整流程图 | 方法不是拼接检测器，而是一个单一功能投影聚合原则 |
| Figure 2 | Association Geometry | 二维或 PCA 示意：更新被分解为 clean-learning direction 与 backdoor-association direction | ASAGuard 只移除后门关联分量，而不是粗暴丢弃整个更新 |
| Figure 3 | Association Coefficient over Rounds | 不同防御下 AC 和 ASR 随通信轮次变化的双轴曲线 | AC 下降应领先或同步于 ASR 下降，说明机制有效 |
| Figure 4 | Projection Heatmap | 投影前后每层或每个 q 方向上的投影系数热图 | 被移除的确实是后门敏感分量，而非随机削弱更新 |
| Figure 5 | ACC-ASR Pareto Plot | 横轴 ASR、纵轴 ACC，比对各防御方法 | ASAGuard 应位于低 ASR、高 ACC 的 Pareto 优势区域 |

其中 Figure 3 是最关键的机制图。它应同时展示 ASR 与 Association Coefficient：如果 ASAGuard 有效，投影后 AC 应显著下降并维持在低位，而 ASR 也随之受到抑制。这能直接支撑“后门是功能关联增长，而 ASAGuard 阻断了关联增长”的论文主张。

## 9. 实验设计总览
实验目标是证明四件事：ASAGuard 能降低后门 ASR；ASAGuard 能保持 clean accuracy；ASAGuard 的关联系数确实被投影压低；ASAGuard 在 stealthy、distributed、persistent 和 adaptive 后门下比传统过滤范式更稳定。

### 9.1 数据集与模型

| Dataset | 作用 | 建议位置 |
| --- | --- | --- |
| CIFAR-10 | 标准视觉 FL/backdoor 基础 benchmark，用于主文核心结果和机制图 | 主文 |
| CIFAR-100 | 类别更多，更能体现 non-IID 与目标类别复杂性 | 主文或附录 |
| Tiny-ImageNet | 更复杂视觉场景，用于验证方法扩展性 | 主文精简表或附录 |
| GTSRB | 交通标志场景，适合贴片式触发器现实展示 | 附录 |

模型建议以 ResNet-18 为主，必要时补充 VGG 或轻量 CNN，以避免结果只依赖某个架构。数据划分应覆盖 IID 与 Dirichlet non-IID，主文建议使用 alpha = 0.3 或 0.5，附录补充 alpha = 0.1 和 1.0。

### 9.2 攻击基线

| 主文攻击 | 代表威胁 | 保留理由 |
| --- | --- | --- |
| Model Replacement | 经典 FL 后门模型替换攻击 | 验证方法是否能处理最经典、最强的 targeted FL backdoor |
| DBA | 分布式局部触发后门 | 检验参数异常较弱时的功能投影能力 |
| Neurotoxin / persistent backdoor | 持久化与参数隐蔽后门 | 测试方法是否能处理更耐久、更不易被参数空间防御捕获的攻击 |
| Cerberus / stealthy colluded backdoor | 协同隐蔽后门 | 直接回应恶意更新统计上接近良性更新的情形 |
| Adaptive trigger attack | 攻击者知道防御算法但不知道本轮秘密探针 | 检验 trigger-family 子空间估计的泛化能力 |

### 9.3 防御基线

| 防御方法 | 类别 | 比较意义 |
| --- | --- | --- |
| FedAvg | 无防御基线 | 展示攻击原始强度 |
| Multi-Krum | 距离型鲁棒聚合 | 代表经典 Byzantine-robust aggregation |
| Trimmed Mean / Median | 坐标型鲁棒聚合 | 代表 coordinate-wise robust aggregation |
| RLR | server learning-rate 调节 | 代表轻量服务器端 backdoor defense |
| FLAME | 聚类、裁剪与噪声结合 | 代表经典服务器侧后门防御 |
| FedInv | update inversion 检测 | 代表模型更新反演路线 |
| Snowball | 个体视角双向选举 | 代表近年 AAAI 后门过滤路线 |

### 9.4 评价指标
- ACC：clean test accuracy，用于衡量主任务性能。
- ASR：attack success rate，用于衡量后门攻击成功率。
- Association Coefficient (AC)：聚合更新在后门敏感子空间中的能量占比，用于解释 ASR 降低的机制来源。
- Projected Energy Ratio：每轮被投影移除的能量比例，用于展示 ASAGuard 是否只是轻微编辑而非粗暴削弱更新。
- Runtime Overhead：服务器端额外前向/反向计算与投影开销。
- Communication Overhead：应基本不变，因为 ASAGuard 不要求客户端上传额外对象。

## 10. 主文表图建议

| 表/图 | 内容 | 论文作用 |
| --- | --- | --- |
| Table 1 | 不同数据集与攻击下的 ACC/ASR 主结果 | 证明 ASAGuard 的防御效果与精度保持能力 |
| Table 2 | 与强基线平均比较，含 overhead | 证明方法不是以高开销换效果 |
| Figure 1 | 方法总览图 | 一眼展示新范式：反事实探针 -> 子空间 -> 投影聚合 |
| Figure 2 | Association geometry 示意图 | 解释为什么不需要识别恶意客户端 |
| Figure 3 | ASR 与 AC 随轮次变化曲线 | 机制级证明：投影压低关联，关联降低带来 ASR 降低 |
| Figure 4 | 投影热图与层级分析 | 展示投影具体移除了哪些后门敏感方向 |
| Figure 5 | ACC-ASR Pareto | 展示相较其他防御的综合优势 |

## 11. 推荐标题
推荐最终标题：
- ASAGuard: Association-Safe Aggregation for Federated Backdoor Defense
- Association-Safe Aggregation via Functional Projection for Federated Backdoor Defense
- From Client Filtering to Association Editing: Backdoor-Robust Federated Aggregation
其中第一版最适合作为主标题：简洁、明确，并突出新视角 Association-Safe Aggregation。第三版适合作为引言中的核心叙事句。

## 12. 贡献点
1. A new defense perspective. We reformulate federated backdoor defense as association-safe aggregation, shifting the goal from malicious-client detection to trigger-target association removal.
2. A counterfactual functional subspace. We estimate backdoor-sensitive directions using clean/trigger counterfactual probes and target-margin gradient differences.
3. A single projection-based aggregation rule. We propose a simple and principled aggregator that projects each submitted update away from the estimated backdoor-sensitive subspace before aggregation.
4. Mechanism-level visualization. We introduce association coefficient and projection heatmaps to directly visualize how the method suppresses backdoor association growth.
5. Comprehensive evaluation. We evaluate ASAGuard against classic, distributed, persistent, stealthy, and adaptive federated backdoor attacks under IID and non-IID settings.

## 13. 摘要草案
Federated learning is vulnerable to backdoor attacks launched by malicious clients. Existing defenses commonly identify, downweight, or reject suspicious client updates, but this client-filtering paradigm becomes fragile when malicious updates are statistically indistinguishable from benign ones under non-IID data and stealthy collusion. We propose ASAGuard, an association-safe aggregation framework that defends against federated backdoors by directly removing trigger-target association components from model updates. Instead of asking which client is malicious, ASAGuard asks which functional direction would amplify the target-class margin of triggered inputs relative to their clean counterparts. The server constructs counterfactual clean/trigger probe pairs from a small clean reference buffer, estimates a backdoor-sensitive subspace using target-margin gradient differences, and projects each submitted update onto the orthogonal complement of this subspace before aggregation. This single projection-based rule requires no client-side modification, no trusted execution, and no additional communication. We further introduce association coefficient and projection heatmaps to visualize how backdoor association is suppressed during training. Experiments against classic, distributed, persistent, stealthy, and adaptive federated backdoor attacks are designed to demonstrate that ASAGuard substantially reduces attack success rates while preserving clean accuracy under both IID and non-IID settings.

## 14. 最终判断
ASAGuard 的最终定位应是：一种基于反事实功能投影的关联安全聚合范式，而不是一个由探针、打分、阈值和降权规则拼接而成的工程系统。它的核心概念足够单一：估计后门敏感子空间，并将每个客户端更新投影到其正交补。
该方案的核心定位是 association editing：将防御对象从“客户端”转化为“功能关联方向”，避免回到 server-side client auditing、admission control 或客户端风险评估范式。
后续实现时应优先验证 Figure 3 所需的机制证据：Association Coefficient 是否能被投影稳定压低，并且该指标下降是否与 ASR 下降强相关。若该机制图成立，整篇论文的故事线会非常清晰：后门通过关联增长形成，ASAGuard 通过关联安全投影阻断增长。

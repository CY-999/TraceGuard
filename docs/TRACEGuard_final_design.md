# TRACEGuard: Server-Side Trigger-Family Auditing for Federated Backdoor Defense

**最终方案设计（攻击与防御基线定稿版）**

## 0. 一句话定位

TRACEGuard 是一种面向视觉联邦学习后门防御的服务器侧功能审计框架。它不要求客户端执行任何本地筛选、净化或可信训练流程，而是由服务器在聚合前使用秘密触发族探针直接审计每个客户端上传 update 的功能行为。其核心思想是：一个恶意 update 即使在参数空间中不显著异常，也会选择性增强触发输入上的目标类别响应；TRACEGuard 通过成对触发放大统计量捕获这种行为，并据此对 update 进行接受、降权或拒绝。

该设计直接回应两个核心问题：一是恶意客户端为什么会执行本地筛选；二是方法是否只检测固定白色 trigger。TRACEGuard 的回答是：不要求客户端执行本地筛选，也不依赖单一 trigger 模板。服务器只检查上传 update 的功能行为，并通过秘密触发族探针覆盖 patch、blend/low-alpha、frequency、warping 和 model-inverted/adaptive-like trigger families。

## 1. 核心设计动机

联邦后门攻击的关键挑战在于：攻击者可以控制部分客户端的本地数据和训练过程，并上传看似正常但包含后门行为的模型更新。传统服务器侧防御通常检查 update 的参数距离、梯度方向或聚类异常，但分布式后门攻击会削弱每个恶意客户端的 update-level anomaly，使得纯参数空间检测不稳定。

客户端侧样本筛选看似可以在本地训练前阻断 poisoned samples，但它依赖一个脆弱假设：恶意客户端会诚实执行防御模块。若攻击者已经控制客户端训练流程，它可以关闭筛选器、篡改筛选结果或直接上传 poisoned update。因此，将安全性建立在客户端自愿执行防御上并不稳妥。

TRACEGuard 的设计转向是：不验证客户端本地做了什么，而验证它上传的 update 会造成什么功能行为。服务器为每个候选 update 构造 shadow model，并用秘密触发族探针测试该 update 是否选择性增强 trigger-conditioned target response。这样，防御逻辑从“客户端侧本地筛选”转变为“服务器侧功能审计”。

## 2. 威胁模型

TRACEGuard 面向视觉联邦学习中的 targeted backdoor attack。服务器是诚实验证方，拥有一个小规模 clean reference buffer，并能在每轮聚合前对候选客户端 update 执行轻量前向评估。

攻击者可以：

- 控制部分客户端；
- 任意修改本地数据和标签；
- 使用任意本地训练过程；
- 上传任意模型 update；
- 知道 TRACEGuard 的整体算法；
- 使用经典、分布式、持久化或自适应后门攻击；
- 完全跳过任何客户端侧防御逻辑。

防御者可以：

- 维护小规模 clean reference buffer；
- 每轮秘密生成触发族探针；
- 对每个上传 update 构造 shadow model；
- 比较 update 前后 clean probes 与 trigger probes 的响应变化；
- 在聚合前接受、降权或拒绝异常 update。

TRACEGuard 不假设恶意客户端会执行本地筛选，不要求客户端提交真实训练日志，不使用 TEE、ZKP 或区块链，也不要求服务器知道客户端真实本地数据。其安全边界是：如果一个恶意 update 在服务器秘密触发族探针上表现出 trigger-conditioned target amplification，则它会被降权或拒绝；若攻击者能构造一个在所有秘密探针上无异常、但仍能在真实触发器上产生高 ASR 的 update，则该攻击超出当前功能审计的可观测范围。

## 3. 方法总览

TRACEGuard 由三个组件构成：

1. **Trigger-Family Probe Bank**：服务器从 clean reference buffer 中生成秘密触发族探针。
2. **Update Response Auditor**：服务器将每个客户端 update 应用于 shadow model，并计算成对触发放大分数。
3. **Robust Admission Controller**：服务器基于鲁棒归一化后的风险分数，对 update 进行接受、降权或拒绝。

整体流程如下：

```text
Server maintains a small clean reference buffer B.

For each FL round t:
  1. The server samples a secret trigger-family probe bank P_t from B.
  2. Clients train locally and submit updates Delta w_i^t.
  3. For each update, the server constructs a shadow model:
         w_i' = w_t + Delta w_i^t.
  4. The server evaluates paired clean/trigger probes on w_t and w_i'.
  5. The server computes a paired trigger amplification score R_i^t.
  6. The server robustly normalizes risk scores across clients.
  7. The server accepts, downweights, or rejects updates before aggregation.
```

TRACEGuard 不包含本地净化器。客户端可以使用任意训练流程，服务器只审计其上传 update 的功能后果。

## 4. 核心组件一：Trigger-Family Probe Bank

Trigger-Family Probe Bank 是 TRACEGuard 的探针生成模块。服务器从小规模 clean reference buffer 中采样 clean probes，并为每个 clean probe 生成多个 trigger-family variants。探针族并不是固定白色 patch，而是覆盖视觉后门中常见的触发机制。

| Trigger Family | 覆盖的攻击类型 | 设计目的 |
|---|---|---|
| Patch / Shape | BadNets、DBA、局部贴片 trigger | 覆盖显式局部触发 |
| Blend / Low-alpha | Blended、低可见性 trigger | 覆盖弱可见或透明触发 |
| Frequency / Sine | SIG、频域 trigger | 覆盖非局部频域触发 |
| Warping / Geometric | WaNet、形变类 trigger | 覆盖几何变换触发 |
| Model-inverted / Adaptive-like | 面向当前模型反向优化的触发 | 模拟自适应攻击者 |

每轮训练中，服务器随机采样 location、size、color、alpha、shape、frequency、phase、warp strength 和 target label。因此，攻击者即使知道 TRACEGuard 使用哪些 trigger families，也不知道本轮具体 probe instances。

Probe Bank 的设计原则包括 secret、lightweight、family-level coverage 和 paired。也就是说，探针每轮随机生成、实例不公开，只需要几十到几百个 probes，覆盖触发机制而不是匹配某一个具体 trigger，并且每个 trigger probe 都有对应 clean probe，用于抵消 non-IID 和正常学习带来的混杂因素。

## 5. 核心组件二：Update Response Auditor

服务器收到客户端 update 后，不直接聚合，而是构造 shadow model：

$$
w_i' = w_t + \Delta w_i^t.
$$

对于每个 clean probe $x$ 及其触发族变换 $T_f(x)$，TRACEGuard 比较 update 对 clean probe 与 trigger probe 的目标类 margin 增益差异。定义目标类 margin：

$$
M_y(x; w) = z_y(x; w) - \max_{c \neq y} z_c(x; w),
$$

其中 $z_c(x; w)$ 是模型 $w$ 对类别 $c$ 的 logit，$y$ 是服务器为该 probe 指定的目标类别。对于客户端 update $\Delta w_i^t$，定义成对触发放大统计量：

$$
A_i(x, f) =
\left[M_y(T_f(x); w_t + \Delta w_i^t) - M_y(T_f(x); w_t)\right]
-
\left[M_y(x; w_t + \Delta w_i^t) - M_y(x; w_t)\right].
$$

直观上，$A_i(x,f)$ 衡量该 update 是否相对于 clean probe，选择性增强了 trigger probe 上的目标类 margin。良性 update 可能提升整体模型性能，但不应系统性地只提升 trigger-conditioned target margin。后门 update 的本质是建立 trigger-to-target shortcut，因此会在该统计量上表现异常。

客户端级风险分数定义为：

$$
R_i^t = \operatorname{median}_{x, f} A_i(x, f).
$$

该设计不使用 alpha、beta、gamma 等多信号手工加权项，避免被认为是调参 heuristic。Trigger-clean gap 被直接编码在 paired difference 中；cross-family consistency 通过对多个 probes 和 trigger families 取鲁棒中位数自然体现。Entropy collapse、feature alignment shift 等信号可以作为诊断分析，但不进入默认判决公式。

## 6. 核心组件三：Robust Admission Controller

Robust Admission Controller 将每个客户端 update 的风险分数转化为聚合权重。首先，对同一轮客户端的风险分数做 cohort-wise robust normalization：

$$
z_i^t = \frac{R_i^t - \operatorname{median}_j R_j^t}{\operatorname{MAD}_j(R_j^t) + \epsilon}.
$$

其中 MAD 是 median absolute deviation。相比均值和方差，median/MAD 更不容易被恶意客户端污染。然后，将归一化风险分数映射为聚合权重：

$$
a_i^t = \operatorname{clip}\left(1 - \frac{z_i^t}{\tau}, 0, 1\right).
$$

其中 $\tau$ 由服务器在 benign calibration updates 上设定，用于控制目标 false rejection rate。$a_i^t = 1$ 表示完全接受，$0 < a_i^t < 1$ 表示降权，$a_i^t = 0$ 表示拒绝。最终聚合为：

$$
w_{t+1} = w_t + \frac{\sum_i a_i^t \Delta w_i^t}{\sum_i a_i^t}.
$$

该设计避免了复杂的三类规则或多个手工阈值。接受、降权和拒绝都来自同一个校准风险分数和一个连续权重函数。

## 7. 为什么 TRACEGuard 能回应两个核心疑虑

### 7.1 不依赖客户端执行本地防御

TRACEGuard 的安全判断完全发生在服务器端。客户端可以关闭任何本地防御，也可以任意训练和上传 update。服务器不检查客户端声明，而是检查 update 在秘密触发族探针上的功能行为。因此，TRACEGuard 对“恶意客户端为什么会运行本地筛选？”的回答是：它不需要运行。只要它上传 update，就必须通过服务器侧功能审计。

### 7.2 不依赖固定 trigger 模板

TRACEGuard 不训练一个固定 trigger detector，也不依赖 DBA-style white patch。服务器每轮生成多触发族、随机化、保密的 probes，并使用 paired trigger amplification score 评估 update 的功能响应。因此，其目标不是识别某个视觉 pattern，而是识别 update 是否引入或增强 trigger-conditioned target shortcut。

### 7.3 区分 non-IID 与后门行为

non-IID 客户端可能对某些类别有不同贡献，导致普通参数空间或 logit 变化异常。但 TRACEGuard 使用 paired clean-trigger difference，能够抵消正常类别学习、模型整体进步和客户端数据异质性带来的共同变化。只有当 update 对 trigger probe 的目标类 margin 增强显著高于 clean probe 时，风险分数才会升高。

## 8. 与现有方法的区别

TRACEGuard 不属于传统 client-side sample filtering，也不属于纯参数空间 robust aggregation。它是一种 server-side functional auditing 方法。

- 与客户端侧防御相比，LeadFL、FLIP 等方法主要增强良性客户端或进行 trigger inversion，而 TRACEGuard 不要求客户端执行任何防御逻辑。
- 与服务器侧鲁棒聚合相比，Krum、Trimmed Mean、FLAME 等方法主要检测参数空间或更新分布异常，而 TRACEGuard 检测 update 对 trigger-conditioned behavior 的功能影响。
- 与 trigger inversion 类方法相比，TRACEGuard 将 trigger-family probes 用于 update admission control，即在聚合前审计每个客户端 update。

## 9. 实验设计总览

实验设计的目标是证明三件事：TRACEGuard 能在不信任客户端执行本地防御的情况下抑制后门；TRACEGuard 对分布式、持久化和自适应后门具有鲁棒性；TRACEGuard 在 non-IID 视觉 FL 中保持良好 clean accuracy 和较低额外开销。

### 9.1 数据集

| Dataset | 作用 |
|---|---|
| CIFAR-10 | 标准视觉 FL/backdoor 基础 benchmark |
| CIFAR-100 | 类别更多，更能体现 non-IID 与复杂分类设置 |
| Tiny-ImageNet | 更复杂视觉场景，用于验证方法在高难度数据上的鲁棒性 |

GTSRB 可作为附录或补充实验，用于展示 traffic-sign/sticker-style backdoor 的现实场景，但不作为主文核心数据集。

### 9.2 数据划分

实验应覆盖 IID 与 non-IID。主实验建议使用 Dirichlet non-IID split，例如 alpha = 0.3 或 alpha = 0.5；更极端的 alpha = 0.1 和更接近 IID 的 alpha = 1.0 可放入附录。

### 9.3 攻击基线（定稿）

主文攻击基线应少而精，并且全部围绕 targeted federated backdoor 展开。不要把 ALIE、IPM、Sign-flipping 等 untargeted Byzantine attacks 混入主表；也不要把 Blended、SIG、WaNet 全部堆进主文主攻击表。TRACEGuard 的核心是 trigger-conditioned functional auditing，因此主文应选择能够代表 FL 后门攻击发展主线的四个攻击。

| 主文攻击 | 代表威胁 | 保留理由 |
|---|---|---|
| Model Replacement | 经典 FL 后门模型替换攻击 | 作为 federated backdoor 的基础参照，测试方法能否处理最经典的 targeted FL backdoor。 |
| DBA | 分布式局部触发后门 | 对齐 distributed trigger 场景，检验参数异常较弱时的功能审计能力。 |
| Neurotoxin | 持久化/参数隐蔽后门 | 测试方法是否能处理更耐久、更不易被参数空间防御捕获的后门 update。 |
| A3FL | 自适应后门攻击 | 直接回应未知与 adaptive trigger 质疑，测试攻击者考虑全局训练动态时的鲁棒性。 |

这四个攻击形成清晰递进：classic -> distributed -> persistent/stealthy -> adaptive。它们比简单枚举 Blended、SIG、Low-alpha 等触发形态更符合主文实验叙事。

附录可补充 BadNets、Blended、SIG/Frequency、Low-alpha Patch、WaNet、ALIE、IPM、Sign-flipping、Min-Max/Min-Sum。其中，前五个用于 trigger-family generalization analysis，后四个用于 Byzantine stress test，但不作为主 claim。

### 9.4 防御基线（定稿）

主文防御基线应控制在 6 个核心方法以内，并覆盖无防御、经典鲁棒聚合、经典 FL 后门防御、近年顶会 client-hardening 路线和 heterogeneous FL 后门防御。最终主文比较方法如下：

| 主文防御 | 类别 | 保留理由 |
|---|---|---|
| FedAvg | 无防御基线 | 显示攻击原始强度，是所有 FL backdoor 实验的必要参照。 |
| Multi-Krum | 经典距离型鲁棒聚合 | 代表 Byzantine-robust aggregation 中的距离选择类方法。 |
| Trimmed Mean | 经典坐标型鲁棒聚合 | 代表 coordinate-wise robust aggregation，与 Multi-Krum 互补。 |
| FLAME | 经典服务器侧 FL 后门防御 | 代表 clustering、clipping 和 noise 结合的服务器侧 backdoor defense。 |
| FLIP | 近年顶会 trigger inversion / client-hardening 防御 | 代表通过触发器反演和良性客户端硬化进行后门缓解的路线。 |
| FDCR | 近年顶会 heterogeneous FL 后门防御 | 代表在 non-IID / heterogeneous FL 中利用参数重要性差异进行防御的强基线。 |

主文最终比较列表为：FedAvg、Multi-Krum、Trimmed Mean、FLAME、FLIP、FDCR、TRACEGuard。该组合精炼且覆盖类别完整，避免 baseline 过多导致主线分散。

附录可补充 Median、FoolsGold、FLTrust、Lockdown、DeepSight、RFLBAT、FedGrad、Fedward。其中 Median 与 Trimmed Mean 功能接近，FoolsGold 更偏 sybil/similarity setting，FLTrust 需要服务器 trusted data 假设，Lockdown 改变训练协议较多，因此更适合作为附录 additional comparison。

### 9.5 评价指标

- ACC：clean test accuracy；
- ASR：attack success rate；
- Backdoor Risk Score：TRACEGuard 的 normalized risk；
- False Rejection Rate：良性客户端被拒绝比例；
- Malicious Rejection / Downweight Rate：恶意 update 被拒绝或降权比例；
- Aggregation Overhead：每轮额外前向评估开销；
- Communication Overhead：应基本不变，因为 TRACEGuard 不要求客户端上传额外大对象。

### 9.6 主文表图建议

主文建议包含以下表图：

- **Table 1: Main performance across datasets and attacks.** 展示 CIFAR-10、CIFAR-100、Tiny-ImageNet 上四个主文攻击下各防御的 ACC/ASR。
- **Table 2: Average comparison with strong defenses.** 重点比较 FedAvg、Multi-Krum、Trimmed Mean、FLAME、FLIP、FDCR 和 TRACEGuard 的平均 ACC/ASR。
- **Figure 1: Method overview.** 展示服务器如何用 secret trigger-family probes 审计客户端 update。
- **Figure 2: Risk score separation.** 展示 benign 与 malicious updates 在 normalized risk score 上的分离。
- **Figure 3: Adaptive and persistent attack robustness.** 展示 TRACEGuard 在 Neurotoxin 和 A3FL 下的 ASR 抑制效果。
- **Figure 4: ASR-ACC trade-off.** 展示 TRACEGuard 相比其他 defense 的 Pareto 优势。

### 9.7 附录实验建议

- 完整 baseline × attack × dataset 表；
- trigger-family generalization：BadNets、Blended、SIG/Frequency、Low-alpha、WaNet；
- Byzantine stress tests：ALIE、IPM、Sign-flipping、Min-Max/Min-Sum；
- malicious ratio：4%、10%、20%、30%；
- non-IID severity：Dirichlet alpha = 0.1、0.3、0.5、1.0；
- probe bank size：16、32、64、128；
- trigger family ablation：去除 patch/blend/frequency/warping/adaptive-like；
- runtime overhead、false rejection analysis、risk score over training rounds。

## 10. 推荐标题

推荐最终标题：

> TRACEGuard: Server-Side Trigger-Family Auditing for Federated Backdoor Defense

该标题同时突出 TRACEGuard、server-side、trigger-family、auditing 和 federated backdoor defense，适合作为顶会论文标题。

## 11. 贡献点

1. **A server-side functional auditing paradigm.** We reformulate federated backdoor defense as update-level functional auditing rather than client-side sample filtering or parameter-space anomaly detection.
2. **Secret trigger-family probe bank.** We design a lightweight and randomized trigger-family probe bank that covers patch, blend, frequency, warping, and adaptive-like visual triggers without relying on a fixed trigger template.
3. **Paired trigger amplification score.** We propose a single paired statistic that measures whether a client update selectively amplifies trigger-conditioned target margins relative to clean probes, avoiding heuristic multi-signal weighting.
4. **Robust admission control.** We convert update risk into aggregation weights through cohort-wise robust normalization and calibrated admission, enabling continuous accept/downweight/reject decisions.
5. **Comprehensive evaluation.** We evaluate TRACEGuard against Model Replacement, DBA, Neurotoxin and A3FL across CIFAR-10, CIFAR-100 and Tiny-ImageNet, comparing with FedAvg, Multi-Krum, Trimmed Mean, FLAME, FLIP and FDCR.

## 12. 摘要草案

Federated learning is vulnerable to backdoor attacks launched by malicious clients. Existing client-side sample filtering defenses attempt to remove poisoned samples before local training, but they rely on a fragile assumption: compromised clients are expected to execute the deployed filter faithfully. Moreover, detectors trained on fixed trigger patterns often fail to generalize to unseen or adaptive triggers. We propose TRACEGuard, a server-side trigger-family auditing framework for federated backdoor defense. Instead of trusting clients to perform local purification, TRACEGuard audits each submitted update using a secret trigger-family probe bank maintained by the server. For each candidate update, the server applies it to a shadow model and computes a paired trigger amplification score, which measures whether the update selectively increases target-class margins on trigger probes relative to clean probes. The resulting risk scores are robustly normalized across clients and converted into aggregation weights, allowing suspicious updates to be downweighted or rejected before aggregation. TRACEGuard requires no trusted client-side execution, no hardware attestation, and no additional client communication. Experiments on CIFAR-10, CIFAR-100, and Tiny-ImageNet against Model Replacement, DBA, Neurotoxin, and A3FL demonstrate that TRACEGuard substantially reduces attack success rates while preserving clean accuracy and incurring only lightweight server-side evaluation overhead.

## 13. 最终判断

TRACEGuard 的最终版应被定位为一种服务器侧、无需信任客户端执行防御、基于秘密触发族探针的功能性 update 审计方法。它不再延续客户端样本过滤的脆弱假设，而是保留原思路中关于“触发器会造成局部/功能响应异常”的核心洞察，并将其提升为服务器端 update admission control。

最终方案的关键特征是：无本地净化器、无多版本设计、无 TEE/ZKP 强假设、无多信号手工加权、只使用一个成对触发放大分数、用秘密触发族探针解决 trigger generalization、用服务器侧功能审计解决 client bypass、用 robust admission control 保证 non-IID 下的稳定性。

最终攻击和防御基线也已收敛为顶会主文风格：主文攻击为 Model Replacement、DBA、Neurotoxin、A3FL；主文防御为 FedAvg、Multi-Krum、Trimmed Mean、FLAME、FLIP、FDCR 和 TRACEGuard。这样既保证代表性，又避免实验设计过度混杂。

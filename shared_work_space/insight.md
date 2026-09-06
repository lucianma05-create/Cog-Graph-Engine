# BDI-E：基于图状认知状态的主动对话用户模拟器

## 1. 研究背景

情感支持、劝说捐赠、讨价还价等主动对话任务，本质上都是长期、多轮、用户状态依赖的序列决策问题：agent 并不只是需要生成一条"当前看起来合适"的回复，而是需要通过连续交互逐步改变用户的认知、情绪和行为倾向，从而推动长期任务目标。

例如，在情感支持中，一个有效的对话过程可能经历：

$$
\text{负面情绪}
\rightarrow
\text{被理解}
\rightarrow
\text{重新评价问题}
\rightarrow
\text{恢复控制感}
\rightarrow
\text{情绪缓解}
$$

在捐赠说服中可能经历：

$$
\text{不感兴趣}
\rightarrow
\text{理解捐赠价值}
\rightarrow
\text{顾虑下降}
\rightarrow
\text{产生捐赠意愿}
\rightarrow
\text{采取行动}
$$

讨价还价则可能经历：

$$
\text{价格分歧}
\rightarrow
\text{推断对方约束}
\rightarrow
\text{调整价格预期}
\rightarrow
\text{形成接受意向}
\rightarrow
\text{达成交易}
$$

训练和评估这类 agent 需要与真实用户反复交互，成本高、不可复现、且存在伦理风险。用户模拟器（user simulator）因此是主动对话研究的关键基础设施：它同时充当 agent 的训练环境与可复现、可诊断的评估环境。

然而，现有用户模拟器大多难以胜任这一角色：

* 早期规则/议程驱动的模拟器仅适用于槽位填充类任务，无法表达认知状态的演化；
* 当前主流的 LLM 用户模拟器通常直接根据 Persona、历史对话和 agent 回复生成下一轮用户话语：

$$
u_{t+1}
\sim
P(
u_{t+1}
\mid
P,H_t,a_t
)
$$

虽然这种方式能够生成语言上自然的用户响应，但模型内部没有一个明确、持续、可追踪的用户认知状态。因此很难回答：

* 当前回复具体改变了用户什么？
* 用户为什么会产生下一轮回复？
* 用户状态是否在朝某一行为目标演化？
* 不同 agent action 是否会导致不同的长期用户状态？

而在主动对话中，用户认知状态的演化恰恰是任务的核心。传统 user simulator 因此更接近一个 conditional response generator，而不是一个真正意义上的动态环境模型。

本文的目标是构建一个面向主动对话的动态环境模型：

$$
\boxed{
\text{Agent Action}
\rightarrow
\text{User Cognitive Transition}
\rightarrow
\text{User Response}
}
$$

即：用图状结构显式维护用户认知状态，每轮由 agent 回复驱动状态演化，再由状态驱动用户话语生成；并配套可视化交互窗口，使认知状态演化过程可见、可调试。

---

## 2. 需要解决的问题

围绕"构建一个可用的主动对话用户模拟器"，本文聚焦四个问题。

### 2.1 状态表示：用什么样的结构化状态刻画用户认知

主动对话中的用户状态既包含慢变化的认知结构（信念、目标、行动意向），也包含快变化的事件反应（对某条回复的评价与情绪）。需要一个最小而统一的本体，在表达能力、跨任务通用性与标注稳定性之间取得平衡。

### 2.2 状态演化：状态如何随对话逐轮、一致地更新

状态必须随 agent 行为而演化，且演化过程本身要可追踪——哪条回复、改变了哪个节点、如何改变。同时，长程对话中若让模型每轮重新生成完整状态，会出现节点 ID 漂移、历史遗忘与前后不一致；更新机制必须增量、可执行、可审计。

### 2.3 状态-话语耦合：话语必须由当前状态驱动

模拟器的用户话语不应只是"像用户说的话"，还必须能被当前认知状态解释（为什么这轮拒绝/犹豫/接受）；反之，状态轨迹也必须足够真实，才能作为未来训练与评估的有效信号。

### 2.4 轻量化：模拟器会被高频调用

用户模拟器在 agent 训练与评估中会被大量、反复调用，每轮推断成本与延迟直接决定其可用性；设计上必须保持轻量。

---

## 3. 核心研究假设

本文提出如下核心假设：

> 主动对话可以被建模为 agent 对用户潜在认知状态进行连续干预的过程。

我们使用一个结构化潜在状态：

$$
z_t=(G_t,A_t,E_t)
$$

描述用户当前状态。

其中：

* $G_t$：持久 BDI 认知图；
* $A_t$：当前轮 Appraisal；
* $E_t$：短期 Emotion。

用户模拟过程写为：

$$
z_{t+1}
\sim
P_\phi(
z_{t+1}
\mid
z_t,H_t,a_t,P,S
)
$$

随后：

$$
u_{t+1}
\sim
P_\phi(
u_{t+1}
\mid
z_{t+1},H_t,a_t,P,S
)
$$

即：

$$
\boxed{
\text{Agent Reply}
\rightarrow
\text{Cognitive Transition}
\rightarrow
\text{User Response}
}
$$

这一 structured latent cognitive state 并非用户真实心理状态，而是环境模型基于当前交互证据维护的认知状态后验估计。

因此，$z_t$ 不是 agent observation，也不是心理学意义上的 ground truth，而是：

$$
\boxed{
\text{simulator-maintained structured latent user state}
}
$$

表示约定：$z_t$ 的内容一律以用户的**第一人称命题**书写（"我的工作压力很大"、"我想保住这份收入"、"我准备报价 80 元"）——图上维护的是"用户自己心中的想法"，而不是"模拟器对用户的第三人称描述"。这一表示方式不影响其认识论定位：$z_t$ 仍是模拟器基于交互证据维护的后验估计，只是以用户心智自身的语言表达，使"话语由状态驱动"的因果链可以直接阅读与审计。

---

## 4. BDI-E 图状用户认知状态

## 4.1 BDI 节点

本文采用 BDI 作为持久认知状态的最小统一 ontology。

### Belief

Belief 表示用户对于世界、自身、其他参与者、环境约束或行动后果的判断。

形式上可表示为一个 proposition：

$$
B_i=(p_i,s_i)
$$

例如：

* "这个慈善组织值得信任。"
* "小额捐赠仍然具有意义。"
* "卖家大概率不会接受 80 元以下的价格。"
* "没有人真正理解我的处境。"

其中 $s_i$ 表示当前 belief strength。

### Desire

Desire 表示用户希望实现或避免的状态。

例如：

* "希望帮助有需要的人。"
* "希望避免浪费钱。"
* "希望尽快卖出商品。"
* "希望自己的情绪被理解。"

Desire 本身不等价于具体行动。

### Intention

Intention 表示用户对具体行为形成的行动倾向或承诺。

例如：

* "愿意捐赠 10 元。"
* "准备继续看其他工作机会。"
* "愿意接受 80 元成交。"
* "准备明天和朋友谈谈。"

从结构上可以理解为：

$$
\text{Belief}+\text{Desire}
\rightarrow
\text{Intention}
$$

节点内容统一以第一人称书写：Belief 写作用户持有的命题本身（或"I think/believe …"），Desire 写作"I want / I want to avoid …"，Intention 写作"I will / I am ready to …"。图上的边刻画的是用户自身的实践推理结构（慎思、手段-目的推理），而不是观察者对用户认知的元判断。

---

## 4.2 节点属性

每个 BDI 节点包含：

* `type`
* `content`
* `strength`

其中 $strength_i\in[0,4]$ 由 LLM 直接给出：0–1 弱/试探性，2–3 明确存在，3.5–4 核心/稳定承诺。

早期版本曾使用 0–4 五档概率分布推导强度（期望 + 熵 + 按分布采样）。2026-09-06 精简审查确认：只有期望被消费，熵与采样从未实现，属声明性冗余——按"如无必要勿增实体"原则改为直接输出强度。

---

## 4.3 图关系

为保证跨任务通用性与标注稳定性，本文只保留对状态转移有直接作用的少量关系：

* `facilitates`：一个状态会促进另一个状态；
* `inhibits`：一个状态会抑制另一个状态；
* `means_for`：某个 Intention 是实现某个 Desire 的具体方式；
* `conflicts_with`：两个 Desire 在当前情境下存在明显目标冲突。

其中，`facilitates` 和 `inhibits` 遵循认知影响的前向传递方向（B → D → I），只允许：

$$
B\xrightarrow{}D,\quad
D\xrightarrow{}I,\quad
B\xrightarrow{}I
$$

同层（B→B、D→D、I→I）与逆向（D→B、I→B、I→D）边由引擎拒绝。例如：

$$
B_1
\xrightarrow{\text{facilitates}}
I_1
$$

表示某一 Belief 会增强相应的行动意向；

$$
B_2
\xrightarrow{\text{inhibits}}
D_1
$$

表示某一 Belief 会抑制某个目标或需求。

三个正向方向对应三种不同的实践推理机制：$B\xrightarrow{}D$ 刻画信念对目标合意性的评价（desirability appraisal），$D\xrightarrow{}I$ 刻画慎思（欲望的强度驱动行动承诺，deliberation），$B\xrightarrow{}I$ 刻画手段评估（信念直接作用于行动本身的可行性、代价或手段质量，means evaluation）。$I\xrightarrow{}B$ 在任何情况下都不合法：意向不产生信念——从意向反推信念属于一厢情愿的推断（Bratman 不对称论题）；意向对信念只存在一致性约束（不能一边打算做某事、一边相信它不可能），不存在生成性影响。

Intention 与 Desire 之间使用：

$$
I_1
\xrightarrow{\text{means\_for}}
D_1
$$

表示该 Intention 是实现 Desire 的具体行动方式。

当两个较强 Desire 难以同时满足时，可以表示为：

$$
D_1
\xleftrightarrow{\text{conflicts\_with}}
D_2
$$

该冲突可以进一步影响当前轮 Appraisal 中的 `goal_conflict`，并间接影响 Emotion。


## 5. Appraisal 与 Emotion

Emotion 不作为长期 BDI 节点，而被看作 agent action 与 BDI 状态共同作用下产生的短期 affective response。

定义：

$$
BDI_t+a_t
\rightarrow
A_{t+1}
\rightarrow
E_{t+1}
$$

Appraisal 使用连续属性表示：

$$
A_t=
(
goal\_congruence,
controllability,
goal\_conflict
)
$$

Emotion 包含：

* category；
* valence；
* arousal；
* appraisal target。

字段保持最小集（如无必要勿增实体）：`certainty` 由 Belief strength 承载（样本中恒定、模型未使用，已删除）；`intensity` 与 `arousal` 高度共变（环形模型 valence × arousal 已足够表达感受强度，已删除）。

因此，$G_t$ 主要描述慢变化认知状态，而 $A_t,E_t$ 描述快速变化的事件评价和情绪反应。

---

## 6. 初始状态与逐轮状态更新

初始状态由：

$$
G_0=Init(P,S,H_0)
$$

得到。

其中：

* $P$：对话前已知的稳定信息（元信息：自述问题、前测人格、私有谈判立场等）；
* $S$：任务场景；
* $H_0$：未被干预影响的对话前缀——用户自发表达自身状态、agent 尚未开始施加影响（劝说/建议/重构）之前的对话历史。

初始图必须遵循：

$$
\boxed{
\text{minimal evidence-supported initialization}
}
$$

不能利用前缀之后的对话信息反推用户在干预开始时刻的状态。

此外，$G_0$ 是**确定性构造**（temperature=0 + 明确的分步指令），且每个种子首次生成的种子图会被持久化复用：同一种子重复初始化得到同一基础图，保证可复现、可人工审校。

每一轮模型不重新生成完整图，而只生成：

$$
Updates_t
$$

然后由 deterministic updater 执行：

$$
G_t=Apply(G_{t-1},Updates_t)
$$

支持：

* `add`
* `update`
* `deactivate`
* edge `add`
* edge `remove`

这一设计避免 LLM 每轮重新构图导致的 ID 漂移、历史遗忘和状态不一致，同时让每轮状态变化可追踪、可审计。

被拒绝的更新不会静默丢失：updater 拒绝的操作（非法边方向、噪声强度、重复 ID 等）连同原因会被回注到下一轮提示中，使模型心目中的图与真实图保持一致，避免话语与最终状态脱节。

---

## 7. BDI-E 用户模拟器

## 7.1 模型目标

本文的目标模型是一个 action-conditioned cognitive user simulator。

输入：

$$
(G_t,A_t,E_t,H_t,a_t,P,S)
$$

首先预测：

$$
\Delta G_{t+1},A_{t+1},E_{t+1}
$$

再由更新器得到：

$$
G_{t+1}
$$

随后生成：

$$
u_{t+1}
$$

整个模型可以写成：

$$
P_\phi(
\Delta G_{t+1},
A_{t+1},
E_{t+1},
u_{t+1}
\mid
G_t,A_t,E_t,H_t,a_t,P,S
)
$$

在概念上分解为：

$$
P_\phi(
z_{t+1}
\mid
z_t,H_t,a_t
)
$$

以及：

$$
P_\phi(
u_{t+1}
\mid
z_{t+1},H_t,a_t
)
$$

---

## 7.2 轻量化实现

* 每轮单次结构化生成：一次模型调用同时输出更新操作（add / update / deactivate / edge add / edge remove）、$A_{t+1}$、$E_{t+1}$ 与 $u_{t+1}$（结构化 JSON），由确定性执行器落地；
* 认知图与执行器是纯程序组件：模型只负责"读图 + 提出更新"，不负责维护一致性；
* 可选蒸馏：以收集到的（对话，状态轨迹）数据蒸馏小模型，进一步降低高频调用成本。

---

## 8. 模拟器的用途与质量验证

### 用途

* 为主动对话 agent 的训练提供可交互环境（agent 侧训练与奖励分配暂缓）；
* 为 agent 评估提供可复现、可诊断的测试环境：可直接对比不同 agent 带来的用户状态轨迹差异；
* 状态轨迹为未来的过程级奖励设计保留接口。

### 模拟器自身的质量指标

* **行为真实性**：最终结局分布（如捐赠率、成交率、情绪缓解率）与真实对话数据对齐；
* **状态-话语一致性**：每轮话语可由当前状态解释；
* **可操纵性**：改变 Persona / 场景能引起对应的状态与行为变化；
* **多样性**：同一状态可采样出不同但合理的后续轨迹。

### 可视化交互窗口

实时展示每轮状态图的节点、边与强度变化，用于调试、分析与演示。

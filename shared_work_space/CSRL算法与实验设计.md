# CSRL：认知状态引导的强化学习

> **Cognitive-State-Guided Reinforcement Learning**  
> 算法与实验设计，2026-09-07。本文依据当前仓库代码与 PPDPP 设计；明确区分已经实现的用户模拟器与尚待实现的训练模块。文中超参数是预注册起点，不是实验结论。

## 1. 核心问题与论文主线

研究问题是：**在相同模拟成本下，结构化用户认知状态能否帮助我们选择更值得展开的对话节点，并学习更准确的长期价值，从而提高对话策略强化学习的采样效率？**

本文只保留一条在线训练主线和一个离线复用实验：

| 部分 | 学习对象 | 数据来源 | 作用 |
|---|---|---|---|
| CSRL-Planner-RL | 根据公开历史选择离散策略的 planner | 主轨迹＋认知状态引导的树分支 | 主算法，检验采样效率与信用分配 |
| Offline Response-DPO | 直接根据公开历史生成回复的模型 | **原样复用** Planner-RL 已采集的分支 | 次实验，检验同一批 rollout 能否转化为偏好数据 |

不再设置 Planner-Pref，也不为 DPO 维护第二套在线树搜索、价值函数或调度分数。DPO 不反向控制主采样器；它是在 Planner-RL 数据冻结后进行的离线蒸馏实验。因此，两项实验互补但不并列：Planner-RL 回答“认知引导是否提高在线强化学习效率”，DPO 回答“已经支付成本的树分支是否还能训练直接回复模型”。

认知状态 `z` 是训练期特权信息。部署时 planner 和 DPO 模型都只读取合法公开历史 `h` 及 agent 自身任务信息，不读取真实用户图、用户私有 persona 或未来结果。

```mermaid
flowchart LR
    H[公开历史 h] --> P[planner π(a|h)]
    P --> F[冻结回复模型 f(y|h,a)]
    F --> E[Cog-Graph engine]
    E --> H
    E --> Z[训练期认知状态 z]
    H --> V[认知价值集成 V(h,z)]
    Z --> V
    V --> S[选择高不确定节点]
    S --> B[固定 K×R 完整分支]
    B --> C[训练下一轮 V]
    C --> P
    B --> D[离线导出同父回复偏好对]
    D --> M[Response-DPO]
```

### 1.1 为什么需要认知状态

主动多轮对话优化的是后续任务结果，而不是单句流畅度。相同的“我再考虑一下”可能来自价格顾虑、信任不足、目标冲突或尚未形成承诺；公开话语相同并不保证后续转移相同。仓库显式维护 BDI 图、Appraisal 和 Emotion，因而可以检验这些状态是否包含公开历史之外的回报预测信息。

PPDPP 已证明“小型策略规划器＋冻结回复 LLM＋模拟交互回报”是可执行的主动对话训练范式。[PPDPP，ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/29e8437db7b549160ce03d336ff66f65-Paper-Conference.pdf) Tree-GRPO 等工作说明共享前缀和树状 rollout 可提供更细的训练信号。[Tree-GRPO，ICLR 2026](https://openreview.net/pdf?id=ZpQwAFhU13) 因此，本工作的贡献不能只写成“换了模拟器”或“使用了树”，而应由以下证据链支撑：

1. `h+z` 比等容量、等 token 的 `h-only` 模型更准确地预测长期回报；
2. 该增量信息能在相同 rollout 费用下选出更需要补采样的父节点；
3. 更好的采样和价值估计能提高 planner 的学习速度或最终表现；
4. 结论在非本仓库模拟器或人工盲评中仍有保留。

如果只有树相对链式采样的收益，结论应限定为树采样有效；如果认知调度不优于 history-only 调度，则不能把收益归因于 Cognitive-State-Guided。

### 1.2 主要挑战与对应设计

| 挑战 | 当前设计 | 必须报告的证据 |
|---|---|---|
| 内部状态部署时不可见 | actor 只读 `h`，训练期 critic 读 `h+z` | 输入泄漏检查、去掉 `z` 的匹配对照 |
| 终局奖励稀疏且用户反应随机 | 同父状态固定 `K` 个候选，每个候选独立续采样 `R` 次 | 留出价值误差、回报方差、有效分支率 |
| 自适应采样容易引入选择偏差 | 只自适应选择父节点；选中后固定 `K×R`，不按中途结果追加 | 选择概率、失败和费用日志 |
| DPO 与策略梯度目标不同 | DPO 只复用具体回复和长期回报，不参与 actor 更新 | 独立报告 RL 与 DPO，不用一个替代另一个 |
| 模拟器可能自洽但不真实 | 固定外部评价器、环境迁移与人工抽查 | 跨模拟器结果、盲评和偏差探针 |

### 1.3 与相关工作的边界

| 相关工作 | 本文借鉴 | 不能直接继承的结论 |
|---|---|---|
| [PPDPP](https://proceedings.iclr.cc/paper_files/paper/2024/file/29e8437db7b549160ce03d336ff66f65-Paper-Conference.pdf) | planner 控制冻结回复模型，以交互回报训练策略 | 更换用户模拟器本身构成新算法 |
| [DPO](https://arxiv.org/abs/2305.18290) | 参考模型约束下的回复偏好损失 | 用多轮模拟回报生成的偏好自动满足原论文假设或保证全局最优 |
| [Tree-GRPO](https://openreview.net/pdf?id=ZpQwAFhU13) | 完整 agent step 作为树节点，共享前缀 | 本文的父节点选择与 critic 更新自动具有其理论性质 |
| [非对称强化学习](https://arxiv.org/abs/2105.11674) | 训练期特权状态辅助部分可观测策略学习 | 任意认知 critic 都能降低方差，或 `z` 是充分 Markov 状态 |
| [Deep Ensembles](https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) | 多模型分歧作为经验不确定性代理 | 集成方差是严格置信区间或心理不确定性 |
| [Yoon et al., NAACL 2024](https://aclanthology.org/2024.naacl-long.83/) | 用户模拟器与真人行为的偏差需要独立测量 | 图与话语一致就足以证明真实用户有效性 |

主版不引入认知教师、独立 Q 网络、GNN、学习式调度、心理稠密奖励或树分支直接 actor 更新。它们只在文末作为可能扩展。

## 2. 当前用户模拟器的建模与接口

### 2.1 已实现能力与缺口

| 文件 | 当前能力 |
|---|---|
| [schema.py](../engine/schema.py) | Seed、BDI 节点和边、Appraisal、Emotion、工具契约 |
| [prompts.py](../engine/prompts.py) | 初始化、用户转移、条件重写、agent 输入隔离 |
| [updater.py](../engine/updater.py) | 图增量落地、阈值过滤与承诺守卫 |
| [simulator.py](../engine/simulator.py) | `init`、`step`、恢复、审计与日志，当前 `SCHEMA_VERSION=3` |
| [llm.py](../engine/llm.py) | 结构化调用、模型配置与重试 |
| [agent.py](../engine/agent.py) | 普通回复生成，目前没有可训练 planner |
| [eval_tree.py](../tools/eval_tree.py) | 深复制模拟器、共享前缀和手写话术分支 |

当前 engine 已具备 CSRL 所需的核心环境能力：接收一条具体 agent 回复、更新隐藏认知状态、生成用户回复，并可通过深复制从同一父状态展开不同未来。但仓库尚未实现 RL wrapper、任务奖励适配器、认知价值模型、父节点调度器、统一 rollout 记录、DPO 导出器和训练循环。现有代码因此能作为用户模拟环境，不能直接等同于完整 CSRL 训练系统。

### 2.2 Seed 输入与可见性

| 字段 | 用途 | 训练可见性 |
|---|---|---|
| `seed_id` | 缓存、分组、去重 | 不作为模型捷径特征 |
| `task` | emotional_support / persuasion_donation / price_negotiation | 环境与策略 |
| `persona` | 对话前公开用户信息 | actor、环境 |
| `private_persona` | 用户私有立场或目标 | 仅环境 |
| `agent_private` | agent 自身私有约束，如卖方底线 | actor 合法输入 |
| `scenario` | 场景 | actor、环境 |
| `pre_context` / `u0` | 干预前缀或初始用户话语 | actor、环境 |
| `reference_transcript` | 参考后续对话 | 禁止进入在线状态、actor、critic 和 rollout |
| `cognitive_style` | 第一人称认知更新风格 | 仅环境提示词 |
| `cognitive_profile` | 守卫档位 | 仅引擎逻辑 |
| `notes` | 来源与任务元数据 | 白名单使用 |

数据预处理必须先确定干预前缀，再抽取 persona；不得从参考后续对话的成交、捐赠或缓解结果反推初始状态。

### 2.3 认知状态 `z=(G,A,E)`

BDI 节点字段是 `id/type/content/strength`。B 表示信念，D 表示希望实现或避免的状态，I 表示行动承诺；`strength∈[0,4]` 是直接强度，不是概率或置信度。节点 ID 只在单次会话内有效。

| 边类型 | 合法方向 |
|---|---|
| `facilitates` / `inhibits` | B→D、D→I、B→I |
| `means_for` | I→D |
| `conflicts_with` | D↔D |

引擎支持节点新增、更新、失活以及边增删；非法方向会被拒绝。代码不会沿边自动传播强度，状态变化由 LLM 提议后经程序守卫落地。因此 G 是结构化模拟状态，不是真实心理因果图。

Appraisal 包含 `goal_congruence∈[-1,1]`、`controllability∈[0,1]`、`goal_conflict∈[0,1]`。Emotion 包含闭集类别、`valence∈[-1,1]`、`arousal∈[0,1]` 和 `appraisal_target`。初始化将 A 设为 `(0,0.5,0)`，E 设为 neutral、valence 0、arousal 0.2；这是代码默认值，需检查它是否影响早期价值预测。

### 2.4 初始化、单步输入与输出

初始化为

\[
G_0=Init(P,S,H_0;style).
\]

输入包括用户 persona、私有 persona、认知风格、场景和干预前缀。`initialize_cognitive_state` 返回节点和边，`build_initial_graph` 落地状态。初始化模型由 `ANTHROPIC_INIT_MODEL` 指定，temperature 为 0；缓存位于 session 的 `g0/<seed_id>.json`。训练前应把 seed 内容、prompt、模型和 schema 哈希加入缓存键，避免修改种子后误读旧图。

训练环境每轮接收具体回复：

```python
turn = sim.step(agent_reply=y_t)
```

engine 不接收抽象策略 `a_t`。planner 路线必须先调用冻结回复模型 `f(y|h,a)` 得到文本 `y_t`，再交给 engine。单轮执行顺序是：

1. 读取 persona、认知风格、当前 G/A/E、历史、最新 `y_t` 和上一轮审计反馈；
2. 联合生成图更新、A/E、用户话语和终止字段；
3. updater 应用合法图操作、截断范围并执行守卫；
4. 实质性拒绝或警告触发话语重写，重写不再修改图；
5. 提交状态和历史，写入 turn 日志。

正常轮是一次联合结构化生成，不是独立的“先更新认知、再生成话语”两阶段模型。程序校验、API 重试和条件重写可能产生额外调用，成本统计必须覆盖它们。

| 输出 | CSRL 用法 |
|---|---|
| `agent_reply`、`user_utterance` | 更新公开历史 `h` |
| `graph_after`、`deactivated_after` | 构成下一时刻 G，并支持恢复 |
| `appraisal`、`emotion` | 构成下一时刻 A/E |
| `deltas`、`ops_applied` | 状态变化诊断，不能替代完整前后图 |
| `ops_rejected`、`notes` | 审计，不直接作为负奖励 |
| `done`、`done_reason` | 环境终止提示，不自动等于任务成功 |
| `llm_raw`、`retry_raw` | 调试，不进入训练输入 |

模拟器读取完整前缀和最近 8 条新增话语；已有 16 条新增历史后，提示词要求尽快结束，这不是硬截断。训练版应由 wrapper 统一执行 `T_max`，并使所有实验组使用完全相同的时域配置。

## 3. 形式化与模型输入输出

可恢复的完整环境状态写为

\[
x_t=(Seed,G_t,A_t,E_t,history_t,feedback_t,terminal_t,config).
\]

`h_t` 是部署时合法的公开历史和 agent 自身信息；`z_t=(G_t,A_t,E_t)` 是仅供训练的结构化认知状态。环境转移和目标为

\[
(x_{t+1},u_{t+1})\sim\mathcal E(\cdot\mid x_t,y_t),\qquad
J(\theta)=\mathbb E\!\left[\sum_{t=0}^{T-1}\gamma^t r_t\right].
\]

| 模块 | 输入 | 输出 | 是否部署 |
|---|---|---|---|
| planner `πθ` | `h_t` | 离散策略分布 `π(a_t|h_t)` | 是 |
| 冻结回复模型 `f` | `h_t,a_t` | 具体回复 `y_t` | 是 |
| engine `E` | 完整 `x_t,y_t` | `x_{t+1},u_{t+1}` | 训练与评估 |
| 认知 critic 集成 | `h_t,z_t` | `V_m(h_t,z_t)` | 否 |
| 父节点调度器 | 集成分歧、预计费用 | 父节点采样概率 `q_t` | 否 |
| reward adapter | 完整对话、合法任务元信息 | reward、outcome、有效性 | 评估时需要 |
| DPO actor | `h_t` | 直接回复概率 `π_D(y|h_t)` | DPO 实验部署 |

主 critic 不读取 `private_persona`、`cognitive_style` 或 `cognitive_profile`，否则无法区分图状态与其他私有元数据的贡献。可额外设置“全部特权字段”上界，但不能作为 CSRL 主结果。actor 和 DPO prompt 均禁止读取 `z`。

认知编码器序列化活跃节点的 type/content/strength、边、A/E，并使用固定 token 上限。必须记录截断率。主版采用 `M=3` 个独立初始化或按原始案例 bootstrap 的 V 模型；集成标准差只是需要校准的经验不确定性代理。

## 4. 奖励、终止和评价

主实验保持任务奖励固定，使认知状态改变经验采集和价值估计，而不改变优化目标：

\[
r_t=\begin{cases}
-c,&\text{对话继续},\\
R_{task}(\tau),&\text{任务终止或达到时限}.
\end{cases}
\]

启动配置取 `γ=0.99`、`c=0.02`，任务终局奖励归一化到 `[0,1]`；终局轮不重复扣轮成本。

| 任务 | 主终局奖励 | 同时报告 |
|---|---|---|
| 价格谈判，agent 为 seller | 有效成交且满足卖方底线：`0.5+0.5×归一化卖方效用`；否则 0 | 成交率、价格、期望效用、约束违反率 |
| 情感支持 | 冻结 rubric：无改善/恶化 0，部分改善 0.5，显著改善 1 | 支持效果、理解程度、质量盲评 |
| 捐赠劝说 | 对话中明确承诺捐赠为 1，否则 0 | 承诺率、明确金额、质量 |

谈判效用可使用 `clip((price-seller_floor)/(listed_price-seller_floor),0,1)`，前提是元信息可靠且分母为正。捐赠承诺不等于实际支付，模拟支持得分不等于临床效果。

reward adapter 至少输出：

```json
{
  "terminated": false,
  "truncated": false,
  "outcome": "ongoing",
  "reward": -0.02,
  "task_score": null,
  "valid_transition": true,
  "termination_source": null,
  "evaluator_version": "task-rubric-v1"
}
```

API 或 schema 失败在有限重试后标为无效转移，不能伪造成拒绝、成交或成功；调用费用仍计入。`done` 节点不可继续 `step`。engine 终止而任务评价器不判成功时，记录非成功终止及冲突，不为寻找成功继续对话。

谈判评价器必须区分双方明确同意、单方报价、条件式接受和离场，并保存支持判断的文本片段。情感支持评价器采用冻结对话级 rubric；“谢谢”、礼貌收尾或更强的缓解愿望不能直接判为改善。训练 judge 与最终测试 judge 分离，rubric、模型和解码参数在验证集校准后冻结。

## 5. 单一 CSRL rollout 流程

### 5.1 主轨迹

第 `k` 轮开始时冻结 planner `π_k`、回复模型 `f`、实际解码配置、critic 集成 `V_k`、engine 和评价器版本。先从训练案例均匀采样完整主轨迹：

\[
a_t\sim\pi_k(\cdot|h_t),\quad
y_t\sim f(\cdot|h_t,a_t),\quad
x_{t+1}\sim\mathcal E(\cdot|x_t,y_t).
\]

planner 的动作按其合法分类分布采样，保存采样时的 log probability 和 action mask。每个前动作节点保存完整私有快照 `x_t`、合法公开输入 `h_t`、认知输入 `z_t`、策略版本、回复模型版本、回报、失败状态和全部调用成本。

### 5.2 只选择父节点，不自适应改变树形

对主轨迹中的非终止父节点计算

\[
U(x)=\operatorname{Std}_{m=1..M}\big[V_m(h,z)\big],\qquad
S(x)=\frac{\widetilde U(x)}{\widehat C_{remain}(x)+\epsilon}.
\]

`Ũ` 是在独立验证 rollout 上校准后的集成分歧，`Ĉ_remain` 是从该父节点完成一个固定分支的预计费用。父节点选择概率为

\[
q(x)=\eta\frac1{|\mathcal P|}+(1-\eta)
\frac{\exp(S(x)/\tau_s)}{\sum_{x'\in\mathcal P}\exp(S(x')/\tau_s)},
\]

其中 `η=0.2` 是均匀探索起点。warm-up 阶段或校准失败时使用均匀选择。每个案例设置父节点上限，防止某些长对话占满预算。

主版不在 WIDEN、REPEAT、DEEPEN 之间做第二层自适应决策。选中父节点后执行同一预注册协议：

1. 从当前 planner 和冻结回复模型独立采样 `K` 个原始候选 `(a_i,y_i)`；
2. 对每个**具体回复** `y_i`，从同一父快照独立恢复 `R` 次；每次固定首回复为 `y_i`，重采用户反应；
3. 首轮之后统一按当前 `π_k` 和 `f` 续采样，直到任务终止或 `T_max`；
4. 所有候选使用相同 `R`，不因中途均值高低追加、提前停止或删除；
5. 只有完整且有效的分支进入回报标签，失败率和费用仍完整报告。

启动值取 `K=4,R=4`；高方差任务可在先导实验后统一改为 `R=8`。`K`、`R`、最大父节点数和预算必须在比较前冻结。

原始候选必须保留重复项。因为 `(a_i,y_i)` 来自当前行为策略的独立采样，保留策略或回复重复及相同 `R` 后，父节点 Monte Carlo 标签

\[
\widehat V(x)=\frac1{KR}\sum_{i=1}^{K}\sum_{r=1}^{R}G_{i,r}
\]

才对应当前 planner＋回复模型＋续采样规则下的价值估计。为了“增加多样性”而去重、只保留最好候选或按结果分配不同重复数，会改变该估计对象。这里的有限样本估计仍可能有噪声，不能表述为无偏真值。

候选回复的长期价值为

\[
\widehat Q(x,y_i)=\frac1R\sum_{r=1}^{R}G_{i,r}.
\]

同一组 `G_{i,r}` 同时提供父节点 V 标签和 DPO 回复排序，因此不需要第二套 DPO rollout。

### 5.3 统一记录格式

每条树记录至少包含：

```json
{
  "case_id": "train_case_001",
  "iteration": 2,
  "parent_id": "tree_001_node_003",
  "snapshot_id": "private_snapshot_123",
  "public_prompt": [{"role": "user", "content": "..."}],
  "cognitive_state_ref": "private_state_123",
  "parent_selection_probability": 0.02,
  "candidate_id": "candidate_02",
  "strategy": "Provide Information",
  "strategy_logp": -1.37,
  "reply": "...",
  "repeat_id": 3,
  "return": 0.73,
  "outcome": "success",
  "valid_transition": true,
  "policy_version": "planner_iter_2",
  "generator_version": "reply_sft_v1",
  "environment_version": "engine_schema3_prompt_hash",
  "evaluator_version": "task-rubric-v1",
  "cost": {"api_calls": 6, "input_tokens": 0, "output_tokens": 0}
}
```

私有快照和 `z` 与公开 prompt 分开存储；训练 loader 使用显式白名单。共享前缀只存一次，分支引用父节点，统计上始终按原始案例聚类。

## 6. CSRL-Planner-RL 更新

### 6.1 planner 与策略空间

planner 采用文本编码器加分类头，先从训练集内的合法策略标注进行 SFT。冻结回复模型读取 `h_t` 和策略描述 `a_t` 生成具体文本。合法 action mask 只能由公开历史和 agent 自身约束构造，训练、采样和部署保持一致。

谈判可从 PPDPP 的 11 类策略表开始；情感支持可使用 ESConv 的 8 类策略。接入前必须核验当前数据中的原始标签与映射。捐赠任务若没有可靠策略标注，不应临时编造策略表；先把它作为跨任务评估或 Response-DPO 任务，等标注协议确定后再加入 planner 主实验。

### 6.2 actor 更新

actor 只使用均匀抽取案例所产生的主轨迹，树分支不直接进入 actor loss：

\[
G_t=\sum_{j=t}^{T-1}\gamma^{j-t}r_j,\qquad
Adv_t=G_t-\operatorname{stopgrad}(V_k(h_t,z_t)),
\]

\[
\mathcal L_{actor}=-\frac1N\sum_{i=1}^{N}\sum_t
\gamma^t\log\pi_\theta(a_{it}|h_{it})Adv_{it}.
\]

`V_k` 在本轮采样前冻结，不能先拟合当前结果再回填 baseline。固定 `h,z` 时，该 baseline 与本轮抽到的动作无关，因此不会仅因读取隐藏状态就改变 score-function 梯度的期望；它是否降低方差取决于预测质量，必须实测。主版采用一次 on-policy policy-gradient 更新；若以后换 PPO，所有 RL 对照必须同步更换。

### 6.3 critic 更新

本轮 actor 更新后训练 `V_{k+1}`。监督包括：

- 主轨迹各前动作状态的完整 Monte Carlo 回报；
- 被选父节点的 `V̂(x)`；
- 分支在固定首回复之后产生的状态及其从该状态开始的完整回报，因为后缀按当前行为策略采样。

按原始案例均衡采样，主轨迹与树分支各占 batch 的 50% 起步，最多训练 3 epoch，并用**均匀抽取的独立父节点留出集**早停和校准。自适应父节点数据改变训练状态分布，但不应改变每个入选状态的回报定义；主轨迹监督与均匀留出评价用于防止只在高不确定节点上过拟合。

### 6.4 完整伪代码

```text
输入：训练 Seeds、SFT planner、冻结回复模型、engine、reward、总调用预算
均匀 warm-up：采完整主轨迹，训练并校准 V 集成；费用计入

for iteration k:
    冻结 π_k、回复模型、V_k、解码、环境和评价器版本
    D_main = 对均匀抽取案例采样完整 on-policy 主轨迹
    P = D_main 中所有有效、非终止的前动作父节点
    用 V_k 集成分歧/预计费用计算 q(x)，按 q 选择父节点
    对每个入选父节点：
        独立采 K 个原始 (a_i, y_i)
        每个具体 y_i 从父快照固定执行 R 次
        之后按 π_k 和冻结回复模型完整续采样
    用 D_main 的 G - V_k 更新 planner 一次
    用 D_main 和树分支完整回报训练 V_{k+1}
    冻结本轮全部 rollout 记录，供离线 DPO 导出
    记录费用、失败、策略 KL、均匀留出误差和验证表现

输出：部署时只依赖公开 h 的 planner＋回复模型
```

该保守版本让树分支通过下一轮 critic 改善 actor baseline。它有意避免把自适应选择的分支直接塞入 actor loss，因为那需要额外处理父节点选择、动作采样、共享前缀权重和停止规则。分支是否值得其成本必须由等预算学习曲线回答。

## 7. 同一批 rollout 的离线 Response-DPO

### 7.1 为什么可以复用，复用的是什么

每个树父节点已经保存 `K` 个具体回复，并为每个回复执行了相同次数的独立完整后续模拟。因此可以在不再调用 engine 的情况下，从同一父节点构造 `(h,y^+,y^-)`。

复用成立需要同时满足：

1. 两个回复来自完全相同的父快照和公开 prompt；
2. 比较对象是实际执行过的具体文本 `y`，不是只比较抽象策略标签；
3. 每个 `y` 使用相同的固定 `R`，且后缀策略、环境、评价器版本一致；
4. 标签只使用完整有效回报，不使用认知强度变化直接指定 winner；
5. DPO prompt 只保留公开 `h`，移除策略提示 `a`、`z` 和私有快照；
6. planner 的 V 数据保留全部原始候选；DPO 导出阶段才过滤相同或近重复回复。

树中的回复原本由 `a~π(a|h)` 后经 `f(y|h,a)` 生成。DPO 则学习直接策略 `π_D(y|h)`，相当于把 planner 与回复模型的边缘行为及长期结果蒸馏到一个直接回复模型。这是模型级迁移实验，不能与 planner 的绝对分数直接互证。

### 7.2 偏好对导出

对每个父节点：

1. 按规范化文本和语义阈值过滤相同回复；
2. 使用已经保存的 `Q̂(x,y_i)` 对候选排序；
3. 仅当均值差 `Q̂(y^+)-Q̂(y^-)≥δ` 且对两组独立重复回报进行 bootstrap 后，同方向比例达到 `ρ` 时导出；
4. 每父节点最多导出一对，避免长对话或高分支节点主导训练；
5. 不按差值大小加 loss 权重，不在看到结果后追加 rollout，也不把方向相反的候选临时翻转成新实验假设。

启动值为 `δ=0.05,ρ=0.8`，阈值在验证集冻结。小 `R` 下的 top-bottom 选择仍有 winner's curse；应另取少量父节点做高重复审计，报告偏好误标率、可导出率和覆盖分布，不把 bootstrap 过滤称为严格置信保证。

### 7.3 DPO 目标

固定直接回复 SFT 模型为 `π_ref`，并从相同权重初始化 `π_D`：

\[
\mathcal L_{DPO}=-\mathbb E\log\sigma\!\left(\beta\left[
\log\frac{\pi_D(y^+|h)}{\pi_{ref}(y^+|h)}-
\log\frac{\pi_D(y^-|h)}{\pi_{ref}(y^-|h)}
\right]\right).
\]

只计算当前 assistant 回复 token 的序列 log probability，mask prompt；后续用户和 assistant 轨迹只用于形成长期回报，不拼接为 completion。主配置取 `β=0.1`，LoRA rank 16、学习率 `5e-6`，并由验证集选择 checkpoint。[DPO 原论文](https://arxiv.org/abs/2305.18290)

DPO 数据在 Planner-RL rollout 全部冻结后一次性导出。它不迭代生成新偏好、不更新 CSRL 调度器，也不训练独立 DPO critic。

## 8. 实验设置

### 8.1 数据、划分与共同协议

当前少量 seeds 只适合流程验证，不能支撑论文结论。正式实验按原始用户或场景分组划分 train/validation/test，去除近重复 persona 和参考对话；test 在所有阈值冻结后只运行预设 checkpoint。建议每个主任务至少达到数百个训练用户状态，并报告各任务实际规模。

所有组固定：seed 划分、engine/prompt/schema、初始化缓存、回复模型、解码参数、`T_max`、奖励、评价器、错误重试、训练 token 上限和硬件。主效率横轴同时报告：

- engine step 数及成功/失败数；
- 初始化、重写、judge 和重试在内的 API 调用；
- 输入与输出 token；
- wall-clock 与 GPU 小时；
- 主轨迹数、分支数和有效终局数。

首轮配置起点：`T_max=8` 个 agent 回合、`γ=0.99`、轮成本 0.02、每轮 `K=4,R=4`、均匀探索 `η=0.2`、3 个 critic、至少 5 个训练随机种子。超参数只在 validation 调整，并给所有匹配组相同调参次数。

### 8.2 训练前门槛实验

正式训练前先回答两个问题：

**状态价值门槛。** 在均匀父节点上收集高重复完整 rollout，比较等容量、等文本预算的 `V(h)` 与 `V(h,z)`。指标包括 return MSE、MAE、Spearman、校准误差和集成分歧对绝对误差的 AUROC/AUPRC。如果 `h+z` 没有稳定预测增益，CSRL 的认知主张不成立。

**父节点选择门槛。** 用冻结 critic 在独立池中比较均匀、history-only 不确定性和 cognitive-state 不确定性。固定父节点数与每节点 `K×R`，检查所选节点的事后价值误差、补采样后的整体留出 MSE 降幅以及单位调用收益。如果认知选择不优于 history-only，主训练不应继续堆叠更复杂模块。

### 8.3 Planner-RL 主实验

| 组 | 采样和 critic | 回答的问题 |
|---|---|---|
| P0 | PPDPP 风格 chain policy-gradient；普通用户环境 | 外部参考，不作为唯一归因基线 |
| P1 | engine chain，无 learned baseline 或使用 batch 常数 | 换环境后的基础 planner 能否学习 |
| P2 | engine chain＋`V(h)` | 普通 history critic 的收益 |
| P3 | engine chain＋`V(h,z)` | 认知信息仅用于 baseline 是否有益 |
| P4 | engine uniform-tree＋`V(h,z)`，固定 `K×R` | 普通树分支的收益 |
| P5 | engine history-guided tree＋`V(h)` | 不依赖认知图的自适应采样 |
| P6 | engine cognitive-guided tree＋`V(h,z)` | 完整 CSRL |

核心归因不是简单比较 P6 与 P1。`P2 vs P3` 检验认知价值表示，`P4 vs P6` 检验认知父节点选择相对均匀树的收益，`P5 vs P6` 检验结构化认知相对公开历史调度的增量。P4–P6 必须使用相同总调用预算、`K`、`R` 和每案例上限。

主指标是在固定累计调用成本下的任务回报/AUC、达到目标分数的调用数和最终测试回报。机制指标包括留出 V 误差、优势方差、有效样本率、父节点覆盖、策略熵与 KL。树分支只改善 critic 时，若最终策略没有等成本收益，应如实结论为“价值估计改善但不足以偿还采样成本”。

### 8.4 DPO 数据复用实验

DPO 不重新采样。分别从 P4、P5、P6 已保存的树数据导出回复对：

| 组 | 训练数据 | 作用 |
|---|---|---|
| D0 | 直接回复 SFT | DPO 基线与参考模型 |
| D1 | P4 的 uniform-tree 数据 | 普通树 rollout 是否可蒸馏 |
| D2 | P5 的 history-guided 数据 | 公开历史调度生成的数据质量 |
| D3 | P6 的 cognitive-guided 数据 | 认知引导数据是否更有效 |

固定 DPO 初始化、`π_ref`、训练器和最大训练对数。先在相同原始 rollout 成本下比较 D1–D3，再从各数据集中等量下采样偏好对比较标签质量；两种口径分开报告。指标包括独立测试环境任务回报、pairwise win rate、偏好可导出率、审计误标率、任务/风格覆盖和 KL。

这组实验若成功，支撑的是“CSRL rollout 具有可复用的回复监督价值”。它不能单独证明 planner 的策略梯度更好；Planner-RL 主实验仍是核心证据。

### 8.5 必要消融与泛化

最低限度消融包括：

1. `z` 去除、节点强度打乱、边去除、仅 A/E、仅节点集合；
2. cognitive critic 与 history critic 等参数、等 token；
3. cognitive-guided、history-guided 与 uniform 父节点选择等成本；
4. `K∈{2,4,8}`、`R∈{1,4,8}` 的小规模灵敏度，不为单组增加预算；
5. 去掉均匀探索或费用归一化；
6. DPO 不做稳定性过滤、只用单次回报、等量 pair 对照。

独立泛化至少包括一个不由 Cog-Graph Engine 产生的用户环境，或冻结真人对话数据上的离线评价，并加入人工盲评子集。真实用户研究可留到后续，但论文不得把 simulator 内收益表述为真实用户因果效果。

统计以原始案例为聚类单位进行 bootstrap，并报告跨训练种子的均值与区间。共享前缀叶子、同一用户变体和同一回复的 `R` 次重复都不是独立用户样本。

## 9. 实施前必须修复或锁定的问题

| 当前风险 | 代码表现 | 处理要求 |
|---|---|---|
| 调用失败可能被算作成交 | `eval_tree` 异常终止后，部分 outcome 分类默认 deal | 用结构化错误状态与任务结果分离 |
| 重写后的 `done_reason` 可能仍是旧值 | simulator 在重写前保存局部 reason | 提交最终 `out.done_reason` 并做一致性测试 |
| updater 使用重写前的 done | 图更新先于话语重写 | 明确定义终止翻转语义并覆盖测试 |
| 强度阈值文案不一致 | updater 使用 0.4，部分 prompt 写 0.5 | 统一数值并记录 prompt hash |
| G0 缓存缺少版本键 | 当前主要按 `seed_id` 读取 | 加入 seed/prompt/model/schema 哈希 |
| 已终止会话仍可继续 step | simulator 缺少硬拒绝 | wrapper 阻止并测试 |
| prompt 的收尾提示影响时域 | 16 条新增历史后要求尽快结束 | 参数化，与统一 `T_max` 对齐 |
| 成本日志不完整 | turn 日志未覆盖全部 usage/retry | 记录所有调用、重写、judge 和 token |
| actor 与环境模型配置耦合 | 普通回复与用户模拟配置未完全拆开 | 分离可训练 actor、冻结 generator 和 engine 配置 |
| 情感支持存在策略先验 | prompt 对 relief 与快速建议有硬编码倾向 | 做倾听/澄清/建议的环境偏差探针 |

拟新增的最小模块为：

```text
training/csrl/
  env_adapter.py       # reset/fork/step、终止和合法输入
  rewards.py           # 任务奖励与冻结评价协议
  policies.py          # planner、冻结回复模型和 action mask
  state_encoder.py     # h/z 编码及输入隔离
  value_models.py      # V 集成、校准和 policy version
  rollout.py           # 主轨迹和固定 K×R 分支
  scheduler.py         # 单一父节点选择分数与费用约束
  preferences.py       # 从同一树日志离线导出 DPO 对
  train_planner.py     # policy-gradient 与 critic 更新
  train_response.py    # 离线 Response-DPO
  evaluate.py          # 等成本、独立环境和案例级统计
  configs/             # 数据、模型、预算和实验组 manifest
```

实施顺序是：先修复环境终止、缓存和成本记录；再做状态价值与父节点选择门槛实验；通过后实现 P1–P6；最后从已经冻结的 P4–P6 日志导出 D1–D3。DPO 阶段不得再次调用 engine，否则“直接复用同一批 rollout”的实验定义被破坏。

关键验证包括：快照往返后提示词与状态一致；不同分支互不污染；每个具体回复确实从原父快照独立执行 `R` 次；V 数据保留重复候选；DPO prompt 不含策略标签或私有字段；actor baseline 不含当前动作后的信息；失败轨迹不成为正样本；全部重试计费；部署模型不加载 engine 或 critic。

## 10. 论文可支持的结论

只有当 `V(h,z)` 的留出预测优于 `V(h)`、认知选择在等成本下优于均匀和 history-only 选择，并最终改善 planner 学习曲线时，才能称主方法为 **Cognitive-State-Guided Reinforcement Learning**。

如果 D3 同时优于 D1/D2，可以进一步说明认知引导的树 rollout 不仅改善在线 RL，还产生质量更高的长期回复偏好数据。这是数据复用与迁移证据，不是第二个 CSRL 主算法。

如果结果只在当前 engine 内成立，结论应限定为“在结构化认知用户模拟器中的采样效率”；结构化认知状态是模拟器维护的变量，不是用户心理真值，也不构成真实用户因果识别。

## 11. 参考资料

- [PPDPP: Plug-and-Play Policy Planner for Large Language Model Powered Dialogue Agents，ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/29e8437db7b549160ce03d336ff66f65-Paper-Conference.pdf)
- [Direct Preference Optimization，NeurIPS 2023](https://arxiv.org/abs/2305.18290)
- [Tree Search for LLM Agent Reinforcement Learning，ICLR 2026](https://openreview.net/pdf?id=ZpQwAFhU13)
- [Unbiased Asymmetric Reinforcement Learning under Partial Observability](https://arxiv.org/abs/2105.11674)
- [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles，NeurIPS 2017](https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)
- [Evaluating Large Language Models as Generative User Simulators for Conversational Recommendation，NAACL 2024](https://aclanthology.org/2024.naacl-long.83/)
- [Simulated Rewards, Skewed Strategies，AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39348)
- [Towards Emotional Support Dialog Systems，ACL 2021](https://aclanthology.org/2021.acl-long.269/)
- [Decoupling Strategy and Generation in Negotiation Dialogues，EMNLP 2018](https://aclanthology.org/D18-1256/)
- [仓库认知建模说明](insight.md)

## 12. 可能扩展：不属于当前主算法和主实验

### 12.1 认知教师

认知教师是训练时读取 `h+z`、根据模拟用户内部状态提出候选策略或回复的辅助模型。它与本方案中只做估值的 critic 不同，也会直接改变候选分布。当前版本不创建认知教师、不用教师候选、不定义教师损失，也不把它计入 CSRL 主结果。

只有 current-policy 候选下的认知价值与调度收益已经成立，才适合比较 current policy、只读 `h` 的公开教师和读取 `h+z` 的认知教师。教师不得把用户私有底线或未公开事实写入回复；教师候选必须单独完整 rollout，并记录来源和全部费用。由于它改变行为分布，不能直接放入当前 planner 的 on-policy actor loss。

### 12.2 其他扩展

| 扩展 | 启用条件 | 新问题 |
|---|---|---|
| WIDEN/REPEAT/DEEPEN 自适应树形 | 固定 `K×R` 已证明有等成本收益 | 中途选择和停止偏差、额外调参公平性 |
| 树分支直接更新 actor | critic 路线有效但分支利用率成为瓶颈 | 父节点/动作选择校正、共享前缀权重 |
| 独立 Q 网络或 GNN | V-only 出现明确表示瓶颈 | 标签条件、容量和训练成本对照 |
| 心理势函数奖励 | 固定任务奖励主结论已经成立 | 需满足势函数塑形条件，任意认知增量不保证策略不变 |
| 学习式调度或递归搜索 | 简单调度稳定优于基线 | 搜索策略价值与部署策略价值分离 |
| 真实用户研究 | 模拟器和跨环境结果稳定 | 伦理、招募、标注和独立效果评估 |

当前仍需在实施配置中补齐具体 checkpoint、GPU 预算、数据 manifest、人工校准规模和正式测试样本量；这些是实验执行项，不能由算法描述或文献引用替代。

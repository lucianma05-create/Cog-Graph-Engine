# CogWM 改造方案（2026-09-04）

## 1. 改造目标

现有 CogWM 适合根据用户已说出的话识别 BDI/E，但不适合直接作为 RL 环境和奖励源。新建 **agent-reply-conditioned 用户认知状态转移模型**：

\[
(G_t,E_t,u_t)=F(G_{t-1},E_{t-1},P,H_{t-1},a_t)
\]

- `G_t`：跨轮持续的 BDI 认知图。
- `E_t`：当前轮的短期情绪评价状态。
- `P`：分支点前已知、无未来泄漏的用户先验。
- `H_{t-1}`：对话历史；`a_t`：当前 agent 回复。
- `u_t`：下一句用户话语，是 agent 可观察的环境反馈。

保留现有 CogWM-14B 作为独立评估模型，避免破坏已报告的评估结果。

## 2. 核心问题与方案

| 问题 | 解决方案 |
|---|---|
| 状态对当前 agent 回复不敏感 | 用“同一历史+多个反事实回复”监督 `reply → state transition` |
| 每轮独立抽取 BDI | 给状态项分配稳定 ID，按轮更新认知图 |
| B、D、I 并行建模 | Desire 表示目标，Belief 表示判断/约束，二者共同影响 Intention |
| Emotion 与 BDI 当作同类状态 | BDI 持久维护；E 每轮评估，表示分离、转移耦合 |
| 空 BDI 导致奖励稀疏 | 每轮输出转移判断，允许 neutral，增加小权重过程奖励 |
| 离散分数造成奖励突跳 | 预测五档序数概率，用期望值生成连续奖励 |
| 强制非空会虚构变化 | 保留安全但无效回复和真实零转移 |
| 画像使用完整对话 | 按分支点构造 prefix profile，删除未来接受/拒绝、后测和 outcome |
| Big Five 大量为低置信默认值 | 删除单段对话推断的 Big Five，只可选保留真实前测属性 |
| 环境模型可能用话语自证状态 | 独立评估模型从生成话语复核转移，不一致轨迹降权 |

## 3. BDIE 状态定义

BDI 是环境模型基于当前证据得到的用户潜在认知状态后验估计，不是客观心理事实，也不是 agent 已知状态。agent 默认只观察用户话语。

- **Desire**：用户的目标、需求和价值偏好。
- **Belief**：用户对环境、自身、约束和行动后果的判断。
- **Intention**：在 Belief 和 Desire 共同作用下形成的具体行动承诺。

每个 BDI 项记录 `confidence` 和 `observability`：`explicit`（明确表达）、`inferred`（已见证据支持）、`hypothesized`（证据不足，不进入主奖励或降权）。

Emotion 加入图的因果结构，但不作为持久 BDI 节点：

```text
BDI + agent reply → Appraisal → Emotion
```

E 每轮输出情绪类别、`valence`、`arousal`、`intensity` 和 `appraisal_target`。

## 4. 图结构与边

使用 JSON 属性图，包含 `nodes`、`edges`、`appraisal`、`emotion` 和当轮 `updates`。不构建完全图，只标注有明确证据、影响状态更新或奖励解释的边。

### 4.1 最小边集

| 边 | 含义 |
|---|---|
| `facilitates` | 促进，统一 supports/motivates/enables |
| `inhibits` | 抑制，统一 conflicts/constrains/opposes |
| `means_for` | 实现手段，Intention 是实现某个 Desire 的方式 |
| `based_on` | Appraisal 的评价依据 |
| `elicits` | Appraisal 引发 Emotion |

这是为状态转移与奖励计算设计的最小、任务无关关系集，不声称穷尽所有心理关系。

Appraisal 优先使用连续属性，不继续扩张边类型：

```json
{
  "goal_congruence": -0.6,
  "controllability": 0.3,
  "certainty": 0.7,
  "goal_conflict": 0.8
}
```

## 5. 初始图

\[
G_0=Init(P,S,u_0)
\]

- `P`：对话前已知的稳定偏好和约束。
- `S`：任务场景。
- `u_0`：用户第一段有效表达。

初始图是对话起点有证据支持的最小认知图，不是用户所有旧有心理状态，不能用完整对话的 global BDI 倒推。证据不足时允许为空或仅含前置 metadata 明确提供的状态。

## 6. 节点量化

强度表示状态当前对用户认知和行为的影响程度，不表示好坏。

| 强度 | 含义 |
|---:|---|
| 0 | 不存在或已失活 |
| 1 | 微弱、试探性 |
| 2 | 明确存在，但不稳定 |
| 3 | 较强，明显影响判断或行为 |
| 4 | 核心、稳定或已形成承诺 |

- Belief 强度：对命题的确信程度。
- Desire 强度：需求或目标的当前驱动力。
- Intention 强度：对具体行动的承诺程度。

标注器输出 0–4 的五档概率 `level_probs`，程序计算连续期望：

\[
strength_i=\sum_{k=0}^{4}p(k)k
\]

每个节点另保留：

- `importance`：对用户当前核心问题的重要程度。
- `goal_impact∈[-1,1]`：状态增强对用户核心目标的影响。
- `confidence`：状态判断的把握。
- `observability`：`explicit/inferred/hypothesized`。

`strength` 和 `goal_impact` 分离：对负向信念非常确信，可以同时是高 strength 和负 goal impact。

## 7. 按轮图更新

\[
G_t=Apply(G_{t-1},Updates_t)
\]

模型只输出图操作，由确定性更新器合并进旧图。

### 7.1 基本操作

节点：

- `add`：产生有证据支持的新状态。
- `update`：更新已有节点的 `level_probs` 或内容。
- `deactivate`：当前不再活跃，但保留 ID 和历史。

边：`add/remove`。

不要求模型另外选择 strengthen/weaken/persist；程序根据新旧 strength 自动判定，避免操作类型与数值矛盾。

### 7.2 更新示例

```json
{
  "node_updates": [
    {
      "operation": "update",
      "node_id": "B1",
      "new_level_probs": [0.01, 0.04, 0.15, 0.55, 0.25],
      "trigger_span": "...",
      "confidence": 0.82
    },
    {
      "operation": "add",
      "node": {
        "id": "I2",
        "type": "intention",
        "content": "先查看其他工作机会",
        "level_probs": [0.05, 0.45, 0.35, 0.15, 0.0]
      }
    }
  ],
  "edge_updates": [
    {"operation": "add", "source": "B1", "target": "I2", "relation": "facilitates"}
  ],
  "appraisal_update": {},
  "emotion_update": {}
}
```

### 7.3 合并顺序

1. 复制 `G_{t-1}`，校验操作和 ID。
2. 添加新节点。
3. 更新旧节点的概率、连续强度或内容。
4. 将强度接近 0 的节点标记为 inactive。
5. 添加/删除边，拒绝悬空边和非法类型组合。
6. 更新当轮 Appraisal 和 Emotion。
7. 执行 schema 与语义校验，生成 `G_t`。

`delta` 由程序根据新旧 strength 计算，不让 LLM 同时填写前值、后值和 delta。

## 8. 反事实数据

同一 `group_id` 固定 profile、history prefix 和 previous graph，只改变 candidate reply：

```json
{
  "group_id": "esconv_001_turn_4",
  "task": "esconv",
  "profile": {"stable_facts": [], "relevant_preferences": [], "constraints": []},
  "history_prefix": [],
  "previous_graph": {"nodes": [], "edges": [], "emotion": {}},
  "candidates": [
    {"reply": "...", "updates": {}, "next_user_utterance": "..."}
  ]
}
```

每组包含：明显有效、轻度有效、安全但无效、与用户状态/约束不匹配的 hard negative、有害回复，以及当前 SFT/RL agent 的真实输出。

失败判断必须来自明确约束违反、数据集原生反馈、真实用户反应或多标注器对比判断。非 neutral 标注保存 `target_state/trigger_span/mechanism/confidence`，但 good/bad 标签不作为模型输入。

### 8.1 画像取舍

- 删除从单段完整对话推断的 Big Five。
- communication style 只使用分支点前的表达习惯，主要服务 utterance 生成。
- decision style 只保留前测或已见前缀支持的结构化参数。
- ESConv 删除 final emotion、事后问卷和后半段接受倾向。
- P4G 删除最终捐赠金额 B6 及由结局推断的 willingness。
- DuRecDial 只保留当前领域相关偏好/约束，删除完整未来 goal path。
- DailyDialog 第一版不使用 profile，不作为 CSRL 环境的重点任务。

## 9. 连续稠密奖励

节点变化与奖励：

\[
\Delta s_i=strength_{i,t}-strength_{i,t-1}
\]

\[
r_i=importance_i\cdot confidence_i\cdot goalImpact_i\cdot\Delta s_i
\]

正向状态增强和负向状态减弱都产生正奖励。边不重复计分，只用于验证变化的合理路径，并对无因果支持的跳变降权。

第一版总奖励：

\[
r_t=0.7r_{BDIE-transition}+0.2r_{process}+0.1r_{terminal}-r_{risk}-r_{inconsistency}
\]

- `BDIE-transition`：节点级认知转移和当轮情绪转移。
- `process`：回应当前需求、解决疑虑、信息增益和阶段匹配。
- `terminal`：有锚点的分级 GO，仅作小权重终局信号。
- `risk`：安全违反、欺骗、操控、事实错误和明确约束违反。
- `inconsistency`：潜在转移与用户话语/独立复核不一致。

neutral 是有效标签。同组奖励差小于重复判分噪声时，跳过该 GRPO 组或重新采样，不强行放大 advantage。

## 10. 其他工程修复

- GO 改用分级评分，降低终局权重，删除 `Be GENEROUS`。
- CTS 用于轨迹评估，不直接作逐轮奖励。
- CogWM 使用完整 JSON Schema guided decoding、`max_tokens=2048`、紧凑输入、语义校验和 invalid trajectory 过滤。
- 为 `run.py/run_cot.py` 中当前 agent 回复传递修复增加回归测试。

## 11. 评估闭环

```text
agent reply
    ↓
环境模型预测 BDI graph update + Emotion
    ↓
环境模型生成 user utterance
    ↓
独立 CogWM 从话语复核状态
    ↓
一致性检查
```

先用 Qwen3-8B 做 sanity check，达标后再决定是否扩大到 14B 或增加独立 reward head。

## 12. 质量门

1. 好回复相对坏回复 sign-rate > 80%。
2. 能区分明显改善、轻度改善、neutral 和恶化。
3. 同义改写差异明显小于好坏对照差。
4. 同一回复面对冲突偏好/约束时产生不同转移。
5. profile 和 previous graph 只使用分支点前信息。
6. 非 neutral 转移可定位到 state item、reply trigger 和 mechanism。
7. 图中无悬空边、非法 ID 或操作与数值矛盾。
8. 图转移、Emotion、用户话语与独立复核基本一致。
9. 重复判分噪声明显低于有效组内奖励间隔。
10. JSON/schema 失败率 = 0，结构缺失率 < 1%。
11. neutral 比例符合人工验证集，不追求空状态率趋零。

## 13. 实施路线

### Phase 0：定义与小样本

- 固定 BDI 节点、边、图操作、Appraisal 和 E schema。
- 每任务选 30–50 个分支点，每点构造 5–6 个反事实回复。
- 多标注器排序+少量人工审核，验证标注敏感性和图操作一致性。
- 落地 guided JSON、图更新器、语义校验和回归测试。

### Phase 1：数据生产

- 按 group 生成 ESConv/P4G/DuRecDial 反事实数据。
- 先保证分支点、失败类型和对话阶段覆盖，再扩大数量。
- 按对话/group 切分 train/validation/test，同组候选不得跨 split。

### Phase 2：环境模型

- 8B sanity check，评估图更新、节点级转移、连续奖励和话语一致性。
- 通过质量门后再训练 14B 或增加独立 reward head。

### Phase 3：RL 重启

- 启用新环境模型、分级 GO、PPO clip/KL 和 invalid-group 过滤。
- 监控奖励分量、组内方差、判分噪声、CTS 和独立 GO。
- 用现有保守 CogWM 做最终轨迹评估。

## 14. Phase 0 待定参数

1. BDI 节点最大活跃数和归档规则。
2. Appraisal 是显式中间节点还是状态字段。
3. 关系边由模型直接生成还是后处理构建。
4. `importance/goal_impact` 是初始时固定，还是允许低频更新。
5. 五档强度的标注一致性和概率校准方法。
6. 三个任务的 BDI/E 权重和 process reward 子项。
7. 根据重复判分噪声确定 GRPO 低信息组阈值。
8. 8B 是否足以达到转移敏感性和话语一致性要求。

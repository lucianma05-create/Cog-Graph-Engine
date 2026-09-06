# Cog-Graph-Engine

可解释因果用户模拟器：**BDI 图结构用户心智** + 可视化交互窗口，面向主动对话（情感支持 / 劝说捐赠 / 讨价还价）。设计文档：[shared_work_space/insight.md](shared_work_space/insight.md)。

```
Agent Reply → 认知状态转移（BDI 图增量更新 + Appraisal + Emotion）→ User Reply
```

---

## 1. 名词速查

| 名词 | 含义 |
|---|---|
| **BDI 图** | 用户心智的结构化表示：B=Belief（用户持有的命题）、D=Desire（想实现/避免的状态）、I=Intention（对具体行动的承诺倾向）。节点字段仅 `id/type/content/strength` |
| **第一人称立场** | 图上维护的是"用户自己心中的想法"：节点 content 一律用户口吻（B=命题本身，D="I want/avoid..."，I="I will..."），禁 "The user..." 包装；引擎扮演用户心智而非观察者 |
| **G0** | 初始认知图 = Init(P, S, H0)，temperature=0 + 最强模型（ANTHROPIC_INIT_MODEL）生成，持久化于 `sessions/g0/<seed>.json` 复用；删文件或 `--reinit` 重新生成 |
| **P / S / H0** | Init 的三个证据源：P=persona（对话前已知：人格问卷/自报问题/私有谈判立场）、S=场景、H0=干预前的对话前缀。**前缀之后的对话绝不进 LLM 上下文** |
| **三层证据政策** | G0 建节点的规则：Tier 1 固有态度（量表直接建：情境相关+强度按量表极端度+忠实转写）、Tier 2 情境激活（自报问题/私有立场）、Tier 3 前缀证据；**结果变量（捐赠/成交意向等）留给干预**，G0 不预植 |
| **strength** | LLM 直接输出 0–4 浮点（0-1 弱、2-3 明确、≥3.5 强）——早期五档概率分布机制已于 2026-09-06 删除（只有期望被消费，熵/采样从未实现） |
| **Appraisal** | 每轮对 agent 回复的事件评价：`goal_congruence / controllability / goal_conflict` 三字段（certainty 已删，由 belief strength 承载） |
| **Emotion** | 短期情绪：`category(闭合 17 标签) / valence / arousal / appraisal_target`（intensity 已删，由 arousal 承载；target 指向受影响节点或 `#agent_reply`，是因果归因） |
| **边关系（4 种）** | 见 §4 边规则 |
| **done / done_reason** | 终局标志；done_reason 是用户终局立场的第一人称表述（"I will take it at $70"） |
| **CAUSAL DISCIPLINE** | 因果纪律：u_t 立场切换必须有同轮节点变化；无切换时空更新合法（不逼造变化） |
| **ENGINE FEEDBACK** | 引擎把上一轮被拒操作及原因、⚠ 审计标记回注下一轮提示，保持 LLM 心理图与真实图一致 |
| **认知风格（3 维）** | 见 §5 |

## 2. 每轮发生什么（pipeline）

```
sim.step(agent_reply)
 ├─ 组装提示词（见 §3 上下文构成）——1 次 LLM 结构化调用（forced tool_choice，1 次校验重试）
 ├─ deterministic updater 落地：G_t = Apply(G_{t-1}, Updates_t)
 │    节点 add/update/deactivate、边 add/remove；非法方向/噪声/违约操作被拒
 ├─ 守卫与审计（全确定性，不改状态只记账）：
 │    噪声守卫、承诺持久守卫、价格审计、施压反制 ⚠、传播一致性 ⚠
 └─ 记录 turn（deltas、ops_applied/ops_rejected、notes、graph_after）并持久化
```

每轮模拟器转移只调 **1 次 LLM**；auto 模式额外 1 次 agent 回复调用。

## 3. 上下文构成（每个 LLM 调用里放什么）

### 3.1 Init 调用（构建 G0，每种子只跑一次）

| 位置 | 内容 |
|---|---|
| system | SYSTEM_CORE（本体论/边规则/更新纪律）+ INIT_SYSTEM_EXTRA（三步证据+三层政策+确定性要求）+ TASK_RULES[task] |
| user | **PERSONA(P)**（含 PRIVATE 私有谈判立场）、**COGNITIVE STYLE**（用户自述更新习惯）、**SCENARIO**、**H0 前缀** |

### 3.2 Turn 调用（每轮一次）

| 位置 | 内容 |
|---|---|
| system | SYSTEM_CORE + TURN_SYSTEM_EXTRA（CAUSAL DISCIPLINE/情绪闭合集/done 规则）+ TASK_RULES[task] |
| user | **PERSONA**（含 PRIVATE）、**COGNITIVE STYLE**、**CURRENT GRAPH**（活跃节点+边+最近 10 条失活）、**PREVIOUS APPRAISAL/EMOTION**、**DIALOGUE HISTORY**（前缀全量 + 最近 8 句）、**LATEST AGENT REPLY**、**ENGINE FEEDBACK**（上一轮被拒+⚠，首轮用 Init 的）、≥16 句时追加**确定性收尾提示** |

### 3.3 Agent 调用（auto / auto_editable 模式，扮演对方）

| 内容 | 可见性 |
|---|---|
| 公开 persona、场景、前缀、对话历史；auto_editable 可编辑系统提示词 | ✅ |
| 谈判任务的 agent 私有立场（agent_private，如卖家底价） | ✅ 仅 agent 上下文 |
| **认知图、private_persona（买家目标价）、cognitive_style** | ❌ **永不进 agent 上下文**（有防泄漏测试） |

## 4. 边规则（Bratman/Rao & Georgeff 实践推理）

认知前向流动 B→D→I，`facilitates/inhibits` 只允许三个方向，各对应一种机制：

| 方向 | 机制 |
|---|---|
| B→D | 合意性评价：信念改变对目标的渴望程度 |
| D→I | 慎思：欲望强度驱动行动承诺 |
| B→I | 手段评估：信念作用于行动本身的可行性/代价 |
| I→D `means_for` | 目的归属（手段-目的推理）：意向**服务**于哪个欲望；驱动欲望减弱后仍存续（承诺持久性） |
| D↔D `conflicts_with` | 欲望对立（欲望间唯一的合法连接；同层/逆向边一律被引擎拒绝） |
| I→B | **永非法**（不对称论题：意向不产生信念，反推即 wishful thinking） |

## 5. 认知风格（3 维，文献支撑，LLM 只见 NL）

**认知风格 = 用户认知"怎么变"的习惯**（不是"想什么"——那是图的内容）。3 个维度 × 2~3 档，组合即用户类型。**LLM 只看到第一人称 NL 块**（种子里的 `cognitive_style` 字段），**引擎只读枚举档位**（`cognitive_profile` 字段）——维度名和档位名从不进 LLM 上下文。

### 维度 ① 更新阻抗：同样的证据，认知动多快、被什么触发

| 档位 | 种子里的 NL 示例（真实文本） | 行为含义 |
|---|---|---|
| **malleable** | "I shift my view fairly easily — a friendly manner, a sense that others want the item, or an easy way to pay is enough to move me beyond my original number."（craigslist_02） | 奉承/稀缺/社会线索就能推动信念；小步移动频繁 |
| **normal** | "I move when the evidence is solid: concrete facts about where money goes persuade me, flattery does not."（p4g_01） | 需要实质证据才动 |
| **resistant** | "I hold my position unless the evidence is overwhelming — only new facts about the item move me; flattery, repetition or pressure does not."（craigslist_01） | 只有压倒性证据才动，且一步一格（esconv_02："when I am low, even hopeful claims barely reach me. I move only gradually, through small felt steps"） |

触发方式在 NL 里写明：谈判/捐赠任务多是**事实触发**，情感支持任务可以是**情感确认触发**（esconv_01："being genuinely listened to opens me up to see my situation differently"）。理论：ELM 双路径与 Need for Cognition（高 NC 者只认论据质量、态度持久抗反驳）、态度强度、Edwards 信念保守主义。

### 维度 ② 反向敏感性：被施压时，认知顺向还是反向

| 档位 | 种子里的 NL 示例（真实文本） | 行为含义 |
|---|---|---|
| **reactant** | "pushy or guilt-tripping appeals make me dig in rather than move me."（p4g_02） | 压力让立场**反向强化**（boomerang）；实测话语："Whoa, hold on... guilt-tripping me is not how you ask" |
| **non-reactant** | （无此句，即缺省） | 压力不引发反向；正常档用户对压力的回应是"要证据"而非"反弹"（p4g_01："That's pressure, not information"） |

引擎挂钩：reactant 用户 + 施压措辞 + goal_conflict≥0.6 + 意向上升 → ⚠ 审计标记。理论：Brehm 心理阻抗（trait 量表效度有争议，**此档只给有量表证据的种子**——目前仅 p4g_02，freedom=6.0/6）。

### 维度 ③ 承诺粘性：意向一旦形成，多难被放弃

| 档位 | 种子里的 NL 示例（真实文本） | 行为含义 |
|---|---|---|
| **persistent** | "Once I commit to a price, I stick with it until the deal clearly fails."（craigslist_01） | 意向存续到目的失败/欲望消亡；当前 6 种子全是此档 |
| **flexible** | "I readily revisit my plans when the situation shifts."（无种子使用，预留） | 情境变化即重新考虑（R&G 承诺策略之 open-minded） |

引擎挂钩：**承诺持久守卫**（persistent 档）——挂着活跃 means_for 的意向不得被 deactivate/清零。理论：Bratman 承诺持久性、Rao & Georgeff 承诺策略（blind/single/open-minded）。

### 种子里的实际写法

```json
{
  "cognitive_style": "I hold my position unless the evidence is overwhelming — only new facts about the item move me; flattery, repetition or pressure does not. Once I commit to a price, I stick with it until the deal clearly fails.",
  "cognitive_profile": { "reactance": "normal", "commitment": "persistent" }
}
```

`cognitive_style` 只进模拟器上下文（Init/Turn），**永不进 agent 上下文**（有防泄漏测试）；`cognitive_profile` 只被引擎读取（阻抗维度 ① 无机械实现，只由 NL 承担——建模审查结论：噪声过滤与人格阻抗是两个因果角色，不得共用阈值）。

### 6 种子当前配置

| 种子 | ① 更新阻抗 | ② 反向敏感性 | ③ 承诺粘性 |
|---|---|---|---|
| craigslist_01 | resistant（事实触发） | non-reactant | persistent |
| craigslist_02 | malleable | non-reactant | persistent |
| p4g_01 | normal（事实触发） | non-reactant | persistent |
| p4g_02 | resistant | **reactant** | persistent |
| esconv_01 | normal（情感确认触发） | non-reactant | persistent |
| esconv_02 | resistant（低落时门槛更高） | non-reactant | persistent |

## 6. 守卫与审计（引擎兜底，确定性）

| 机制 | 作用 |
|---|---|
| **噪声守卫** | 全局 \|Δ\|<0.4 的纯强度抖动拒绝；**携带真实内容修改的亚阈值移动放行**（内容重写=真实变化而非抖动） |
| **承诺持久守卫** | 挂着活跃 means_for 的意向不得 deactivate/清零，除非：欲望同轮死亡 / 同轮新意向+指向同欲望的 means_for 边 / 边同轮移除 / done / commitment=flexible |
| **价格审计** | 新出价高于历史最高且无 worth-belief/urgency 正向 Δ（或同轮新增）时 ⚠；带让步条件且涨幅 ≤5% 豁免（"throw in the adapter" 类） |
| **施压反制 ⚠** | reactant 用户：施压措辞 + goal_conflict≥0.6 + 意向上升 → ⚠（不拒绝，进回馈） |
| **传播一致性 ⚠** | 同轮相邻边（facilitates/inhibits）反向移动 ≥0.4 → ⚠（post-apply 边集，同轮切边不算） |
| **空 G0 守卫** | 非空 persona 下 Init 返回空图 → 带提醒重试一次 |

⚠ 标记与拒绝对（含原因）经 **ENGINE FEEDBACK** 回注下一轮提示。

## 7. 运行

```bash
python -m unittest discover tests           # 88 个单元测试（无需 LLM）
python run_demo.py --seed esconv_01         # 可视化服务（默认 8644；8642/8643 被占用）
python replay_cli.py --seed p4g_01 --turns 6   # 真实 agent 回复回放，对照真实/模拟用户
python tests/smoke_e2e.py --live --turns 3      # 冒烟（6 种子 × Init+3 轮）
python tools/eval_session.py --all          # 确定性日志指标（schema 拒绝/⚠ 审计/噪声计数）
python tools/eval_probes.py                 # 风格探针（reactance/facts-first/commitment，3/3 断言）
python tools/eval_tree.py --runs 3          # 共享前缀树 rollout（因果性/区分度聚合）
python tools/eval_dims.py                   # 3 维风格符合度对照审查
```

## 8. LLM 配置（.env，已被 gitignore）

```ini
ANTHROPIC_BASE_URL=...
ANTHROPIC_AUTH_TOKEN=...
ANTHROPIC_MODEL=...        # turn 更新 + agent 回复（高频路径）
ANTHROPIC_INIT_MODEL=...   # G0 种子图（一次性、缓存复用，用最强模型）
```

**`.env` 优先于 shell 环境变量**（曾经被 shell 里的旧导出压住，已修）；缺省 fallback `claude-sonnet-4-6`。注意：代理默认 thinking 模式拒绝强制 tool_choice，结构化调用已传 `thinking={"type": "disabled"}`。

## 9. 目录结构

```
engine/   schema（pydantic + LLM tool schema）、updater（确定性图引擎+守卫）、prompts、
          llm（anthropic 封装+模型分层）、agent（auto 模式）、simulator（会话编排+审计）
server/   stdlib HTTP 服务 + 可视化前端（vis-network 本地化，无 CDN）
seeds/    6 种子（真实对话；persona/私有立场/前缀/风格块/风格 profile）
tools/    种子提取 + 评估工具（eval_session/eval_probes/eval_tree/eval_dims）
sessions/ 会话日志（JSON 机器日志 + g0 缓存 + markdown 报告）
tests/    单元测试 + 冒烟测试
```

## 10. 已知边界（记录在案）

- 价格审计的文本正则读不懂让步语义的全貌（引用报价/条件句存在已知盲点）
- ⚠ 协议为字符串前缀约定（"⚠" 前缀 + 关键词匹配），改措辞会破坏路由
- ENGINE FEEDBACK 措辞把 ⚠ 警告与拒绝混称"rejected proposals"
- 历史会话日志含旧格式字段（读兼容，无害）；SCHEMA_VERSION=2

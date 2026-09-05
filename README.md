# Cog-Graph-Engine

轻量化的 **BDI 图结构用户模拟器** + 可视化交互窗口，面向主动对话（情感支持 / 劝说捐赠 / 讨价还价）。

设计文档：[shared_work_space/insight.md](shared_work_space/insight.md)

## 原理

```
Agent Reply → 认知状态转移（BDI 图增量更新 + Appraisal + Emotion）→ User Reply
```

- **图是用户自己的第一人称心智**：节点 content 一律以用户口吻书写（Belief=用户持有的命题、Desire="I want/avoid ..."、Intention="I will ..."），禁止 "The user ..." 第三人称包装；Appraisal/Emotion 同样以用户视角产出。模拟器定位仍是 insight.md §3 的"结构化潜在状态后验估计"——第一人称是表示语言，不是 ground truth 声明；
- 每轮 LLM 只输出**图更新操作**（节点 `add/update/deactivate`、边 `add/remove`）与 appraisal/emotion/用户话语，**确定性引擎** `G_t = Apply(G_{t-1}, Updates_t)` 负责落地——节点 ID 永不漂移、状态全程可审计；
- 节点强度 = LLM 输出的 0–4 五档概率分布的期望（`strength = Σ p(k)·k`），引擎程序计算，LLM 永不直接写强度；
- **边方向遵循 BDI 实践推理的三个机制**（同层与逆向边被引擎拒绝）：
  - `facilitates/inhibits`：B→D = 合意性评价，D→I = 慎思，B→I = 手段评估（信念作用于行动本身）；
  - `means_for` 仅 I→D = 意向的目的归属（手段-目的推理，意向在驱动欲望减弱后仍存续）；
  - `conflicts_with` 仅 D↔D；欲望之间的任何连接只能用 conflicts_with（对立）或不建边；
  - I→B 非法（Bratman 不对称论题：意向不产生信念，反推即 wishful thinking）；
- **G0 三层证据政策**（`G0 = Init(P, S, H0)`）：Tier 1 固有态度——persona 量表的价值观/人格直接建节点（情境相关 + 强度按量表极端度 + 忠实转写）；Tier 2 情境激活——自报问题/私有谈判立场；Tier 3 前缀证据——H0 自发表达。**结果变量（捐赠意向/成交意向等）留给干预**：G0 不预植，除非 H0 明说。前缀之后的对话绝不进入 LLM 上下文；
- **认知风格（cognitive_style）**：种子内置一段第一人称"条件化更新习惯"（如"I only change my mind on verifiable facts"），决定用户认知状态**怎么变**而非**想什么**；仅进模拟器上下文、绝不进 agent 上下文。`cognitive_profile` 是引擎专属的守卫开关（commitment / reactance），LLM 只看到 NL 描述；
- **确定性守卫与审计**（引擎侧兜底，违规尝试经 ENGINE FEEDBACK 回注下一轮提示）：
  - 噪声守卫：|Δ|<0.4 的纯抖动拒绝；**携带真实内容修改的亚阈值移动生效**（内容重写是真实认知变化的证据）；
  - 承诺持久守卫：仍挂着活跃 means_for 欲望的意向不能被 deactivate/清零（欲望同轮死亡/同轮替代/边同轮移除/done 可解封）；
  - 价格审计：新出价高于历史最高且无 worth-belief/urgency 正向 Δ 时打 ⚠（**带让步条件且涨幅 ≤5% 的提价豁免**）；
  - 施压反制 ⚠（reactance=pronounced 用户：施压 + 高 goal_conflict + 意向上升）与传播一致性 ⚠（同轮相邻边反向移动）；
  - **CAUSAL DISCIPLINE**：u_t 立场切换必须有同轮节点变化，无切换时空更新合法；
- 字段最小集（如无必要勿增实体）：Appraisal = goal_congruence / controllability / goal_conflict；Emotion = category / valence / arousal / appraisal_target（intensity、certainty 已删——分别由 arousal 与 belief strength 承载）；
- 种子图稳定：Init 使用 temperature=0 + 分步明确指令，且首次生成的 G0 持久化于 `sessions/g0/<seed_id>.json` 复用——同一种子每次初始化的基础图一致（删除该文件或 `--reinit` 可重新生成）；8 轮后追加确定性收尾提示防漂移；
- 三种驱动模式：**手动输入** / **LLM 自动扮演 agent** / **LLM 自定义提示词**（基础提示词可编辑）。

## 运行

```bash
# 1. 单元测试（无需 LLM）
python -m unittest discover tests

# 2. 启动可视化服务（默认 http://127.0.0.1:8644/）
python run_demo.py --seed esconv_01        # 可选 --seed / --port / --host / --reinit
```

可视化窗口：左侧 BDI 图（蓝=Belief、橙=Desire、绿=Intention，四种边关系区分样式），每轮变更高亮（新增绿 / 强度变化橙 + Δ / 失活灰虚线 / 边增删）；侧栏显示双方话语、appraisal/emotion、本轮更新操作、真实参考对话；底部支持逐轮回放与三种模式输入。

```bash
# 3. Headless 回放：用真实 agent 回复逐轮喂模拟器，对照真实/模拟用户回复
python replay_cli.py --seed p4g_01 --turns 6

# 4. 冒烟测试（真实 LLM 调用，6 个种子各 Init + 3 轮 manual + 1 轮 auto）
python tests/smoke_e2e.py --live --turns 3

# 5. 评估工具
python tools/eval_session.py --all   # 确定性日志分析：schema 拒绝/审计 ⚠/噪声计数
python tools/eval_probes.py          # 风格探针套件（reactance / facts-first / commitment，3/3 断言）
python tools/eval_tree.py --runs 3   # 共享前缀树状 rollout：因果性（状态分叉）与区分度（结局分化）聚合
```

## 目录结构

```
engine/     核心：schema（pydantic + LLM tool schema）、updater（确定性图引擎 + 守卫）、
            prompts、llm（anthropic 封装）、agent（auto 模式）、simulator（会话编排 + 审计）
server/     极简 stdlib HTTP 服务 + 可视化前端（vis-network 本地化，无 CDN）
seeds/      6 个种子（ESConv / P4G / CraigslistBargain 各 2，来自真实对话，
            含第一人称 cognitive_style + cognitive_profile）
tools/      种子提取脚本 + 评估工具（eval_session / eval_probes / eval_tree）
sessions/   会话日志（JSON 机器日志 + g0 缓存 + markdown 报告，API 不回传 llm_raw）
tests/      单元测试（84 个，无需 LLM）+ 冒烟测试
```

## 数据来源

- **ESConv / P4G**：`/data/user21300120/mmh/CogWM/bdi-annotation/output/` 的标注 jsonl（persona 只用对话前自述/前测字段，事后标注的 global_bdi 绝不进入种子）；
- **CraigslistBargain**：Stanford NLP 官方数据（Codalab bundle，下载缓存于 `seeds/raw/`）。

## LLM 配置

在项目根目录的 **`.env`** 中配置（该文件已被 `.gitignore` 排除，勿提交/分享）：

```ini
ANTHROPIC_BASE_URL=...
ANTHROPIC_AUTH_TOKEN=...
ANTHROPIC_MODEL=...          # 高频路径：turn 状态更新 + agent 回复
ANTHROPIC_INIT_MODEL=...     # G0 种子图（一次性、持久化缓存，建议用最强模型）
```

模型分层：种子图 G0 用 `ANTHROPIC_INIT_MODEL`（默认回退 `ANTHROPIC_MODEL`），每轮状态更新与 agent 回复用 `ANTHROPIC_MODEL`。**`.env` 的键优先于 shell 环境变量**；未写入 `.env` 的键回退到进程环境变量，两者都缺省时 fallback 为 `claude-sonnet-4-6`。空 G0 守卫：非空 persona 下 Init 返回空图会带提醒自动重试一次。

## 说明

- 每轮模拟器转移只调 1 次 LLM（forced tool_choice 结构化输出 + pydantic 校验 + 1 次重试）；auto 模式额外 1 次 agent 回复调用；
- 会话日志含每轮 `llm_raw`（审计用，API 不回传）；
- 评估方法论：结局对齐用 replay（模拟 vs 真实对话）、风格一致性用探针套件、可操纵性用树状 rollout 的剂量-反应曲线、结构健康度用 eval_session——中间认知状态只做诊断/路由（potential-based shaping），**credit 只从结局来**，避免模拟器自证陷阱；
- 奖励分配 / RL 训练等后续工作尚未开始。

# Cog-Graph-Engine

轻量化的 **BDI 图结构用户模拟器** + 可视化交互窗口，面向主动对话（情感支持 / 劝说捐赠 / 讨价还价）。

设计文档：[shared_work_space/insight.md](shared_work_space/insight.md)

## 原理

```
Agent Reply → 认知状态转移（BDI 图增量更新 + Appraisal + Emotion）→ User Reply
```

- 每轮 LLM 只输出**图更新操作**（节点 `add/update/deactivate`、边 `add/remove`）与 appraisal/emotion/用户话语，**确定性引擎** `G_t = Apply(G_{t-1}, Updates_t)` 负责落地——节点 ID 永不漂移、状态全程可审计；
- 节点强度 = LLM 输出的 0–4 五档概率分布的期望（`strength = Σ p(k)·k`），引擎程序计算，LLM 永不直接写强度；
- 边关系遵循认知前向传递 **B→D→I**：`facilitates/inhibits` 只允许 B→D、D→I、B→I（同层与逆向边被引擎拒绝）；`means_for` 仅 I→D；`conflicts_with` 仅 D↔D；
- 种子 = **对话元信息 + 未被干预影响的对话前缀 H0**，初始图 `G0 = Init(P, S, H0)` 最小证据支持（前缀之后的对话绝不进入 LLM 上下文，仅作界面上的真实对话参考）；
- 种子图稳定：Init 使用 temperature=0 + 分步明确指令，且首次生成的 G0 持久化于 `sessions/g0/<seed_id>.json` 复用——同一种子每次初始化的基础图一致（删除该文件或 `--reinit` 可重新生成）；
- 三种驱动模式：**手动输入** / **LLM 自动扮演 agent** / **LLM 自定义提示词**（基础提示词可编辑）。

## 运行

```bash
# 1. 单元测试（无需 LLM）
python -m unittest discover tests

# 2. 启动可视化服务（默认 http://127.0.0.1:8642/）
python run_demo.py --seed esconv_01        # 可选 --seed / --port / --host / --reinit
```

可视化窗口：左侧 BDI 图（蓝=Belief、橙=Desire、绿=Intention，四种边关系区分样式），每轮变更高亮（新增绿 / 强度变化橙 + Δ / 失活灰虚线 / 边增删）；侧栏显示双方话语、appraisal/emotion、本轮更新操作、真实参考对话；底部支持逐轮回放与三种模式输入。

```bash
# 3. Headless 回放：用真实 agent 回复逐轮喂模拟器，对照真实/模拟用户回复
python replay_cli.py --seed p4g_01 --turns 6

# 4. 冒烟测试（真实 LLM 调用，6 个种子各 Init + 3 轮 manual + 1 轮 auto）
python tests/smoke_e2e.py --live --turns 3
```

## 目录结构

```
engine/     核心：schema（pydantic + LLM tool schema）、updater（确定性图引擎）、
            prompts、llm（anthropic 封装）、agent（auto 模式）、simulator（会话编排）
server/     极简 stdlib HTTP 服务 + 可视化前端（vis-network 本地化，无 CDN）
seeds/      6 个种子（ESConv / P4G / CraigslistBargain 各 2，来自真实对话）
tools/      种子提取脚本（--dry-run 候选表 → --pick 生成）
sessions/   会话日志（每 seed 一个 JSON，服务重启可回放）
tests/      单元测试 + 冒烟测试
```

## 数据来源

- **ESConv / P4G**：`/data/user21300120/mmh/CogWM/bdi-annotation/output/` 的标注 jsonl（persona 只用对话前自述/前测字段，事后标注的 global_bdi 绝不进入种子）；
- **CraigslistBargain**：Stanford NLP 官方数据（Codalab bundle，下载缓存于 `seeds/raw/`）。

## LLM 配置

在项目根目录的 **`.env`** 中配置（该文件已被 `.gitignore` 排除，勿提交/分享）：

```ini
ANTHROPIC_BASE_URL=...
ANTHROPIC_AUTH_TOKEN=...
ANTHROPIC_MODEL=...
```

未写入 `.env` 的键会回退到进程环境变量；模型缺省 fallback 为 `claude-sonnet-4-6`。

## 说明

- 每轮模拟器转移只调 1 次 LLM（forced tool_choice 结构化输出 + pydantic 校验 + 1 次重试）；auto 模式额外 1 次 agent 回复调用；
- 会话日志含每轮 `llm_raw`（审计用，API 不回传）；奖励分配/RL 训练等后续工作尚未开始。

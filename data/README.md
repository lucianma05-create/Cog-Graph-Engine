# 实验数据留存

- `ab_results_*.json`：策略 A/B 效应量实验的 outcome 结果（keys 形如
  `seed_id|strategy`，值为每局 outcome 浮点列表）。波次 1：16 种子 ×
  good/bad/third/random × 3 局，max 6 轮，全部 flash。
- 原始输出日志（含每局进度行与 [FAIL] 行）随波次存为 `ab_wave*_raw.log`。
- outcome 定义与策略集见 `tools/eval_ab.py`（P4G 十策略 / 谈判策略 /
  ESConv 策略文献锚定）。

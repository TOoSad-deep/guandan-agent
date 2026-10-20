# 公开证据说明

本目录收录 B10 的原始摘要副本，便于读者核对首页与实验报告的数值。文件从原工作区复制，未为了展示修改内容。

| 文件 | 来源 | 用途 |
| --- | --- | --- |
| [frozen-plan.json](b10/frozen-plan.json) | `artifacts/b10/frozen-plan.json` | 版本、配置摘要、受测座位与总额度 |
| [comparison.json](b10/comparison.json) | `artifacts/b10/comparison.json` | 三版本结果、实际调用覆盖、时延与缺失值 |
| [verification.json](b10/verification.json) | `artifacts/b10/verification.json` | 历史核验结论、实际调用量与交付状态 |

这些摘要记录的是 2026-10-20 的本地实验。`verification.json` 中的测试数与交付状态是当时观测，不随当前 CI 自动更新。文件内的 `artifacts/`、`runs/` 路径仍指向原工作区。

**可复核范围：** 公开仓库支持阅读配置、核对摘要之间的一致性、检查核验脚本以及重跑离线测试。完整 B10 事件流、原模型回复、逐场审计与课程 ZIP 未公开，因此不能仅凭本目录独立重算全部 B10 结论。若取得对应的完整关闭归档，可使用 `scripts/verify_b10_run.py` 与 `scripts/summarize_b10.py` 重新核验。

仓库另保留 `experiments/phase3/results/probe_platform_run_02/` 中的早期平台回放，供协议解析和状态归约测试使用；它与 B10 三场实验不同。`tests/fixtures/web/` 是固定页面演示数据，不应计入真实对战结果。

摘要不支持胜率提升结论：每版本只有一场，发牌未配对，座位未轮换，V1/V2 额度耗尽后大量走规则备选。模型返回合法动作与解释文字正确也是两项独立指标。

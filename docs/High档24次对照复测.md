# High档24次对照复测

> 公开版本说明：本文中标为“原工作区”的材料未随仓库提供；可公开核对的内容见[证据说明](evidence/README.md)。

日期：2026-10-19。按用户要求，用DeepSeek high档重测[上一轮24次扩展复测](扩展24次复测.md)的同一组案例。

**本批最终文字表现改善：未发现明确错误21条、明确错误1条、含混2条，上一轮为14/6/4。调用与程序事实仍全部通过；耗时和Token用量上升，不能直接把这套评测预算用于在线比赛。**

## High如何设置与核验

模型仍为 `deepseek-flash`，按官方Chat Completions参数开启 `thinking={"type":"enabled"}`，并设置 `reasoning_effort="high"`。官方文档说明了这两个控制字段及思考输出字段。[Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/)、[Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)。

现有正式Provider默认关闭思考，RunConfig也不接受thinking=true。因此本轮在 `artifacts/retest24_high/` 使用隔离Provider，复用原传输、JSON模式、用量和错误处理，不修改正式配置。`review-config.json`只供生产生成器重放结果，thinking仍为false；实际API配置以plan的experiment_profile及HTTP观察值为准，完整报告generation也注明这一点。

调用前8项HTTP Mock检查证明绑定参数覆盖旧disabled设置；运行器另有21项检查。实际24个HTTP请求均观察到enabled/high/8192/json_object；24份响应均为HTTP200、finish_reason=stop，并返回非空思考字段与思考Token计数。这里只保留存在性和数字，不保存思考正文或headers。它证明所发参数和所获字段，不能独立检查服务端内部的思考强度实现。

## 对比条件

| 条件 | 上一轮 | 本轮High |
| --- | --- | --- |
| 模型、业务提示版本 | deepseek-flash；在线/离线v3 | 相同 |
| 任务与来源 | 12任务×2；9个请求、同一B04比赛 | 相同 |
| 原始messages、候选与真值 | 冻结记录 | 逐字段保持 |
| 思考 | disabled | enabled、high |
| 单次生成Token上限 | 在线384、离线1600 | 8192，供思考与最终答案使用 |
| 单次超时 | 20秒 | 90秒 |
| 重试、并发、调用上限 | 0、1、24 | 相同 |
| 最终答案结构与长度门槛 | action_id/reason；离线四字段 | 相同 |

review依赖本轮main的真实结果并重新计算风险，因此实际review输入不保证与旧轮完全相同；非review输入保持相同。判定标准逐字复用旧rubric，未因high放宽或收紧。

这是**整套high配置的比较**，不是只改变reasoning_effort的严格单变量实验。重复样本、同场比赛和review依赖仍然相关，不能据此计算总体准确率、显著性或胜率提升。

## 结果

| 指标 | 上一轮关闭思考 | 本轮High |
| --- | --- | --- |
| 真实调用/传输成功 | 24/24 | 24/24 |
| 生产契约通过 | 24/24 | 24/24 |
| 离线程序事实通过 | 12/12 | 12/12 |
| 模型文字未发现明确错误 | 14 | **21** |
| 模型文字明确错误 | 6 | **1** |
| 模型文字含混 | 4 | **2** |
| 12个任务中两次均未发现明确错误 | 5 | **9** |
| 至少一次有明确错误的任务 | 5 | **1** |

High的12条在线main/review理由均未发现明确事实错误；12条离线文字为9条未发现明确错误、1条错误、2条含混。仅检查实际写出的最终答案，程序事实不计入模型文字成绩；多数回复仍只覆盖部分预定目标。

逐条配对后：原14条未发现明确错误的回复本轮仍为该状态；原6条错误变为3条未发现明确错误、2条含混、1条仍有错误；原4条含混本轮均未发现明确错误。三路子代理分别独立审查动作、离线r1和r2，另对Q配牌r1交叉审计，双方均未发现明确错误，交叉项不重复计数。

本轮4次review均维持本批真实main。座位1残局在本轮两次均选择小王；上一轮两次分别选小王和大王。这是选择变化的观察，不证明小王策略更优，也不证明review带来纠错收益。

## 残余问题与覆盖局限

- **确定错误：** `offline_lead_baseline_r1` 仍写“拆掉6的一张”。原动作打出C6、D6两张6，留下H6一张；不能将剩余一张误作只打出一张。上一轮“两张10”的错误已不再出现，但该回复仍存在另一处数牌错误。
- **含混：** `offline_duplicate_bomb_r1` 写“11张散牌”，没有交代其中仍有三张4；没有将这一粗略描述直接判为全部单张算错。它已不再虚构后来本人重新领出。
- **含混：** `offline_wildcard_q_r2` 写“A炸弹消耗AA”“保留配牌与A组”，却未清晰说明炸弹扣4张A后只剩1张A。前文又正确写出炸弹出4张，因此保留歧义，不强判为确定数量错误。
- **引用问题另记：** 还贡r1仍将请求事件e1309写成执行确认依据。D2已执行这一事实确有其他直接证据，但具体inline引用不精确；不因为程序补齐事实引用就认为模型引用完全正确。
- H3配牌身份、本人配牌数量0、两张HT等检查点并非每次都明确讨论；未提及没有计作理解验证。最终答案平均字符数由592降至426，陈述更简短也可能减少暴露错误的机会，不能把结果变化全部归因为内部推理能力。

## 用量与速度

| 指标 | 上一轮 | 本轮High |
| --- | --- | --- |
| 输入tokens | 85,891 | 86,457 |
| 输出tokens | 8,755 | 37,848，其中思考31,604 |
| 总tokens | 94,646 | **124,305，约增加31.3%** |
| 单次Provider耗时均值 | 2.66秒 | 9.39秒 |
| 中位数 | 2.91秒 | 7.67秒 |
| P95（nearest rank，24条） | 4.67秒 | 14.32秒 |
| 最大耗时 | 5.47秒 | **39.96秒** |

费用没有供应商金额证据，保持未知，不能把Token增幅直接等同于费用增幅。输出Token包含思考；扣除已报告的31,604思考tokens后，最终输出部分为6,244 tokens。

这些是Provider调用耗时，不是整步出牌端到端耗时。`review_near_finish_seat1_r2`单次39.96秒，已超过正式配置默认20秒模型超时；同一请求的main与review实测耗时之和约43.27秒，还未计工具、提交等时间，超过当前30秒决策总预算。该调用之所以在本次完成，是使用了90秒实验超时，不能直接照搬完成率到比赛环境。

在这组归档样本上，high对文字质量有正向观察结果；它更适合先用于没有出牌时限压力的离线复盘。在线启用前仍需按真实时限验证预算和降级策略。当前生产默认没有切换，B10和新平台比赛均未启动。

## 记录与保持

实际24次调用、24次HTTP传输，零重试、无供应商错误，没有触发连续3次错误停止条件；额度用完后不追加调用。原生产代码与256份冻结文件未改。本轮生成报告的事实层与旧轮按同case比较；12份程序事实均经生产生成器和读层核验，模型原文独立保留。

- High冻结计划（原工作区 `artifacts/retest24_high/plan.json`）、前置参数核验（原工作区 `artifacts/retest24_high/preflight.json`）
- 真实调用汇总（原工作区 `artifacts/retest24_high/live-run/summary.json`）、实际HTTP参数及思考计数观察（原工作区 `artifacts/retest24_high/live-run/provider-observations.json`）
- 模型文字汇总（原工作区 `artifacts/retest24_high/adjudicated-audit.json`）、逐案对照（原工作区 `artifacts/retest24_high/comparison.json`）、耗时与用量（原工作区 `artifacts/retest24_high/usage-latency-comparison.json`）
- 生产契约与报告重放（原工作区 `artifacts/retest24_high/contract-audit.json`）、验证索引（原工作区 `artifacts/retest24_high/verification.json`）
- 动作审计（原工作区 `artifacts/retest24_high/action-audit.json`）、离线r1审计（原工作区 `artifacts/retest24_high/offline-audit-r1.json`）、离线r2与交叉审计（原工作区 `artifacts/retest24_high/offline-audit-r2.json`）

# V0 审计器固定样例

`valid_scripted_run/` 是 B03 四席、两场最小脚本运行的原样记录，用于检查审计器本身。它不是实际平台比赛，不用于 B04 的真实验收结论。

- 来源：`artifacts/b03/test-output/test_four_seats_two_scripted_m0/d45606da-be40-49ae-9172-d4d6a9854917/`。
- run_id：`d45606da-be40-49ae-9172-d4d6a9854917`。
- 内容：config.json、events.jsonl、summary.json；157 个事件，6 条脚本确认链。
- 测试先复制到临时目录，再注入非法动作、重复写出、缺失反馈等损坏。

样例随源码保存，测试无需依赖被忽略的 artifacts 目录、当前时间、服务或凭证。记录中的来源版本属于生成时刻；审计器单独报告当前源码是否一致，不会改写历史摘要。

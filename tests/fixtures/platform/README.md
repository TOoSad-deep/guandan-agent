# B02 平台原始帧

`historical_frames.json` 保存阶段三已完成实测的 18 条精选接收消息，覆盖创建、加入、公告、三种动作请求、公开通知、贡还、抗贡和两级结算。这里没有新增对局，也不是连续对局回放。

每条记录的 `raw` 与 `source.path` 指向 JSONL 的 `source.line` 中 `message` 完全一致；`source.sha256` 是该原始 JSONL 行 **包含行尾换行** 的 SHA-256。`seat` 来自记录的接收座位，不能用 act.curPos 代替。测试只加载当前案例，未将多座位手牌聚合为决策输入。

`back_request` 保留重复动作和全部原始候选。`lead_request`、`play_request`、`pass_request` 为避免复制大列表分别选取满足场景的较短真实请求。畸形消息与发送故障是单元测试显式生成的负例，不属于历史平台证据。

固定源码提交：`e92db7fa9d915fb318b3be78703d5ccd4a09c887`。

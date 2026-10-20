# 阶段三关键技术验证

> 公开版本说明：本文中标为“原工作区”的材料未随仓库提供；可公开核对的内容见[证据说明](../../docs/evidence/README.md)。

此目录保存小规模验证脚本与证据，不是最终项目结构。当前平台实验使用官方源码快照 `e92db7fa9d915fb318b3be78703d5ccd4a09c887`，不是教师另行发放版本的验证。

## 已完成与待完成

| 验证 | 当前状态 | 证据 |
| --- | --- | --- |
| Java 服务构建与四人完整对局 | 已运行：1 整场、15 小局、1328 次动作发送及通知匹配确认 | 平台总结果（原工作区 `experiments/phase3/results/probe_platform_run_02/summary.json`） |
| 非法动作、过期动作重放、断线、短时等待 | 已运行独立本地房间 | 同上；每项有单独 summary 与四座位日志 |
| Java 动作引擎与 Python bridge | 官方 smoke 通过；固定首局复现成功，跨局转换失败 | smoke（原工作区 `experiments/phase3/results/bridge-smoke.log`）、固定发牌实验（原工作区 `experiments/phase3/results/bridge/summary.json`） |
| LangGraph 与输出校验 | 模拟路径及 15 项离线测试通过，包含本地 MockTransport | 测试日志（原工作区 `experiments/phase3/results/model-tests.log`）、模拟结果（原工作区 `experiments/phase3/results/model-simulated/summary.json`） |
| DeepSeek 真实调用 | 三个合成局势、四次真实请求全部通过输出校验，包含一次复核 | 真实结果（原工作区 `experiments/phase3/results/model-live-run-01/summary.json`）、明细统计（原工作区 `experiments/phase3/results/model-live-run-01/analysis.json`） |
| 牌桌页面 | 当前服务根路径返回 WebSocket 握手错误；源码树无牌桌资源 | HTTP 结果（原工作区 `experiments/phase3/results/http-root.json`）、[静态审计](source_audit.json) |

模型模拟耗时不能作为 DeepSeek 延迟。平台一场成功不证明长期稳定或规则完全正确。完整解释和后续影响见 [架构与技术选型](../../docs/架构与技术选型.md)。

## 文件用途

- `prepare_platform.py`：复制官方源码至隔离工作目录、仅修改本机监听地址、构建服务端和动作引擎。
- `probe_platform.py`：四个独立 WebSocket 客户端，正常对局和受控异常实验。
- `probe_bridge.py`：Python 桥接的固定初始发牌与跨小局转换检查。
- `probe_model.py`：固定合成局势下的模拟或真实模型输出、LangGraph 分支与预算验证。
- `test_probe_model.py`：无密钥、无外网的异常路径与适配器请求测试。
- `fixtures/model_cases.json`：人工构造的三种局势，仅验证输出和流程，不证明牌型引擎正确。
- `requirements.lock.txt`：本次实际安装的 Python 版本依赖；环境信息见 environment.json（原工作区 `experiments/phase3/results/environment.json`）。

## 环境与复现

以下命令从项目根目录运行。需要 Python、uv、Maven 和 JDK。此快照的 Lombok 较旧，本次明确使用 JDK 21 构建，不使用本机 Maven 默认选择的 JDK 26。

```bash
uv venv .venv-phase3
uv pip install --python .venv-phase3/bin/python -r experiments/phase3/requirements.lock.txt
.venv-phase3/bin/python experiments/phase3/prepare_platform.py --java-home /Library/Java/JavaVirtualMachines/jdk-21.jdk/Contents/Home
```

其他机器将 `--java-home` 改为其 JDK 17 或 21 的实际目录。下载和构建需要联网，验证产物和缓存保存在实验目录。准备脚本不修改 `research/sources/` 原始快照。

启动本地服务器：

```bash
java -Dserver.port=18181 -jar experiments/phase3/workspace/guandan-websocket/target/guandan-java-pure-1.0.0.jar
```

服务器仅绑定 `127.0.0.1`，该适配见 source-adaptation.json（原工作区 `experiments/phase3/results/source-adaptation.json`）。测试结束后在启动它的终端按 Ctrl+C。不要同时启动第二个占用同端口的实例。

另开终端运行：

```bash
.venv-phase3/bin/python experiments/phase3/probe_platform.py --url ws://127.0.0.1:18181 --mode all --out experiments/phase3/results/platform-new-run
.venv-phase3/bin/python experiments/phase3/probe_bridge.py --output experiments/phase3/results/bridge-new-run
.venv-phase3/bin/python -m unittest discover -s experiments/phase3 -p test_probe_model.py -v
.venv-phase3/bin/python experiments/phase3/probe_model.py --mode simulated --output-dir experiments/phase3/results/model-simulated-new-run
```

平台异常探针只应对自己的本地验证服务运行。每项新建独立房间；非法动作和重放实验会故意提交候选之外的动作，用于观察该服务的实际行为。

## DeepSeek 配置与实测复现

使用项目根目录的 [.env.example](../../.env.example) 作为本地填写模板，密钥由使用者填写到 `.env`。该文件已被 Git 忽略。探针只读取进程环境，不会主动读取任何密钥文件。

在自己的终端中把配置导入当前进程后运行：

```bash
set -a
source .env
set +a
.venv-phase3/bin/python experiments/phase3/probe_model.py --mode live --output-dir experiments/phase3/results/model-live
```

这条 live 命令会调用 DeepSeek 并产生相应费用。当前三个合成局势每个最多两次请求，总计最多六次；不会向比赛平台出牌。默认选择 `deepseek-flash`、关闭 thinking、使用 JSON Output，再进行本地结构与候选校验。请求重试为零；模型失败不会自动切换供应商。

CLI 参数优先于环境变量。`DEEPSEEK_MODEL` 控制模型名，`GUANDAN_PROBE_TIMEOUT_SECONDS` 控制单次等待预算，当前默认 20 秒；总决策预算默认 30 秒，可用 `--total-budget` 调整。这些是探针参数，不是课程比赛时限。`--max-calls 1` 可检查不启用复核的路径。

本次已运行真实验证：四次请求耗时约 0.76–0.88 秒，带复核的决策约 1.62 秒，接口报告总用量 2081 Token。样本为三个固定合成局势，不代表实战胜率、稳定延迟或比赛时限达标。费用尚未核算，保持未知。

后续重跑会再次产生模型请求，应写入新的结果目录。没有密钥时 live 模式退出码为 2，并保存 blocked 记录；现有 `model-blocked` 是用户补充配置之前的历史尝试，不代表当前配置缺失。

## 证据使用边界

- 多座位原始日志仅用于离线核查。在线规则选择函数只接收对应座位的动作请求。
- 异常实验中“收到处理通知”证明服务处理了该动作，不证明它符合掼蛋规则。
- 首局固定发牌成功不等于整场可配对复现；本次 Python bridge 跨局已复现异常。
- 当前脚本没有完成真实 DeepSeek 与真实对局的端到端整合，也没有测量策略胜率。
- 模型探针的唯一候选分支只证明“输入候选只剩一项时可跳过模型”，不证明平台只有一个合法动作；实际接入时必须使用完整合法集合判断。
- `probe_platform_run_01` 是沙箱禁止本地连接的失败记录，未创建房间；实际平台实验使用 `run_02`。

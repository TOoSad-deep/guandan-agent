"use strict";

// Reading never starts a model call. Model review requires its explicit button.
const API = "/api/v1";
const $ = (id) => document.getElementById(id);
const ui = {
  runs: [], configs: [], run: null, session: null, view: null,
  runEpoch: 0, sessionEpoch: 0, detailEpoch: 0, reportListEpoch: 0, reportDetailEpoch: 0, evidenceEpoch: 0,
  pollBusy: false, decisionsBusy: false, cursor: 0, retainedFrom: 1,
  decisions: [], selectedRequest: null, mode: "table", report: null,
  historyStatus: "idle", historyError: null, historyGap: false, historyNew: false, historyMore: false,
  reportActionBusy: false, reportGenerationEnabled: null,
  modelReview: null, reviewSessionId: "", reviewSessionManual: false, reviewSessionsKey: null, reviewStarting: false,
  reviewTasks: new Map(), reviewErrors: new Map(), reviewPollBusy: false, capabilitiesEpoch: 0,
  observedActiveMatchId: null,
  followActiveMatch: true,
};
const labels = {
  starting: "正在启动", running: "运行中", stopping: "正在停止", completed: "已完成",
  interrupted: "已中断", waiting: "等待开局", play: "出牌", tribute: "进贡", back: "还贡",
  settled: "已结算", connecting: "正在连接", joined: "已入座", closed: "已关闭",
  main: "主模型", review: "条件复核", repair: "输出修复", rule: "规则决策",
  unique: "唯一动作", fallback: "规则兜底", reserved: "已预留", writing: "写入中",
  sent: "已发送 · 未确认", confirmed: "服务端已确认", cancelled_before_write: "写入前取消",
  unknown: "执行结果未知", rejected: "服务端拒绝", kept: "保留原选", changed: "改选动作",
  repaired: "修复成功", skipped: "跳过", failed: "失败", partial: "部分完成",
  trusted: "已知事实", degraded: "信息降级", none: "无", evaluation: "统计报告",
  preparing: "准备候选", deciding: "正在决策", submitting: "正在提交", invalidated: "请求已失效",
  ok: "正常", legal_pass: "合法不出", finishes_hand: "可出完手牌", opponent_close: "应对对手即将出完",
  rule_choice: "规则首选", top_score: "规则高分",
};
const actionLabels = {
  Single: "单张", Pair: "对子", Trips: "三张", ThreeWithTwo: "三带二",
  TwoTrips: "钢板", ThreePair: "三连对", Straight: "顺子", StraightFlush: "同花顺",
  Bomb: "炸弹", FourKings: "四王炸", PASS: "不出", tribute: "进贡", back: "还贡",
};
const featureLabels = {
  remaining_count: "出后余牌", remaining_cards: "出后余牌", estimated_turns: "估计轮数",
  split_cost: "拆牌代价", single_count: "单牌数", singleton_count: "单牌数", rule_score: "规则评分",
  card_count: "出牌张数", played_count: "出牌张数", is_pass: "不出", is_bomb: "炸弹", can_finish: "出完手牌", finishes_hand: "出完手牌", beats_leader: "压过领出",
  rank_strength: "点数强度", deep_used: "深度分析", breaks_bomb: "拆炸弹",
};

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}
function replace(id, ...children) { $(id).replaceChildren(...children); }
function name(value) { return value == null ? "—" : (labels[value] || String(value)); }
function short(value, length = 8) { return value ? String(value).slice(0, length) : "—"; }
function rankName(value) { return ({ T: "10", B: "小王", R: "大王" })[value] || value || "—"; }
function actionName(wire) {
  const kind = actionLabels[wire[0]] || wire[0];
  return ["PASS", "FourKings", "tribute", "back"].includes(wire[0]) ? kind : `${kind} · ${rankName(wire[1])}`;
}
function valueText(value) {
  if (value == null) return "—";
  if (typeof value === "boolean") return value ? "是" : "否";
  if (Array.isArray(value)) return value.map(valueText).join("、") || "无";
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
}
function badge(text, tone = "neutral") { return element("span", `badge ${tone}`, text); }
function errorMessage(error) { return error?.safe_message || error?.message || "读取失败，请稍后重试。"; }
function message(text, tone = "") {
  const node = $("app-message");
  node.textContent = text || "";
  node.className = `message ${tone}`;
  node.hidden = !text;
}
function empty(title, description, compact = false) {
  const box = element("div", `empty-state${compact ? " compact" : ""}`);
  box.append(element("span", compact ? "" : "empty-symbol", "≋"), element("h4", "", title));
  if (description) box.append(element("p", "", description));
  return box;
}
function displayError(container, error, fallback = "读取失败") {
  container.replaceChildren(empty(fallback, errorMessage(error), true));
}
function errorLine(error) {
  return element("p", "error-text", error ? `${error.code ? `${error.code} · ` : ""}${errorMessage(error)}` : "");
}
function timestamp(value) {
  if (!value) return "时间未知";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString("zh-CN", { hour12: false });
}
function toneFor(value) {
  if (["confirmed", "completed", "kept", "changed", "repaired"].includes(value)) return "good";
  if (["failed", "rejected"].includes(value)) return "danger";
  if (["partial", "unknown", "interrupted", "fallback", "skipped"].includes(value)) return "warning";
  return "neutral";
}
async function api(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    cache: "no-store", credentials: "same-origin", ...options,
    headers: { Accept: "application/json", ...(options.body ? { "Content-Type": "application/json" } : {}), ...options.headers },
  });
  let data;
  try { data = await response.json(); } catch { data = {}; }
  if (!response.ok) {
    const error = new Error(data.error?.safe_message || `请求失败（${response.status}）`);
    error.code = data.error?.code;
    error.status = response.status;
    error.safe_message = data.error?.safe_message;
    const headerCursor = response.headers?.get("X-Retained-From-Seq");
    error.retained_from_seq = data.retained_from_seq ?? data.error?.retained_from_seq ?? (headerCursor ? Number(headerCursor) : undefined);
    throw error;
  }
  return data;
}
function runPath() { return `/runs/${encodeURIComponent(ui.run.run_id)}`; }
function sessionPath() { return `${runPath()}/sessions/${encodeURIComponent(ui.session.session_id)}`; }
function active(run) { return run && ["starting", "running", "stopping"].includes(run.status); }
function reviewPending(task) { return task.status === "running" || (["completed", "partial"].includes(task.status) && !task.report_ready); }
function updateControls() {
  const archival = ui.run?.read_only === true;
  $("start-run").disabled = !ui.configs.length || archival || active(ui.run) || ui.runs.some((run) => run.run_id !== ui.run?.run_id && active(run));
  $("stop-run").disabled = !ui.run || archival || !active(ui.run) || ui.run.status === "stopping";
  $("config-select").disabled = archival;
  $("start-run").title = archival ? "当前为只读归档；选择“新运行”后可启动配置" : "使用本地允许的配置创建运行";
  $("refresh-reports").disabled = !ui.run || ui.reportActionBusy;
  $("generate-report").disabled = !ui.run || active(ui.run) || ui.reportActionBusy || !ui.reportGenerationEnabled;
  const generationHint = ui.reportGenerationEnabled === false ? "固定演示归档只读，可浏览已有报告。"
    : ui.reportGenerationEnabled !== true ? "报告生成能力尚未加载。"
    : active(ui.run) ? "运行结束后，可生成统计报告。模型调用 0 次。" : "生成统计报告 · 模型调用 0 次";
  $("generate-report").title = generationHint;
  $("report-generation-hint").textContent = generationHint;
  renderReviewControls();
}
function renderReviewControls() {
  const sessions = ui.run?.sessions || [];
  const select = $("review-session-select");
  const existing = sessions.find((session) => session.session_id === ui.reviewSessionId);
  const current = sessions.find((session) => session.session_id === ui.session?.session_id);
  const chosen = ui.reviewSessionManual ? existing : current;
  ui.reviewSessionId = chosen?.session_id || "";
  const sessionsKey = JSON.stringify([ui.run?.run_id, sessions.map((session) => [session.session_id, session.seat, session.match_id])]);
  if (ui.reviewSessionsKey !== sessionsKey) {
    ui.reviewSessionsKey = sessionsKey;
    select.replaceChildren(element("option", "", "请选择复盘座位"));
    select.firstElementChild.value = "";
    for (const session of sessions) {
      const option = element("option", "", `座位 ${session.seat} · ${short(session.match_id, 6)} · ${short(session.session_id, 6)}`);
      option.value = session.session_id;
      select.append(option);
    }
  }
  if (select.value !== ui.reviewSessionId) select.value = ui.reviewSessionId;
  select.disabled = !sessions.length || ui.reviewStarting;
  const capability = ui.modelReview;
  const running = [...ui.reviewTasks.values()].find((record) => reviewPending(record.task));
  const reason = !ui.run ? "先选择已结束的运行。"
    : active(ui.run) ? "运行结束后可生成模型复盘。"
    : !capability ? "模型复盘配置尚未加载。"
    : !capability.enabled ? capability.disabled_reason || "模型复盘尚未启用。"
    : !(capability.remaining_calls > 0) ? "本批模型复盘额度已用完。"
    : ui.reviewStarting ? "正在创建复盘任务…"
    : running ? "已有模型复盘正在生成。"
    : !ui.reviewSessionId ? "请选择要复盘的座位。" : "";
  $("generate-model-review").disabled = Boolean(reason) || ui.reportActionBusy;
  $("generate-model-review").title = reason || "按所选座位生成一份模型复盘";
  const description = capability
    ? `每份最多 ${valueText(capability.max_decisions)} 条关键决策 · ${valueText(capability.calls_per_report)} 次 ${capability.model || "DeepSeek"} 调用 · 剩余 ${valueText(capability.remaining_calls)} / ${valueText(capability.max_calls)} 次`
    : "模型复盘配置尚未加载。";
  $("model-review-hint").textContent = `${description}${reason ? `\n${reason}` : ""}`;
  const record = ui.reviewTasks.get(ui.run?.run_id), status = $("model-review-status");
  const error = ui.reviewErrors.get(ui.run?.run_id);
  status.hidden = !record && !error;
  status.className = "inline-message";
  if (record) {
    const task = record.task;
    const taskSeat = sessions.find((session) => session.session_id === record.sessionId)?.seat;
    const scope = taskSeat == null ? "后台复盘任务" : `座位 ${taskSeat} 的模型复盘`;
    const state = task.report_ready ? (task.status === "partial" ? "已部分生成" : task.status === "failed" ? "已保存失败报告" : "已生成")
      : task.status === "running" ? "正在生成…" : task.status === "failed" ? "生成失败" : "正在保存报告…";
    status.textContent = `${scope} · ${state} · 实际模型调用 ${valueText(task.calls_started)} 次${task.error ? `\n${errorMessage(task.error)}` : ""}${record.pollError ? `\n${record.pollError}` : ""}${error ? `\n${error}` : ""}`;
  } else status.textContent = error || "";
  const ready = record?.task.report_ready === true && Boolean(record.task.report_id);
  $("open-model-review").hidden = !ready;
}
function renderRun() {
  $("run-mode").textContent = ui.run ? (ui.run.read_only ? "只读归档" : "当前进程") : "等待选择";
  $("run-status").textContent = ui.run ? name(ui.run.status) : "无运行";
  $("run-status").className = `badge ${toneFor(ui.run?.status)}`;
  $("log-status").textContent = ui.run ? (ui.run.log_complete ? "记录完整" : "记录不完整") : "记录状态 —";
  const runError = $("run-error");
  runError.hidden = !ui.run?.error;
  runError.textContent = ui.run?.error ? errorMessage(ui.run.error) : "";
  updateControls();
}
function renderRunOptions() {
  const select = $("run-select");
  select.replaceChildren(element("option", "", "新运行 / 未选择"));
  select.firstElementChild.value = "";
  for (const run of ui.runs) {
    const option = element("option", "", `${run.read_only ? "归档" : "当前"} · ${run.label || "未命名运行"} · ${short(run.run_id)} · ${name(run.status)}`);
    option.title = run.run_id;
    option.value = run.run_id;
    select.append(option);
  }
  select.value = ui.run?.run_id || "";
}
async function loadRuns({ initial = false } = {}) {
  const epoch = ui.runEpoch;
  try {
    const result = await api("/runs");
    if (epoch !== ui.runEpoch) return;
    ui.runs = result.items || [];
    renderRunOptions();
    if (initial) {
      let saved;
      try { saved = localStorage.getItem("guandan.run_id"); } catch { /* Storage is optional. */ }
      const selected = ui.runs.find((run) => run.run_id === saved) || ui.runs.find(active) || ui.runs[0];
      if (selected) await selectRun(selected.run_id);
      else message("暂无运行记录。选择本地配置后，可新建一次运行。");
    } else if (ui.run) {
      const item = ui.runs.find((run) => run.run_id === ui.run.run_id);
      if (item) { ui.run = item; renderRun(); renderSessions(); }
    }
    updateControls();
  } catch (error) { if (epoch === ui.runEpoch) message(errorMessage(error), "error"); }
}
async function loadConfigs() {
  const epoch = ++ui.capabilitiesEpoch;
  try {
    const result = await api("/configs");
    ui.configs = result.items || [];
    if (epoch === ui.capabilitiesEpoch) applyReportCapabilities(result);
    const select = $("config-select");
    select.replaceChildren();
    for (const config of ui.configs) {
      const option = element("option", "", `${config.config_name} · ${config.policy} · 调用上限 ${valueText(config.model_calls)}`);
      option.value = config.config_name;
      select.append(option);
    }
    if (!ui.configs.length) { const option = element("option", "", "暂无允许的配置"); option.value = ""; select.append(option); }
  } catch (error) { replace("config-select", element("option", "", "配置读取失败")); message(errorMessage(error), "error"); }
  updateControls();
}
function applyReportCapabilities(result) {
  ui.reportGenerationEnabled = typeof result.report_generation_enabled === "boolean" ? result.report_generation_enabled : null;
  ui.modelReview = result.model_review || null;
  if (ui.modelReview?.active_task) rememberReviewTask(ui.modelReview.active_task);
}
async function loadReviewCapabilities() {
  const epoch = ++ui.capabilitiesEpoch;
  try {
    const result = await api("/configs");
    if (epoch !== ui.capabilitiesEpoch) return;
    applyReportCapabilities(result); updateControls();
  } catch { /* Task state is still read independently; generation is never retried. */ }
}
function persistReviewTasks() {
  try {
    localStorage.setItem("guandan.review_tasks", JSON.stringify([...ui.reviewTasks.values()].map((record) => ({
      task_id: record.task.task_id, source_run_id: record.task.source_run_id, session_id: record.sessionId,
    }))));
  } catch { /* Server capabilities can recover an active task without storage. */ }
}
function restoreReviewTasks() {
  try {
    const records = JSON.parse(localStorage.getItem("guandan.review_tasks") || "[]");
    if (!Array.isArray(records)) return;
    for (const record of records) {
      if (typeof record.task_id !== "string" || typeof record.source_run_id !== "string") continue;
      rememberReviewTask({ task_id: record.task_id, source_run_id: record.source_run_id, status: "running", report_ready: false }, record.session_id);
    }
  } catch { /* Browser storage is optional and may contain obsolete data. */ }
}
function rememberReviewTask(task, sessionId) {
  if (!task?.task_id || !task.source_run_id) return null;
  let record = ui.reviewTasks.get(task.source_run_id);
  if (record?.task.task_id === task.task_id) {
    if (record.task.status === "running" || task.status !== "running") record.task = task;
    if (sessionId) record.sessionId = sessionId;
  } else {
    record = { task, sessionId: sessionId || "", presentedEpoch: -1, pollError: "",
      watchRunEpoch: ui.run?.run_id === task.source_run_id ? ui.runEpoch : null,
      detailEpoch: ui.reportDetailEpoch };
    ui.reviewTasks.set(task.source_run_id, record);
    persistReviewTasks();
  }
  return record;
}
async function selectRun(id) {
  ui.runEpoch += 1;
  ui.sessionEpoch += 1;
  ui.detailEpoch += 1;
  ui.reportListEpoch += 1; ui.reportDetailEpoch += 1;
  ui.evidenceEpoch += 1;
  ui.run = ui.runs.find((run) => run.run_id === id) || null;
  ui.session = null; ui.view = null; ui.report = null;
  ui.reviewSessionId = ""; ui.reviewSessionManual = false;
  ui.observedActiveMatchId = null;
  ui.followActiveMatch = true;
  ui.decisions = []; ui.cursor = 0; ui.selectedRequest = null;
  ui.decisionsBusy = false;
  ui.historyStatus = "idle"; ui.historyError = null; ui.historyGap = false; ui.historyNew = false; ui.historyMore = false;
  ui.reportActionBusy = false;
  $("report-action-status").hidden = true;
  message("");
  replace("decision-detail", empty("选择一条决策", "查看候选动作、模型理由和实际执行。"));
  replace("report-body", empty("选择一份已有报告", "浏览报告不会触发新的模型调用。"));
  closeEvidence();
  renderRunOptions(); renderRun(); renderSessions(); renderTable(); renderDecisions();
  const reviewTask = ui.reviewTasks.get(id);
  if (reviewTask) {
    reviewTask.watchRunEpoch = ui.runEpoch; reviewTask.detailEpoch = ui.reportDetailEpoch;
    if (ui.run?.sessions.some((session) => session.session_id === reviewTask.sessionId)) {
      ui.reviewSessionId = reviewTask.sessionId; ui.reviewSessionManual = true;
    }
    renderReviewControls();
  }
  try { if (id) localStorage.setItem("guandan.run_id", id); else localStorage.removeItem("guandan.run_id"); } catch { /* Optional preference only. */ }
  if (!ui.run) { replace("report-list", empty("暂无运行", "先选择一个已有运行。", true)); return; }
  const epoch = ui.runEpoch;
  try {
    const result = await api(runPath());
    if (epoch !== ui.runEpoch) return;
    ui.run = { ...ui.run, ...result };
    renderRun(); renderSessions();
  } catch (error) { if (epoch === ui.runEpoch) message(errorMessage(error), "error"); }
  if (ui.mode === "reports" && epoch === ui.runEpoch) await loadReports();
  if (ui.modelReview && epoch === ui.runEpoch) loadReviewCapabilities();
}
function renderSessions() {
  const select = $("session-select");
  const sessions = ui.run?.sessions || [];
  const existing = ui.session?.session_id;
  const preferredSeat = ui.session?.seat;
  ui.observedActiveMatchId = ui.run?.active_match_id ?? null;
  select.replaceChildren();
  for (const session of sessions) {
    const repeatedSeat = sessions.filter((item) => item.seat === session.seat).length > 1;
    const option = element("option", "", `座位 ${session.seat}${repeatedSeat ? ` · ${short(session.match_id, 5)}` : ""}`);
    option.value = session.session_id;
    select.append(option);
  }
  select.disabled = !sessions.length;
  if (!sessions.length) { select.append(element("option", "", "暂无座位")); ui.session = null; return; }
  const needsCurrentMatch = ui.followActiveMatch && ui.run.active_match_id && ui.session?.match_id !== ui.run.active_match_id;
  const nextMatchSameSeat = needsCurrentMatch && preferredSeat != null
    ? sessions.find((session) => session.match_id === ui.run.active_match_id && session.seat === preferredSeat) : null;
  const selected = nextMatchSameSeat || sessions.find((session) => session.session_id === existing) || sessions.find((session) => session.match_id === ui.run.active_match_id) || sessions[0];
  select.value = selected.session_id;
  if (selected.session_id !== existing) selectSession(selected.session_id);
}
function selectSession(id, { manual = false } = {}) {
  ui.sessionEpoch += 1; ui.detailEpoch += 1;
  ui.session = ui.run?.sessions.find((session) => session.session_id === id) || null;
  if (manual) ui.followActiveMatch = !ui.run?.active_match_id || ui.session?.match_id === ui.run.active_match_id;
  ui.view = null; ui.cursor = 0; ui.decisions = []; ui.selectedRequest = null; ui.decisionsBusy = false;
  ui.historyStatus = "idle"; ui.historyError = null; ui.historyGap = false; ui.historyNew = false; ui.historyMore = false;
  ui.retainedFrom = 1;
  $("session-select").value = id;
  $("history-message").hidden = true; $("recover-history").hidden = true;
  replace("decision-detail", empty("选择一条决策", "当前详情始终标明所属座位和请求。"));
  renderTable(); renderDecisions();
  renderReviewControls();
  $("refresh-decisions").disabled = !ui.session;
  if (ui.session) { pollState(); loadDecisions(true); }
}
function player(seat, relative, counts, currentSeat) {
  const container = element("div", `player-info`);
  const isSelf = seat === currentSeat;
  const isPartner = seat === (currentSeat + 2) % 4;
  const title = element("p", "player-name", `座位 ${seat}`);
  title.append(element("span", "player-tag", isSelf ? "本人" : isPartner ? "队友" : "对手"));
  const count = element("div", "player-count");
  count.append(element("strong", "", valueText(counts?.[seat])), document.createTextNode("张"));
  container.append(title, count);
  const node = $(`seat-${relative}`);
  node.className = `player player-${relative}${isSelf ? " self" : isPartner ? " partner" : ""}`;
  node.replaceChildren(element("span", "seat-avatar", seat), container);
}
function cardDetails(card) {
  const suits = { S: "♠", H: "♥", C: "♣", D: "♦" };
  if (card === "SB") return { rank: "JOKER", suit: "♛", red: false, joker: true, label: "小王" };
  if (card === "HR") return { rank: "JOKER", suit: "♛", red: true, joker: true, label: "大王" };
  const raw = String(card);
  const rank = raw.slice(1) === "T" ? "10" : raw.slice(1);
  return { rank, suit: suits[raw[0]] || "", red: ["H", "D"].includes(raw[0]), joker: false, label: `${suits[raw[0]] || ""}${rank}` };
}
function cardNode(card, mini = false) {
  const info = cardDetails(card);
  const node = element("span", `${mini ? "mini-card" : "card"}${info.red ? " red" : ""}${info.joker ? " joker" : ""}`);
  node.setAttribute("aria-label", info.label);
  node.title = info.label;
  if (mini) node.textContent = `${info.suit}${info.rank}`;
  else node.append(element("span", "card-rank", info.rank), element("span", "card-suit", info.suit));
  return node;
}
function actionCards(wire, mini = false) {
  const cards = element("div", mini ? "mini-cards" : "leading-cards");
  if (!wire) { cards.append(element("span", "muted", "动作未知")); return cards; }
  if (wire[0] === "PASS") { cards.append(element("span", "badge neutral", "不出")); return cards; }
  const values = Array.isArray(wire[2]) ? wire[2] : [];
  for (const card of values) cards.append(cardNode(card, mini));
  return cards;
}
function renderTable() {
  const state = ui.view?.state;
  const seat = ui.session?.seat ?? 0;
  $("round-value").textContent = ui.view?.round_id ? `第 ${ui.view.round_id} 局` : "—";
  $("level-value").textContent = rankName(state?.level_rank);
  $("stage-value").textContent = state ? name(state.stage) : "等待牌桌";
  $("decision-stage").textContent = name(ui.view?.decision_stage);
  $("table-subtitle").textContent = ui.session ? `座位 ${seat} 的独立视角 · 队友为座位 ${(seat + 2) % 4}` : "每个视角，只展示该座位可见的信息";
  const counts = state?.remaining_counts;
  player((seat + 2) % 4, "north", counts, seat);
  player((seat + 1) % 4, "east", counts, seat);
  player((seat + 3) % 4, "west", counts, seat);
  player(seat, "south", counts, seat);
  $("leader-label").textContent = state?.leader_seat != null ? `领出 · 座位 ${state.leader_seat}` : (state ? "当前无领出动作" : "等待牌桌信息");
  replace("leading-cards", ...(state?.leading_action ? Array.from(actionCards(state.leading_action).childNodes) : []));
  $("leading-type").textContent = state?.leading_action ? actionName(state.leading_action) : (state ? "公开领出动作将在这里显示" : "平台公开信息将在这里显示");
  $("hand-title").textContent = ui.session ? `座位 ${seat} · 本人手牌` : "当前座位手牌";
  $("hand-count").textContent = state ? `${state.self_hand.length} 张` : "— 张";
  if (!state) replace("self-hand", element("p", "empty-inline", ui.session ? "尚未收到本座位牌桌信息" : "选择运行和观察座位后查看手牌"));
  else if (!state.self_hand.length) replace("self-hand", element("p", "empty-inline", "当前手牌为空"));
  else replace("self-hand", ...state.self_hand.map((card) => cardNode(card)));
  $("sync-label").textContent = ui.view ? `${ui.run?.read_only ? "归档快照" : "当前状态"} · v${ui.view.view_version} · ${timestamp(ui.view.updated_at)}` : "等待同步";
  $("facts-status").textContent = ui.view?.log_complete === false ? "记录存在缺口，详情可能不完整" : state?.facts_status === "degraded" ? "牌桌信息降级 · 未知数值显示 —" : "其他座位仅显示公开余牌";
}
async function pollState() {
  if (!ui.run || ui.pollBusy) return;
  const runEpoch = ui.runEpoch;
  let sessionEpoch = ui.sessionEpoch;
  ui.pollBusy = true;
  try {
    const wasActive = active(ui.run);
    const run = await api(runPath());
    if (runEpoch !== ui.runEpoch) return;
    ui.run = { ...ui.run, ...run };
    ui.runs = ui.runs.map((item) => item.run_id === ui.run.run_id ? { ...item, ...ui.run } : item);
    renderRun(); renderSessions();
    if (wasActive && !active(ui.run) && ui.session) loadDecisions(true);
    if (!ui.session || ui.mode !== "table") return;
    sessionEpoch = ui.sessionEpoch;
    const result = await api(`${sessionPath()}/state`);
    if (runEpoch !== ui.runEpoch || sessionEpoch !== ui.sessionEpoch) return;
    if (result.session_id !== ui.session.session_id || result.run_id !== ui.run.run_id) return;
    const changed = !ui.view || result.view_version > ui.view.view_version;
    if (changed) {
      ui.historyNew = Boolean(ui.view && ui.decisions.length);
      ui.view = result;
      if (!ui.decisions.length && ui.historyStatus === "ready") loadDecisions(true);
      renderDecisions();
    }
    // A successful unchanged snapshot also clears a previous transport error.
    renderTable();
  } catch (error) {
    if (runEpoch === ui.runEpoch && sessionEpoch === ui.sessionEpoch) $("sync-label").textContent = `同步失败 · ${errorMessage(error)}`;
  } finally { ui.pollBusy = false; }
}
function renderDecisions() {
  const list = $("decision-list");
  list.replaceChildren();
  if (!ui.decisions.length) {
    let title = "暂无决策记录", description = active(ui.run)
      ? "本座位暂未记录决策，收到新的牌桌状态后会自动检查。" : "本座位没有可查询的决策记录。可切换座位或刷新记录。";
    if (!ui.session) { title = "尚未选择座位"; description = "选择运行和观察座位后查看决策记录。"; }
    else if (ui.historyStatus === "loading") { title = "正在读取决策记录…"; description = "正在查询当前座位的历史。"; }
    else if (ui.historyStatus === "error") { title = "决策记录读取失败"; description = "点击“刷新记录”重试。"; }
    else if (ui.historyStatus === "unavailable") { title = "较早记录暂不可查询"; description = "当前查询窗口存在缺口，请从可用记录继续或刷新重试。"; }
    else if (ui.historyGap) { title = "可用尾部中没有决策"; description = "较早记录不可查询，不能据此认为本座位从未决策。"; }
    list.append(empty(title, description, true));
  }
  for (const item of ui.decisions) {
    const button = element("button", `decision-row${item.request_id === ui.selectedRequest ? " selected" : ""}`);
    button.setAttribute("aria-label", `查看第 ${item.round_id} 局 ${item.request_id.split(":").at(-1)} 决策，${name(item.source)}，${name(item.execution_status)}`);
    const label = element("span", "decision-label");
    label.append(element("strong", "", `第 ${item.round_id} 局 · ${name(item.stage)}`), element("small", "", `${item.request_id.split(":").at(-1)} · ${name(item.source)}${item.log_complete === false ? " · 记录不完整" : ""}`));
    button.append(element("span", "decision-number", item.opened_seq), label, badge(name(item.execution_status), toneFor(item.execution_status)), element("span", "row-arrow", "›"));
    button.addEventListener("click", () => openDecision(item.request_id));
    list.append(button);
  }
  $("history-count").textContent = `${ui.decisions.length} 条记录`;
  $("refresh-decisions").disabled = !ui.session || ui.decisionsBusy;
  $("more-decisions").disabled = ui.decisionsBusy || !ui.historyMore;
  const notes = [];
  if (ui.historyGap) notes.push("记录覆盖存在缺口，当前可用记录不代表完整历史。");
  if (ui.historyError) notes.push(errorMessage(ui.historyError));
  if (ui.historyNew) notes.push("牌桌状态已有更新，可点击“刷新记录”查看最新决策。");
  $("history-message").textContent = notes.join(" ");
  $("history-message").hidden = !notes.length;
  $("recover-history").hidden = !(ui.historyStatus === "unavailable" && ui.retainedFrom > ui.cursor + 1);
  $("recover-history").disabled = ui.decisionsBusy;
}
async function loadDecisions(reset = false, forcedCursor) {
  if (!ui.session || ui.decisionsBusy) return;
  const runEpoch = ui.runEpoch, sessionEpoch = ui.sessionEpoch;
  ui.decisionsBusy = true;
  if (reset) {
    ui.decisions = []; ui.cursor = 0;
    ui.historyGap = forcedCursor != null && forcedCursor > 0;
    ui.retainedFrom = 1;
  }
  if (forcedCursor != null) ui.cursor = forcedCursor;
  ui.historyStatus = "loading"; ui.historyError = null; ui.historyNew = false; ui.historyMore = false;
  renderDecisions();
  try {
    let result;
    // A moving live tail may require another cursor, but never retry indefinitely.
    for (let attempt = 0; attempt < 3; attempt += 1) {
      try {
        result = await api(`${sessionPath()}/decisions?after_seq=${ui.cursor}&limit=30`);
        break;
      } catch (error) {
        if (runEpoch !== ui.runEpoch || sessionEpoch !== ui.sessionEpoch) return;
        const retained = error.retained_from_seq;
        if (error.status === 410) {
          ui.historyGap = true;
          if (Number.isInteger(retained) && retained >= 1) ui.retainedFrom = retained;
        }
        if (error.status !== 410 || !Number.isInteger(retained) || retained - 1 <= ui.cursor || attempt === 2) throw error;
        ui.cursor = retained - 1;
      }
    }
    if (runEpoch !== ui.runEpoch || sessionEpoch !== ui.sessionEpoch) return;
    const known = new Set(ui.decisions.map((item) => item.request_id));
    ui.decisions.push(...(result.items || []).filter((item) => !known.has(item.request_id)));
    ui.cursor = result.next_after_seq ?? ui.cursor;
    ui.retainedFrom = result.retained_from_seq ?? 1;
    ui.historyMore = result.has_more === true;
    ui.historyStatus = "ready";
    ui.historyGap = ui.historyGap || result.log_complete === false;
  } catch (error) {
    if (runEpoch !== ui.runEpoch || sessionEpoch !== ui.sessionEpoch) return;
    ui.historyStatus = error.status === 410 ? "unavailable" : "error";
    ui.historyError = error;
  } finally {
    if (runEpoch === ui.runEpoch && sessionEpoch === ui.sessionEpoch) { ui.decisionsBusy = false; renderDecisions(); }
  }
}
function section(title) {
  const node = element("section", "detail-section");
  node.append(element("h4", "", title)); return node;
}
function dataGrid(values) {
  const grid = element("dl", "detail-grid");
  for (const [title, value] of values) { const pair = element("div"); pair.append(element("dt", "", title), element("dd", "", valueText(value))); grid.append(pair); }
  return grid;
}
function decisionView(data, { sessionId, requestId, seat, fromReport = false }) {
  const fragment = document.createDocumentFragment();
  const state = data.state?.state || data.state;
  const binding = state?.binding || data.execution?.binding || data.proposal?.binding;
  const intro = element("div", "detail-intro");
  const title = element("div");
  title.append(element("h4", "", `座位 ${binding?.seat ?? seat ?? "—"} · ${requestId.split(":").at(-1)}`), element("div", "request-id", requestId));
  intro.append(title, badge(name(data.execution?.status), toneFor(data.execution?.status))); fragment.append(intro);
  if (fromReport) fragment.append(element("p", "muted", "报告引用的原决策；牌桌观察座位保持不变。"));
  if (data.log_complete === false) fragment.append(element("div", "message warning", "记录不完整：缺失字段保留未知，不能据此补判动作结果。"));
  fragment.append(dataGrid([["所属小局", binding?.round_id], ["决策阶段", name(state?.stage)], ["动作来源", name(data.execution?.source || data.proposal?.source)], ["Session", short(sessionId)]]));
  if (state) {
    const facts = section("当时的可见事实");
    facts.append(dataGrid([["级牌", state.level_rank], ["领出座位", state.leader_seat], ["各座位余牌", state.remaining_counts], ["事实状态", name(state.facts_status)]]));
    const hand = element("div", "mini-cards");
    for (const card of state.self_hand || []) hand.append(cardNode(card, true));
    facts.append(element("p", "muted", "本人当时手牌"), hand);
    fragment.append(facts);
  }
  const execution = section("最终执行");
  execution.append(dataGrid([["动作编号", data.execution?.action_id || data.proposal?.action_id], ["确认时间", data.execution?.confirmed_at ? timestamp(data.execution.confirmed_at) : "—"]]));
  execution.append(actionCards(data.execution?.wire_action, true));
  if (!data.execution) execution.append(element("p", "muted", "尚无执行记录；建议不等于已提交。"));
  if (data.execution?.error) execution.append(errorLine(data.execution.error));
  if (data.execution?.feedback_event_id) execution.append(element("p", "request-id", `确认事件：${data.execution.feedback_event_id}`));
  fragment.append(execution);
  if (data.program_explanation?.length) {
    const explanation = section("程序事实说明 · 不评价策略优劣");
    for (const text of data.program_explanation) explanation.append(element("p", "", text));
    explanation.append(element("p", "muted", "根据原请求手牌和记录动作计算；不代表模型理由已通过核验。"));
    fragment.append(explanation);
  }
  const modelSource = ["main", "review", "repair"].includes(data.proposal?.source);
  const proposal = section(modelSource ? "模型原始理由 · 未核验" : "原始规则或兜底说明");
  proposal.append(element("div", "reason-box", data.proposal?.reason || "暂无已记录理由"));
  if (data.proposal?.cause_code) proposal.append(element("p", "candidate-reasons", `原因：${data.proposal.cause_code}`));
  fragment.append(proposal);
  const review = data.review || data.proposal?.review;
  const reviewSection = section("条件复核 / 输出修复");
  if (review) {
    reviewSection.append(element("p", "muted", "以下模型理由保留原文，未核验；复核状态不代表文字事实正确。"));
    reviewSection.append(badge(`${name(review.kind)} · ${name(review.status)}`, toneFor(review.status)));
    reviewSection.append(element("p", "candidate-reasons", `触发原因：${valueText(review.trigger_codes)}`));
    if (review.before) reviewSection.append(element("p", "", `原选 ${review.before.action_id}：${review.before.reason}`));
    if (review.after) reviewSection.append(element("div", "reason-box", `复核后 ${review.after.action_id}：${review.after.reason}`));
    if (review.error) reviewSection.append(errorLine(review.error));
  } else reviewSection.append(element("p", "muted", "无复核或修复记录。"));
  fragment.append(reviewSection);
  const candidates = section("候选动作与规则特征");
  const snapshot = data.candidates?.snapshot || data.candidates;
  if (snapshot?.candidates?.length) {
    candidates.append(element("p", "muted", `${snapshot.candidates.length} 个候选 · 工具 ${name(snapshot.tool_status)} · 规则首选 ${snapshot.rule_action_id || "—"}`));
    const list = element("div", "candidate-list");
    for (const candidate of snapshot.candidates) {
      const chosen = candidate.action_id === (data.execution?.action_id || data.proposal?.action_id);
      const row = element("div", `candidate${chosen ? " chosen" : ""}`);
      const heading = element("div", "candidate-title");
      heading.append(element("strong", "", candidate.action_id));
      if (chosen) heading.append(badge("最终选择", "good"));
      row.append(heading);
      const legal = (data.candidate_actions || []).find((action) => action.action_id === candidate.action_id);
      if (legal?.wire_action) {
        row.append(element("p", "muted", actionName(legal.wire_action)), actionCards(legal.wire_action, true));
      }
      const fields = Object.entries(candidate.features || {}).map(([key, value]) => `${featureLabels[key] || key} ${valueText(value)}`);
      row.append(element("div", "candidate-features", fields.join(" · ") || "特征未知"));
      row.append(element("div", "candidate-reasons", `保留原因：${valueText(candidate.keep_reasons?.map(name))}`));
      list.append(row);
    }
    candidates.append(list);
    if (snapshot.degradation_codes?.length) candidates.append(element("p", "error-text", `工具降级：${valueText(snapshot.degradation_codes)}`));
  } else candidates.append(element("p", "muted", "此请求没有已记录的候选快照。"));
  fragment.append(candidates);
  const calls = section("模型调用");
  if (!data.calls?.length) calls.append(element("p", "muted", "0 条调用记录 · 未记录用量显示 —"));
  for (const call of data.calls || []) {
    const row = element("div", "call-row");
    const heading = element("div", "call-heading");
    heading.append(element("span", "", name(call.role)), badge(call.error ? "调用失败" : call.decision ? "返回有效建议" : "结果未知", call.error ? "warning" : "neutral"));
    row.append(heading);
    const metrics = element("div", "call-metrics");
    metrics.append(element("span", "", `耗时 ${call.elapsed_ms == null ? "—" : `${Math.round(call.elapsed_ms)} ms`}`), element("span", "", `Tokens ${valueText(call.usage?.total_tokens)}`), element("span", "", `传输 ${valueText(call.transport_requests)}`));
    row.append(metrics);
    if (call.decision) row.append(element("p", "", `模型原文 · 未核验 · ${call.decision.action_id}：${call.decision.reason}`));
    if (call.error) row.append(errorLine(call.error));
    calls.append(row);
  }
  fragment.append(calls);
  return fragment;
}
async function openDecision(requestId) {
  if (!ui.session) return;
  const epoch = ++ui.detailEpoch, runEpoch = ui.runEpoch, sessionEpoch = ui.sessionEpoch;
  const sessionId = ui.session.session_id, seat = ui.session.seat;
  ui.selectedRequest = requestId; renderDecisions();
  replace("decision-detail", empty("正在读取决策…", "", true));
  try {
    const data = await api(`${sessionPath()}/decisions/${encodeURIComponent(requestId)}`);
    if (epoch !== ui.detailEpoch || runEpoch !== ui.runEpoch || sessionEpoch !== ui.sessionEpoch) return;
    replace("decision-detail", decisionView(data, { sessionId, requestId, seat }));
  } catch (error) { if (epoch === ui.detailEpoch && runEpoch === ui.runEpoch && sessionEpoch === ui.sessionEpoch) displayError($("decision-detail"), error, error.status === 410 ? "原决策超出保留窗口" : "决策不可查询"); }
}
function setMode(mode) {
  ui.mode = mode;
  $("table-view").hidden = mode !== "table"; $("reports-view").hidden = mode !== "reports";
  $("table-tab").classList.toggle("active", mode === "table"); $("reports-tab").classList.toggle("active", mode === "reports");
  $("table-tab").setAttribute("aria-pressed", String(mode === "table")); $("reports-tab").setAttribute("aria-pressed", String(mode === "reports"));
  $("workspace-title").textContent = mode === "table" ? "对局工作台" : "复盘与统计";
  if (mode === "reports") loadReports(); else pollState();
}
function renderReportList(result) {
  const list = $("report-list"); list.replaceChildren();
  if (!result.items?.length) list.append(empty("尚未生成报告",
    active(ui.run) ? "运行结束后可生成统计报告。" : ui.reportGenerationEnabled
      ? "可生成统计报告查看对局汇总，或选择座位生成模型复盘。"
      : ui.reportGenerationEnabled === false ? "此演示归档没有已有报告。" : "可刷新报告目录；报告生成能力尚未加载。", true));
  for (const report of result.items || []) {
    const button = element("button", `report-button${report.report_id === ui.report?.report_id ? " selected" : ""}`);
    button.title = report.report_id;
    button.append(element("strong", "", `${report.kind === "review" ? "复盘报告" : name(report.kind)} · ${short(report.report_id, 12)}`), element("small", "", timestamp(report.created_at)), badge(name(report.status), toneFor(report.status)));
    button.addEventListener("click", () => { for (const sibling of list.querySelectorAll("button")) sibling.classList.remove("selected"); button.classList.add("selected"); openReport(report.report_id); });
    list.append(button);
  }
  if (result.log_complete === false) list.append(element("p", "error-text", "源运行记录不完整"));
  for (const error of result.errors || []) list.append(errorLine(error));
}
async function loadReports({ refresh = false, afterAction = false } = {}) {
  if (ui.reportActionBusy && !afterAction) return;
  const epoch = ++ui.reportListEpoch, runEpoch = ui.runEpoch;
  if (!ui.run) { replace("report-list", empty("暂无运行", "先选择一个已有运行。", true)); return; }
  if (refresh) { ui.reportActionBusy = true; updateControls(); }
  replace("report-list", empty("正在读取…", "", true));
  try {
    const result = await api(`${runPath()}/reports${refresh ? "/refresh" : ""}`, refresh ? { method: "POST", body: "{}" } : {});
    if (epoch !== ui.reportListEpoch || runEpoch !== ui.runEpoch) return;
    renderReportList(result);
  } catch (error) {
    if (epoch === ui.reportListEpoch && runEpoch === ui.runEpoch) {
      const title = error.code === "RUN_NOT_CLOSED" ? "运行尚未关闭"
        : error.code === "RECORD_NOT_REGISTERED" ? "归档尚未就绪"
        : error.code === "REPORT_BUSY" ? "报告操作进行中" : "报告列表不可读取";
      displayError($("report-list"), error, title);
    }
  } finally {
    if (refresh && runEpoch === ui.runEpoch) { ui.reportActionBusy = false; updateControls(); }
  }
}
async function generateReport() {
  if ($("generate-report").disabled || !ui.run) return;
  const runEpoch = ui.runEpoch, detailEpoch = ui.reportDetailEpoch;
  ui.reportActionBusy = true; updateControls();
  const status = $("report-action-status");
  status.hidden = false; status.className = "inline-message";
  status.textContent = "正在根据记录生成统计报告… 模型调用 0 次。";
  closeEvidence();
  try {
    const report = await api(`${runPath()}/reports/evaluate`, { method: "POST", body: "{}" });
    if (runEpoch !== ui.runEpoch) return;
    status.textContent = "统计报告已生成 · 本次模型调用 0 次。";
    if (detailEpoch === ui.reportDetailEpoch) {
      ui.reportDetailEpoch += 1;
      ui.report = report;
      renderReport(report);
    }
    await loadReports({ afterAction: true });
  } catch (error) {
    if (runEpoch === ui.runEpoch) { status.textContent = `统计报告生成失败 · ${errorMessage(error)}`; status.className = "inline-message error-text"; }
  } finally {
    if (runEpoch === ui.runEpoch) { ui.reportActionBusy = false; updateControls(); }
  }
}
async function generateModelReview() {
  if ($("generate-model-review").disabled || !ui.run) return;
  const runId = ui.run.run_id, sessionId = ui.reviewSessionId;
  if (!ui.run.sessions.some((session) => session.session_id === sessionId)) return;
  const watchRunEpoch = ui.runEpoch, detailEpoch = ui.reportDetailEpoch;
  ui.reviewStarting = true; ui.reviewErrors.delete(runId); updateControls();
  try {
    const task = await api(`/runs/${encodeURIComponent(runId)}/reports/review`, {
      method: "POST", body: JSON.stringify({ session_id: sessionId }),
    });
    if (task.source_run_id !== runId) throw new Error("返回的复盘任务与源运行不一致。");
    const record = rememberReviewTask(task, sessionId);
    record.watchRunEpoch = watchRunEpoch; record.detailEpoch = detailEpoch;
    persistReviewTasks();
  } catch (error) {
    ui.reviewErrors.set(runId, `模型复盘未能创建 · ${errorMessage(error)}`);
  } finally {
    ui.reviewStarting = false; updateControls();
    // A read can recover an accepted task after a lost response; it never retries POST.
    await loadReviewCapabilities();
  }
}
async function presentReviewResult(record) {
  const task = record.task, runEpoch = ui.runEpoch;
  if (!task.report_ready || !task.report_id || ui.run?.run_id !== task.source_run_id
      || record.presentedEpoch === runEpoch || ui.reportActionBusy) return;
  record.presentedEpoch = runEpoch;
  await loadReports();
  if (runEpoch !== ui.runEpoch || ui.reviewTasks.get(task.source_run_id) !== record) return;
  const sameSeat = Boolean(record.sessionId) && record.sessionId === ui.reviewSessionId;
  if (record.watchRunEpoch === runEpoch && record.detailEpoch === ui.reportDetailEpoch && sameSeat) {
    await openReport(task.report_id, { reviewSessionId: record.sessionId });
  }
}
async function pollReviewTasks() {
  if (ui.reviewPollBusy) return;
  ui.reviewPollBusy = true;
  try {
    for (const record of [...ui.reviewTasks.values()]) {
      const task = record.task;
      if (reviewPending(task)) {
        try {
          const result = await api(`/runs/${encodeURIComponent(task.source_run_id)}/reports/review-tasks/${encodeURIComponent(task.task_id)}`);
          if (ui.reviewTasks.get(task.source_run_id) !== record || result.task_id !== task.task_id || result.source_run_id !== task.source_run_id) continue;
          record.task = result; record.pollError = "";
          if (result.report_ready || result.status === "failed") await loadReviewCapabilities();
        } catch (error) {
          if (ui.reviewTasks.get(task.source_run_id) !== record) continue;
          record.pollError = `任务状态读取失败 · ${errorMessage(error)}`;
          if (error.status === 404) {
            record.task = { ...task, status: "failed", error: { safe_message: "任务状态已失效，可刷新报告目录查找已保存的报告。" } };
            await loadReviewCapabilities();
          }
        }
      }
      await presentReviewResult(record);
    }
  } finally {
    ui.reviewPollBusy = false;
    renderReviewControls();
  }
}
function factBlock(title, value, later = false) {
  const block = element("div", `report-fact${later ? " later" : ""}`);
  block.append(element("h5", "", title));
  if (Array.isArray(value)) {
    if (!value.length) block.append(element("p", "muted", "无已记录内容"));
    for (const line of value) block.append(element("p", "", valueText(line)));
  } else block.append(element("p", "", valueText(value)));
  return block;
}
function renderReport(report) {
  const body = $("report-body"); body.replaceChildren();
  const review = report.kind === "review" || Array.isArray(report.items);
  body.append(element("h3", "", review ? "对局复盘" : "统计报告"), badge(name(report.status), toneFor(report.status)));
  body.append(element("p", "report-intro", `报告 ${report.report_id || "—"}\n源运行 ${report.run_id || valueText(report.run_ids)} · ${timestamp(report.created_at)}`));
  if (report.status === "partial") body.append(element("div", "message warning", "此报告仅部分完成。未覆盖的请求和未知结果不作推断。"));
  if (report.status === "failed") body.append(element("div", "message error", "报告生成失败。以下内容仅保留已有证据，不代表完整复盘。"));
  if (review) {
    if (report.coverage?.selection_scope === "selected_requests") {
      const selected = Array.isArray(report.coverage.selected_request_ids) ? report.coverage.selected_request_ids.length : "—";
      const reviewed = Array.isArray(report.coverage.reviewed_request_ids) ? report.coverage.reviewed_request_ids.length : "—";
      body.append(element("p", "report-intro", `复盘范围：已复盘 ${reviewed} / ${selected} 个选定请求；该座位共有 ${valueText(report.coverage.available_request_count)} 个可用请求。结论限于本次选取的证据。`));
    }
    body.append(element("p", "muted", "事实来源与核验状态见各条目。模型分析和建议仍可能出错，事实核验不代表策略有效。"));
    if (!report.items?.length) body.append(empty("报告没有复盘条目", "请结合报告状态和错误信息查看。", true));
    (report.items || []).forEach((item, index) => {
      const entry = element("section", "report-entry");
      const heading = element("div", "report-entry-header");
      heading.append(element("h4", "", `决策 ${String(index + 1).padStart(2, "0")} · ${item.request_id?.split(":").at(-1) || "未知请求"}`));
      const button = element("button", "text-button", "查看原决策 ↗");
      const canLink = item.evidence_available === true && report.run_id === ui.run?.run_id && Boolean(item.session_id && item.request_id);
      button.disabled = !canLink;
      button.addEventListener("click", () => openEvidence(report, item));
      heading.append(button); entry.append(heading);
      const programFacts = item.facts_source === "program";
      const verifiedFacts = programFacts && item.facts_verified === true && ["b09-offline-v2", "b09-offline-v3"].includes(report.generation?.prompt_version);
      const factsLabel = verifiedFacts ? "程序提取 · 已与源记录核对" : programFacts ? "声明为程序提取 · 未通过源记录核验" : "模型归纳 · 未核验";
      if (programFacts && !verifiedFacts) entry.append(element("div", "message warning", "本条目声明的程序事实未通过源记录核验，原文仅供查看，不能当作已核验事实。"));
      entry.append(factBlock(`当时事实 · ${factsLabel}`, item.at_time_facts), factBlock(`事后事实 · ${factsLabel} · 当时不可知`, item.later_facts, true), factBlock("模型分析 · 仍可能出错", item.assessment), factBlock("模型建议 · 待验证", item.suggestion));
      entry.append(element("p", "muted", "引用的原始事件"));
      const ids = element("div", "evidence-ids");
      for (const id of item.evidence_event_ids || []) ids.append(element("span", "evidence-id", id));
      if (!item.evidence_event_ids?.length) ids.append(element("span", "error-text", "未提供事件依据"));
      entry.append(ids);
      if (!canLink) entry.append(element("p", "error-text", item.evidence_error ? errorMessage(item.evidence_error) : "引用依据不足或不可查询，无法跳转原决策。"));
      body.append(entry);
    });
  } else {
    body.append(element("p", "report-intro", "程序根据对局记录汇总。需要决策分析，可生成模型复盘。"));
    const counts = report.counts || {}, rate = report.performance?.win_rate;
    const usage = report.model_usage?.online || {};
    const summary = section("关键统计");
    summary.append(dataGrid([
      ["样本单位", report.unit === "match" ? "整场" : report.unit === "round" ? "小局" : "—"],
      ["开始 / 已结算", `${valueText(counts.started)} / ${valueText(counts.settled)}`],
      ["胜 / 负 / 平", `${valueText(counts.win)} / ${valueText(counts.loss)} / ${valueText(counts.draw)}`],
      ["已结算样本胜率", typeof rate?.value === "number" ? `${(rate.value * 100).toFixed(1)}%（${valueText(rate.numerator)} / ${valueText(rate.denominator)}）` : "—"],
      ["待结算 / 中断", `${valueText(counts.pending)} / ${valueText(counts.interrupted)}`],
      ["原对局模型调用", usage.calls_started?.value],
      ["原对局总 Tokens", usage.total_tokens?.value],
      ["数据完整性", report.quality?.metrics_complete === true ? "统计所需记录完整" : report.quality?.metrics_complete === false ? "存在缺口，部分指标未知" : "—"],
    ]));
    summary.append(element("p", "muted", "未知指标保留为 —；胜率只以已有明确结算为分母。"));
    body.append(summary);
    for (const [key, title] of [["cohort", "评测条件"], ["counts", "样本与结算"], ["performance", "表现"], ["latency", "延迟"], ["model_usage", "原对局模型用量"], ["quality", "数据质量"]]) {
      const block = element("details", "report-entry report-data");
      block.append(element("summary", "", `${title} · 查看明细`), element("pre", "json-data", valueText(report[key]))); body.append(block);
    }
  }
  if (report.usage) body.append(element("p", "report-intro", `本报告用量 · Tokens ${valueText(report.usage.total_tokens)} · 成本 ${valueText(report.usage.cost_amount)}${report.usage.currency ? ` ${report.usage.currency}` : ""}`));
  for (const error of report.errors || []) body.append(errorLine(error));
}
async function openReport(reportId, { reviewSessionId = null } = {}) {
  const epoch = ++ui.reportDetailEpoch, runEpoch = ui.runEpoch;
  ui.report = null; closeEvidence();
  replace("report-body", empty("正在读取报告…", "", true));
  try {
    const report = await api(`${runPath()}/reports/${encodeURIComponent(reportId)}`);
    if (epoch !== ui.reportDetailEpoch || runEpoch !== ui.runEpoch) return;
    if (reviewSessionId && reviewSessionId !== ui.reviewSessionId) {
      replace("report-body", empty("复盘已生成", "复盘座位已切换，可从报告目录查看刚生成的原座位报告。"));
      return;
    }
    ui.report = report; renderReport(report);
  } catch (error) { if (epoch === ui.reportDetailEpoch && runEpoch === ui.runEpoch) displayError($("report-body"), error, "报告不可读取"); }
}
function closeEvidence() {
  ui.evidenceEpoch += 1;
  $("report-evidence-panel").hidden = true;
  $("reports-view").classList.remove("with-evidence");
  replace("report-decision-detail");
}
async function openEvidence(report, item) {
  if (item.evidence_available !== true || report.run_id !== ui.run?.run_id || !item.session_id || !item.request_id) return;
  const epoch = ++ui.evidenceEpoch, runEpoch = ui.runEpoch;
  $("report-evidence-panel").hidden = false; $("reports-view").classList.add("with-evidence");
  replace("report-decision-detail", empty("正在读取原决策…", "不改变牌桌观察座位。", true));
  try {
    const data = await api(`/runs/${encodeURIComponent(report.run_id)}/sessions/${encodeURIComponent(item.session_id)}/decisions/${encodeURIComponent(item.request_id)}`);
    if (epoch !== ui.evidenceEpoch || runEpoch !== ui.runEpoch || report !== ui.report) return;
    replace("report-decision-detail", decisionView(data, { sessionId: item.session_id, requestId: item.request_id, fromReport: true }));
  } catch (error) { if (epoch === ui.evidenceEpoch && runEpoch === ui.runEpoch) displayError($("report-decision-detail"), error, "引用的原决策不可查询"); }
}
async function startRun() {
  if ($("start-run").disabled) return;
  $("start-run").disabled = true; message("正在创建运行…");
  try {
    const result = await api("/runs", { method: "POST", body: JSON.stringify({ config_name: $("config-select").value }) });
    const run = { ...result, read_only: false };
    ui.runs.unshift(run); await selectRun(run.run_id);
    message("运行已创建。平台连接和启动结果会在状态中显示。");
  } catch (error) { message(errorMessage(error), "error"); }
  updateControls();
}
async function stopRun() {
  if (!ui.run || $("stop-run").disabled) return;
  const epoch = ui.runEpoch; $("stop-run").disabled = true;
  try {
    const result = await api(`${runPath()}/stop`, { method: "POST", body: "{}" });
    if (epoch !== ui.runEpoch) return;
    ui.run = { ...ui.run, ...result }; renderRun();
    message("已请求安全停止。未取得平台结算时，不会补判胜负。");
  } catch (error) { if (epoch === ui.runEpoch) message(errorMessage(error), "error"); }
  if (epoch === ui.runEpoch) updateControls();
}

$("run-select").addEventListener("change", (event) => selectRun(event.target.value));
$("session-select").addEventListener("change", (event) => selectSession(event.target.value, { manual: true }));
$("refresh-runs").addEventListener("click", () => loadRuns());
$("start-run").addEventListener("click", startRun);
$("stop-run").addEventListener("click", stopRun);
$("table-tab").addEventListener("click", () => setMode("table"));
$("reports-tab").addEventListener("click", () => setMode("reports"));
$("refresh-decisions").addEventListener("click", () => loadDecisions(true));
$("more-decisions").addEventListener("click", () => loadDecisions());
$("recover-history").addEventListener("click", () => loadDecisions(true, Math.max(0, ui.retainedFrom - 1)));
$("refresh-reports").addEventListener("click", () => loadReports({ refresh: true }));
$("generate-report").addEventListener("click", generateReport);
$("generate-model-review").addEventListener("click", generateModelReview);
$("review-session-select").addEventListener("change", (event) => {
  ui.reviewSessionId = event.target.value;
  ui.reviewSessionManual = true;
  renderReviewControls();
});
$("open-model-review").addEventListener("click", () => {
  const task = ui.reviewTasks.get(ui.run?.run_id)?.task;
  if (task?.report_ready && task.report_id) openReport(task.report_id);
});
$("close-evidence").addEventListener("click", closeEvidence);
restoreReviewTasks();
renderTable();
Promise.allSettled([loadConfigs(), loadRuns({ initial: true })]);
setInterval(pollState, 1000);
setInterval(pollReviewTasks, 1000);

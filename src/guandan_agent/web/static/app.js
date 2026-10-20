"use strict";

// This reader owns presentation only. No model, action, or replay endpoints exist.
const API = "/api/v1";
const $ = (id) => document.getElementById(id);
const ui = {
  runs: [], configs: [], run: null, session: null, view: null,
  runEpoch: 0, sessionEpoch: 0, detailEpoch: 0, reportEpoch: 0, evidenceEpoch: 0,
  pollBusy: false, decisionsBusy: false, cursor: 0, retainedFrom: 1,
  decisions: [], selectedRequest: null, mode: "table", report: null,
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
function updateControls() {
  const archival = ui.run?.read_only === true;
  $("start-run").disabled = !ui.configs.length || archival || active(ui.run) || ui.runs.some((run) => run.run_id !== ui.run?.run_id && active(run));
  $("stop-run").disabled = !ui.run || archival || !active(ui.run) || ui.run.status === "stopping";
  $("config-select").disabled = archival;
  $("start-run").title = archival ? "当前为只读归档；选择“新运行”后可启动配置" : "使用本地允许的配置创建运行";
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
    const option = element("option", "", `${run.read_only ? "归档" : "当前"} · ${run.label || short(run.run_id)} · ${name(run.status)}`);
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
  try {
    ui.configs = (await api("/configs")).items || [];
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
async function selectRun(id) {
  ui.runEpoch += 1;
  ui.sessionEpoch += 1;
  ui.detailEpoch += 1;
  ui.reportEpoch += 1;
  ui.evidenceEpoch += 1;
  ui.run = ui.runs.find((run) => run.run_id === id) || null;
  ui.session = null; ui.view = null; ui.report = null;
  ui.observedActiveMatchId = null;
  ui.followActiveMatch = true;
  ui.decisions = []; ui.cursor = 0; ui.selectedRequest = null;
  ui.decisionsBusy = false;
  message("");
  replace("decision-detail", empty("选择一条决策", "查看候选动作、模型理由和实际执行。"));
  replace("report-body", empty("选择一份已有报告", "浏览报告不会触发新的模型调用。"));
  closeEvidence();
  renderRunOptions(); renderRun(); renderSessions(); renderTable(); renderDecisions();
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
  $("session-select").value = id;
  $("history-message").hidden = true; $("recover-history").hidden = true;
  replace("decision-detail", empty("选择一条决策", "当前详情始终标明所属座位和请求。"));
  renderTable(); renderDecisions();
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
  $("level-value").textContent = state?.level_rank || "—";
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
  $("leading-type").textContent = state?.leading_action ? `${state.leading_action[0]} · ${state.leading_action[1]}` : (state ? "公开领出动作将在这里显示" : "平台公开信息将在这里显示");
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
    const run = await api(runPath());
    if (runEpoch !== ui.runEpoch) return;
    ui.run = { ...ui.run, ...run };
    ui.runs = ui.runs.map((item) => item.run_id === ui.run.run_id ? { ...item, ...ui.run } : item);
    renderRun(); renderSessions();
    if (!ui.session || ui.mode !== "table") return;
    sessionEpoch = ui.sessionEpoch;
    const result = await api(`${sessionPath()}/state`);
    if (runEpoch !== ui.runEpoch || sessionEpoch !== ui.sessionEpoch) return;
    if (result.session_id !== ui.session.session_id || result.run_id !== ui.run.run_id) return;
    if (ui.view && result.view_version <= ui.view.view_version) return;
    ui.view = result; renderTable();
  } catch (error) {
    if (runEpoch === ui.runEpoch && sessionEpoch === ui.sessionEpoch) $("sync-label").textContent = `同步失败 · ${errorMessage(error)}`;
  } finally { ui.pollBusy = false; }
}
function renderDecisions() {
  const list = $("decision-list");
  list.replaceChildren();
  if (!ui.decisions.length) list.append(empty("暂无决策记录", "等待请求到达后，点击“刷新记录”查看。", true));
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
}
async function loadDecisions(reset = false, forcedCursor) {
  if (!ui.session || ui.decisionsBusy) return;
  const runEpoch = ui.runEpoch, sessionEpoch = ui.sessionEpoch;
  ui.decisionsBusy = true;
  if (reset) { ui.decisions = []; ui.cursor = 0; }
  if (forcedCursor != null) ui.cursor = forcedCursor;
  $("more-decisions").disabled = true; $("refresh-decisions").disabled = true;
  const note = $("history-message"); note.hidden = true; $("recover-history").hidden = true;
  try {
    const result = await api(`${sessionPath()}/decisions?after_seq=${ui.cursor}&limit=30`);
    if (runEpoch !== ui.runEpoch || sessionEpoch !== ui.sessionEpoch) return;
    const known = new Set(ui.decisions.map((item) => item.request_id));
    ui.decisions.push(...(result.items || []).filter((item) => !known.has(item.request_id)));
    ui.cursor = result.next_after_seq ?? ui.cursor;
    ui.retainedFrom = result.retained_from_seq ?? 1;
    $("more-decisions").disabled = !result.has_more;
    if (result.log_complete === false) { note.textContent = "记录不完整，当前列表可能存在缺口。"; note.hidden = false; }
  } catch (error) {
    if (runEpoch !== ui.runEpoch || sessionEpoch !== ui.sessionEpoch) return;
    note.textContent = error.status === 410 ? "较早记录已超出可查询窗口，不能将当前尾部视为完整历史。" : errorMessage(error);
    note.hidden = false;
    if (error.status === 410) {
      if (Number.isInteger(error.retained_from_seq) && error.retained_from_seq >= 1) {
        ui.retainedFrom = error.retained_from_seq;
        $("recover-history").hidden = false;
      }
    }
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
        row.append(element("p", "muted", `${legal.wire_action[0]} · ${legal.wire_action[1]}`), actionCards(legal.wire_action, true));
      }
      const fields = Object.entries(candidate.features || {}).map(([key, value]) => `${featureLabels[key] || key} ${valueText(value)}`);
      row.append(element("div", "candidate-features", fields.join(" · ") || "特征未知"));
      row.append(element("div", "candidate-reasons", `保留原因：${valueText(candidate.keep_reasons)}`));
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
  $("workspace-title").textContent = mode === "table" ? "对局工作台" : "已生成复盘";
  if (mode === "reports") loadReports(); else pollState();
}
async function loadReports() {
  const epoch = ++ui.reportEpoch, runEpoch = ui.runEpoch;
  if (!ui.run) { replace("report-list", empty("暂无运行", "先选择一个已有运行。", true)); return; }
  replace("report-list", empty("正在读取…", "", true));
  try {
    const result = await api(`${runPath()}/reports`);
    if (epoch !== ui.reportEpoch || runEpoch !== ui.runEpoch) return;
    const list = $("report-list"); list.replaceChildren();
    if (!result.items?.length) list.append(empty("尚无报告", "报告需由离线复盘流程生成。", true));
    for (const report of result.items || []) {
      const button = element("button", "report-button");
      button.append(element("strong", "", `${report.kind === "review" ? "复盘报告" : name(report.kind)} · ${report.report_id}`), element("small", "", timestamp(report.created_at)), badge(name(report.status), toneFor(report.status)));
      button.addEventListener("click", () => { for (const sibling of list.querySelectorAll("button")) sibling.classList.remove("selected"); button.classList.add("selected"); openReport(report.report_id); });
      list.append(button);
    }
    if (result.log_complete === false) list.append(element("p", "error-text", "源运行记录不完整"));
    for (const error of result.errors || []) list.append(errorLine(error));
  } catch (error) { if (epoch === ui.reportEpoch && runEpoch === ui.runEpoch) displayError($("report-list"), error, error.status === 409 ? "运行尚未关闭" : "报告列表不可读取"); }
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
      const verifiedFacts = programFacts && item.facts_verified === true && report.generation?.prompt_version === "b09-offline-v2";
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
    for (const [key, title] of [["cohort", "评测条件"], ["counts", "样本与结算"], ["performance", "表现"], ["latency", "延迟"], ["model_usage", "模型用量"], ["quality", "数据质量"]]) {
      const block = element("section", "report-entry");
      block.append(element("h4", "", title), element("pre", "json-data", valueText(report[key]))); body.append(block);
    }
  }
  if (report.usage) body.append(element("p", "report-intro", `本报告用量 · Tokens ${valueText(report.usage.total_tokens)} · 成本 ${valueText(report.usage.cost_amount)}${report.usage.currency ? ` ${report.usage.currency}` : ""}`));
  for (const error of report.errors || []) body.append(errorLine(error));
}
async function openReport(reportId) {
  const epoch = ++ui.reportEpoch, runEpoch = ui.runEpoch;
  ui.report = null; closeEvidence();
  replace("report-body", empty("正在读取报告…", "", true));
  try {
    const report = await api(`${runPath()}/reports/${encodeURIComponent(reportId)}`);
    if (epoch !== ui.reportEpoch || runEpoch !== ui.runEpoch) return;
    ui.report = report; renderReport(report);
  } catch (error) { if (epoch === ui.reportEpoch && runEpoch === ui.runEpoch) displayError($("report-body"), error, "报告不可读取"); }
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
$("refresh-reports").addEventListener("click", loadReports);
$("close-evidence").addEventListener("click", closeEvidence);
renderTable();
Promise.allSettled([loadConfigs(), loadRuns({ initial: true })]);
setInterval(pollState, 1000);

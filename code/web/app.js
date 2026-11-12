"use strict";
/* 种蛋模型网页界面。所有来自文件的内容都用 textContent 放进页面，不拼 HTML。 */

const TOKEN = document.querySelector('meta[name="egg-token"]').content;
const SVG_NS = "http://www.w3.org/2000/svg";
const POLL_RUNNING_MS = 700;
const POLL_IDLE_MS = 3000;
const SERIES_COLORS = 8;            // style.css 里定义了 --series-1 … --series-8
const MAX_DIRECT_LABELS = 4;        // 线不多于这个数才在线尾直接标名字
const TABLE_NAMES = {
  "ablation.csv": "消融结果（决策日）",
  "ablation_per_day.csv": "消融结果（每天）",
  "results_per_day.csv": "按天成绩",
  "results_per_fold.csv": "按折成绩",
  "decision_day_comparison.csv": "决策日对比",
  "decision_day_per_batch.csv": "按批次",
  "per_day_comparison.csv": "逐日比较（校正）",
  "calibration.csv": "概率校准",
  "failure_summary.csv": "错误分组",
  "failure_cases.csv": "失败案例",
  "repeat_cv.csv": "重复交叉验证",
  "holdout.csv": "独立测试集",
  "holdout_调试_非独立验证.csv": "独立测试集（调试）",
  "dev_mae.csv": "发育误差",
  "folds.csv": "分折",
  "predictions_oof.csv": "逐蛋预测",
};
const TABLE_ORDER = Object.keys(TABLE_NAMES);
const CHART_METRICS = [["决策效用均值", "决策效用"], ["准确率均值", "准确率"], ["AUC均值", "AUC"]];

// ---------- 小工具 ----------
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child);
  }
  return node;
}

function svg(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Egg-Token": TOKEN },
    body: JSON.stringify(body),
  };
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new Error("连不上后台程序：启动网页界面的那个黑色窗口是不是被关掉了？");
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `请求失败（${response.status}）`);
  return data;
}

let toastTimer = 0;
function toast(message) {
  const box = document.getElementById("toast");
  box.textContent = message;
  box.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { box.hidden = true; }, 5000);
}

const isNumber = (text) => text !== "" && text !== null && Number.isFinite(Number(text));
const fmt = (value, digits = 3) => (value === null || value === undefined || !Number.isFinite(Number(value)))
  ? "—" : Number(value).toFixed(digits);
const signed = (value, digits = 3) => (Number(value) > 0 ? "+" : "") + fmt(value, digits);
const seriesColor = (index) => `var(--series-${(index % SERIES_COLORS) + 1})`;

// ---------- 页面切换 ----------
const views = ["tasks", "predictions", "runs"];
let currentRun = null;

function route() {
  const [view, ...rest] = decodeURIComponent(location.hash.slice(1)).split("/");
  const name = views.includes(view) ? view : "tasks";
  for (const v of views) document.getElementById(`view-${v}`).hidden = v !== name;
  for (const item of document.querySelectorAll(".nav-item")) {
    if (item.dataset.view === name) item.setAttribute("aria-current", "page");
    else item.removeAttribute("aria-current");
  }
  if (name === "predictions") loadPredictions();
  if (name === "runs") loadRuns(rest.join("/") || null);
}

for (const item of document.querySelectorAll(".nav-item")) {
  item.addEventListener("click", () => { location.hash = item.dataset.view; });
}
window.addEventListener("hashchange", route);

// ---------- 运行任务 ----------
const logBox = document.getElementById("task-log");
const actionButtons = [...document.querySelectorAll("[data-action]")];
let nextLine = 0;
let wasRunning = false;
let seenTask = null;
let pollTimer = 0;

function taskParams(action) {
  if (action === "predict") return { data: document.getElementById("predict-data").value };
  if (action === "train" || action === "self_check" || action === "ablation" || action === "repeat_cv") {
    return {
      data: document.getElementById("train-data").value,
      decision_day: document.getElementById("train-day").value,
      skip_holdout: document.getElementById("train-skip").checked,
    };
  }
  return {};
}

async function startTask(action) {
  try {
    await api("/api/run", { action, ...taskParams(action) });
    logBox.textContent = "";
    nextLine = 0;
    seenTask = null;
    poll();
  } catch (error) {
    toast(error.message);
  }
}

for (const button of actionButtons) {
  if (button.type === "submit") continue;
  button.addEventListener("click", () => startTask(button.dataset.action));
}
for (const form of document.querySelectorAll("form")) {
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    startTask(form.querySelector('[type="submit"]').dataset.action);
  });
}
document.getElementById("task-stop").addEventListener("click", async () => {
  try { await api("/api/stop", {}); } catch (error) { toast(error.message); }
});
for (const button of document.querySelectorAll("[data-pick]")) {
  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      const { path } = await api("/api/pick", {});
      if (path) document.getElementById(button.dataset.pick).value = path;
    } catch (error) {
      toast(error.message);
    } finally {
      button.disabled = false;
    }
  });
}

const OUTCOME_STYLE = { ok: "ok", self_check_failed: "bad", holdout_blocked: "warn", stopped: "warn", error: "bad" };

function showTask(task) {
  const dot = document.getElementById("task-dot");
  const status = document.getElementById("task-status");
  const message = document.getElementById("task-message");
  const stop = document.getElementById("task-stop");
  const open = document.getElementById("task-open-run");
  const running = Boolean(task && task.running);
  for (const button of actionButtons) button.disabled = running;
  stop.hidden = !running;
  if (!task) return;

  const key = `${task.name}@${task.started}`;
  if (key !== seenTask) {            // 换了任务（或刚打开页面）：日志从头显示
    seenTask = key;
    logBox.textContent = "";
  }
  if (task.lines.length) {
    const atBottom = logBox.scrollHeight - logBox.scrollTop - logBox.clientHeight < 40;
    logBox.append(task.lines.join("\n") + "\n");
    if (atBottom) logBox.scrollTop = logBox.scrollHeight;
  }
  nextLine = task.next;

  if (running) {
    dot.className = "dot running";
    status.textContent = `正在${task.name}…（${task.started.slice(11)} 开始）`;
    message.hidden = true;
    open.hidden = true;
    return;
  }
  const style = OUTCOME_STYLE[task.outcome] || "bad";
  dot.className = `dot ${style}`;
  status.textContent = task.outcome === "ok" ? `${task.name}完成` : `${task.name}没有完成`;
  message.hidden = task.outcome === "ok";
  message.className = `notice ${style}`;
  message.textContent = task.message;
  if (task.outcome === "error") {     // 出错时把日志最后一行（通常就是原因）直接摆出来
    const last = logBox.textContent.trimEnd().split("\n").pop().replace(/^\w*(Error|Exception):\s*/, "");
    if (last) message.append(el("div", {}, el("b", { text: "原因：" }), last));
  }
  const toPredictions = task.action === "predict";
  open.hidden = task.outcome === "stopped" || task.outcome === "error" || (toPredictions && task.outcome !== "ok");
  open.textContent = toPredictions ? "查看判断结果" : "查看这次运行的记录";
  open.href = toPredictions ? "#predictions" : "#runs";
}

function showModel(model) {
  const box = document.getElementById("model-info");
  box.textContent = "";
  if (!model) {
    box.textContent = "还没有模型，请先训练。";
    return;
  }
  if (!model.run) {
    box.append(`保存于 ${model.saved}`, el("small", { text: "对不上任何一次运行记录。" }));
    return;
  }
  box.append(el("strong", { text: model.chosen || "深度模型" }),
    `，决策日第 ${model.decision_day} 天`,
    el("small", { text: `来自 ${model.run}` }));
}

async function poll() {
  clearTimeout(pollTimer);
  let running = false;
  try {
    const state = await api(`/api/state?since=${nextLine}`);
    showTask(state.task);
    showModel(state.model);
    const placeholder = state.sample_csv ? "不填 = 用样例数据" : "填写新蛋的 meta.csv 位置";
    document.getElementById("predict-data").placeholder = placeholder;
    running = Boolean(state.task && state.task.running);
    if (wasRunning && !running && !document.getElementById("view-runs").hidden) loadRuns(null);
    wasRunning = running;
  } catch (error) {
    document.getElementById("task-status").textContent = error.message;
    document.getElementById("task-dot").className = "dot bad";
  }
  pollTimer = setTimeout(poll, running ? POLL_RUNNING_MS : POLL_IDLE_MS);
}

// ---------- 表格 ----------
function buildTable(table, decorate) {
  if (!table.columns.length) return el("p", { class: "empty", text: "这张表是空的。" });
  const numeric = table.columns.map((_, i) => table.rows.length > 0
    && table.rows.every((row) => row[i] === "" || row[i] === undefined || isNumber(row[i])));
  const head = el("tr", {}, table.columns.map((name, i) =>
    el("th", { class: numeric[i] ? "n" : null, title: name, scope: "col", text: name })));
  const body = table.rows.map((row) => el("tr", {}, table.columns.map((name, i) => {
    const cell = el("td", { class: numeric[i] ? "n" : null });
    const custom = decorate ? decorate(name, row[i] ?? "") : null;
    cell.append(custom || (row[i] ?? ""));
    return cell;
  })));
  const wrap = el("div", { class: "table-wrap", tabindex: "0" },
    el("table", {}, el("thead", {}, head), el("tbody", {}, body)));
  if (table.total <= table.rows.length) return wrap;
  return el("div", {}, wrap, el("p", { class: "table-note",
    text: `只显示前 ${table.rows.length} 行，共 ${table.total} 行；完整内容请打开文件。` }));
}

// ---------- 判断结果 ----------
async function loadPredictions() {
  const body = document.getElementById("pred-body");
  const sub = document.getElementById("pred-sub");
  try {
    const table = await api("/api/predictions");
    body.textContent = "";
    if (!table.saved) {
      sub.textContent = "每颗蛋在决策日的判断。";
      body.append(el("div", { class: "panel empty", text: "还没有判断结果。到“运行任务”里点“开始判断”。" }));
      return;
    }
    const judgeAt = table.columns.indexOf("判断");
    const judged = table.rows.filter((row) => row[judgeAt] !== "无法判断").length;
    sub.textContent = `${table.total} 颗蛋，其中 ${judged} 颗在决策日给出了判断。保存于 ${table.saved}。`;
    const decorate = (name, value) => (name === "判断" && value === "无法判断")
      ? el("span", { class: "badge blocked", text: value }) : null;
    body.append(el("div", { class: "panel" }, buildTable(table, decorate),
      el("p", { class: "table-note", text: `每天一行的明细在 ${table.daily_csv}` })));
  } catch (error) {
    body.textContent = "";
    body.append(el("div", { class: "panel empty", text: error.message }));
  }
}

// ---------- 历史运行 ----------
async function loadRuns(wanted) {
  const list = document.getElementById("run-list");
  let runs;
  try {
    runs = await api("/api/runs");
  } catch (error) {
    list.textContent = error.message;
    return;
  }
  list.textContent = "";
  const detail = document.getElementById("run-detail");
  if (!runs.length) {
    detail.textContent = "";
    detail.append(el("div", { class: "panel empty", text: "还没有运行记录。" }));
    return;
  }
  const names = runs.map((run) => run.name);
  const selected = names.includes(wanted) ? wanted : names[0];      // 没指定就看最新的一次
  for (const run of runs) {
    const time = run.name.slice(0, 10) + " " + run.name.slice(11, 13) + ":" + run.name.slice(13, 15);
    list.append(el("button", {
      class: "run-item", role: "option", "data-name": run.name, "aria-selected": String(run.name === selected),
      onclick: () => { location.hash = `runs/${run.name}`; },
    },
    el("div", { class: "run-item-top" },
      el("span", { class: "run-kind", text: run.kind }),
      el("span", { class: `badge ${run.status}`, text: run.status_text })),
    el("span", { class: "run-time", text: time })));
  }
  showRun(selected);
}

function tile(label, value, note, textual) {
  return el("div", { class: "tile" },
    el("div", { class: "label", text: label }),
    el("div", { class: `tile-value${textual ? " text" : ""}`, text: value }),
    note ? el("div", { class: "tile-note", text: note }) : null);
}

function checkRow(what, state, text, why) {
  return el("div", { class: "check-row" },
    el("span", { class: "what", text: what }),
    el("span", { class: `badge ${state}`, text }),
    el("span", { class: "why", text: why || "" }));
}

function buildChecks(config) {
  const rows = [];
  const check = config["自检"];
  if (check) {
    const ok = check["总体"] === "通过";
    const bad = Object.entries(check["各方法"] || {}).filter(([, v]) => v !== "通过").map(([k, v]) => `${k}：${v}`);
    rows.push(checkRow("打乱标签自检", ok ? "ok" : "failed", check["总体"],
      ok ? "标签打乱后各方法都回到瞎猜水平" : bad.join("；")));
  } else if (typeof config["自检通过"] === "boolean") {
    rows.push(checkRow("打乱标签自检", config["自检通过"] ? "ok" : "failed", config["自检通过"] ? "通过" : "未通过"));
  }
  const controls = config["负对照"];
  if (controls) {
    const background = controls["背景对照"];
    if (background) {
      rows.push(checkRow("背景负对照", background["结果"] === "通过" ? "ok" : "failed", background["结果"],
        `只看蛋外面的背景猜性别，准确率 ${fmt(background["平均准确率"])}`));
    } else {
      rows.push(checkRow("背景负对照", "", "没做", "数据里没有背景光谱"));
    }
    const probe = controls["批次可分性"];
    if (probe) {
      rows.push(checkRow("批次差异", probe["批次差异很大"] ? "blocked" : "ok", probe["批次差异很大"] ? "很大" : "不大",
        `从光谱猜批次的准确率 ${fmt(probe["认批次准确率"])}，瞎猜水平 ${fmt(probe["瞎猜水平"])}`));
    }
  }
  const release = config["模型发布"];
  if (release) {
    rows.push(checkRow("模型发布", release["已发布"] ? "ok" : "failed", release["已发布"] ? "已保存" : "未保存",
      (release["未发布原因"] || []).join("；")));
  }
  if (!rows.length) return null;
  return el("div", { class: "panel" }, el("h2", { text: "检查" }), el("div", { class: "checks" }, rows));
}

function buildTiles(config) {
  const result = config["决策日结果"];
  if (!result) return null;
  const tiles = [];
  const eggs = config["开发集蛋数"] ?? config["蛋数"];
  const batches = config["开发集批次数"] ?? config["分组数"];
  if (config["部署方法"]) {
    tiles.push(tile("选定的方法", config["部署方法"],
      result["深度模型明显更好"] ? "深度模型明显好过基线" : "深度模型没有明显好过基线", true));
  }
  tiles.push(tile(`深度模型（第 ${config["决策日"]} 天）`, fmt(result["深度模型决策效用"]), "交叉验证决策效用"));
  tiles.push(tile(`基线 ${result["主比较基线"] || result["最好的基线"] || ""}`, fmt(result["基线决策效用"]), "交叉验证决策效用"));
  const interval = result["差值95%区间"];
  tiles.push(tile("深度减基线", signed(result["差值"]),
    `单侧 p = ${fmt(result["单侧p"])}${interval ? `，95% 区间 ${signed(interval[0])} 到 ${signed(interval[1])}` : ""}`));
  if (eggs !== undefined) tiles.push(tile("开发集", `${eggs} 颗`, batches !== undefined ? `${batches} 个批次` : ""));
  return el("div", { class: "tiles" }, tiles);
}

function buildHoldout(config, table) {
  const info = config["独立测试集"];
  if (!info || !table) return null;
  const at = (name) => table.columns.indexOf(name);
  const all = table.rows.find((row) => row[at("范围")] === "全部");
  if (!all) return null;
  const formal = info["正式结果"] ?? (info["holdout_views"] === undefined || info["holdout_views"] === 1);
  const perBatch = info["按批次决策效用"];
  const tiles = [
    tile("独立测试集决策效用", fmt(all[at("决策效用")]),
      `${all[at("方法")]}，${all[at("判对的蛋数")]} / ${all[at("蛋数")]} 颗判对`),
    tile("测试集", `${info["蛋数"]} 颗`, `批次 ${(info["批次"] || []).join("、")}`),
  ];
  if (perBatch) {
    tiles.push(tile("按批次", fmt(perBatch["平均"]),
      `${perBatch["批次数"]} 个批次，95% 区间 ${fmt(perBatch["95%区间"][0])} 到 ${fmt(perBatch["95%区间"][1])}`));
  }
  return el("div", { class: "panel" },
    el("div", { class: "panel-head" }, el("h2", { text: "独立测试集" }),
      el("span", { class: `badge ${formal ? "ok" : "blocked"}`, text: info["性质"] || `第 ${info["holdout_views"] ?? "?"} 次查看` })),
    el("div", { class: "tiles" }, tiles));
}

async function showRun(name) {
  currentRun = name;
  const detail = document.getElementById("run-detail");
  let run;
  try {
    run = await api(`/api/runs/${encodeURIComponent(name)}`);
  } catch (error) {
    detail.textContent = "";
    detail.append(el("div", { class: "panel empty", text: error.message }));
    return;
  }
  if (currentRun !== name) return;          // 等回应的时候又点了别的
  const config = run.config || {};
  const tableCache = new Map();
  const getTable = async (file) => {
    if (!tableCache.has(file)) {
      tableCache.set(file, await api(`/api/runs/${encodeURIComponent(name)}/table/${encodeURIComponent(file)}`));
    }
    return tableCache.get(file);
  };
  const has = (file) => run.tables.includes(file);
  const [perDay, holdout] = await Promise.all([
    has("results_per_day.csv") ? getTable("results_per_day.csv").catch(() => null) : null,
    has("holdout.csv") ? getTable("holdout.csv").catch(() => null) : null,
  ]);
  if (currentRun !== name) return;

  detail.textContent = "";
  detail.append(el("div", { class: "detail-head" },
    el("h2", { text: run.kind }),
    el("span", { class: `badge ${run.status}`, text: run.status_text }),
    el("span", { class: "detail-name", text: run.name })));
  if (config["标签含义"]) detail.append(el("div", { class: "notice warn block", text: config["标签含义"] }));
  if (config["说明"]) detail.append(el("div", { class: "notice block", text: config["说明"] }));
  if (config["已拦截"]) detail.append(el("div", { class: "notice warn block", text: config["已拦截"] }));
  if (!run.config) {
    detail.append(el("div", { class: "notice block", text: "这次运行没有留下配置记录，多半是中途出错或被停止了。下面的日志里有原因。" }));
  }
  for (const part of [buildTiles(config), buildHoldout(config, holdout), buildChecks(config)]) {
    if (part) detail.append(part);
  }
  if (perDay && perDay.rows.length) detail.append(buildChartPanel(perDay, config));
  if (run.tables.length) detail.append(buildTablesPanel(run.tables, getTable));
  if (run.log) {
    detail.append(el("details", { class: "panel", open: !run.config || Boolean(config["已拦截"]) },
      el("summary", { text: "运行日志" }), el("pre", { class: "log", tabindex: "0", text: run.log })));
  }
}

function buildTablesPanel(files, getTable) {
  const ordered = [...files].sort((a, b) => {
    const [ia, ib] = [TABLE_ORDER.indexOf(a), TABLE_ORDER.indexOf(b)];
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
  });
  const body = el("div");
  const tabs = el("div", { class: "tabs", role: "tablist" });
  const show = async (file) => {
    for (const tab of tabs.children) tab.setAttribute("aria-selected", String(tab.dataset.file === file));
    body.textContent = "读取中…";
    try {
      const table = await getTable(file);
      body.textContent = "";
      body.append(buildTable(table));
    } catch (error) {
      body.textContent = error.message;
    }
  };
  for (const file of ordered) {
    tabs.append(el("button", { class: "tab", role: "tab", "data-file": file, title: file,
      text: TABLE_NAMES[file] || file, onclick: () => show(file) }));
  }
  show(ordered[0]);
  return el("div", { class: "panel" }, el("h2", { text: "结果表" }), tabs, body);
}

// ---------- 图表 ----------
const tooltip = document.getElementById("tooltip");

function showTooltip(event, title, rows) {
  tooltip.textContent = "";
  tooltip.append(el("div", { class: "tt-title", text: title }));
  for (const row of rows) {
    const key = el("span", {}, row.color ? el("i", { class: "swatch", style: `background:${row.color}` }) : null, row.name);
    tooltip.append(el("div", { class: "tt-row" }, key, el("span", { text: row.value })));
  }
  tooltip.hidden = false;
  const box = tooltip.getBoundingClientRect();
  const left = event.clientX + 16 + box.width > window.innerWidth ? event.clientX - box.width - 16 : event.clientX + 16;
  tooltip.style.left = `${Math.max(8, left)}px`;
  tooltip.style.top = `${Math.max(8, Math.min(event.clientY - 12, window.innerHeight - box.height - 8))}px`;
}
const hideTooltip = () => { tooltip.hidden = true; };

function buildChartPanel(perDay, config) {
  const at = (name) => perDay.columns.indexOf(name);
  const methods = [...new Set(perDay.rows.map((row) => row[at("方法")]))];
  const days = [...new Set(perDay.rows.map((row) => Number(row[at("孵化日")])))].sort((a, b) => a - b);
  const metrics = CHART_METRICS.filter(([column]) => at(column) >= 0);
  const holder = el("div", { class: "chart" });
  const title = el("h2");
  const seg = el("div", { class: "seg", role: "group", "aria-label": "指标" });
  let current = null;
  let drawnWidth = 0;
  const draw = (column, label) => {
    current = [column, label];
    drawnWidth = Math.max(Math.round(holder.clientWidth), 240);
    for (const button of seg.children) button.setAttribute("aria-pressed", String(button.dataset.column === column));
    const series = methods.map((method, index) => ({
      name: method, color: seriesColor(index),
      values: days.map((day) => {
        const row = perDay.rows.find((r) => r[at("方法")] === method && Number(r[at("孵化日")]) === day);
        return row && isNumber(row[at(column)]) ? Number(row[at(column)]) : null;
      }),
    }));
    title.textContent = days.length > 1 ? `各孵化日的${label}（交叉验证）` : `第 ${days[0]} 天的${label}（交叉验证）`;
    holder.textContent = "";
    holder.append(days.length > 1 ? lineChart(series, days, label, config["决策日"], drawnWidth)
      : barChart(series, label, drawnWidth));
  };
  // 按面板的实际宽度画，字号不随窗口缩放；宽度变了就重画
  new ResizeObserver(() => {
    if (current && holder.isConnected && Math.abs(holder.clientWidth - drawnWidth) > 8) draw(...current);
  }).observe(holder);
  for (const [column, label] of metrics) {
    seg.append(el("button", { type: "button", "data-column": column, text: label, onclick: () => draw(column, label) }));
  }
  current = metrics[0];
  requestAnimationFrame(() => draw(...current));       // 等面板放进页面、有了宽度再画
  return el("div", { class: "panel" }, el("div", { class: "panel-head" }, title, metrics.length > 1 ? seg : null), holder);
}

function legend(series, box) {
  return el("div", { class: "legend" }, series.map((s) => el("span", { class: "legend-item" },
    el("i", { class: `swatch${box ? " box" : ""}`, style: `background:${s.color}` }), s.name)));
}

/* 折线图：横轴孵化日，纵轴 0–1；0.5 是瞎猜水平，竖虚线是决策日。 */
function lineChart(series, days, label, decisionDay, W) {
  const direct = series.length <= MAX_DIRECT_LABELS && W >= 460;
  const H = 300, left = 40, right = direct ? 96 : 16, top = 14, bottom = 40;
  const x = (i) => left + (i / (days.length - 1)) * (W - left - right);
  const y = (v) => top + (1 - v) * (H - top - bottom);
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img",
    "aria-label": `${label}随孵化日的变化，共 ${series.length} 种方法；具体数字见下方的结果表` });

  for (const tick of [0, 0.25, 0.5, 0.75, 1]) {
    if (tick !== 0.5) root.append(svg("line", { class: "grid", x1: left, x2: W - right, y1: y(tick), y2: y(tick) }));
    root.append(svg("text", { class: "tick", x: left - 8, y: y(tick) + 4, "text-anchor": "end" }, tick.toFixed(2)));
  }
  root.append(svg("line", { class: "ref", x1: left, x2: W - right, y1: y(0.5), y2: y(0.5) }));
  root.append(svg("text", { class: "ref-label", x: W - right - 4, y: y(0.5) + 14, "text-anchor": "end" }, "瞎猜水平"));
  days.forEach((day, i) => root.append(svg("text", { class: "tick", x: x(i), y: H - bottom + 18, "text-anchor": "middle" }, String(day))));
  root.append(svg("text", { class: "axis-title", x: (left + W - right) / 2, y: H - 4, "text-anchor": "middle" }, "孵化日（天）"));
  const decisionAt = days.indexOf(Number(decisionDay));
  if (decisionAt >= 0) {
    root.append(svg("line", { class: "mark-day", x1: x(decisionAt), x2: x(decisionAt), y1: top, y2: H - bottom }));
    root.append(svg("text", { class: "ref-label", x: x(decisionAt) + 5, y: top + 10 }, "决策日"));
  }

  for (const s of series) {
    let path = "", pen = "M";
    s.values.forEach((value, i) => {
      if (value === null) { pen = "M"; return; }
      path += `${pen}${x(i).toFixed(1)},${y(value).toFixed(1)}`;
      pen = "L";
    });
    root.append(svg("path", { class: "line", d: path, stroke: s.color }));
    s.values.forEach((value, i) => {
      if (value !== null) root.append(svg("circle", { class: "pt", cx: x(i), cy: y(value), r: 4, fill: s.color }));
    });
  }

  if (direct) {                // 线尾标名字；挤在一起时上下错开
    const labels = series.map((s) => ({ s, at: s.values.findLastIndex((v) => v !== null) }))
      .filter((item) => item.at === days.length - 1)
      .map((item) => ({ ...item, y: y(item.s.values[item.at]) }))
      .sort((a, b) => a.y - b.y);
    const GAP = 15;
    for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + GAP);
    const overflow = labels.length ? labels[labels.length - 1].y - (H - bottom) : 0;
    if (overflow > 0) for (const item of labels) item.y -= overflow;
    for (let i = labels.length - 2; i >= 0; i--) labels[i].y = Math.min(labels[i].y, labels[i + 1].y - GAP);
    for (const item of labels) {
      root.append(svg("text", { class: "end-label", x: W - right + 10, y: item.y + 4 }, item.s.name));
    }
  }

  const cross = svg("line", { class: "cross", y1: top, y2: H - bottom, visibility: "hidden" });
  root.append(cross);
  const hit = svg("rect", { class: "hit", x: left - 10, y: top, width: W - left - right + 20, height: H - top - bottom });
  root.append(hit);
  hit.addEventListener("pointermove", (event) => {
    const box = root.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * W;
    const i = Math.max(0, Math.min(days.length - 1, Math.round(((px - left) / (W - left - right)) * (days.length - 1))));
    cross.setAttribute("x1", x(i));
    cross.setAttribute("x2", x(i));
    cross.setAttribute("visibility", "visible");
    const rows = series.filter((s) => s.values[i] !== null)
      .sort((a, b) => b.values[i] - a.values[i])
      .map((s) => ({ name: s.name, color: s.color, value: fmt(s.values[i]) }));
    showTooltip(event, `第 ${days[i]} 天 · ${label}`, rows);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTooltip(); });
  return el("div", {}, root, legend(series, false));
}

/* 只有一个孵化日时：每种方法一根横条，数值直接标在条尾。 */
function barChart(series, label, W) {
  const rows = series.filter((s) => s.values[0] !== null);
  const rowH = 34, left = 130, right = 64, top = 8, bottom = 30;
  const H = top + rows.length * rowH + bottom;
  const x = (v) => left + v * (W - left - right);
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img",
    "aria-label": `${rows.length} 种方法的${label}；具体数字标在每根横条末端` });
  for (const tick of [0, 0.25, 0.5, 0.75, 1]) {
    root.append(svg("line", { class: tick === 0.5 ? "ref" : "grid", x1: x(tick), x2: x(tick), y1: top, y2: H - bottom }));
    root.append(svg("text", { class: "tick", x: x(tick), y: H - bottom + 17, "text-anchor": "middle" }, tick.toFixed(2)));
  }
  root.append(svg("text", { class: "ref-label", x: x(0.5) + 5, y: H - 2 }, "0.50 = 瞎猜水平"));
  rows.forEach((s, i) => {
    const value = s.values[0], cy = top + i * rowH + rowH / 2, h = 18, w = Math.max(x(value) - x(0), 1), r = Math.min(4, w);
    root.append(svg("text", { class: "bar-label", x: left - 10, y: cy + 4.5, "text-anchor": "end" }, s.name));
    root.append(svg("path", { fill: "var(--series-1)",       // 一种量度，一种颜色；名字标在左边
      d: `M${x(0)},${cy - h / 2}h${w - r}a${r},${r} 0 0 1 ${r},${r}v${h - 2 * r}a${r},${r} 0 0 1 ${-r},${r}h${-(w - r)}z` }));
    root.append(svg("text", { class: "bar-value", x: x(value) + 8, y: cy + 4.5 }, fmt(value)));
    const hit = svg("rect", { class: "hit", x: 0, y: cy - rowH / 2, width: W, height: rowH });
    hit.addEventListener("pointermove", (event) => showTooltip(event, s.name, [{ name: label, value: fmt(value) }]));
    hit.addEventListener("pointerleave", hideTooltip);
    root.append(hit);
  });
  return el("div", {}, root);
}

route();
poll();

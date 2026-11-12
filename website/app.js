/* MENA Risk Monitor - reads one data bundle (build_site_data.py). The map is the
   full-screen background (#stage); dark glass panels float above it.
   Routes: #/ overview, #/c/ISO country, #/about data & model. EN / 中文.
   Anything built on the transformer forecast (forecasts, severity, risk) carries
   the glowing azurite-violet "fc" marker; observed data stays plain. */
import { createMap3D, webglAvailable } from "./map3d.js";

const CFG = window.SITE_CONFIG;
const S = { data: null, world: null, byNum: {}, byIso: {}, metric: "risk", list: "risk", lanesOn: true, acledOff: new Set(), map: null,
  lang: (() => { let v = null; try { v = localStorage.getItem("lang"); } catch { /* private mode */ } return v || ((navigator.language || "").startsWith("zh") ? "zh" : "en"); })() };
const $ = (sel, el = document) => el.querySelector(sel);
const tip = $("#tip");
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const C = {};
const USE_3D = webglAvailable() && !new URLSearchParams(location.search).has("2d");
const narrow = () => window.matchMedia("(max-width: 900px)").matches;

const fmt0 = d3.format(".0f"), fmt1 = d3.format(".1f"), fmt2 = d3.format("+.2f"), fmtInt = d3.format(",d"), pct = d3.format(".0%");
const spct = (v) => `${v >= 0 ? "+" : ""}${pct(v)}`;
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ------------------------------------------------------------ language
const T = {
  en: {
    title: "MENA Risk Monitor", asof: (w) => `Forecasts as of the week of ${w}`, updated: (d) => `Updated ${d}`, snapshot: (d) => `Snapshot ${d}`, about: "Data & model",
    export: "Export", fcTag: "Forecast-based", next13: "next 13 weeks", overview: "← Overview", of20: (r) => `#${r} of 20`,
    bands: { Severe: "Severe", High: "High", Elevated: "Elevated", Low: "Low" }, risk: "risk",
    regional: "Regional risk", sinceLast: "since the last refresh", alertsAhead: "surge alerts ahead", deaths4w: "conflict deaths, last 4 weeks",
    thisRefresh: "This refresh", watchlist: "Watchlist", tabRisk: "Highest risk", tabWorse: "Deteriorating", tabLanes: "Shipping lanes",
    mRisk: "Risk", mSev: "Severity", lower: "lower", higher: "higher risk", calmer: "calmer", severe: "more severe", relations: "hostility", lanes: "lanes",
    aiYes: "AI brief, refreshed with the data", aiNo: "Data summary (AI brief not configured)",
    tipFc: "◆ Forecast-based · next 13 weeks", tipObs: "Observed · last 4 weeks", tRisk: "Risk", tRel: "Relations vs normal", tAlerts: "Surge alerts", tDeaths: "ACLED conflict deaths",
    riskCard: "Risk", severity: "Severity", exposure: "Trade at stake", builtFrom: "Severity is built from",
    pHost: "Forecast hostility", pSurge: "Forecast surge risk", pViol: "Recent violence (observed)", brief: "Brief",
    outlook: "Outlook", pastYear: "past year observed", forecast: "Forecast", vsNormal: "vs normal", surgeAlerts: "Surge alerts",
    rel: "Relationships", partner: "Partner", evwk: "Ev/wk", tone: "◆ Tone", vsN: "◆ vs normal", context: "Context",
    trade12: "China trade, 12 months", yoy: "y/y", deaths: "conflict deaths, 4 weeks", mostly: (t) => `mostly ${t}`,
    acledLegend: "ACLED events, last 28 days · height = deaths", acledNone: "ACLED overlay unavailable",
    status: { disrupted: "disrupted", reduced: "reduced", normal: "normal" }, transits: "ships / week", vsNormalLane: "vs normal",
    laneLittoral: "States on its shores", laneStake: "China trade with partners that depend on it", laneTankers: "tankers / week",
    exportItems: [["countries", "Country risk scores", "CSV · one row per country"], ["dyads", "Relationship forecasts", "CSV · all 400 country pairs"],
      ["lanes", "Shipping lanes", "CSV · transits and status"], ["events", "ACLED events shown", "CSV · last 28 days"], ["json", "Everything", "JSON · the full data bundle"]],
    footer: "Sources: GDELT 2.0 · ACLED · IMF IMTS · IMF PortWatch. Forecasts are model estimates for horizon scanning, not predictions of specific events.",
    meaning: "What this means:", measured: "How it is measured:",
    chinaCard: "China exposure", atRisk: "What could be at risk for Chinese entities", sources: "Sources",
  },
  zh: {
    title: "中东北非风险监测", asof: (w) => `预测基准周：${w}`, updated: (d) => `更新于 ${d}`, snapshot: (d) => `快照 ${d}`, about: "数据与模型",
    export: "导出", fcTag: "基于预测", next13: "未来13周", overview: "← 总览", of20: (r) => `20国中第${r}位`,
    bands: { Severe: "严重", High: "高", Elevated: "偏高", Low: "低" }, risk: "风险",
    regional: "区域风险", sinceLast: "较上期", alertsAhead: "项冲突激增预警", deaths4w: "过去四周冲突死亡人数",
    thisRefresh: "本期概要", watchlist: "关注列表", tabRisk: "风险最高", tabWorse: "恶化最快", tabLanes: "航运通道",
    mRisk: "风险", mSev: "严重度", lower: "较低", higher: "较高风险", calmer: "平稳", severe: "更严重", relations: "敌对", lanes: "航道",
    aiYes: "AI 简报，随数据更新", aiNo: "数据摘要（AI 简报未启用）",
    tipFc: "◆ 基于预测 · 未来13周", tipObs: "实际观测 · 过去四周", tRisk: "风险", tRel: "关系较常态", tAlerts: "激增预警", tDeaths: "ACLED 冲突死亡",
    riskCard: "风险", severity: "严重度", exposure: "涉及贸易", builtFrom: "严重度的构成",
    pHost: "预测敌对程度", pSurge: "预测激增风险", pViol: "近期暴力（观测）", brief: "简报",
    outlook: "前景", pastYear: "过去一年（观测）", forecast: "预测", vsNormal: "常态", surgeAlerts: "激增预警",
    rel: "双边关系", partner: "对象国", evwk: "周事件", tone: "◆ 关系预测", vsN: "◆ 较常态", context: "背景",
    trade12: "对华贸易（12个月）", yoy: "同比", deaths: "四周冲突死亡", mostly: (t) => `主要为${t}`,
    acledLegend: "ACLED 事件（近28天）· 高度 = 死亡人数", acledNone: "ACLED 图层不可用",
    status: { disrupted: "严重受阻", reduced: "通行减少", normal: "正常" }, transits: "艘/周", vsNormalLane: "较常态",
    laneLittoral: "沿岸国家", laneStake: "依赖该通道伙伴的对华贸易", laneTankers: "油轮/周",
    exportItems: [["countries", "国家风险评分", "CSV · 每国一行"], ["dyads", "双边关系预测", "CSV · 全部400组国家对"],
      ["lanes", "航运通道", "CSV · 通行量与状态"], ["events", "ACLED 事件", "CSV · 近28天"], ["json", "全部数据", "JSON · 完整数据包"]],
    footer: "数据来源：GDELT 2.0 · ACLED · IMF IMTS · IMF PortWatch。预测为用于前瞻研判的模型估计，并非对具体事件的预言。",
    meaning: "含义：", measured: "计算方法：",
    chinaCard: "对华关联", atRisk: "中国主体可能面临的风险", sources: "来源",
  },
};
const t = (k) => T[S.lang][k];
const nm = (c) => (S.lang === "zh" ? c.name_zh || c.name : c.name);
const nmIso = (iso) => nm(S.byIso[iso]).replace("United Arab Emirates", "UAE");

// hints: [what this means, how it is measured]
const H = {
  en: {
    region: ["How much risk the region as a whole faces over the next quarter: higher means more countries combine serious conflict with significant trade at stake.", "The average of the 20 country risk scores (0–100). Bands: severe ≥ 55, high ≥ 45, elevated ≥ 32."],
    map: ["Where risk is concentrated. Darker red countries need more attention.", "Colour shows the measure chosen in this bar. Hover a country to lift it and see its relationships; click to open it. Drag to rotate, scroll to zoom."],
    threads: ["The hovered country's troubled relationships: orange threads are heading for hostility (deeper orange = more hostile). Calm, cooperative relationships are not drawn – unless they carry a surge alert, when they appear in white. A glowing ball travelling along a thread warns of a possible surge in conflict – yellow for a moderate risk, red for a sharp one.", "Colour: the transformer's forecast tone for the pair over the next 13 weeks. Thickness: how often the two interact. Ball colour: the forecast surge probability (about 50% yellow, 60% orange, 75%+ red)."],
    lanesMap: ["Sea lanes through which the region's trade moves, and whether ships are getting through. A red, pulsing ring means traffic has collapsed.", "Ships per week through each chokepoint over the last four weeks compared with the 2019–2023 normal (IMF PortWatch). Dashed: the main Europe–Gulf route."],
    brief: ["The main points of this refresh in plain words.", "Written from this refresh's numbers and regenerated with every data refresh."],
    watch: ["Where to look first. Highest risk: countries with the most at stake. Deteriorating: where relations are expected to worsen most. Shipping lanes: chokepoints whose traffic is disrupted.", "Risk scores, the forecast change from each country's normal, and PortWatch transits."],
    gauge: ["How exposed an organisation trading with this country is over the next quarter: it needs both serious conflict and significant trade at stake to score high.", "Risk (0–100) = √(severity × trade at stake)."],
    sev: ["How bad the conflict environment is expected to be, regardless of trade.", "0.40 × forecast hostility + 0.30 × forecast surge risk + 0.30 × recent violence, each scaled 0–1."],
    exp: ["The amount of trade value threatened if this country's situation deteriorates.", "China's trade (exports + imports) with the country over the last 12 months (IMF), on a log scale relative to the largest partner in the region."],
    hostility: ["How hostile the country's relations are expected to be over the next quarter.", "Transformer forecast of the activity-weighted Goldstein score of all its relationships; +1.5 → 0%, −4.0 → 100%."],
    surge: ["How likely the country is to see a sudden spike in conflict with some partner over the next quarter.", "Transformer forecast: the average probability that a relationship's worst month is unusually intense by its own standards; 60% or more → 100%."],
    violence: ["How much deadly violence is already happening.", "ACLED-recorded deaths in the last four weeks; log scale, 2,000 or more → 100%."],
    outlook: ["Whether relations are expected to get better or worse than usual over the next quarter.", "Black: the last year (observed, −10 hostile to +10 cooperative). Blue-violet: the transformer forecast for the next 13 weeks. Dashed: the 52-week normal."],
    alerts: ["Partners with which a sharp surge in conflict is possible next quarter.", "Pairs flagged by the transformer's conflict-surge model, with the probability that the coming worst month is unusually intense."],
    rel: ["Which relationships matter most for this country and where each is heading.", "Ev/wk: average events per week (observed). Tone: forecast Goldstein score for the next 13 weeks (−10 hostile, +10 cooperative). vs normal: forecast minus the pair's 52-week average."],
    ctxTrade: ["The amount of trade value threatened if relations deteriorate.", "China's goods trade with the country over the last 12 months published by the IMF, and its change on the year before."],
    ctxDeaths: ["How much deadly violence is already under way on the ground.", "ACLED-recorded deaths in the last four weeks and the most common type of event."],
    cmap: ["Where violence is happening inside the country right now.", "Pillars mark ACLED events in the last 28 days: colour is the event type, height is the number of deaths."],
    fc: ["This figure comes from, or is built on, our transformer forecast – the model's own view of the next 13 weeks.", "Release v2.2: a Goldstein forecast and a conflict-surge forecast. Risk and severity include them. Plain figures are observed data."],
    china: ["How this country's situation could affect Chinese companies and organisations: the trade relationship with China and what could be at risk.", "Written by AI (DeepSeek) at each refresh from IMF trade data, the forecasts on this page and recent Guardian / New York Times coverage of the country and China (listed under Sources). Specific projects or deals are named only when a cited article reports them."],
    lanes: ["Which sea lanes are disrupted and how much trade depends on them.", "Ships per week over the last four weeks vs the 2019–2023 normal (IMF PortWatch); status: disrupted ≤ −50%, reduced ≤ −20%."],
  },
  zh: {
    region: ["整个地区在下一季度面临的总体风险：数值越高，同时存在严重冲突与大量贸易利益的国家越多。", "20国风险评分（0–100）的平均值。等级：严重 ≥ 55，高 ≥ 45，偏高 ≥ 32。"],
    map: ["风险集中在哪里。越红的国家越需要关注。", "颜色表示本栏所选指标。悬停国家可抬起并显示其双边关系，点击进入详情。拖动旋转，滚轮缩放。"],
    threads: ["所悬停国家存在隐患的双边关系：橙色线表示趋向敌对（越深越敌对）。平稳、合作的关系不显示——除非触发激增预警，此时以白色显示。沿线移动的发光球提示可能出现冲突激增——黄色为中度风险，红色为急剧升级。", "颜色：Transformer 对未来13周双边关系的预测；粗细：双方互动频率；光球颜色：预测激增概率（约50%为黄，60%为橙，75%以上为红）。"],
    lanesMap: ["本地区贸易经过的海上通道，以及船只能否通行。红色脉冲环表示通行量已大幅下降。", "过去四周各咽喉要道每周通行船只数与2019–2023年常态的比较（IMF PortWatch）。虚线：欧洲—海湾主航线。"],
    brief: ["用通俗语言概括本期要点。", "根据本期数据生成，并随每次数据更新重新生成。"],
    watch: ["优先关注的对象。风险最高：利害最大的国家；恶化最快：关系预计恶化最明显的国家；航运通道：通行受阻的咽喉要道。", "风险评分、相对常态的预测变化，以及 PortWatch 通行量。"],
    gauge: ["与该国开展贸易的机构在下一季度面临的风险：只有冲突严重且涉及贸易规模大，得分才会高。", "风险（0–100）= √（严重度 × 涉及贸易）。"],
    sev: ["预计冲突环境的严重程度，不考虑贸易。", "0.40 × 预测敌对程度 + 0.30 × 预测激增风险 + 0.30 × 近期暴力，各项折算为0–1。"],
    exp: ["若该国局势恶化，可能受威胁的贸易价值。", "过去12个月中国与该国的贸易额（进口+出口，IMF），按对数相对于本地区最大贸易伙伴折算。"],
    hostility: ["预计下一季度该国对外关系的敌对程度。", "Transformer 对其全部双边关系（按活跃度加权）戈德斯坦值的预测；+1.5 → 0%，−4.0 → 100%。"],
    surge: ["下一季度该国与某一伙伴之间冲突突然升级的可能性。", "Transformer 预测：各组关系最糟月份按其自身标准异常激烈的平均概率；60%及以上 → 100%。"],
    violence: ["当前已在发生的致命暴力的程度。", "ACLED 记录的过去四周死亡人数；对数尺度，2,000人及以上 → 100%。"],
    outlook: ["下一季度对外关系预计比平常更好还是更差。", "黑线：过去一年（观测，−10敌对至+10合作）；蓝紫：Transformer 对未来13周的预测；虚线：52周常态。"],
    alerts: ["下一季度可能出现冲突急剧升级的对象国。", "被 Transformer 冲突激增模型标记的国家对，及其未来最糟月份异常激烈的概率。"],
    rel: ["对该国最重要的双边关系及其走向。", "周事件：每周平均事件数（观测）；关系预测：未来13周戈德斯坦值预测（−10敌对，+10合作）；较常态：预测值减去52周平均。"],
    ctxTrade: ["若关系恶化，可能受威胁的贸易价值。", "IMF 公布的过去12个月中国与该国的货物贸易额及同比变化。"],
    ctxDeaths: ["当地正在发生的致命暴力程度。", "ACLED 记录的过去四周死亡人数及最常见的事件类型。"],
    cmap: ["该国境内当前暴力事件的分布。", "柱体表示近28天 ACLED 事件：颜色为事件类型，高度为死亡人数。"],
    fc: ["该数值来自或基于我们的 Transformer 预测——即模型对未来13周的判断。", "v2.2 版本：戈德斯坦关系预测与冲突激增预测；风险与严重度均包含二者。普通数值为观测数据。"],
    china: ["该国局势可能如何影响中国企业和机构：与中国的贸易关系及可能面临的风险。", "由 AI（DeepSeek）在每次更新时根据 IMF 贸易数据、本页预测以及《卫报》《纽约时报》近期有关该国与中国的报道撰写（见“来源”）。具体项目或交易仅在被引用文章提及时才会列出。"],
    lanes: ["哪些海上通道受阻，以及有多少贸易依赖它们。", "过去四周每周通行船只数与2019–2023年常态之比（IMF PortWatch）；状态：严重受阻 ≤ −50%，通行减少 ≤ −20%。"],
  },
};
const hintText = (k) => H[S.lang][k];
const hint = (k) => `<span class="i" tabindex="0" data-hk="${k}">i</span>`;
const fc = (html) => `<span class="fc">${html}</span>`;
const fcTag = (label) => `<span class="fc-tag" tabindex="0" data-hk="fc">${esc(label || t("fcTag"))}</span>`;
const band = (risk) => [[55, "Severe", "b4"], [45, "High", "b3"], [32, "Elevated", "b2"], [0, "Low", "b1"]].find(([x]) => risk >= x);  // lamps on the red scale
const bandName = (risk) => t("bands")[band(risk)[1]];
const lamp = (cls, pulse) => `<span class="lamp ${cls}${pulse ? " pulse" : ""}"></span>`;

const ACLED_COLORS = { "Battles": "#b8262b", "Explosions/Remote violence": "#c8692b", "Violence against civilians": "#7a4a8a", "Protests": "#2f5d8a", "Riots": "#2f8f7d", "Strategic developments": "#6c6e72" };
const ACLED_ZH = { "Battles": "交战", "Explosions/Remote violence": "爆炸/远程暴力", "Violence against civilians": "针对平民的暴力", "Protests": "抗议", "Riots": "骚乱", "Strategic developments": "战略事态" };
const shortType = (t0) => (S.lang === "zh" ? ACLED_ZH[t0] : t0.replace("Explosions/Remote violence", "Explosions").replace("Violence against civilians", "Civilians").replace("Strategic developments", "Strategic"));
const LANE_COLORS = { disrupted: "#e0443a", reduced: "#e08a2c", normal: "#2f8f7d" };
const laneName = (l) => (S.lang === "zh" ? l.name_zh : l.name);

const METRICS = {
  risk: { label: () => t("mRisk"), get: (c) => c.risk, t: (v) => (v - 15) / 50, legend: () => [t("lower"), t("higher")] },
  severity: { label: () => t("mSev"), get: (c) => c.severity, t: (v) => (v - 5) / 70, legend: () => [t("calmer"), t("severe")] },
};

function scales() {
  // single red scale: the higher, the redder; the least severe a pale grey-red
  C.ramp = d3.interpolateRgbBasis(["#e4e0df", "#e8b5ae", "#d6675d", "#b8262b", "#7c1218"]);
  // thread tones: white for cooperative / neutral pairs, orange only for hostile ones (deeper = more hostile)
  C.tone = (g) => (g >= HOSTILE ? "#ffffff" : d3.interpolateRgbBasis(["#ffb36b", "#ff8a2a", "#e8500f", "#c23a08"])(Math.min(1, (-g - 0.25) / 3.75)));
  // surge sharpness: yellow (~50%) -> orange (~60%) -> red (75%+)
  C.surge = (p) => d3.interpolateRgbBasis(["#ffd84a", "#ff8a1e", "#ff2b1f"])(Math.max(0, Math.min(1, (p - 0.45) / 0.32)));
  const vmax = d3.max(S.data.threads, (x) => x.volume) || 1;
  C.width = d3.scaleSqrt().domain([0, vmax]).range([1, 9]);
  C.width3d = d3.scaleSqrt().domain([0, vmax]).range([0.09, 0.62]);
}
const metricT = (c, m = S.metric) => Math.max(0, Math.min(1, METRICS[m].t(METRICS[m].get(c))));
const fillFor = (c, m = S.metric) => C.ramp(metricT(c, m));
const riskColor = (v) => C.ramp(Math.max(0, Math.min(1, (v - 15) / 50)));
const fracColor = (f) => C.ramp(Math.max(0, Math.min(1, f)));
const topByMetric = (n) => [...S.data.countries].sort((a, b) => metricT(b) - metricT(a)).slice(0, n).map((c) => c.iso3);

// ------------------------------------------------------------ boot
const getJson = (url) => fetch(url, { cache: "no-cache" }).then((r) => { if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); });
// live bundle first; the committed copy if it is unreachable (or when previewing locally: ?local)
async function loadData() {
  const local = /localhost|127\.0\.0\.1/.test(location.hostname) || new URLSearchParams(location.search).has("local");
  if (!local || !CFG.fallbackUrl) {
    try { return await getJson(CFG.dataUrl); } catch (e) { if (!CFG.fallbackUrl) throw e; console.warn("live data unavailable, using bundled copy", e); }
  }
  return getJson(CFG.fallbackUrl);
}

async function init() {
  try {
    const [data, world] = await Promise.all([
      loadData(),
      fetch(CFG.mapUrl).then((r) => r.json()),
    ]);
    S.data = data;
    S.world = topojson.feature(world, world.objects.countries);
    S.world.features.forEach((f) => { S.byNum[f.id] = f; });
    data.countries.forEach((c) => { S.byIso[c.iso3] = c; const f = S.byNum[c.num]; if (f) f.properties.iso = c.iso3; });
    scales();
    if (document.fonts?.ready) await document.fonts.ready;
    chrome();
    hintListeners();
    window.addEventListener("hashchange", route);
    route();
  } catch (e) {
    console.error(e);
    $("#view").innerHTML = `<div class="loading">Could not load the data bundle (${esc(e.message)}).</div>`;
  }
}

// top bar, footer, language switch, export menu
function chrome() {
  const m = S.data.meta;
  document.documentElement.lang = S.lang === "zh" ? "zh-CN" : "en";
  document.title = t("title");
  $("#brand-title").textContent = t("title");
  $("#sub").textContent = t("asof")(m.as_of_week);
  const gen = new Date(m.generated_utc);
  const stale = (Date.now() - gen.getTime()) / 864e5 > 16;
  $("#status").innerHTML = CFG.frozen
    ? `${lamp("orange")}${t("snapshot")(gen.toISOString().slice(0, 10))}`
    : `${lamp(stale ? "orange" : "ok", !stale)}${t("updated")(gen.toISOString().slice(0, 10))}`;
  pausedNotice(gen);
  $("#fclegend").innerHTML = fcTag();
  $("#nav-about").textContent = t("about");
  $("#export-btn").textContent = t("export");
  $("#export-drop").innerHTML = t("exportItems").map(([k, l, s]) => `<button data-x="${k}">${esc(l)}<small>${esc(s)}</small></button>`).join("");
  $("#footer").textContent = t("footer");
  document.querySelectorAll("#lang button").forEach((b) => b.classList.toggle("on", b.dataset.l === S.lang));
  if (!chrome.wired) {
    chrome.wired = true;
    $(".brand").addEventListener("click", () => { location.hash = "#/"; });
    document.querySelectorAll("#lang button").forEach((b) => b.addEventListener("click", () => {
      S.lang = b.dataset.l;
      try { localStorage.setItem("lang", S.lang); } catch { /* private mode */ }
      chrome(); route();
    }));
    $("#export-btn").addEventListener("click", (ev) => { ev.stopPropagation(); $("#export").classList.toggle("open"); });
    document.addEventListener("click", () => $("#export").classList.remove("open"));
    $("#export-drop").addEventListener("click", (ev) => { const b = ev.target.closest("button"); if (b) exportData(b.dataset.x); });
  }
}

// "refresh paused" alert in the top bar (CFG.frozen): message and the developers' name cards
const ALERT_ICON = `<svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true"><path d="M12 3.2 22 20.5H2L12 3.2z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M12 9.5v5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="12" cy="17.4" r="1.2" fill="currentColor"/></svg>`;
const LINKEDIN_ICON = `<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><rect width="24" height="24" rx="4" fill="#0a66c2"/><path fill="#fff" d="M7.1 9.6h-2.6v8.3h2.6V9.6zM5.8 5.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zM19.5 13.4c0-2.3-1.2-4-3.6-4-1.2 0-2 .6-2.4 1.3V9.6h-2.5v8.3h2.6v-4.3c0-1.1.3-2.1 1.6-2.1s1.6 1.2 1.6 2.2v4.2h2.7v-4.5z"/></svg>`;

function pausedNotice(gen) {
  const el = $("#paused");
  if (!CFG.frozen) { el.hidden = true; return; }
  el.hidden = false;
  const zh = S.lang === "zh";
  const day = zh ? `${gen.getUTCFullYear()}年${gen.getUTCMonth() + 1}月${gen.getUTCDate()}日`
    : gen.toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" });
  const people = CFG.contacts || [];
  const mail = (p) => `<a href="mailto:${esc(p.email)}">${esc(p.email)}</a>`;
  const names = people.map((p) => `<b>${esc(p.name)}</b> ${zh ? "（" : "at "}${mail(p)}${zh ? "）" : ""}`).join(zh ? "与" : " and ");
  const msg = zh
    ? `本原型的数据更新已由开发者暂停，原因是本阶段开发目标已顺利完成。您看到的是截至${day}的快照。如需了解项目详情，欢迎联系${names}，期待与您交流！`
    : `The refresh of this prototype has been paused by the developers due to successful completion of the objectives of this development stage - you are seeing a snapshot as of ${day}. For any inquiries on the details of this project, please do not hesitate to contact ${names} - we are happy to connect!`;
  const initials = (n) => n.split(/\s+/).map((w) => w[0]).join("").slice(0, 2).toUpperCase();
  const card = (p) => `<a class="ncard" href="${esc(p.linkedin)}" target="_blank" rel="noopener">
      <span class="av"><span>${esc(initials(p.name))}</span>${p.photo ? `<img src="${esc(p.photo)}" alt="" onerror="this.remove()">` : ""}</span>
      <span class="who"><b>${esc(p.name)}</b><small>${esc(p.email)}</small></span>
      <span class="li">${LINKEDIN_ICON}LinkedIn</span></a>`;
  el.innerHTML = `<button class="pill alert-btn" type="button" aria-label="${zh ? "更新已暂停" : "Refresh paused"}">${ALERT_ICON}<span>${zh ? "已暂停更新" : "Refresh paused"}</span></button>
    <div class="paused-pop glass" role="dialog">
      <div class="pp-h">${ALERT_ICON}${zh ? "数据更新已暂停" : "Refresh paused"}</div>
      <p>${msg}</p>
      <div class="ncards">${people.map(card).join("")}</div>
    </div>`;
  if (!pausedNotice.wired) {  // tap to open on touch screens; click elsewhere closes
    pausedNotice.wired = true;
    el.addEventListener("click", (ev) => { if (ev.target.closest(".alert-btn")) { ev.stopPropagation(); el.classList.toggle("open"); } });
    document.addEventListener("click", (ev) => { if (!el.contains(ev.target)) el.classList.remove("open"); });
    if (new URLSearchParams(location.search).has("alert")) el.classList.add("open");  // preview helper
  }
}

function hintListeners() {
  const html = (el) => {
    if (el.dataset.hk) { const [a, b] = hintText(el.dataset.hk); return `<div class="hint-text"><b>${t("meaning")}</b> ${esc(a)}<br><span style="color:var(--muted)"><b>${t("measured")}</b> ${esc(b)}</span></div>`; }
    return `<div class="hint-text">${esc(el.dataset.hint)}</div>`;
  };
  const sel = "[data-hk], [data-hint]";
  const show = (ev) => { const el = ev.target.closest?.(sel); if (el) showTip(html(el), ev.clientX ? ev : rectEvent(el)); };
  document.addEventListener("mouseover", (ev) => { if (ev.target.closest?.(sel)) show(ev); });
  document.addEventListener("mousemove", (ev) => { if (ev.target.closest?.(sel)) moveTip(ev); });
  document.addEventListener("mouseout", (ev) => { if (ev.target.closest?.(sel)) hideTip(); });
  document.addEventListener("focusin", show);
  document.addEventListener("focusout", hideTip);
}
const rectEvent = (el) => { const r = el.getBoundingClientRect(); return { clientX: r.right, clientY: r.bottom }; };

// floating columns start one gap below the top bar's actual bottom edge
function layout() {
  const gap = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--gap")) || 14;
  document.documentElement.style.setProperty("--top", `${Math.round($(".topbar").getBoundingClientRect().bottom + gap)}px`);
}
window.addEventListener("resize", layout);

function route() {
  hideTip();
  layout();
  if (S.map) { S.map.destroy(); S.map = null; }
  $("#stage").innerHTML = "";
  $("#stage").classList.remove("dim");
  S.threadG = null;
  const m = location.hash.match(/^#\/c\/([A-Z]{3})/);
  const about = location.hash.startsWith("#/about");
  $("#nav-about").classList.toggle("on", about);
  $("#view").classList.toggle("withmap", !about);
  $("#footer").style.display = about ? "none" : "";
  if (about) renderAbout();
  else if (m && S.byIso[m[1]]) renderCountry(m[1]);
  else renderOverview();
  window.scrollTo({ top: 0 });
}

function inset() {
  if (narrow()) return { left: 0, right: 0, top: $(".topbar").getBoundingClientRect().bottom, bottom: 0 };
  const L = $(".float.left > *")?.getBoundingClientRect(), R = $(".float.right > *")?.getBoundingClientRect();  // the cards, not the padded columns
  const Tp = $(".crumbs")?.getBoundingClientRect() || $(".topbar").getBoundingClientRect();
  const B = $(".map-ui")?.getBoundingClientRect();
  return { left: L ? L.right + 10 : 0, right: R ? innerWidth - R.left + 10 : 0, top: Tp.bottom + 10, bottom: B ? innerHeight - B.top + 10 : 0 };
}

// ------------------------------------------------------------ tooltip
function showTip(html, ev) { tip.innerHTML = html; tip.classList.add("show"); moveTip(ev); }
function moveTip(ev) {
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
  if (y + h > innerHeight - 8) y = ev.clientY - h - pad;
  tip.style.left = `${Math.max(8, x)}px`; tip.style.top = `${Math.max(8, y)}px`;
}
function hideTip() { tip.classList.remove("show"); }

const countryTip = (c) => `<div class="t">${lamp(band(c.risk)[2])}${esc(nm(c))}</div>
  <div class="cap f">${t("tipFc")}</div>
  <div class="r">${t("tRisk")} <b>${fc(fmt0(c.risk))} · ${bandName(c.risk)}</b></div>
  <div class="r">${t("tRel")} <b>${fc(fmt2(c.change))}</b></div>
  <div class="r">${t("tAlerts")} <b>${fc(c.alerts)}</b></div>
  ${c.acled_fatalities_4w != null ? `<div class="sep"></div><div class="cap">${t("tipObs")}</div><div class="r">${t("tDeaths")} <b>${fmtInt(c.acled_fatalities_4w)}</b></div>` : ""}`;

const laneTip = (l) => `<div class="t"><span class="lamp" style="background:${LANE_COLORS[l.status]}"></span>${esc(laneName(l))} · ${t("status")[l.status]}</div>
  <div class="cap">${S.lang === "zh" ? "观测 · 过去四周" : "Observed · last 4 weeks"}</div>
  <div class="r">${t("transits")} <b>${fmt0(l.transits_week)} (${spct(l.change)} ${t("vsNormalLane")})</b></div>
  <div class="r">${t("laneTankers")} <b>${fmt1(l.tankers_week)} / ${fmt0(l.normal_tankers)}</b></div>
  ${l.china_trade_at_stake_bn ? `<div class="r">${t("laneStake")} <b>$${fmt0(l.china_trade_at_stake_bn)}bn</b></div>` : ""}
  ${l.littoral.length ? `<div class="sep"></div><div class="cap f">${t("laneLittoral")} · ◆ ${t("tRisk")}</div>${l.littoral.map((x) => `<div class="r">${esc(nmIso(x.iso3))} <b>${fc(fmt0(x.risk))}</b></div>`).join("")}` : ""}`;

// ------------------------------------------------------------ overview
function renderOverview() {
  const D = S.data, cs = D.countries, R = D.region;
  const fat = d3.sum(cs, (c) => c.acled_fatalities_4w || 0);
  const lanesBad = (D.lanes || []).filter((l) => l.on_map && l.status !== "normal").length;
  $("#view").innerHTML = `
    <aside class="float left">
      <section class="card glass framed regional">
        <h2>${t("regional")} ${hint("region")}</h2>
        <div class="regional-body">
          <div id="rgauge"></div>
          <div>
            <div class="chip">${lamp(band(R.risk)[2], R.risk >= 45)}${bandName(R.risk)}</div>
            <div class="mini" style="margin-top:10px">${deltaHtml(R.risk - R.risk_prev, true)} ${t("sinceLast")}</div>
            <div class="mini" style="margin-top:8px">${fcTag()}</div>
          </div>
        </div>
        <div class="summary-line"><span><b>${R.alerts}</b> ${t("alertsAhead")}</span><span><b>${fmtInt(fat)}</b> ${t("deaths4w")}</span></div>
      </section>
      <section class="card glass framed">
        <h2>${t("thisRefresh")} ${hint("brief")}</h2>
        <div class="brief">${esc(summary("region"))}</div>
        <div class="prov-row">${provTag()}</div>
      </section>
    </aside>
    <aside class="float right">
      <section class="card glass framed">
        <h2>${t("watchlist")} ${hint("watch")}</h2>
        <div class="seg" id="listtab" style="margin-bottom:10px">
          <button data-k="risk">${t("tabRisk")}</button><button data-k="worse">${t("tabWorse")}</button>
          ${D.lanes ? `<button data-k="lanes">${t("tabLanes")}${lanesBad ? ` ${lamp("red", true)}` : ""}</button>` : ""}
        </div>
        <div id="watch"></div>
      </section>
    </aside>
    <div class="map-ui glass">
      <div class="seg" id="metric">${Object.entries(METRICS).map(([k, m]) => `<button data-k="${k}" class="${k === S.metric ? "on" : ""}">${m.label()}</button>`).join("")}</div>
      <div class="legend">
        <span class="sw"><span id="lg-lo">${METRICS[S.metric].legend()[0]}</span><span class="bar"></span><span id="lg-hi">${METRICS[S.metric].legend()[1]}</span> ${hint("map")}</span>
        <span class="sw">${t("relations")} <span class="thr"></span> ${S.lang === "zh" ? "激增" : "surge"} <span class="ball" style="background:#ffd84a;box-shadow:0 0 6px #ffd84a"></span><span class="ball" style="background:#ff8a1e;box-shadow:0 0 6px #ff8a1e"></span><span class="ball" style="background:#ff2b1f;box-shadow:0 0 6px #ff2b1f"></span> ${hint("threads")}</span>
        ${D.lanes ? `<button class="chip" id="lanes-toggle" style="cursor:pointer">${lamp(S.lanesOn ? "red" : "off")} ${t("lanes")}</button> ${hint("lanesMap")}` : ""}
      </div>
    </div>`;

  gauge("#rgauge", R.risk, 124);
  watchlist();
  document.querySelectorAll("#listtab button").forEach((b) => b.addEventListener("click", () => { S.list = b.dataset.k; watchlist(); }));
  document.querySelectorAll("#metric button").forEach((b) => b.addEventListener("click", () => {
    S.metric = b.dataset.k;
    document.querySelectorAll("#metric button").forEach((x) => x.classList.toggle("on", x === b));
    $("#lg-lo").textContent = METRICS[S.metric].legend()[0];
    $("#lg-hi").textContent = METRICS[S.metric].legend()[1];
    if (S.map) S.map.setStyle(); else d3.selectAll(".mena").attr("fill", (f) => fillFor(S.byIso[f.properties.iso]));
  }));
  $("#lanes-toggle")?.addEventListener("click", () => {
    S.lanesOn = !S.lanesOn;
    $("#lanes-toggle").innerHTML = `${lamp(S.lanesOn ? "red" : "off")} ${t("lanes")}`;
    if (S.map) S.map.setLanes(S.lanesOn); else d3.selectAll(".lane2d").attr("display", S.lanesOn ? null : "none");
  });
  if (USE_3D) regionMap3D(); else drawRegionMap();
  const focus = new URLSearchParams(location.search).get("focus");  // preview hook
  if (focus && S.byIso[focus]) { focusCountry(focus); const ins = inset(); showTip(countryTip(S.byIso[focus]), { clientX: ins.left + 30, clientY: ins.top + 10 }); }
}

function summary(key, iso) {
  const s = S.data.summaries, zh = S.lang === "zh";
  if (key === "region") return (zh && s.region_zh) || s.region;
  return (zh && s.countries_zh?.[iso]) || s.countries[iso] || "";
}

function watchlist() {
  document.querySelectorAll("#listtab button").forEach((x) => x.classList.toggle("on", x.dataset.k === S.list));
  const cs = S.data.countries;
  let html;
  if (S.list === "lanes") {
    html = `<table class="tbl"><tbody>${S.data.lanes.map((l) => `<tr data-lane="${l.id}"><td><span class="lane-dot" style="background:${LANE_COLORS[l.status]};box-shadow:0 0 8px ${LANE_COLORS[l.status]}"></span>${esc(laneName(l))}<br><small style="color:var(--muted)">${t("status")[l.status]}</small></td><td class="num">${fmt0(l.transits_week)}<br><small style="color:var(--muted)">${t("transits")}</small></td><td class="num">${spct(l.change)}</td></tr>`).join("")}</tbody></table>
      <div class="mini" style="margin-top:8px">${hint("lanes")} ${esc(S.data.meta.lanes?.source || "")}</div>`;
  } else {
    const rows = S.list === "risk"
      ? cs.slice(0, 6).map((c) => row(c, (c.risk - 15) / 50, fc(fmt0(c.risk)), riskColor(c.risk), deltaHtml(c.risk_change, true)))
      : [...cs].sort((a, b) => a.change - b.change).slice(0, 6).map((c) => row(c, Math.max(0, -c.change), fc(fmt2(c.change)), "", "", true));
    html = `<div class="list">${rows.join("")}</div>`;
  }
  $("#watch").innerHTML = html;
  document.querySelectorAll("#watch .row").forEach((r) => {
    r.addEventListener("click", () => { location.hash = `#/c/${r.dataset.iso}`; });
    r.addEventListener("mouseenter", () => focusCountry(r.dataset.iso));
    r.addEventListener("mouseleave", () => focusCountry(null));
  });
  document.querySelectorAll("#watch tr[data-lane]").forEach((r) => {
    const l = S.data.lanes.find((x) => x.id === r.dataset.lane);
    r.addEventListener("mousemove", (ev) => showTip(laneTip(l), ev));
    r.addEventListener("mouseleave", hideTip);
  });
}

function provTag() {
  const prov = S.data.summaries.provenance || {};
  return prov.source === "ai" ? `<span class="ai-tag">${lamp("ok")}${t("aiYes")}</span>` : `<span class="ai-tag">${lamp("off")}${t("aiNo")}</span>`;
}
function deltaHtml(d, upIsBad, suffix = "", digits = 1) {
  const f = d3.format(`.${digits}f`), sfx = suffix ? " " + suffix : "";
  if (d == null || Math.abs(d) < 0.5 * 10 ** -digits) return `<span class="delta">±${f(0)}${sfx}</span>`;
  const worse = upIsBad ? d > 0 : d < 0;
  return `<span class="delta ${worse ? "up" : "down"}">${d > 0 ? "▲" : "▼"} ${f(Math.abs(d))}${sfx}</span>`;
}
function row(c, frac, label, color, extra, isFc = false) {
  return `<div class="row" data-iso="${c.iso3}">
    <span class="nm">${lamp(band(c.risk)[2], c.risk >= 55)}${esc(nm(c))}</span>
    <span><div class="meter${isFc ? " fc-meter" : ""}"><i style="width:${Math.max(4, Math.min(100, frac * 100))}%;background:${color}"></i></div>${extra}</span>
    <span class="val">${label}</span></div>`;
}

const ANCHORS = { PSE: [35.25, 31.95], ISR: [34.9, 31.1] };
const HOSTILE = -0.25;  // forecast tone below this counts as heading for hostility (drawn orange)
const SMALL = ["BHR", "QAT", "PSE", "LBN", "KWT", "CYP", "DJI", "ISR"];
function landNear(bbox) {
  return S.world.features.filter((f) => {
    if (f.properties.iso) return false;
    const b = d3.geoBounds(f);
    return b[1][0] > bbox[0] && b[0][0] < bbox[2] && b[1][1] > bbox[1] && b[0][1] < bbox[3] && (b[1][0] - b[0][0]) < 180;
  });
}

function regionMap3D() {
  const mena = S.data.countries.map((c) => ({ feature: S.byNum[c.num], iso: c.iso3 })).filter((x) => x.feature);
  S.map = createMap3D($("#stage"), {
    menaFeatures: mena, landFeatures: landNear([-25, -15, 95, 60]),
    fit: { type: "FeatureCollection", features: mena.map((m) => m.feature) }, anchors: ANCHORS, inset,
    color: (iso) => fillFor(S.byIso[iso]),
    glow: (iso) => topByMetric(3).includes(iso),
    // calm map: draw only relationships heading for hostility, plus any pair with a surge alert
    // (a cooperative-looking pair appears, in white, only when it carries an alert)
    threads: (iso) => S.data.threads.filter((x) => (x.a === iso || x.b === iso) && x.volume > 0.05 && (x.alert || x.tone < HOSTILE))
      .map((x) => ({ other: x.a === iso ? x.b : x.a, color: C.tone(x.tone), width: C.width3d(x.volume), alert: x.alert, surge: x.surge })),
    surgeColor: (p) => C.surge(p),
    lanes: (S.data.lanes || []).filter((l) => l.on_map).map((l) => ({ ...l, color: LANE_COLORS[l.status], alert: l.status === "disrupted" })),
    onLaneHover: (l, ev) => { if (l) showTip(laneTip(l), ev); else hideTip(); },
    threadStyle: new URLSearchParams(location.search).get("threads") || "flow",
    labels: true, hideLabels: ["BHR", "QAT", "LBN", "KWT", "CYP", "DJI"], labelText: nmIso,
    onHover: (iso, ev) => { if (iso) showTip(countryTip(S.byIso[iso]), ev); else hideTip(); document.querySelectorAll(".row").forEach((r) => r.classList.toggle("hl", r.dataset.iso === iso)); },
    onClick: (iso) => { location.hash = `#/c/${iso}`; },
  });
  S.map.setLanes(S.lanesOn);
}

function drawRegionMap(host = null, dimmed = false) {  // 2D fallback; also the backdrop of the about page
  host = host || $("#stage").appendChild(Object.assign(document.createElement("div"), { className: "svgmap" }));
  const W = innerWidth, H2 = innerHeight, ins = dimmed ? { left: 0, right: 0, top: 60, bottom: 0 } : inset();
  const mena = S.data.countries.map((c) => S.byNum[c.num]).filter(Boolean);
  const proj = d3.geoMercator().fitExtent([[ins.left + 20, ins.top + 20], [W - ins.right - 20, H2 - ins.bottom - 20]], { type: "FeatureCollection", features: mena });
  const path = d3.geoPath(proj);
  S.centroid = {};
  mena.forEach((f) => { S.centroid[f.properties.iso] = proj(ANCHORS[f.properties.iso] || d3.geoCentroid(f)); });
  const svg = d3.select(host).append("svg").attr("viewBox", `0 0 ${W} ${H2}`);
  svg.append("g").selectAll("path").data(S.world.features).join("path").attr("d", path).attr("fill", "#d6d7d3").attr("stroke", "#ecedeb").attr("stroke-width", 0.6);
  svg.append("g").selectAll("path").data(mena).join("path")
    .attr("class", "mena").attr("d", path).attr("fill", (f) => fillFor(S.byIso[f.properties.iso])).attr("fill-opacity", 0.8).attr("stroke", "rgba(17,18,20,0.35)").attr("stroke-width", 0.7)
    .style("cursor", dimmed ? "default" : "pointer")
    .on("mouseenter", (ev, f) => { if (dimmed) return; focusCountry(f.properties.iso); showTip(countryTip(S.byIso[f.properties.iso]), ev); })
    .on("mousemove", (ev) => { if (!dimmed) moveTip(ev); }).on("mouseleave", () => { if (!dimmed) { focusCountry(null); hideTip(); } })
    .on("click", (ev, f) => { if (!dimmed) location.hash = `#/c/${f.properties.iso}`; });
  S.threadG = svg.append("g");
  if (!dimmed) for (const l of (S.data.lanes || []).filter((x) => x.on_map)) {
    const [x, y] = proj([l.lon, l.lat]);
    svg.append("circle").attr("class", "lane2d").attr("cx", x).attr("cy", y).attr("r", 9).attr("fill", "none").attr("stroke", LANE_COLORS[l.status]).attr("stroke-width", 3)
      .attr("display", S.lanesOn ? null : "none").style("cursor", "help").on("mousemove", (ev) => showTip(laneTip(l), ev)).on("mouseleave", hideTip);
  }
  svg.append("g").selectAll("text").data(mena.filter((f) => !SMALL.includes(f.properties.iso))).join("text")
    .attr("text-anchor", "middle").attr("font-size", 12).attr("fill", "#111214").attr("paint-order", "stroke").attr("stroke", "rgba(255,255,255,0.75)").attr("stroke-width", 3)
    .attr("x", (f) => S.centroid[f.properties.iso][0]).attr("y", (f) => S.centroid[f.properties.iso][1] + 4).style("pointer-events", "none")
    .text((f) => nmIso(f.properties.iso));
}

function focusCountry(iso) {
  document.querySelectorAll(".row").forEach((r) => r.classList.toggle("hl", r.dataset.iso === iso));
  if (S.map) { S.map.focus(iso); return; }
  const g = S.threadG;
  if (!g) return;
  if (!iso) { g.selectAll("*").remove(); return; }
  const th = S.data.threads.filter((x) => (x.a === iso || x.b === iso) && x.volume > 0.05 && (x.alert || x.tone < HOSTILE)).sort((a, b) => a.volume - b.volume);
  const p0 = S.centroid[iso];
  g.selectAll("path").data(th, (x) => x.a + x.b).join("path").attr("fill", "none").attr("stroke-linecap", "round").style("pointer-events", "none")
    .attr("d", (x) => {
      const p1 = S.centroid[x.a === iso ? x.b : x.a];
      const mx = (p0[0] + p1[0]) / 2, my = (p0[1] + p1[1]) / 2, dx = p1[0] - p0[0], dy = p1[1] - p0[1];
      return `M${p0[0]},${p0[1]} Q${mx - dy * 0.18},${my + dx * 0.18} ${p1[0]},${p1[1]}`;
    })
    .attr("stroke", (x) => C.tone(x.tone)).attr("stroke-width", (x) => C.width(x.volume));
}

// ------------------------------------------------------------ country page
function renderCountry(iso) {
  const D = S.data, c = S.byIso[iso];
  const pairs = D.threads.filter((x) => x.a === iso || x.b === iso).map((x) => ({ ...x, other: x.a === iso ? x.b : x.a })).sort((a, b) => b.volume - a.volume);
  const alerts = pairs.filter((p) => p.alert);
  const ac = c.acled;
  const topType = ac ? Object.entries(ac.by_type).sort((a, b) => b[1] - a[1])[0] : null;

  $("#view").innerHTML = `
    <div class="crumbs glass">
      <button class="btn" id="back">${t("overview")}</button>
      <h2 class="title-big">${esc(nm(c))}</h2>
      <span class="chip">${lamp(band(c.risk)[2], c.risk >= 55)}${bandName(c.risk)} ${t("risk")} · ${t("of20")(c.rank)}</span>
    </div>
    <aside class="float left">
      <section class="card glass framed">
        <h2>${t("riskCard")} ${hint("gauge")}</h2>
        <div class="score"><div id="gauge"></div><div class="parts">
          ${part(t("severity"), c.severity / 100, fc(fmt0(c.severity)), "", "sev", true)}
          ${part(t("exposure"), c.exposure / 100, fmt0(c.exposure), css("--neutral"), "exp")}
          <div class="mini">${deltaHtml(c.risk_change, true)} ${t("sinceLast")}</div>
        </div></div>
        <div class="parts" style="margin-top:14px;padding-top:12px;border-top:1px solid var(--hair)">
          <div class="mini" style="margin:0 0 2px">${t("builtFrom")}</div>
          ${part(t("pHost"), c.severity_parts.hostility, fc(pct(c.severity_parts.hostility)), "", "hostility", true)}
          ${part(t("pSurge"), c.severity_parts.surge, fc(pct(c.severity_parts.surge)), "", "surge", true)}
          ${c.severity_parts.violence != null ? part(t("pViol"), c.severity_parts.violence, pct(c.severity_parts.violence), fracColor(c.severity_parts.violence), "violence") : ""}
        </div>
      </section>
      <section class="card glass framed">
        <h2>${t("brief")} ${hint("brief")}</h2>
        <div class="brief">${esc(summary("country", iso))}</div>
        <div class="prov-row">${provTag()}</div>
      </section>
      ${chinaCard(iso)}
    </aside>
    <aside class="float right">
      <section class="card glass framed chart">
        <h2>${t("outlook")} ${hint("outlook")}</h2>
        <div class="sub2">${t("pastYear")} · ${fcTag(t("next13"))}</div>
        <div id="ch-gold"></div>
        <div class="mini" style="margin-top:8px">${t("forecast")} <b>${fc(fmt2(c.forecast))}</b>&nbsp; ${t("vsNormal")} <b>${fmt2(c.normal)}</b></div>
        ${alerts.length ? `<div class="mini" style="margin-top:8px">${lamp("red", true)} ${t("surgeAlerts")}: ${alerts.map((p) => `<b>${esc(nmIso(p.other))}</b> ${fc(pct(p.surge))}`).join(", ")} ${hint("alerts")}</div>` : ""}
      </section>
      <section class="card glass framed">
        <h2>${t("rel")} ${hint("rel")}</h2>
        <table class="tbl"><thead><tr><th>${t("partner")}</th><th class="num">${t("evwk")}</th><th class="num fch">${t("tone")}</th><th class="num fch">${t("vsN")}</th></tr></thead>
        <tbody>${pairs.slice(0, 6).map((p) => `<tr data-iso="${p.other}"><td>${p.alert ? lamp("red", true) + " " : ""}${esc(nmIso(p.other))}</td><td class="num">${fmt0(p.volume)}</td><td class="num">${fc(fmt2(p.tone))}</td><td class="num">${fc(fmt2(p.change))}</td></tr>`).join("")}</tbody></table>
      </section>
      <section class="card glass framed">
        <h2>${t("context")}</h2>
        <div class="ctx">
          <div><div class="big">$${fmt1(c.trade.total_bn)}bn ${hint("ctxTrade")}</div><div class="note">${t("trade12")}${c.trade.yoy != null ? ` · ${spct(c.trade.yoy)} ${t("yoy")}` : ""}</div></div>
          <div><div class="big">${ac ? fmtInt(c.acled_fatalities_4w) : "—"} ${hint("ctxDeaths")}</div><div class="note">${t("deaths")}${topType ? ` · ${esc(t("mostly")(shortType(topType[0]).toLowerCase()))}` : ""}</div></div>
        </div>
      </section>
    </aside>
    <div class="map-ui glass">
      ${ac ? `<div class="chips" id="acled-chips">${Object.keys(ACLED_COLORS).map((x) => `<button data-t="${esc(x)}" class="${S.acledOff.has(x) ? "off" : ""}"><span class="dot" style="background:${ACLED_COLORS[x]}"></span>${esc(shortType(x))}</button>`).join("")}</div>
      <div class="legend">${t("acledLegend")} ${hint("cmap")}</div>` : `<div class="legend">${t("acledNone")}</div>`}
    </div>`;

  $("#back").addEventListener("click", () => { location.hash = "#/"; });
  document.querySelectorAll("tr[data-iso]").forEach((r) => r.addEventListener("click", () => { location.hash = `#/c/${r.dataset.iso}`; }));
  document.querySelectorAll("#acled-chips button").forEach((b) => b.addEventListener("click", () => {
    const x = b.dataset.t;
    S.acledOff.has(x) ? S.acledOff.delete(x) : S.acledOff.add(x);
    b.classList.toggle("off");
    if (S.map) S.map.setEventFilter(S.acledOff);
    d3.selectAll(".event-dot").attr("display", (e) => (S.acledOff.has(e.t) ? "none" : null));
  }));
  gauge("#gauge", c.risk, 124);
  if (USE_3D) countryMap3D(c); else zoomMap(c);
  lineChart("#ch-gold", c.history.goldstein, { normal: c.normal, forecast: c.forecast });
}

function chinaCard(iso) {
  const d = S.data.summaries.detail?.[iso];
  if (!d) return "";
  const zh = S.lang === "zh";
  const items = zh ? d.at_risk_zh : d.at_risk;
  return `<section class="card glass framed">
    <h2>${t("chinaCard")} ${hint("china")}</h2>
    <div class="brief">${esc(zh ? d.china_zh : d.china)}</div>
    <div class="mini" style="margin-top:12px;color:var(--ink)"><b style="font-family:inherit">${t("atRisk")}</b></div>
    <ul class="risks">${items.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
    ${d.sources?.length ? `<div class="srcs"><span>${t("sources")}:</span> ${d.sources.map((a) => `<a href="${esc(a.url)}" target="_blank" rel="noopener noreferrer">${esc(a.source)}, ${esc(a.date)}</a>`).join(" · ")}</div>` : ""}
    <div class="prov-row">${provTag()}</div>
  </section>`;
}

const part = (label, frac, val, color, hk, isFc = false) => `<div class="part"><div class="lab"><span>${esc(label)} ${hk ? hint(hk) : ""}</span><b>${val}</b></div>
  <div class="meter${isFc ? " fc-meter" : ""}"><i style="width:${Math.max(2, Math.min(100, frac * 100))}%;background:${color}"></i></div></div>`;

function countryMap3D(c) {
  const f = S.byNum[c.num];
  const b = d3.geoBounds(f);
  const pad = Math.max(3, (b[1][0] - b[0][0]) * 0.45, (b[1][1] - b[0][1]) * 0.45);
  const box = { type: "MultiPoint", coordinates: [[b[0][0] - pad, b[0][1] - pad], [b[0][0] - pad, b[1][1] + pad], [b[1][0] + pad, b[1][1] + pad], [b[1][0] + pad, b[0][1] - pad]] };
  const near = S.data.countries.filter((x) => {
    const fb = d3.geoBounds(S.byNum[x.num]);
    return fb[1][0] > b[0][0] - pad * 3 && fb[0][0] < b[1][0] + pad * 3 && fb[1][1] > b[0][1] - pad * 3 && fb[0][1] < b[1][1] + pad * 3;
  });
  S.map = createMap3D($("#stage"), {
    menaFeatures: near.map((x) => ({ feature: S.byNum[x.num], iso: x.iso3 })),
    landFeatures: landNear([b[0][0] - pad * 6, b[0][1] - pad * 5, b[1][0] + pad * 6, b[1][1] + pad * 5]),
    fit: box, anchors: ANCHORS, focusIso: c.iso3, inset,
    color: (iso) => fillFor(S.byIso[iso], "risk"), glow: () => false,
    events: (c.acled?.events || []).map((e) => ({ ...e, color: ACLED_COLORS[e.t] || "#6c6e72" })),
    lanes: (S.data.lanes || []).filter((l) => l.on_map).map((l) => ({ ...l, color: LANE_COLORS[l.status], alert: l.status === "disrupted" })),
    onLaneHover: (l, ev) => { if (l) showTip(laneTip(l), ev); else hideTip(); },
    labels: true, labelText: nmIso,
    onHover: (iso, ev) => { if (iso && iso !== c.iso3) showTip(countryTip(S.byIso[iso]), ev); else if (!iso) hideTip(); },
    onClick: (iso) => { if (iso !== c.iso3) location.hash = `#/c/${iso}`; },
    onEventHover: (e, ev) => {
      if (!e) return;
      showTip(`<div class="t"><span class="lamp" style="background:${ACLED_COLORS[e.t]}"></span>${esc(e.s || shortType(e.t))}</div><div class="cap">ACLED</div>
        <div class="r">${esc(e.d)} <b>${esc(e.l)}</b></div><div class="r">${S.lang === "zh" ? "死亡" : "Deaths"} <b>${fmtInt(e.f)}</b></div>
        ${e.a ? `<div class="r">${S.lang === "zh" ? "行为体" : "Actor"} <b>${esc(e.a)}</b></div>` : ""}<div style="margin-top:4px;color:var(--ink-2)">${esc(e.n)}</div>`, ev);
    },
  });
  S.map.setEventFilter(S.acledOff);
  S.map.setLanes(S.lanesOn);
}

function zoomMap(c) {  // 2D fallback country map
  const host = $("#stage").appendChild(Object.assign(document.createElement("div"), { className: "svgmap" }));
  const W = innerWidth, H2 = innerHeight, ins = inset();
  const f = S.byNum[c.num];
  const svg = d3.select(host).append("svg").attr("viewBox", `0 0 ${W} ${H2}`);
  const fit = d3.geoMercator().fitExtent([[ins.left + 60, ins.top + 60], [W - ins.right - 60, H2 - ins.bottom - 60]], f);
  const proj = fit.scale(Math.min(fit.scale() * 0.75, 14000));
  const path = d3.geoPath(proj);
  svg.append("g").selectAll("path").data(S.world.features).join("path").attr("d", path).attr("fill", "#d6d7d3").attr("stroke", "#ecedeb").attr("stroke-width", 0.6);
  svg.append("path").datum(f).attr("d", path).attr("fill", fillFor(c, "risk")).attr("fill-opacity", 0.8).attr("stroke", "#111214").attr("stroke-width", 1.2);
  if (c.acled) {
    const rs = d3.scaleSqrt().domain([0, d3.max(c.acled.events, (e) => e.f) || 1]).range([2.6, 16]);
    svg.append("g").selectAll("circle").data([...c.acled.events].sort((a, b2) => b2.f - a.f)).join("circle")
      .attr("class", "event-dot").attr("cx", (e) => proj([e.lon, e.lat])[0]).attr("cy", (e) => proj([e.lon, e.lat])[1])
      .attr("r", (e) => rs(e.f)).attr("fill", (e) => ACLED_COLORS[e.t]).attr("opacity", 0.8).attr("stroke", "#fff").attr("stroke-width", 0.6)
      .attr("display", (e) => (S.acledOff.has(e.t) ? "none" : null));
  }
}

// ------------------------------------------------------------ export
function exportData(kind) {
  const D = S.data, m = D.meta, stamp = m.as_of_week;
  const csv = (rows) => {
    const cols = Object.keys(rows[0]);
    const q = (v) => (v == null ? "" : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
    return "﻿" + [cols.join(","), ...rows.map((r) => cols.map((k) => q(r[k])).join(","))].join("\n");  // BOM: Excel opens UTF-8 (Chinese names)
  };
  let name, body, type = "text/csv;charset=utf-8";
  if (kind === "countries") {
    name = `country_risk_${stamp}.csv`;
    body = csv(D.countries.map((c) => ({
      as_of_week: stamp, model: m.model, iso3: c.iso3, name: c.name, name_zh: c.name_zh, rank: c.rank,
      risk: c.risk, risk_previous_refresh: c.risk_prev, risk_change: c.risk_change, band: band(c.risk)[1],
      severity: c.severity, forecast_hostility: c.severity_parts.hostility, forecast_surge_risk: c.severity_parts.surge, recent_violence: c.severity_parts.violence,
      trade_exposure: c.exposure, china_trade_bn_12m: c.trade.total_bn, china_exports_bn: c.trade.exports_bn, china_imports_bn: c.trade.imports_bn, trade_yoy: c.trade.yoy,
      goldstein_forecast_13w: c.forecast, goldstein_normal_52w: c.normal, forecast_vs_normal: c.change, surge_alerts: c.alerts,
      acled_deaths_4w: c.acled_fatalities_4w, hormuz_dependence_low: c.trade.hormuz_low, hormuz_dependence_high: c.trade.hormuz_high,
    })));
  } else if (kind === "dyads") {
    name = `relationship_forecasts_${stamp}.csv`;
    body = csv(D.dyads.map((d) => ({ as_of_week: stamp, actor1: d.a, actor2: d.b, goldstein_forecast_13w: d.f, goldstein_normal_52w: d.n,
      forecast_vs_normal: +(d.f - d.n).toFixed(3), surge_probability: d.s, surge_alert: d.al, events_per_week_52w: d.v })));
  } else if (kind === "lanes") {
    name = `shipping_lanes_${stamp}.csv`;
    body = csv((D.lanes || []).map((l) => ({ lane: l.name, lane_zh: l.name_zh, weeks: l.weeks, status: l.status, ships_per_week: l.transits_week,
      normal_ships_per_week: l.normal_week, change_vs_normal: l.change, tankers_per_week: l.tankers_week, normal_tankers_per_week: l.normal_tankers,
      china_trade_at_stake_bn: l.china_trade_at_stake_bn, littoral_states: l.littoral.map((x) => x.iso3).join(" "), highest_littoral_risk: l.littoral_risk })));
  } else if (kind === "events") {
    name = `acled_events_${stamp}.csv`;
    body = csv(D.countries.flatMap((c) => (c.acled?.events || []).map((e) => ({ country: c.iso3, date: e.d, type: e.t, sub_type: e.s, actor: e.a,
      location: e.l, latitude: e.lat, longitude: e.lon, deaths: e.f, notes: e.n }))));
  } else {
    name = `mena_risk_bundle_${stamp}.json`; body = JSON.stringify(D, null, 1); type = "application/json";
  }
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(new Blob([body], { type })), download: name });
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

// ------------------------------------------------------------ data & model page
const ABOUT = {
  en: {
    title: "Data & model",
    lede: "We forecast how relations between 20 Middle East and North Africa states will develop over the next quarter, and turn those forecasts into a country risk score for organisations with trade at stake. The forecasts come from a transformer model that reads the whole region at once.",
    whyH: "Why a transformer: forecasting the region as a system",
    why: ["Most operational forecasting systems predict each country, or each pair of countries, as a separate row: a model scores that row from its own features, and the rest of the world enters only through summaries an analyst decided to compute in advance.",
      "States do not set policy towards one counterpart in isolation. Israel's posture towards Iran is shaped by what happens between Israel and the Houthis, between Iran and the Gulf monarchies, and between both and the great powers. A model that scores each relationship from its own past cannot learn which other relationships matter, or when.",
      "Our model treats every pair of countries as one token. After reading each pair's own history, the model lets all pairs attend to each other before it makes any forecast, so every forecast is conditioned on the state of the whole system – and which relationships matter is learned from the data, week by week."],
    rowLab: "Row-by-row prediction (conventional)", sysLab: "System-aware prediction (this model)",
    rowNote: "each row scored on its own; context only through hand-made features", sysNote: "every forecast conditioned on the whole regional system",
    feat: "features", hist: "history", mdl: "model", forecast: "forecast", attn: ["self-attention", "across all", "1,959 pairs"],
    pipeH: "How it works", dataH: "The data", modelH: "The model", perfH: "How well it works", limH: "What it cannot do", refH: "Refresh",
    data: [["GDELT 2.0", " machine-codes news reports worldwide into events between countries, each scored from −10 (most hostile) to +10 (most cooperative). We aggregate them by week for every pair: 400 pairs among the 20 states, 1,559 links between them and the rest of the world, and 5,122 pairs elsewhere used only for training."],
      ["ACLED", " records political violence on the ground; the last four weeks of deaths are observed context."],
      ["IMF trade statistics", " give China's trade with each country over the last 12 months – the trade value at stake."],
      ["IMF PortWatch", " counts ships through Hormuz, Bab el-Mandeb, Suez and other chokepoints every day – whether the region's trade is getting through."]],
    model: "Two versions are trained on 2015–2022: one forecasts the average tone of each relationship over the next 13 weeks, the other the chance that its worst month will be unusually hostile. Nothing from the test period (2024–2026) was used to train or tune them.",
    stats: (P) => [[fmt0(P.goldstein.r2 * 100) + "%", "of the variation in relations over the next quarter explained", `vs ${fmt0(P.goldstein.r2_trailing_mean * 100)}% by assuming the past year repeats, and ${fmt0(P.goldstein.r2_linear_raw * 100)}% by a regression on the same data`],
      [pct(P.surge.top10_hit), "of the top-10 weekly surge alerts were followed by a surge", `vs ${pct(P.surge.base_rate)} if pairs were picked at random`],
      [pct(P.goldstein.direction_accuracy), "of forecasts got the direction (better or worse than normal) right", "test period March 2024 – June 2026, never used in training"]],
    lim: ["GDELT counts news coverage, not events on the ground, and it starts in 2015: the model has seen only a few region-wide crises.",
      "It forecasts drift better than sudden breaks; use it for horizon scanning alongside expert judgement, not to predict specific events.",
      "Surge probabilities rank relationships well but run high in absolute terms; read them as a ranking."],
    ref: "Forecasts are recomputed every two weeks from the latest complete week of GDELT data; trade and shipping data weekly. This page updates with them.",
    arch: { s: ["1 · DATA", "2 · TRANSFORMER ◆", "3 · FORECASTS ◆", "4 · RISK SCORE"], gd: ["GDELT 2.0 news events", "coded by country pair,", "aggregated by week", "2015 – today"],
      pn: ["Weekly pair panel", "400 pairs within MENA", "1,559 links to the world", "5,122 world pairs", "(training only)"],
      tf: ["Transformer (release v2.2)", "1  Each pair reads its last 13 weeks", "   (temporal attention)", "2  Countries learn an embedding", "3  All 1,959 pairs attend to", "   each other (cross-system)", "", "Trained 2015–2022 · 5 seeds"],
      f1: ["Relations forecast", "mean Goldstein tone of each", "pair, next 13 weeks"], f2: ["Conflict-surge forecast", "chance a pair's worst month", "is unusually hostile"],
      rk: ["Country risk", "Severity = forecast hostility", "+ forecast surge risk", "+ recent violence", "Trade at stake = China trade", "Risk = √(severity × trade)"],
      ob: [["ACLED", "violence"], ["IMF", "trade"]], note: "Blue-violet = transformer forecast (the model's own output); grey dashed = observed data used alongside it." },
  },
  zh: {
    title: "数据与模型",
    lede: "我们预测中东北非20国之间的关系在下一季度如何演变，并将预测转化为面向涉及贸易利益的机构的国家风险评分。预测由一个同时读取整个地区的 Transformer 模型生成。",
    whyH: "为什么用 Transformer：把地区当作一个体系来预测",
    why: ["多数业务化预测系统把每个国家或每组国家对当作独立的一行：模型只根据这一行自身的特征打分，其他国家的信息只能通过分析人员事先设计的汇总特征进入。",
      "国家不会孤立地制定对某一对象国的政策。以色列对伊朗的态度，受到以色列与胡塞武装、伊朗与海湾君主国、以及双方与大国之间关系的影响。只根据每组关系自身历史打分的模型，无法学到哪些其他关系重要、何时重要。",
      "我们的模型把每一对国家当作一个 token：先读取每组关系自身的历史，再让所有国家对彼此“注意”，然后才做出预测。因此每个预测都以整个体系的状态为条件——哪些关系重要，由模型逐周从数据中学习。"],
    rowLab: "逐行预测（传统方法）", sysLab: "体系感知预测（本模型）",
    rowNote: "每行单独打分；背景信息只能来自人工特征", sysNote: "每个预测都以整个地区体系为条件",
    feat: "特征", hist: "历史", mdl: "模型", forecast: "预测", attn: ["自注意力", "覆盖全部", "1,959 组国家对"],
    pipeH: "工作流程", dataH: "数据", modelH: "模型", perfH: "效果如何", limH: "局限", refH: "更新频率",
    data: [["GDELT 2.0", " 将全球新闻报道自动编码为国家之间的事件，每个事件按 −10（最敌对）至 +10（最合作）打分。我们按周汇总每组国家对：20国之间的400组、它们与世界其他国家的1,559组联系，以及仅用于训练的5,122组其他国家对。"],
      ["ACLED", " 记录实地政治暴力事件；过去四周的死亡人数作为观测背景。"],
      ["IMF 贸易统计", " 提供过去12个月中国与各国的贸易额——即涉及的贸易价值。"],
      ["IMF PortWatch", " 每日统计通过霍尔木兹、曼德海峡、苏伊士等咽喉要道的船只——即本地区贸易能否通行。"]],
    model: "模型在2015–2022年数据上训练两个版本：一个预测未来13周每组关系的平均基调，另一个预测其最糟月份异常敌对的概率。测试期（2024–2026）的数据从未用于训练或调参。",
    stats: (P) => [[fmt0(P.goldstein.r2 * 100) + "%", "下一季度关系变化中可被解释的比例", `若假设过去一年重演为${fmt0(P.goldstein.r2_trailing_mean * 100)}%，同数据线性回归为${fmt0(P.goldstein.r2_linear_raw * 100)}%`],
      [pct(P.surge.top10_hit), "每周前10项激增预警中随后确实出现激增的比例", `随机挑选为${pct(P.surge.base_rate)}`],
      [pct(P.goldstein.direction_accuracy), "预测方向（好于或差于常态）正确的比例", "测试期2024年3月–2026年6月，从未用于训练"]],
    lim: ["GDELT 统计的是新闻报道而非实地事件，且始于2015年：模型见过的全区域危机不多。",
      "模型更擅长预测渐变而非突变；应作为前瞻研判工具与专家判断结合使用，而非预测具体事件。",
      "激增概率对关系的排序较准，但绝对值偏高；请将其视为排序。"],
    ref: "预测每两周根据最新完整一周的 GDELT 数据重新计算；贸易与航运数据每周更新。本页随之更新。",
    arch: { s: ["1 · 数据", "2 · TRANSFORMER ◆", "3 · 预测 ◆", "4 · 风险评分"], gd: ["GDELT 2.0 新闻事件", "按国家对编码", "按周汇总", "2015年至今"],
      pn: ["每周国家对面板", "中东北非内400组", "与世界1,559组联系", "5,122组其他国家对", "（仅用于训练）"],
      tf: ["Transformer（v2.2版）", "1  每组读取其近13周历史", "   （时间注意力）", "2  学习国家嵌入表示", "3  全部1,959组国家对", "   相互注意（跨体系）", "", "2015–2022训练 · 5个种子"],
      f1: ["关系预测", "每组国家对未来13周", "平均戈德斯坦基调"], f2: ["冲突激增预测", "最糟月份异常敌对", "的概率"],
      rk: ["国家风险", "严重度 = 预测敌对程度", "+ 预测激增风险", "+ 近期暴力", "涉及贸易 = 对华贸易", "风险 = √(严重度 × 贸易)"],
      ob: [["ACLED", "暴力"], ["IMF", "贸易"]], note: "蓝紫色 = Transformer 预测（模型自身输出）；灰色虚线 = 同时使用的观测数据。" },
  },
};

function renderAbout() {
  const A = ABOUT[S.lang], P = S.data.meta.performance;
  $("#stage").classList.add("dim");
  drawRegionMap(null, true);
  $("#view").innerHTML = `
    <article class="page glass">
      <h2 class="pt">${A.title}</h2>
      <p class="lede">${A.lede}</p>
      <h3>${A.whyH}</h3>
      ${A.why.map((p) => `<p>${p}</p>`).join("")}
      ${conceptDiagram(A)}
      <h3>${A.pipeH}</h3>
      ${archDiagram(A.arch)}
      <h3>${A.dataH}</h3>
      <ul>${A.data.map(([b, s]) => `<li><b>${b}</b>${s}</li>`).join("")}</ul>
      <h3>${A.modelH}</h3>
      <p>${A.model}</p>
      <h3>${A.perfH}</h3>
      ${P ? `<div class="stats">${A.stats(P).map(([big, lab, vs]) => `<div class="stat"><div class="big">${fc(big)}</div><div class="lab">${lab}</div><div class="vs">${vs}</div></div>`).join("")}</div>` : ""}
      <h3>${A.limH}</h3>
      <ul>${A.lim.map((x) => `<li>${x}</li>`).join("")}</ul>
      <h3>${A.refH}</h3>
      <p>${A.ref}</p>
    </article>`;
}

// the paper's Figure 1, redrawn: row-by-row vs system-aware prediction
function conceptDiagram(A) {
  const pairs = ["ISR–IRN", "ISR–PSE", "SAU–YEM", "IRN–QAT", "TUR–SYR"];
  const ink = "#e9eaec", mut = "#9a9ea6", fa = "#5fd3ff";
  const box = (x, y, w, h, l1, l2, stroke = mut, fill = "rgba(255,255,255,0.04)") => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="7" fill="${fill}" stroke="${stroke}" stroke-width="1.2"/>
    <text x="${x + w / 2}" y="${y + (l2 ? h / 2 - 3 : h / 2 + 4)}" text-anchor="middle" font-size="12" fill="${ink}">${l1}</text>${l2 ? `<text x="${x + w / 2}" y="${y + h / 2 + 12}" text-anchor="middle" font-size="11" fill="${mut}">${l2}</text>` : ""}`;
  const arr = (x1, y, x2, col = mut) => `<path d="M${x1},${y} L${x2 - 5},${y}" stroke="${col}" stroke-width="1.5" marker-end="url(#${col === fa ? "cf" : "cg"})"/>`;
  let left = "", right = "";
  pairs.forEach((p, i) => {
    const y = 46 + i * 52;
    left += box(10, y, 112, 40, p, A.feat) + arr(122, y + 20, 152) + box(152, y, 92, 40, A.mdl, "", mut, "rgba(255,255,255,0.08)") + arr(244, y + 20, 274) + box(274, y, 96, 40, A.forecast);
    right += box(510, y, 112, 40, p, A.hist) + arr(622, y + 20, 664, fa) + arr(776, y + 20, 808, fa) + box(808, y, 96, 40, A.forecast);
  });
  return `<svg class="arch concept" viewBox="0 0 920 340" role="img" aria-label="${esc(A.rowLab)} / ${esc(A.sysLab)}">
    <defs>
      <marker id="cg" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="${mut}"/></marker>
      <marker id="cf" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="${fa}"/></marker>
      <linearGradient id="cgr" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="rgba(95,211,255,0.28)"/><stop offset="1" stop-color="rgba(180,155,255,0.22)"/></linearGradient>
    </defs>
    <text x="10" y="22" font-size="13" font-weight="650" fill="${ink}">(a) ${A.rowLab}</text>
    <text x="510" y="22" font-size="13" font-weight="650" fill="${fa}">(b) ${A.sysLab} ◆</text>
    ${left}${right}
    <rect x="664" y="40" width="112" height="258" rx="10" fill="url(#cgr)" stroke="${fa}" stroke-width="1.6"/>
    ${A.attn.map((l, i) => `<text x="720" y="${152 + i * 18}" text-anchor="middle" font-size="12.5" font-weight="650" fill="${ink}">${l}</text>`).join("")}
    <text x="190" y="328" text-anchor="middle" font-size="11" fill="${mut}">${A.rowNote}</text>
    <text x="707" y="328" text-anchor="middle" font-size="11" fill="${mut}">${A.sysNote}</text>
  </svg>`;
}

function archDiagram(a) {
  const ink = "#e9eaec", mut = "#9a9ea6";
  const box = (x, y, w, h, lines, kind = "data") => {
    const stroke = kind === "fc" ? "url(#gfc)" : kind === "risk" ? "#e9eaec" : "#7a7e86";
    return `<g><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" fill="rgba(255,255,255,${kind === "fc" ? 0.07 : 0.04})" stroke="${stroke}" stroke-width="${kind === "data" ? 1 : 1.8}" ${kind === "obs" ? 'stroke-dasharray="4 3"' : ""}/>
      <text x="${x + 14}" y="${y + 24}" font-size="13" font-weight="650" fill="${kind === "fc" ? "#5fd3ff" : ink}">${lines[0]}</text>
      ${lines.slice(1).map((l, i) => `<text x="${x + 14}" y="${y + 44 + i * 17}" font-size="11.5" fill="${mut}">${l}</text>`).join("")}</g>`;
  };
  const arrow = (x1, y1, x2, y2, f = false) => `<path d="M${x1},${y1} L${x2 - 6},${y2}" stroke="${f ? "#5fd3ff" : "#9a9ea6"}" stroke-width="1.6" fill="none" marker-end="url(#${f ? "afc" : "a"})"/>`;
  return `<svg class="arch" viewBox="0 0 1000 330" role="img">
    <defs>
      <linearGradient id="gfc" x1="0" x2="1"><stop offset="0" stop-color="#5fd3ff"/><stop offset="1" stop-color="#b49bff"/></linearGradient>
      <marker id="a" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#9a9ea6"/></marker>
      <marker id="afc" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#5fd3ff"/></marker>
    </defs>
    ${a.s.map((l, i) => `<text x="${[20, 268, 540, 790][i]}" y="18" font-size="10" letter-spacing="1.5" fill="${i === 1 || i === 2 ? "#5fd3ff" : mut}" font-family="JetBrains Mono, monospace">${l}</text>`).join("")}
    ${box(20, 32, 210, 92, a.gd)}${box(20, 146, 210, 110, a.pn)}${arrow(125, 124, 125, 146)}
    <rect x="268" y="32" width="236" height="224" rx="12" fill="rgba(255,255,255,0.07)" stroke="url(#gfc)" stroke-width="2"/>
    <text x="284" y="58" font-size="13" font-weight="650" fill="#5fd3ff">${a.tf[0]}</text>
    ${a.tf.slice(1).map((l, i) => `<text x="284" y="${86 + i * 21}" font-size="12" fill="${i === 6 ? mut : ink}">${l}</text>`).join("")}
    ${arrow(230, 200, 268, 200)}
    ${box(540, 40, 214, 92, a.f1, "fc")}${box(540, 156, 214, 92, a.f2, "fc")}${arrow(504, 100, 540, 86, true)}${arrow(504, 190, 540, 202, true)}
    ${box(790, 40, 192, 152, a.rk, "risk")}${arrow(754, 86, 790, 96, true)}${arrow(754, 202, 790, 150, true)}
    ${box(790, 222, 92, 64, a.ob[0], "obs")}${box(890, 222, 92, 64, a.ob[1], "obs")}${arrow(836, 222, 836, 198)}${arrow(936, 222, 936, 198)}
    <text x="20" y="318" font-size="11" fill="${mut}">${a.note}</text>
  </svg>`;
}

// ------------------------------------------------------------ charts
function gauge(sel, value, size) {
  const r = size * 0.4, th = size * 0.075, start = -0.75 * Math.PI, end = 0.75 * Math.PI;
  const id = `g${Math.random().toString(36).slice(2, 7)}`;
  const svg = d3.select(sel).append("svg").attr("viewBox", `0 0 ${size} ${size}`).style("width", `${size}px`);
  const lg = svg.append("defs").append("linearGradient").attr("id", id).attr("x1", 0).attr("x2", 1);
  lg.append("stop").attr("offset", 0).attr("stop-color", css("--fc-a"));
  lg.append("stop").attr("offset", 1).attr("stop-color", css("--fc-b"));
  const g = svg.append("g").attr("transform", `translate(${size / 2},${size / 2 + 6})`);
  d3.range(0, 41).forEach((i) => {
    const a = start + (end - start) * (i / 40), r0 = r + 4, r1 = r + (i % 5 === 0 ? 9 : 6);
    g.append("line").attr("x1", Math.sin(a) * r0).attr("y1", -Math.cos(a) * r0).attr("x2", Math.sin(a) * r1).attr("y2", -Math.cos(a) * r1).attr("stroke", css("--line")).attr("stroke-width", i % 5 === 0 ? 1 : 0.6);
  });
  const arc = d3.arc().innerRadius(r - th).outerRadius(r).cornerRadius(6);
  g.append("path").attr("d", arc({ startAngle: start, endAngle: end })).attr("fill", "rgba(255,255,255,0.1)");
  const frac = Math.min(1, value / 100);
  const fg = g.append("path").attr("fill", riskColor(value));
  const t0 = performance.now();
  const step = (now) => {
    const k = Math.min(1, (now - t0) / 700), e = 1 - Math.pow(1 - k, 3);
    fg.attr("d", arc({ startAngle: start, endAngle: start + (end - start) * Math.max(0.02, frac * e) }));
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
  g.append("text").attr("text-anchor", "middle").attr("y", size * 0.05).attr("font-size", size * 0.22).attr("font-weight", 650).attr("fill", `url(#${id})`).attr("font-family", "JetBrains Mono, monospace").style("filter", "drop-shadow(0 0 6px rgba(120,190,255,0.6))").text(fmt0(value));
  g.append("text").attr("text-anchor", "middle").attr("y", size * 0.18).attr("font-size", 9.5).attr("fill", css("--muted")).attr("letter-spacing", 1.2).text("/ 100");
}

function lineChart(sel, values, o) {
  const el = $(sel), weeks = S.data.meta.chart_weeks.map((w) => new Date(w));
  const W = el.clientWidth || 300, H2 = 140, m = { t: 8, r: 52, b: 20, l: 28 };
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H2}`);
  const lastWeek = weeks[weeks.length - 1], fEnd = d3.timeWeek.offset(lastWeek, 13);
  const x = d3.scaleTime().domain([weeks[0], fEnd]).range([m.l, W - m.r]);
  const smooth = values.map((v, i) => d3.mean(values.slice(Math.max(0, i - 3), i + 1)));
  const y = d3.scaleLinear().domain(d3.extent([...smooth, o.normal, o.forecast])).nice().range([H2 - m.b, m.t]);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H2 - m.b})`).call(d3.axisBottom(x).ticks(3).tickSizeOuter(0));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(3).tickSize(-(W - m.l - m.r))).call((g) => g.select(".domain").remove());
  const gid = `fcg${Math.random().toString(36).slice(2, 6)}`;  // user-space gradient: a horizontal line has a zero-height bounding box
  const lg = svg.append("defs").append("linearGradient").attr("id", gid).attr("gradientUnits", "userSpaceOnUse").attr("x1", x(lastWeek)).attr("x2", x(fEnd)).attr("y1", 0).attr("y2", 0);
  lg.append("stop").attr("offset", 0).attr("stop-color", css("--fc-a"));
  lg.append("stop").attr("offset", 1).attr("stop-color", css("--fc-b"));
  svg.append("line").attr("x1", m.l).attr("x2", x(fEnd)).attr("y1", y(o.normal)).attr("y2", y(o.normal)).attr("stroke", css("--muted")).attr("stroke-dasharray", "3 4");
  svg.append("path").attr("d", d3.line().x((d, i) => x(weeks[i])).y((d) => y(d)).curve(d3.curveMonotoneX)(smooth)).attr("fill", "none").attr("stroke", css("--ink")).attr("stroke-width", 1.6);
  svg.append("rect").attr("x", x(lastWeek)).attr("width", x(fEnd) - x(lastWeek)).attr("y", m.t).attr("height", H2 - m.b - m.t).attr("fill", "rgba(95,211,255,0.08)");
  svg.append("line").attr("x1", x(lastWeek)).attr("x2", x(fEnd)).attr("y1", y(smooth[smooth.length - 1])).attr("y2", y(o.forecast)).attr("stroke", `url(#${gid})`).attr("stroke-width", 1).attr("stroke-dasharray", "2 3");
  svg.append("line").attr("x1", x(lastWeek) + 3).attr("x2", x(fEnd)).attr("y1", y(o.forecast)).attr("y2", y(o.forecast)).attr("stroke", `url(#${gid})`).attr("stroke-width", 3.5).attr("stroke-linecap", "round").style("filter", "drop-shadow(0 0 5px rgba(120,190,255,0.8))");
  svg.append("text").attr("x", x(fEnd) + 6).attr("y", y(o.forecast) + 4).attr("font-size", 10.5).attr("font-weight", 650).attr("fill", css("--fc-a")).text(`◆ ${fmt2(o.forecast)}`);
  const cross = svg.append("line").attr("y1", m.t).attr("y2", H2 - m.b).attr("stroke", css("--muted")).attr("opacity", 0);
  svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", x(lastWeek) - m.l).attr("height", H2 - m.b - m.t).attr("fill", "transparent")
    .on("mousemove", (ev) => {
      const [px] = d3.pointer(ev);
      const i = Math.max(0, Math.min(weeks.length - 1, d3.bisectCenter(weeks.map(Number), +x.invert(px))));
      cross.attr("x1", x(weeks[i])).attr("x2", x(weeks[i])).attr("opacity", 0.5);
      showTip(`<div class="t">${weeks[i].toISOString().slice(0, 10)}</div><div class="r">${S.lang === "zh" ? "4周平均" : "4-week average"} <b>${fmt2(smooth[i])}</b></div>`, ev);
    })
    .on("mouseleave", () => { cross.attr("opacity", 0); hideTip(); });
}

init();

"""
Build the merged paper (conflict forecasting + trade module + risk monitor) as one Word document.
It combines the transformer section (paper/build_docx.py) and the trade section
(paper/trade_section_draft.md) into one product paper; numbers come from the same files.

    python paper/build_paper.py      -> paper/risk_paper.docx

Figures and tables are numbered by label (FIGS / TABS below), so cross-references follow the order
in which they appear.
"""

import json
import math
import re

import joblib
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

P = "paper"
S = json.load(open(f"{P}/stats.json"))                     # data facts, baselines, GDELT example
R = json.load(open(f"{P}/v21_stats.json"))                 # release v2.1 numbers, attention, experiments
F = json.load(open(f"{P}/figure_numbers.json"))            # improvement path, regional ablation, direction
L = json.load(open("logs/ladder_results.json"))            # rudimentary networks
LC = json.load(open(f"{P}/learning_curve.json"))
BT = json.load(open(f"{P}/backtest.json"))
LIN = json.load(open(f"{P}/linear_baselines.json"))
V22S = json.load(open(f"{P}/v22_stats.json"))
A22 = json.load(open(f"{P}/v22_analysis.json"))           # v2.2 variance, examples, attention (compute_v22_analysis.py)
SITE = json.load(open("website/data/latest.json", encoding="utf-8"))  # the monitor's current bundle
B = S["goldstein_benchmark"]
V21, V20, GBM = R["benchmark"]["v2.1"], R["benchmark"]["v2.0"], R["gbm"]
V22, FL = V22S["v2.2"], V22S["flag"]
OLS1, OLS44 = LIN["OLS current week"], LIN["OLS 44 history features"]
AB, PATH = F["ablation"], F["path"]
d = S["data"]
SURGE_ART = joblib.load("artifacts/surge_v2.2_preprocessing.joblib")  # the release's surge cut-off and calibration
SURGE_T, SURGE_CAL = SURGE_ART["surge_threshold"], SURGE_ART["calibration"]
p_at = lambda gap: 1 / (1 + math.exp(-(SURGE_CAL["a"] * gap + SURGE_CAL["b"])))

# ------------------------------------------------------------------ numbering
FIGS = ["fig1_concept", "fig10_gdelt", "fig2_architecture", "fig3_timeline", "fig4_benchmark", "fig5_path",
        "fig6_ablation", "fig13_backtest", "fig12_learning_curve", "fig11_scaling", "fig8_attention",
        "fig7_examples", "fig14_surge_isr", "fig_trade_1_closure", "fig_trade_2_regression",
        "fig_trade_3_backtest", "fig15_website"]
TABS = ["components", "hormuz_params", "gdelt_spec", "external", "bench", "path", "experiments", "flags",
        "closure", "reg", "robust", "trade_models", "backtest", "by_year", "decision", "current", "codes"]
fig = lambda name: f"Figure {FIGS.index(name) + 1}"
tab = lambda key: f"Table {TABS.index(key) + 1}"

INK = RGBColor(0x0B, 0x0B, 0x0B)
GREY = RGBColor(0x52, 0x51, 0x4E)
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.4)
sec.top_margin = sec.bottom_margin = Cm(2.2)
st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.font.size = Pt(11)
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.15
for lvl, size in [(1, 14), (2, 12), (3, 11)]:
    h = doc.styles[f"Heading {lvl}"]
    h.font.name = "Times New Roman"
    h.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    h.font.size, h.font.bold, h.font.italic = Pt(size), True, lvl == 3
    h.font.color.rgb = INK
    h.paragraph_format.space_before = Pt(14 if lvl == 1 else 10)
    h.paragraph_format.space_after = Pt(4)
    h.paragraph_format.keep_with_next = True

fig_no, tab_no = [0], [0]


def _runs(p, text, italic=False, size=None, color=None):
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not part:
            continue
        r = p.add_run(part[2:-2] if part.startswith("**") else part[1:-1] if part.startswith("*") else part)
        r.bold = part.startswith("**")
        r.italic = italic or (part.startswith("*") and not part.startswith("**"))
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = color


def para(text, italic=False, size=None, align=None, after=None, color=None):
    p = doc.add_paragraph()
    _runs(p, text, italic=italic, size=size, color=color)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT if align == "left" else \
        WD_ALIGN_PARAGRAPH.CENTER if align == "center" else WD_ALIGN_PARAGRAPH.JUSTIFY
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def bullet(text, style="List Bullet"):
    p = doc.add_paragraph(style=style)
    _runs(p, text)
    p.paragraph_format.space_after = Pt(2)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


_num = {"el": None, "n": 0}


def numbered(text):
    """Numbered item; a list restarts at 1 unless it directly follows another numbered item
    (Word's List Number style would continue one count through the whole document)."""
    last = doc.element.body[-2] if len(doc.element.body) > 1 else None  # [-1] is the section properties
    _num["n"] = _num["n"] + 1 if last is not None and last is _num["el"] else 1
    p = doc.add_paragraph()
    _runs(p, f"{_num['n']}.\t{text}")
    p.paragraph_format.left_indent = Cm(0.75)
    p.paragraph_format.first_line_indent = Cm(-0.75)
    p.paragraph_format.tab_stops.add_tab_stop(Cm(0.75))
    p.paragraph_format.space_after = Pt(2)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    _num["el"] = p._p


def code(lines):
    for ln in lines:
        p = doc.add_paragraph()
        r = p.add_run(ln)
        r.font.name, r.font.size = "Courier New", Pt(8.5)
        p.paragraph_format.space_after = Pt(0)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def figure(name, caption, width=16.0):
    fig_no[0] += 1
    assert FIGS.index(name) + 1 == fig_no[0], f"figure order: {name}"
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(f"{P}/{name}.png", width=Cm(width))
    c = doc.add_paragraph()
    _runs(c, f"**Figure {fig_no[0]}.** {caption}", size=9.5, color=GREY)
    c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    c.paragraph_format.space_after = Pt(10)


def shade(cell, hex_fill):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    cell._tc.get_or_add_tcPr().append(shd)


def table(key, caption, header, rows, widths, bold_last=False, note=None):
    tab_no[0] += 1
    assert TABS.index(key) + 1 == tab_no[0], f"table order: {key}"
    c = doc.add_paragraph()
    _runs(c, f"**Table {tab_no[0]}.** {caption}", size=9.5, color=GREY)
    c.paragraph_format.keep_with_next = True
    c.paragraph_format.space_after = Pt(3)
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(header):
        cell = t.rows[0].cells[j]
        cell.text = ""
        r = cell.paragraphs[0].add_run(h)
        r.bold, r.font.size = True, Pt(9)
        shade(cell, "E8E7E3")
    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            cell = t.rows[i].cells[j]
            cell.text = ""
            _runs(cell.paragraphs[0], str(v), size=9)
            if bold_last and i == len(rows):
                for r in cell.paragraphs[0].runs:
                    r.bold = True
            if j > 0:
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for i, row in enumerate(t.rows):
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for j, w in enumerate(widths):
            row.cells[j].width = Cm(w)
            if i < len(t.rows) - 1:
                for cp in row.cells[j].paragraphs:
                    cp.paragraph_format.keep_with_next = True
    n = doc.add_paragraph()
    if note:
        _runs(n, note, size=8.5, color=GREY)
    n.paragraph_format.space_after = Pt(10 if note else 4)


def merged_table(key, caption, header, rows, merges, widths, note=None):
    """Table with merged cells. rows: lists of strings (None = covered by a merge);
    merges: ((r0, c0), (r1, c1)) body-cell ranges, row 0 = first body row."""
    tab_no[0] += 1
    assert TABS.index(key) + 1 == tab_no[0], f"table order: {key}"
    c = doc.add_paragraph()
    _runs(c, f"**Table {tab_no[0]}.** {caption}", size=9.5, color=GREY)
    c.paragraph_format.keep_with_next = True
    c.paragraph_format.space_after = Pt(3)
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(header):
        cell = t.rows[0].cells[j]
        cell.text = ""
        r = cell.paragraphs[0].add_run(h)
        r.bold, r.font.size = True, Pt(9)
        shade(cell, "E8E7E3")
    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            t.rows[i].cells[j].text = ""
            if v:
                _runs(t.rows[i].cells[j].paragraphs[0], v, size=9)
    for i, row in enumerate(t.rows):
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for j, w in enumerate(widths):
            row.cells[j].width = Cm(w)
            for cp in row.cells[j].paragraphs:
                cp.paragraph_format.keep_with_next = i < len(t.rows) - 1
    for (r0, c0), (r1, c1) in merges:
        m = t.cell(r0 + 1, c0).merge(t.cell(r1 + 1, c1))
        for extra in m.paragraphs[1:]:  # merging appends the covered cells' empty paragraphs
            if not extra.text:
                extra._p.getparent().remove(extra._p)
        m.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        if c0 == 0 or c1 > c0:
            shade(m, "F4F3F0")
    n = doc.add_paragraph()
    if note:
        _runs(n, note, size=8.5, color=GREY)
    n.paragraph_format.space_after = Pt(10 if note else 4)


f3 = lambda x: f"{x:.3f}"
pct = lambda x: f"{x:.1%}"
DIRN = FL["v2.0 direction model (4-week escalation probability)"]
SURG = FL["v2.2 conflict-surge model"]
NAIV = FL["Naive: last quarter's peak conflict"]
LINF = FL["Linear regression, hand-built features*"]
GOLD = FL["v2.2 Goldstein forecast (fall below normal)"]

# ============================================================== title + summary
t = doc.add_paragraph()
r = t.add_run("Geopolitical Risk to China's Trade with the Middle East and North Africa: System-Level Conflict "
              "Forecasting, Trade Exposure and a Live Risk Monitor")
r.bold, r.font.size = True, Pt(16)
t.paragraph_format.space_after = Pt(4)
para("Risk-A-Lab geopolitical risk forecasting project. Conflict models: release v2.2; trade module: status October "
     "2026. All results are reproducible from the project repository (Appendix A3). Items marked **[verify]** rest on a "
     "source that should be re-checked before submission.", italic=True, size=9.5, color=GREY, align="left", after=12)

doc.add_heading("Summary", level=2)
para(
    "Chinese enterprises in the Middle East and North Africa (MENA) are exposed to conflict in two ways: directly, "
    "through violence where they operate, and indirectly, through disruption of the shipping routes that carry the "
    "region's trade with China. This paper describes an early-warning product built around both channels. It has "
    "two analytical parts, a forecast of how relations between MENA states will develop over the next quarter and "
    "a measurement of how conflict reaches China's trade, which are combined into a country risk score and published "
    "through a bilingual, automatically refreshed risk monitor.")
para(
    "**Forecasting conflict.** Most operational conflict-forecasting systems treat each country or country pair as a "
    "separate row of data and predict it from its own history, with the rest of the world entering, if at all, "
    "through hand-built aggregate features. We develop and test an alternative, closer to how states make decisions: "
    "a transformer that forecasts every relationship in the system jointly. Each directed country pair (dyad) is a "
    "token; a temporal transformer encodes each dyad's recent history, and a cross-dyad transformer lets every dyad "
    "attend to every other. The model forecasts the 400 dyads among 20 MENA states while attending to a wider system "
    "of 1,959 dyads that also includes the region's relations with the rest of the world. Using GDELT event data "
    "aggregated by week, it forecasts each dyad's mean Goldstein cooperation–conflict score over the next 13 weeks. "
    f"Under a strict out-of-sample protocol the release model (v2.2) reaches a test R² of {f3(V22['r2'])}, about "
    f"twice that of a linear regression on the same raw weekly inputs ({f3(OLS1['r2'])}), and ahead of the strongest "
    f"naive forecast (a trailing 52-week mean, {f3(B['trailing 52-week mean']['r2'])}), gradient boosting "
    f"({f3(GBM['r2'])}) and a linear regression on 44 history features that an analyst designed by hand, with "
    f"hindsight ({f3(OLS44['r2'])}). The transformer learns those summaries itself: trained on the region alone "
    f"(v2.1) it already matches the hand-engineered model ({f3(V21['r2'])}), and training it on "
    f"{V22S['world_dyads']:,} additional dyads from outside the region moves it ahead. A second model with the same architecture forecasts the *worst* "
    f"month of conflict in the coming quarter: {SURG['conflict_surge']['top10']:.0%} of its top-ten weekly alerts "
    f"are followed by a conflict surge, against a base rate of {V22S['surge_rate']:.0%}.")
para(
    "**Conflict and China's trade.** Combining monthly IMF trade data (China and 20 MENA partners plus the United "
    "States, 2017-01 to 2026-06) with GDELT conflict data, IMF PortWatch chokepoint transits and a route-based measure "
    "of each partner's exposure to the Strait of Hormuz, we find that conflict reaches China's trade mainly through "
    "the shipping route, not through partners' own instability. When the strait was effectively closed in March "
    "2026, China's imports from Iraq, Qatar and Kuwait fell by 83–89% year on year in the second quarter, while "
    "imports from Saudi Arabia, which can export from the Red Sea, rose by 12%. In a fixed-effects design a Gulf "
    "conflict shock lowers import growth from Hormuz-exposed partners by a cumulative −0.45 log points per standard "
    "deviation sustained over four months (permutation p = 0.004), an effect identified by the 2026 episode alone. "
    "For forecasting China's trade one quarter ahead, a two-parameter shrunk-persistence rule beat a country × week "
    "transformer out of sample and is used operationally; its skill is modest, and no model anticipated the 2026 "
    "collapse.")
para(
    "**The product.** Risk is defined for an organisation with trade at stake: it is high only when the conflict "
    "environment is expected to be serious and much trade is exposed to it. The two transformer forecasts, recent "
    "ACLED violence and China's trade with each country form the score; shipping-lane status and Hormuz exposure are "
    "reported alongside it. The contribution is methodological and practical: a demonstration of how "
    "self-attention can bring the international system into dyad-level forecasts, evidence on the channel through "
    "which conflict actually reaches China's trade, and a working monitor that re-scores the region every week from "
    "live data, with recommendations for the enterprises it serves.")

# ============================================================== 1
doc.add_heading("1  Introduction", level=1)
doc.add_heading("1.1  Two channels of risk", level=2)
para("Chinese firms in the Middle East are exposed to conflict through two channels:")
bullet("**Directly:** violence and deteriorating relations at the place of operation.")
bullet("**Indirectly:** disruption of the routes that carry the region's trade with China.")
para("The second channel is concentrated in a few chokepoints. The most important is the Strait of Hormuz, through "
     "which most Gulf oil and LNG exports pass. We ask four questions:")
numbered("Can relations among MENA states be forecast a quarter ahead, and does modelling the region as a system "
         "rather than one relationship at a time help? (Section 5)")
numbered("How exposed is China's trade with each MENA partner to a Hormuz disruption? (Section 3.4)")
numbered("Does Gulf conflict reduce China's trade more for more exposed partners? (Section 6.2)")
numbered("Can China's trade with each partner be forecast one quarter ahead, and does conflict information help? "
         "(Section 6.3)")
para("Section 2 defines risk and Section 3 the indicator system built on these answers. Section 4 describes the data, "
     "their coding and the analysis and refresh process. Sections 5 and 6 report the findings, Section 7 the risk "
     "monitor that puts them to use, Section 8 recommendations for enterprises, and Section 9 the limitations and "
     "notes needed to use the results correctly.")

doc.add_heading("1.2  Forecasting countries one row at a time", level=2)
para(
    "Quantitative conflict forecasting has matured into operational systems. The Violence & Impacts Early-"
    "Warning System (VIEWS) publishes monthly forecasts of state-based, non-state and one-sided violence at the "
    "country-month and PRIO-GRID-month levels (Hegre et al. 2019, 2021). ConflictForecast derives topic "
    "proportions from millions of newspaper articles and feeds them, with conflict history, into "
    "machine-learning classifiers that forecast armed-conflict onset and intensity at the country-month level "
    "(Mueller and Rauh 2018, 2022). Both are carefully validated and widely used, and the VIEWS prediction "
    "competitions have pushed the field towards rigorous out-of-sample evaluation (Vesco et al. 2022).")
para(
    "Their dominant modelling paradigm shares two features. First, the core learners are mostly "
    "non-deep-learning models, chiefly tree ensembles such as random forests and gradient boosting, which work "
    "on fixed-length feature vectors. Second, and following from that, each unit of observation is predicted as "
    "a separate *row*: the model maps that row's features to that row's outcome, and other units enter only "
    "through features the analyst decided in advance to compute, such as spatial lags, neighbourhood averages "
    "or global trends. Deep-learning entries exist in the competitions, but the systems in routine use remain "
    "row-by-row in this sense.")
para(
    "For international relations this can be a substantive restriction, not only a technical one. States do not "
    "set policy towards one counterpart in isolation. Israel's posture towards Iran is shaped by what happens "
    "between Israel and the Houthis, between Iran and the Gulf monarchies, and between both and the great "
    "powers; a Saudi–Iranian détente changes the calculus in Yemen, Lebanon and Iraq at once. Systemic theories "
    "of international politics hold that outcomes depend on the configuration of the whole system rather than "
    "on unit attributes alone (Waltz 1979). A model that scores each relationship from its own past, plus a few "
    "analyst-chosen summaries of the rest, cannot learn which other relationships matter, when, or how much. "
    f"{fig('fig1_concept')} contrasts the two approaches.")
figure("fig1_concept", "Row-by-row versus system-aware forecasting. (a) Conventional set-up: each country pair is "
       "a separate row; context enters only through features computed in advance. (b) This study: each pair's "
       "history is encoded, then all pairs exchange information through self-attention before any forecast is "
       "made.", 15.5)

# ============================================================== 2
doc.add_heading("2  Risk definition", level=1)
para(
    "Risk is defined for an organisation with trade at stake in a country: exporters, importers and energy buyers, "
    "contractors and investors, shippers, and staff on the ground. It is high only when the conflict environment is "
    "expected to be serious **and** much trade is exposed to it. A country at war with almost no trade with China, "
    "or a large trading partner in calm conditions, should not rank at the top. The horizon is one quarter (13 weeks), "
    "which leaves organisations time to act.")
para("Each country receives three scores on a 0–100 scale, recomputed at every refresh:")
bullet("**Conflict severity**: how serious the conflict environment is expected to be. It combines the two "
       "transformer forecasts (forecast hostility and forecast surge risk) with one observed signal, recent violence.")
bullet("**Trade at stake** (exposure): how much of China's trade is with the country.")
bullet("**Risk** = √(severity × trade at stake), the geometric mean of the two. Unlike a product it stays on the "
       "0–100 scale, and unlike an average it rewards neither alone: a country at war with almost no trade at "
       "stake, or a large partner in calm conditions, scores moderately. Bands are severe (≥ 55), high (≥ 45), "
       "elevated (≥ 32) and low. **Regional risk** is the average of the 20 country scores, and every score is "
       "also reported as its change since the previous refresh.")
para(
    "The indirect channel is measured, but deliberately kept beside the score rather than inside it. The status of "
    "the shipping lanes and each partner's dependence on the Strait of Hormuz (Sections 3.4–3.5) describe a shock that "
    "hits many countries at once and whose scale is identified from one episode (Section 6.2); folding it into a "
    "country score would hide which channel is driving the risk. The monitor therefore reports route risk "
    "separately, and the written briefs connect the two for each country.")
para(
    "Two conventions run through every indicator. **Forecast and observation are kept apart**: everything produced by, "
    "or built on, the transformer is marked as a forecast, and observed data (recent violence, trade, shipping) are "
    "shown as such. **The score orders countries by business risk; it is not a probability.** Probabilities are "
    "reported where they are calibrated, for conflict surges at the level of country pairs.")

# ============================================================== 3
doc.add_heading("3  The indicator system and its construction logic", level=1)
para(
    f"The indicators fall into three layers ({tab('components')}): the **conflict outlook**, forecast by the "
    "transformers for every country pair; **observed conditions**, recent violence and the size of China's trade; "
    "and the **transmission channel**, Hormuz exposure and the status of the shipping lanes, together with a "
    "trade outlook from the trade module.")
merged_table(
    "components", "The indicator system: the risk score and how it is built, and the indicators reported beside it.",
    ["Score", "Component", "Indicator", "Source", "Kind", "Scaling (weight)"],
    [["**Risk** = √(severity × trade at stake), 0–100", "**Conflict severity** = 100 × weighted sum of its three "
      "indicators", "Forecast hostility", "Transformer v2.2, 13-week Goldstein forecast", "Forecast",
      "+1.5 → 0, −4.0 → 1 (0.40)"],
     [None, None, "Forecast surge risk", "Transformer v2.2, conflict-surge probability", "Forecast",
      "0 → 0, 0.6 → 1 (0.30)"],
     [None, None, "Recent violence", "ACLED deaths, last 4 weeks", "Observed", "log, 2,000 deaths → 1 (0.30)"],
     [None, "**Trade at stake**", None, "IMF IMTS, China's trade, last 12 months", "Observed",
      "log, relative to largest partner"],
     ["**Reported beside the score** (transmission channel and trade outlook)", "Hormuz exposure", None,
      "China's 2024 energy imports × route dependence (EIA, IEA)", "Observed, predetermined",
      "share 0–1, low and high case"],
     [None, "Shipping-lane status", None, "IMF PortWatch transits vs pre-crisis normal", "Observed",
      "disrupted ≤ −50%, reduced ≤ −20%"],
     [None, "Trade outlook", None, "Shrunk persistence on IMF trade", "Forecast", "% change, 80% interval"]],
    [((0, 0), (3, 0)), ((0, 1), (2, 1)), ((3, 1), (3, 2)), ((4, 0), (6, 0)), ((4, 1), (4, 2)), ((5, 1), (5, 2)),
     ((6, 1), (6, 2))],
    [2.7, 2.7, 2.2, 3.8, 1.9, 2.9],
    note="Risk and severity are forecast-based scores; the weights in brackets are those of the severity score. The "
         "indicators beside the score are shown on the lanes panel, in the briefs and, for the trade outlook, in the "
         "trade module's output.")

doc.add_heading("3.1  Construction logic", level=2)
para("Six choices shape every indicator:")
bullet("**One quarter ahead.** All forecasts look 13 weeks ahead, long enough for an organisation to act and short "
       "enough to be forecastable.")
bullet("**Relative to each relationship's own normal.** Israel–Palestine is always far more conflictual than "
       "Oman–Qatar. Change indicators are measured against each dyad's past year, so a forecast or alert signals a "
       "departure from what is usual for that relationship, not a permanent difference in level.")
bullet("**Hostility measured by volume, not diluted by cooperation.** Surges are measured from hostile events only, "
       f"by volume, because humanitarian and diplomatic events can raise the average score during a war "
       f"({fig('fig10_gdelt')}).")
bullet("**Busiest relationships count most.** Country indicators average a country's dyads weighted by their "
       "activity, each dyad's average weekly event count over the past year.")
bullet("**Log scales and a geometric mean.** Deaths and trade span orders of magnitude; log scales keep a mid-sized "
       "partner from being dwarfed by Saudi Arabia or the United Arab Emirates, and the geometric mean lets neither "
       "severity nor trade dominate alone.")
bullet("**Parameters fixed in advance, never fitted to outcomes.** Score weights and scaling points are set so that "
       "each component spans its full range across the region; Hormuz route parameters are set from route data and "
       "never fitted to the 2026 trade outcomes, which would make the analysis circular.")

doc.add_heading("3.2  Conflict outlook: average tone and conflict surges", level=2)
para(
    "The first target is the **mean Goldstein score of a dyad over the next 13 weeks**: the average tone of a "
    "relationship, from −10 (most conflictual) to +10 (most cooperative). Read against the dyad's trailing 52-week "
    "mean, it says whether relations are expected to be better or worse than usual, and by how much. Forecast "
    "hostility averages these forecasts over a country's dyads, including its domestic dyad.")
para(
    "The second target serves early warning: the **conflict-surge risk**, the probability that in the next 13 weeks a "
    "pair of countries goes through a month far more hostile than is usual for that pair. Average tone is a poor "
    "early-warning signal, because cooperative events can mask fighting and a three-month average smooths away "
    "short flare-ups; organisations need to know how bad the coming quarter could get. The indicator is built in "
    "five steps.")
numbered("**Weekly conflict intensity.** For each dyad and week, the Goldstein scores of all *hostile* events "
         "(threats, sanctions, expulsions, attacks) are summed in absolute value and put on a log scale, "
         "log(1 + sum). Cooperative events are ignored, so aid or diplomacy cannot offset fighting, and volume counts: "
         "ten attacks weigh more than one.")
numbered("**The worst month ahead (the target).** Over the next 13 weeks, the highest four-week mean of that intensity. "
         "Four-week windows are used rather than single weeks because a single week of a quiet dyad can hold one or "
         "two events, so its extreme is mostly noise.")
numbered("**The dyad's normal.** Its typical quarterly peak over the past year: for each of the last 52 weeks, take the "
         "worst four-week stretch of the 13 weeks before it, and average these 52 values. Only data available on the "
         "forecast date are used, and every dyad is judged against itself: a jump that is routine for Israel–Iran is "
         "a large one for Oman–Qatar.")
numbered(f"**What counts as a surge.** A quarter whose worst month exceeds the normal by at least {SURGE_T:.2f} on the "
         f"log scale, which means a worst month roughly {math.exp(SURGE_T):.1f} times the dyad's usual peak. The cut-off "
         "is one number for all dyads: the largest tenth of gaps between the realised worst month and normal across "
         f"all 400 dyads in the training period, 2015–2022. On the test weeks {V22S['surge_rate']:.0%} of dyad-weeks "
         "were followed by a surge.")
numbered("**From forecast to probability and alerts.** The surge model (Section 4.3) forecasts the worst month; its "
         "forecast gap over the normal is converted into a probability by a logistic curve fitted on the 2023 "
         f"validation year: about {p_at(0):.0%} when the model expects no gap, 50% at a forecast gap of "
         f"{-SURGE_CAL['b'] / SURGE_CAL['a']:.2f}, and higher beyond. Each week the ten dyads with the highest "
         "probability are flagged as **surge alerts**.")
para(
    "Read this way, a surge probability of 74% for a pair means the model gives about a three-in-four chance that the "
    "pair's worst month in the coming quarter will clearly break out of its usual range. On held-out data, "
    f"{SURG['conflict_surge']['top10']:.0%} of alerts were followed by a surge, against the "
    f"{V22S['surge_rate']:.0%} base rate (Section 5.7). Because the baseline is each pair's own recent peak, the "
    "indicator measures a *new flare-up*, not absolute danger: a quiet pair can surge from a low base, and a pair "
    "already at war can score low if fighting is expected to stay at its usual intensity. Sustained conflict shows up "
    "in forecast hostility and recent violence instead, which is why the severity score combines all three.")

doc.add_heading("3.3  Severity, trade at stake and the risk score", level=2)
bullet("**Conflict severity** = 100 × (0.40 × hostility + 0.30 × surge risk + 0.30 × violence). *Forecast hostility* "
       "is the activity-weighted mean of the Goldstein forecasts of all the country's dyads, including its domestic "
       "dyad, mapped linearly so that +1.5 or above scores 0 and −4.0 or below scores 1. *Forecast surge risk* is the "
       "activity-weighted mean conflict-surge probability of those dyads, with 0.6 or above scoring 1. *Recent "
       "violence* is the number of deaths recorded by ACLED in the last four weeks on a log scale, with 2,000 or more "
       "scoring 1. Without ACLED the two forecast parts are re-weighted 4:3.")
bullet("**Trade at stake** is China's goods trade with the country, exports plus imports, over the latest 12 months "
       "published by the IMF, on a log scale relative to the largest partner in the region: 100 × log(1 + trade) / "
       "log(1 + largest trade).")
para(
    "The weights and scaling points are design choices, set so that each component spans its full range across "
    "the region and fixed in advance rather than fitted, so that scores remain comparable across refreshes. They "
    "are published with the scores and can be changed in one place.")

doc.add_heading("3.4  Hormuz exposure", level=2)
para("For partner *i* and scenario *s* ∈ {low, high}:")
para("exposed(i, s) = crude(i) × dep_crude(i, s) + gas(i) × dep_gas(i, s)", align="center")
bullet("crude(i) and gas(i) are China's 2024 imports of HS 2709 and HS 2711 from *i*, in USD.")
bullet("dep is the share of the partner's exports that cannot bypass the strait.")
para("The regressor of Section 6.2 is the exposed value divided by China's total 2024 imports from *i*: a "
     "predetermined share in [0, 1]. Parameters are set from EIA descriptions of export infrastructure, checked on "
     f"2026-10-14 (Appendix A1; {tab('hormuz_params')}).")
table("hormuz_params", "Hormuz dependence parameters (share of exports that cannot bypass the strait).",
      ["Partner", "dep (crude), low – high", "Basis"],
      [["Iraq", "0.88 – 1.00", "All 2024 seaborne exports left from southern Gulf terminals (EIA, July 2025). The "
        "Iraq–Türkiye pipeline carried about 0.4 mb/d of northern crude before its 2023–2025 closure and cannot carry "
        "Basra grades."],
       ["Kuwait, Qatar, Bahrain", "1.00", "No outlet outside the strait; only Saudi Arabia and the UAE can reroute (EIA)."],
       ["UAE", "0.44 – 0.66", "1.8 mb/d Habshan–Fujairah pipeline against 3.2 mb/d of total exports (EIA, July 2026). "
        "LNG loads at Das Island, inside the Gulf: 1.00."],
       ["Saudi Arabia", "0.08 – 0.45", "East–West pipeline to Yanbu, about 5 mb/d available for exports; spare capacity "
        "estimates differ (IEA; EIA, June 2025)."],
       ["Iran", "0.88 – 1.00", "Goreh–Jask pipeline, about 0.3 mb/d effective. China's recorded crude imports from Iran "
        "are near zero, so the parameter does not affect exposure."],
       ["Oman", "0", "Terminals at Mina al-Fahal and Sohar (Gulf of Oman) and Duqm (Arabian Sea) lie outside the strait "
        "(EIA)."]],
      [3.0, 3.0, 10.0])
bullet("China's Hormuz-exposed energy imports in 2024 are **$79.7–107.1 bn**, or **3.1–4.1%** of its total imports "
       "($2,587 bn).")
bullet("Five partners carry material exposure: Iraq, Saudi Arabia, the UAE, Qatar and Kuwait.")
bullet("Oman, China's third-largest energy supplier in the region ($27.4 bn), is unexposed by route. Treating Oman as "
       "fully exposed would raise the upper bound by about $27 bn.")

doc.add_heading("3.5  Shipping-lane status", level=2)
para(
    "Weekly vessel transits through the Strait of Hormuz, Bab el-Mandeb and the Suez Canal over the last four weeks "
    "are compared with their 2019 – September 2023 average, before the Red Sea attacks; a lane is *disrupted* at −50% "
    "or worse and *reduced* at −20% or worse. For Hormuz the monitor also reports China's trade with partners whose "
    "oil exports depend on the strait, weighted by the share of each partner's crude exports that cannot bypass it "
    "(high case). In the trade module, a **Hormuz flag** marks exposed partners when tanker transits over the last "
    "four weeks fall below half of the 2019–2025 average; the flag does not change the trade forecast.")

# ============================================================== 4
doc.add_heading("4  Data, coding and the analysis process", level=1)
doc.add_heading("4.1  Event data and their coding", level=2)
para(
    "The conflict models use one source, the Global Database of Events, Language and Tone (GDELT 2.0; Leetaru and "
    "Schrodt 2013), which machine-codes news reports into events with a source actor, a target actor, a CAMEO event "
    "type (Gerner et al. 2002) and a Goldstein score from −10 (most conflictual) to +10 (most cooperative; Goldstein "
    "1992). We query GDELT's public BigQuery dataset and aggregate events to **directed dyad-weeks** "
    "(Monday–Sunday). The forecasting targets are the 400 ordered pairs of 20 MENA states, including each "
    "state's domestic dyad: the United Arab Emirates, Bahrain, Cyprus, Djibouti, Egypt, Iran, Iraq, Israel, "
    "Jordan, Kuwait, Lebanon, Libya, Oman, Palestine, Qatar, Saudi Arabia, Sudan, Syria, Turkey and Yemen, "
    f"observed over {d['weeks']} weeks ({d['first_week']} to {d['last_week']}). The cross-system model also "
    "sees **1,559 dyads linking these states to the rest of the world**: every MENA–world dyad with events in "
    "at least half of all weeks. These are inputs and training examples only; evaluation is always the 400 "
    f"MENA dyads. Release v2.2 additionally trains on the {V22S['world_dyads']:,} dyads between countries outside "
    "the region that are active in at least half of all weeks (same instrument, same weekly aggregation), as "
    "extra training examples only.")
para(
    "Inputs for each week are the log counts of the 20 CAMEO root event types, the week's mean Goldstein score, and "
    "three long-memory features: the dyad's trailing 26- and 52-week mean Goldstein and its 52-week activity level "
    f"({tab('gdelt_spec')}). The surge model adds weekly conflict intensity and summed Goldstein, each with its "
    "52-week mean.")
table("gdelt_spec", "Data and model specification of the conflict models (release v2.2).", ["Item", "Specification"],
      [["Source", "GDELT 2.0 events, public BigQuery dataset gdelt-bq.gdeltv2.events"],
       ["Forecast targets", "400 directed MENA dyad-weeks, incl. domestic"],
       ["System (tokens per week)", "400 MENA dyads + 1,559 MENA–world dyads = 1,959"],
       ["Extra training data (v2.2)", f"{V22S['world_dyads']:,} non-MENA dyads, in three extra 1,959-token "
                                      "sequences per training week; never evaluated"],
       ["Period", f"{d['first_week']} to {d['last_week']} ({d['weeks']} weeks)"],
       ["Targets", "Mean Goldstein score over the next 13 weeks; worst four-week conflict intensity over the next "
                   "13 weeks (surge model)"],
       ["Inputs per week", "log CAMEO root counts (20), mean Goldstein, trailing 26/52-week mean Goldstein, "
                           "52-week activity (24 features)"],
       ["Look-back", "13 weeks directly; up to 64 weeks through long-memory features"],
       ["Model size", f"{A22['params']:,} parameters per model; 5-model seed ensemble"]], [4.0, 12.0])
para(
    "The coding has known biases. GDELT counts track media attention as much as events, coding errors and "
    f"duplicates are common, and the Goldstein average mixes very different events. {fig('fig10_gdelt')} shows a "
    "direct consequence: after 7 October 2023 the Israel → Palestine mean Goldstein score *rose*, because the war "
    "generated many statements, appeals and humanitarian events with positive scores, so averaged event scores "
    "can understate a crisis precisely when it is most severe. This is why the surge indicator counts hostile events "
    "only, by volume (Section 3.2).")
figure("fig10_gdelt", "GDELT understates the October 2023 crisis for Israel → Palestine. (a) Weekly mean Goldstein "
       "score. (b) Shares of cooperative (CAMEO 01–09) and violent (CAMEO 18–20) events.", 15.5)
para(
    "For the trade analysis, conflict is measured from the same GDELT events aggregated by event location "
    "(ActionGeo, FIPS codes mapped to ISO3) to country × month, 2017-01 to 2026-10. The main measure is the share of "
    "events in CAMEO root codes 15, 18, 19 and 20: military posture, assault, fight and mass violence. A share rather "
    "than a count removes the drift in GDELT's media coverage. The **Gulf shock** is the mean of this share across "
    "the eight Hormuz littoral states (IRN, IRQ, KWT, SAU, ARE, BHR, QAT, OMN), standardised over months.")

doc.add_heading("4.2  Trade, shipping and other data", level=2)
para(
    "**Trade.** China's monthly bilateral goods trade comes from the IMF International Merchandise Trade Statistics "
    "(IMTS), China-reported: exports FOB, imports CIF; 20 MENA partners plus the United States and the World "
    "aggregate; 2017-01 to 2026-06, 9,681 observations. Comtrade's China monthly data end in December 2024 and cannot "
    "cover the 2026 events, so UN Comtrade is used only for the commodity composition of China's 2024 imports from "
    "each partner: crude oil (HS 2709) and petroleum gases (HS 2711). The data were checked as follows:")
bullet("*Units.* IMF values are full US dollars. China's 2024 totals in the IMF data ($3,580 bn exports, $2,587 bn "
       "imports) match the customs (GACS) headline figures **[verify GACS figures against the January 2025 release]**.")
bullet("*Missing months.* Import values are missing for Palestine (13 months), Djibouti (2) and Libya (1). They are "
       "treated as unknown, never as zero.")
bullet("*Mirror data.* The partner-reported series are not independent of China's figures for six partners: IRQ, "
       "ARE, IRN, DJI, YEM and LBY. For these, the IMF reports estimates (status \"e\") at a constant 1.06 ratio to "
       "China's data, a standard CIF/FOB adjustment. Kuwait and Oman report almost no oil exports to China. A "
       "robustness check that replaces China-reported with partner-reported data is therefore not informative for the "
       "exposed partners, and we do not use one.")
bullet("*Coding.* Palestine is coded WBG in IMF data and mapped to PSE (Appendix A2).")
para(
    "**Chokepoint transits.** IMF PortWatch reports daily vessel transits and cargo capacity through 28 maritime "
    "chokepoints, from 2019-01-01. We use the Strait of Hormuz, Bab el-Mandeb, the Suez Canal, the Cape of Good Hope "
    "and the Strait of Malacca. PortWatch serves as descriptive evidence, as the lane status and as the Hormuz flag; "
    "it is not yet a model input.")
para(
    f"**Recent violence, news and maps.** {tab('external')} lists every external source used by the models and the "
    "monitor, including ACLED for recent violence and news coverage used as cited context in the briefs.")
table("external", "External data used by the forecasting models, the trade module and the risk monitor.",
      ["Source", "Provider and access", "Used for", "Refresh"],
      [["GDELT 2.0 event data", "GDELT Project; public BigQuery dataset gdelt-bq.gdeltv2.events",
        "All conflict-model inputs and targets (400 MENA dyads, 1,559 MENA–world links, 5,122 world dyads for "
        "training); country-month conflict index of the trade module", "Every 2 weeks"],
       ["ACLED", "Armed Conflict Location & Event Data; API (registered account)",
        "Recent violence in severity; event map; context in briefs", "Weekly"],
       ["IMF International Trade in Goods (IMTS)", "IMF; SDMX 3.0 API",
        "China's trade with each country: trade at stake, regressions, trade forecast", "Weekly (published with a "
        "3–4 month lag)"],
       ["UN Comtrade", "United Nations; API", "Commodity composition of China's 2024 imports (HS 2709, 2711) for "
        "Hormuz exposure", "Static (2024)"],
       ["IMF PortWatch", "IMF and University of Oxford; public ArcGIS API",
        "Daily vessel transits through Hormuz, Bab el-Mandeb, Suez, Cape of Good Hope, Malacca", "Weekly"],
       ["Strait of Hormuz dependence", "Compiled from EIA and IEA route data (data/hormuz)",
        "Share of each Gulf exporter's crude and gas that cannot bypass Hormuz", "When sources change"],
       ["The Guardian Open Platform; New York Times Article Search", "Guardian Media Group; The New York Times; APIs",
        "Recent coverage of each country and China for the briefs", "Weekly"],
       ["DeepSeek API (deepseek-chat)", "DeepSeek; API", "Writing the bilingual briefs from the inputs above",
        "Weekly"],
       ["World Bank WDI", "World Bank; API", "Trade exposure in the earlier risk-index module", "Every 2 weeks"],
       ["Natural Earth via world-atlas", "Natural Earth (public domain)", "Country boundaries for the maps", "Static"]],
      [4.0, 4.4, 5.6, 2.4],
      note="News articles are used as cited context only; they are not model inputs.")

doc.add_heading("4.3  The conflict forecasting model", level=2)
para(
    "The transformer (Vaswani et al. 2017) was designed for sequences of words, but its central mechanism, "
    "self-attention, is defined over any *set* of tokens. Each token emits a query, a key and a value; every "
    "token then updates itself with a weighted average of all other tokens' values, with weights computed from "
    "query–key similarity. Three properties make it a natural way to represent a system of states in a "
    "forecasting model:")
bullet("**Every unit sees every other unit.** Attention connects all pairs of tokens in one step, so the forecast "
       "for any relationship can depend on any other relationship in the system.")
bullet("**The weights are learned and state-dependent.** Which relationships matter is not fixed in advance by the "
       "analyst, and can change from week to week, for example when a Red Sea crisis makes Israel–Yemen relevant "
       "to Iran–Gulf relations.")
bullet("**No arbitrary ordering.** Without positional encodings, self-attention treats its input as an unordered "
       "set, which matches a system of states better than any fixed ordering of countries.")
para(
    "Each **directed country pair (dyad)** is therefore a token, and the set of dyads in a week is the sequence, "
    "so one forward pass forecasts the whole system jointly. A temporal transformer first summarises each "
    "dyad's own recent history, so the model reasons about both *time* and *system*. We test three expectations:")
bullet("**E1 (benchmark).** Under a strict out-of-sample protocol, the transformer forecasts more accurately than "
       "naive forecasts and a competitive non-deep-learning model given the same information.")
bullet("**E2 (system context).** Attention across the system improves forecasts over modelling each dyad from its "
       "own history alone.")
bullet("**E3 (scaling).** The benefit of the approach grows with the amount of training data, so that larger "
       "systems and longer histories are a credible path to better forecasts.")
para(
    "The 'system' can be drawn at two scales. The **regional** system is the 400 dyads among the 20 MENA states. "
    "The **cross-system** view adds the region's relations with the rest of the world (Israel–United States, "
    "Iran–Russia, Turkey–Germany and so on), on the view that outside powers are part of the regional system. "
    "We test both; the release uses the second.")
para(
    f"{fig('fig2_architecture')} shows the architecture. Each dyad's 13-week input window is projected to "
    "64-dimensional tokens with a learned temporal position embedding and passed through two **temporal transformer** "
    "blocks, in which the dyad attends only over its own weeks. The representation of the latest week is "
    "concatenated with two **country embeddings**, learned 8-dimensional vectors for the source and target state "
    "from one shared table, so that 'Iran' means the same on either side of a dyad while direction is preserved by "
    "position. The resulting dyad tokens, 1,959 per week, then pass through two **cross-system transformer** "
    "blocks in which every dyad attends to every other. All blocks use four-head attention, SwiGLU feed-forward "
    "layers (Shazeer 2020), residual connections and layer normalisation.")
para(
    "The output is **residual**: the network produces a correction that is added to the dyad's trailing 52-week "
    "mean, so the model starts from the strongest naive forecast and learns when and how far the coming quarter "
    "will depart from a relationship's normal. Forecasts are read off the 400 MENA tokens. The release averages "
    "five identically configured models trained from different random seeds. The conflict-surge model uses the same "
    "architecture and horizon with the surge target of Section 3.2; it weights each dyad's error by its historical "
    "volatility, so that the many stable relationships do not teach it to stay flat, and its score is calibrated on "
    "validation into a probability.")
figure("fig2_architecture", "Architecture of release v2.2. Shaded blocks are the two attention stages: temporal "
       "attention within each dyad's own history, then cross-system attention across all 1,959 dyads of the "
       "week. During training, three further sequences of dyads from outside the region pass through the same "
       "blocks. Tensor shapes in brackets.", 16)

doc.add_heading("4.4  Evaluation protocol", level=2)
para(
    "Forecasting results are easy to inflate, and an earlier internal version of this model was, so the protocol "
    f"is deliberately strict ({fig('fig3_timeline')}). Data are split chronologically into training (May 2015 – "
    "October 2022), validation (February – December 2023) and test (March 2024 – June 2026) periods. Because each "
    "target averages the 13 weeks *after* its row, 13-week embargoes separate the periods; otherwise training targets "
    "would overlap the evaluation periods. **All design choices were made on validation loss alone**; the test "
    "period was used only to report results. The test period is long (119 weeks) and difficult, covering the "
    "Gaza war, the Red Sea crisis and the 2026 Israel–Iran escalation.")
figure("fig3_timeline", "Chronological evaluation design with 13-week embargoes between periods.", 15.5)
para(
    "Besides R² and mean absolute error, which reward getting each relationship's *level* right, we report "
    "three measures of *change*, which matter more for horizon scanning: **direction accuracy** (whether the "
    "forecast correctly says the next quarter will be above or below the dyad's 52-week normal), **change "
    "correlation** (whether the size of the forecast departure from normal tracks the actual one) and "
    "**escalation AUC** (how well the forecast ranks dyad-weeks whose next quarter ends at least one Goldstein "
    "point below normal). Benchmarks are naive forecasts (trailing means); **linear regression**, both on the "
    "same raw weekly inputs the transformer receives and on hand-built history features; a **competing "
    "algorithm**, gradient boosting (HistGradientBoosting) on the same hand-built features, i.e. a strong "
    "row-by-row model of the kind used in operational systems; and earlier versions of the transformer.")
para(
    "The hand-built feature set deserves a note. Its 44 features (trailing 4-, 13-, 26- and 52-week means and "
    "volatility, momentum, CAMEO conflict and cooperation shares, the reverse dyad's history, both countries' "
    "recent context and a region-wide average) were designed during this project, *after* studying the data and "
    "seeing which summaries predict well, for example that a 52-week mean outperforms a 13-week one. Both the "
    "linear and the gradient-boosting benchmarks therefore benefit from hindsight that a forecaster facing a new "
    "region or data source would not have. The transformer receives only the raw weekly event counts and "
    "Goldstein scores (plus three long-memory averages) and learns which summaries matter. The comparison is "
    "therefore deliberately favourable to the benchmarks: they are given the product of the analyst's feature "
    "engineering, while the transformer must discover it.")

doc.add_heading("4.5  Trade analysis design", level=2)
para("**Regression.** For partner *i* and month *t* we estimate two specifications:")
bullet("(A) own conflict: y(i,t) = β c(i, t−k) + α(i) + γ(t) + ε(i,t)")
bullet("(B) Gulf shock × exposure: y(i,t) = β gulf(t−k) × exposure(i) + α(i) + γ(t) + ε(i,t)")
para(
    "y is the 12-month log change of China's imports from (or exports to) *i*, which removes seasonality and level "
    "differences; c is the standardised conflict share of partner *i*. Partner and month fixed effects absorb the "
    "main effects of exposure and of the shock, so identification comes from the interaction: whether partners with "
    "higher predetermined exposure fare worse in months of high Gulf conflict. This is a continuous-treatment "
    "difference-in-differences, or shift-share design (Borusyak, Hull and Jaravel 2022). Each lag k = 0…3 is "
    "estimated separately, and all four lags are also estimated jointly; because the Gulf shock is autocorrelated, "
    "single-lag estimates partly proxy for one another, so the **cumulative effect** (the sum of the joint lag "
    "coefficients) is the main quantity. The sample runs 2018-01 to 2026-06: 21 partners, 102 months, 2,117 import "
    "observations.")
para(
    "**Inference.** The shock is common to all partners in a month, so the effective number of independent "
    "observations is close to the number of months, and only five partners carry material exposure. We report four "
    "kinds of inference:")
numbered("two-way clustered standard errors (partner and month; Cameron, Gelbach and Miller 2011), with eigenvalue "
         "clipping when the variance matrix is not positive semi-definite;")
numbered("month-clustered standard errors;")
numbered("Driscoll–Kraay standard errors (three lags; Driscoll and Kraay 1998);")
numbered("randomisation inference: exposure is reassigned across partners 999 times.")
para(
    "The permutation p-value is the most credible of the four, given the small number of treated units. On a "
    "simulated panel with a planted lag-1 effect of −0.3, the estimator recovers −0.29; clustering by partner alone, "
    "as in an earlier draft of this analysis, produced spuriously small p-values.")
para(
    "**Trade forecasting.** At the end of each week, for each of the 20 MENA partners and each flow, we forecast the "
    "year-on-year log change of China's trade over the next 13 weeks:")
para("y = log(trade over days [O, O+91)) − log(trade over the same days one year earlier)", align="center")
bullet("O is the forecast origin; exp(y) − 1 is the % change. Monthly values are spread evenly over days, and the "
       "year-on-year form removes Chinese New Year seasonality. Flows below $100 m in the last published 12 months "
       "are not graded.")
bullet("A trade month is used only after it would have been published: 100 days after month end for the IMF data "
       "that the automated pipeline uses (on 2026-10-14 the IMF's latest month was June 2026); 28 days for China "
       "customs releases.")
para(
    "Two kinds of model are compared. The first mirrors the conflict transformer, with countries as tokens (45,714 "
    "parameters: model width 32, four heads, SwiGLU feed-forward layers and an eight-dimensional country embedding). "
    "A temporal transformer lets each partner attend over its own last 13 weeks of 25 inputs (weekly GDELT conflict "
    "measures, UCDP deaths, monthly location-based GDELT, published trade growth, China's world trade growth, data "
    "staleness and season); a contextual transformer lets each partner attend to the other 19 in the same week. It "
    "is trained as five-seed ensembles with early stopping on a validation year (2023-10 to 2024-09) and tested on "
    "2025-01 to 2026-03 with 13-week embargoes. The second is **shrunk persistence**:")
para("ŷ = k_flow × (latest published 3-month year-on-year log change)", align="center")
para(
    "where k is a single least-squares slope through the origin for each flow, common to all partners. An 80% "
    "interval adds the 10th and 90th percentiles of past out-of-sample errors. In the walk-forward backtest, k is "
    "refitted each year on origins whose target window ended before that year began; then the year 2020–2026 is "
    "scored.")

doc.add_heading("4.6  Continuous scanning: the refresh cycle", level=2)
para(
    "The analysis runs as a scanning cycle without manual intervention. Every second week a scheduled cloud job "
    "rebuilds the weekly GDELT panel through the last complete week, runs the release models on all 400 dyads and "
    "writes the forecasts to BigQuery. A second job refreshes the IMF trade and PortWatch shipping series every week. "
    "A third weekly job, run after both, recomputes the indicators of Section 3 for every country, pulls the last 28 "
    "days of ACLED events and the last 90 days of relevant news, writes the briefs, and publishes a single data "
    "bundle (scores, dyad forecasts, histories, recent events, shipping lanes, briefs) to cloud storage. The monitor "
    "reads the latest bundle on every visit, so a refresh reaches users without redeploying the site, and the date "
    "of the data is shown on every page. The trade regressions and the trade forecast are rerun after each trade "
    "data refresh.")

# ============================================================== 5
doc.add_heading("5  Main findings (I): forecasting conflict", level=1)
doc.add_heading("5.1  Benchmark performance", level=2)
para(
    f"{tab('bench')} and {fig('fig4_benchmark')} report test performance. Release v2.2 reaches R² = {f3(V22['r2'])} "
    f"and MAE = {f3(V22['mae'])} Goldstein points. Given the same raw weekly inputs, a linear regression reaches only "
    f"{f3(OLS1['r2'])}: the transformer's ability to learn from raw history roughly doubles the explained "
    f"variance. It is also ahead of every naive forecast, of the regional-attention v2.0 ({f3(V20['r2'])}) and of "
    f"gradient boosting on hand-built features ({f3(GBM['r2'])}), including on all three change measures "
    f"(direction {pct(V22['dir_acc'])} vs {pct(GBM['dir_acc'])}, change correlation {V22['change_corr']:.3f} vs "
    f"{GBM['change_corr']:.3f}, escalation AUC {V22['esc_auc']:.3f} vs {GBM['esc_auc']:.3f}).")
para(
    f"The comparison with the raw-input regression is the central one. Given the 44 hand-built features, the "
    f"same linear regression rises from {f3(OLS1['r2'])} to {f3(OLS44['r2'])}. The whole of that gain comes from "
    "feature engineering done with hindsight (Section 4.4). The transformer achieves it by learning from the raw "
    "series which summaries of the past matter: trained on the region alone (v2.1, R² "
    f"{f3(V21['r2'])}), it is already level with the engineered model (change correlation "
    f"{V21['change_corr']:.3f} against {OLS44['change_corr']:.3f}, escalation AUC {V21['esc_auc']:.3f} against "
    f"{OLS44['esc_auc']:.3f}), so a well-engineered classical model is matched without anyone having to engineer "
    "it. That property is what allows the approach to carry over to new regions, actors or data sources, where no "
    "tested feature set exists. Given more training data, the release moves ahead of the engineered model "
    f"(R² {f3(V22['r2'])} against {f3(OLS44['r2'])}, change correlation {V22['change_corr']:.3f} against "
    f"{OLS44['change_corr']:.3f}, escalation AUC {V22['esc_auc']:.3f} against {OLS44['esc_auc']:.3f}, direction "
    f"{pct(V22['dir_acc'])} against {pct(OLS44['dir_acc'])}), whereas the engineered features would have to be "
    "redesigned to gain from it. Beyond this, the transformer's specific advantages appear where Section 5.4 "
    "locates them: in periods when shocks propagate across the system, and increasingly as training data grow. "
    f"Relative to the trailing 52-week mean the release reduces squared error by {A22['variance']['mse_reduction']:.1%} "
    f"and is more accurate for {A22['variance']['share_dyads_better']:.0%} of the 400 dyads. The trailing 52-week "
    "mean is a demanding baseline: a smoothed quarterly target is largely persistent, and carrying a relationship's "
    "annual average forward already explains 60% of test variance.")
table("bench", "Test-set performance of the conflict models (400 MENA dyads × 119 weeks, March 2024 – June 2026).",
      ["Forecast", "R²", "MAE", "Direction", "Change corr.", "Escalation AUC"],
      [["Trailing 13-week mean", f3(B["trailing 13-week mean"]["r2"]), f3(B["trailing 13-week mean"]["mae"]),
        pct(L["trailing 13-week mean"]["dir_acc"]), f3(L["trailing 13-week mean"]["change_corr"]),
        f3(L["trailing 13-week mean"]["esc_auc"])],
       ["Trailing 52-week mean", f3(B["trailing 52-week mean"]["r2"]), f3(B["trailing 52-week mean"]["mae"]),
        "—", "—", "0.500"],
       ["Linear regression, raw weekly inputs (this week)", f3(OLS1["r2"]), f3(OLS1["mae"]), pct(OLS1["dir_acc"]),
        f3(OLS1["change_corr"]), f3(OLS1["esc_auc"])],
       ["Linear regression, 44 hand-built features*", f3(OLS44["r2"]), f3(OLS44["mae"]), pct(OLS44["dir_acc"]),
        f3(OLS44["change_corr"]), f3(OLS44["esc_auc"])],
       ["Rudimentary network (counts + Goldstein + embeddings)", f3(L["rudimentary: counts + goldstein + embeddings"]["r2"]),
        f3(L["rudimentary: counts + goldstein + embeddings"]["mae"]), pct(L["rudimentary: counts + goldstein + embeddings"]["dir_acc"]),
        f3(L["rudimentary: counts + goldstein + embeddings"]["change_corr"]), f3(L["rudimentary: counts + goldstein + embeddings"]["esc_auc"])],
       ["Gradient boosting, 44 hand-built features*", f3(GBM["r2"]), f3(GBM["mae"]), pct(GBM["dir_acc"]),
        f3(GBM["change_corr"]), f3(GBM["esc_auc"])],
       ["v2.0: regional attention (400 tokens)", f3(V20["r2"]), f3(V20["mae"]), pct(V20["dir_acc"]),
        f3(V20["change_corr"]), f3(V20["esc_auc"])],
       ["v2.1: cross-system attention (1,959 tokens)", f3(V21["r2"]), f3(V21["mae"]), pct(V21["dir_acc"]),
        f3(V21["change_corr"]), f3(V21["esc_auc"])],
       ["v2.2: v2.1 + world dyads as training data", f3(V22["r2"]), f3(V22["mae"]), pct(V22["dir_acc"]),
        f3(V22["change_corr"]), f3(V22["esc_auc"])]],
      [6.2, 1.6, 1.6, 2.0, 2.2, 2.4], bold_last=True,
      note="Direction and change measures are relative to each dyad's trailing 52-week mean, so they are undefined "
           "for that baseline. Persistence of the latest week scores R² −1.62. The rudimentary network sees only the "
           "current week (no history, no transformer). Linear regressions are fit on the training weeks only, as "
           "the transformer is. *Hand-built features were designed with hindsight about what predicts well "
           "(Section 4.4).")
figure("fig4_benchmark", "Test R² of the release models against naive forecasts, the earlier transformer recipe and "
       "gradient boosting. Orange: models using the 44 hand-built features. Dark blue: the current release. "
       "The axis starts at 0.30.", 15.5)

doc.add_heading("5.2  What produced the improvement", level=2)
pv = list(PATH.values())
para(
    f"{fig('fig5_path')} and {tab('path')} trace the path from the original v1.0 recipe to the release, adding one "
    f"change at a time. The original recipe, once evaluated honestly, scored {pv[0]:.3f}, below the trailing mean. "
    f"Stronger regularisation (+{pv[1] - pv[0]:.3f}), the residual formulation (+{pv[2] - pv[1]:.3f}) and the "
    f"long-memory inputs (+{pv[3] - pv[2]:.3f}) contributed similar amounts; averaging five seeds added "
    f"+{pv[4] - pv[3]:.3f} (v2.0), and extending attention to the region's relations with the rest of the world "
    f"added +{pv[5] - pv[4]:.3f} (v2.1). Training on dyads from outside the region then added +{pv[6] - pv[5]:.3f} "
    "(v2.2). No single change was sufficient; together they move the model from below the "
    "naive baseline to above gradient boosting. The regularisation gain reflects how quickly the original recipe "
    "overfitted, with its best validation epoch the eighth.")
figure("fig5_path", "Test R² as each change is added (single seed until the five-seed ensembles). Dashed lines: "
       "trailing 52-week mean and gradient boosting.", 15.5)
E = {e["tag"]: e for e in (json.loads(l) for l in open("logs/goldstein_experiments.jsonl"))}
rows = [["v1.0 recipe (dropout 0.1, learning rate 1e-3)", "—", "0.556"]]
for label, tag in [("+ regularisation (dropout 0.2, lr 3e-4, halving)", "lb13_do0.2_lr0.0003_plat_s42"),
                   ("+ residual on 52-week mean", "lb13_res52_do0.2_lr0.0003_plat_s42"),
                   ("+ long-memory inputs", "lb13_res52_long_do0.2_lr0.0003_plat_s42")]:
    rows.append([label, f"{E[tag]['va_mse']:.3f}", f3(E[tag]["te_r2"])])
rows += [["v2.0: + 5-seed ensemble (regional attention)", "0.874", f3(V20["r2"])],
         ["v2.1: + 1,559 MENA–world dyads, cross-system attention, 5 seeds", f"{R['v21_val_mse']:.3f}", f3(V21["r2"])],
         [f"v2.2: + {V22S['world_dyads']:,} world dyads as training data, 5 seeds", f"{V22['val_mse']:.3f}",
          f3(V22["r2"])]]
table("path", "Configurations on the path to the release. Selection used validation MSE only; test R² for reference.",
      ["Configuration", "Validation MSE", "Test R²"], rows, [10.6, 2.7, 2.7], bold_last=True)

doc.add_heading("5.3  What limits the model", level=2)
para(
    "Before the release was fixed we asked why forecasts of *change* remain modest: around 61% direction accuracy "
    "for every model, including the transformers. Three diagnostics come first. Weeks with no events are "
    "recorded as zero, but changes in the number of active weeks explain only 1% of the variance of quarterly "
    "change; sampling noise from low-volume weeks accounts for about 7%; and cooperative events masking conflict "
    f"({fig('fig10_gdelt')}) is real but partial: in weeks where violent events doubled, the mean Goldstein score "
    "still fell by two points on average, but rose in 12% of cases. The target, in other words, mostly measures what "
    "it should. An honest re-run of the original modelling ladder also locates the gains: a rudimentary network "
    f"that sees only the current week scores R² {L['rudimentary: CAMEO counts']['r2']:.2f}, country embeddings "
    f"raise it to {L['rudimentary: counts + goldstein + embeddings']['r2']:.2f}, and the transformer recipe to "
    "about 0.64. Most of that gain is in levels; on change, the rudimentary network is already within about "
    "0.06 of the transformer's change correlation.")
para(
    f"We then tested whether more information would help ({tab('experiments')}), one experiment at a time, each "
    "against the release recipe on the same rows. Measuring conflict only, by summing the Goldstein scores of "
    "negative events, did not make change more predictable. Its event-weighted average even missed 7 October 2023, "
    "because conflict events are always roughly equally severe and escalation shows up as *volume*. Adding all 69 "
    "fields of the GDELT table (tone, media volume, conflict shares), country-level ACLED event data "
    "(Raleigh et al. 2010) or news text from the Guardian, encoded with a pretrained sentence model, made "
    "forecasts *worse*: weak inputs give a small model more ways to overfit, and a direct linear check found "
    "almost no signal about next-quarter change in either ACLED or the news features. Older GDELT 1.0 history "
    "(2008–2015) did not help either, because it is a different instrument with a different source base. "
    "**What did help was more training examples from the same instrument**: adding MENA–world dyads as training "
    "examples improved validation and test results consistently. The model is limited by data, not by the "
    "richness of its inputs, which is what motivates the scaling analysis below. We then tested the conclusion "
    f"directly: training the release model on {V22S['world_dyads']:,} additional dyads between countries outside "
    "the region, which the MENA forecasts never attend to, lowered validation error from "
    f"{R['v21_val_mse']:.3f} to {V22['val_mse']:.3f} and raised test R² from {f3(V21['r2'])} to {f3(V22['r2'])}, "
    f"with every one of the five seeds ({min(V22['seed_r2']):.3f}–{max(V22['seed_r2']):.3f}) above the v2.1 "
    "ensemble. The gain comes from learning how relationships evolve in general, not from more context at "
    "forecast time. This is release v2.2.")
trows = [[e["name"], f"{e['val_mse']:.3f}", f3(e["r2"]), pct(e["dir_acc"]), f3(e["change_corr"]), f3(e["esc_auc"])]
         for e in R["experiments"]]
table("experiments", "Data and model experiments (single seed 42 unless marked; same test rows). Lower validation "
      "MSE is better.",
      ["Experiment", "Val. MSE", "Test R²", "Direction", "Change corr.", "Esc. AUC"], trows,
      [6.4, 1.7, 1.7, 1.9, 2.1, 2.2],
      note="Conflict-only target (log sum of negative Goldstein scores; reported on its own scale): change correlation "
           f"{R['conflict_target']['change_corr']:.3f}, escalation AUC {R['conflict_target']['esc_auc']:.3f}, against "
           "0.356 and 0.648 for a simple 13-week momentum baseline on that target.")

doc.add_heading("5.4  When does system attention help?", level=2)
para(
    "Attention across the system is the central idea of the conflict model, so we tested it from four directions. "
    f"**First, a regional ablation** ({fig('fig6_ablation')}) retrains the v2.0 configuration without cross-dyad "
    "attention, without country embeddings, and without both. On validation the full model is best and each removal "
    f"hurts (MSE {AB['full']['val_mse_ensemble']:.3f} full, {AB['no_crossdyad']['val_mse_ensemble']:.3f} without "
    f"attention, {AB['neither']['val_mse_ensemble']:.3f} without both). On the 2024–2026 test period, however, "
    f"country embeddings help ({AB['full']['r2']:.3f} vs {AB['no_embedding']['r2']:.3f} without them) while "
    f"cross-dyad attention does not ({AB['no_crossdyad']['r2']:.3f} without it). The result is mixed, and "
    "on its own would not settle E2.")
figure("fig6_ablation", "Regional ablation (v2.0 configuration, three seeds each). Dots are seeds, diamonds their "
       "average.", 15.5)
bi, bc = BT["independent"], BT["cross-system"]
para(
    f"**Second, a rolling backtest** ({fig('fig13_backtest')}) repeats the comparison on four test periods, each with "
    "its own training and validation window. Attention loses in the two early folds, which have only 151 and 256 "
    f"training weeks (change correlation {bc['change_corr'][0]:.2f} vs {bi['change_corr'][0]:.2f}, and "
    f"{bc['change_corr'][1]:.2f} vs {bi['change_corr'][1]:.2f}). It wins clearly in 2023 "
    f"({bc['change_corr'][2]:.2f} vs {bi['change_corr'][2]:.2f}; escalation AUC {bc['esc_auc'][2]:.2f} vs "
    f"{bi['esc_auc'][2]:.2f}), the year in which the 7 October attack propagated through Israel, Lebanon, Yemen, "
    "Iran and the Gulf, exactly the kind of system-wide shock attention is designed for. 2023 also favours "
    "attention when it is the validation period of the final fold. In 2024–2026, a regional configuration the "
    "training data never contained, the two are level.")
figure("fig13_backtest", "Rolling backtest: independent-dyad versus cross-dyad-attention models on four test periods "
       "(three seeds each; labels give the number of training weeks).", 15.5)
ci, cc = LC["independent"], LC["cross-system"]
para(
    f"**Third, a learning curve** ({fig('fig12_learning_curve')}) trains both architectures on 12.5%, 25%, 50% and "
    "100% of the training weeks. Both improve with data, but the attention model improves faster: it starts worse "
    f"(validation MSE {cc['val'][0]:.3f} vs {ci['val'][0]:.3f} with 49 weeks) and ends clearly better "
    f"({cc['val'][-1]:.3f} vs {ci['val'][-1]:.3f} with 391 weeks), and its validation error is still falling "
    f"by about {abs(cc['val_change_per_doubling'][-1]):.3f} per doubling of data, against "
    f"{abs(ci['val_change_per_doubling'][-1]):.3f} for the independent model. On test change correlation it "
    f"closes from {cc['corr'][0]:.3f} vs {ci['corr'][0]:.3f} to parity. This crossover is the characteristic "
    "signature of a higher-capacity model that pays off as data grows (E3).")
figure("fig12_learning_curve", "Learning curves: models trained on 12.5–100% of the training weeks (MENA only, "
       "three seeds each).", 15.5)
para(
    f"**Fourth, the cross-system view** ({fig('fig11_scaling')}) adds MENA–world dyads to the system. At every size "
    "from 400 to 1,959 dyads, cross-system attention achieves lower validation error than the independent model given "
    "the same dyads, and the full 1,959-dyad system gives the lowest validation error of any configuration, which "
    "is why it was selected as the release. On test, results are flat across sizes within run-to-run noise "
    "(about ±0.01): adding *different* dyads is not the same as adding more of the task itself, and the test "
    "period's new regional configuration caps what extra training data can do.")
figure("fig11_scaling", "Adding MENA–world dyads to the system (three seeds each). Validation error is the "
       "selection criterion.", 15.5)
para(
    "Taken together, the evidence supports a qualified version of E2 and E3: **system-level attention adds value "
    "when there is enough data and when shocks propagate across the system, and its advantage grows with data**. "
    "At current data volumes it does not yet outperform simpler models in a regime the training data never "
    "covered.")

doc.add_heading("5.5  What the cross-system attention learns", level=2)
att = A22["attention"]
ext = att["external_share"]
h1, h2 = att["selective_heads"]
lift = lambda head, dyad: next(x["lift"] for x in att["received"][head] if x["dyad"] == dyad)
head_name = lambda k: f"block {int(k[15]) + 1}, head {int(k[-1]) + 1}"
para(
    "Attention weights are not explanations of a model's output (Jain and Wallace 2019), but they show where "
    "information flows. Averaged over heads, MENA forecasts spread attention across MENA–world dyads roughly in "
    f"proportion to their number ({min(ext.values()):.0%}–{max(ext.values()):.0%} of attention, against "
    f"{att['uniform_external_share']:.0%} if uniform). Two heads, however, are sharply selective "
    f"({fig('fig8_attention')}). The most selective ({head_name(h1)}) concentrates on a handful of relationships "
    f"around the region's central conflict: Palestine → Israel receives {lift(h1, 'PSE-ISR'):.0f} times its uniform "
    f"share, the United States → Israel {lift(h1, 'USA-ISR'):.0f} times, Israel → Palestine "
    f"{lift(h1, 'ISR-PSE'):.0f} times, Israel's domestic dyad {lift(h1, 'ISR-ISR'):.0f} times and Israel → Egypt "
    f"{lift(h1, 'ISR-EGY'):.0f} times, together with events GDELT attributes to Africa as a regional actor directed "
    f"at Egypt (AFR → EGY, {lift(h1, 'AFR-EGY'):.0f} times). The second ({head_name(h2)}) spreads over Israel, the "
    f"United States and Iran: Israel's domestic dyad ({lift(h2, 'ISR-ISR'):.1f} times uniform), the United States "
    f"with Israel and with Iran, Israel–Palestine in both directions, and Russia with Iran. Without being told which "
    "relationships matter, the model has learned to watch the region's central conflict and how outside powers "
    "engage with it and with Iran, which is precisely the systemic information that row-by-row models cannot use.")
figure("fig8_attention", "Dyads receiving the most attention from MENA forecasts in the two most selective "
       "cross-system heads of release v2.2 (seed 42), as a multiple of uniform attention (test-period average; note "
       "the different scales). Blue: MENA–world dyads; green: MENA–MENA dyads. AFR: GDELT's code for Africa as a "
       "regional actor.", 15.5)

doc.add_heading("5.6  Where the model adds information", level=2)
vv = A22["variance"]
ex = vv["example_reduction"]
para(
    f"Most of the variation the model is scored on lies *between* dyads: {vv['between_share']:.0%} of test "
    "variance is the difference in long-run level between relationships (Israel–Palestine is always far more "
    "conflictual than Oman–Qatar). Levels are what the trailing mean already captures, so the model's "
    "value-added is best measured on movements: its forecast departures from each dyad's normal correlate at "
    f"{vv['change_corr']:.2f} with the realised departures. {fig('fig7_examples')} illustrates this for four salient "
    f"relationships. The model improves on the 52-week mean for Turkey → Syria (−{ex['TUR-SYR']:.0%} squared "
    f"error), Israel → Palestine (−{ex['ISR-PSE']:.0%}) and Israel → Iran (−{ex['ISR-IRN']:.0%}), and is less "
    f"accurate for Saudi Arabia → Yemen (+{-ex['SAU-YEM']:.0%}), whose largest move in 2026 was an abrupt break. "
    "Anticipating such breaks a quarter ahead is the hardest part of the problem.")
figure("fig7_examples", "Forecast versus outcome for four dyads over the test period (release v2.2). Titles give "
       "the change in squared error relative to the trailing 52-week mean (negative = better).", 16)

doc.add_heading("5.7  Flagging conflict surges", level=2)
para(
    "Users of a risk product care less about how the average tone of a relationship will move than about "
    "whether it is heading into a crisis. An earlier companion model addressed this with a four-week direction "
    "task: will the next four weeks be more conflictual than the current week? It predicts that label well "
    f"(ROC-AUC {V22S['direction_own_auc']:.2f}), but the label mostly measures regression to the mean: a high "
    "escalation probability marks a dyad that has just had an unusually good week. Scored against the outcome "
    f"that matters, a surge in conflict over the next quarter, it ranks dyad-weeks no better than chance "
    f"(AUC {DIRN['conflict_surge']['auc']:.2f}). We therefore changed what the model forecasts, to the worst-month "
    "surge target of Section 3.2. Because the validation year, 2023, was unusually turbulent, the calibrated "
    f"probabilities run high in 2024–2026 (mean {V22S['surge_mean_prob_test']:.0%} against a realised "
    f"{V22S['surge_rate_test_all']:.0%}); the ranking, which drives the alerts, is unaffected, and the calibration "
    "will be refitted as data accrue.")
para(
    f"{tab('flags')} compares crisis flags on the test weeks shared by every model ({V22S['flag_weeks']} weeks). If "
    f"each week an analyst reviewed the ten dyads ranked highest, {SURG['conflict_surge']['top10']:.0%} of the surge "
    f"model's alerts would be followed by a surge, against {GOLD['conflict_surge']['top10']:.0%} for the v2.2 "
    f"Goldstein forecast and {DIRN['conflict_surge']['top10']:.0%} for the direction model "
    f"(base rate {V22S['surge_rate']:.0%}). Simple benchmarks are competitive on this target: conflict peaks are "
    f"persistent, so last quarter's peak already scores AUC {NAIV['conflict_surge']['auc']:.3f}, and a linear "
    f"regression on the hand-built features {LINF['conflict_surge']['auc']:.3f}, against "
    f"{SURG['conflict_surge']['auc']:.3f} for the transformer. On this new target the hand-built features, "
    "designed with hindsight for this region (Section 4.4), keep an edge in ranking; on the average-Goldstein "
    "target the transformer closed the same gap once it had more training data (Section 5.1), and the surge "
    "target is the natural next candidate for that. The surge model also departs from the normal more readily "
    f"than the average-Goldstein forecast ({fig('fig14_surge_isr')}): for Israel → Palestine its forecast of the "
    f"coming peak correlates {V22S['isr_pse']['ISR-PSE']['surge_corr']:.2f} with the actual change and moves "
    f"{V22S['isr_pse']['ISR-PSE']['surge_amp']:.0%} as much as reality, whereas the v2.2 forecast of average "
    f"Goldstein stays close to its normal ({V22S['isr_pse']['ISR-PSE']['gold_amp']:.0%}). Both still miss the "
    "sustained easing of 2025–2026, which a quarter-ahead model cannot see coming. Each target flags "
    "its own kind of crisis: the surge model is weaker on falls in average Goldstein "
    f"(AUC {SURG['gold_crisis']['auc']:.3f} against {GOLD['gold_crisis']['auc']:.3f} for the v2.2 Goldstein forecast), so the two "
    "models are complementary, and the release runs both.")
para(
    "Three alternatives did not improve on this. Forecasting the worst month of *average* Goldstein flagged "
    "Goldstein crises slightly better than the average-Goldstein forecast it was tested against but did not track "
    "conflict; a single network trained on both "
    "targets did both jobs slightly worse than two separate models; and the world training data that improves "
    "v2.2 left the surge model's ranking unchanged. The surge target is a rare, noisy quantity, and like the "
    "main model it is likely to benefit most from a larger, global training set.")
table("flags", "Crisis flags on the shared test weeks (400 MENA dyads). AUC ranks all dyad-weeks; top-10 is the share "
      "of each week's ten highest-scored dyads that experience the event in the next 13 weeks.",
      ["Flag", "Surge AUC", "Surge top-10", "Goldstein-crisis AUC", "Goldstein-crisis top-10"],
      [[name, f3(r["conflict_surge"]["auc"]), f"{r['conflict_surge']['top10']:.0%}", f3(r["gold_crisis"]["auc"]),
        f"{r['gold_crisis']['top10']:.0%}"] for name, r in FL.items()],
      [6.4, 1.9, 2.0, 2.6, 2.6], bold_last=True,
      note=f"Base rates: conflict surge {V22S['surge_rate']:.0%}, Goldstein crisis {V22S['gold_crisis_rate']:.0%} "
           "(the next quarter's worst four-week mean Goldstein falls below the dyad's typical quarterly low by more "
           "than the worst tenth of historical gaps). *Hand-built features as in Table "
           f"{TABS.index('bench') + 1}, plus conflict-intensity history; fit on training weeks.")
figure("fig14_surge_isr", "Israel → Palestine on the test weeks. (a) The v2.2 forecast of mean Goldstein stays near "
       "the dyad's normal. (b) The surge model's forecast of the coming quarter's peak conflict intensity departs "
       "from its normal more often, but also misses the sustained decline. Forecasts are plotted at the week they "
       "are made.", 16)

doc.add_heading("5.8  Validating outputs against the decision they serve", level=2)
para(
    "Elsewhere in the project the forecasts also feed a country-level risk index in which risk is the product of "
    "likelihood and impact; there the transformer supplies the likelihood component, a calibrated probability that "
    "a country's relations deteriorate materially below their own 52-week normal over the next quarter, which ranks "
    "countries with ROC-AUC 0.70 on held-out data with the v2.2 forecasts. One finding from that work concerns model "
    "outputs: the direction model's escalation probability, used naively as a likelihood, ranks countries *worse* "
    "than random (AUC 0.37), because a high value mostly flags a country that has just had an unusually good week. "
    "Outputs must be validated against the decision they serve, not only against their training target; this is why "
    "the live pipeline pairs the Goldstein forecast with the conflict-surge flag instead of the direction model.")

# ============================================================== 6
doc.add_heading("6  Main findings (II): how conflict reaches China's trade", level=1)
doc.add_heading("6.1  The 2026 Hormuz closure", level=2)
bullet("Violent events in the weekly GDELT panel rose from about 3,800 a week in January–February 2026 to **30,352 in "
       "the week of 2 March**, eight times the earlier level.")
bullet("In the monthly index, the Gulf shock reached **7.5 standard deviations** in March 2026. The largest value in "
       "the preceding nine years was 2.0 (January 2024).")
bullet("PortWatch shows Hormuz tanker transits averaging **50 a day in 2019–2025** and 37 a day in January–February "
       "2026. They fell below five a day on **2 March 2026** and averaged **2.5 a day** from then to October, and 0.8 a "
       "day in the most recent four weeks.")
figure("fig_trade_1_closure", "(a) Weekly tanker transits through the Strait of Hormuz, IMF PortWatch. (b) China's "
       "imports from Gulf partners, % change on the same month a year earlier. The dotted line marks 2 March 2026, "
       "when transits fell below five a day.", 16)
table("closure", "Effect of the closure on China's imports, April–June 2026 vs April–June 2025 (USD value).",
      ["Partner", "Change", "Implied energy dependence", "Assumed dep (low – high)"],
      [["Iraq", "−89%", "0.91", "0.88 – 1.00"],
       ["Qatar", "−88%", "0.96", "1.00"],
       ["Kuwait", "−83%", "0.93", "1.00"],
       ["UAE", "−75%", "1.01", "0.44 – 0.66"],
       ["Oman", "−20% (flat Apr–May, −76% in June)", "0.23", "0"],
       ["Saudi Arabia", "+12%", "< 0", "0.08 – 0.45"],
       ["World (China's total imports)", "+30%", "—", "—"]],
      [4.2, 4.6, 3.6, 3.6],
      note="Implied energy dependence is the observed fall divided by the partner's energy share of China's imports. "
           "It is a lower bound, because oil prices rose and values understate volumes.")
para("How this compares with the assumed route parameters of Section 3.4:")
bullet("**Consistent:** Iraq, Kuwait, Qatar and Saudi Arabia match the route parameters.")
bullet("**UAE:** the fall is larger than the route parameters imply. Fujairah capacity appears not to have been "
       "allocated to China-bound cargoes.")
bullet("**Oman:** the June drop points to indirect war-risk effects in the Gulf of Oman that a route parameter cannot "
       "capture.")
bullet("**Saudi Arabia:** EIA reports that attacks on the East–West pipeline in September 2026 temporarily halted the "
       "bypass. Saudi dependence is therefore time-varying.")

doc.add_heading("6.2  Gulf conflict and trade with exposed partners", level=2)
table("reg", "Cumulative effect of the Gulf shock × exposure, lags 0–3 jointly.",
      ["Specification", "β", "Two-way p", "Permutation p"],
      [["Imports, exposure high", "−0.446", "0.001", "0.004"],
       ["Imports, exposure low", "−0.491", "<0.001", "0.004"],
       ["Exports, exposure high", "−0.209", "0.006", "0.015"],
       ["Exports, exposure low", "−0.225", "0.004", "0.011"],
       ["Own conflict (A), imports", "−0.192", "0.43", "—"]],
      [6.4, 3.2, 3.2, 3.2])
para(
    "For imports at high exposure, the single-lag estimates are −0.28 (k = 0), −0.25 (k = 1), −0.31 (k = 2) and −0.13 "
    "(k = 3), with permutation p-values of 0.012, 0.011, 0.003 and 0.082. The cumulative effect is the change in "
    "12-month import growth for a fully exposed partner (exposure = 1) when the Gulf shock is one standard deviation "
    "higher for four consecutive months. As a check, the Gulf shock averaged about 3.2 standard deviations over "
    "March–June 2026; a fully exposed partner would then lose roughly 0.45 × 3.2 ≈ 1.4 log points, about −76%, "
    "against observed second-quarter falls of −83% to −89% for Iraq, Qatar and Kuwait. The linear specification "
    "captures most, but not all, of the collapse.")
bullet("**Own-country conflict has no detectable effect on trade** (specification A). Conflict matters for trade "
       "through the shipping route, not through the partner's own instability.")
bullet("**China's exports to exposed partners also fell**, by about half the import effect: inbound shipping to Gulf "
       "ports was disrupted too.")
table("robust", "Robustness: imports, high exposure, cumulative effect, lags 0–3.",
      ["Variant", "β", "Permutation p"],
      [["Baseline", "−0.446", "0.004"],
       ["Sample from 2023", "−0.395", "0.028"],
       ["Excluding the USA", "−0.441", "0.004"],
       ["Excluding SYR, SDN, YEM, LBY (war-affected)", "−0.382", "0.017"],
       ["Excluding IRN and ARE (sanctions, re-exports)", "−0.457", "0.005"],
       ["Import flows ≥ $100 m in 2024 only", "−0.379", "0.018"],
       ["Shock = material-conflict share", "−0.390", "0.111"],
       ["Shock includes ISR and YEM", "−0.326", "0.058"],
       ["Oman dependence = 1", "−0.375", "0.011"],
       ["Saudi dependence = 0.8", "−0.402", "0.006"],
       ["Energy share without route parameters", "−0.226", "0.087"],
       ["**Sample ending 2025-12 (no 2026 closure)**", "**−0.075**", "**0.739**"]],
      [9.0, 3.5, 3.5])
figure("fig_trade_2_regression", "Cumulative effect (lags 0–3) of Gulf conflict × Hormuz exposure across "
       "specifications, with 95% intervals from two-way clustered standard errors and permutation p-values. Orange: "
       "the sample ending before the 2026 closure.", 16)
para("Two results stand out.")
para(
    "**The effect is identified by one episode.** Without 2026, the coefficient is a small fraction of its full-sample "
    "value and indistinguishable from zero. Gulf conflict between 2018 and 2025 included the 2019 tanker attacks, the "
    "Red Sea crisis of 2023–24 and the June 2025 Israel–Iran war; none of these moved China's trade with exposed "
    "partners measurably. The estimate therefore describes what happens when the strait actually closes, not a "
    "general elasticity of trade to Gulf conflict.")
para(
    "**Route information matters.** Replacing route-based exposure with the raw energy share of imports roughly halves "
    "the coefficient and weakens its significance. Partners whose energy exports could bypass the strait (Saudi "
    "Arabia, Oman) behaved differently from those that could not. The results are not sensitive to the parameter "
    "revision of October 2026: with the earlier, secondary-source parameters the cumulative effects were −0.45 (high) "
    "and −0.55 (low).")

doc.add_heading("6.3  Forecasting China's trade one quarter ahead", level=2)
table("trade_models", "Trade transformer and benchmarks: test skill, 1 − MSE / MSE of \"no change\" (2025-01 to "
      "2026-03, 28-day lag).",
      ["Model", "Imports", "Exports", "Validation MSE"],
      [["Shrunk persistence (2 parameters)", "0.171", "**0.280**", "—"],
       ["LightGBM, same inputs", "0.032", "−0.042", "—"],
       ["Transformer, full", "0.083", "0.016", "0.133"],
       ["Transformer, no contextual attention", "0.151", "0.091", "0.149"],
       ["Transformer, no conflict inputs", "0.046", "−0.007", "0.132"],
       ["Transformer, residual on shrunk persistence", "**0.187**", "0.187", "0.149"]],
      [7.0, 3.0, 3.0, 3.0])
numbered("**The full transformer did not beat the two-parameter baseline.** Its validation advantage did not carry over "
         "to the test period.")
numbered("**Removing all conflict inputs left validation loss unchanged.** Conflict data added no validated forecasting "
         "skill, consistent with the zero pre-2026 regression effect.")
numbered("**No model anticipated the closure.** At the 2 March 2026 origin, after the violence spike was already in the "
         "inputs, the models forecast Iraqi, Kuwaiti and Qatari imports between −17% and +23% of a year earlier. The "
         "realised change was at least −78% (the target is capped at ±1.5 log points). The training history contained "
         "no sustained chokepoint closure; the models had learned that sharp falls in China's oil imports, which are "
         "usually cargo timing, revert within a quarter.")
para(
    "With about 220 training weeks (roughly 17 independent quarters), rankings among neural variants are unstable, so "
    "the trade transformer is kept as a research track and shrunk persistence is the operational forecast. Fitted on "
    "all labelled origins with the 100-day lag, **k = 0.236 for imports** and **k = 0.380 for exports**: about a "
    "quarter of a recent import swing, and about two-fifths of an export swing, persists into the next quarter. "
    "Imports, dominated by oil and gas cargoes, are lumpier and revert faster.")
table("backtest", "Operational trade forecast: walk-forward backtest, 2020–2026.",
      ["Information", "Imports skill", "Exports skill", "Imports R²", "Exports R²", "80% coverage (imp / exp)"],
      [["IMF, 100-day lag", "0.057", "0.110", "0.048", "0.062", "71% / 79%"],
       ["China customs, 28-day lag", "0.088", "0.199", "0.082", "0.155", "73% / 80%"]],
      [4.2, 2.2, 2.2, 2.2, 2.2, 3.0],
      note="R² is computed against the mean outcome of the evaluated rows, which no forecaster knows in advance. "
           "Skill is computed against \"no change\".")
figure("fig_trade_3_backtest", "Walk-forward skill of the operational trade forecast by year (IMF data, 100-day "
       "publication lag). Skill = 1 − MSE / MSE of forecasting no change.", 16)
table("by_year", "Operational trade forecast by year, IMF lag (imports / exports).",
      ["Year", "2020", "2021", "2022", "2023", "2024", "2025", "2026"],
      [["k", "0.26 / 0.48", "0.28 / 0.41", "0.27 / 0.40", "0.23 / 0.44", "0.21 / 0.36", "0.24 / 0.34", "0.24 / 0.35"],
       ["Skill", "0.13 / 0.02", "0.08 / 0.28", "0.00 / 0.22", "0.04 / −0.03", "0.07 / −0.03", "0.10 / 0.18",
        "−0.07 / 0.09"]],
      [1.4, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1])
bullet("**Small but positive overall skill**, and k is stable across years.")
bullet("**Data speed matters:** with customs data, export skill nearly doubles.")
bullet("**Import intervals are too narrow:** 71% coverage for a nominal 80%, because oil-cargo errors are heavy-tailed.")
bullet("**The closure breaks the rule:** in 2026 import skill turns negative and coverage falls to 50–60%. Persistence "
       "cannot know that a closure will continue.")

# ============================================================== 7
doc.add_heading("7  Application value: the risk monitor", level=1)
para(
    "The findings are put to use through a bilingual (English and Chinese) web-based risk monitor for organisations "
    "with trade at stake in the region, in particular Chinese exporters, energy buyers, contractors and shippers. Its "
    "design follows four principles. **Forecast and observation are kept apart:** everything produced by, or built "
    "on, the transformer carries one consistent visual marker, and observed data (recent violence, trade, shipping) "
    "are shown plainly, so users can tell the model's view of the next quarter from what has already happened. **The "
    "system view is kept:** a country's relationships are drawn on the map, so a user sees where pressure comes from, "
    "not only how much. **Decisions, not data:** every figure is accompanied by a plain statement of what it means "
    "for an organisation with interests in the country. **Calm by default:** the map shows only relationships "
    "forecast to turn hostile and those with a conflict-surge alert, so attention goes where it is needed "
    f"({fig('fig15_website')}).")
figure("fig15_website", "The risk monitor. (a) Regional overview: countries coloured by risk; the hovered country "
       "(Saudi Arabia) is lifted and its troubled relationships are drawn - orange for those forecast to turn "
       "hostile, glowing with a travelling ball where a conflict surge is flagged; rings mark the Strait of Hormuz, "
       "Bab el-Mandeb and the Suez Canal, coloured by shipping disruption. (b) Country page (Yemen): the risk score "
       "and its parts, the 13-week outlook against the past year, the relationships most at risk, the China "
       "exposure brief, and ACLED events of the last 28 days as pillars on the map.", 16)

doc.add_heading("7.1  What the monitor shows", level=2)
bullet("**Regional overview.** A three-dimensional map, coloured on a single red scale by risk or severity, is the "
       "background of the page. Hovering a country lifts it and draws its relationships: orange arcs for dyads "
       "whose forecast tone is hostile (below −0.25), deeper orange for more hostile, and, for any dyad with a "
       "conflict-surge alert, a glowing arc with a travelling ball whose colour shows the surge probability "
       "(yellow near 50%, orange near 60%, red at 75% or more). Calm, cooperative relationships are not drawn. "
       "A watchlist ranks the highest-risk countries, those forecast to deteriorate most against their own "
       "normal, and the disrupted shipping lanes.")
bullet("**Country page.** The risk score and its components, a one-year history of the country's relations with "
       "the 13-week forecast added, its most active relationships with their forecast tone, the trade at stake, "
       "recent violence (ACLED events of the last 28 days drawn as pillars on the map, height proportional to "
       "deaths) and a China exposure brief (Section 7.2).")
bullet("**Shipping lanes.** The lane status of Section 3.5 for Hormuz, Bab el-Mandeb and the Suez Canal, with the "
       "China trade at stake through Hormuz.")
bullet("**Data & model page and export.** A plain-language account of the data, the model and its limits, with "
       "the test results of Section 5; all outputs (country scores, the 400 dyad forecasts, shipping lanes, the "
       "ACLED events shown and the full data bundle) can be downloaded.")

doc.add_heading("7.2  Written briefs", level=2)
para(
    "Each refresh produces a regional brief and, for each country, an outlook, a summary of its trade relationship "
    "with China and two to four points on what could be at risk for Chinese entities, each naming who (exporters, "
    "crude buyers, contractors, shipping lines, staff on the ground) and how. They are written in English and "
    "Chinese by a large language model (DeepSeek, deepseek-chat) from a fixed set of inputs: the dashboard figures, "
    "the IMF trade series, the country's Hormuz dependence, the shipping lanes it borders, and recent articles "
    "mentioning both China and the country, retrieved from the Guardian and New York Times APIs over the previous "
    "90 days. The instructions require every figure to come from those inputs, named projects, companies or "
    "events to come from a cited article, and no causal claims the inputs do not state; without a relevant "
    "article the brief stays at the level of sectors. Sources are listed under each brief, briefs are labelled "
    "as machine-written, and if the model is unavailable the page falls back to a fixed template built from the "
    "same figures.")

doc.add_heading("7.3  Value for users", level=2)
para(
    "The monitor turns the two analyses into answers an enterprise can act on. The conflict models are current to "
    "the last complete week, while official trade data arrive three to four months late (Section 6.3), so the "
    "monitor's forecasts and shipping data lead what trade statistics can show. The risk score directs attention to "
    "countries where both conflict and trade are large; the relationship view shows where the pressure comes from; "
    "the lanes panel isolates the route channel, which Section 6 shows is how conflict actually reached China's trade "
    "in 2026; and the briefs translate the figures into who could be affected and how.")

# ============================================================== 8
doc.add_heading("8  Recommendations for enterprises", level=1)
doc.add_heading("8.1  Reading the conflict and trade signals together", level=2)
para(
    "The conflict forecasts and the trade forecast answer different questions with information of different age. "
    f"The conflict models are current through last week: the Goldstein forecast has {-V22['mse_vs_52']:.1%} lower MSE "
    f"than a 52-week mean, and {SURG['conflict_surge']['top10']:.0%} of the surge model's top-10 flags are followed "
    f"by a surge against a {V22S['surge_rate']:.0%} base rate. The trade forecast uses trade data three to four months "
    f"old, with 6–11% skill against \"no change\". {tab('decision')} sets out how to read them together.")
table("decision", "How the conflict and trade outputs are read together.",
      ["Conflict forecasts (13 weeks)", "Trade forecast (13 weeks)", "Reading"],
      [["Surge alert on, or Goldstein forecast below its 52-week normal, for dyads involving the partner",
        "Flat or positive", "Trade data (3–4 months old) cannot show it yet. For Hormuz-exposed partners, plan around "
        "the lower end of the trade interval."],
       ["Stable", "Sharp recent drop", "Probably not conflict-driven (prices, cargo timing). Expect partial reversion, "
        "as the baseline does."],
       ["Surge alert or deterioration", "Falling, Hormuz flag on", "Disruption likely to persist. Use the latest "
        "change or the interval's lower end."],
       ["Stable", "Stable", "Normal conditions. Use the point forecast."]],
      [4.6, 3.4, 8.0])

doc.add_heading("8.2  Recommendations", level=2)
bullet("**Watch the route, not only the partner.** Own-country conflict left no detectable trace in China's trade, "
       "while the closure of the Strait of Hormuz cut imports from partners without a bypass by four-fifths or more "
       "(Section 6). Energy buyers and traders dependent on Iraq, Kuwait, Qatar and the UAE should hold a closure "
       "scenario in their plans; the regression gives its scale, about −0.45 log points of import growth per standard "
       "deviation of sustained Gulf conflict for a fully exposed partner, to be used for scenarios rather than as a "
       "forecast.")
bullet("**Value supply that can bypass the strait, but monitor the bypass.** Saudi exports through the Red Sea rose "
       "during the closure, and Oman's terminals lie outside the strait; but the Saudi East–West pipeline was attacked "
       "in September 2026, and Oman's June 2026 drop shows indirect war-risk effects. Bypass capacity reduces "
       "dependence; it does not remove it.")
bullet("**Treat surge alerts as triggers for review, not as predictions.** Roughly one alert in two is followed by a "
       "surge, about three times the base rate. An alert on a relationship involving a country where the organisation "
       "operates is a signal to review contingency plans, security arrangements and exposure during the 13-week "
       "window, not proof that a crisis will occur.")
bullet("**Do not wait for trade statistics.** Official trade data arrive three to four months late, and no "
       "statistical trade model anticipated the 2026 collapse. Conflict forecasts and shipping transits are the "
       "leading signals; when they deteriorate while trade still looks normal, plan around the lower end of the trade "
       "interval (Table "
       f"{TABS.index('decision') + 1}).")
bullet("**Plan on intervals, and widen them for imports.** Import intervals cover 71% of outcomes rather than 80%, "
       "and fall to 50–60% during the closure; energy-import planning should allow for wider swings than the "
       "interval states.")
bullet("**Combine the monitor with analyst judgement on discrete crises.** The models anticipate gradual drifts "
       "better than abrupt breaks (Section 5.6). They are horizon-scanning tools that direct attention, to be read "
       "alongside expert assessment of specific events.")

doc.add_heading("8.3  The current reading", level=2)
top = SITE["countries"][:5]
lanes = {l["id"]: l for l in SITE.get("lanes") or []}
lane_txt = "; ".join(l["name"] + " " + f"{l['change']:+.0%}".replace("-", "−") for k in ("hormuz", "bab_el_mandeb", "suez", "cape_of_good_hope")
                     if (l := lanes.get(k)))
para(
    f"At the refresh for the week of {SITE['meta']['as_of_week']}, the highest country risks are "
    + ", ".join(f"{c['name']} ({c['risk']:.0f})" for c in top)
    + f", and regional risk is {SITE['region']['risk']:.0f} with {SITE['region']['alerts']} surge alerts. Shipping "
    f"transits over the last four weeks against their pre-crisis normal: {lane_txt}. For trade, the forecast for "
    f"12 October 2026 – 10 January 2027 uses trade data through June 2026; with Hormuz tanker transits at 2% of "
    f"normal, the Hormuz flag is on ({tab('current')}).")
table("current", "Current trade forecast for Hormuz-exposed partners (origin 12 October 2026).",
      ["Partner", "Latest 3-month change (imports)", "Import forecast", "80% interval"],
      [["Iraq", "−87%", "−38%", "−71% to +9%"],
       ["Kuwait", "−83%", "−34%", "−69% to +15%"],
       ["UAE", "−75%", "−28%", "−67% to +25%"],
       ["Saudi Arabia", "+12%", "+3%", "−52% to +79%"]],
      [3.6, 4.4, 3.6, 4.4])
para("The point forecasts assume partial reversion. While the strait remains closed, the lower end of the interval, or "
     "the latest observed change, is the relevant scenario.")

# ============================================================== 9
doc.add_heading("9  Necessary notes", level=1)
doc.add_heading("9.1  Limitations of the conflict models", level=2)
para("**Data volume.** The model is trained on 391 weeks of the regional system, and Section 5.3 showed it is "
     "limited by the number of training examples rather than by input richness. Seeds vary "
     f"({min(V22['seed_r2']):.3f}–{max(V22['seed_r2']):.3f} test R² across the five v2.2 seeds), which is why the release "
     "averages five.")
para("**Regime shifts.** System-level patterns learned from 2015–2022 transferred well to 2023 but not to the "
     "2024–2026 configuration of direct Israel–Iran exchanges and a wider Red Sea conflict. A model can only "
     "attend to relationships whose dynamics it has seen.")
para(f"**One input, with known biases.** GDELT counts track media attention as much as events, and averaged event "
     f"scores can understate a crisis precisely when it is most severe ({fig('fig10_gdelt')}); the surge indicator "
     "mitigates but does not remove this.")
para("**A short history from a single instrument.** GDELT 2.0 begins in 2015, so the model learns from about "
     "eleven years in which the region went through only a handful of system-wide crises. Earlier GDELT 1.0 data "
     "cannot simply be appended: it draws on a different set of sources and coding rules, and adding it made "
     "forecasts worse (Section 5.3). Because the model is limited by data rather than by architecture, this "
     "ceiling is set by the instrument, not by the method; the results reported here should be read as a lower "
     "bound on what the approach can achieve with a longer, consistently coded record.")
para("**Gradual shifts rather than sudden breaks.** The model anticipates gradual drifts better than abrupt "
     "breaks, and is best used as a horizon-scanning tool alongside analyst judgement on discrete crises.")
para("**Interpretability.** Attention patterns (Section 5.5) indicate where information flows but do not "
     "establish what drives a given forecast.")

doc.add_heading("9.2  Limitations of the trade module", level=2)
numbered("**One episode.** The exposure effect is identified by the 2026 closure alone. It is an estimate of what a "
         "closure does, not of how trade responds to ordinary Gulf tension.")
numbered("**Routing assumption.** Each exporter's aggregate routing is applied to its exports to China. China-bound "
         "cargoes may differ: the UAE's 2026 fall suggests they were more exposed than average. Destination-specific "
         "routing data (AIS by destination) are not available.")
numbered("**Secondary parameters.** Route parameters rest on public EIA and IEA analysis, not operational data, and "
         "Saudi dependence varies over time (September 2026 pipeline attacks). Low and high scenarios bracket the "
         "uncertainty.")
numbered("**Recorded trade ≠ dependence.** Sanctioned flows are under-recorded: China's recorded crude imports from "
         "Iran are near zero. Re-exports blur origins, notably through the UAE. Values are in USD, so price movements "
         "mix with volumes.")
numbered("**Energy-only exposure.** Exposure covers crude oil and petroleum gases, not other goods. It is a "
         "conservative measure.")
numbered("**Predetermined but not strictly ex ante.** Exposure uses 2024 commodity data while the regression sample "
         "starts in 2018. The 2023-onward sample gives a similar estimate (−0.40).")
numbered("**Small sample.** 21 partners, of which five are materially exposed. About 102 months in the regression and "
         "about 17 independent quarters for forecasting.")
numbered("**Data lag.** The IMF publishes three to four months after the reference month, which halves forecast skill "
         "relative to customs data.")
numbered("**GDELT.** Media coverage drifts over time, the United States is noisy, and events are located at country "
         "level, not at specific infrastructure.")

doc.add_heading("9.3  Using the scores correctly", level=2)
bullet("**Scores rank, they do not predict.** The country risk score orders countries by business risk; it is not a "
       "probability. Its weights are fixed design choices, published with the scores.")
bullet("**Surge probabilities run high.** Calibrated on the turbulent year 2023, they overstate the realised rate in "
       "2024–2026; the ranking and the alerts are unaffected (Section 5.7).")
bullet("**Different data, different ages.** Forecasts are current to the last complete week, ACLED to the last four "
       "weeks, trade to the latest IMF month (three to four months back); each page shows the date of its data.")
bullet("**Briefs are machine-written.** They are generated from the listed inputs and cited articles under fixed "
       "rules, labelled as such, and should be checked before being relied on for a specific decision.")
bullet("**Route risk is reported beside the score.** A chokepoint closure affects many countries at once and is not "
       "fully reflected in any single country's score; read the lanes panel together with the score (Section 2).")

doc.add_heading("9.4  Scaling up", level=2)
para("The experiments point to where scale is most likely to pay off.")
para("**More of the same instrument.** The learning curve shows the attention model still improving at full "
     "data, and same-instrument examples (MENA–world dyads) helped where different instruments (GDELT 1.0) did "
     "not. A global model, trained on all states' dyads coded the same way, multiplies the training data by "
     "an order of magnitude. Full attention over all of them is expensive, since cost grows with the square of "
     "the number of tokens, but sparse or hierarchical attention (dyads attend within regions, regions to each "
     "other) or attention restricted to active relationships keeps it tractable. Release v2.2 is a first step: "
     f"adding {V22S['world_dyads']:,} dyads from outside the region as training data raised test R² from "
     f"{f3(V21['r2'])} to {f3(V22['r2'])} without any change to the model, and its 5-seed ensemble trains in "
     "under an hour on a single consumer GPU.")
para("**From coded events to text, at the right scale.** A pretrained sentence model applied to one newspaper's "
     "headlines added no signal here; the coverage was too sparse (8% of dyad-weeks) and too generic. The "
     "promising route is dyad-specific representations learned from large multi-source news corpora, with "
     "encoders fine-tuned to extract actors, event types, intensity and certainty. That also fixes the dilution "
     f"problem in {fig('fig10_gdelt')} at its source, and requires GPU infrastructure beyond a single laptop.")
para("**A longer record through in-house coding.** The largest gain is likely to come from an event-coding "
     "pipeline of our own, applied uniformly to a news archive reaching back several decades. One instrument "
     "across that span would turn eleven years of training data into thirty or more, with many more crises, "
     "ceasefires and realignments to learn from, the kind of shifts the current model has seen too rarely to "
     "anticipate (Section 9.1, regime shifts). Owning the coding also means controlling what is measured: actors "
     "resolved consistently, hostile and cooperative events kept apart, and intensity recorded rather than "
     "averaged away. Given how directly v2.2 improved with more same-instrument data, a record of that length "
     "could change the results materially, and the model, evaluation protocol and live pipeline built here "
     "carry over to it unchanged.")
para("**Observing the route directly.** For trade, the value of a richer model will depend on data that observe the "
     "route: PortWatch transits are the obvious next model input, though their history begins only in 2019; "
     "automated China customs data would roughly double export forecast skill; and destination-specific shipping "
     "data would replace the routing assumption.")
para("**From point forecasts to calibrated risk.** Operational use needs uncertainty: quantile outputs, seed and "
     "data ensembles as uncertainty estimates, and regular re-training. The live pipeline already re-scores the "
     "full system from the latest complete week of data, so these extensions slot into an existing production loop.")

# ============================================================== 10
doc.add_heading("10  Conclusion", level=1)
para(
    "This paper has described an early-warning product for organisations with trade at stake in the Middle East and "
    "North Africa, and the evidence behind it. For conflict, it demonstrates a new way of applying deep learning to "
    "international relations: each directed country pair becomes a token, a temporal transformer models each "
    "relationship's history, and a cross-system transformer forecasts the region jointly while attending to its "
    "relations with the rest of the world. Built on a single open data source and evaluated under a strict "
    "out-of-sample protocol, the model forecasts quarterly relations among 400 MENA dyads far more accurately than a "
    "linear model on the same raw inputs and than naive forecasts, outperforms gradient boosting, and matches a linear "
    "model built on hand-engineered features without needing that engineering; trained on more dyads from the same "
    "instrument, it moves ahead of it. A second model with the same architecture flags conflict surges far better "
    "than forecasts of average relations.")
para(
    "For trade, conflict reaches China's trade with the Middle East mainly through the shipping route, not through the "
    "partners' own instability. When the Strait of Hormuz closed in March 2026, China's imports from partners without "
    "a bypass fell by four-fifths or more within a quarter, while Saudi imports, which could be rerouted to the Red "
    "Sea, rose; exposure built from route data predicts which partners were hit. In the eight years before 2026, "
    "however, Gulf conflict left no measurable trace in China's trade, and statistical models trained on that history "
    "had no closure to learn from: the most reliable trade forecast was the simplest, paired with an explicit flag for "
    "the chokepoint's status.")
para(
    "Combined in a transparent risk score and a bilingual monitor that refreshes itself from live data, the two parts "
    "give enterprises an early, current view of where conflict and trade exposure meet, and of the route channel that "
    "matters most. Both are limited by data rather than by method, which makes a global, same-instrument conflict "
    "system, a longer record coded by an in-house pipeline, and direct observation of shipping routes the natural "
    "next steps; the framework built here is designed to take them.")

# ============================================================== appendix
doc.add_heading("Appendix", level=1)
doc.add_heading("A1  Hormuz parameters and sources", level=2)
para("See data/hormuz/hormuz_dependence.csv. Each row records the low and high dependence for crude and gas, the "
     "basis, the sources, a confidence rating and the 2026 validation check. Sources:")
for s_ in ["EIA, *World Oil Transit Chokepoints* (updated 3 March 2026)", "EIA, *Today in Energy*, 16 June 2025",
           "EIA Country Analysis Briefs: Iraq (July 2025), United Arab Emirates (July 2026), Oman background (2019), "
           "Saudi Arabia (October 2024)", "EIA, *Short-Term Energy Outlook*, August–October 2026",
           "IEA, *Strait of Hormuz Factsheet* (February 2026) **[verify]**", "GRC commentary (March 2026) **[verify]**"]:
    bullet(s_)
doc.add_heading("A2  Country codes", level=2)
table("codes", "Country code mapping.", ["ISO3", "IMF code", "GDELT (FIPS)", "Note"],
      [["PSE", "WBG", "WE, GZ", "West Bank and Gaza merged"],
       ["ARE, BHR, CYP, DJI, EGY, IRN, IRQ, ISR, JOR, KWT, LBN, LBY, OMN, QAT, SAU, SDN, SYR, TUR, YEM, USA",
        "same as ISO3", "AE, BA, CY, DJ, EG, IR, IZ, IS, JO, KU, LE, LY, MU, QA, SA, SU, SY, TU, YM, US", "—"]],
      [5.6, 2.4, 5.6, 2.4])
doc.add_heading("A3  Reproduction", level=2)
code(["# conflict models (release v2.2) and their statistics",
      "python release_v2.1_goldstein.py && python release_v2.2_goldstein.py && python release_v2.2_surge.py",
      "KERAS_BACKEND=torch python paper/compute_v21_stats.py && python paper/compute_v22_stats.py",
      "KERAS_BACKEND=torch python paper/compute_v22_analysis.py",
      "python paper/make_figures.py && python paper/make_learning_curve.py",
      "python paper/make_backtest_figure.py && python paper/make_scaling_figure.py",
      "# trade module",
      "python src/fetch_imf_imts.py && python src/fetch_portwatch.py",
      "python src/build_exposure.py && python src/build_panel.py && python src/check_trade_data.py",
      "python src/run_regression.py && python src/trade_baseline.py",
      "python paper/compute_trade_stats.py && python paper/make_trade_figures.py",
      "# risk monitor data bundle, then this paper",
      "python build_site_data.py",
      "python paper/build_paper.py"])

doc.add_heading("References", level=1)
for ref in [
    "Borusyak, K., Hull, P., and Jaravel, X. (2022). Quasi-experimental shift-share research designs. *Review of "
    "Economic Studies*, 89(1), 181–213.",
    "Cameron, A. C., Gelbach, J. B., and Miller, D. L. (2011). Robust inference with multiway clustering. *Journal of "
    "Business & Economic Statistics*, 29(2), 238–249.",
    "Driscoll, J. C., and Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel "
    "data. *Review of Economics and Statistics*, 80(4), 549–560.",
    "Gerner, D. J., Schrodt, P. A., Yilmaz, Ö., and Abu-Jabr, R. (2002). Conflict and Mediation Event Observations "
    "(CAMEO): A new event data framework for the analysis of foreign policy interactions. Paper presented at the "
    "Annual Meeting of the International Studies Association, New Orleans.",
    "Goldstein, J. S. (1992). A conflict-cooperation scale for WEIS events data. *Journal of Conflict Resolution*, "
    "36(2), 369–385.",
    "Hegre, H., Allansson, M., Basedau, M., Colaresi, M., Croicu, M., Fjelde, H., et al. (2019). ViEWS: A political "
    "violence early-warning system. *Journal of Peace Research*, 56(2), 155–174.",
    "Hegre, H., Bell, C., Colaresi, M., Croicu, M., Hoyles, F., Jansen, R., et al. (2021). ViEWS2020: Revising and "
    "evaluating the ViEWS political Violence Early-Warning System. *Journal of Peace Research*, 58(3), 599–611.",
    "International Monetary Fund. *International Merchandise Trade Statistics (IMTS)*. https://data.imf.org "
    "(accessed October 2026).",
    "International Monetary Fund and University of Oxford. *IMF PortWatch: monitoring trade disruptions from "
    "space*. https://portwatch.imf.org (chokepoint transit data, accessed October 2026).",
    "Jain, S., and Wallace, B. C. (2019). Attention is not explanation. *Proceedings of NAACL-HLT 2019*, 3543–3556.",
    "Leetaru, K., and Schrodt, P. A. (2013). GDELT: Global data on events, location, and tone, 1979–2012. Paper "
    "presented at the Annual Convention of the International Studies Association, San Francisco.",
    "Mueller, H., and Rauh, C. (2018). Reading between the lines: Prediction of political violence using newspaper "
    "text. *American Political Science Review*, 112(2), 358–375.",
    "Mueller, H., and Rauh, C. (2022). The hard problem of prediction for conflict prevention. *Journal of the "
    "European Economic Association*, 20(6), 2440–2467.",
    "Raleigh, C., Linke, A., Hegre, H., and Karlsen, J. (2010). Introducing ACLED: An Armed Conflict Location and "
    "Event Dataset. *Journal of Peace Research*, 47(5), 651–660.",
    "Shazeer, N. (2020). GLU variants improve Transformer. arXiv:2002.05202.",
    "United Nations. *UN Comtrade Database*. https://comtradeplus.un.org (accessed October 2026).",
    "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. "
    "(2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.",
    "Vesco, P., Hegre, H., Colaresi, M., Jansen, R. B., Lo, A., Reisch, G., and Weidmann, N. B. (2022). United they "
    "stand: Findings from an escalation prediction competition. *International Interactions*, 48(4), 860–896.",
    "Waltz, K. N. (1979). *Theory of International Politics*. Reading, MA: Addison-Wesley.",
]:
    p = doc.add_paragraph()
    _runs(p, ref, size=10)
    p.paragraph_format.left_indent = Cm(0.8)
    p.paragraph_format.first_line_indent = Cm(-0.8)
    p.paragraph_format.space_after = Pt(4)

assert fig_no[0] == len(FIGS) and tab_no[0] == len(TABS), (fig_no[0], tab_no[0])
import sys  # noqa: E402
out = sys.argv[1] if len(sys.argv) > 1 else f"{P}/risk_paper.docx"
doc.save(out)
print("saved", out, "| figures:", fig_no[0], "| tables:", tab_no[0])

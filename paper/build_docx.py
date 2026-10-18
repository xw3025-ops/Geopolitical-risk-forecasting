"""
Build the paper section (release v2.2) as a Word document from saved results.

    KERAS_BACKEND=torch python paper/compute_v21_stats.py    # v2.1 numbers, experiments
    python paper/compute_v22_stats.py                         # v2.2 numbers, surge flag
    KERAS_BACKEND=torch python paper/compute_v22_analysis.py  # v2.2 variance, examples, attention
    python paper/make_figures.py paper/make_learning_curve.py paper/make_backtest_figure.py paper/make_scaling_figure.py
    python paper/build_docx.py                               # -> paper/transformer_section.docx
"""

import json
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
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
SC = json.load(open(f"{P}/scaling_summary.json"))
B = S["goldstein_benchmark"]
V21, V20, GBM = R["benchmark"]["v2.1"], R["benchmark"]["v2.0"], R["gbm"]
LIN = json.load(open(f"{P}/linear_baselines.json"))      # OLS, fit on training weeks only
V22S = json.load(open(f"{P}/v22_stats.json"))            # release v2.2: world training data, surge flag
V22, FL = V22S["v2.2"], V22S["flag"]
A22 = json.load(open(f"{P}/v22_analysis.json"))          # v2.2 variance, examples, attention (compute_v22_analysis.py)
RISK = json.load(open("artifacts/risk_v1_calibration.json"))
OLS1, OLS44 = LIN["OLS current week"], LIN["OLS 44 history features"]
AB, PATH, DIR = F["ablation"], F["path"], F["direction"]

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
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT if align == "left" else WD_ALIGN_PARAGRAPH.JUSTIFY
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def bullet(text):
    p = doc.add_paragraph(style="List Bullet")
    _runs(p, text)
    p.paragraph_format.space_after = Pt(2)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def figure(name, caption, width=16.0):
    fig_no[0] += 1
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


def table(caption, header, rows, widths, bold_last=False, note=None):
    tab_no[0] += 1
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
            r = cell.paragraphs[0].add_run(str(v))
            r.font.size = Pt(9)
            r.bold = bold_last and i == len(rows)
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


f3 = lambda x: f"{x:.3f}"
pct = lambda x: f"{x:.1%}"
d = S["data"]

# ============================================================== title + summary
t = doc.add_paragraph()
r = t.add_run("Forecasting the Region as a System: A Spatio-Temporal Transformer for Interstate "
              "Relations in the Middle East and North Africa")
r.bold, r.font.size = True, Pt(16)
t.paragraph_format.space_after = Pt(4)
para("Section draft for the Risk-A-Lab geopolitical risk forecasting paper. Model release v2.2. All results are "
     "reproducible from the project repository (benchmark_honest.py, benchmark_ladder.py, benchmark_targets.py, "
     "experiment_*.py, release_v2.1_goldstein.py, release_v2.2_*.py and the paper/ scripts).", italic=True, size=9.5, color=GREY, align="left", after=12)

doc.add_heading("Section summary", level=2)
para(
    "Most operational conflict-forecasting systems treat each country or country pair as a separate row of data "
    "and predict it from its own history, with the rest of the world entering, if at all, through hand-built "
    "aggregate features. This section develops and tests an alternative, closer to how states make decisions: a "
    "transformer that forecasts every relationship in the system jointly. Each directed country pair (dyad) is a "
    "token; a temporal transformer encodes each dyad's recent history, and a cross-dyad transformer lets every "
    "dyad attend to every other. The model forecasts the 400 dyads among 20 Middle East and North Africa (MENA) "
    "states while attending to a wider system of 1,959 dyads that also includes the region's relations with the "
    "rest of the world. Using GDELT event data aggregated by week, it forecasts each dyad's mean Goldstein "
    "cooperation–conflict score over the next 13 weeks. Under a strict out-of-sample protocol the release model "
    f"(v2.2) reaches a test R² of {f3(V22['r2'])}, about twice that of a linear regression on the same raw weekly "
    f"inputs ({f3(OLS1['r2'])}), and ahead of the strongest naive forecast (a trailing 52-week mean, "
    f"{f3(B['trailing 52-week mean']['r2'])}), gradient boosting ({f3(GBM['r2'])}) and a linear regression on 44 "
    f"history features that an analyst designed by hand, with hindsight about what predicts well "
    f"({f3(OLS44['r2'])}). The transformer learns those summaries itself, which is what lets the approach extend "
    f"to new regions and data sources: trained on the region alone (v2.1) it already matches the hand-engineered "
    f"model ({f3(V21['r2'])}), and training it on {V22S['world_dyads']:,} additional dyads from outside the region "
    "moves it ahead.")
para(
    "A systematic programme of experiments clarifies what drives these gains and what limits them. Attention "
    "across the system helps when there is enough training data and when crises propagate across the region, "
    "as in 2023, and its advantage grows with data. The model is currently limited by the number of training "
    "examples rather than by the richness of its inputs, and adding examples from the same instrument improves "
    "it directly. For early warning, a second model with the same architecture forecasts the *worst* month of "
    "conflict in the coming quarter rather than the average; it flags conflict surges far better than a "
    "forecast of average relations. The contribution is methodological: a demonstration "
    "of how self-attention can bring the international system into dyad-level forecasts, a working prototype "
    "that re-forecasts the region every two weeks from live data, and evidence-based directions for scaling the "
    "approach. Section 5 shows how the forecasts drive a bilingual risk monitor for organisations with trade at "
    "stake in the region, and lists every external data source it uses. Elsewhere in the paper the model also "
    "supplies the likelihood component of a risk index for "
    "commercial users.")

# ============================================================== 1
doc.add_heading("1  The problem: forecasting countries one row at a time", level=1)
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
    "Figure 1 contrasts the two approaches.")
figure("fig1_concept", "Row-by-row versus system-aware forecasting. (a) Conventional set-up: each country pair is "
       "a separate row; context enters only through features computed in advance. (b) This study: each pair's "
       "history is encoded, then all pairs exchange information through self-attention before any forecast is "
       "made.", 15.5)

# ============================================================== 2
doc.add_heading("2  Core assumption: self-attention as a model of the international system", level=1)
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

# ============================================================== 3
doc.add_heading("3  Design", level=1)
doc.add_heading("3.1  Data", level=2)
para(
    "The sole input is the Global Database of Events, Language and Tone (GDELT 2.0; Leetaru and Schrodt 2013), "
    "which machine-codes news reports into events with a source actor, a target actor, a CAMEO event type "
    "(Gerner et al. 2002) and a Goldstein score from −10 (most conflictual) to +10 (most cooperative; Goldstein "
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
    "The target is the **mean Goldstein score of a dyad over the next 13 weeks**. Inputs for each week are the "
    "log counts of the 20 CAMEO root event types, the week's mean Goldstein score, and three long-memory "
    "features: the dyad's trailing 26- and 52-week mean Goldstein and its 52-week activity level (Table 1).")
table("Data and model specification (release v2.2).", ["Item", "Specification"],
      [["Source", "GDELT 2.0 events, public BigQuery dataset gdelt-bq.gdeltv2.events"],
       ["Forecast targets", "400 directed MENA dyad-weeks, incl. domestic"],
       ["System (tokens per week)", "400 MENA dyads + 1,559 MENA–world dyads = 1,959"],
       ["Extra training data (v2.2)", f"{V22S['world_dyads']:,} non-MENA dyads, in three extra 1,959-token "
                                      "sequences per training week; never evaluated"],
       ["Period", f"{d['first_week']} to {d['last_week']} ({d['weeks']} weeks)"],
       ["Target", "Mean Goldstein score over the next 13 weeks (continuous)"],
       ["Inputs per week", "log CAMEO root counts (20), mean Goldstein, trailing 26/52-week mean Goldstein, "
                           "52-week activity (24 features)"],
       ["Look-back", "13 weeks directly; up to 64 weeks through long-memory features"],
       ["Model size", f"{A22['params']:,} parameters per model; 5-model seed ensemble"]], [4.0, 12.0])

doc.add_heading("3.2  Architecture", level=2)
para(
    "Figure 2 shows the architecture. Each dyad's 13-week input window is projected to 64-dimensional tokens "
    "with a learned temporal position embedding and passed through two **temporal transformer** blocks, in which "
    "the dyad attends only over its own weeks. The representation of the latest week is concatenated with two "
    "**country embeddings**, learned 8-dimensional vectors for the source and target state from one shared "
    "table, so that 'Iran' means the same on either side of a dyad while direction is preserved by position. "
    "The resulting dyad tokens, 1,959 per week, then pass through two **cross-system transformer** "
    "blocks in which every dyad attends to every other. All blocks use four-head attention, SwiGLU feed-forward "
    "layers (Shazeer 2020), residual connections and layer normalisation.")
para(
    "The output is **residual**: the network produces a correction that is added to the dyad's trailing 52-week "
    "mean, so the model starts from the strongest naive forecast and learns when and how far the coming quarter "
    "will depart from a relationship's normal. Forecasts are read off the 400 MENA tokens. The release averages "
    "five identically configured models trained from different random seeds.")
figure("fig2_architecture", "Architecture of release v2.2. Shaded blocks are the two attention stages: temporal "
       "attention within each dyad's own history, then cross-system attention across all 1,959 dyads of the "
       "week. During training, three further sequences of dyads from outside the region pass through the same "
       "blocks. Tensor shapes in brackets.", 16)

doc.add_heading("3.3  Evaluation protocol", level=2)
para(
    "Forecasting results are easy to inflate, and an earlier internal version of this model was, so the protocol "
    "is deliberately strict (Figure 3). Data are split chronologically into training (May 2015 – October 2022), "
    "validation (February – December 2023) and test (March 2024 – June 2026) periods. Because each target "
    "averages the 13 weeks *after* its row, 13-week embargoes separate the periods; otherwise training targets "
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

# ============================================================== 4
doc.add_heading("4  Results", level=1)
doc.add_heading("4.1  Benchmark performance", level=2)
para(
    f"Table 2 and Figure 4 report test performance. Release v2.2 reaches R² = {f3(V22['r2'])} and MAE = "
    f"{f3(V22['mae'])} Goldstein points. Given the same raw weekly inputs, a linear regression reaches only "
    f"{f3(OLS1['r2'])}: the transformer's ability to learn from raw history roughly doubles the explained "
    f"variance. It is also ahead of every naive forecast, of the regional-attention v2.0 ({f3(V20['r2'])}) and of "
    f"gradient boosting on hand-built features ({f3(GBM['r2'])}), including on all three change measures "
    f"(direction {pct(V22['dir_acc'])} vs {pct(GBM['dir_acc'])}, change correlation {V22['change_corr']:.3f} vs "
    f"{GBM['change_corr']:.3f}, escalation AUC {V22['esc_auc']:.3f} vs {GBM['esc_auc']:.3f}).")
para(
    f"The comparison with the raw-input regression is the central one. Given the 44 hand-built features, the "
    f"same linear regression rises from {f3(OLS1['r2'])} to {f3(OLS44['r2'])}. The whole of that gain comes from "
    "feature engineering done with hindsight (Section 3.3). The transformer achieves it by learning from the raw "
    "series which summaries of the past matter: trained on the region alone (v2.1, R² "
    f"{f3(V21['r2'])}), it is already level with the engineered model (change correlation "
    f"{V21['change_corr']:.3f} against {OLS44['change_corr']:.3f}, escalation AUC {V21['esc_auc']:.3f} against "
    f"{OLS44['esc_auc']:.3f}), so a well-engineered classical model is matched without anyone having to engineer "
    "it. That property is what allows the approach to carry over to new regions, actors or data sources, where no "
    "tested feature set exists. Given more training data (Section 4.3), the release moves ahead of the engineered "
    f"model (R² {f3(V22['r2'])} against {f3(OLS44['r2'])}, change correlation {V22['change_corr']:.3f} against "
    f"{OLS44['change_corr']:.3f}, escalation AUC {V22['esc_auc']:.3f} against {OLS44['esc_auc']:.3f}, direction "
    f"{pct(V22['dir_acc'])} against {pct(OLS44['dir_acc'])}), whereas the engineered features would have to be "
    "redesigned to gain from it. Beyond this, the transformer's specific advantages appear where Section 4.4 "
    "locates them: in periods when shocks propagate across the system, and increasingly as training data grow. "
    f"Relative to the trailing 52-week mean the release reduces squared error by {A22['variance']['mse_reduction']:.1%} "
    f"and is more accurate for {A22['variance']['share_dyads_better']:.0%} of the 400 dyads. The trailing 52-week "
    "mean is a demanding baseline: a smoothed quarterly target is largely persistent, and carrying a relationship's "
    "annual average forward already explains 60% of test variance.")
table("Test-set performance (400 MENA dyads × 119 weeks, March 2024 – June 2026).",
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
           "(Section 3.3).")
figure("fig4_benchmark", "Test R² of the release models against naive forecasts, the earlier transformer recipe and "
       "gradient boosting. Orange: models using the 44 hand-built features. Dark blue: the current release. "
       "The axis starts at 0.30.", 15.5)

doc.add_heading("4.2  What produced the improvement", level=2)
pv = list(PATH.values())
para(
    "Figure 5 and Table 3 trace the path from the original v1.0 recipe to the release, adding one change at a "
    f"time. The original recipe, once evaluated honestly, scored {pv[0]:.3f}, below the trailing mean. Stronger "
    f"regularisation (+{pv[1] - pv[0]:.3f}), the residual formulation (+{pv[2] - pv[1]:.3f}) and the long-memory "
    f"inputs (+{pv[3] - pv[2]:.3f}) contributed similar amounts; averaging five seeds added +{pv[4] - pv[3]:.3f} "
    f"(v2.0), and extending attention to the region's relations with the rest of the world added "
    f"+{pv[5] - pv[4]:.3f} (v2.1). Training on dyads from outside the region then added +{pv[6] - pv[5]:.3f} "
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
table("Configurations on the path to the release. Selection used validation MSE only; test R² for reference.",
      ["Configuration", "Validation MSE", "Test R²"], rows, [10.6, 2.7, 2.7], bold_last=True)

doc.add_heading("4.3  What limits the model", level=2)
para(
    "Before the release was fixed we asked why forecasts of *change* remain modest: around 61% direction accuracy "
    "for every model, including the transformers. Three diagnostics come first. Weeks with no events are "
    "recorded as zero, but changes in the number of active weeks explain only 1% of the variance of quarterly "
    "change; sampling noise from low-volume weeks accounts for about 7%; and cooperative events masking conflict "
    "(Section 5) is real but partial: in weeks where violent events doubled, the mean Goldstein score still fell "
    "by two points on average, but rose in 12% of cases. The target, in other words, mostly measures what it "
    "should. An honest re-run of the original modelling ladder also locates the gains: a rudimentary network "
    f"that sees only the current week scores R² {L['rudimentary: CAMEO counts']['r2']:.2f}, country embeddings "
    f"raise it to {L['rudimentary: counts + goldstein + embeddings']['r2']:.2f}, and the transformer recipe to "
    "about 0.64. Most of that gain is in levels; on change, the rudimentary network is already within about "
    "0.06 of the transformer's change correlation.")
para(
    "We then tested whether more information would help (Table 4), one experiment at a time, each against the "
    "release recipe on the same rows. Measuring conflict only, by summing the Goldstein scores of negative "
    "events, did not make change more predictable. Its event-weighted average even missed 7 October 2023, "
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
table("Data and model experiments (single seed 42 unless marked; same test rows). Lower validation MSE is better.",
      ["Experiment", "Val. MSE", "Test R²", "Direction", "Change corr.", "Esc. AUC"], trows,
      [6.4, 1.7, 1.7, 1.9, 2.1, 2.2],
      note="Conflict-only target (log sum of negative Goldstein scores; reported on its own scale): change correlation "
           f"{R['conflict_target']['change_corr']:.3f}, escalation AUC {R['conflict_target']['esc_auc']:.3f}, against "
           "0.356 and 0.648 for a simple 13-week momentum baseline on that target.")

doc.add_heading("4.4  When does system attention help?", level=2)
para(
    "Attention across the system is the central idea of this section, so we tested it from four directions. "
    "**First, a regional ablation** (Figure 6) retrains the v2.0 configuration without cross-dyad attention, "
    "without country embeddings, and without both. On validation the full model is best and each removal hurts "
    f"(MSE {AB['full']['val_mse_ensemble']:.3f} full, {AB['no_crossdyad']['val_mse_ensemble']:.3f} without "
    f"attention, {AB['neither']['val_mse_ensemble']:.3f} without both). On the 2024–2026 test period, however, "
    f"country embeddings help ({AB['full']['r2']:.3f} vs {AB['no_embedding']['r2']:.3f} without them) while "
    f"cross-dyad attention does not ({AB['no_crossdyad']['r2']:.3f} without it). The result is mixed, and "
    "on its own would not settle E2.")
figure("fig6_ablation", "Regional ablation (v2.0 configuration, three seeds each). Dots are seeds, diamonds their "
       "average.", 15.5)
bi, bc = BT["independent"], BT["cross-system"]
para(
    "**Second, a rolling backtest** (Figure 7) repeats the comparison on four test periods, each with its own "
    "training and validation window. Attention loses in the two early folds, which have only 151 and 256 "
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
    "**Third, a learning curve** (Figure 8) trains both architectures on 12.5%, 25%, 50% and 100% of the training "
    "weeks. Both improve with data, but the attention model improves faster: it starts worse "
    f"(validation MSE {cc['val'][0]:.3f} vs {ci['val'][0]:.3f} with 49 weeks) and ends clearly better "
    f"({cc['val'][-1]:.3f} vs {ci['val'][-1]:.3f} with 391 weeks), and its validation error is still falling "
    f"by about {abs(cc['val_change_per_doubling'][-1]):.3f} per doubling of data, against "
    f"{abs(ci['val_change_per_doubling'][-1]):.3f} for the independent model. On test change correlation it "
    f"closes from {cc['corr'][0]:.3f} vs {ci['corr'][0]:.3f} to parity. This crossover is the characteristic "
    "signature of a higher-capacity model that pays off as data grows (E3).")
figure("fig12_learning_curve", "Learning curves: models trained on 12.5–100% of the training weeks (MENA only, "
       "three seeds each).", 15.5)
para(
    "**Fourth, the cross-system view** (Figure 9) adds MENA–world dyads to the system. At every size from 400 to "
    "1,959 dyads, cross-system attention achieves lower validation error than the independent model given the "
    "same dyads, and the full 1,959-dyad system gives the lowest validation error of any configuration, which "
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

doc.add_heading("4.5  What the cross-system attention learns", level=2)
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
    f"(Figure 10). The most selective ({head_name(h1)}) concentrates on a handful of relationships "
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

doc.add_heading("4.6  Where the model adds information", level=2)
vv = A22["variance"]
ex = vv["example_reduction"]
para(
    f"Most of the variation the model is scored on lies *between* dyads: {vv['between_share']:.0%} of test "
    "variance is the difference in long-run level between relationships (Israel–Palestine is always far more "
    "conflictual than Oman–Qatar). Levels are what the trailing mean already captures, so the model's "
    "value-added is best measured on movements: its forecast departures from each dyad's normal correlate at "
    f"{vv['change_corr']:.2f} with the realised departures. Figure 11 illustrates this for four salient "
    f"relationships. The model improves on the 52-week mean for Turkey → Syria (−{ex['TUR-SYR']:.0%} squared "
    f"error), Israel → Palestine (−{ex['ISR-PSE']:.0%}) and Israel → Iran (−{ex['ISR-IRN']:.0%}), and is less "
    f"accurate for Saudi Arabia → Yemen (+{-ex['SAU-YEM']:.0%}), whose largest move in 2026 was an abrupt break. "
    "Anticipating such breaks a quarter ahead is the hardest part of the problem.")
figure("fig7_examples", "Forecast versus outcome for four dyads over the test period (release v2.2). Titles give "
       "the change in squared error relative to the trailing 52-week mean (negative = better).", 16)

doc.add_heading("4.7  A second task: flagging conflict surges", level=2)
DIRN = FL["v2.0 direction model (4-week escalation probability)"]
SURG = FL["v2.2 conflict-surge model"]
NAIV = FL["Naive: last quarter's peak conflict"]
LINF = FL["Linear regression, hand-built features*"]
GOLD = FL["v2.2 Goldstein forecast (fall below normal)"]
para(
    "Users of a risk product care less about how the average tone of a relationship will move than about "
    "whether it is heading into a crisis. An earlier companion model addressed this with a four-week direction "
    "task: will the next four weeks be more conflictual than the current week? It predicts that label well "
    f"(ROC-AUC {V22S['direction_own_auc']:.2f}), but the label mostly measures regression to the mean: a high "
    "escalation probability marks a dyad that has just had an unusually good week. Scored against the outcome "
    f"that matters, a surge in conflict over the next quarter, it ranks dyad-weeks no better than chance "
    f"(AUC {DIRN['conflict_surge']['auc']:.2f}).")
para(
    "We therefore changed what the model forecasts. The **conflict-surge model** keeps the architecture and the "
    "13-week horizon, which leaves organisations a quarter to act, but forecasts the *worst month* of the coming "
    "quarter: the highest four-week mean of a dyad's weekly conflict intensity, the log of the summed Goldstein "
    "scores of its hostile events. Counting hostile events only, by volume, removes the dilution shown in "
    "Figure 14, where humanitarian and diplomatic events raised the average score during a war. A **surge** is a "
    "quarter whose peak exceeds the dyad's typical quarterly peak by more than the worst tenth of historical "
    f"gaps ({V22S['surge_rate']:.0%} of test dyad-weeks). The model adds conflict intensity and summed Goldstein to "
    "its inputs and weights each dyad's error by its historical volatility, so that the many stable "
    "relationships do not teach it to stay flat. Its score is calibrated on validation into a probability of a "
    "surge. Because the validation year, 2023, was unusually turbulent, these probabilities run high in "
    f"2024–2026 (mean {V22S['surge_mean_prob_test']:.0%} against a realised {V22S['surge_rate_test_all']:.0%}); "
    "the ranking, which drives the alerts, is unaffected, and the calibration will be refitted as data accrue.")
para(
    f"Table 5 compares crisis flags on the test weeks shared by every model ({V22S['flag_weeks']} weeks). If each "
    f"week an analyst reviewed the ten dyads ranked highest, {SURG['conflict_surge']['top10']:.0%} of the surge "
    f"model's alerts would be followed by a surge, against {GOLD['conflict_surge']['top10']:.0%} for the v2.2 "
    f"Goldstein forecast and {DIRN['conflict_surge']['top10']:.0%} for the direction model "
    f"(base rate {V22S['surge_rate']:.0%}). Simple benchmarks are competitive on this target: conflict peaks are "
    f"persistent, so last quarter's peak already scores AUC {NAIV['conflict_surge']['auc']:.3f}, and a linear "
    f"regression on the hand-built features {LINF['conflict_surge']['auc']:.3f}, against "
    f"{SURG['conflict_surge']['auc']:.3f} for the transformer. On this new target the hand-built features, "
    "designed with hindsight for this region (Section 3.3), keep an edge in ranking; on the average-Goldstein "
    "target the transformer closed the same gap once it had more training data (Section 4.1), and the surge "
    "target is the natural next candidate for that. The surge model also departs from the normal more readily "
    "than the average-Goldstein forecast (Figure 12): for Israel → Palestine its forecast of the coming "
    f"peak correlates {V22S['isr_pse']['ISR-PSE']['surge_corr']:.2f} with the actual change and moves "
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
table("Crisis flags on the shared test weeks (400 MENA dyads). AUC ranks all dyad-weeks; top-10 is the share of "
      "each week's ten highest-scored dyads that experience the event in the next 13 weeks.",
      ["Flag", "Surge AUC", "Surge top-10", "Goldstein-crisis AUC", "Goldstein-crisis top-10"],
      [[name, f3(r["conflict_surge"]["auc"]), f"{r['conflict_surge']['top10']:.0%}", f3(r["gold_crisis"]["auc"]),
        f"{r['gold_crisis']['top10']:.0%}"] for name, r in FL.items()],
      [6.4, 1.9, 2.0, 2.6, 2.6], bold_last=True,
      note=f"Base rates: conflict surge {V22S['surge_rate']:.0%}, Goldstein crisis {V22S['gold_crisis_rate']:.0%} "
           "(the next quarter's worst four-week mean Goldstein falls below the dyad's typical quarterly low by more "
           "than the worst tenth of historical gaps). *Hand-built features as in Table 2, plus conflict-intensity "
           "history; fit on training weeks.")
figure("fig14_surge_isr", "Israel → Palestine on the test weeks. (a) The v2.2 forecast of mean Goldstein stays near "
       "the dyad's normal. (b) The surge model's forecast of the coming quarter's peak conflict intensity departs "
       "from its normal more often, but also misses the sustained decline. Forecasts are plotted at the week they "
       "are made.", 16)

doc.add_heading("4.8  From forecast to risk product", level=2)
para(
    "Section 5 describes how the forecasts become a country risk score and a live risk monitor. Elsewhere in "
    "this paper they also feed a country-level risk index in which risk is the product of likelihood and impact; "
    "there the transformer supplies the likelihood component, a calibrated probability that a country's "
    "relations deteriorate materially below their own 52-week normal over the next quarter, which ranks "
    "countries with ROC-AUC 0.70 on held-out data with the v2.2 forecasts. One "
    "finding from that work belongs here because it concerns model outputs: the direction model's escalation "
    "probability, used naively as a likelihood, ranks countries *worse* than random (AUC 0.37), because a high "
    "value mostly flags a country that has just had an unusually good week. Outputs must be validated against "
    "the decision they serve, not only against their training target; this is why the live pipeline now pairs "
    "the Goldstein forecast with the conflict-surge flag of Section 4.7 instead of the direction model.")

# ============================================================== 5
doc.add_heading("5  From forecasts to a risk monitor", level=1)
para(
    "The forecasts are published through a bilingual (English and Chinese) web-based risk monitor for "
    "organisations with trade at stake in the region, in particular Chinese exporters, energy buyers, "
    "contractors and shippers. Its design follows four principles. **Forecast and observation are kept apart:** "
    "everything produced by, or built on, the transformer carries one consistent visual marker, and observed "
    "data (recent violence, trade, shipping) are shown plainly, so users can tell the model's view of the next "
    "quarter from what has already happened. **The system view is kept:** a country's relationships are drawn "
    "on the map, so a user sees where pressure comes from, not only how much. **Decisions, not data:** every "
    "figure is accompanied by a plain statement of what it means for an organisation with interests in the "
    "country. **Calm by default:** the map shows only relationships forecast to turn hostile and those with a "
    "conflict-surge alert, so attention goes where it is needed (Figure 13).")
figure("fig15_website", "The risk monitor. (a) Regional overview: countries coloured by risk; the hovered country "
       "(Saudi Arabia) is lifted and its troubled relationships are drawn - orange for those forecast to turn "
       "hostile, glowing with a travelling ball where a conflict surge is flagged; rings mark the Strait of Hormuz, "
       "Bab el-Mandeb and the Suez Canal, coloured by shipping disruption. (b) Country page (Yemen): the risk score "
       "and its parts, the 13-week outlook against the past year, the relationships most at risk, the China "
       "exposure brief, and ACLED events of the last 28 days as pillars on the map.", 16)

doc.add_heading("5.1  How risk is defined", level=2)
para(
    "Risk is defined for an organisation with trade at stake in a country: it is high only when the conflict "
    "environment is expected to be serious **and** much trade is exposed to it. Each country receives three "
    "scores on a 0–100 scale, recomputed at every refresh (Table 6).")
bullet("**Conflict severity** combines the two transformer forecasts with one observed signal. *Forecast "
       "hostility* is the activity-weighted mean of the Goldstein forecasts of all the country's dyads, including "
       "its domestic dyad, mapped linearly so that +1.5 or above scores 0 and −4.0 or below scores 1. *Forecast "
       "surge risk* is the activity-weighted mean conflict-surge probability of those dyads, with 0.6 or above "
       "scoring 1. *Recent violence* is the number of deaths recorded by ACLED in the last four weeks on a log "
       "scale, with 2,000 or more scoring 1. Severity = 100 × (0.40 × hostility + 0.30 × surge risk + 0.30 × "
       "violence). Activity weights are each dyad's average weekly event count over the past year, so a country's "
       "busiest relationships count most.")
bullet("**Trade at stake** (exposure) is China's goods trade with the country, exports plus imports, over the latest "
       "12 months published by the IMF, on a log scale relative to the largest partner in the region: "
       "100 × log(1 + trade) / log(1 + largest trade). The log scale keeps a mid-sized partner from being "
       "dwarfed by Saudi Arabia or the United Arab Emirates.")
bullet("**Risk** = √(severity × trade at stake), the geometric mean of the two. Unlike a product it stays on the "
       "0–100 scale, and unlike an average it rewards neither alone: a country at war with almost no trade at "
       "stake, or a large partner in calm conditions, scores moderately. Bands are severe (≥ 55), high (≥ 45), "
       "elevated (≥ 32) and low. **Regional risk** is the average of the 20 country scores, and every score is "
       "also reported as its change since the previous refresh.")
para(
    "The weights and scaling points are design choices, set so that each component spans its full range across "
    "the region and fixed in advance rather than fitted, so that scores remain comparable across refreshes. They "
    "are published with the scores and can be changed in one place. The score orders countries by business "
    "risk; it is not a probability.")
table("Components of the country risk score.",
      ["Component", "Source", "Kind", "Scaling to 0–1"],
      [["Forecast hostility", "Transformer v2.2, 13-week Goldstein forecast", "Forecast", "+1.5 → 0, −4.0 → 1"],
       ["Forecast surge risk", "Transformer v2.2, conflict-surge probability", "Forecast", "0 → 0, 0.6 → 1"],
       ["Recent violence", "ACLED deaths, last 4 weeks", "Observed", "log, 2,000 deaths → 1"],
       ["Trade at stake", "IMF IMTS, China trade, last 12 months", "Observed", "log, relative to largest partner"],
       ["Severity", "0.40 / 0.30 / 0.30 of the first three", "Forecast-based", "× 100"],
       ["Risk", "√(severity × trade at stake)", "Forecast-based", "0–100"]],
      [3.4, 6.2, 2.4, 4.0])

doc.add_heading("5.2  What the monitor shows", level=2)
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
       "deaths) and a China exposure brief (Section 5.3).")
bullet("**Shipping lanes.** Weekly vessel transits through the Strait of Hormuz, Bab el-Mandeb and the Suez Canal "
       "over the last four weeks are compared with their 2019 – September 2023 average, before the Red Sea "
       "attacks; a lane is *disrupted* at −50% or worse and *reduced* at −20% or worse. For Hormuz the monitor "
       "also reports China's trade with partners whose oil exports depend on the strait, weighted by the share "
       "of each partner's crude exports that cannot bypass it (high case).")
bullet("**Data & model page and export.** A plain-language account of the data, the model and its limits, with "
       "the test results of Section 4; all outputs (country scores, the 400 dyad forecasts, shipping lanes, the "
       "ACLED events shown and the full data bundle) can be downloaded.")

doc.add_heading("5.3  Written briefs", level=2)
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

doc.add_heading("5.4  A dynamic system", level=2)
para(
    "The monitor is refreshed without manual intervention. Every second week a scheduled cloud job rebuilds the "
    "weekly GDELT panel through the last complete week, runs the release models and writes the forecasts to "
    "BigQuery. A second job refreshes the IMF trade and PortWatch shipping series every week. A third weekly job, "
    "run after both, builds a single data bundle (scores, dyad forecasts, histories, recent ACLED events, shipping "
    "lanes, briefs) and publishes it to cloud storage. The website is a static front end that reads the latest bundle on every visit, "
    "so a refresh reaches users without redeploying the site, and the date of the data is shown on every page. "
    "Table 7 lists every external source used by the models and the monitor.")
table("External data used by the forecasting models and the risk monitor.",
      ["Source", "Provider and access", "Used for", "Refresh"],
      [["GDELT 2.0 event data", "GDELT Project; public BigQuery dataset gdelt-bq.gdeltv2.events",
        "All model inputs and targets: weekly dyad panel (400 MENA dyads, 1,559 MENA–world links, 5,122 world dyads for training)", "Every 2 weeks"],
       ["ACLED", "Armed Conflict Location & Event Data; API (registered account)",
        "Recent violence in severity; event map; context in briefs", "Every 2 weeks"],
       ["IMF International Trade in Goods (IMTS)", "IMF; SDMX 3.0 API",
        "China's trade with each country: trade at stake, trade briefs", "Weekly (published with a 3–4 month lag)"],
       ["IMF PortWatch", "IMF and University of Oxford; public ArcGIS API",
        "Daily vessel transits through Hormuz, Bab el-Mandeb, Suez, Cape of Good Hope, Malacca", "Weekly"],
       ["Strait of Hormuz dependence", "Compiled from IEA and EIA route data (data/hormuz)",
        "Share of each Gulf exporter's crude that cannot bypass Hormuz", "When sources change"],
       ["The Guardian Open Platform; New York Times Article Search", "Guardian Media Group; The New York Times; APIs",
        "Recent coverage of each country and China for the briefs", "Every 2 weeks"],
       ["DeepSeek API (deepseek-chat)", "DeepSeek; API",
        "Writing the bilingual briefs from the inputs above", "Every 2 weeks"],
       ["World Bank WDI", "World Bank; API", "Trade exposure in the earlier risk-index module", "Every 2 weeks"],
       ["Natural Earth via world-atlas", "Natural Earth (public domain)", "Country boundaries for the maps", "Static"]],
      [4.0, 4.4, 5.6, 2.4],
      note="UN Comtrade (HS 2709/2711, 2024) is used only for the optional Hormuz energy detail of the trade module. "
           "News articles are used as cited context only; they are not model inputs.")

# ============================================================== 6
doc.add_heading("6  Limitations", level=1)
para("As a prototype, the model has limitations that define its current scope and point to the next steps in "
     "Section 7.")
para("**Data volume.** The model is trained on 391 weeks of the regional system, and Section 4.3 showed it is "
     "limited by the number of training examples rather than by input richness. Seeds vary "
     f"({min(V22['seed_r2']):.3f}–{max(V22['seed_r2']):.3f} test R² across the five v2.2 seeds), which is why the release "
     "averages five.")
para("**Regime shifts.** System-level patterns learned from 2015–2022 transferred well to 2023 but not to the "
     "2024–2026 configuration of direct Israel–Iran exchanges and a wider Red Sea conflict. A model can only "
     "attend to relationships whose dynamics it has seen.")
para("**One input, with known biases.** GDELT counts track media attention as much as events, coding errors and "
     "duplicates are common, and the Goldstein average mixes very different events. Figure 14 shows a direct "
     "consequence: after 7 October 2023 the Israel → Palestine mean Goldstein score *rose*, because the war "
     "generated many statements, appeals and humanitarian events with positive scores, so averaged event scores "
     "can understate a crisis precisely when it is most severe.")
figure("fig10_gdelt", "GDELT understates the October 2023 crisis for Israel → Palestine. (a) Weekly mean Goldstein "
       "score. (b) Shares of cooperative (CAMEO 01–09) and violent (CAMEO 18–20) events.", 15.5)
para("**A short history from a single instrument.** GDELT 2.0 begins in 2015, so the model learns from about "
     "eleven years in which the region went through only a handful of system-wide crises. Earlier GDELT 1.0 data "
     "cannot simply be appended: it draws on a different set of sources and coding rules, and adding it made "
     "forecasts worse (Section 4.3). Because the model is limited by data rather than by architecture, this "
     "ceiling is set by the instrument, not by the method; the results reported here should be read as a lower "
     "bound on what the approach can achieve with a longer, consistently coded record.")
para("**Gradual shifts rather than sudden breaks.** The model anticipates gradual drifts better than abrupt "
     "breaks, and is best used as a horizon-scanning tool alongside analyst judgement on discrete crises.")
para("**Interpretability.** Attention patterns (Section 4.5) indicate where information flows but do not "
     "establish what drives a given forecast.")

# ============================================================== 6
doc.add_heading("7  Scaling up", level=1)
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
     "problem in Figure 14 at its source, and requires GPU infrastructure beyond a single laptop.")
para("**A longer record through in-house coding.** The largest gain is likely to come from an event-coding "
     "pipeline of our own, applied uniformly to a news archive reaching back several decades. One instrument "
     "across that span would turn eleven years of training data into thirty or more, with many more crises, "
     "ceasefires and realignments to learn from, the kind of shifts the current model has seen too rarely to "
     "anticipate (Section 5, regime shifts). Owning the coding also means controlling what is measured: actors "
     "resolved consistently, hostile and cooperative events kept apart, and intensity recorded rather than "
     "averaged away. Given how directly v2.2 improved with more same-instrument data, a record of that length "
     "could change the results materially, and the model, evaluation protocol and live pipeline built here "
     "carry over to it unchanged.")
para("**From point forecasts to calibrated risk.** Operational use needs uncertainty: quantile outputs, seed and "
     "data ensembles as uncertainty estimates, and regular re-training. The live pipeline already re-scores the "
     "full system every two weeks from the latest complete week of GDELT data, so these extensions slot into an "
     "existing production loop.")

# ============================================================== 7
doc.add_heading("8  Conclusion", level=1)
para(
    "This section has demonstrated a new way of applying deep learning to forecasting international relations: "
    "each directed country pair becomes a token, a temporal transformer models each relationship's history, and "
    "a cross-system transformer forecasts the region jointly while attending to its relations with the rest of "
    "the world. Built on a single open data source and evaluated under a strict out-of-sample protocol, the "
    "prototype forecasts quarterly relations among 400 MENA dyads far more accurately than a linear model on the "
    "same raw inputs and than naive forecasts, outperforms gradient boosting, and matches a linear model built "
    "on hand-engineered features without needing that engineering; trained on more dyads from the same "
    "instrument, it moves ahead of it. A second model with the same architecture forecasts the worst month of "
    "conflict in the coming quarter and flags conflict surges far better than forecasts of average relations. "
    "Both run live every two weeks and feed a bilingual risk monitor that combines them with observed violence, "
    "trade and shipping data into a transparent country risk score for organisations with trade at stake. The experiment programme shows "
    "when the systemic view pays off: with enough data and when crises propagate across the system, with an "
    "advantage that grows as data grow. The model is data-limited rather than feature-limited, which makes a "
    "global, same-instrument system, and a longer record coded by an in-house pipeline, the natural next steps; "
    "the framework built here is designed to take them.")

# ============================================================== references
doc.add_heading("References", level=1)
for ref in [
    "Gerner, D. J., Schrodt, P. A., Yilmaz, Ö., and Abu-Jabr, R. (2002). Conflict and Mediation Event Observations "
    "(CAMEO): A new event data framework for the analysis of foreign policy interactions. Paper presented at the "
    "Annual Meeting of the International Studies Association, New Orleans.",
    "Goldstein, J. S. (1992). A conflict-cooperation scale for WEIS events data. *Journal of Conflict Resolution*, "
    "36(2), 369–385.",
    "International Monetary Fund and University of Oxford. *IMF PortWatch: monitoring trade disruptions from "
    "space*. https://portwatch.imf.org (chokepoint transit data, accessed October 2026).",
    "Hegre, H., Allansson, M., Basedau, M., Colaresi, M., Croicu, M., Fjelde, H., et al. (2019). ViEWS: A political "
    "violence early-warning system. *Journal of Peace Research*, 56(2), 155–174.",
    "Hegre, H., Bell, C., Colaresi, M., Croicu, M., Hoyles, F., Jansen, R., et al. (2021). ViEWS2020: Revising and "
    "evaluating the ViEWS political Violence Early-Warning System. *Journal of Peace Research*, 58(3), 599–611.",
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

doc.save(f"{P}/transformer_section.docx")
print("saved", f"{P}/transformer_section.docx", "| figures:", fig_no[0], "| tables:", tab_no[0])

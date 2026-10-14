"""
Build the paper section (release v2.1) as a Word document from saved results.

    KERAS_BACKEND=torch python paper/compute_v21_stats.py    # v2.1 numbers + attention analysis
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
para("Section draft for the Risk-A-Lab geopolitical risk forecasting paper. Model release v2.1. All results are "
     "reproducible from the project repository (benchmark_honest.py, benchmark_ladder.py, experiment_*.py, "
     "release_v2.1_goldstein.py and the paper/ scripts).", italic=True, size=9.5, color=GREY, align="left", after=12)

doc.add_heading("Section summary", level=2)
para(
    "Most operational conflict-forecasting systems treat each country or country pair as a separate row of data "
    "and predict it from its own history, with the rest of the world entering, if at all, through hand-built "
    "aggregate features. This section develops and tests an alternative, closer to how states make decisions: a "
    "transformer that forecasts every relationship in the system jointly. Each directed country pair (dyad) is a "
    "token; a temporal transformer encodes each dyad's recent history, and a cross-dyad transformer lets every "
    "dyad attend to every other. The released model (v2.1) forecasts the 400 dyads among 20 Middle East and "
    "North Africa (MENA) states while attending to a wider system of 1,959 dyads that also includes the region's "
    "relations with the rest of the world. Using GDELT event data aggregated by week, it forecasts each dyad's "
    f"mean Goldstein cooperation–conflict score over the next 13 weeks. Under a strict out-of-sample protocol it "
    f"reaches a test R² of {f3(V21['r2'])}, ahead of the strongest naive forecast (a trailing 52-week mean, "
    f"{f3(B['trailing 52-week mean']['r2'])}) and of a gradient-boosting model on 44 hand-engineered features "
    f"({f3(GBM['r2'])}), and it is also ahead on every measure of predicted *change*.")
para(
    "A systematic programme of experiments clarifies what drives these gains and what limits them. Attention "
    "across the system helps when there is enough training data and when crises propagate across the region, "
    "as in 2023, and its advantage grows with data. The model is currently limited by the number of training "
    "examples rather than by the richness of its inputs. The contribution is methodological: a demonstration "
    "of how self-attention can bring the international system into dyad-level forecasts, a working prototype "
    "that re-forecasts the region every two weeks from live data, and evidence-based directions for scaling the "
    "approach. Elsewhere in the paper the model supplies the likelihood component of a risk index for "
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
    "MENA dyads.")
para(
    "The target is the **mean Goldstein score of a dyad over the next 13 weeks**. Inputs for each week are the "
    "log counts of the 20 CAMEO root event types, the week's mean Goldstein score, and three long-memory "
    "features: the dyad's trailing 26- and 52-week mean Goldstein and its 52-week activity level (Table 1).")
table("Data and model specification (release v2.1).", ["Item", "Specification"],
      [["Source", "GDELT 2.0 events, public BigQuery dataset gdelt-bq.gdeltv2.events"],
       ["Forecast targets", "400 directed MENA dyad-weeks, incl. domestic"],
       ["System (tokens per week)", "400 MENA dyads + 1,559 MENA–world dyads = 1,959"],
       ["Period", f"{d['first_week']} to {d['last_week']} ({d['weeks']} weeks)"],
       ["Target", "Mean Goldstein score over the next 13 weeks (continuous)"],
       ["Inputs per week", "log CAMEO root counts (20), mean Goldstein, trailing 26/52-week mean Goldstein, "
                           "52-week activity (24 features)"],
       ["Look-back", "13 weeks directly; up to 64 weeks through long-memory features"],
       ["Model size", f"{R['params']:,} parameters per model; 5-model seed ensemble"]], [4.0, 12.0])

doc.add_heading("3.2  Architecture", level=2)
para(
    "Figure 2 shows the architecture. Each dyad's 13-week input window is projected to 64-dimensional tokens "
    "with a learned temporal position embedding and passed through two **temporal transformer** blocks, in which "
    "the dyad attends only over its own weeks. The representation of the latest week is concatenated with two "
    "**country embeddings**, learned 8-dimensional vectors for the source and target state from one shared "
    "table, so that 'Iran' means the same on either side of a dyad while direction is preserved by position. "
    "The resulting dyad tokens, 1,959 per week in v2.1, then pass through two **cross-system transformer** "
    "blocks in which every dyad attends to every other. All blocks use four-head attention, SwiGLU feed-forward "
    "layers (Shazeer 2020), residual connections and layer normalisation.")
para(
    "The output is **residual**: the network produces a correction that is added to the dyad's trailing 52-week "
    "mean, so the model starts from the strongest naive forecast and learns when and how far the coming quarter "
    "will depart from a relationship's normal. Forecasts are read off the 400 MENA tokens. The release averages "
    "five identically configured models trained from different random seeds.")
figure("fig2_architecture", "Architecture of release v2.1. Shaded blocks are the two attention stages: temporal "
       "attention within each dyad's own history, then cross-system attention across all 1,959 dyads of the "
       "week. Tensor shapes in brackets.", 16)

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
    "point below normal). Benchmarks are naive forecasts; a **competing algorithm**, gradient boosting "
    "(HistGradientBoosting) on 44 hand-engineered lag features including the reverse dyad, both countries' "
    "context and a region-wide average, i.e. a strong row-by-row model of the kind used in operational systems; "
    "and earlier versions of the transformer.")

# ============================================================== 4
doc.add_heading("4  Results", level=1)
doc.add_heading("4.1  Benchmark performance", level=2)
para(
    f"Table 2 and Figure 4 report test performance. Release v2.1 reaches R² = {f3(V21['r2'])} and MAE = "
    f"{f3(V21['mae'])} Goldstein points, ahead of every naive forecast, of gradient boosting ({f3(GBM['r2'])}) "
    f"and of the regional-attention v2.0 ({f3(V20['r2'])}). It is also ahead on all three change measures: "
    f"direction accuracy {pct(V21['dir_acc'])} against {pct(GBM['dir_acc'])} for gradient boosting, change "
    f"correlation {V21['change_corr']:.3f} against {GBM['change_corr']:.3f}, and escalation AUC "
    f"{V21['esc_auc']:.3f} against {GBM['esc_auc']:.3f}. Relative to the trailing 52-week mean it reduces "
    f"squared error by {R['variance']['mse_reduction']:.1%} and is more accurate for "
    f"{R['variance']['share_dyads_better']:.0%} of the 400 dyads. The trailing 52-week mean is a demanding "
    "baseline: a smoothed quarterly target is largely persistent, and carrying a relationship's annual average "
    "forward already explains 60% of test variance.")
table("Test-set performance (400 MENA dyads × 119 weeks, March 2024 – June 2026).",
      ["Forecast", "R²", "MAE", "Direction", "Change corr.", "Escalation AUC"],
      [["Trailing 13-week mean", f3(B["trailing 13-week mean"]["r2"]), f3(B["trailing 13-week mean"]["mae"]),
        pct(L["trailing 13-week mean"]["dir_acc"]), f3(L["trailing 13-week mean"]["change_corr"]),
        f3(L["trailing 13-week mean"]["esc_auc"])],
       ["Trailing 52-week mean", f3(B["trailing 52-week mean"]["r2"]), f3(B["trailing 52-week mean"]["mae"]),
        "—", "—", "0.500"],
       ["Rudimentary network (counts + Goldstein + embeddings)", f3(L["rudimentary: counts + goldstein + embeddings"]["r2"]),
        f3(L["rudimentary: counts + goldstein + embeddings"]["mae"]), pct(L["rudimentary: counts + goldstein + embeddings"]["dir_acc"]),
        f3(L["rudimentary: counts + goldstein + embeddings"]["change_corr"]), f3(L["rudimentary: counts + goldstein + embeddings"]["esc_auc"])],
       ["Gradient boosting, 44 lag features", f3(GBM["r2"]), f3(GBM["mae"]), pct(GBM["dir_acc"]),
        f3(GBM["change_corr"]), f3(GBM["esc_auc"])],
       ["v2.0: regional attention (400 tokens)", f3(V20["r2"]), f3(V20["mae"]), pct(V20["dir_acc"]),
        f3(V20["change_corr"]), f3(V20["esc_auc"])],
       ["v2.1: cross-system attention (1,959 tokens)", f3(V21["r2"]), f3(V21["mae"]), pct(V21["dir_acc"]),
        f3(V21["change_corr"]), f3(V21["esc_auc"])]],
      [6.2, 1.6, 1.6, 2.0, 2.2, 2.4], bold_last=True,
      note="Direction and change measures are relative to each dyad's trailing 52-week mean, so they are undefined "
           "for that baseline. Persistence of the latest week scores R² −1.62. The rudimentary network sees only the "
           "current week (no history, no transformer).")
figure("fig4_benchmark", "Test R² of the release models against naive forecasts, the earlier transformer recipe and "
       "gradient boosting. The axis starts at 0.45.", 15.5)

doc.add_heading("4.2  What produced the improvement", level=2)
pv = list(PATH.values())
para(
    "Figure 5 and Table 3 trace the path from the original v1.0 recipe to the release, adding one change at a "
    f"time. The original recipe, once evaluated honestly, scored {pv[0]:.3f}, below the trailing mean. Stronger "
    f"regularisation (+{pv[1] - pv[0]:.3f}), the residual formulation (+{pv[2] - pv[1]:.3f}) and the long-memory "
    f"inputs (+{pv[3] - pv[2]:.3f}) contributed similar amounts; averaging five seeds added +{pv[4] - pv[3]:.3f} "
    f"(v2.0), and extending attention to the region's relations with the rest of the world added "
    f"+{pv[5] - pv[4]:.3f} (v2.1). No single change was sufficient; together they move the model from below the "
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
         ["v2.1: + 1,559 MENA–world dyads, cross-system attention, 5 seeds", f"{R['v21_val_mse']:.3f}", f3(V21["r2"])]]
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
    "richness of its inputs, which is what motivates the scaling analysis below.")
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
ext = R["attention"]["external_share"]
para(
    "Attention weights are not explanations of a model's output (Jain and Wallace 2019), but they show where "
    "information flows. Averaged over heads, MENA forecasts spread attention across MENA–world dyads roughly in "
    f"proportion to their number ({min(ext.values()):.0%}–{max(ext.values()):.0%} of attention, against "
    f"{R['attention']['uniform_external_share']:.0%} if uniform). Two heads, however, are sharply selective "
    "(Figure 10). The most selective attends up to four times more than uniform to the United Kingdom, Germany "
    "and the European Union in relation to Israel, the United States and United Kingdom in relation to Palestine, "
    "and Russia in relation to Iran. The second concentrates on Spain, Ireland, Germany, Canada and the European "
    "Union in relation to Israel; Spain and Ireland recognised Palestine in 2024 and were central to Israel's "
    "diplomatic relations. Without being told which relationships matter, the model has learned to watch how "
    "outside powers engage with the region's central conflict, which is precisely the systemic information that "
    "row-by-row models cannot use.")
figure("fig8_attention", "Dyads receiving the most attention from MENA forecasts in the two most selective "
       "cross-system heads, as a multiple of uniform attention (test-period average). Blue: MENA–world dyads; "
       "green: MENA–MENA dyads.", 15.5)

doc.add_heading("4.6  Where the model adds information", level=2)
vv = R["variance"]
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
figure("fig7_examples", "Forecast versus outcome for four dyads over the test period (release v2.1). Titles give "
       "the change in squared error relative to the trailing 52-week mean (negative = better).", 16)

doc.add_heading("4.7  A second task: four-week direction", level=2)
para(
    "A companion model with the same architecture, structural-relation biases in the attention stage, a richer "
    "GDELT feature set and the United States and China as context actors classifies whether a dyad's next four "
    f"weeks will be more conflictual than the current week (escalation). It reaches {DIR['accuracy']:.1%} test "
    "accuracy (ROC-AUC 0.894), level with gradient boosting (81.6%) and ahead of a one-line mean-reversion rule "
    "(78.1%) and the majority class (53.6%) (Figure 12a). When the current week is far from its recent mean, "
    "regression to the mean makes the direction easy (≈95% for every method). In the hardest quartile, where "
    f"the current week is near normal, the transformer is correct {DIR['q_transformer'][0]:.0%} of the time "
    f"against {DIR['q_rule'][0]:.0%} for the rule, which is the regime that matters for early warning.")
figure("fig9_direction", "Four-week direction task. (a) Test accuracy. (b) Accuracy by how far the current week is "
       "from its 13-week mean (quartiles).", 16)

doc.add_heading("4.8  From forecast to risk product", level=2)
para(
    "Elsewhere in this paper the forecasts feed a country-level risk index for commercial users, in which risk "
    "is the product of likelihood and impact. The transformer supplies the likelihood component: a calibrated "
    "probability that a country's relations deteriorate materially below their own 52-week normal over the "
    "next quarter, which ranks countries with ROC-AUC 0.69 on held-out data. One finding from that work belongs "
    "here because it concerns model outputs: the direction model's escalation probability, used naively as a "
    "likelihood, ranks countries *worse* than random (AUC 0.37), because a high value mostly flags a country "
    "that has just had an unusually good week. Outputs must be validated against the decision they serve, not "
    "only against their training target.")

# ============================================================== 5
doc.add_heading("5  Limitations", level=1)
para("As a prototype, the model has limitations that define its current scope and point to the next steps in "
     "Section 6.")
para("**Data volume.** The model is trained on 391 weeks of the regional system, and Section 4.3 showed it is "
     "limited by the number of training examples rather than by input richness. Seeds vary "
     f"({min(R['v21_seed_r2']):.3f}–{max(R['v21_seed_r2']):.3f} test R² for v2.1), which is why the release "
     "averages five.")
para("**Regime shifts.** System-level patterns learned from 2015–2022 transferred well to 2023 but not to the "
     "2024–2026 configuration of direct Israel–Iran exchanges and a wider Red Sea conflict. A model can only "
     "attend to relationships whose dynamics it has seen.")
para("**One input, with known biases.** GDELT counts track media attention as much as events, coding errors and "
     "duplicates are common, and the Goldstein average mixes very different events. Figure 13 shows a direct "
     "consequence: after 7 October 2023 the Israel → Palestine mean Goldstein score *rose*, because the war "
     "generated many statements, appeals and humanitarian events with positive scores, so averaged event scores "
     "can understate a crisis precisely when it is most severe.")
figure("fig10_gdelt", "GDELT understates the October 2023 crisis for Israel → Palestine. (a) Weekly mean Goldstein "
       "score. (b) Shares of cooperative (CAMEO 01–09) and violent (CAMEO 18–20) events.", 15.5)
para("**Gradual shifts rather than sudden breaks.** The model anticipates gradual drifts better than abrupt "
     "breaks, and is best used as a horizon-scanning tool alongside analyst judgement on discrete crises.")
para("**Interpretability.** Attention patterns (Section 4.5) indicate where information flows but do not "
     "establish what drives a given forecast.")

# ============================================================== 6
doc.add_heading("6  Scaling up", level=1)
para("The experiments point to where scale is most likely to pay off.")
para("**More of the same instrument.** The learning curve shows the attention model still improving at full "
     "data, and same-instrument examples (MENA–world dyads) helped where different instruments (GDELT 1.0) did "
     "not. A global model, trained on all states' dyads coded the same way, multiplies the training data by "
     "an order of magnitude. Full attention over all of them is expensive, since cost grows with the square of "
     "the number of tokens, but sparse or hierarchical attention (dyads attend within regions, regions to each "
     "other) or attention restricted to active relationships keeps it tractable. The 1,959-token release "
     "already trains in minutes on a single consumer GPU.")
para("**From coded events to text, at the right scale.** A pretrained sentence model applied to one newspaper's "
     "headlines added no signal here; the coverage was too sparse (8% of dyad-weeks) and too generic. The "
     "promising route is dyad-specific representations learned from large multi-source news corpora, with "
     "encoders fine-tuned to extract actors, event types, intensity and certainty. That also fixes the dilution "
     "problem in Figure 13 at its source, and requires GPU infrastructure beyond a single laptop.")
para("**From point forecasts to calibrated risk.** Operational use needs uncertainty: quantile outputs, seed and "
     "data ensembles as uncertainty estimates, and regular re-training. The live pipeline already re-scores the "
     "full system every two weeks from the latest complete week of GDELT data, so these extensions slot into an "
     "existing production loop.")

# ============================================================== 7
doc.add_heading("7  Conclusion", level=1)
para(
    "This section has demonstrated a new way of applying deep learning to forecasting international relations: "
    "each directed country pair becomes a token, a temporal transformer models each relationship's history, and "
    "a cross-system transformer forecasts the region jointly while attending to its relations with the rest of "
    "the world. Built on a single open data source and evaluated under a strict out-of-sample protocol, the "
    "prototype forecasts quarterly relations among 400 MENA dyads more accurately than naive forecasts and a "
    "strong gradient-boosting benchmark, on levels and on change, runs live every two weeks, and supplies the "
    "likelihood component of the risk index developed elsewhere in this paper. The experiment programme shows "
    "when the systemic view pays off: with enough data and when crises propagate across the system, with an "
    "advantage that grows as data grow. The model is data-limited rather than feature-limited, which makes a "
    "global, same-instrument system the natural next step, and the framework built here is designed to take it.")

# ============================================================== references
doc.add_heading("References", level=1)
for ref in [
    "Gerner, D. J., Schrodt, P. A., Yilmaz, Ö., and Abu-Jabr, R. (2002). Conflict and Mediation Event Observations "
    "(CAMEO): A new event data framework for the analysis of foreign policy interactions. Paper presented at the "
    "Annual Meeting of the International Studies Association, New Orleans.",
    "Goldstein, J. S. (1992). A conflict-cooperation scale for WEIS events data. *Journal of Conflict Resolution*, "
    "36(2), 369–385.",
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

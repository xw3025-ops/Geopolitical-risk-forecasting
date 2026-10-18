"""
Chinese edition of the merged paper (written in Chinese, not translated; just under 5,000 Chinese
characters). Same layout as paper/build_paper.py; numbers come from the same files; figures are the
English originals with Chinese captions.

    python paper/build_paper_zh.py      -> paper/risk_paper_zh.docx
"""

import json
import math
import re
import sys

import joblib
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

P = "paper"
S = json.load(open(f"{P}/stats.json"))
R = json.load(open(f"{P}/v21_stats.json"))
LIN = json.load(open(f"{P}/linear_baselines.json"))
V22S = json.load(open(f"{P}/v22_stats.json"))
A22 = json.load(open(f"{P}/v22_analysis.json"))
SITE = json.load(open("website/data/latest.json", encoding="utf-8"))
SA = joblib.load("artifacts/surge_v2.2_preprocessing.joblib")
B = S["goldstein_benchmark"]
V21, GBM, V22, FL = R["benchmark"]["v2.1"], R["gbm"], V22S["v2.2"], V22S["flag"]
OLS1, OLS44 = LIN["OLS current week"], LIN["OLS 44 history features"]
SURG = FL["v2.2 conflict-surge model"]
DIRN = FL["v2.0 direction model (4-week escalation probability)"]
CAL, THR = SA["calibration"], SA["surge_threshold"]
p_at = lambda gap: 1 / (1 + math.exp(-(CAL["a"] * gap + CAL["b"])))
att = A22["attention"]
lift = lambda dyad: next(x["lift"] for x in att["received"][att["selective_heads"][0]] if x["dyad"] == dyad)
f3 = lambda x: f"{x:.3f}"
pc = lambda x: f"{100 * x:.1f}%"
p0 = lambda x: f"{100 * x:.0f}%"

FIGS = ["fig1_concept", "fig10_gdelt", "fig2_architecture", "fig3_timeline", "fig4_benchmark", "fig5_path",
        "fig6_ablation", "fig13_backtest", "fig12_learning_curve", "fig11_scaling", "fig8_attention",
        "fig7_examples", "fig14_surge_isr", "fig_trade_1_closure", "fig_trade_2_regression",
        "fig_trade_3_backtest", "fig15_website"]
TABS = ["components", "data", "bench", "flags", "closure", "decision"]
fig = lambda n: f"图{FIGS.index(n) + 1}"
tab = lambda k: f"表{TABS.index(k) + 1}"

# ------------------------------------------------------------------ document and styles (as the English edition)
LATIN, CJK = "Times New Roman", "SimSun"
INK, GREY = RGBColor(0x0B, 0x0B, 0x0B), RGBColor(0x52, 0x51, 0x4E)
def fonts(style, cjk=CJK):
    style.font.name = LATIN
    rpr = style.element.get_or_add_rPr()
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):  # theme fonts override the named ones
        rpr.rFonts.attrib.pop(qn(a), None)
    rpr.rFonts.set(qn("w:eastAsia"), cjk)
    lang = OxmlElement("w:lang")  # zh-CN: Chinese line-breaking rules
    lang.set(qn("w:val"), "en-US")
    lang.set(qn("w:eastAsia"), "zh-CN")
    rpr.append(lang)


def new_document():
    d = Document()
    sec = d.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.4)
    sec.top_margin = sec.bottom_margin = Cm(2.2)
    st = d.styles["Normal"]
    fonts(st)
    st.font.size = Pt(11)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.line_spacing = 1.15
    for lvl, size in [(1, 14), (2, 12)]:
        h = d.styles[f"Heading {lvl}"]
        fonts(h, "SimHei")  # 黑体 headings, 宋体 body
        h.font.size, h.font.bold, h.font.italic = Pt(size), True, False
        h.font.color.rgb = INK
        h.paragraph_format.space_before = Pt(14 if lvl == 1 else 10)
        h.paragraph_format.space_after = Pt(4)
        h.paragraph_format.keep_with_next = True
    return d


doc = new_document()
tdoc = new_document()  # the tables, one per page; rendered by Word and placed in the paper as images
TABLE_DIR = f"{P}/zh_tables"
pending = []  # (key, placeholder paragraph)
fig_no, tab_no = [0], [0]


def _runs(p, text, size=None, color=None, italic=False):
    for part in re.split(r"(\*\*[^*]+\*\*)", text):
        if not part:
            continue
        r = p.add_run(part[2:-2] if part.startswith("**") else part)
        r.bold = part.startswith("**")
        r.italic = italic
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = color


def para(text, align=None, size=None, color=None, after=None):
    p = doc.add_paragraph()
    _runs(p, text, size=size, color=color)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT if align == "left" else \
        WD_ALIGN_PARAGRAPH.CENTER if align == "center" else WD_ALIGN_PARAGRAPH.JUSTIFY
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    if align is None:  # body text: first-line indent of two characters
        p._p.get_or_add_pPr().get_or_add_ind().set(qn("w:firstLineChars"), "200")
    return p


_num = {"el": None, "n": 0}


def numbered(text):
    last = doc.element.body[-2] if len(doc.element.body) > 1 else None
    _num["n"] = _num["n"] + 1 if last is not None and last is _num["el"] else 1
    p = doc.add_paragraph()
    _runs(p, f"{_num['n']}.\t{text}")
    p.paragraph_format.left_indent = Cm(0.75)
    p.paragraph_format.first_line_indent = Cm(-0.75)
    p.paragraph_format.tab_stops.add_tab_stop(Cm(0.75))
    p.paragraph_format.space_after = Pt(2)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    _num["el"] = p._p


def figure(name, caption, width=15.5):
    fig_no[0] += 1
    assert FIGS.index(name) + 1 == fig_no[0], name
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(f"{P}/{name}.png", width=Cm(width))
    c = doc.add_paragraph()
    _runs(c, f"**图{fig_no[0]}** {caption}", size=9.5, color=GREY)
    c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    c.paragraph_format.space_after = Pt(10)


def shade(cell, fill):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shd)


def table(key, caption, header, rows, widths, merges=(), note=None):
    tab_no[0] += 1
    assert TABS.index(key) + 1 == tab_no[0], key
    c = doc.add_paragraph()
    _runs(c, f"**表{tab_no[0]}** {caption}", size=9.5, color=GREY)
    c.paragraph_format.keep_with_next = True
    c.paragraph_format.space_after = Pt(3)
    ph = doc.add_paragraph()  # the table image goes here (render_tables)
    ph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    ph.paragraph_format.space_after = Pt(0)
    pending.append((key, ph))
    t = tdoc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(header):
        cell = t.rows[0].cells[j]
        cell.text = ""
        _runs(cell.paragraphs[0], f"**{h}**", size=9)
        shade(cell, "E8E7E3")
    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            cell = t.rows[i].cells[j]
            cell.text = ""
            if v:
                _runs(cell.paragraphs[0], v, size=9)
            if j > 0 and not merges:
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for i, row in enumerate(t.rows):
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for j, w in enumerate(widths):
            row.cells[j].width = Cm(w)
            for cp in row.cells[j].paragraphs:
                cp.paragraph_format.keep_with_next = i < len(t.rows) - 1
    for (r0, c0), (r1, c1) in merges:
        m = t.cell(r0 + 1, c0).merge(t.cell(r1 + 1, c1))
        for extra in m.paragraphs[1:]:
            if not extra.text:
                extra._p.getparent().remove(extra._p)
        m.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        shade(m, "F4F3F0")
    tdoc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    n = doc.add_paragraph()
    if note:
        _runs(n, note, size=8.5, color=GREY)
    n.paragraph_format.space_after = Pt(10 if note else 4)


H1 = lambda t: doc.add_heading(t, level=1)
H2 = lambda t: doc.add_heading(t, level=2)

# ============================================================== title, abstract
t = doc.add_paragraph()
r = t.add_run("中东和北非地区地缘政治风险与中国贸易：基于体系注意力的冲突预测、贸易暴露测度与动态风险监测")
r.bold, r.font.size = True, Pt(16)
t.paragraph_format.space_after = Pt(4)
para("Risk-A-Lab 地缘政治风险预测项目（冲突模型 v2.2，2026年10月）", align="left", size=9.5, color=GREY, after=12)

H2("摘要")
para(
    "中资企业在中东和北非地区面临的冲突风险来自两条渠道：一是经营所在地的暴力与关系恶化，二是承载中国与该地区"
    f"贸易的海上通道受阻。本文围绕这两条渠道构建地缘政治风险预警体系。在冲突预测上，本文把每组有向国家关系视为一个词元（token），以时间注意力和跨体系注意力联合预测地区内外1,959组关系，将国际体系结构纳入双边层面的预测。严格样本外检验显示，发布模型对未来13周关系走向的预测决定系数为{f3(V22['r2'])}，"
    "显著优于各类基准；同架构的冲突激增模型每周前十项预警中有"
    f"{p0(SURG['conflict_surge']['top10'])}随后应验，而基准概率仅为{p0(V22S['surge_rate'])}。在贸易上，本文发现冲突主要"
    "经由航运通道而非伙伴国自身动荡冲击中国贸易：2026年3月霍尔木兹海峡事实封锁后，中国自伊拉克、卡塔尔和科威特的"
    "进口同比下降83%至89%。上述成果整合为透明的国家风险评分，经每周自动更新的双语平台发布。")
para("**关键词：**地缘政治风险；冲突预测；自注意力机制；霍尔木兹海峡；海外利益保护", align="left")

# ============================================================== 一 引言
H1("一、引言")
para(
    "VIEWS、ConflictForecast 等业务化冲突预测系统多以随机森林、梯度提升等方法为核心，把每个国家或国家对视为独立样本，仅依据其自身历史和预先构造的汇总特征作出预测。这种“逐行预测”与国际关系的基本事实"
    "存在张力。一国对另一国的政策从不孤立形成，以色列对伊朗的立场同时受以色列与胡塞武装、伊朗与海湾国家以及各方"
    "与大国关系的影响；体系结构本身就是解释变量。本文的出发点是：自注意力机制恰好能够表示这种结构（"
    f"{fig('fig1_concept')}）。在此基础上，本文考察冲突向中国贸易的传导，并将两方面结果整合为风险产品。")
figure("fig1_concept", "逐行预测（a）与体系感知预测（b）。")

# ============================================================== 二 风险定义
H1("二、风险定义")
para(
    "本文所称风险，是指在某国拥有贸易利益的机构（出口商、能源采购方、工程承包商、投资者、航运企业及驻外人员）面临的经营性风险。风险高须同时满足两个条件：未来一个季度的冲突环境预计严峻，且涉及的贸易规模较大。"
    "处于战争状态但对华贸易很少的国家，或贸易规模大而局势平稳的国家，都不应居于榜首。据此，国家风险定义为冲突严重度"
    "与涉及贸易两项得分的几何平均：")
para("风险 = √（冲突严重度 × 涉及贸易）", align="center")
para(
    "几何平均既保持0—100的量纲，又使任一因素都无法单独推高风险。风险分为严重（≥55）、高（≥45）、偏高（≥32）和低"
    "四档，地区风险取20国均值。需要说明两点：其一，评分用于排序而非概率；"
    "其二，航运通道风险单列而不并入国家评分，因为海峡封锁同时冲击多国，且其量级仅由一次事件识别，折算进单一国家"
    "得分反而会掩盖风险来源。")

# ============================================================== 三 指标体系
H1("三、指标体系构建逻辑")
H2("（一）指标层级")
para(f"指标体系分为三层：Transformer 预测的冲突前景、已观测的现实条件，以及以海峡和航道为核心的传导渠道（{tab('components')}）。")
table("components", "风险指标体系",
      ["评分", "构成", "指标", "来源", "性质", "折算（权重）"],
      [["**风险** = √（严重度 × 涉及贸易）", "**冲突严重度**", "预测敌对程度", "v2.2 平均戈德斯坦预测", "预测",
        "+1.5→0，−4.0→1（0.40）"],
       [None, None, "冲突激增风险", "v2.2 激增概率", "预测", "0→0，0.6→1（0.30）"],
       [None, None, "近期暴力", "ACLED 近4周死亡人数", "观测", "对数，2,000人→1（0.30）"],
       [None, "**涉及贸易**", None, "IMF 近12个月对华贸易额", "观测", "对数，相对最大伙伴"],
       ["**评分外单列**", "霍尔木兹暴露度", None, "2024年能源进口 × 无法绕行比例", "观测", "0—1，高低情景"],
       [None, "航道状态", None, "PortWatch 通行量对比常态", "观测", "降幅≥50%为严重受阻"],
       [None, "贸易前景", None, "收缩持续性预测", "预测", "变化率及80%区间"]],
      [2.9, 2.4, 2.4, 3.9, 1.4, 3.2],
      merges=[((0, 0), (3, 0)), ((0, 1), (2, 1)), ((3, 1), (3, 2)), ((4, 0), (6, 0)), ((4, 1), (4, 2)),
              ((5, 1), (5, 2)), ((6, 1), (6, 2))])

H2("（二）构建原则")
para(
    "指标构建遵循六项原则：统一采用13周预测期，兼顾可预测性与机构的反应时间；一切变化都相对于该关系自身近一年的常态"
    "衡量，避免把长期水平差异误读为风险变化；敌对程度按事件数量和强度累计而非取平均，防止合作事件稀释冲突信号"
    f"（{fig('fig10_gdelt')}）；国家指标按各关系的活跃度加权；死亡人数与贸易额取对数；全部权重和参数事先设定，不依据结果拟合，以免循环论证。")

H2("（三）冲突激增指标")
para("冲突激增指标是本文在指标设计上的主要创新，回答的是“未来一个季度最坏会到什么程度”。其构建分为五步：")
numbered("**周度冲突强度。**将某关系当周全部敌对事件（威胁、制裁、驱逐、袭击等）的戈德斯坦得分取绝对值求和后取对数，"
         "合作事件不计入。")
numbered("**未来最坏月份。**未来13周内冲突强度的最高四周均值。采用四周窗口，是为了过滤低活跃度关系的偶然波动。")
numbered("**关系常态。**对过去52周的每一周，回看此前13周取最高四周均值，再求平均，即该关系近一年的典型季度峰值。"
         "每组关系只与自身比较。")
numbered(f"**激增判定。**未来最坏月份超出常态{THR:.2f}（对数尺度）以上，约相当于常态峰值的{math.exp(THR):.1f}倍。"
         "该阈值取2015—2022年训练期全部关系差值分布的前10%。")
numbered(f"**概率与预警。**模型预测的差值经2023年验证数据拟合的逻辑曲线转为概率：预测差值为零时约为{p0(p_at(0))}，"
         f"达到{-CAL['b'] / CAL['a']:.2f}时为50%。每周概率最高的10组关系列为预警。")
para(
    "由于基准是各关系自身的近期峰值，该指标刻画的是新的升级，而非绝对危险程度。长期处于战争状态但强度平稳的关系，"
    "激增概率未必高，这部分风险由敌对程度预测和近期暴力承担。")

H2("（四）霍尔木兹暴露度")
para(
    "伙伴国的暴露额等于中国2024年自该国的原油与天然气进口分别乘以无法绕行海峡的比例之和，比例依据 EIA 与 IEA 的出口设施资料按高低情景设定。测算表明，2024年中国受该海峡影响的能源进口为"
    "797亿至1,071亿美元，占进口总额的3.1%至4.1%，集中于伊拉克、沙特、阿联酋、卡塔尔和科威特；阿曼的出口码头位于"
    "海峡之外，不受直接影响。")

# ============================================================== 四 数据编码与扫描分析
H1("四、数据编码与扫描分析过程")
H2("（一）事件数据编码")
para(
    "冲突模型以 GDELT 2.0 为唯一输入。该库以机器编码方式把新闻报道转化为带 CAMEO 事件类型的双边事件，并赋予−10（最具冲突性）至+10（最具合作性）的戈德斯坦得分。事件按周汇总为有向国家对，覆盖地区20国之间的400组关系（含各国国内），自2015年起约600周；另以1,559组地区与区外国家的"
    f"关系作为体系背景，以{V22S['world_dyads']:,}组区外国家间关系作为额外训练样本。GDELT 记录的是媒体报道而非实地事件，危机中平均得分可能失真：2023年10月7日后，以色列对巴勒斯坦关系的平均得分不降反升，原因在于战争伴随大量人道"
    f"和外交事件（{fig('fig10_gdelt')}）。激增指标只计敌对事件，正是针对这一问题。")
figure("fig10_gdelt", "以色列对巴勒斯坦关系的周平均得分与事件构成。")

H2("（二）模型结构")
para(
    f"模型设计的核心，是把国际体系直接写进网络结构（{fig('fig2_architecture')}）。每组关系近13周的输入先经两层时间"
    "注意力，只在自身历史内部建模；其最新状态与施动国、受动国的国家嵌入拼接，形成关系词元；全部1,959个词元再经两层"
    "跨体系注意力，任一关系都可以参照其他任一关系。这一设计有三个特点：关系之间的依存不由研究者预设，而由数据学习并"
    "随时局变化；无须对国家人为排序；输出为相对过去52周均值的修正量，使模型专注于学习何时偏离常态、偏离多少。"
    "发布版本为5个随机种子模型的集成，冲突激增模型沿用同一结构。")
figure("fig2_architecture", "发布模型 v2.2 的结构。")

H2("（三）评估方案")
para(
    "数据按时间严格划分为训练期（2015年5月—2022年10月）、验证期（2023年）和测试期（2024年3月—2026年6月），"
    f"各期之间设13周隔离期（{fig('fig3_timeline')}），一切设计选择仅依据验证集作出。对比基准中的44个人工特征是在研究数据之后设计的，带有事后之明，比较因而对基准有利。")
figure("fig3_timeline", "按时间划分的评估设计。")

H2("（四）贸易分析设计")
para(
    "贸易部分以中国申报的 IMF 月度货物贸易（2017年1月—2026年6月）为核心，结合 GDELT 国家—月冲突指数与 PortWatch 航运数据。回归采用份额—冲击设计，以双向固定效应识别海湾冲突冲击与预先确定的暴露度之交互项，以置换检验推断。贸易预测比较了国家—周 Transformer 与两参数的收缩持续性规则。")

H2("（五）扫描与更新机制")
para(
    "系统每两周重建 GDELT 面板并运行模型，每周更新贸易与航运数据，随后重算指标、提取近期 ACLED 事件与相关报道、生成简报并统一发布，全程无需人工干预。" f"外部数据见{tab('data')}。")
table("data", "外部数据来源", ["数据源", "用途", "更新频率"],
      [["GDELT 2.0", "冲突模型输入与目标；贸易冲突指数", "每两周"],
       ["ACLED", "近期暴力、事件地图", "每周"],
       ["IMF 国际货物贸易统计", "对华贸易额、回归与预测", "每周（滞后3—4个月）"],
       ["UN Comtrade", "2024年能源进口构成", "静态"],
       ["IMF PortWatch", "咽喉要道每日通行量", "每周"],
       ["霍尔木兹依赖度（EIA、IEA）", "无法绕行比例", "按需"],
       ["《卫报》《纽约时报》API", "简报引用的新闻", "每周"],
       ["DeepSeek API", "生成中英文简报", "每周"],
       ["Natural Earth", "地图国界", "静态"]],
      [5.0, 7.0, 4.2])

# ============================================================== 五 主要发现与应用价值
H1("五、主要发现与应用价值")
H2("（一）体系注意力提升了冲突预测精度")
para(
    f"{tab('bench')}和{fig('fig4_benchmark')}显示，v2.2 在测试期的决定系数为{f3(V22['r2'])}，同等原始输入的线性回归仅为"
    f"{f3(OLS1['r2'])}。线性回归加入44个人工特征后升至{f3(OLS44['r2'])}；不含区外训练样本的 v2.1 已与之持平"
    f"（{f3(V21['r2'])}），且无需任何特征工程；补充区外样本后，v2.2 在各项指标上全面领先。相对于52周均值，v2.2 的平方"
    f"误差降低{pc(A22['variance']['mse_reduction'])}，对{p0(A22['variance']['share_dyads_better'])}的关系更为准确。"
    f"{fig('fig5_path')}给出各项改进的贡献。")
table("bench", "冲突预测测试集表现（400组关系 × 119周）", ["方法", "R²", "方向准确率", "恶化识别 AUC"],
      [["过去52周均值", f3(B["trailing 52-week mean"]["r2"]), "—", "0.500"],
       ["线性回归（原始输入）", f3(OLS1["r2"]), pc(OLS1["dir_acc"]), f3(OLS1["esc_auc"])],
       ["线性回归（44个人工特征）", f3(OLS44["r2"]), pc(OLS44["dir_acc"]), f3(OLS44["esc_auc"])],
       ["梯度提升树（44个人工特征）", f3(GBM["r2"]), pc(GBM["dir_acc"]), f3(GBM["esc_auc"])],
       ["v2.1 跨体系注意力", f3(V21["r2"]), pc(V21["dir_acc"]), f3(V21["esc_auc"])],
       ["**v2.2 加入区外训练样本**", f"**{f3(V22['r2'])}**", f"**{pc(V22['dir_acc'])}**", f"**{f3(V22['esc_auc'])}**"]],
      [6.6, 2.6, 3.2, 3.8])
figure("fig4_benchmark", "各模型测试集 R² 对比。")
figure("fig5_path", "逐项改进后的测试集 R²。")
para(
    f"体系注意力何时起作用？消融实验与滚动回测（{fig('fig6_ablation')}、{fig('fig13_backtest')}）表明，训练数据不足时"
    "注意力并无优势，但在冲击沿地区扩散的2023年明显胜出；学习曲线"
    f"（{fig('fig12_learning_curve')}）显示，注意力模型随数据增加改进更快，在全部数据下仍未饱和；扩大体系规模"
    f"（{fig('fig11_scaling')}）也降低了验证误差。可见模型受限于数据量而非特征。")
figure("fig6_ablation", "地区消融实验。")
figure("fig13_backtest", "四个测试期的滚动回测。")
figure("fig12_learning_curve", "学习曲线。")
figure("fig11_scaling", "扩大体系规模后的表现。")
para(
    f"注意力分析进一步揭示了模型学到的内容（{fig('fig8_attention')}）。最具选择性的注意力头集中于巴以冲突及其外部"
    f"联系：巴勒斯坦对以色列关系获得的注意力为均匀水平的{lift('PSE-ISR'):.0f}倍，美国对以色列为{lift('USA-ISR'):.0f}倍。"
    "在未被告知哪些关系重要的情况下，模型自行学会关注地区核心冲突及大国介入，这正是逐行模型无法利用的体系信息。"
    f"{fig('fig7_examples')}展示了四组典型关系的预测轨迹。")
figure("fig8_attention", "最具选择性的两个注意力头。")
figure("fig7_examples", "四组关系测试期的预测与实际值。")

H2("（二）冲突激增预警具备实用价值")
para(
    f"以未来最坏月份为目标后，预警能力显著提升（{tab('flags')}）：激增模型每周前十项预警的命中率为"
    f"{p0(SURG['conflict_surge']['top10'])}，原四周方向模型仅为{p0(DIRN['conflict_surge']['top10'])}，接近随机水平。"
    f"{fig('fig14_surge_isr')}显示，激增模型对巴以关系峰值变化的跟踪明显优于平均值预测。基于人工特征的线性模型在该目标上仍略占优势。")
names = [("v2.0 四周方向模型", "v2.0 direction model (4-week escalation probability)"),
         ("v2.2 平均戈德斯坦预测", "v2.2 Goldstein forecast (fall below normal)"),
         ("上季度峰值（朴素）", "Naive: last quarter's peak conflict"),
         ("线性回归（人工特征）", "Linear regression, hand-built features*"),
         ("**v2.2 冲突激增模型**", "v2.2 conflict-surge model")]
table("flags", f"冲突激增预警效果（测试期共同{V22S['flag_weeks']}周）", ["预警方法", "激增 AUC", "前十命中率"],
      [[n, f3(FL[k]["conflict_surge"]["auc"]), p0(FL[k]["conflict_surge"]["top10"])] for n, k in names],
      [8.0, 4.0, 4.0], note=f"基准概率为{p0(V22S['surge_rate'])}。")
figure("fig14_surge_isr", "以色列对巴勒斯坦关系的平均值预测与峰值预测。")

H2("（三）冲突经由航运通道冲击中国贸易")
para(
    f"2026年3月2日起，霍尔木兹海峡油轮通行量由日均约50艘降至5艘以下（{fig('fig_trade_1_closure')}）。第二季度中国"
    f"自伊拉克、卡塔尔和科威特的进口同比下降83%至89%，而自可经红海出口的沙特进口反增12%（{tab('closure')}）。回归结果"
    f"（{fig('fig_trade_2_regression')}）表明，海湾冲突上升一个标准差并持续四个月，完全暴露伙伴的进口增速累计下降0.45"
    "个对数点（置换检验 p = 0.004），伙伴国自身冲突则无显著影响。但剔除2026年样本后效应接近于零，该估计刻画的是海峡真正封锁时的冲击，而非一般弹性。")
figure("fig_trade_1_closure", "霍尔木兹海峡油轮通行量与中国自海湾伙伴进口的同比变化。")
table("closure", "2026年第二季度中国自海湾伙伴进口的同比变化", ["伙伴国", "进口同比", "假定无法绕行比例"],
      [["伊拉克", "−89%", "0.88—1.00"], ["卡塔尔", "−88%", "1.00"], ["科威特", "−83%", "1.00"],
       ["阿联酋", "−75%", "0.44—0.66"], ["阿曼", "−20%", "0"], ["沙特阿拉伯", "+12%", "0.08—0.45"]],
      [5.4, 5.4, 5.4])
figure("fig_trade_2_regression", "交互项累计效应及稳健性检验。")

H2("（四）贸易预测宜简不宜繁")
para(
    "国家—周 Transformer 未能在样本外胜过收缩持续性规则，删除冲突输入亦无影响，也没有模型预见到2026年的贸易断崖。业务上因此采用后者（进口系数0.236，出口0.380），相对“无变化”预测的技能为6%至11%"
    f"（{fig('fig_trade_3_backtest')}）。")
figure("fig_trade_3_backtest", "收缩持续性预测的逐年样本外技能。")

H2("（五）应用价值")
para(
    f"上述成果通过中英双语风险监测平台落地（{fig('fig15_website')}）。地区总览以三维地图呈现国家风险及预计恶化或触发"
    "预警的双边关系；国家页给出风险构成、13周前景、对华贸易与近期暴力事件；航运面板单列通道状态；大语言模型据此并"
    "引用新闻生成对华简报。冲突预测按周更新，而官方贸易数据滞后三至四个月，平台因此能在贸易统计有所反映之前发出预警。")
figure("fig15_website", "风险监测平台：（a）地区总览；（b）国家页（也门）。")

# ============================================================== 六 企业建议
H1("六、企业建议")
numbered("**关注通道，而不只是伙伴国。**依赖伊拉克、科威特、卡塔尔和阿联酋能源供应的企业，应将海峡封锁纳入情景规划，"
         "并以回归结果估算冲击量级。")
numbered("**重视可绕行的供应，同时监测绕行设施本身。**沙特东西管道已于2026年9月遇袭。")
numbered("**把激增预警视为复核信号，而非确定性预测。**约半数预警随后应验，约为基准的三倍，企业应据此复核应急预案、"
         "安保措施与风险敞口。")
numbered(f"**不要等待贸易统计。**冲突预测与航运数据是领先信号；二者恶化而贸易数据仍显正常时，应按预测区间下限安排"
         f"计划（{tab('decision')}）。")
numbered("**为进口留足余量。**进口预测区间偏窄（实际覆盖率71%，名义80%），能源采购应预留更大的波动空间。")
numbered("**结合专家研判。**模型更擅长捕捉渐进变化，对突发事件须辅以专业判断。")
table("decision", "冲突与贸易信号的综合解读", ["冲突预测", "贸易预测", "解读"],
      [["出现激增预警或关系恶化", "平稳或上升", "贸易数据尚未反映，暴露伙伴按区间下限规划"],
       ["平稳", "近期骤降", "多为价格或船期因素，预计部分回升"],
       ["出现激增预警或关系恶化", "下降且海峡受阻", "中断可能持续，按最新变化或区间下限规划"],
       ["平稳", "平稳", "常态，采用点预测"]],
      [4.6, 3.6, 8.0])

# ============================================================== 七 必要说明
H1("七、必要说明")
para(
    "一是数据量与历史长度有限。GDELT 2.0 始于2015年，可供学习的地区性危机不多，本文结果应视为该方法能力的下限。二是贸易冲击仅由2026年一次事件识别，航线参数来自公开资料，受制裁"
    "贸易存在低报。三是各类数据时效不一，贸易数据滞后三至四个月。四是激增概率以2023年"
    "数据校准，在2024—2026年整体偏高，但不影响排序与预警。五是简报由大语言模型生成，用于具体决策前应予核实。"
    "下一步工作包括：以全球同源数据训练更大规模的模型，建立自有事件编码流程以延长历史记录，并把航运数据纳入贸易模型。")

# ============================================================== 八 结论
H1("八、结论")
para(
    "本文表明，把双边关系作为词元、以自注意力联合预测整个地区，是将国际体系结构引入冲突预测的有效途径；以各关系自身"
    "常态为基准的“最坏月份”指标显著提升了冲突激增预警能力；冲突对中国贸易的冲击则主要经由航运通道传导。三者共同构成"
    "一套透明、可持续更新的风险预警体系，可为中资企业的海外经营决策提供前瞻性支持。")

# ============================================================== references
H1("参考文献")
for ref in [
    "Borusyak, K., Hull, P., and Jaravel, X. (2022). Quasi-experimental shift-share research designs. Review of "
    "Economic Studies, 89(1), 181–213.",
    "Goldstein, J. S. (1992). A conflict-cooperation scale for WEIS events data. Journal of Conflict Resolution, "
    "36(2), 369–385.",
    "Hegre, H., et al. (2019). ViEWS: A political violence early-warning system. Journal of Peace Research, 56(2), "
    "155–174.",
    "Leetaru, K., and Schrodt, P. A. (2013). GDELT: Global data on events, location, and tone, 1979–2012. ISA Annual "
    "Convention, San Francisco.",
    "Mueller, H., and Rauh, C. (2018). Reading between the lines: Prediction of political violence using newspaper "
    "text. American Political Science Review, 112(2), 358–375.",
    "Raleigh, C., Linke, A., Hegre, H., and Karlsen, J. (2010). Introducing ACLED. Journal of Peace Research, 47(5), "
    "651–660.",
    "Vaswani, A., et al. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30.",
    "Waltz, K. N. (1979). Theory of International Politics. Reading, MA: Addison-Wesley.",
]:
    p = doc.add_paragraph()
    _runs(p, ref, size=10)
    p.paragraph_format.left_indent = Cm(0.8)
    p.paragraph_format.first_line_indent = Cm(-0.8)
    p.paragraph_format.space_after = Pt(4)

assert fig_no[0] == len(FIGS) and tab_no[0] == len(TABS)


def render_tables():
    """Lay the tables out in Word (tdoc, one per page), export to PDF, crop each table to a PNG and put it in
    the paper at its true size. Needs Microsoft Word (Windows) and PyMuPDF."""
    import os
    import subprocess

    import pymupdf
    os.makedirs(TABLE_DIR, exist_ok=True)
    src, pdf = os.path.abspath(f"{TABLE_DIR}/tables.docx"), os.path.abspath(f"{TABLE_DIR}/tables.pdf")
    tdoc.save(src)
    ps = (f"$w = New-Object -ComObject Word.Application; $w.Visible = $false; "
          f"$d = $w.Documents.Open('{src}', $false, $true); $d.SaveAs([ref]'{pdf}', [ref]17); "
          f"$d.Close($false); $w.Quit()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    pages = pymupdf.open(pdf)
    for (key, ph), page in zip(pending, pages):
        rects = [d_["rect"] for d_ in page.get_drawings()]
        box = rects[0]
        for r_ in rects[1:]:
            box |= r_
        box = pymupdf.Rect(box.x0 - 1, box.y0 - 1, box.x1 + 1, box.y1 + 1)
        png = f"{TABLE_DIR}/{key}.png"
        page.get_pixmap(clip=box, dpi=300).save(png)
        ph.add_run().add_picture(png, width=Pt(box.width))
    pages.close()
    os.remove(src)
    os.remove(pdf)


render_tables()
out = sys.argv[1] if len(sys.argv) > 1 else f"{P}/risk_paper_zh.docx"
doc.save(out)
body = "\n".join(p.text for p in doc.paragraphs)
leftover = re.findall(r"\{[A-Za-z_][^{}\s,]*\}", body)  # an f-string placeholder that was never filled in
assert not leftover, f"unfilled placeholders: {leftover}"
print(f"saved {out} | Chinese characters in text: {len(re.findall(r'[一-鿿]', body))} | figures {fig_no[0]} | tables {tab_no[0]} (as images)")

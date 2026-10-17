"""
Chinese edition of the transformer section (condensed, <= 5,000 Chinese characters).
Numbers come from the same files as the English edition (paper/build_docx.py).

    python paper/build_docx_zh.py      -> paper/transformer_section_zh.docx
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
S = json.load(open(f"{P}/stats.json"))
R = json.load(open(f"{P}/v21_stats.json"))
V22S = json.load(open(f"{P}/v22_stats.json"))
LIN = json.load(open(f"{P}/linear_baselines.json"))
B = S["goldstein_benchmark"]
V21, V22, GBM = R["benchmark"]["v2.1"], V22S["v2.2"], R["gbm"]
OLS1, OLS44 = LIN["OLS current week"], LIN["OLS 44 history features"]
FL = V22S["flag"]
SURG, DIRN = FL["v2.2 conflict-surge model"], FL["v2.0 direction model (4-week escalation probability)"]
f3 = lambda x: f"{x:.3f}"
pct = lambda x: f"{100 * x:.0f}%"

FONT = "Microsoft YaHei"
GREY = RGBColor(0x52, 0x51, 0x4E)
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.4)
sec.top_margin = sec.bottom_margin = Cm(2.2)


def set_font(style, size, bold=None):
    st = doc.styles[style]
    st.font.name = FONT
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    lang = OxmlElement("w:lang")  # zh-CN so Word applies Chinese line-break rules
    st.element.rPr.append(lang)
    lang.set(qn("w:val"), "en-US")
    lang.set(qn("w:eastAsia"), "zh-CN")
    st.font.size = Pt(size)
    if bold is not None:
        st.font.bold = bold
    st.font.color.rgb = RGBColor(0x0B, 0x0B, 0x0B)


set_font("Normal", 10.5)
doc.styles["Normal"].paragraph_format.line_spacing = 1.35
doc.styles["Normal"].paragraph_format.space_after = Pt(6)
for lvl, size in [(1, 14), (2, 12)]:
    set_font(f"Heading {lvl}", size, True)
    doc.styles[f"Heading {lvl}"].paragraph_format.space_before = Pt(12)
    doc.styles[f"Heading {lvl}"].paragraph_format.keep_with_next = True

fig_no, tab_no = [0], [0]


def runs(p, text, size=None, color=None):
    for i, part in enumerate(re.split(r"\*\*", text)):
        r = p.add_run(part)
        r.bold = i % 2 == 1
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = color


def para(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    runs(p, text)


def bullet(text):
    runs(doc.add_paragraph(style="List Bullet"), text)


def figure(name, caption, width=15.5):
    fig_no[0] += 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(f"{P}/{name}.png", width=Cm(width))
    runs(doc.add_paragraph(), f"**图{fig_no[0]}** {caption}", size=9, color=GREY)


def table(caption, header, rows, widths):
    tab_no[0] += 1
    runs(doc.add_paragraph(), f"**表{tab_no[0]}** {caption}", size=9, color=GREY)
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(header):
        c = t.rows[0].cells[j]
        c.text = ""
        runs(c.paragraphs[0], f"**{h}**", size=9)
    for i, row in enumerate(rows, 1):
        for j, v in enumerate(row):
            c = t.rows[i].cells[j]
            c.text = ""
            runs(c.paragraphs[0], str(v), size=9)
    for row in t.rows:
        for j, w in enumerate(widths):
            row.cells[j].width = Cm(w)
    doc.add_paragraph()


title = doc.add_heading("把地区当作一个体系来预测：用 Transformer 预测中东北非国际关系", level=0)
for r in title.runs:
    r.font.name = FONT
    r._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    r.font.size = Pt(17)
runs(doc.add_paragraph(), "Risk-A-Lab 地缘政治风险预测项目 · 模型版本 v2.2 · 英文版的中文精简版", size=9, color=GREY)

doc.add_heading("摘要", level=1)
para(f"现有的冲突预测系统大多把每个国家或每对国家当作独立的一行数据，只依据其自身历史作出预测，体系中其他国家的影响只能通过分析人员事先设计的汇总特征进入。"
     f"本文提出并检验另一种做法：用 Transformer 同时预测整个地区体系中的全部双边关系。每一对有向国家关系（dyad）是一个 token，"
     f"时间注意力层读取其近13周历史，跨体系注意力层再让所有关系相互“注意”。模型预测中东北非20国之间400组关系未来13周的平均戈德斯坦（Goldstein）合作—冲突得分，"
     f"同时关注这些国家与世界其他国家的1,559组联系。在严格的样本外检验中，跨体系模型（v2.1）的测试集 R² 为 {f3(V21['r2'])}，"
     f"明显高于同样原始输入的线性回归（{f3(OLS1['r2'])}）和最强的朴素预测（过去52周均值，{f3(B['trailing 52-week mean']['r2'])}），也高于梯度提升（{f3(GBM['r2'])}）。"
     f"再加入5,122组区外国家对作为训练数据（v2.2），R² 升至 {f3(V22['r2'])}，超过所有基准，包括依赖人工特征的线性模型（{f3(OLS44['r2'])}）。"
     f"预测结果驱动一个中英双语的动态风险监测网站，面向在本地区有贸易利益的机构，尤其是中国企业。")

doc.add_heading("1 问题：逐行预测的局限", level=1)
para("VIEWS 与 ConflictForecast 等业务化系统已相当成熟，但核心模型多为随机森林、梯度提升等非深度学习方法，处理定长特征向量，并逐行预测每个单位。"
     "对国际关系而言，这是实质性的限制：国家不会孤立地制定对某一对象国的政策。以色列对伊朗的态度，受到以色列与胡塞武装、伊朗与海湾君主国、以及双方与大国之间关系的影响。"
     "只根据每组关系自身历史打分的模型，无法学到哪些其他关系重要、何时重要。图1对比了两种做法。")
figure("fig1_concept", "逐行预测与体系感知预测。（a）传统做法：每对国家单独预测，背景信息只能来自人工特征。（b）本文：先编码每对关系的历史，再通过自注意力让所有关系交换信息，然后才作出预测。")

doc.add_heading("2 核心假设：自注意力即国际体系的模型", level=1)
para("自注意力定义在任意 token 集合上：每个 token 用所有其他 token 的加权平均更新自身，权重由数据学习并随时间变化。"
     "它恰好对应体系视角：任何关系的预测都可以依赖体系中任何其他关系；哪些关系重要不由分析人员预先指定；也不需要对国家人为排序。"
     "我们检验三个预期：模型优于朴素预测和非深度学习基准（E1）；跨体系注意力优于只看自身历史（E2）；收益随数据量增加（E3）。")

doc.add_heading("3 设计", level=1)
para("唯一的模型输入是 GDELT 2.0：它把全球新闻自动编码为国家之间的事件，每个事件按 −10（最敌对）至 +10（最合作）计分。"
     "我们按周（周一至周日）汇总每对有向国家关系，共20国、400组关系，2015年起约600周。"
     "每周输入为20类 CAMEO 事件计数的对数、当周平均戈德斯坦值，以及过去26周和52周的均值与活跃度；目标是未来13周的平均戈德斯坦值。"
     "模型以过去52周均值为基线、只预测偏离部分，采用5个随机种子的集成。"
     "检验严格按时间切分：训练期2015年5月至2022年10月，验证期为2023年，测试期为2024年3月至2026年6月，各期之间留出13周隔离期，以防目标重叠；模型选择只看验证集。")

doc.add_heading("4 结果", level=1)
para(f"表1列出主要结果。仅凭同样的原始周度数据，线性回归的 R² 只有 {f3(OLS1['r2'])}；Transformer 从原始历史中学习，解释的方差约为其两倍。"
     f"给线性回归加入44个人工设计的历史特征后，其 R² 升至 {f3(OLS44['r2'])}，与 v2.1 持平——但这些特征是在研究数据之后设计的，带有事后之明。"
     f"Transformer 无需人工特征即可达到同等水平；加入更多同源数据后（v2.2）则全面领先：方向准确率 {pct(V22['dir_acc'])}，变化相关系数 {V22['change_corr']:.2f}。")
table("测试集表现（400组关系 × 119周，2024年3月–2026年6月）",
      ["预测方法", "R²", "方向准确率", "恶化识别 AUC"],
      [["过去52周均值（朴素）", f3(B["trailing 52-week mean"]["r2"]), "—", "0.500"],
       ["线性回归（当周原始数据）", f3(OLS1["r2"]), pct(OLS1["dir_acc"]), f3(OLS1["esc_auc"])],
       ["线性回归（44个人工特征）", f3(OLS44["r2"]), pct(OLS44["dir_acc"]), f3(OLS44["esc_auc"])],
       ["梯度提升（44个人工特征）", f3(GBM["r2"]), pct(GBM["dir_acc"]), f3(GBM["esc_auc"])],
       ["v2.1 跨体系注意力", f3(V21["r2"]), pct(V21["dir_acc"]), f3(V21["esc_auc"])],
       ["v2.2 + 区外训练数据", f3(V22["r2"]), pct(V22["dir_acc"]), f3(V22["esc_auc"])]],
      [6.2, 2.4, 3.0, 3.4])
para("消融与扩展实验给出一致的图景：跨体系注意力在训练数据充足、冲击在地区内传导时（如2023年）优势明显，在训练数据从未覆盖的2024–2026年格局中与简单模型持平；"
     "学习曲线显示注意力模型在全部数据下仍在改进。加入新闻文本、ACLED 或更早的 GDELT 1.0 均未改善结果，而增加同源数据（v2.2）直接提高了效果：模型受限于数据量而非特征。")
para(f"我们还训练了第二个同架构模型，预测未来13周中“最糟糕的一个月”的冲突强度（只计敌对事件），用于预警冲突激增。"
     f"每周排名前10的预警中，有 {pct(SURG['conflict_surge']['top10'])} 随后确实出现激增，而随机基准为 {pct(V22S['surge_rate'])}；旧的四周方向模型仅为 {pct(DIRN['conflict_surge']['top10'])}。")

doc.add_heading("5 从预测到风险监测", level=1)
para("预测结果通过一个中英双语网站发布（图2）。设计遵循四条原则：**区分预测与观测**——凡由 Transformer 产生或以其为基础的数值都用统一标识，观测数据单独呈现；"
     "**保留体系视角**——在地图上画出一国的双边关系，让用户看到压力来自何处；**面向决策**——每个数值都配有“含义”说明；**默认简洁**——地图只显示预计趋向敌对或触发激增预警的关系。")
figure("fig15_website", "风险监测网站。（a）地区总览：国家按风险着色，悬停国家被抬起并显示其存在隐患的双边关系；圆环标示霍尔木兹海峡、曼德海峡与苏伊士运河，按航运受阻程度着色。（b）国家页（也门）：风险及其构成、13周前景、对华关联简报与近28天 ACLED 事件。", 14)
para("**风险的定义。**风险面向在该国有贸易利益的机构：只有冲突环境预计严重、且涉及贸易规模大时，风险才会高。各项均为0–100分：")
bullet("**严重度** = 100 ×（0.40 × 预测敌对程度 + 0.30 × 预测激增风险 + 0.30 × 近期暴力）。前两项来自 Transformer 预测（按各关系的活跃度加权），近期暴力为 ACLED 记录的过去四周死亡人数（对数尺度）。")
bullet("**涉及贸易** = 过去12个月中国与该国的贸易额（IMF），按对数相对于本地区最大贸易伙伴折算。")
bullet("**风险** = √（严重度 × 涉及贸易）。几何平均使任何单一因素都无法独自拉高风险。等级：严重 ≥ 55，高 ≥ 45，偏高 ≥ 32。地区风险为20国平均值。")
para("权重与折算点为事先固定的设计选择，随评分一同公布，以保证各期结果可比；评分用于排序，而非概率。")
para("**网站内容。**地区总览包括风险地图、关注列表（风险最高、恶化最快、航运通道）；国家页包括风险构成、过去一年关系走势与13周预测、主要双边关系及背景数据。"
     "航运通道比较霍尔木兹、曼德海峡、苏伊士运河过去四周的每周通行船只数与2019年至2023年9月的常态（降幅超过50%为“严重受阻”，超过20%为“通行减少”）。所有结果均可导出。")
para("**AI 简报。**每次更新时，大语言模型（DeepSeek）根据仪表盘数据、IMF 贸易数据、霍尔木兹依赖度、所邻航道，以及《卫报》《纽约时报》近90天同时提及中国与该国的报道，"
     "用中英文撰写各国前景、对华贸易关系及“中国主体可能面临的风险”。规则要求：数字必须来自输入数据，具体项目或事件必须有被引用的报道，不得作无依据的因果判断；来源列于简报下方，模型不可用时改用固定模板。")
para("**动态更新。**云端定时任务每两周重建 GDELT 周度面板、运行模型、生成数据包并发布到云存储；另一任务每周更新 IMF 贸易与 PortWatch 航运数据。网站每次访问时读取最新数据包，因此无需重新部署即可更新。表2列出全部外部数据。")
table("模型与风险监测使用的外部数据",
      ["数据源", "提供方与获取方式", "用途", "更新"],
      [["GDELT 2.0 事件数据", "GDELT 项目；BigQuery 公共数据集", "全部模型输入与目标", "每两周"],
       ["ACLED", "ACLED；API（注册账户）", "近期暴力、事件地图、简报背景", "每两周"],
       ["IMF 国际货物贸易（IMTS）", "IMF；SDMX API", "对华贸易额、涉及贸易", "每周"],
       ["IMF PortWatch", "IMF 与牛津大学；ArcGIS API", "咽喉要道每日通行船只", "每周"],
       ["霍尔木兹依赖度", "据 IEA、EIA 航线数据整理", "各海湾产油国原油无法绕行的比例", "按需"],
       ["《卫报》《纽约时报》API", "两家媒体的开放接口", "简报所引用的近期报道", "每两周"],
       ["DeepSeek API", "DeepSeek", "撰写中英文简报", "每两周"],
       ["世界银行 WDI", "世界银行；API", "早期风险指数模块的贸易暴露", "每两周"],
       ["Natural Earth 边界", "公有领域", "地图国界", "静态"]],
      [3.6, 4.2, 5.0, 2.0])

doc.add_heading("6 局限", level=1)
bullet("**数据量与历史长度。**GDELT 2.0 始于2015年，模型只见过少数全区域危机；GDELT 1.0 编码方式不同，加入后反而变差。在数据受限的前提下，本文结果应视为该方法能力的下限。")
bullet("**单一数据源的偏差。**GDELT 统计的是媒体报道而非实地事件；2023年10月7日后以色列—巴勒斯坦的平均戈德斯坦值反而上升，因为战争带来大量人道与外交事件。冲突激增模型只计敌对事件，部分缓解了这一问题。")
bullet("**渐变优于突变。**模型更擅长预测趋势性变化，应作为前瞻研判工具与专家判断结合使用，而非预测具体事件。")

doc.add_heading("7 扩展方向", level=1)
para("最直接的收益来自更多同源数据：覆盖全球所有国家对的模型可把训练数据扩大一个数量级，并通过稀疏或分层注意力控制计算量；"
     "更长远的方向是建立自有的事件编码流程，对数十年的新闻档案统一编码，让模型见到更多危机与格局转换。")

doc.add_heading("8 结论", level=1)
para("本文展示了将深度学习用于国际关系预测的一种新方式：把每对国家关系当作一个 token，用时间注意力读取各自历史，再用跨体系注意力联合预测整个地区。"
     "在严格的样本外检验中，模型在不依赖人工特征的情况下达到并超过精心设计的基准，增加同源数据后进一步领先；第二个同架构模型能有效预警冲突激增。"
     "两者每两周自动更新，驱动一个面向在本地区有贸易利益机构的双语动态风险监测网站。")

doc.save(f"{P}/transformer_section_zh.docx")
text = "\n".join(p.text for p in doc.paragraphs) + "\n".join(c.text for t in doc.tables for row in t.rows for c in row.cells)
cjk = len(re.findall(r"[一-鿿]", text))
print(f"saved {P}/transformer_section_zh.docx | Chinese characters: {cjk} | figures {fig_no[0]} | tables {tab_no[0]}")

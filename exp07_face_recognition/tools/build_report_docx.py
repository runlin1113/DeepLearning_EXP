"""
把 实验报告.md 转成排版好的 Word（.docx）。
支持：#~#### 标题 / 段落 / 无序·有序列表 / 表格 / 代码块 / 图片+图注 / 引用 / $公式$（转 Unicode）。

用法：
    python tools/build_report_docx.py                # 生成 实验报告.docx
    python tools/build_report_docx.py --md a.md --out b.docx
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from docx import Document                       # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT   # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH    # noqa: E402
from docx.oxml import OxmlElement                # noqa: E402
from docx.oxml.ns import qn                      # noqa: E402
from docx.shared import Cm, Pt, RGBColor          # noqa: E402


# ------------------------------------------------------------- LaTeX → 文本
_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "theta": "θ", "lambda": "λ",
    "mu": "μ", "pi": "π", "sigma": "σ", "tau": "τ", "phi": "φ", "omega": "ω",
    "Delta": "Δ", "Sigma": "Σ", "Omega": "Ω", "Theta": "Θ",
}
_SYM = {
    "top": "ᵀ", "cdot": "·", "times": "×", "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥",
    "approx": "≈", "lesssim": "≲", "sim": "~", "in": "∈", "neq": "≠", "ne": "≠",
    "log": "log ", "exp": "exp ", "max": "max ", "min": "min ", "sqrt": "√",
    "cos": "cos ", "sin": "sin ", "tan": "tan ", "argmax": "argmax ",
    "sum": "Σ", "qquad": "    ", "quad": "  ", "to": "→", "rightarrow": "→",
    "ldots": "…", "cdots": "⋯", "pm": "±", "infty": "∞",
}


def _match_brace(s: str, start: int):
    """s[start] == '{'，返回匹配的 '}' 下标（不含）"""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _clean(s: str) -> str:
    # \frac{A}{B} → (A)/(B)，支持嵌套
    out = ""
    i = 0
    while i < len(s):
        if s.startswith(r"\frac", i):
            j = s.find("{", i)
            if j != -1:
                k = _match_brace(s, j)
                if k != -1:
                    l = s.find("{", k + 1)
                    if l == k + 1:
                        m = _match_brace(s, l)
                        if m != -1:
                            num, den = s[j + 1:k], s[l + 1:m]
                            out += f"({_clean(num)})/({_clean(den)})"
                            i = m + 1
                            continue
        if s.startswith(r"\text", i):
            j = s.find("{", i)
            if j != -1:
                k = _match_brace(s, j)
                if k != -1:
                    out += s[j + 1:k]
                    i = k + 1
                    continue
        out += s[i]
        i += 1

    s = out
    s = s.replace(r"\|", "‖")
    # 上标/下标去掉花括号
    s = re.sub(r"\^\{([^{}]*)\}", r"^(\1)", s)
    s = re.sub(r"_\{([^{}]*)\}", r"_\1", s)
    s = s.replace("^{", "^").replace("_{", "_").replace("{", "").replace("}", "")
    # \命令 → 符号（长名优先）
    s = re.sub(r"\\([A-Za-z]+)",
               lambda m: _SYM.get(m.group(1), _GREEK.get(m.group(1), m.group(1))), s)
    s = re.sub(r"\^2(?![0-9(])", "²", s)
    s = re.sub(r"\^3(?![0-9(])", "³", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = s.replace(" )", ")").strip()
    return s


def latex_to_text(s: str) -> str:
    s = s.replace("$$", "\x00").replace("$", "\x00")
    parts = s.split("\x00")
    out = []
    for idx, seg in enumerate(parts):
        out.append(_clean(seg) if idx % 2 else seg)
    return "".join(out)


# ------------------------------------------------------------------ 字体工具
def set_font(run, name="宋体", size=None, bold=None, color=None, italic=False):
    run.font.name = "Times New Roman"
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), name)
    rFonts.set(qn("w:ascii"), "Times New Roman")
    rFonts.set(qn("w:hAnsi"), "Times New Roman")
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic:
        run.font.italic = True
    if color:
        run.font.color.rgb = RGBColor(*color)


def shade(cell, hex_color="F2F2F2"):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


_TOKEN = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\$[^$]+\$)")


def _segments(text: str):
    """拆出 **粗体** / `代码` / $公式$，返回 [(文本, 类型)]，类型 ∈ n/b/c/m"""
    out, i = [], 0
    for m in _TOKEN.finditer(text):
        if m.start() > i:
            out.append((text[i:m.start()], "n"))
        tok = m.group(0)
        if tok.startswith("**"):
            out.append((tok[2:-2], "b"))
        elif tok.startswith("`"):
            out.append((tok[1:-1], "c"))
        else:
            out.append((_clean(tok[1:-1]), "m"))
        i = m.end()
    if i < len(text):
        out.append((text[i:], "n"))
    return out or [(text, "n")]


def add_paragraph(doc, text, size=10.5, indent=True, color=None, align=None,
                  space_after=4, font="宋体"):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.line_spacing = 1.35
    if indent:
        pf.first_line_indent = Pt(size * 2)
    if align is not None:
        p.alignment = align
    for seg, kind in _segments(text):
        run = p.add_run(seg)
        if kind == "c":
            set_font(run, "Consolas", size - 1, color=(0x33, 0x33, 0x33))
        elif kind == "m":
            set_font(run, "Cambria Math", size, color=(0x1F, 0x3B, 0x73))
        else:
            set_font(run, font, size, bold=(kind == "b"), color=color)
    return p


def add_code_block(doc, lines):
    for ln in lines:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.left_indent = Cm(0.4)
        run = p.add_run(ln if ln.strip() else " ")
        set_font(run, "Consolas", 8.5, color=(0x2F, 0x2F, 0x2F))
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_table(doc, rows):
    ncol = len(rows[0])
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j in range(ncol):
            cell = t.cell(i, j)
            txt = row[j] if j < len(row) else ""
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if (j > 0 or i == 0) else WD_ALIGN_PARAGRAPH.LEFT
            for seg, kind in _segments(txt):
                run = p.add_run(seg)
                if kind == "m":
                    set_font(run, "Cambria Math", 9)
                elif kind == "c":
                    set_font(run, "Consolas", 8.5)
                else:
                    set_font(run, "宋体", 9, bold=(i == 0 or kind == "b"))
            if i == 0:
                shade(cell, "E8EEF7")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_image(doc, path: Path, caption: str, max_w_cm=15.4, max_h_cm=18.5):
    if not path.exists():
        add_paragraph(doc, f"[缺图] {caption} -> {path}", size=9, indent=False,
                      color=(0xC0, 0x39, 0x2B))
        return
    # 按图片真实长宽比缩放，避免"竖长图"超出页高被挤到下一页
    try:
        from PIL import Image
        with Image.open(path) as im:
            pw, ph = im.size
        width_cm = min(max_w_cm, max_h_cm * pw / max(1, ph))
    except Exception:
        width_cm = max_w_cm
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.space_after = Pt(10)
    run = c.add_run(latex_to_text(caption))
    set_font(run, "楷体", 9, color=(0x44, 0x44, 0x44))


def build_cover(doc, title="人脸识别系统设计"):
    def line(text, size=14, bold=False, font="宋体", space=6, align=WD_ALIGN_PARAGRAPH.CENTER):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_after = Pt(space)
        run = p.add_run(text)
        set_font(run, font, size, bold)

    line("课程编号：IB00267", 10.5, align=WD_ALIGN_PARAGRAPH.LEFT)
    line("", 10)
    line("", 10)
    line("深圳技术大学实验报告", 22, True, "黑体", 20)
    line("", 10)
    line("", 10)

    rows = [("课程名称", "深度学习方法与应用"),
            ("实验名称", title),
            ("指导教师", "彭小江"),
            ("实验地点/时间", "C5-361 / 2026-11"),
            ("姓名", ""),
            ("学号", ""),
            ("班级", "")]
    t = doc.add_table(rows=len(rows), cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (k, v) in enumerate(rows):
        for j, txt in enumerate((k, v)):
            cell = t.cell(i, j)
            cell.width = Cm(4.4 if j == 0 else 8.2)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run(txt)
            set_font(run, "宋体", 12)
    doc.add_page_break()


_STRUCT_RE = re.compile(r"^\s*(```|\||!\[|#|>|[-*+]\s|\d+\.\s|---|\*\*\*|\$\$)")
# 这些行不可能有"续行"（标题/表格/图片/引用/分隔线/代码围栏）
_BLOCK_PREV_RE = re.compile(r"^\s*(```|\||!\[|#|>|---|\*\*\*|\$\$)")
# 上一行以这些字符收尾 → 句子没说完，下一行必然是续行
_OPEN_TAIL = "，、；：,;（(“"


def _merge_continuations(lines):
    """把 markdown 里"软换行"的续行并回上一行。

    中文写作常在一句话中间手动折行，严格按行解析会把同一句话拆成两段，docx 里
    就出现莫名其妙的空行。判据有两条（宁缺勿滥，避免把"元信息块"多行粘成一行）：
        ① 下一行有缩进（列表项内的续行，最常见）；
        ② 上一行以顿号/逗号/分号等未收尾字符结尾。
    代码块内不处理。
    """
    out, in_code = [], False
    for ln in lines:
        if ln.strip().startswith("```"):
            in_code = not in_code
            out.append(ln)
            continue
        if in_code or not ln.strip():
            out.append(ln)
            continue
        prev = out[-1] if out else ""
        indented = ln[:1] in (" ", "\t")
        open_tail = prev.rstrip()[-1:] in tuple(_OPEN_TAIL)
        joinable = (bool(prev.strip()) and not _BLOCK_PREV_RE.match(prev)
                    and (indented or open_tail) and _STRUCT_RE.match(ln) is None)
        if joinable:
            a, b = prev.rstrip(), ln.strip()
            sep = " " if (a[-1:].isascii() and a[-1:].isalnum()
                          and b[:1].isascii() and b[:1].isalnum()) else ""
            out[-1] = a + sep + b
        else:
            out.append(ln)
    return out


def convert(md_path: Path, out_path: Path):
    lines = _merge_continuations(md_path.read_text(encoding="utf-8").splitlines())
    doc = Document()
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Cm(2.4)
    sec.top_margin = sec.bottom_margin = Cm(2.2)

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    footer_p = sec.footer.paragraphs[0]
    footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    footer_p._p.append(fld)

    title_done = False
    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()

        if s.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i].rstrip())
                i += 1
            add_code_block(doc, buf)
            i += 1
            continue

        if s.startswith("|") and i + 1 < n and re.match(r"^\|[\s:\-|]+\|$", lines[i + 1].strip()):
            rows = [[c.strip() for c in s.strip("|").split("|")]]
            i += 2
            while i < n and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            add_table(doc, rows)
            continue

        m = re.match(r"!\[(.*?)\]\((.*?)\)", s)
        if m:
            add_image(doc, (md_path.parent / m.group(2)).resolve(), m.group(1))
            i += 1
            continue

        if s.startswith("$$") and s.endswith("$$") and len(s) > 4:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(6)
            run = p.add_run(_clean(s.strip("$")))
            set_font(run, "Cambria Math", 11.5, color=(0x1F, 0x3B, 0x73))
            i += 1
            continue

        if s.startswith("#"):
            level = len(s) - len(s.lstrip("#"))
            text = s[level:].strip()
            if not title_done:
                build_cover(doc, "人脸识别系统设计")
                title_done = True
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after = Pt(12)
                run = p.add_run(latex_to_text(text))
                set_font(run, "黑体", 16, True)
                i += 1
                continue
            sizes = {1: 15, 2: 13.5, 3: 11.5, 4: 10.5}
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10 if level <= 2 else 6)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(latex_to_text(text))
            set_font(run, "黑体", sizes.get(level, 10.5), True)
            i += 1
            continue

        if s in ("---", "***"):
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(2)
            pr = p._p.get_or_add_pPr()
            bd = OxmlElement("w:pBdr")
            bottom = OxmlElement("w:bottom")
            bottom.set(qn("w:val"), "single")
            bottom.set(qn("w:sz"), "6")
            bottom.set(qn("w:color"), "BBBBBB")
            bd.append(bottom)
            pr.append(bd)
            i += 1
            continue

        if s.startswith(">"):
            add_paragraph(doc, s[1:].strip(), size=9.5, indent=False,
                          color=(0x55, 0x55, 0x55), space_after=6)
            i += 1
            continue

        m = re.match(r"^(\d+)\.\s+(.*)$", s)
        if m:
            add_paragraph(doc, f"{m.group(1)}. {m.group(2)}", indent=False, space_after=3)
            i += 1
            continue

        if s.startswith("- ") or s.startswith("* "):
            add_paragraph(doc, "· " + s[2:], indent=False, space_after=3)
            i += 1
            continue

        if not s:
            i += 1
            continue

        add_paragraph(doc, s)
        i += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    print(f"✓ 已生成 {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=str(ROOT / "实验报告.md"))
    ap.add_argument("--out", default=str(ROOT / "实验报告.docx"))
    args = ap.parse_args()
    convert(Path(args.md), Path(args.out))


if __name__ == "__main__":
    main()

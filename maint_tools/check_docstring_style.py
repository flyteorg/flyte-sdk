#!/usr/bin/env python3
"""Fail on docstring markup this SDK does not use.

Docstrings here are read in three places: the published API reference, an IDE
tooltip, and `help()`. Only one of those renders anything but plain text, so
markup aimed at a documentation generator we do not run shows up verbatim to
everyone.

The SDK has no Sphinx (no conf.py, no .rst sources, no dependency), so every
reStructuredText construct in a docstring is inert. And the API-reference
generator parses Google sections and Sphinx field lists only, so a NumPy
`Parameters` section is dropped outright and the rendered parameter table comes
out with empty descriptions.

Rejected:

  1. Cross-reference roles     :class:`X`, :meth:`X`, :func:`X`, ...
     Write a plain code span instead: `flyte.io.Dir`. The docs site links
     qualified identifiers automatically, and a code span reads correctly
     everywhere else.

  2. Literal-block markers     a line ending in `::`
     The marker renders as a stray double colon and the indented block that
     follows becomes a code block with no language, so nothing highlights.
     Use a fenced block with an explicit language.

  3. NumPy sections           Parameters / Returns / ... over a dashed rule
     Use Google style: Args: / Returns: / Raises:, entries indented under the
     header as `name: description`.

  4. RST directives           .. warning::, .. code-block:: python, ...
     Use plain prose, or a fenced code block.

  5. Sphinx field lists       :param x:, :returns:, :rtype:, :raises X:
     Parsed correctly by the generator, but still RST, so it reads as markup
     in an editor. Use Google style: Args: / Returns: / Raises:.

  6. Inline literals          ``value``
     Renders correctly only by accident, since a double backtick is also a
     valid Markdown code span. Use a single-backtick span: `value`.

  7. Hyperlink targets        `text <url>`_
     Use a Markdown link: [text](url).

  8. Grid tables              +----+----+
     Renders as garbage. Use a Markdown table.

  9. Footnote references      [1]_
     Inline the reference, or use a Markdown link.

Rules 5 to 9 apply to docstrings only; 1 to 4 also apply to comments.

Content inside a fenced code block is code and is never flagged, so an example
that deliberately shows RST is fine.

Only docstrings and comments are inspected, both located syntactically, so a
runtime string that happens to contain one of these patterns is never flagged.

Usage:
    python maint_tools/check_docstring_style.py [PATH ...]

With no arguments it checks src/ and every plugins/*/src/ tree.
"""

from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
from pathlib import Path

ROLES = (
    "class|func|meth|attr|data|mod|obj|exc|ref|term|doc|option|envvar|"
    "const|type|paramref|keyword|abbr|command|file|kbd|guilabel|menuselection"
)
ROLE_RE = re.compile(rf":(?:py:)?(?:{ROLES}):`[^`]*`")
LITERAL_BLOCK_RE = re.compile(r"[^:\s]\s*::\s*$")
DIRECTIVE_RE = re.compile(r"^\s*\.\.\s+[\w-]+::")
NUMPY_HEADERS = (
    "Parameters",
    "Other Parameters",
    "Returns",
    "Yields",
    "Raises",
    "Warns",
    "Attributes",
    "Notes",
    "Examples",
    "See Also",
)
NUMPY_HDR_RE = re.compile(rf"^\s*(?:{'|'.join(map(re.escape, NUMPY_HEADERS))})\s*$")
NUMPY_RULE_RE = re.compile(r"^\s*-{3,}\s*$")

# Sphinx field lists. Parsed correctly by the generator, but still RST: inert
# markup in every other place a docstring is read.
SPHINX_FIELDS = (
    "param|parameter|arg|argument|key|keyword|type|return|returns|rtype|"
    "raise|raises|except|exception|var|ivar|cvar|meta|yield|yields"
)
SPHINX_FIELD_RE = re.compile(rf"^\s*:(?:{SPHINX_FIELDS})\b")
# RST inline literal. Renders right only by accident, since a double backtick
# is also a Markdown code span.
DOUBLE_BACKTICK_RE = re.compile(r"(?<!`)``[^`\n]+``(?!`)")
# Any double backtick at all, however it is arranged. The house rule is that a
# docstring never contains one, so the check may as well say that rather than
# recognise well-formed literals: the three live shapes were a same-line literal,
# a literal wrapped at the line limit (invisible to a line-at-a-time scan), and a
# stray backtick leaving the delimiters mismatched (`x`` / ``x`), which pairs with
# nothing and so matched no pattern that looked for a pair.
ANY_DOUBLE_BACKTICK_RE = re.compile(r"``")
# `text <url>`_ wrapped across a line.
WRAPPED_RE = {
    "rst-hyperlink": re.compile(r"`[^`]*<[^>]*\n[^>]*>`__?"),
}
# `text <url>`_ and the anonymous `text <url>`__ form.
RST_HYPERLINK_RE = re.compile(r"`[^`\n]*<[^>\n]+>`__?")
RST_GRID_TABLE_RE = re.compile(r"^\s*\+[-=+]{3,}\+")
# A literal in quotes, or a bare flag, renders as prose. The CLI reference shows
# the effect inside one table row: the Option column is monospaced and the
# Description column beside it is not, for the same flag. Backticks were already
# the majority style before this was enforced.
QUOTED_LITERAL_RE = re.compile(r"(?<![`\w\[])'([A-Za-z0-9_][A-Za-z0-9_.:/@=-]*|\S[^'\n]*\s[^'\n]*\S)'(?!\w|\])")
BARE_FLAG_RE = re.compile(r"(?<![`\w-])(--[a-z][a-z0-9-]{2,})(?![\w-])")
# Lines that are example code, not prose: the quotes there are Python syntax and
# the flags are being demonstrated, so both must be left alone.
CODEISH_RE = re.compile(r"^(\s{4,}|\s*(\$|>>>|flyte |python |uv |pip |make ))")
RST_FOOTNOTE_RE = re.compile(r"\[\d+\]_")
FENCE_RE = re.compile(r"^\s*```")

FIXES = {
    "rst-role": "use a plain code span, qualified where it is public: `flyte.io.Dir`",
    "literal-block": "end the sentence with a single ':' and fence the block with a language",
    "numpy-section": "use Google style: 'Args:' with entries indented beneath it",
    "rst-directive": "use plain prose, or a fenced code block",
    "sphinx-field": "use Google style: 'Args:' / 'Returns:' / 'Raises:'",
    "double-backtick": "use a single-backtick Markdown code span: `value`",
    "quoted-literal": "wrap the literal in backticks so it renders as monospace: `value`",
    "bare-flag": "wrap the flag in backticks so it renders as monospace: `--flag`",
    "rst-hyperlink": "use a Markdown link: [text](url)",
    "rst-grid-table": "use a Markdown table",
    "rst-footnote": "inline the reference, or use a Markdown link",
}


class Finding:
    __slots__ = ("kind", "line", "path", "text")

    def __init__(self, path, line, kind, text):
        self.path = path
        self.line = line
        self.kind = kind
        self.text = text


def scan_block(path: Path, start_line: int, text: str, kind_prefix: str) -> list[Finding]:
    found: list[Finding] = []
    lines = text.split("\n")
    in_fence = False
    for n, line in enumerate(lines):
        lineno = start_line + n
        # Inside a fenced block everything is code and is left alone. The
        # fence itself is not content, so it is skipped too.
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for m in ROLE_RE.finditer(line):
            found.append(Finding(path, lineno, "rst-role", m.group(0)))
        if DIRECTIVE_RE.match(line):
            found.append(Finding(path, lineno, "rst-directive", line.strip()))
        elif LITERAL_BLOCK_RE.search(line):
            found.append(Finding(path, lineno, "literal-block", line.strip()))
        if kind_prefix in ("docstring", "help") and not CODEISH_RE.match(line):
            # `line` here already has code spans left intact, so blank them first
            # or a literal that is ALREADY backticked reports itself.
            prose = re.sub(r"`[^`\n]*`", lambda m: " " * len(m.group(0)), line)
            for m in QUOTED_LITERAL_RE.finditer(prose):
                found.append(Finding(path, lineno, "quoted-literal", m.group(0)[:60]))
            for m in BARE_FLAG_RE.finditer(prose):
                found.append(Finding(path, lineno, "bare-flag", m.group(1)))
        if kind_prefix == "docstring":
            if SPHINX_FIELD_RE.match(line):
                found.append(Finding(path, lineno, "sphinx-field", line.strip()))
            for m in RST_HYPERLINK_RE.finditer(line):
                found.append(Finding(path, lineno, "rst-hyperlink", m.group(0)))
            if RST_GRID_TABLE_RE.match(line):
                found.append(Finding(path, lineno, "rst-grid-table", line.strip()[:40]))
            for m in RST_FOOTNOTE_RE.finditer(line):
                found.append(Finding(path, lineno, "rst-footnote", m.group(0)))
        if (
            kind_prefix == "docstring"
            and NUMPY_HDR_RE.match(line)
            and n + 1 < len(lines)
            and NUMPY_RULE_RE.match(lines[n + 1])
        ):
            found.append(Finding(path, lineno, "numpy-section", line.strip()))

    if kind_prefix == "docstring":
        found += _scan_wrapped(path, start_line, text)
    return found


def _scan_wrapped(path: Path, start_line: int, text: str) -> list[Finding]:
    """Catch constructs whose delimiters land on different lines.

    Runs over the whole block rather than line by line, which is the only way to
    see them. Fenced regions are blanked first so a wrapped literal inside an
    example is left alone, exactly as the per-line pass leaves it alone.
    """
    lines = text.split("\n")
    in_fence = False
    scannable = []
    for line in lines:
        if FENCE_RE.match(line):
            in_fence = not in_fence
            scannable.append("")
            continue
        scannable.append("" if in_fence else line)
    block = "\n".join(scannable)

    found: list[Finding] = []
    for m in ANY_DOUBLE_BACKTICK_RE.finditer(block):
        lineno = start_line + block[: m.start()].count("\n")
        line = block.split("\n")[block[: m.start()].count("\n")]
        found.append(Finding(path, lineno, "double-backtick", line.strip()[:70]))
    for kind, pattern in WRAPPED_RE.items():
        for m in pattern.finditer(block):
            lineno = start_line + block[: m.start()].count("\n")
            found.append(Finding(path, lineno, kind, " ".join(m.group(0).split())))
    return found


def check_file(path: Path) -> list[Finding]:
    src = path.read_bytes().decode("utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [Finding(path, e.lineno or 0, "syntax-error", str(e))]

    found: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for stmt in getattr(node, "body", []):
            if (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            ):
                found += scan_block(path, stmt.value.lineno, stmt.value.value, "docstring")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if (
                kw.arg in ("help", "short_help")
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ):
                found += scan_block(path, kw.value.lineno, kw.value.value, "help")

    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                found += scan_block(path, tok.start[0], tok.string, "comment")
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass

    # a docstring can be reached twice when nodes nest; keep one report per site
    seen = set()
    unique = []
    for f in found:
        key = (f.line, f.kind, f.text)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def roots(repo: Path) -> list[Path]:
    out = [repo / "src"]
    out += sorted(repo.glob("plugins/*/src"))
    out += sorted(repo.glob("plugins/*/*/src"))
    return [p for p in out if p.is_dir()]


def main(argv: list[str]) -> int:
    repo = Path(__file__).resolve().parent.parent
    targets = [Path(a) for a in argv[1:]] or roots(repo)

    files: list[Path] = []
    for t in targets:
        if t.is_dir():
            files += sorted(t.rglob("*.py"))
        elif t.suffix == ".py":
            files.append(t)

    findings: list[Finding] = []
    for f in files:
        findings += check_file(f)

    if not findings:
        print(f"check-docstrings: {len(files)} files clean")
        return 0

    by_kind: dict[str, int] = {}
    for f in findings:
        by_kind[f.kind] = by_kind.get(f.kind, 0) + 1

    for f in findings:
        rel = f.path.relative_to(repo) if f.path.is_absolute() and repo in f.path.parents else f.path
        print(f"{rel}:{f.line}: [{f.kind}] {f.text[:110]}")

    print()
    print(f"check-docstrings: {len(findings)} problem(s) in {len(files)} files")
    for kind, n in sorted(by_kind.items()):
        print(f"  {kind}: {n}  -> {FIXES.get(kind, '')}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))

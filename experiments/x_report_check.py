"""Number check for reports/v3/report.md (implementation plan §14.5).

For every number in the report, look for it (after rounding) among the numeric values of the JSON
files cited in the same line, falling back to the JSON files cited anywhere in the enclosing
experiment section (### X<k>). Unmatched numbers are listed in reports/v3/report_check.txt.

A number "matches" a JSON value v when, at the number's printed precision d (decimals),
|x − c·v| ≤ 0.5·10^−d for one of the conversions c·v below (percent, GB, GiB, minutes, ms, ...),
also with the sign flipped. Rounding to a coarser precision is therefore accepted; anything else
is reported. Small integers (|x| ≤ 10), dates, section references (§1.4), and numbers inside code
spans, links or image paths are skipped (they are labels, arm sizes or keys, not results).

Sources recognised in a line: the v3 summaries (x<k>.json, gpu_benchmarks.json), the v3 raw files
(raw/x<k>.jsonl: every number in every row, plus the row count; only on the line that cites it,
never as a section default, because its many values would match almost anything) and the v2
results cited for the
v2 comparison (v2/<name>.json, read from reports/v2/).

Documented derivations. Some printed numbers are computed from JSON values rather than copied
(a difference, a ratio, a sum, an sd from a variance, a seed × arm product, a mean over raw rows).
A line that states this with the word "derived" (e.g. "(derived: 50 − 14)") has its unmatched
numbers listed in a separate "Documented derivations" section of the output instead of under
"Unmatched". The other numbers on such a line are still checked as usual, and every derivation
stays visible in the output for a reviewer to recompute.

usage: python experiments/x_report_check.py [--report reports/v3/report.md] [--out reports/v3/report_check.txt]
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "reports" / "v3"

NUM = re.compile(r"(?<![\w.\-/])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?(?![\w])|(?<![\w.\-/])-?\d+(?:\.\d+)?(?:e-?\d+)?%?(?![\w])")
JSON_REF = re.compile(r"\b((?:raw/)?x[\w]*\.json|raw/x\d+\.jsonl|gpu_benchmarks\.json|v2/[\w.]+?\.json)\b")
DERIVED = re.compile(r"\bderived\b", re.IGNORECASE)
STRIP = [
    re.compile(r"`[^`]*`"),                      # code spans (keys, commands)
    re.compile(r"!\[[^\]]*\]\([^)]*\)"),         # images
    re.compile(r"\]\([^)]*\)"),                  # link targets
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),        # dates
    re.compile(r"\b[Xx]\d+\b|\b[A-Z]\d+[a-z]?\b"),  # experiment / decision ids (X10, D7, R31, 4a)
    re.compile(r"cuda:\d"),
    re.compile(r"§\s?\d+(?:\.\d+)*"),           # section references (plan §1.4, §14)
]
# value conversions tried for every JSON number
CONVERSIONS = (1.0, 100.0, 1e-9, 2.0 ** -30, 2.0 ** -20, 1 / 60, 1e3, 1e-6, 1e-3, 1 / 3600)


def numbers_in(obj, out: list[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        for m in re.finditer(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", obj):
            try:
                out.append(float(m.group()))
            except ValueError:
                pass
    elif isinstance(obj, dict):
        for k, v in obj.items():
            numbers_in(k, out)
            numbers_in(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            numbers_in(v, out)


_cache: dict[str, list[float]] = {}


def values_of(name: str) -> list[float]:
    if name not in _cache:
        path = (V3.parent / name) if name.startswith("v2/") else (V3 / name)
        vals: list[float] = []
        if path.suffix == ".jsonl" and path.exists():  # cited raw file: every row, plus the row count
            n = 0
            with path.open() as fh:
                for line in fh:
                    if line.strip():
                        numbers_in(json.loads(line), vals)
                        n += 1
            vals.append(float(n))
        elif path.suffix == ".json" and path.exists():
            numbers_in(json.loads(path.read_text()), vals)
            raw = V3 / "raw" / (path.stem + ".jsonl")  # sections quote "(N rows)" of the raw file
            if raw.exists():
                with raw.open() as fh:
                    vals.append(float(sum(1 for _ in fh)))
        # drop NaN/inf: they break the sorted order that bisect relies on
        conv = sorted({round(c * s * v, 12) for v in vals if math.isfinite(v)
                       for c in CONVERSIONS for s in (1, -1)})
        _cache[name] = conv
    return _cache[name]


def matches(x: float, decimals: int, files: list[str]) -> bool:
    tol = 0.5 * 10 ** (-decimals) + 1e-9 * max(1.0, abs(x))
    for f in files:
        vals = values_of(f)
        i = bisect.bisect_left(vals, x - tol)
        if i < len(vals) and vals[i] <= x + tol:
            return True
    return False


def parse(tok: str) -> tuple[float, int]:
    t = tok.replace(",", "").rstrip("%")
    dec = len(t.split(".")[1].split("e")[0]) if "." in t else 0
    if "e" in t:  # 6e-8 style: precision of the mantissa in absolute terms
        mant, exp = t.split("e")
        dec = (len(mant.split(".")[1]) if "." in mant else 0) - int(exp)
    return float(t), dec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(V3 / "report.md"))
    ap.add_argument("--out", default=str(V3 / "report_check.txt"))
    a = ap.parse_args()
    lines = Path(a.report).read_text().split("\n")

    # section default sources
    section_files: list[list[str]] = []
    cur: list[str] = []
    sec_start = 0
    starts = [i for i, l in enumerate(lines) if l.startswith("## ") or l.startswith("### ")] + [len(lines)]
    owner = [0] * len(lines)
    for s, e in zip(starts, starts[1:]):
        # section defaults: summaries only; a raw file counts only on the line that cites it
        files = sorted(f for f in set(JSON_REF.findall("\n".join(lines[s:e]))) if not f.endswith(".jsonl"))
        m = re.match(r"### (X\d+)", lines[s])
        if m:
            files = sorted(set(files) | {f"{m.group(1).lower()}.json"})
        section_files.append(files)
        for i in range(s, e):
            owner[i] = len(section_files) - 1
    del cur, sec_start

    checked = matched = 0
    unmatched: list[str] = []
    derived: list[str] = []
    no_source: list[str] = []
    in_code = False
    for i, raw_line in enumerate(lines):
        if raw_line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code or raw_line.startswith("#") or raw_line.lstrip().startswith(">"):
            continue  # headings and quoted pre-registered predictions are not results
        line_files = sorted(set(JSON_REF.findall(raw_line)))
        files = line_files or (section_files[owner[i]] if starts[0] <= i else [])
        text = raw_line.replace("−", "-").replace("–", " – ")
        for rx in STRIP:
            text = rx.sub(" ", text)
        for m in NUM.finditer(text):
            tok = m.group()
            x, dec = parse(tok)
            if abs(x) <= 10 and dec == 0 and "%" not in tok:
                continue
            checked += 1
            if not files:
                no_source.append(f"L{i + 1}: {tok}  | {raw_line.strip()[:140]}")
                continue
            if matches(x, dec, files) or (tok.endswith("%") and matches(x / 100, dec + 2, files)):
                matched += 1
            elif DERIVED.search(raw_line):
                derived.append(f"L{i + 1}: {tok}  [{', '.join(files)}]  | {raw_line.strip()[:140]}")
            else:
                unmatched.append(f"L{i + 1}: {tok}  [{', '.join(files)}]  | {raw_line.strip()[:140]}")

    out = [
        f"report: {a.report}",
        f"numbers checked: {checked}; matched: {matched}; documented derivations: {len(derived)}; "
        f"unmatched: {len(unmatched)}; no cited source: {len(no_source)}",
        "",
        "## Unmatched (number not found, after rounding, in the cited JSON files)",
        *unmatched,
        "",
        "## Documented derivations (not a JSON value; the line says how it is derived)",
        *derived,
        "",
        "## No cited source (line outside any experiment section and without a JSON link)",
        *no_source,
    ]
    Path(a.out).write_text("\n".join(out) + "\n")
    print(out[1])


if __name__ == "__main__":
    main()

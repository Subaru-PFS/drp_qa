#!/usr/bin/env python3
"""Report Ruff findings only on lines a change actually touched.

The repository is not Ruff-clean, so gating CI on whole files would fail pull
requests over findings they did not introduce — a one-line import fix in a
1200-line module would be blocked by seven pre-existing violations elsewhere in
that file. CI that fails for reasons the author cannot act on gets ignored, and
then it gates nothing at all.

So this runs ``ruff check`` over the changed files, then keeps only the
diagnostics whose line falls inside a range the diff actually added or
modified. New code is held to the full rule set; old code is left for the
deliberate cleanup described in ``doc/qa-rebuild-plan.md``.

Usage
-----
    python .github/scripts/ruff_changed_lines.py <base-ref>

Exits 1 if any finding lands on a changed line, 0 otherwise. Stdlib only, so it
needs no environment beyond Ruff itself being on PATH.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

# "@@ -old,count +new,count @@" — we only care about the + side.
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def changedLines(baseRef: str) -> dict[str, set[int]]:
    """Map each changed Python file to the set of line numbers it touched.

    Parameters
    ----------
    baseRef : str
        Git ref to diff against.

    Returns
    -------
    dict[str, set[int]]
        Absolute file path to the set of added or modified line numbers.
    """
    # -U0 so each hunk covers only changed lines, not surrounding context.
    diff = subprocess.run(
        ["git", "diff", "-U0", "--diff-filter=ACMR", baseRef, "HEAD", "--", "*.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    result: dict[str, set[int]] = {}
    current: str | None = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = str(Path(line[6:]).resolve())
            result.setdefault(current, set())
        elif current and (m := _HUNK.match(line)):
            start = int(m.group(1))
            count = int(m.group(2)) if m.group(2) is not None else 1
            result[current].update(range(start, start + count))
    # Files deleted in the diff have no + side; drop anything empty.
    return {f: lines for f, lines in result.items() if lines}


def main(argv: list[str]) -> int:
    """Run Ruff over changed files and report findings on changed lines.

    Parameters
    ----------
    argv : list[str]
        Command-line arguments; ``argv[0]`` is the base ref.

    Returns
    -------
    int
        1 if any finding lands on a changed line, 0 otherwise.
    """
    if len(argv) != 1:
        print(__doc__)
        return 2
    baseRef = argv[0]

    touched = changedLines(baseRef)
    if not touched:
        print("No changed Python files.")
        return 0

    existing = [f for f in touched if Path(f).exists()]
    if not existing:
        print("No changed Python files remain on disk.")
        return 0

    print(f"Checking {len(existing)} changed file(s) against {baseRef}\n")

    proc = subprocess.run(
        ["ruff", "check", "--force-exclude", "--output-format=json", "--", *existing],
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0, 1):
        sys.stderr.write(proc.stderr)
        return proc.returncode

    findings = json.loads(proc.stdout or "[]")

    onChangedLines = []
    preExisting = 0
    for f in findings:
        path = str(Path(f["filename"]).resolve())
        row = (f.get("location") or {}).get("row")
        if row is not None and row in touched.get(path, set()):
            onChangedLines.append(f)
        else:
            preExisting += 1

    for f in onChangedLines:
        rel = Path(f["filename"]).name
        loc = f.get("location") or {}
        print(f"{rel}:{loc.get('row')}:{loc.get('column')}: {f.get('code')} {f.get('message')}")

    print(
        f"\n{len(onChangedLines)} finding(s) on changed lines; "
        f"{preExisting} pre-existing finding(s) in those files ignored."
    )
    if onChangedLines:
        print("\nFix the findings above, or adjust the code you changed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

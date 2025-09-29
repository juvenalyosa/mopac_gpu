#!/usr/bin/env python3
"""Quick inspection helper: counts how often each iab/jba block size occurs in MOZYME Fock pair loops."""
from pathlib import Path
import re
import sys
from collections import Counter

def parse_counts(text: str) -> Counter:
    counter: Counter[tuple[int, int]] = Counter()
    pattern = re.compile(r"\(iab=(\d+), jba=(\d+)\)")
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            counter[(int(match.group(1)), int(match.group(2)))] += 1
    return counter

def main() -> None:
    if len(sys.argv) < 2:
        print("usage: probe_mozyme_pairs.py LOG", file=sys.stderr)
        sys.exit(1)
    log_path = Path(sys.argv[1])
    if not log_path.exists():
        print(f"File not found: {log_path}", file=sys.stderr)
        sys.exit(1)
    counts = parse_counts(log_path.read_text(encoding="utf-8", errors="ignore"))
    for (iab, jba), count in sorted(counts.items()):
        print(f"{iab:2d} x {jba:2d}: {count}")

if __name__ == "__main__":
    main()

"""Trim / sample a large CSV down to a RAG-friendly size.

Large datasets (e.g. a 1.1 GB Spotify export) exceed what local CPU
embedding can process in reasonable time. This script keeps the header
plus the first N data rows (or a random sample) and writes a new CSV
that fits the upload limit.

Usage:

    python scripts/trim_csv.py input.csv -o sample.csv --rows 50000
    python scripts/trim_csv.py input.csv -o sample.csv --rows 10000 --random
"""

from __future__ import annotations

import argparse
import csv
import random
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="large source CSV")
    parser.add_argument("-o", "--output", required=True, help="trimmed CSV to write")
    parser.add_argument(
        "--rows",
        type=int,
        default=50_000,
        help="number of data rows to keep (default: 50000)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="randomly sample rows instead of keeping the first N "
        "(reads the whole file twice)",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            print("input is empty")
            return 1
        if args.random:
            print("counting rows (pass 1/2)…")
            total = sum(1 for _ in reader)
            rows_to_keep = min(args.rows, total)
            keep = set(random.sample(range(total), rows_to_keep))
            fh.seek(0)
            next(reader)  # skip header
            picked = (row for i, row in enumerate(reader) if i in keep)
        else:
            picked = (
                row
                for i, row in enumerate(reader)
                if i < args.rows
            )
            rows_to_keep = args.rows
        print(f"writing up to {rows_to_keep} rows…")
        with open(args.output, "w", encoding="utf-8", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(header)
            written = 0
            for row in picked:
                writer.writerow(row)
                written += 1

    print(f"done: {written} data rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

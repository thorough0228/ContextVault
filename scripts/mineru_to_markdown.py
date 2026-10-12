"""MinerU output → clean markdown converter (CLI wrapper).

Heavy lifting lives in ``apps/api/app/ingest/mineru_convert.py`` so
the worker's PDF parser can reuse the same implementation.

Usage:
    python scripts/mineru_to_markdown.py <mineru_output_dir> -o out.md
    python scripts/mineru_to_markdown.py <dir> --from-content-list -o out.md
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps" / "api"))

from app.ingest.mineru_convert import convert_directory  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert a MinerU output directory into clean markdown."
    )
    ap.add_argument("directory", help="MinerU output directory")
    ap.add_argument("-o", "--output", required=True, help="clean .md path to write")
    ap.add_argument(
        "--from-content-list",
        action="store_true",
        help="rebuild from content_list.json instead of the .md",
    )
    args = ap.parse_args()

    text = convert_directory(
        Path(args.directory), from_content_list=args.from_content_list
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text)} chars)")


if __name__ == "__main__":
    main()

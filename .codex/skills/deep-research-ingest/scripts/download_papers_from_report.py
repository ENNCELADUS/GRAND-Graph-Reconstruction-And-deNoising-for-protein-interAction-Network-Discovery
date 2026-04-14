#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

URL_PATTERN = re.compile(r"https?://[^\s`)>]+")
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
TITLE_PATTERN = re.compile(r"^\s*(?:\d+\)\s*)?\*\*Full title:\*\*\s*(.+?)\s*$")
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "towards",
    "using",
    "via",
    "with",
}
HOST_ALIASES = {
    "openreview.net": "openreview",
    "arxiv.org": "arxiv",
    "biorxiv.org": "biorxiv",
    "nature.com": "nature",
    "academic.oup.com": "oup",
    "pmc.ncbi.nlm.nih.gov": "pmc",
    "link.springer.com": "springer",
    "springer.com": "springer",
    "openaccess.thecvf.com": "cvf",
    "proceedings.iclr.cc": "iclr",
    "proceedings.mlr.press": "pmlr",
}


@dataclasses.dataclass
class PaperEntry:
    title: str
    year: int | None = None
    pdf_url: str | None = None
    abstract_url: str | None = None
    section: str = "all"


def clean_text(text: str) -> str:
    text = re.sub(r".*?", "", text)
    text = MARKDOWN_LINK_PATTERN.sub(r"\1", text)
    text = text.replace("`", "")
    text = text.replace("**", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -*")


def extract_first_url(text: str) -> str | None:
    match = URL_PATTERN.search(text)
    if not match:
        return None
    return match.group(0).rstrip(").,")


def extract_year(text: str) -> int | None:
    match = YEAR_PATTERN.search(text)
    if not match:
        return None
    return int(match.group(0))


def parse_field_entries(text: str) -> list[PaperEntry]:
    entries: list[PaperEntry] = []
    current: PaperEntry | None = None
    section = "all"

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("## A."):
            section = "top"
            continue
        if line.startswith("## B."):
            section = "secondary"
            continue

        title_match = TITLE_PATTERN.match(line)
        if title_match:
            if current and current.title:
                entries.append(current)
            current = PaperEntry(title=clean_text(title_match.group(1)), section=section)
            continue

        if current is None:
            continue

        if "**Year:**" in line and current.year is None:
            current.year = extract_year(line)
        elif "**Direct PDF URL:**" in line and current.pdf_url is None:
            current.pdf_url = extract_first_url(line)
        elif "**URL to abstract page:**" in line and current.abstract_url is None:
            current.abstract_url = extract_first_url(line)

    if current and current.title:
        entries.append(current)

    return entries


def parse_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_separator_row(line: str) -> bool:
    stripped = line.strip().strip("|").replace(":", "").replace("-", "").replace(" ", "")
    return stripped == ""


def parse_table_entries(text: str) -> list[PaperEntry]:
    entries: list[PaperEntry] = []
    lines = text.splitlines()
    index = 0

    while index < len(lines):
        line = lines[index].strip()
        if (
            line.startswith("|")
            and "paper" in line.lower()
            and ("pdf route" in line.lower() or "recommended vault filename" in line.lower())
        ):
            headers = [header.lower() for header in parse_markdown_row(line)]
            index += 1
            while index < len(lines) and lines[index].strip().startswith("|"):
                row_line = lines[index].strip()
                if is_separator_row(row_line):
                    index += 1
                    continue

                cells = parse_markdown_row(row_line)
                if len(cells) != len(headers):
                    index += 1
                    continue

                row = dict(zip(headers, cells))
                title = clean_text(row.get("paper", ""))
                if not title:
                    index += 1
                    continue

                priority = clean_text(row.get("priority", "")).upper()
                section = "all"
                if priority == "A":
                    section = "top"
                elif priority in {"B", "C"}:
                    section = "secondary"

                entries.append(
                    PaperEntry(
                        title=title,
                        year=extract_year(row.get("year", "")),
                        pdf_url=extract_first_url(row.get("pdf route", "")),
                        section=section,
                    )
                )
                index += 1
            continue

        index += 1

    return entries


def deduplicate(entries: list[PaperEntry]) -> list[PaperEntry]:
    deduped: list[PaperEntry] = []
    seen: set[str] = set()
    for entry in entries:
        key = clean_text(entry.title).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def match_section(entry_section: str, requested_section: str) -> bool:
    if requested_section == "all":
        return True
    if entry_section == "all":
        return True
    return entry_section == requested_section


def slugify(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def short_title_slug(title: str, max_words: int = 8, max_length: int = 64) -> str:
    raw_words = [word for word in slugify(title).split("_") if word]
    filtered_words = [word for word in raw_words if word not in STOPWORDS]
    words = filtered_words or raw_words or ["paper"]
    slug = "_".join(words[:max_words])
    return slug[:max_length].strip("_") or "paper"


def source_from_url(url: str | None) -> str:
    if not url:
        return "source"
    host = urllib.parse.urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    for suffix, alias in HOST_ALIASES.items():
        if host.endswith(suffix):
            return alias
    fallback = host.split(".")[0]
    fallback = re.sub(r"[^a-z0-9]+", "_", fallback)
    return fallback or "source"


def guess_pdf_url(abstract_url: str | None) -> str | None:
    if not abstract_url:
        return None

    if "arxiv.org/abs/" in abstract_url:
        return abstract_url.replace("/abs/", "/pdf/")
    if "openreview.net/forum?id=" in abstract_url:
        return abstract_url.replace("/forum?", "/pdf?")
    if "nature.com/articles/" in abstract_url and not abstract_url.endswith(".pdf"):
        return abstract_url.rstrip("/") + ".pdf"
    if "biorxiv.org/content/" in abstract_url and ".full.pdf" not in abstract_url:
        return abstract_url.rstrip("/") + ".full.pdf"

    return None


def build_filename(entry: PaperEntry, url: str) -> str:
    year = str(entry.year) if entry.year is not None else "unknown"
    source = source_from_url(url)
    short_name = short_title_slug(entry.title)
    return f"{year}_{source}_{short_name}.pdf"


def download_pdf(url: str, destination: pathlib.Path, timeout: int) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; DeepResearchIngest/1.0)",
            "Accept": "application/pdf,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
        content_type = (response.headers.get("Content-Type") or "").lower()

    if b"%PDF" not in data[:1024] and "application/pdf" not in content_type:
        raise ValueError(
            f"response did not look like a PDF (content-type={content_type or 'unknown'})"
        )

    temp_path = destination.with_suffix(destination.suffix + ".part")
    temp_path.write_bytes(data)
    temp_path.replace(destination)


def load_entries(report_path: pathlib.Path) -> list[PaperEntry]:
    text = report_path.read_text(encoding="utf-8")
    entries = parse_field_entries(text)
    entries.extend(parse_table_entries(text))
    return deduplicate(entries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PDFs from a Deep Research markdown report."
    )
    parser.add_argument(
        "--report",
        default="notes/03_Projects/deep-research-report.md",
        help="Path to the Deep Research markdown report.",
    )
    parser.add_argument(
        "--dest",
        default="literature",
        help="Directory to store downloaded PDFs.",
    )
    parser.add_argument(
        "--section",
        choices=("top", "secondary", "all"),
        default="top",
        help="Which report section to download from.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of entries to process after filtering.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files with the same generated filename.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned downloads without writing files.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-download timeout in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = pathlib.Path(args.report)
    destination_dir = pathlib.Path(args.dest)

    if not report_path.exists():
        print(f"ERROR: report not found: {report_path}", file=sys.stderr)
        return 1

    entries = [
        entry for entry in load_entries(report_path) if match_section(entry.section, args.section)
    ]
    if args.limit is not None:
        entries = entries[: args.limit]

    if not entries:
        print("ERROR: no matching report entries were found.", file=sys.stderr)
        return 1

    if not args.dry_run:
        destination_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0

    for entry in entries:
        pdf_url = entry.pdf_url or guess_pdf_url(entry.abstract_url)
        if not pdf_url:
            print(f"SKIP: no PDF URL for '{entry.title}'")
            skipped += 1
            continue

        filename = build_filename(entry, pdf_url)
        output_path = destination_dir / filename

        if args.dry_run:
            print(f"PLAN: {output_path} <- {pdf_url}")
            continue

        if output_path.exists() and not args.overwrite:
            print(f"SKIP: exists {output_path}")
            skipped += 1
            continue

        try:
            download_pdf(pdf_url, output_path, timeout=args.timeout)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as error:
            print(f"FAIL: {entry.title} <- {pdf_url} ({error})")
            failed += 1
            continue

        print(f"OK: {output_path}")
        downloaded += 1

    if args.dry_run:
        print(f"DRY RUN COMPLETE: {len(entries)} planned, section={args.section}")
        return 0

    print(
        f"DONE: downloaded={downloaded} skipped={skipped} failed={failed} from report={report_path}"
    )
    return 0 if downloaded or skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())

import glob
import os
import re
import sys

import pandas
from pypdf import PdfReader

try:
    from web.scraper import DATA_DIR
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from web.scraper import DATA_DIR


AGENTS_DIR = os.path.join(DATA_DIR, "agents")
TOBACCO_KEYWORDS_PATH = os.path.join(DATA_DIR, "tobacco-list.txt")
OUTPUT_PATH = os.path.join(DATA_DIR, "kentucky_tobacco_lobbying_2009_2023.csv")
AUDIT_PATH = os.path.join(DATA_DIR, "source_audit.md")
START_YEAR = 2009
END_YEAR = 2023
MANUAL_TOTAL_REGISTRATIONS = {
    2009: 640,
    2010: 625,
    2011: 595,
    2012: 628,
    2013: 588,
    2014: 557,
    2015: 597,
    2016: 611,
    2017: 576,
    2018: 581,
    2019: 593,
    2020: 733,
    2021: 652,
    2022: 675,
    2023: 694,
}

YEAR_RE = re.compile(r"(\d{4})")
PHONE_RE = re.compile(r"\s+(?:\(?\d{3}\)?[- ]?)\d{3}-\d{4}\s*$")
ZIP_RE = re.compile(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?")


def build_kentucky_tobacco_lobbying_dataset():
    existing_rows = _read_existing_output()
    tobacco_counts = _count_tobacco_agent_registrations_by_year()
    manually_preserved_years = []
    rows = []

    for year in range(START_YEAR, END_YEAR + 1):
        existing = existing_rows.get(year, {})
        total_count = _total_registrations_for_year(year, existing)
        existing_tobacco_count = _blank_if_missing(
            existing.get("tobacco_lobbyist_registrations")
        )
        tobacco_count = tobacco_counts.get(year, existing_tobacco_count)
        if year not in tobacco_counts and existing_tobacco_count:
            manually_preserved_years.append(year)
        ratio = _ratio(tobacco_count, total_count)

        rows.append(
            {
                "state": "Kentucky",
                "year": year,
                "tobacco_lobbyist_registrations": tobacco_count,
                "total_lobbyist_registrations": total_count,
                "tobacco_lobbyist_ratio": ratio,
            }
        )

    output = pandas.DataFrame(rows)
    _validate_years(output)
    output.to_csv(OUTPUT_PATH, index=False)
    _write_source_audit(tobacco_counts, manually_preserved_years)
    return output


def _read_existing_output():
    if not os.path.exists(OUTPUT_PATH):
        return {}

    data = pandas.read_csv(OUTPUT_PATH, dtype=str).fillna("")
    return {
        int(row["year"]): row.to_dict()
        for _, row in data.iterrows()
        if str(row.get("year", "")).strip().isdigit()
    }


def _count_tobacco_agent_registrations_by_year():
    keyword_pattern = _tobacco_keyword_pattern()
    counts = {}

    for pdf_path in sorted(glob.glob(os.path.join(AGENTS_DIR, "*.pdf"))):
        year = _year_from_path(pdf_path)
        if year is None or not START_YEAR <= year <= END_YEAR:
            continue

        count = 0
        for line in _extract_pdf_lines(pdf_path):
            employer = _extract_employer_from_registration_line(line)
            if employer and keyword_pattern.search(employer):
                count += 1

        counts[year] = count

    return counts


def _tobacco_keyword_pattern():
    with open(TOBACCO_KEYWORDS_PATH, encoding="utf-8") as keyword_file:
        keywords = [
            line.strip()
            for line in keyword_file
            if line.strip() and not line.strip().startswith("#")
        ]

    if not keywords:
        raise RuntimeError(f"No tobacco keywords found in {TOBACCO_KEYWORDS_PATH}")

    return re.compile("|".join(re.escape(keyword) for keyword in keywords), re.I)


def _year_from_path(path):
    match = YEAR_RE.search(os.path.basename(path))
    if not match:
        return None
    return int(match.group(1))


def _extract_pdf_lines(pdf_path):
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text = page.extract_text() or ""
        for line in text.splitlines():
            cleaned = " ".join(line.split())
            if cleaned:
                yield cleaned


def _extract_employer_from_registration_line(line):
    zip_matches = list(ZIP_RE.finditer(line))
    if not zip_matches:
        return ""

    employer_with_phone = line[zip_matches[-1].end() :].strip()
    return PHONE_RE.sub("", employer_with_phone).strip()


def _blank_if_missing(value):
    if value is None or pandas.isna(value):
        return ""

    value = str(value).strip()
    return "" if value.lower() == "nan" else value


def _total_registrations_for_year(year, existing):
    value = _blank_if_missing(existing.get("total_lobbyist_registrations"))
    if value:
        return value

    fallback_value = MANUAL_TOTAL_REGISTRATIONS.get(year)
    return fallback_value if fallback_value is not None else ""


def _ratio(tobacco_count, total_count):
    if tobacco_count == "" or total_count == "":
        return ""

    return int(tobacco_count) / int(total_count)


def _validate_years(output):
    expected_years = set(range(START_YEAR, END_YEAR + 1))
    actual_years = set(output["year"])
    missing_years = expected_years - actual_years
    if missing_years:
        raise RuntimeError(f"Missing required years: {sorted(missing_years)}")


def _write_source_audit(tobacco_counts, manually_preserved_years):
    lines = [
        "# Kentucky Tobacco Lobbying Source Audit",
        "",
        "## Output",
        "",
        f"- CSV: `data/{os.path.basename(OUTPUT_PATH)}`",
        "- Required years 2009-2022 are present; 2023 is also present.",
        "- Years without a local KLEC agent PDF are left blank for `tobacco_lobbyist_registrations` and `tobacco_lobbyist_ratio` unless those values were manually entered before running the builder.",
        "- `total_lobbyist_registrations` values are preserved from the manually entered CSV.",
        "",
        "## Numerator: tobacco_lobbyist_registrations",
        "",
        f"- Source files: `data/agents/*.pdf`",
        f"- Keyword file: `data/{os.path.basename(TOBACCO_KEYWORDS_PATH)}`",
        "- Extraction method: parsed each KLEC active-agent PDF with `pypdf`, extracted employer names from registration rows, and counted each row whose employer matched at least one keyword.",
        "- Count type: lobbyist registrations, not unique lobbyists and not unique employers.",
        "- Matching is done against parsed employer names only, so address-only matches such as a street named `Tobacco Road` are not counted.",
        f"- PDF-extracted years: {', '.join(str(year) for year in sorted(tobacco_counts)) or 'none'}.",
        f"- Manually preserved tobacco-count years: {', '.join(str(year) for year in sorted(manually_preserved_years)) or 'none'}.",
        "",
        "## Denominator: total_lobbyist_registrations",
        "",
        "- Source: manually entered totals in `data/kentucky_tobacco_lobbying_2009_2023.csv`.",
        "- Denominator type: total registered lobbyists/agents as entered by the user.",
        "",
        "## Assumptions and warnings",
        "",
        "- The keyword list is treated as authoritative. Broad keywords like `tobacco` can match advocacy employers as well as industry employers if they appear in the employer name.",
        "- Ratios are computed only when both numerator and denominator are present.",
        "- Do not treat blank numerator or ratio values as zero.",
    ]
    with open(AUDIT_PATH, "w", encoding="utf-8") as audit_file:
        audit_file.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    build_kentucky_tobacco_lobbying_dataset()

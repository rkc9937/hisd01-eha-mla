from io import BytesIO
import os
import unicodedata
import zipfile

import pandas
import requests

from web.scraper import (
    ACS_EDUCATION_HIGH_SCHOOL_OR_HIGHER_VARS,
    ACS_EDUCATION_TOTAL_VAR,
    ACS_END_YEAR,
    ACS_START_YEAR,
    DATA_DIR,
    DIRECT_EXPENDITURE_PREFIXES,
    ENV_PATH,
    GOVERNMENT_FINANCE_ZIP_URLS,
    INTERGOVERNMENTAL_EXPENDITURE_PREFIXES,
    _clean_text,
    _divide,
    _fetch_census_json,
    _parse_int,
    _parse_number,
    _percent,
    _sum_numbers,
)


KENTUCKY_STATE_FIPS = "21"
KENTUCKY_ADOPTION_PATH = os.path.join(DATA_DIR, "kentucky_community_smoke_free.csv")
KENTUCKY_SMOKER_PATH = os.path.join(DATA_DIR, "kentucky_smoker_percentage_year.csv")
KENTUCKY_STATE_POPULATION_PATH = os.path.join(DATA_DIR, "kentucky_state_pop.csv")
KENTUCKY_GEOGRAPHIES_PATH = os.path.join(
    DATA_DIR,
    "kentucky_refinement_1_geographies.csv",
)
KENTUCKY_POPULATION_BY_YEAR_PATH = os.path.join(
    DATA_DIR,
    "kentucky_refinement_2_population_by_year.csv",
)
KENTUCKY_EDUCATION_PATH = os.path.join(DATA_DIR, "kentucky_refinement_2_education.csv")
KENTUCKY_INCOME_PATH = os.path.join(
    DATA_DIR,
    "kentucky_refinement_2_per_capita_income.csv",
)
KENTUCKY_WHITE_PATH = os.path.join(
    DATA_DIR,
    "kentucky_refinement_2_non_hispanic_white.csv",
)
KENTUCKY_GOVERNMENT_SPENDING_PATH = os.path.join(
    DATA_DIR,
    "kentucky_refinement_3_government_spending.csv",
)
KENTUCKY_EHA_PANEL_PATH = os.path.join(DATA_DIR, "kentucky_refinement_5_eha_panel.csv")
KENTUCKY_DROPPED_MUNICIPALITIES_PATH = os.path.join(
    DATA_DIR,
    "kentucky_dropped_municipalities.csv",
)

KENTUCKY_PLACE_ALIASES = {
    "lexington fayette county": ("place", "lexington fayette"),
}
KENTUCKY_EXCLUDED_MUNICIPALITIES = {
    "ashland": "Missing 2022 Census government-finance match.",
    "middlesboro": "Missing ACS place match and government-spending match.",
}


def get_kentucky_covariates(
    start_year=ACS_START_YEAR,
    end_year=ACS_END_YEAR,
    government_spending_year=2022,
    census_api_key=None,
):
    geographies = get_kentucky_geographies()
    yearly_covariates = get_kentucky_acs_covariates(
        geographies,
        start_year,
        end_year,
        census_api_key,
    )
    state_population = get_kentucky_state_population(
        start_year,
        end_year,
        census_api_key,
    )
    government_spending = get_kentucky_government_spending(
        geographies,
        government_spending_year,
    )
    smoker = get_kentucky_smoker_percentages(start_year, end_year)
    panel = get_kentucky_eha_panel(
        geographies,
        yearly_covariates,
        government_spending,
        smoker,
        state_population,
        start_year,
        end_year,
    )
    return {
        "geographies": geographies,
        **yearly_covariates,
        "government_spending": government_spending,
        "smoker": smoker,
        "state_population": state_population,
        "eha_panel": panel,
    }


def get_kentucky_geographies():
    adoption = pandas.read_csv(KENTUCKY_ADOPTION_PATH)
    rows = []

    for _, row in adoption.iterrows():
        name = _clean_text(row["municipality"])
        geography_type, lookup_name = _kentucky_lookup(name)
        rows.append(
            {
                "id": row["id"],
                "Municipality": name,
                "geography_type": geography_type,
                "lookup_name": lookup_name,
                "adoption_month": _clean_text(row["adoption_month"]),
                "adoption_year": int(row["adoption_year"]),
                "adoption_date_note": _clean_text(row["adoption_date_note"]),
            }
        )

    geographies = pandas.DataFrame(rows).drop_duplicates(
        subset=["Municipality", "geography_type"],
        keep="first",
    )
    geographies.to_csv(KENTUCKY_GEOGRAPHIES_PATH, index=False)
    return geographies


def get_kentucky_acs_covariates(
    geographies=None,
    start_year=ACS_START_YEAR,
    end_year=ACS_END_YEAR,
    census_api_key=None,
):
    if geographies is None:
        geographies = get_kentucky_geographies()

    population_rows = []
    education_rows = []
    income_rows = []
    white_rows = []

    for year in range(start_year, end_year + 1):
        acs_records = _fetch_kentucky_acs_records(year, census_api_key)

        for _, geography in geographies.iterrows():
            record = _find_acs_record(geography, acs_records)
            base_row = _kentucky_base_year_row(geography, year, record)

            population_rows.append(
                {
                    **base_row,
                    "Population": record.get("population", "") if record else "",
                    "Source": "Census ACS 5-year B01003_001E",
                }
            )
            education_rows.append(
                {
                    **base_row,
                    "percent_25_plus_high_school_or_higher": record.get(
                        "percent_25_plus_high_school_or_higher",
                        "",
                    )
                    if record
                    else "",
                }
            )
            income_rows.append(
                {
                    **base_row,
                    "per_capita_income": record.get("per_capita_income", "")
                    if record
                    else "",
                }
            )
            white_rows.append(
                {
                    **base_row,
                    "non_hispanic_white_share": record.get(
                        "non_hispanic_white_share",
                        "",
                    )
                    if record
                    else "",
                    "non_hispanic_white_count": record.get(
                        "non_hispanic_white_count",
                        "",
                    )
                    if record
                    else "",
                    "total_population_for_race": record.get(
                        "total_population_for_race",
                        "",
                    )
                    if record
                    else "",
                }
            )

    outputs = {
        "population_by_year": pandas.DataFrame(population_rows),
        "education": pandas.DataFrame(education_rows),
        "per_capita_income": pandas.DataFrame(income_rows),
        "non_hispanic_white": pandas.DataFrame(white_rows),
    }
    outputs["population_by_year"].to_csv(KENTUCKY_POPULATION_BY_YEAR_PATH, index=False)
    outputs["education"].to_csv(KENTUCKY_EDUCATION_PATH, index=False)
    outputs["per_capita_income"].to_csv(KENTUCKY_INCOME_PATH, index=False)
    outputs["non_hispanic_white"].to_csv(KENTUCKY_WHITE_PATH, index=False)
    return outputs


def get_kentucky_government_spending(geographies=None, year=2022):
    if geographies is None:
        geographies = get_kentucky_geographies()

    finance_zip = _fetch_government_finance_zip(year)
    finance_units = _parse_government_finance_units(finance_zip)
    expenditures = _parse_government_finance_expenditures(finance_zip)
    unit_index = _build_finance_unit_index(finance_units)
    output_rows = []

    for _, geography in geographies.iterrows():
        unit = _find_finance_unit(geography, unit_index)
        spending = expenditures.get(unit["government_id"], {}) if unit else {}
        total_expenditure = spending.get("total_expenditure", "")
        finance_population = unit.get("population", "") if unit else ""
        output_rows.append(
            {
                "Municipality": geography["Municipality"],
                "geography_type": geography["geography_type"],
                "adoption_year": geography["adoption_year"],
                "Year": year,
                "Government Unit ID": unit.get("government_id", "") if unit else "",
                "Government Name": unit.get("government_name", "") if unit else "",
                "Government County": unit.get("county", "") if unit else "",
                "Place FIPS": unit.get("place_fips", "") if unit else "",
                "Finance Population": finance_population,
                "Finance Population Year": unit.get("population_year", "")
                if unit
                else "",
                "Total Municipal Expenditure": total_expenditure,
                "Direct Expenditure": spending.get("direct_expenditure", ""),
                "Intergovernmental Expenditure": spending.get(
                    "intergovernmental_expenditure",
                    "",
                ),
                "Per Capita Spending": _divide(total_expenditure, finance_population),
                "Amount Units": "dollars",
                "Source": "Census Annual Survey of State and Local Government Finances Individual Unit Files",
            }
        )

    output = pandas.DataFrame(output_rows)
    output.to_csv(KENTUCKY_GOVERNMENT_SPENDING_PATH, index=False)
    return output


def get_kentucky_smoker_percentages(start_year=ACS_START_YEAR, end_year=ACS_END_YEAR):
    if not os.path.exists(KENTUCKY_SMOKER_PATH):
        template = pandas.DataFrame(
            {"Year": list(range(start_year, end_year + 1)), "Percentage": ""}
        )
        template.to_csv(KENTUCKY_SMOKER_PATH, index=False)
        return template

    return pandas.read_csv(KENTUCKY_SMOKER_PATH)


def get_kentucky_state_population(
    start_year=ACS_START_YEAR,
    end_year=ACS_END_YEAR,
    census_api_key=None,
):
    rows = []
    for year in range(start_year, end_year + 1):
        record = _fetch_kentucky_state_population(year, census_api_key)
        rows.append({"Year": year, "Kentucky_Population": record["population"]})

    output = pandas.DataFrame(rows)
    output.to_csv(KENTUCKY_STATE_POPULATION_PATH, index=False)
    return output


def get_kentucky_eha_panel(
    geographies=None,
    yearly_covariates=None,
    government_spending=None,
    smoker=None,
    state_population=None,
    start_year=ACS_START_YEAR,
    end_year=ACS_END_YEAR,
):
    if geographies is None:
        geographies = get_kentucky_geographies()
    if yearly_covariates is None:
        yearly_covariates = {
            "population_by_year": pandas.read_csv(KENTUCKY_POPULATION_BY_YEAR_PATH),
            "education": pandas.read_csv(KENTUCKY_EDUCATION_PATH),
            "per_capita_income": pandas.read_csv(KENTUCKY_INCOME_PATH),
            "non_hispanic_white": pandas.read_csv(KENTUCKY_WHITE_PATH),
        }
    if government_spending is None:
        government_spending = pandas.read_csv(KENTUCKY_GOVERNMENT_SPENDING_PATH)
    if smoker is None:
        smoker = get_kentucky_smoker_percentages(start_year, end_year)
    if state_population is None:
        state_population = pandas.read_csv(KENTUCKY_STATE_POPULATION_PATH)

    _write_kentucky_dropped_municipalities(geographies)

    population = _yearly_index(yearly_covariates["population_by_year"], "Population")
    education = _yearly_index(
        yearly_covariates["education"],
        "percent_25_plus_high_school_or_higher",
    )
    income = _yearly_index(yearly_covariates["per_capita_income"], "per_capita_income")
    white = _yearly_index(
        yearly_covariates["non_hispanic_white"],
        "non_hispanic_white_share",
    )
    spending = _static_index(government_spending)
    smoker_by_year = {
        int(row["Year"]): row["Percentage"]
        for _, row in smoker.iterrows()
        if not pandas.isna(row["Year"])
    }
    state_population_by_year = {
        int(row["Year"]): row["Kentucky_Population"]
        for _, row in state_population.iterrows()
        if not pandas.isna(row["Year"])
    }
    restriction_share = _build_kentucky_restriction_share_by_year(
        geographies,
        population,
        state_population_by_year,
        start_year,
        end_year,
    )
    geographies = _drop_excluded_kentucky_municipalities(geographies)
    rows = []

    for _, geography in geographies.iterrows():
        adoption_year = int(geography["adoption_year"])
        if adoption_year < start_year:
            continue

        key = _geo_key(geography)
        spend = spending.get(key, {})
        for year in range(start_year, min(adoption_year, end_year) + 1):
            year_key = (*key, year)
            pop_record = population.get(year_key, {})
            rows.append(
                {
                    "Municipality": geography["Municipality"],
                    "geography_type": geography["geography_type"],
                    "Adoption Year": adoption_year,
                    "Year": year,
                    "time_index": year - start_year + 1,
                    "adopted_this_year": int(year == adoption_year),
                    "municipal_population": pop_record.get("value", ""),
                    "percent_25_plus_high_school_or_higher": education.get(
                        year_key,
                        {},
                    ).get("value", ""),
                    "non_hispanic_white_share": white.get(year_key, {}).get(
                        "value",
                        "",
                    ),
                    "per_capita_income": income.get(year_key, {}).get("value", ""),
                    "per_capita_spending": spend.get("Per Capita Spending", ""),
                    "total_municipal_expenditure": spend.get(
                        "Total Municipal Expenditure",
                        "",
                    ),
                    "government_spending_year": spend.get("Year", ""),
                    "kentucky_smoker_percentage": smoker_by_year.get(year, ""),
                    "kentucky_population": state_population_by_year.get(year, ""),
                    "proportion_state_population_local_restriction": restriction_share[
                        year
                    ]["proportion_state_population_local_restriction"],
                    "restricted_local_population": restriction_share[year][
                        "restricted_local_population"
                    ],
                    "Census Geography": pop_record.get("Census Geography", ""),
                    "State FIPS": pop_record.get("State FIPS", ""),
                    "Geo FIPS": pop_record.get("Geo FIPS", ""),
                }
            )

    output = pandas.DataFrame(rows)
    output.to_csv(KENTUCKY_EHA_PANEL_PATH, index=False)
    return output


def _build_kentucky_restriction_share_by_year(
    geographies,
    population,
    state_population_by_year,
    start_year,
    end_year,
):
    restriction_share = {}

    for year in range(start_year, end_year + 1):
        restricted_population = 0
        for _, geography in geographies.iterrows():
            adoption_year = int(geography["adoption_year"])
            if adoption_year >= year:
                continue

            key = (*_geo_key(geography), year)
            value = population.get(key, {}).get("value", "")
            if value != "" and not pandas.isna(value):
                restricted_population += value

        kentucky_population = state_population_by_year.get(year, "")
        restriction_share[year] = {
            "restricted_local_population": restricted_population,
            "proportion_state_population_local_restriction": _divide(
                restricted_population,
                kentucky_population,
                6,
            ),
        }

    return restriction_share


def _drop_excluded_kentucky_municipalities(geographies):
    return geographies[
        ~geographies["Municipality"]
        .map(_normalize_name)
        .isin(KENTUCKY_EXCLUDED_MUNICIPALITIES)
    ].copy()


def _write_kentucky_dropped_municipalities(geographies):
    rows = []
    for _, geography in geographies.iterrows():
        key = _normalize_name(geography["Municipality"])
        reason = KENTUCKY_EXCLUDED_MUNICIPALITIES.get(key)
        if not reason:
            continue

        rows.append(
            {
                "Municipality": geography["Municipality"],
                "geography_type": geography["geography_type"],
                "adoption_year": geography["adoption_year"],
                "reason": reason,
            }
        )

    pandas.DataFrame(rows).to_csv(KENTUCKY_DROPPED_MUNICIPALITIES_PATH, index=False)


def _fetch_kentucky_acs_records(year, census_api_key=None):
    variables = [
        "NAME",
        "B01003_001E",
        "B19301_001E",
        "B03002_001E",
        "B03002_003E",
        ACS_EDUCATION_TOTAL_VAR,
        *ACS_EDUCATION_HIGH_SCHOOL_OR_HIGHER_VARS,
    ]
    records = {}

    for geography_type, params in {
        "place": {"for": "place:*", "in": f"state:{KENTUCKY_STATE_FIPS}"},
        "county": {"for": "county:*", "in": f"state:{KENTUCKY_STATE_FIPS}"},
    }.items():
        rows = _fetch_census_json(
            _acs_detail_urls(year),
            {
                "get": ",".join(variables),
                **params,
            },
            census_api_key,
        )
        for row in rows:
            key = _normalize_census_geo_name(row["NAME"], geography_type)
            record = _parse_acs_record(row, geography_type)
            records[(geography_type, key)] = record

    return records


def _fetch_kentucky_state_population(year, census_api_key=None):
    rows = _fetch_census_json(
        _acs_detail_urls(year),
        {
            "get": "NAME,B01003_001E",
            "for": f"state:{KENTUCKY_STATE_FIPS}",
        },
        census_api_key,
    )
    row = rows[0]
    return {"name": row["NAME"], "population": _parse_number(row["B01003_001E"])}


def _parse_acs_record(row, geography_type):
    total_population = _parse_number(row["B03002_001E"])
    non_hispanic_white_count = _parse_number(row["B03002_003E"])
    education_total = _parse_number(row[ACS_EDUCATION_TOTAL_VAR])
    high_school_or_higher = _sum_numbers(
        row[var] for var in ACS_EDUCATION_HIGH_SCHOOL_OR_HIGHER_VARS
    )
    geo_fips = row["place"] if geography_type == "place" else row["county"]
    return {
        "name": row["NAME"],
        "state": row["state"],
        "geo_fips": geo_fips,
        "geography_type": geography_type,
        "population": _parse_number(row["B01003_001E"]),
        "per_capita_income": _parse_number(row["B19301_001E"]),
        "total_population_for_race": total_population,
        "non_hispanic_white_count": non_hispanic_white_count,
        "non_hispanic_white_share": _percent(
            non_hispanic_white_count,
            total_population,
        ),
        "percent_25_plus_high_school_or_higher": _percent(
            high_school_or_higher,
            education_total,
        ),
    }


def _kentucky_base_year_row(geography, year, record):
    return {
        "Municipality": geography["Municipality"],
        "geography_type": geography["geography_type"],
        "adoption_year": geography["adoption_year"],
        "Year": year,
        "ACS Year": year,
        "ACS Period": f"{year - 4}-{year}",
        "Census Geography": record.get("name", "") if record else "",
        "State FIPS": record.get("state", "") if record else "",
        "Geo FIPS": record.get("geo_fips", "") if record else "",
    }


def _find_acs_record(geography, acs_records):
    return acs_records.get((geography["geography_type"], geography["lookup_name"]))


def _fetch_government_finance_zip(year):
    errors = []
    for url_template in GOVERNMENT_FINANCE_ZIP_URLS:
        url = url_template.format(year=year)
        response = requests.get(url, timeout=60)
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            errors.append(f"{url}: {exc}")
            continue

        return zipfile.ZipFile(BytesIO(response.content))

    raise RuntimeError("; ".join(errors))


def _parse_government_finance_units(finance_zip):
    pid_name = _find_zip_member(finance_zip, "Fin_PID_", ".txt")
    units = {}

    with finance_zip.open(pid_name) as pid_file:
        for raw_line in pid_file:
            line = raw_line.decode("latin1").rstrip("\r\n")
            government_id = line[0:12]
            if government_id[0:2] != KENTUCKY_STATE_FIPS or government_id[2] not in {
                "1",
                "2",
            }:
                continue

            units[government_id] = {
                "government_id": government_id,
                "government_type": "county" if government_id[2] == "1" else "place",
                "government_name": _clean_text(line[12:76]),
                "county": _clean_text(line[76:111]),
                "place_fips": _clean_text(line[111:116]),
                "population": _parse_int(line[116:125]),
                "population_year": _clean_text(line[125:127]),
            }

    return units


def _parse_government_finance_expenditures(finance_zip):
    data_name = _find_zip_member(finance_zip, "FinEstDAT", ".txt")
    expenditures = {}

    with finance_zip.open(data_name) as data_file:
        for raw_line in data_file:
            line = raw_line.decode("latin1").rstrip("\r\n")
            government_id = line[0:12]
            if government_id[0:2] != KENTUCKY_STATE_FIPS or government_id[2] not in {
                "1",
                "2",
            }:
                continue

            code = line[12:15]
            amount = _parse_int(line[15:27])
            if amount == "":
                continue

            amount *= 1000
            if code[0] in DIRECT_EXPENDITURE_PREFIXES:
                record = _expenditure_record(expenditures, government_id)
                record["direct_expenditure"] += amount
                record["total_expenditure"] += amount
            elif code[0] in INTERGOVERNMENTAL_EXPENDITURE_PREFIXES:
                record = _expenditure_record(expenditures, government_id)
                record["intergovernmental_expenditure"] += amount
                record["total_expenditure"] += amount

    return expenditures


def _expenditure_record(expenditures, government_id):
    return expenditures.setdefault(
        government_id,
        {
            "direct_expenditure": 0,
            "intergovernmental_expenditure": 0,
            "total_expenditure": 0,
        },
    )


def _find_zip_member(finance_zip, contains, suffix):
    for name in finance_zip.namelist():
        if contains in name and name.endswith(suffix):
            return name

    raise RuntimeError(f"Could not find {contains} file in finance ZIP.")


def _build_finance_unit_index(finance_units):
    index = {}
    for unit in finance_units.values():
        key = (unit["government_type"], _normalize_finance_name(unit["government_name"]))
        index[key] = unit
    return index


def _find_finance_unit(geography, unit_index):
    return unit_index.get((geography["geography_type"], geography["lookup_name"]))


def _yearly_index(dataframe, value_column):
    output = {}
    for _, row in dataframe.iterrows():
        output[_geo_key(row, include_year=True)] = {
            "value": row[value_column],
            "Census Geography": row["Census Geography"],
            "State FIPS": row["State FIPS"],
            "Geo FIPS": row["Geo FIPS"],
        }
    return output


def _static_index(dataframe):
    output = {}
    for _, row in dataframe.iterrows():
        output[_geo_key(row)] = {
            "Year": row["Year"],
            "Per Capita Spending": row["Per Capita Spending"],
            "Total Municipal Expenditure": row["Total Municipal Expenditure"],
        }
    return output


def _kentucky_lookup(name):
    normalized = _normalize_name(name)
    if normalized in KENTUCKY_PLACE_ALIASES:
        return KENTUCKY_PLACE_ALIASES[normalized]
    if normalized.endswith(" county"):
        return "county", normalized[: -len(" county")]
    return "place", normalized


def _geo_key(row, include_year=False):
    _, lookup_name = _kentucky_lookup(row["Municipality"])
    key = (
        row["geography_type"],
        row["lookup_name"] if "lookup_name" in row else lookup_name,
    )
    if include_year:
        return (*key, int(row["Year"]))
    return key


def _acs_detail_urls(year):
    return [
        f"https://api.census.gov/data/{year}/acs/acs5",
        f"https://api.census.gov/data/{year}/acs5",
    ]


def _normalize_census_geo_name(value, geography_type):
    normalized = _normalize_name(str(value).split(",")[0])
    suffixes = [" county"] if geography_type == "county" else [
        " city",
        " town",
        " cdp",
        " urban county",
    ]
    for suffix in suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def _normalize_finance_name(value):
    normalized = _normalize_name(value)
    for suffix in [
        " county",
        " city",
        " town",
        " urban county government",
        " urban county",
    ]:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def _normalize_name(value):
    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(
        normalized.replace("\ufeff", "")
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
        .lower()
        .split()
    )


if __name__ == "__main__":
    get_kentucky_covariates()

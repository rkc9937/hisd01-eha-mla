from bs4 import BeautifulSoup
from io import BytesIO
import requests
import os
import pandas
import unicodedata
import zipfile


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ENV_PATH = os.path.join(BASE_DIR, ".env")
SEARCH_RATINGS_PATH = os.path.join(DATA_DIR, "searchratings.csv")
POPULATION_OUTPUT_PATH = os.path.join(DATA_DIR, "refinement_1_population.csv")
REPORTS_URL = "http://shsordinances.uh.edu/reports.asp#new-reports/?view_314_per_page=1000&view_314_page={page}"
ACS_START_YEAR = 2009
ACS_END_YEAR = 2023
ACS_STATE_TEXAS = "48"
ACS_EDUCATION_TOTAL_VAR = "B15002_001E"
ACS_EDUCATION_HIGH_SCHOOL_OR_HIGHER_VARS = [
    "B15002_011E",
    "B15002_012E",
    "B15002_013E",
    "B15002_014E",
    "B15002_015E",
    "B15002_016E",
    "B15002_017E",
    "B15002_018E",
    "B15002_028E",
    "B15002_029E",
    "B15002_030E",
    "B15002_031E",
    "B15002_032E",
    "B15002_033E",
    "B15002_034E",
    "B15002_035E",
]
ACS_OUTPUTS = {
    "education": os.path.join(DATA_DIR, "refinement_2_education.csv"),
    "per_capita_income": os.path.join(DATA_DIR, "refinement_2_per_capita_income.csv"),
    "non_hispanic_white": os.path.join(DATA_DIR, "refinement_2_non_hispanic_white.csv"),
}
POPULATION_BY_YEAR_OUTPUT_PATH = os.path.join(
    DATA_DIR,
    "refinement_4_population_by_year.csv",
)
EHA_PANEL_OUTPUT_PATH = os.path.join(DATA_DIR, "refinement_5_eha_panel.csv")
GOVERNMENT_SPENDING_OUTPUT_PATH = os.path.join(
    DATA_DIR,
    "refinement_3_government_spending.csv",
)
GOVERNMENT_SPENDING_DEFAULT_YEAR = 2022
GOVERNMENT_FINANCE_ZIP_URLS = [
    "https://www2.census.gov/programs-surveys/gov-finances/tables/{year}/{year}_Individual_Unit_Files.zip",
    "https://www2.census.gov/programs-surveys/gov-finances/tables/{year}/{year}_Individual_Unit_File.zip",
]
GOVERNMENT_FINANCE_STATE = "48"
GOVERNMENT_FINANCE_CITY_TYPE = "2"
DIRECT_EXPENDITURE_PREFIXES = {"E", "F", "G", "I", "J", "K", "X", "Y"}
INTERGOVERNMENTAL_EXPENDITURE_PREFIXES = {"L", "M", "Q", "S"}


def _load_env_file(path=ENV_PATH):
    if not os.path.exists(path):
        return

    with open(path, encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_env_file()


def get_population():
    population_table = _fetch_population_table()
    search_ratings = pandas.read_csv(SEARCH_RATINGS_PATH, encoding="utf-8-sig")

    by_city_county, by_city = _build_population_indexes(population_table)
    output_rows = []

    for _, rating in search_ratings.iterrows():
        municipality = _clean_text(rating["Municipality"])
        source_county = _clean_text(rating["County"])
        population_record = _find_population_record(
            municipality,
            source_county,
            by_city_county,
            by_city,
        )

        output_rows.append(
            {
                "Municipality": municipality,
                "Population": population_record.get("Population", "")
                if population_record
                else "",
                "Passage Date": _clean_text(rating["Passage Date"]),
                "County": population_record.get("County", source_county)
                if population_record
                else source_county,
            }
        )

    output = pandas.DataFrame(
        output_rows,
        columns=["Municipality", "Population", "Passage Date", "County"],
    )
    output.to_csv(POPULATION_OUTPUT_PATH, index=False)
    return output


def get_acs_controls(
    start_year=ACS_START_YEAR,
    end_year=ACS_END_YEAR,
    census_api_key=None,
):
    municipalities = _load_population_refinement(start_year)
    acs_by_year = _fetch_available_acs_years(start_year, end_year, census_api_key)
    available_years = sorted(acs_by_year)
    if not available_years:
        raise RuntimeError(
            f"No ACS controls could be fetched for {start_year}-{end_year}."
        )

    education_rows = []
    income_rows = []
    white_rows = []

    for year in available_years:
        place_controls = acs_by_year[year]

        for _, municipality in municipalities.iterrows():
            source_name = _clean_text(municipality["Municipality"])
            control = place_controls.get(_normalize_place_name(source_name), {})
            base_row = {
                "Municipality": source_name,
                "County": _clean_text(municipality["County"]),
                "Passage Date": _clean_text(municipality["Passage Date"]),
                "Year": year,
                "ACS Year": year,
                "ACS Period": f"{year - 4}-{year}",
                "Census Place": control.get("name", ""),
                "State FIPS": control.get("state", ""),
                "Place FIPS": control.get("place", ""),
            }

            education_rows.append(
                {
                    **base_row,
                    "percent_25_plus_high_school_or_higher": control.get(
                        "percent_25_plus_high_school_or_higher",
                        "",
                    ),
                }
            )
            income_rows.append(
                {
                    **base_row,
                    "per_capita_income": control.get("per_capita_income", ""),
                }
            )
            white_rows.append(
                {
                    **base_row,
                    "non_hispanic_white_share": control.get(
                        "non_hispanic_white_share",
                        "",
                    ),
                    "non_hispanic_white_count": control.get(
                        "non_hispanic_white_count",
                        "",
                    ),
                    "total_population_for_race": control.get(
                        "total_population_for_race",
                        "",
                    ),
                }
            )

    outputs = {
        "education": pandas.DataFrame(education_rows),
        "per_capita_income": pandas.DataFrame(income_rows),
        "non_hispanic_white": pandas.DataFrame(white_rows),
    }
    for key, output in outputs.items():
        output.to_csv(ACS_OUTPUTS[key], index=False)

    return outputs


def get_government_spending(year=GOVERNMENT_SPENDING_DEFAULT_YEAR):
    municipalities = _load_population_refinement_without_year_filter()
    finance_zip = _fetch_government_finance_zip(year)
    finance_units = _parse_government_finance_units(finance_zip)
    expenditures = _parse_government_finance_expenditures(finance_zip)
    finance_index = _build_government_finance_index(finance_units)
    output_rows = []

    for _, municipality in municipalities.iterrows():
        name = _clean_text(municipality["Municipality"])
        county = _clean_text(municipality["County"])
        unit = _find_government_finance_unit(name, county, finance_index)
        spending = expenditures.get(unit["government_id"], {}) if unit else {}
        total_expenditure = spending.get("total_expenditure", "")
        finance_population = unit.get("population", "") if unit else ""

        output_rows.append(
            {
                "Municipality": name,
                "County": county,
                "Passage Date": _clean_text(municipality["Passage Date"]),
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
    output.to_csv(GOVERNMENT_SPENDING_OUTPUT_PATH, index=False)
    return output


def get_eha_panel(start_year=ACS_START_YEAR, end_year=ACS_END_YEAR):
    municipalities = _load_unique_population_refinement_without_year_filter()
    municipalities = _prepare_adoption_years(municipalities)
    population_by_year = _read_yearly_population()
    education = _read_yearly_covariate(
        ACS_OUTPUTS["education"],
        "percent_25_plus_high_school_or_higher",
    )
    income = _read_yearly_covariate(
        ACS_OUTPUTS["per_capita_income"],
        "per_capita_income",
    )
    white = _read_yearly_covariate(
        ACS_OUTPUTS["non_hispanic_white"],
        "non_hispanic_white_share",
    )
    government_spending = _read_static_government_spending()
    smoker = _read_texas_smoker_percentages()
    state_population = _read_state_population()
    restriction_share = _build_restriction_share_by_year(
        municipalities,
        population_by_year,
        state_population,
        start_year,
        end_year,
    )
    output_rows = []

    for _, municipality in municipalities.iterrows():
        adoption_year = municipality["adoption_year"]
        if pandas.isna(adoption_year) or adoption_year < start_year:
            continue

        last_risk_year = min(int(adoption_year), end_year)
        if last_risk_year < start_year:
            continue

        key = _panel_key(municipality)
        spending = government_spending.get(key, {})

        for year in range(start_year, last_risk_year + 1):
            row_key = (key[0], key[1], year)
            population = population_by_year.get(row_key, {})
            output_rows.append(
                {
                    "Municipality": _clean_text(municipality["Municipality"]),
                    "County": _clean_text(municipality["County"]),
                    "Passage Date": _clean_text(municipality["Passage Date"]),
                    "Adoption Year": int(adoption_year),
                    "Year": year,
                    "time_index": year - start_year + 1,
                    "adopted_this_year": int(year == adoption_year),
                    "ordinance_profile_population": _parse_population_string(
                        municipality["Population"]
                    ),
                    "municipal_population": population.get("Population", ""),
                    "percent_25_plus_high_school_or_higher": education.get(
                        row_key,
                        "",
                    ),
                    "non_hispanic_white_share": white.get(row_key, ""),
                    "per_capita_income": income.get(row_key, ""),
                    "per_capita_spending": spending.get("Per Capita Spending", ""),
                    "total_municipal_expenditure": spending.get(
                        "Total Municipal Expenditure",
                        "",
                    ),
                    "government_spending_year": spending.get("Year", ""),
                    "texas_smoker_percentage": smoker.get(year, ""),
                    "texas_population": state_population.get(year, ""),
                    "proportion_state_population_local_restriction": restriction_share[
                        year
                    ]["proportion_state_population_local_restriction"],
                    "proportion_state_population_local_restriction_prior": restriction_share[
                        year
                    ]["proportion_state_population_local_restriction_prior"],
                    "restricted_municipal_population": restriction_share[year][
                        "restricted_municipal_population"
                    ],
                    "restricted_municipal_population_prior": restriction_share[year][
                        "restricted_municipal_population_prior"
                    ],
                    "Census Place": population.get("Census Place", ""),
                    "State FIPS": population.get("State FIPS", ""),
                    "Place FIPS": population.get("Place FIPS", ""),
                }
            )

    output = pandas.DataFrame(output_rows)
    output.to_csv(EHA_PANEL_OUTPUT_PATH, index=False)
    return output


def get_population_by_year(
    start_year=ACS_START_YEAR,
    end_year=ACS_END_YEAR,
    census_api_key=None,
):
    municipalities = _load_unique_population_refinement_without_year_filter()
    population_by_year = _fetch_available_acs_population_years(
        start_year,
        end_year,
        census_api_key,
    )
    available_years = sorted(population_by_year)
    if not available_years:
        raise RuntimeError(
            f"No ACS population data could be fetched for {start_year}-{end_year}."
        )

    output_rows = []
    for year in available_years:
        place_populations = population_by_year[year]

        for _, municipality in municipalities.iterrows():
            name = _clean_text(municipality["Municipality"])
            control = place_populations.get(_normalize_place_name(name), {})
            output_rows.append(
                {
                    "Municipality": name,
                    "County": _clean_text(municipality["County"]),
                    "Passage Date": _clean_text(municipality["Passage Date"]),
                    "Year": year,
                    "ACS Year": year,
                    "ACS Period": f"{year - 4}-{year}",
                    "Census Place": control.get("name", ""),
                    "State FIPS": control.get("state", ""),
                    "Place FIPS": control.get("place", ""),
                    "Population": control.get("population", ""),
                    "Source": "Census ACS 5-year B01003_001E",
                }
            )

    output = pandas.DataFrame(output_rows)
    output.to_csv(POPULATION_BY_YEAR_OUTPUT_PATH, index=False)
    return output


def _prepare_adoption_years(municipalities):
    municipalities = municipalities.copy()
    municipalities["_passage_date"] = pandas.to_datetime(
        municipalities["Passage Date"],
        errors="coerce",
    )
    municipalities["adoption_year"] = municipalities["_passage_date"].dt.year
    return municipalities.drop(columns=["_passage_date"])


def _read_yearly_population():
    population = pandas.read_csv(
        POPULATION_BY_YEAR_OUTPUT_PATH,
        dtype={"State FIPS": str, "Place FIPS": str},
    )
    population = population.drop_duplicates(
        subset=["Municipality", "County", "Year"],
        keep="first",
    )
    output = {}

    for _, row in population.iterrows():
        output[_panel_key(row, include_year=True)] = {
            "Population": row["Population"],
            "Census Place": row["Census Place"],
            "State FIPS": row["State FIPS"],
            "Place FIPS": row["Place FIPS"],
        }

    return output


def _read_yearly_covariate(path, value_column):
    covariate = pandas.read_csv(path)
    covariate = covariate.drop_duplicates(
        subset=["Municipality", "County", "Year"],
        keep="first",
    )
    output = {}

    for _, row in covariate.iterrows():
        output[_panel_key(row, include_year=True)] = row[value_column]

    return output


def _read_static_government_spending():
    spending = pandas.read_csv(GOVERNMENT_SPENDING_OUTPUT_PATH)
    spending = spending.drop_duplicates(
        subset=["Municipality", "County"],
        keep="first",
    )
    output = {}

    for _, row in spending.iterrows():
        output[_panel_key(row)] = {
            "Year": row["Year"],
            "Per Capita Spending": row["Per Capita Spending"],
            "Total Municipal Expenditure": row["Total Municipal Expenditure"],
        }

    return output


def _read_texas_smoker_percentages():
    smoker = pandas.read_csv(os.path.join(DATA_DIR, "texas_smoker_percentage_year.csv"))
    return {
        int(row["Year"]): row["Percentage"]
        for _, row in smoker.iterrows()
        if not pandas.isna(row["Year"])
    }


def _read_state_population():
    state_population = pandas.read_csv(os.path.join(DATA_DIR, "state_pop.csv"))
    return {
        int(row["Year"]): row["Texas_Population"]
        for _, row in state_population.iterrows()
        if not pandas.isna(row["Year"])
    }


def _build_restriction_share_by_year(
    municipalities,
    population_by_year,
    state_population,
    start_year,
    end_year,
):
    restriction_share = {}

    for year in range(start_year, end_year + 1):
        restricted_population = 0
        restricted_population_prior = 0

        for _, municipality in municipalities.iterrows():
            adoption_year = municipality["adoption_year"]
            if pandas.isna(adoption_year) or adoption_year > year:
                continue

            key = (*_panel_key(municipality), year)
            population = population_by_year.get(key, {}).get("Population", "")
            if population != "":
                restricted_population += population
                if adoption_year < year:
                    restricted_population_prior += population

        texas_population = state_population.get(year, "")
        restriction_share[year] = {
            "restricted_municipal_population": restricted_population,
            "restricted_municipal_population_prior": restricted_population_prior,
            "proportion_state_population_local_restriction": _divide(
                restricted_population,
                texas_population,
                6,
            ),
            "proportion_state_population_local_restriction_prior": _divide(
                restricted_population_prior,
                texas_population,
                6,
            ),
        }

    return restriction_share


def _panel_key(row, include_year=False):
    key = (
        _normalize_place_name(row["Municipality"]),
        _normalize_name(row["County"]),
    )
    if include_year:
        return (*key, int(row["Year"]))

    return key


def _parse_population_string(value):
    if pandas.isna(value):
        return ""

    value = str(value).replace(",", "").strip()
    if not value:
        return ""

    return _parse_int(value)


def _load_population_refinement_without_year_filter():
    population = pandas.read_csv(POPULATION_OUTPUT_PATH)
    return population[
        population["Population"].notna()
        & (population["Population"].astype(str).str.strip() != "")
    ].copy()


def _load_unique_population_refinement_without_year_filter():
    population = _load_population_refinement_without_year_filter()
    population["_passage_date"] = pandas.to_datetime(
        population["Passage Date"],
        errors="coerce",
    )
    population = population.sort_values(
        ["Municipality", "County", "_passage_date"],
        na_position="last",
    )
    return population.drop_duplicates(
        subset=["Municipality", "County"],
        keep="first",
    ).drop(columns=["_passage_date"])


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
            if (
                government_id[0:2] != GOVERNMENT_FINANCE_STATE
                or government_id[2] != GOVERNMENT_FINANCE_CITY_TYPE
            ):
                continue

            units[government_id] = {
                "government_id": government_id,
                "government_name": _clean_text(line[12:76]),
                "county": _clean_text(line[76:111]),
                "place_fips": _clean_text(line[111:116]),
                "population": _parse_int(line[116:125]),
                "population_year": _clean_text(line[125:127]),
                "fiscal_year_ending": _clean_text(line[140:144]),
                "survey_year": _clean_text(line[144:146]),
            }

    return units


def _parse_government_finance_expenditures(finance_zip):
    data_name = _find_zip_member(finance_zip, "FinEstDAT", ".txt")
    expenditures = {}

    with finance_zip.open(data_name) as data_file:
        for raw_line in data_file:
            line = raw_line.decode("latin1").rstrip("\r\n")
            government_id = line[0:12]
            if (
                government_id[0:2] != GOVERNMENT_FINANCE_STATE
                or government_id[2] != GOVERNMENT_FINANCE_CITY_TYPE
            ):
                continue

            code = line[12:15]
            amount = _parse_int(line[15:27])
            if amount == "":
                continue

            amount *= 1000
            if code[0] in DIRECT_EXPENDITURE_PREFIXES:
                record = _get_expenditure_record(expenditures, government_id)
                record["direct_expenditure"] += amount
                record["total_expenditure"] += amount
            elif code[0] in INTERGOVERNMENTAL_EXPENDITURE_PREFIXES:
                record = _get_expenditure_record(expenditures, government_id)
                record["intergovernmental_expenditure"] += amount
                record["total_expenditure"] += amount

    return expenditures


def _get_expenditure_record(expenditures, government_id):
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


def _build_government_finance_index(finance_units):
    by_city_county = {}
    by_city_matches = {}

    for unit in finance_units.values():
        city_key = _normalize_finance_unit_name(unit["government_name"])
        by_city_matches.setdefault(city_key, []).append(unit)

        for county_key in _county_keys(unit["county"]):
            by_city_county[(city_key, county_key)] = unit

    by_city = {
        city_key: matches[0]
        for city_key, matches in by_city_matches.items()
        if len(matches) == 1
    }
    return {"by_city_county": by_city_county, "by_city": by_city}


def _find_government_finance_unit(municipality, county, finance_index):
    city_key = _normalize_municipality_name(municipality)

    for county_key in _county_keys(county):
        unit = finance_index["by_city_county"].get((city_key, county_key))
        if unit:
            return unit

    return finance_index["by_city"].get(city_key)


def _fetch_available_acs_population_years(start_year, end_year, census_api_key=None):
    population_by_year = {}
    failures = {}

    for year in range(start_year, end_year + 1):
        try:
            population_by_year[year] = _fetch_acs_place_populations(
                year,
                census_api_key,
            )
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            failures[year] = str(exc)
            if population_by_year:
                raise RuntimeError(
                    f"ACS population worked through {year - 1}, but failed for "
                    f"{year}: {exc}"
                ) from exc

    if failures and population_by_year:
        first_year = min(population_by_year)
        skipped = ", ".join(str(year) for year in failures if year < first_year)
        print(f"Skipped unavailable ACS population years before {first_year}: {skipped}")

    return population_by_year


def _fetch_acs_place_populations(year, census_api_key=None):
    rows = _fetch_census_json(
        _acs_detail_urls(year),
        {
            "get": "NAME,B01003_001E",
            "for": "place:*",
            "in": f"state:{ACS_STATE_TEXAS}",
        },
        census_api_key,
    )
    populations = {}

    for row in rows:
        key = _normalize_place_name(row["NAME"])
        populations[key] = {
            "name": row["NAME"],
            "state": row["state"],
            "place": row["place"],
            "population": _parse_number(row["B01003_001E"]),
        }

    return populations


def _fetch_available_acs_years(start_year, end_year, census_api_key=None):
    acs_by_year = {}
    failures = {}

    for year in range(start_year, end_year + 1):
        try:
            acs_by_year[year] = _fetch_acs_place_controls(year, census_api_key)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            failures[year] = str(exc)
            if acs_by_year:
                raise RuntimeError(
                    f"ACS controls worked through {year - 1}, but failed for {year}: "
                    f"{exc}"
                ) from exc

    if failures and acs_by_year:
        first_year = min(acs_by_year)
        skipped = ", ".join(str(year) for year in failures if year < first_year)
        print(f"Skipped unavailable ACS years before {first_year}: {skipped}")

    return acs_by_year


def _load_population_refinement(start_year):
    population = pandas.read_csv(POPULATION_OUTPUT_PATH)
    passage_dates = pandas.to_datetime(population["Passage Date"], errors="coerce")
    return population[
        population["Population"].notna()
        & (population["Population"].astype(str).str.strip() != "")
        & (passage_dates.dt.year >= start_year)
    ].copy()


def _fetch_acs_place_controls(year, census_api_key=None):
    geography = {"for": "place:*", "in": f"state:{ACS_STATE_TEXAS}"}

    detail_get_vars = [
        "NAME",
        "B19301_001E",
        "B03002_001E",
        "B03002_003E",
        ACS_EDUCATION_TOTAL_VAR,
        *ACS_EDUCATION_HIGH_SCHOOL_OR_HIGHER_VARS,
    ]
    detail_rows = _fetch_census_json(
        _acs_detail_urls(year),
        {
            "get": ",".join(detail_get_vars),
            **geography,
        },
        census_api_key,
    )

    controls = {}
    for row in detail_rows:
        key = _normalize_place_name(row["NAME"])
        record = controls.setdefault(key, _base_acs_record(row))
        total_population = _parse_number(row["B03002_001E"])
        non_hispanic_white_count = _parse_number(row["B03002_003E"])
        education_total = _parse_number(row[ACS_EDUCATION_TOTAL_VAR])
        high_school_or_higher = _sum_numbers(
            row[var] for var in ACS_EDUCATION_HIGH_SCHOOL_OR_HIGHER_VARS
        )

        record["per_capita_income"] = _parse_number(row["B19301_001E"])
        record["total_population_for_race"] = total_population
        record["non_hispanic_white_count"] = non_hispanic_white_count
        record["non_hispanic_white_share"] = _percent(
            non_hispanic_white_count,
            total_population,
        )
        record["percent_25_plus_high_school_or_higher"] = _percent(
            high_school_or_higher,
            education_total,
        )

    return controls


def _acs_detail_urls(year):
    return [
        f"https://api.census.gov/data/{year}/acs/acs5",
        f"https://api.census.gov/data/{year}/acs5",
    ]


def _fetch_census_json(url, params, census_api_key=None):
    api_key = census_api_key or os.environ.get("CENSUS_API_KEY")
    if api_key:
        params = {**params, "key": api_key}

    errors = []
    for candidate_url in url:
        response = requests.get(candidate_url, params=params, timeout=60)
        if "missing_key.html" in response.url:
            raise RuntimeError(
                "The Census API requires a key for this request. Set CENSUS_API_KEY "
                "in your environment and rerun get_acs_controls()."
            )

        try:
            response.raise_for_status()
            rows = response.json()
        except (requests.RequestException, ValueError) as exc:
            error = _sanitize_secret(str(exc), api_key)
            message = _sanitize_secret(response.text[:300], api_key)
            errors.append(f"{candidate_url}: {error}; {message}")
            continue

        header = rows[0]
        return [dict(zip(header, row)) for row in rows[1:]]

    raise RuntimeError("; ".join(errors))


def _sanitize_secret(value, secret):
    if not secret:
        return value

    return value.replace(secret, "[KEY]")


def _base_acs_record(row):
    return {
        "name": row["NAME"],
        "state": row["state"],
        "place": row["place"],
    }


def _parse_number(value):
    if value in (None, "", "null"):
        return ""

    try:
        number = float(value)
    except ValueError:
        return ""

    if number < 0:
        return ""

    if number.is_integer():
        return int(number)

    return number


def _parse_int(value):
    value = str(value).strip()
    if not value:
        return ""

    try:
        return int(value)
    except ValueError:
        return ""


def _sum_numbers(values):
    total = 0
    found_value = False
    for value in values:
        number = _parse_number(value)
        if number == "":
            continue

        found_value = True
        total += number

    return total if found_value else ""


def _divide(numerator, denominator, ndigits=2):
    if numerator == "" or denominator in ("", 0):
        return ""

    return round(numerator / denominator, ndigits)


def _percent(numerator, denominator):
    if numerator == "" or denominator in ("", 0):
        return ""

    return round((numerator / denominator) * 100, 2)


def _fetch_population_table():
    rows = _fetch_population_table_with_selenium()
    if rows:
        return rows

    rows = []
    for page in range(1, 20):
        response = requests.get(REPORTS_URL.format(page=page), timeout=30)
        response.raise_for_status()

        page_rows = _parse_population_rows(response.text)
        if not page_rows:
            break

        rows.extend(page_rows)
        if len(page_rows) < 1000:
            break

    if not rows:
        raise RuntimeError(
            "Could not find population rows. The reports page likely requires "
            "JavaScript rendering; install Selenium and a browser driver."
        )

    return _dedupe_population_rows(rows)


def _fetch_population_table_with_selenium():
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait
    except ImportError:
        return []

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=options)
    try:
        rows = []
        for page in range(1, 20):
            driver.get(REPORTS_URL.format(page=page))
            try:
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "td.field_29"))
                )
            except Exception:
                break

            page_rows = _parse_population_rows(driver.page_source)
            if not page_rows:
                break

            rows.extend(page_rows)
            if len(page_rows) < 1000:
                break

        return _dedupe_population_rows(rows)
    finally:
        driver.quit()


def _parse_population_rows(html):
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    for table_row in soup.select("tr"):
        municipality_cell = table_row.select_one("td.field_29")
        county_cell = table_row.select_one("td.field_31")
        phr_cell = table_row.select_one("td.field_32")
        population_cell = table_row.select_one("td.field_2902")

        if not municipality_cell or not county_cell or not population_cell:
            continue

        rows.append(
            {
                "Municipality": _clean_text(municipality_cell.get_text(" ", strip=True)),
                "County": _clean_text(county_cell.get_text(" ", strip=True)),
                "PHR": _clean_text(phr_cell.get_text(" ", strip=True))
                if phr_cell
                else "",
                "Population": _clean_text(population_cell.get_text(" ", strip=True)),
            }
        )

    return rows


def _dedupe_population_rows(rows):
    seen = set()
    deduped = []

    for row in rows:
        key = (
            _normalize_name(row["Municipality"]),
            _normalize_name(row["County"]),
            row["Population"],
        )
        if key in seen:
            continue

        seen.add(key)
        deduped.append(row)

    return deduped


def _build_population_indexes(population_rows):
    by_city_county = {}
    by_city_matches = {}

    for row in population_rows:
        city_key = _normalize_name(row["Municipality"])
        county_keys = _county_keys(row["County"])

        by_city_matches.setdefault(city_key, []).append(row)
        for county_key in county_keys:
            key = (city_key, county_key)
            by_city_county[key] = _prefer_populated_record(
                by_city_county.get(key),
                row,
            )

    by_city = {
        city_key: matches[0]
        for city_key, matches in by_city_matches.items()
        if len(matches) == 1
    }
    return by_city_county, by_city


def _prefer_populated_record(current, candidate):
    if current is None:
        return candidate

    if not current.get("Population") and candidate.get("Population"):
        return candidate

    return current


def _find_population_record(municipality, county, by_city_county, by_city):
    city_key = _normalize_name(municipality)

    for county_key in _county_keys(county):
        record = by_city_county.get((city_key, county_key))
        if record:
            return record

    return by_city.get(city_key)


def _county_keys(county):
    normalized = _normalize_name(county)
    if not normalized:
        return [""]

    parts = [
        _normalize_name(part)
        for chunk in county.replace("&", "-").replace("/", "-").split("-")
        for part in chunk.split(",")
    ]
    return [normalized] + [part for part in parts if part and part != normalized]


def _normalize_name(value):
    return " ".join(
        str(value)
        .replace("\ufeff", "")
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
        .lower()
        .split()
    )


def _normalize_place_name(value):
    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    is_census_place_label = "," in normalized
    normalized = normalized.split(",")[0]
    normalized = _normalize_name(normalized)

    if is_census_place_label:
        for suffix in (" city", " town", " village", " cdp"):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break

    return normalized


def _normalize_municipality_name(value):
    return _normalize_place_name(value)


def _normalize_finance_unit_name(value):
    normalized = _normalize_place_name(value)
    for suffix in (
        " city",
        " town",
        " village",
        " municipality",
        " municipal government",
    ):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break

    return normalized


def _clean_text(value):
    if pandas.isna(value):
        return ""

    return " ".join(str(value).replace("\ufeff", "").split())

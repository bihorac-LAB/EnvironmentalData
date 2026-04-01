import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import usaddress
from loguru import logger

try:
    from geopy.geocoders import Nominatim
except Exception:  # pragma: no cover
    Nominatim = None


LOCATION_REQUIRED_COLUMNS = [
    "location_id",
    "address_1",
    "address_2",
    "city",
    "state",
    "zip",
    "county",
    "location_source_value",
    "country_concept_id",
    "country_source_value",
    "latitude",
    "longitude",
]

LOCATION_HISTORY_REQUIRED_COLUMNS = [
    "location_id",
    "relationship_type_concept_id",
    "domain_id",
    "entity_id",
    "start_date",
    "end_date",
]

DEFAULT_RELATIONSHIP_TYPE_CONCEPT_ID = 32848
DEFAULT_DOMAIN_ID = 1147314
DEFAULT_COUNTRY_SOURCE_VALUE = "UNITED STATES OF AMERICA"
DEFAULT_COUNTRY_QUERY_TOKEN = "USA"


STREET_SUFFIX_ABBREVIATIONS = {
    r"\bstreet\b": "St",
    r"\bavenue\b": "Ave",
    r"\broad\b": "Rd",
    r"\bdrive\b": "Dr",
    r"\bboulevard\b": "Blvd",
    r"\blane\b": "Ln",
    r"\bcourt\b": "Ct",
    r"\bcircle\b": "Cir",
    r"\bplace\b": "Pl",
    r"\bterrace\b": "Ter",
    r"\btrail\b": "Trl",
    r"\bparkway\b": "Pkwy",
    r"\bhighway\b": "Hwy",
    r"\bsquare\b": "Sq",
    r"\bexpressway\b": "Expy",
}

UNIT_ABBREVIATIONS = {
    r"\bapartment\b": "Apt",
    r"\bsuite\b": "Ste",
    r"\bunit\b": "Unit",
    r"\bfloor\b": "Fl",
    r"\bbuilding\b": "Bldg",
    r"\broom\b": "Rm",
    r"\bdepartment\b": "Dept",
    r"\blot\b": "Lot",
}

CARDINAL_ABBREVIATIONS = {
    r"\bnorth\b": "N",
    r"\bsouth\b": "S",
    r"\beast\b": "E",
    r"\bwest\b": "W",
    r"\bnortheast\b": "NE",
    r"\bnorthwest\b": "NW",
    r"\bsoutheast\b": "SE",
    r"\bsouthwest\b": "SW",
}

COMMON_ADDRESS_TYPO_CORRECTIONS = {
    r"\bavnue\b": "avenue",
    r"\bavn\b": "avenue",
    r"\bavee\b": "avenue",
    r"\bstreett\b": "street",
    r"\bstreer\b": "street",
    r"\broaad\b": "road",
    r"\brd\.\b": "rd",
    r"\bdr\.\b": "dr",
    r"\bblvd\.\b": "blvd",
    r"\bappartment\b": "apartment",
    r"\bapartmet\b": "apartment",
    r"\bsutie\b": "suite",
    r"\bsuit\b": "suite",
    r"\bnorthh\b": "north",
    r"\bsouthh\b": "south",
    r"\beastt\b": "east",
    r"\bwestt\b": "west",
    r"\bnwe\b": "nw",
    r"\bswe\b": "sw",
    r"\bnee\b": "ne",
    r"\bsee\b": "se",
}


PLACEHOLDER_LOCATION_TOKENS = {
    "not stated",
    "unknown",
    "na",
    "n/a",
    "none",
    "null",
    "missing",
    "not available",
}


def ensure_output_folder(output_folder: str) -> str:
    output_folder = os.path.abspath(output_folder) if output_folder else os.path.join(os.getcwd(), "output")
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def configure_logging(output_folder: str) -> str:
    output_folder = ensure_output_folder(output_folder)
    log_folder = os.path.join(output_folder, "log")
    os.makedirs(log_folder, exist_ok=True)

    log_filename = f"address_to_location_{datetime.now().strftime('%Y%m%d_%H%M%S')}" \
        f".log"
    log_file_path = os.path.join(log_folder, log_filename)

    logger.remove()
    logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
    logger.add(log_file_path, format="{time} {level} {message}", level="DEBUG")
    logger.info(f"Log file configured: {log_file_path}")
    return log_file_path


def create_analysis_preprocessed_folder(output_folder: str) -> str:
    run_token = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(output_folder, "preprocessed_analysis", run_token)
    os.makedirs(folder, exist_ok=True)
    return folder


def cleanup_analysis_preprocessed_folder(folder: str):
    keep_preprocessed = os.getenv("KEEP_PREPROCESSED", "0") == "1"
    if keep_preprocessed:
        logger.info(f"Retaining preprocessed analysis files (KEEP_PREPROCESSED=1): {folder}")
        return

    if folder and os.path.isdir(folder):
        shutil.rmtree(folder, ignore_errors=True)
        logger.info(f"Deleted preprocessed analysis folder: {folder}")


def is_valid_coordinate(lat: object, lon: object) -> bool:
    try:
        lat_val = float(lat)
        lon_val = float(lon)
    except (TypeError, ValueError):
        return False

    if pd.isna(lat_val) or pd.isna(lon_val):
        return False
    return -90.0 <= lat_val <= 90.0 and -180.0 <= lon_val <= 180.0


def normalize_spaces(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def geopy_style_cleanup(value: str) -> str:
    cleaned = re.sub(r"[^\w\s#-]", " ", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_address_text(value: object) -> str:
    text = normalize_spaces(value)
    if not text:
        return ""
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"[^\w\s#&/,-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def smart_title_case(value: object) -> str:
    text = clean_address_text(value)
    if not text:
        return ""

    tokens = []
    for token in text.split():
        raw = token.strip()
        if not raw:
            continue
        if re.fullmatch(r"\d+(st|nd|rd|th)", raw, flags=re.IGNORECASE):
            tokens.append(raw.lower())
            continue
        if raw.isdigit():
            tokens.append(raw)
            continue

        upper_raw = raw.upper()
        if upper_raw in {"N", "S", "E", "W", "NE", "NW", "SE", "SW"}:
            tokens.append(upper_raw)
            continue
        if upper_raw in {"PO", "P.O.", "P O"}:
            tokens.append("PO")
            continue

        tokens.append(raw.lower().capitalize())

    return " ".join(tokens)


def standardize_address_terms(value: object, replacements: Dict[str, str]) -> str:
    text = smart_title_case(value)
    if not text:
        return ""

    result = text.lower()
    for pattern, replacement in COMMON_ADDRESS_TYPO_CORRECTIONS.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    for pattern, replacement in replacements.items():
        result = re.sub(pattern, replacement.lower(), result, flags=re.IGNORECASE)

    result = smart_title_case(result)
    result = re.sub(r"\b(\d+)(St|Nd|Rd|Th)\b", lambda match: f"{match.group(1)}{match.group(2).lower()}", result)
    return result


def normalize_city_name(value: object) -> str:
    text = smart_title_case(value)
    if not text:
        return ""

    corrected = text.lower()
    for pattern, replacement in COMMON_ADDRESS_TYPO_CORRECTIONS.items():
        corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)

    corrected = smart_title_case(corrected)
    corrected = re.sub(r"\s+", " ", corrected).strip()
    return corrected


def format_clean_address(street: str, unit: str, city: str, state: str, zip5: str) -> str:
    components = [component for component in [street, unit] if component]
    primary = " ".join(components).strip()
    trailing = ", ".join(component for component in [city, state] if component)
    if zip5:
        trailing = f"{trailing} {zip5}".strip() if trailing else zip5

    parts = [part for part in [primary, trailing] if part]
    return ", ".join(parts)


def format_modifier_source_value(geocode_level: object, geocode_reason: object = "") -> str:
    level = normalize_spaces(geocode_level).lower()
    reason = normalize_spaces(geocode_reason)

    level_map = {
        "provided": ("Level 1", "lat/long provided"),
        "address": ("Level 2", "lat/long generated from address"),
        "zip9": ("Level 3", "lat/long generated from zip9"),
        "zip5": ("Level 4", "lat/long generated from zip5"),
    }

    if level in level_map:
        return " | ".join(level_map[level])

    if level == "failed":
        parts = ["failed", "geocoding failed"]
        if reason:
            parts.append(reason)
        return " | ".join(parts)

    if not level:
        return ""

    parts = [level]
    if reason:
        parts.append(reason)
    return " | ".join(parts)


def normalize_zip(value: object) -> Tuple[str, str]:
    raw = normalize_spaces(value).replace(".0", "")
    digits = re.sub(r"\D", "", raw)
    return digits[:5] if len(digits) >= 5 else "", digits[:9] if len(digits) >= 9 else ""


def format_zip9_hyphen(zip9: object) -> str:
    zip9_clean = re.sub(r"\D", "", str(zip9 or ""))
    if len(zip9_clean) != 9:
        return ""
    return f"{zip9_clean[:5]}-{zip9_clean[5:]}"


def normalize_state(value: object) -> str:
    state = normalize_spaces(value).upper()
    return state[:2] if len(state) >= 2 else state


def normalize_geocoder_query(value: object) -> str:
    text = normalize_spaces(value)
    if not text:
        return ""
    text = geopy_style_cleanup(text)
    return re.sub(r"\s+", " ", text).strip()


def get_stage_threshold(level_name: str) -> float:
    env_map = {
        "address": "GEOCODER_THRESHOLD_ADDRESS",
        "zip9": "GEOCODER_THRESHOLD_ZIP9",
        "zip5": "GEOCODER_THRESHOLD_ZIP5",
    }
    default_map = {"address": 0.7, "zip9": 0.3, "zip5": 0.1}

    env_name = env_map.get(level_name, "")
    default_value = default_map.get(level_name, 0.7)
    raw_value = os.getenv(env_name, str(default_value)).strip() if env_name else str(default_value)

    try:
        parsed = float(raw_value)
    except (TypeError, ValueError):
        logger.warning(f"Invalid threshold value for {env_name}: {raw_value}. Using default {default_value}.")
        parsed = default_value

    if parsed < 0:
        return 0.0
    if parsed > 1:
        return 1.0
    return parsed


def build_row_address_fields(row: pd.Series) -> Dict[str, str]:
    row_dict = row.to_dict()

    raw_address = normalize_spaces(row_dict.get("address", ""))
    address_1 = normalize_spaces(row_dict.get("address_1", ""))
    address_2 = normalize_spaces(row_dict.get("address_2", ""))
    street = normalize_spaces(row_dict.get("street", ""))
    city = normalize_spaces(row_dict.get("city", ""))
    state = normalize_state(row_dict.get("state", ""))
    zip_raw = row_dict.get("zip", "")

    base_full = raw_address or " ".join([x for x in [address_1, address_2, street, city, state, normalize_spaces(zip_raw)] if x])
    base_full = geopy_style_cleanup(base_full)

    parsed = {}
    if base_full:
        try:
            parsed, _ = usaddress.tag(base_full)
        except Exception:
            parsed = {}

    parsed_street = " ".join(
        part
        for part in [
            parsed.get("AddressNumber", ""),
            parsed.get("StreetNamePreDirectional", ""),
            parsed.get("StreetName", ""),
            parsed.get("StreetNamePostType", ""),
            parsed.get("StreetNamePostDirectional", ""),
        ]
        if part
    ).strip()

    parsed_unit = " ".join(part for part in [parsed.get("OccupancyType", ""), parsed.get("OccupancyIdentifier", "")] if part).strip()

    street_terms = dict(CARDINAL_ABBREVIATIONS)
    street_terms.update(STREET_SUFFIX_ABBREVIATIONS)
    normalized_street = standardize_address_terms(parsed_street or address_1 or street, street_terms)
    normalized_unit = standardize_address_terms(parsed_unit or address_2, UNIT_ABBREVIATIONS)
    normalized_city = normalize_city_name(normalize_spaces(parsed.get("PlaceName", "")) or city)
    normalized_state = normalize_state(parsed.get("StateName", "")) or state

    parsed_zip = normalize_spaces(parsed.get("ZipCode", "")) or normalize_spaces(zip_raw)
    zip5, zip9 = normalize_zip(parsed_zip)

    normalized_street = geopy_style_cleanup(normalized_street)
    normalized_unit = geopy_style_cleanup(normalized_unit)
    normalized_city = geopy_style_cleanup(normalized_city)

    return {
        "normalized_street": normalized_street,
        "normalized_unit": normalized_unit,
        "normalized_city": normalized_city,
        "normalized_state": normalized_state,
        "zip5": zip5,
        "zip9": zip9,
        "normalized_full_address": format_clean_address(normalized_street, normalized_unit, normalized_city, normalized_state, zip5),
    }


def geopy_parse_address_optional(address: str) -> Dict[str, str]:
    enabled = os.getenv("ENABLE_GEOPY_PARSE", "0") == "1"
    if not enabled or not address or Nominatim is None:
        return {}

    try:
        geolocator = Nominatim(user_agent="exposome-geocoder-cleaner", timeout=5)
        location = geolocator.geocode(address, addressdetails=True, exactly_one=True)
        if not location or not getattr(location, "raw", None):
            return {}

        addr = location.raw.get("address", {})
        city = addr.get("city") or addr.get("town") or addr.get("village") or ""
        state = addr.get("state") or ""
        postcode = addr.get("postcode") or ""
        zip5, zip9 = normalize_zip(postcode)
        return {
            "normalized_city": normalize_spaces(city).upper(),
            "normalized_state": normalize_state(state),
            "zip5": zip5,
            "zip9": zip9,
        }
    except Exception as ex:
        logger.debug(f"Optional geopy parsing failed: {ex}")
        return {}


def build_zip5_centroid_lookup(location_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    if location_df.empty or "zip5" not in location_df.columns:
        return {}

    centroid_source = location_df.loc[
        location_df["zip5"].astype(str).str.fullmatch(r"\d{5}", na=False)
        & location_df["latitude"].apply(pd.notna)
        & location_df["longitude"].apply(pd.notna),
        ["zip5", "latitude", "longitude"],
    ].copy()

    if centroid_source.empty:
        return {}

    centroid_source["zip5"] = centroid_source["zip5"].astype(str).str.zfill(5).str[:5]
    centroid_means = centroid_source.groupby("zip5")[["latitude", "longitude"]].mean()

    return {
        zip5: (float(row["latitude"]), float(row["longitude"]))
        for zip5, row in centroid_means.iterrows()
    }


def build_tract_centroid_lookup(location_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    if location_df.empty or "hud_tract" not in location_df.columns:
        return {}

    centroid_source = location_df.loc[
        location_df["hud_tract"].astype(str).str.fullmatch(r"\d{11}", na=False)
        & location_df["latitude"].apply(pd.notna)
        & location_df["longitude"].apply(pd.notna),
        ["hud_tract", "latitude", "longitude"],
    ].copy()

    if centroid_source.empty:
        return {}

    centroid_source["hud_tract"] = centroid_source["hud_tract"].astype(str).str.zfill(11).str[:11]
    centroid_means = centroid_source.groupby("hud_tract")[["latitude", "longitude"]].mean()

    return {
        tract: (float(row["latitude"]), float(row["longitude"]))
        for tract, row in centroid_means.iterrows()
    }


def classify_address_input_reason(row: pd.Series) -> str:
    street = normalize_spaces(row.get("normalized_street", "")) or normalize_spaces(row.get("address_1", "")) or normalize_spaces(row.get("address", ""))
    city = normalize_spaces(row.get("normalized_city", "")) or normalize_spaces(row.get("city", ""))
    state = normalize_spaces(row.get("normalized_state", "")) or normalize_spaces(row.get("state", ""))
    zip5 = normalize_spaces(row.get("zip5", "")) or normalize_spaces(row.get("zip", ""))

    if not any([street, city, state, zip5]):
        return "Blank/Incomplete address"
    if not street:
        return "Street missing"
    if not city:
        return "City missing"
    if not state:
        return "State missing"
    if not zip5:
        return "Zip missing"
    return "Blank/Incomplete address"


def classify_zip9_input_reason(row: pd.Series, zip9_reference: "Zip9Reference") -> str:
    zip9_value = re.sub(r"\D", "", normalize_spaces(row.get("zip9", "")))
    if not zip9_value:
        return "ZIP9 missing"
    if len(zip9_value) != 9:
        return "ZIP9 invalid"
    if not zip9_reference.has_zip9(zip9_value, row.get("normalized_state", "")):
        return "ZIP9 not found in zip9-fips12 crosswalk"
    return ""


def classify_zip5_input_reason(row: pd.Series, hud_zip5_reference: "HUDZip5Reference") -> str:
    zip5_value = re.sub(r"\D", "", normalize_spaces(row.get("zip5", "")))[:5]
    if not zip5_value:
        return "Zip missing"
    if len(zip5_value) != 5:
        return "Zip invalid"
    if not hud_zip5_reference.has_zip5(zip5_value, row.get("year")):
        return "ZIP5 not found in HUD crosswalk"
    return ""


def classify_geocode_failure_reason(geocode_result: object, lat: object, lon: object) -> str:
    result_text = normalize_spaces(geocode_result)
    if is_valid_coordinate(lat, lon):
        return ""
    if result_text:
        normalized_result = result_text.lower()
        geocoder_reason_map = {
            "imprecise_geocode": "Geocoder could not resolve precise coordinates",
            "ungeocodable_address": "Address not geocodable",
            "no_match": "No geocoder match",
            "multiple_matches": "Multiple ambiguous geocoder matches",
        }
        if normalized_result == "geocoded":
            return "No coordinate result returned"
        return geocoder_reason_map.get(normalized_result, f"Geocoder status: {normalized_result}")
    return "No geocode match"


def is_placeholder_value(value: object) -> bool:
    normalized = normalize_spaces(value).lower()
    return normalized in PLACEHOLDER_LOCATION_TOKENS


def classify_stage_failure_reason(stage: str, row: pd.Series, zip9_reference: "Zip9Reference", hud_zip5_reference: "HUDZip5Reference") -> str:
    stage_key = normalize_spaces(stage).lower()

    if stage_key == "address":
        street = normalize_spaces(row.get("normalized_street", "")) or normalize_spaces(row.get("address_1", ""))
        city = normalize_spaces(row.get("normalized_city", "")) or normalize_spaces(row.get("city", ""))
        state = normalize_spaces(row.get("normalized_state", "")) or normalize_spaces(row.get("state", ""))
        zip5 = normalize_spaces(row.get("zip5", "")) or normalize_spaces(row.get("zip", ""))

        if not any([street, city, state, zip5]):
            return "Missing address"
        if not street:
            return "Incomplete address: street missing"
        if not city:
            return "Incomplete address: city missing"
        if not state:
            return "Incomplete address: state missing"
        if not zip5:
            return "Incomplete address: ZIP missing"
        if any(is_placeholder_value(value) for value in [street, city, state]):
            return "Invalid address: placeholder values"

        return classify_geocode_failure_reason(row.get("geocode_result", ""), row.get("lat"), row.get("lon"))

    if stage_key == "zip9":
        zip9_value = re.sub(r"\D", "", normalize_spaces(row.get("zip9", "")))
        if not zip9_value:
            return "Missing ZIP9"
        if len(zip9_value) != 9:
            return "Invalid ZIP9"
        if not zip9_reference.has_zip9(zip9_value, row.get("normalized_state", "")):
            return "ZIP9 not found in crosswalk"

        return "ZIP9 available but could not generate coordinates"

    if stage_key == "zip5":
        zip5_value = re.sub(r"\D", "", normalize_spaces(row.get("zip5", "")))[:5]
        if not zip5_value:
            return "Missing ZIP5"
        if len(zip5_value) != 5:
            return "Invalid ZIP5"
        if not hud_zip5_reference.has_zip5(zip5_value, row.get("year")):
            return "ZIP5 not found in crosswalk"

        return "ZIP5 available but could not generate coordinates"

    return classify_geocode_failure_reason(row.get("geocode_result", ""), row.get("lat"), row.get("lon"))


class Zip9Reference:
    def __init__(self):
        self.base_path = self._resolve_base_path()
        self._cache_by_state = {}

    @staticmethod
    def _resolve_base_path() -> Optional[str]:
        env_override = os.getenv("ZIP9_CROSSWALK_DIR", "").strip()
        candidates = [
            env_override,
            os.path.join(os.getcwd(), "reference_codes", "input", "crosswalks", "zip9-fips12"),
            "/workspace/reference_codes/input/crosswalks/zip9-fips12",
            os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "reference_codes", "input", "crosswalks", "zip9-fips12"),
        ]
        for candidate in candidates:
            if os.path.isdir(candidate):
                return candidate
        return None

    def _load_state_map(self, state_code: str) -> Dict[str, str]:
        if state_code in self._cache_by_state:
            return self._cache_by_state[state_code]
        if not self.base_path:
            self._cache_by_state[state_code] = {}
            return {}

        file_path = os.path.join(self.base_path, f"convert{state_code.upper()}.txt")
        if not os.path.exists(file_path):
            self._cache_by_state[state_code] = {}
            return {}

        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                payload = handle.read().strip()
            if not payload.startswith("{"):
                payload = "{" + payload
            if not payload.endswith("}"):
                payload = payload + "}"
            mapping = json.loads(payload)
            self._cache_by_state[state_code] = mapping
            return mapping
        except Exception as ex:
            logger.warning(f"Unable to load ZIP9 reference file for {state_code}: {ex}")
            self._cache_by_state[state_code] = {}
            return {}

    def has_zip9(self, zip9: str, state_code: str) -> bool:
        zip9_clean = re.sub(r"\D", "", str(zip9 or ""))
        state_clean = normalize_state(state_code)
        if len(zip9_clean) != 9 or len(state_clean) != 2:
            return False
        return zip9_clean in self._load_state_map(state_clean)


class HUDZip5Reference:
    def __init__(self):
        self.base_path = self._resolve_base_path()
        self._file_entries = self._discover_files()
        self._zip_sets_by_file = {}
        self._tract_maps_by_file = {}

    @staticmethod
    def _resolve_base_path() -> Optional[str]:
        env_override = os.getenv("HUD_CROSSWALK_DIR", "").strip()
        candidates = [
            env_override,
            os.path.join(os.getcwd(), "reference_codes", "input", "crosswalks"),
            "/workspace/reference_codes/input/crosswalks",
        ]
        for candidate in candidates:
            if candidate and os.path.isdir(candidate):
                return candidate
        return None

    @staticmethod
    def _parse_year_from_filename(file_path: Path) -> Optional[int]:
        stem = file_path.stem.upper()
        match = re.match(r"ZIP_TRACT_(\d{2})(\d{4})$", stem)
        if not match:
            return None
        return int(match.group(2))

    def _discover_files(self):
        if not self.base_path:
            return []
        entries = []
        for file_path in sorted(Path(self.base_path).glob("ZIP_TRACT_*.xlsx")):
            year = self._parse_year_from_filename(file_path)
            if year is not None:
                entries.append({"path": file_path, "year": year})
        return sorted(entries, key=lambda entry: entry["year"])

    def _select_file_for_year(self, year_value: Optional[object]) -> Optional[Path]:
        if not self._file_entries:
            return None

        parsed_year = None
        if year_value is not None and not pd.isna(year_value):
            try:
                parsed_year = int(float(year_value))
            except (TypeError, ValueError):
                parsed_year = None

        if parsed_year is None:
            return self._file_entries[-1]["path"]

        eligible = [entry for entry in self._file_entries if entry["year"] <= parsed_year]
        if eligible:
            return eligible[-1]["path"]
        return self._file_entries[0]["path"]

    def _load_zip_set(self, file_path: Path):
        cache_key = str(file_path)
        if cache_key in self._zip_sets_by_file:
            return self._zip_sets_by_file[cache_key]

        try:
            header = pd.read_excel(file_path, nrows=0)
            normalized_columns = {str(col).strip().lower(): col for col in header.columns}
            zip_col = normalized_columns.get("zip")
            if zip_col is None:
                logger.warning(f"ZIP column not found in HUD ZIP crosswalk file {file_path.name}")
                self._zip_sets_by_file[cache_key] = set()
                return set()

            frame = pd.read_excel(file_path, dtype=str, usecols=[zip_col])
            zip_set = set(frame[zip_col].astype(str).str.zfill(5).str[:5])
            self._zip_sets_by_file[cache_key] = zip_set
            return zip_set
        except Exception as ex:
            logger.warning(f"Unable to load HUD ZIP crosswalk file {file_path.name}: {ex}")
            self._zip_sets_by_file[cache_key] = set()
            return set()

    def _load_tract_map(self, file_path: Path):
        cache_key = str(file_path)
        if cache_key in self._tract_maps_by_file:
            return self._tract_maps_by_file[cache_key]

        try:
            frame = pd.read_excel(file_path, dtype=str)
            frame.columns = [str(col).strip().lower() for col in frame.columns]
            if "zip" not in frame.columns or "tract" not in frame.columns:
                logger.warning(f"ZIP or TRACT column not found in HUD ZIP crosswalk file {file_path.name}")
                self._tract_maps_by_file[cache_key] = {}
                return {}

            frame = frame[[col for col in ["zip", "tract", "res_ratio", "tot_ratio"] if col in frame.columns]].copy()
            frame["zip"] = frame["zip"].astype(str).str.zfill(5).str[:5]
            frame["tract"] = frame["tract"].astype(str).str.zfill(11).str[:11]
            if "res_ratio" in frame.columns:
                frame["res_ratio"] = pd.to_numeric(frame["res_ratio"], errors="coerce").fillna(0)
            if "tot_ratio" in frame.columns:
                frame["tot_ratio"] = pd.to_numeric(frame["tot_ratio"], errors="coerce").fillna(0)

            if "res_ratio" in frame.columns and (frame["res_ratio"] > 0).any():
                working = frame[frame["res_ratio"] > 0].copy()
                ratio_col = "res_ratio"
            elif "tot_ratio" in frame.columns and (frame["tot_ratio"] > 0).any():
                working = frame[frame["tot_ratio"] > 0].copy()
                ratio_col = "tot_ratio"
            else:
                working = frame.copy()
                ratio_col = "tot_ratio" if "tot_ratio" in frame.columns else "res_ratio"

            if working.empty:
                self._tract_maps_by_file[cache_key] = {}
                return {}

            best = working.sort_values(["zip", ratio_col], ascending=[True, False]).drop_duplicates("zip")
            tract_map = dict(zip(best["zip"], best["tract"]))
            self._tract_maps_by_file[cache_key] = tract_map
            return tract_map
        except Exception as ex:
            logger.warning(f"Unable to load HUD tract mapping from {file_path.name}: {ex}")
            self._tract_maps_by_file[cache_key] = {}
            return {}

    def has_zip5(self, zip5: object, year_value: Optional[object] = None) -> bool:
        zip5_clean = re.sub(r"\D", "", str(zip5 or ""))[:5]
        if len(zip5_clean) != 5:
            return False

        use_union_mode = os.getenv("HUD_ZIP5_VALIDATE_MODE", "union").strip().lower() != "year"
        if use_union_mode:
            union_set = set()
            for entry in self._file_entries:
                union_set.update(self._load_zip_set(entry["path"]))
            return zip5_clean in union_set

        selected_file = self._select_file_for_year(year_value)
        if selected_file is None:
            return False
        return zip5_clean in self._load_zip_set(selected_file)

    def lookup_best_tract(self, zip5: object, year_value: Optional[object] = None) -> str:
        zip5_clean = re.sub(r"\D", "", str(zip5 or ""))[:5]
        if len(zip5_clean) != 5:
            return ""

        use_union_mode = os.getenv("HUD_ZIP5_LOOKUP_MODE", "union").strip().lower() != "year"
        if use_union_mode:
            for entry in reversed(self._file_entries):
                tract_map = self._load_tract_map(entry["path"])
                tract_value = tract_map.get(zip5_clean, "")
                if tract_value:
                    return tract_value
            return ""

        selected_file = self._select_file_for_year(year_value)
        if selected_file is None:
            return ""
        return self._load_tract_map(selected_file).get(zip5_clean, "")


def run_degauss_for_addresses(df_input: pd.DataFrame, output_folder: str, threshold: float, file_tag: str) -> pd.DataFrame:
    if df_input.empty:
        return pd.DataFrame(columns=["_rid", "lat", "lon", "geocode_result"])

    host_base = os.getenv("HOST_PWD", os.getcwd())
    preprocessed_path = os.path.join(output_folder, f"preprocessed_{file_tag}.csv")
    df_input[["_rid", "address"]].to_csv(preprocessed_path, index=False)

    abs_output_folder = os.path.abspath(output_folder)
    container_cwd = os.getcwd()
    rel_path = os.path.relpath(abs_output_folder, container_cwd)
    container_input_path = f"/workspace/{rel_path}/{os.path.basename(preprocessed_path)}"

    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{host_base}:/workspace",
        "ghcr.io/degauss-org/geocoder:3.3.0",
        container_input_path,
        str(threshold),
    ]

    logger.info(f"Running geocoder for {file_tag} with {len(df_input)} records")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.debug(result.stdout)
    except subprocess.CalledProcessError as ex:
        logger.error(f"Geocoder failed at level {file_tag}: {ex.stderr}")
        return pd.DataFrame(columns=["_rid", "lat", "lon", "geocode_result"])

    output_path = os.path.join(output_folder, f"preprocessed_{file_tag}_geocoder_3.3.0_score_threshold_{threshold}.csv")
    if not os.path.exists(output_path):
        logger.error(f"Expected geocoder output missing: {output_path}")
        return pd.DataFrame(columns=["_rid", "lat", "lon", "geocode_result"])

    geocoded = pd.read_csv(output_path)
    for col in ["lat", "lon"]:
        geocoded[col] = pd.to_numeric(geocoded.get(col), errors="coerce")
    if "geocode_result" not in geocoded.columns:
        geocoded["geocode_result"] = ""

    return geocoded[["_rid", "lat", "lon", "geocode_result"]]


def apply_geocoding_fallback(
    location_df: pd.DataFrame,
    output_folder: str,
    zip9_reference: Zip9Reference,
    hud_zip5_reference: HUDZip5Reference,
    analysis_folder: str,
) -> pd.DataFrame:
    location_df = location_df.copy()
    location_df["latitude"] = pd.to_numeric(location_df.get("latitude"), errors="coerce")
    location_df["longitude"] = pd.to_numeric(location_df.get("longitude"), errors="coerce")

    location_df["_rid"] = np.arange(len(location_df))
    location_df["geocode_level"] = location_df.apply(
        lambda row: "provided" if is_valid_coordinate(row.get("latitude"), row.get("longitude")) else "pending",
        axis=1,
    )
    location_df["geocode_reason"] = ""

    parsed = location_df.apply(build_row_address_fields, axis=1, result_type="expand")
    location_df = pd.concat([location_df, parsed], axis=1)

    geopy_updates = location_df["normalized_full_address"].apply(geopy_parse_address_optional)
    geopy_updates_df = pd.DataFrame(list(geopy_updates)) if len(geopy_updates) else pd.DataFrame()
    if not geopy_updates_df.empty:
        for col in ["normalized_city", "normalized_state", "zip5", "zip9"]:
            if col in geopy_updates_df.columns:
                location_df[col] = np.where(
                    geopy_updates_df[col].fillna("") != "",
                    geopy_updates_df[col],
                    location_df[col],
                )

    location_df["address_query"] = location_df["normalized_full_address"].apply(normalize_geocoder_query)
    location_df["zip9_hyphen"] = location_df["zip9"].apply(format_zip9_hyphen)
    location_df["zip9_city_state_query"] = location_df.apply(
        lambda row: normalize_geocoder_query(
            format_clean_address("", "", row.get("normalized_city", ""), row.get("normalized_state", ""), row.get("zip9_hyphen", ""))
        ),
        axis=1,
    )
    location_df["zip9_query"] = location_df["zip9_hyphen"].apply(normalize_geocoder_query)
    location_df["zip5_city_state_query"] = location_df.apply(
        lambda row: normalize_geocoder_query(
            format_clean_address("", "", row.get("normalized_city", ""), row.get("normalized_state", ""), row.get("zip5", ""))
        ),
        axis=1,
    )
    location_df["zip5_query"] = location_df["zip5"].apply(normalize_geocoder_query)

    location_df["zip9_validated"] = location_df.apply(
        lambda row: zip9_reference.has_zip9(row.get("zip9", ""), row.get("normalized_state", "")),
        axis=1,
    )
    location_df["zip5_validated"] = location_df.apply(
        lambda row: hud_zip5_reference.has_zip5(row.get("zip5", ""), row.get("year")),
        axis=1,
    )
    location_df["hud_tract"] = location_df.apply(
        lambda row: hud_zip5_reference.lookup_best_tract(row.get("zip5", ""), row.get("year")),
        axis=1,
    )

    failure_reasons_by_location: Dict[str, List[str]] = {}

    def location_key(location_id: object) -> str:
        key = normalize_spaces(location_id)
        return key if key else str(location_id)

    def add_failure_reason(location_id: object, stage: str, reason: str) -> None:
        reason_clean = normalize_spaces(reason)
        if not reason_clean:
            return

        full_reason = f"{stage}: {reason_clean}" if stage else reason_clean
        key = location_key(location_id)
        bucket = failure_reasons_by_location.setdefault(key, [])
        if full_reason not in bucket:
            bucket.append(full_reason)

    def run_stage(stage: str, query_variant: str, query_column: str, candidate_mask: pd.Series):
        if not candidate_mask.any():
            return

        candidates = location_df.loc[
            candidate_mask,
            [
                "_rid",
                "location_id",
                query_column,
                "normalized_street",
                "address_1",
                "normalized_city",
                "city",
                "normalized_state",
                "state",
                "zip5",
                "zip9",
                "zip",
                "year",
            ],
        ].copy()
        candidates["address"] = candidates[query_column].apply(normalize_geocoder_query)
        candidates = candidates[candidates["address"] != ""].copy()
        if candidates.empty:
            return

        unique_queries = candidates[["address"]].drop_duplicates(subset=["address"]).copy()
        unique_queries["_rid"] = np.arange(len(unique_queries))

        geocoded = run_degauss_for_addresses(
            df_input=unique_queries[["_rid", "address"]],
            output_folder=analysis_folder,
            threshold=get_stage_threshold(stage),
            file_tag=f"{stage}_{query_variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        if geocoded.empty:
            for _, row in candidates.iterrows():
                add_failure_reason(row["location_id"], stage, "Geocoder returned no output")
            return

        unique_results = unique_queries.merge(geocoded, on="_rid", how="left")
        stage_results = candidates.merge(unique_results.drop(columns=["_rid"]), on="address", how="left")
        stage_results["valid"] = stage_results.apply(lambda row: is_valid_coordinate(row.get("lat"), row.get("lon")), axis=1)

        stage_failures = stage_results[~stage_results["valid"]]
        for _, row in stage_failures.iterrows():
            rejection_reason = classify_stage_failure_reason(stage, row, zip9_reference, hud_zip5_reference)
            add_failure_reason(row["location_id"], stage, rejection_reason)

        success = stage_results[stage_results["valid"]].copy()
        for _, row in success.iterrows():
            rid = row["_rid"]
            location_df.loc[location_df["_rid"] == rid, "latitude"] = row.get("lat")
            location_df.loc[location_df["_rid"] == rid, "longitude"] = row.get("lon")
            location_df.loc[location_df["_rid"] == rid, "geocode_level"] = stage
            location_df.loc[location_df["_rid"] == rid, "geocode_reason"] = ""

    pending = location_df["geocode_level"] == "pending"

    address_blank = pending & (location_df["address_query"] == "")
    for _, row in location_df.loc[address_blank, ["location_id", "normalized_street", "address_1", "address", "normalized_city", "city", "normalized_state", "state", "zip5", "zip"]].iterrows():
        add_failure_reason(row["location_id"], "address", classify_address_input_reason(row))

    run_stage("address", "full_address", "address_query", pending & (location_df["address_query"] != ""))

    pending = location_df["geocode_level"] == "pending"
    zip9_missing = pending & (location_df["zip9_query"] == "")
    zip9_not_validated = pending & (location_df["zip9_query"] != "") & (~location_df["zip9_validated"])
    for mask in [zip9_missing, zip9_not_validated]:
        for _, row in location_df.loc[mask, ["location_id", "zip9", "zip9_query", "normalized_state"]].iterrows():
            add_failure_reason(row["location_id"], "zip9", classify_zip9_input_reason(row, zip9_reference))

    run_stage("zip9", "city_state_zip9", "zip9_city_state_query", pending & (location_df["zip9_city_state_query"] != "") & location_df["zip9_validated"])
    pending = location_df["geocode_level"] == "pending"
    run_stage("zip9", "zip9_only", "zip9_query", pending & (location_df["zip9_query"] != "") & location_df["zip9_validated"])

    pending = location_df["geocode_level"] == "pending"
    zip5_missing = pending & (location_df["zip5_query"] == "")
    zip5_not_validated = pending & (location_df["zip5_query"] != "") & (~location_df["zip5_validated"])
    for mask in [zip5_missing, zip5_not_validated]:
        for _, row in location_df.loc[mask, ["location_id", "zip5", "zip5_query", "year"]].iterrows():
            add_failure_reason(row["location_id"], "zip5", classify_zip5_input_reason(row, hud_zip5_reference))

    run_stage("zip5", "city_state_zip5", "zip5_city_state_query", pending & (location_df["zip5_city_state_query"] != "") & location_df["zip5_validated"])
    pending = location_df["geocode_level"] == "pending"
    run_stage("zip5", "zip5_only", "zip5_query", pending & (location_df["zip5_query"] != "") & location_df["zip5_validated"])

    pending = location_df["geocode_level"] == "pending"
    zip5_centroids = build_zip5_centroid_lookup(location_df)
    if zip5_centroids:
        zip5_proxy_mask = pending & location_df["zip5"].astype(str).isin(zip5_centroids)
        for _, row in location_df.loc[zip5_proxy_mask, ["location_id", "zip5"]].iterrows():
            centroid = zip5_centroids.get(normalize_spaces(row.get("zip5", ""))[:5], None)
            if not centroid:
                continue
            lat, lon = centroid
            location_df.loc[location_df["location_id"] == row["location_id"], "latitude"] = lat
            location_df.loc[location_df["location_id"] == row["location_id"], "longitude"] = lon
            location_df.loc[location_df["location_id"] == row["location_id"], "geocode_level"] = "zip5"
            location_df.loc[location_df["location_id"] == row["location_id"], "geocode_reason"] = ""

    pending = location_df["geocode_level"] == "pending"
    if pending.any():
        try:
            import pgeocode
            nomi = pgeocode.Nominatim("us")
            zip5_pgeocode_mask = pending & (location_df["zip5"].fillna("") != "")
            unique_zips = location_df.loc[zip5_pgeocode_mask, "zip5"].unique()
            
            pgeocode_dict = {}
            for z in unique_zips:
                z_clean = str(z).strip()[:5]
                if len(z_clean) == 5:
                    res = nomi.query_postal_code(z_clean)
                    if res is not None and not pd.isna(res.latitude) and not pd.isna(res.longitude):
                        pgeocode_dict[z_clean] = (float(res.latitude), float(res.longitude))
            
            pgeocode_valid_mask = pending & location_df["zip5"].astype(str).str.strip().str[:5].isin(pgeocode_dict)
            for _, row in location_df.loc[pgeocode_valid_mask, ["location_id", "zip5"]].iterrows():
                centroid = pgeocode_dict.get(str(row["zip5"]).strip()[:5])
                if centroid:
                    location_df.loc[location_df["location_id"] == row["location_id"], "latitude"] = centroid[0]
                    location_df.loc[location_df["location_id"] == row["location_id"], "longitude"] = centroid[1]
                    location_df.loc[location_df["location_id"] == row["location_id"], "geocode_level"] = "zip5"
                    location_df.loc[location_df["location_id"] == row["location_id"], "geocode_reason"] = ""
        except ImportError:
            logger.debug("pgeocode is not installed; offline ZIP5 centroid fallback skipped.")

    pending = location_df["geocode_level"] == "pending"
    tract_centroids = build_tract_centroid_lookup(location_df)
    if tract_centroids:
        tract_proxy_mask = pending & location_df["hud_tract"].astype(str).isin(tract_centroids)
        for _, row in location_df.loc[tract_proxy_mask, ["location_id", "hud_tract"]].iterrows():
            tract = normalize_spaces(row.get("hud_tract", ""))[:11]
            centroid = tract_centroids.get(tract, None)
            if not centroid:
                continue
            lat, lon = centroid
            location_df.loc[location_df["location_id"] == row["location_id"], "latitude"] = lat
            location_df.loc[location_df["location_id"] == row["location_id"], "longitude"] = lon
            location_df.loc[location_df["location_id"] == row["location_id"], "geocode_level"] = "zip5"
            location_df.loc[location_df["location_id"] == row["location_id"], "geocode_reason"] = ""

    pending = location_df["geocode_level"] == "pending"
    location_df.loc[pending, "geocode_level"] = "failed"

    for destination, source in [
        ("address_1", "normalized_street"),
        ("address_2", "normalized_unit"),
        ("city", "normalized_city"),
        ("state", "normalized_state"),
        ("zip", "zip5"),
    ]:
        location_df[destination] = np.where(
            location_df[source].fillna("") != "",
            location_df[source],
            location_df[destination].fillna(""),
        )

    location_df["modifier_source_value"] = location_df.apply(
        lambda row: format_modifier_source_value(row.get("geocode_level"), row.get("geocode_reason")),
        axis=1,
    )

    failed_mask = location_df["geocode_level"] == "failed"
    if failure_reasons_by_location:
        location_df.loc[failed_mask, "geocode_reason"] = location_df.loc[failed_mask, "location_id"].apply(
            lambda value: " | ".join(failure_reasons_by_location.get(location_key(value), []))
        )
    location_df.loc[failed_mask & (location_df["geocode_reason"].fillna("") == ""), "geocode_reason"] = "No geocode match"

    location_df.drop(
        columns=["_rid", "address_query", "zip9_hyphen", "zip9_city_state_query", "zip9_query", "zip9_validated", "zip5_city_state_query", "zip5_validated", "zip5_query", "hud_tract"],
        inplace=True,
        errors="ignore",
    )
    return location_df


def prepare_location_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    base.rename(columns={col: col.lower().strip() for col in base.columns}, inplace=True)

    def get_series(col_name: str, default_value: str = "") -> pd.Series:
        if col_name in base.columns:
            return base[col_name].fillna(default_value)
        return pd.Series([default_value] * len(base), index=base.index)

    latitude_col = "latitude" if "latitude" in base.columns else "lat" if "lat" in base.columns else None
    longitude_col = "longitude" if "longitude" in base.columns else "lon" if "lon" in base.columns else None

    if "location_id" not in base.columns:
        base["location_id"] = np.arange(1, len(base) + 1)

    if "address_1" in base.columns:
        base["address_1"] = base["address_1"].fillna("")
    elif "street" in base.columns:
        base["address_1"] = base["street"].fillna("")
    elif "address" in base.columns:
        base["address_1"] = base["address"].fillna("")
    else:
        base["address_1"] = pd.Series([""] * len(base), index=base.index)

    base["address_2"] = get_series("address_2", "")
    base["city"] = get_series("city", "")
    base["state"] = get_series("state", "")
    base["zip"] = get_series("zip", "")
    base["county"] = get_series("county", "")
    base["location_source_value"] = get_series("location_source_value", "")
    base["country_concept_id"] = get_series("country_concept_id", "")
    base["country_source_value"] = get_series("country_source_value", DEFAULT_COUNTRY_SOURCE_VALUE)

    base["latitude"] = pd.to_numeric(base.get(latitude_col, np.nan), errors="coerce")
    base["longitude"] = pd.to_numeric(base.get(longitude_col, np.nan), errors="coerce")

    for col in ["entity_id", "person_id", "year", "start_date", "end_date"]:
        if col not in base.columns:
            base[col] = pd.NA

    selected = LOCATION_REQUIRED_COLUMNS + ["entity_id", "person_id", "year", "start_date", "end_date", "address"]
    for col in selected:
        if col not in base.columns:
            base[col] = pd.NA

    return base[selected].copy()


def prepare_location_history_dataframe(source_df: pd.DataFrame, location_df: pd.DataFrame, existing_history: Optional[pd.DataFrame]) -> pd.DataFrame:
    if existing_history is not None:
        history = existing_history.copy()
        history.rename(columns={col: col.lower().strip() for col in history.columns}, inplace=True)
        for col in LOCATION_HISTORY_REQUIRED_COLUMNS:
            if col not in history.columns:
                history[col] = pd.NA
        history = history[LOCATION_HISTORY_REQUIRED_COLUMNS].copy()
    else:
        history = pd.DataFrame()
        history["location_id"] = location_df["location_id"].values
        history["relationship_type_concept_id"] = DEFAULT_RELATIONSHIP_TYPE_CONCEPT_ID
        history["domain_id"] = DEFAULT_DOMAIN_ID

        entity_series = source_df.get("entity_id")
        if entity_series is None or entity_series.isna().all():
            entity_series = source_df.get("person_id")
        if entity_series is None:
            entity_series = pd.Series(np.arange(1, len(source_df) + 1), index=source_df.index)

        fallback_entity_series = pd.Series(np.arange(1, len(source_df) + 1), index=source_df.index)
        history["entity_id"] = pd.to_numeric(entity_series, errors="coerce").fillna(fallback_entity_series).astype(int)

        if "start_date" in source_df.columns and source_df["start_date"].notna().any():
            start_dates = pd.to_datetime(source_df["start_date"], errors="coerce")
        elif "year" in source_df.columns and source_df["year"].notna().any():
            years = pd.to_numeric(source_df["year"], errors="coerce").fillna(1998).astype(int)
            start_dates = pd.to_datetime(years.astype(str) + "-01-01", errors="coerce")
        else:
            start_dates = pd.Series([pd.NaT] * len(source_df), index=source_df.index)

        if "end_date" in source_df.columns and source_df["end_date"].notna().any():
            end_dates = pd.to_datetime(source_df["end_date"], errors="coerce")
        elif "year" in source_df.columns and source_df["year"].notna().any():
            years = pd.to_numeric(source_df["year"], errors="coerce").fillna(2020).astype(int)
            end_dates = pd.to_datetime(years.astype(str) + "-12-31", errors="coerce")
        else:
            end_dates = pd.Series([pd.NaT] * len(source_df), index=source_df.index)

        history["start_date"] = pd.to_datetime(start_dates, errors="coerce").dt.strftime("%Y-%m-%d")
        history["end_date"] = pd.to_datetime(end_dates, errors="coerce").dt.strftime("%Y-%m-%d")

    for col in ["relationship_type_concept_id", "domain_id"]:
        history[col] = pd.to_numeric(history[col], errors="coerce").fillna(
            DEFAULT_RELATIONSHIP_TYPE_CONCEPT_ID if col == "relationship_type_concept_id" else DEFAULT_DOMAIN_ID
        ).astype(int)

    fallback_history_entity_series = pd.Series(np.arange(1, len(history) + 1), index=history.index)
    history["entity_id"] = pd.to_numeric(history["entity_id"], errors="coerce").fillna(fallback_history_entity_series).astype(int)
    history["start_date"] = pd.to_datetime(history["start_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    history["end_date"] = pd.to_datetime(history["end_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")

    missing_start = (history["start_date"] == "").sum()
    if missing_start:
        logger.warning(
            f"Generated LOCATION_HISTORY contains {int(missing_start)} rows with blank start_date. "
            "Provide start_date or year in source input for OMOP-compliant temporal mapping."
        )

    return history[LOCATION_HISTORY_REQUIRED_COLUMNS].copy()


def load_input_data(input_folder: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    csv_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV files found in input folder.")

    location_file = next((f for f in csv_files if f.lower() == "location.csv"), None)
    location_history_file = next((f for f in csv_files if f.lower() == "location_history.csv"), None)

    if location_file:
        location_df = pd.read_csv(os.path.join(input_folder, location_file), dtype=str)
        history_df = pd.read_csv(os.path.join(input_folder, location_history_file), dtype=str) if location_history_file else None
        return location_df, history_df

    source_frames = []
    for file_name in csv_files:
        if file_name.lower() == "location_history.csv":
            continue
        frame = pd.read_csv(os.path.join(input_folder, file_name), dtype=str)
        for numeric_col in ["year", "latitude", "longitude", "lat", "lon", "entity_id", "person_id"]:
            if numeric_col in frame.columns:
                frame[numeric_col] = pd.to_numeric(frame[numeric_col], errors="ignore")
        source_frames.append(frame)

    if not source_frames:
        raise ValueError("No usable CSV files found. Provide LOCATION.csv or encounter CSV files.")

    return pd.concat(source_frames, ignore_index=True), None


def write_failure_report(location_df: pd.DataFrame, output_folder: str):
    failures = location_df[location_df["geocode_level"] == "failed"].copy()
    if failures.empty:
        return

    report_path = os.path.join(output_folder, f"geocode_failures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    failures[["location_id", "address_1", "address_2", "city", "state", "zip", "geocode_level", "geocode_reason"]].to_csv(report_path, index=False)
    logger.warning(f"Geocoding failures logged to: {report_path}")


def write_geocoding_summary(location_df: pd.DataFrame, output_folder: str):
    total_records = int(len(location_df))
    if total_records == 0:
        logger.warning("No LOCATION records were available to summarize geocoding success.")
        return

    valid_mask = location_df.apply(
        lambda row: is_valid_coordinate(row.get("latitude"), row.get("longitude")),
        axis=1,
    )
    linked_records = int(valid_mask.sum())
    unlinked_records = total_records - linked_records
    success_rate_percent = (linked_records / total_records) * 100

    geocode_level_counts = location_df.get("geocode_level", pd.Series(dtype=str)).fillna("").astype(str).str.lower().value_counts()

    summary_df = pd.DataFrame(
        [
            {
                "total_records": total_records,
                "records_with_coordinates": linked_records,
                "records_without_coordinates": unlinked_records,
                "success_rate_percent": round(success_rate_percent, 2),
                "level1_provided": int(geocode_level_counts.get("provided", 0)),
                "level2_address": int(geocode_level_counts.get("address", 0)),
                "level3_zip9": int(geocode_level_counts.get("zip9", 0)),
                "level4_zip5": int(geocode_level_counts.get("zip5", 0)),
                "failed": int(geocode_level_counts.get("failed", 0)),
            }
        ]
    )

    report_path = os.path.join(output_folder, f"geocoding_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    summary_df.to_csv(report_path, index=False)
    logger.info(
        f"Geocoding success rate: {success_rate_percent:.2f}% ({linked_records}/{total_records}). "
        f"Summary generated: {report_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate LOCATION and LOCATION_HISTORY from address/lat-lon inputs")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input folder path containing CSV files")
    args = parser.parse_args()

    input_folder = os.path.abspath(args.input)
    if not os.path.isdir(input_folder):
        logger.error(f"Input folder not found: {input_folder}")
        sys.exit(1)

    output_folder = ensure_output_folder(os.path.join(os.path.dirname(input_folder), "output"))
    configure_logging(output_folder)
    analysis_folder = create_analysis_preprocessed_folder(output_folder)

    logger.info("FIPS generation is disabled for this iteration. Producing LOCATION-based outputs only.")

    try:
        source_df, existing_history = load_input_data(input_folder)
        location_df = prepare_location_dataframe(source_df)

        zip9_reference = Zip9Reference()
        if not zip9_reference.base_path:
            logger.warning("ZIP9 reference files not found. ZIP9 validation will fail closed.")

        hud_zip5_reference = HUDZip5Reference()
        if not hud_zip5_reference.base_path or not hud_zip5_reference._file_entries:
            logger.warning("HUD ZIP crosswalk files not found. ZIP5 validation will fail closed.")

        location_df = apply_geocoding_fallback(
            location_df,
            output_folder,
            zip9_reference,
            hud_zip5_reference,
            analysis_folder,
        )

        location_df["country_source_value"] = location_df["country_source_value"].replace("", DEFAULT_COUNTRY_SOURCE_VALUE)
        location_df["country_source_value"] = location_df["country_source_value"].fillna(DEFAULT_COUNTRY_SOURCE_VALUE)

        location_output = location_df[LOCATION_REQUIRED_COLUMNS + ["modifier_source_value"]].copy()
        location_output_path = os.path.join(output_folder, "LOCATION.csv")
        location_output.to_csv(location_output_path, index=False)
        logger.info(f"LOCATION output generated: {location_output_path}")

        location_history_output = prepare_location_history_dataframe(source_df, location_df, existing_history)
        location_history_output_path = os.path.join(output_folder, "LOCATION_HISTORY.csv")
        location_history_output.to_csv(location_history_output_path, index=False)
        logger.info(f"LOCATION_HISTORY output generated: {location_history_output_path}")

        write_geocoding_summary(location_df, output_folder)
        write_failure_report(location_df, output_folder)
    finally:
        cleanup_analysis_preprocessed_folder(analysis_folder)


if __name__ == "__main__":
    main()

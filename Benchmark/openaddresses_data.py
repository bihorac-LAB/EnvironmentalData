"""Download and prepare public OpenAddresses-compatible data for benchmarks."""

import csv
import gzip
import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


DEFAULT_SOURCE_URL = (
    "https://gismaps.vdem.virginia.gov/arcgis/rest/services/"
    "VA_Base_Layers/VA_Address_Points/FeatureServer/0"
)
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "data" / "openaddresses_us_encounters.csv"
OPENADDRESSES_API_URL = "https://batch.openaddresses.io/api"
SOURCE_DEFINITION_URL = "https://github.com/openaddresses/openaddresses/blob/master/sources/us/va/statewide.json"
SOURCE_FIELDS = [
    "PREADDRNUM", "ADDRNUM", "ADDRNUMSUF", "STREET_PREMOD", "STREET_PREFIX", "STREET_PRE_TYPE",
    "STREET_NAME", "STREET_TYPE", "STREET_SUFFIX", "PO_NAME", "MUNICIPALITY", "STATE", "ZIP_5",
]
OUTPUT_FIELDS = [
    "entity_id", "location_id", "address_1", "address_2", "city", "state", "zip", "latitude",
    "longitude", "year", "start_date", "end_date", "location_source_value", "country_source_value",
]


class CrossHostRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Remove API authorization when OpenAddresses redirects to signed object storage."""

    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> urllib.request.Request | None:
        """Create a redirect request without leaking the bearer token to another host."""
        redirected = super().redirect_request(request, file_pointer, code, message, headers, new_url)
        old_host = urllib.parse.urlparse(request.full_url).netloc
        new_host = urllib.parse.urlparse(new_url).netloc
        if redirected is not None and old_host != new_host:
            redirected.remove_header("Authorization")
        return redirected


def load_env_file(env_path: Path) -> None:
    """Load missing environment variables from a simple dotenv file without overriding existing values."""
    if not env_path.is_file():
        return

    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid .env entry on line {line_number}: expected NAME=VALUE.")
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            raise ValueError(f"Invalid .env entry on line {line_number}: variable name is blank.")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(name, value)


def get_api_token(warn_if_missing: bool = False) -> str | None:
    """Return the configured OpenAddresses API token, optionally warning when it is unavailable."""
    token = os.getenv("OPENADDRESSES_API_TOKEN", "").strip() or os.getenv("OA_TOKEN", "").strip()
    if token:
        return token
    if warn_if_missing:
        warnings.warn(
            "OpenAddresses API token not found; using the public Virginia statewide address source. "
            "To enable authenticated OpenAddresses downloads, copy .env.example to .env and set "
            "OPENADDRESSES_API_TOKEN.",
            UserWarning,
            stacklevel=2,
        )
    return None


def request_json(url: str, parameters: dict[str, Any], timeout_seconds: float = 120.0) -> dict[str, Any]:
    """Request and decode a JSON document, raising an actionable error for network failures."""
    request_url = f"{url}?{urllib.parse.urlencode(parameters)}"
    request = urllib.request.Request(request_url, headers={"User-Agent": "EnvironmentalData-benchmark/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to download OpenAddresses source data from {url}: {error}") from error

    if "error" in payload:
        raise RuntimeError(f"OpenAddresses source returned an error: {payload['error']}")
    return payload


def count_csv_records(path: Path) -> int:
    """Return the number of data records in a CSV file."""
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return max(sum(1 for _ in csv_file) - 1, 0)


def join_address_parts(attributes: dict[str, Any], field_names: list[str]) -> str:
    """Join nonblank ArcGIS address attributes with normalized whitespace."""
    parts = [str(attributes.get(field_name) or "").strip() for field_name in field_names]
    return " ".join(part for part in parts if part)


def feature_to_encounter(feature: dict[str, Any], row_number: int, coordinate_fraction: float) -> dict[str, Any]:
    """Convert one Virginia address feature into the encounter schema used by the geocoder."""
    attributes = feature.get("attributes", {})
    geometry = feature.get("geometry") or {}
    address_number = join_address_parts(attributes, ["PREADDRNUM", "ADDRNUM", "ADDRNUMSUF"])
    street = join_address_parts(
        attributes,
        ["STREET_PREMOD", "STREET_PREFIX", "STREET_PRE_TYPE", "STREET_NAME", "STREET_TYPE", "STREET_SUFFIX"],
    )
    benchmark_year = 2010 + ((row_number - 1) % 15)
    coordinate_selector = ((row_number * 6_181) % 10_000) / 10_000
    retain_coordinate = coordinate_selector < coordinate_fraction
    location_id = f"OA-VA-{row_number:09d}"
    return {
        "entity_id": row_number,
        "location_id": location_id,
        "address_1": f"{address_number} {street}".strip(),
        "address_2": "",
        "city": str(attributes.get("PO_NAME") or attributes.get("MUNICIPALITY") or "").strip(),
        "state": str(attributes.get("STATE") or "VA").strip(),
        "zip": str(attributes.get("ZIP_5") or "").strip(),
        "latitude": geometry.get("y", "") if retain_coordinate else "",
        "longitude": geometry.get("x", "") if retain_coordinate else "",
        "year": benchmark_year,
        "start_date": f"{benchmark_year}-01-01",
        "end_date": f"{benchmark_year}-12-31",
        "location_source_value": location_id,
        "country_source_value": "UNITED STATES OF AMERICA",
    }


def normalized_feature_to_encounter(
    feature: dict[str, Any],
    row_number: int,
    state_code: str,
    coordinate_fraction: float,
) -> dict[str, Any]:
    """Convert a normalized OpenAddresses GeoJSON feature into one synthetic encounter."""
    properties = feature.get("properties") or {}
    geometry = feature.get("geometry") or {}
    coordinates = geometry.get("coordinates") or []
    if geometry.get("type") != "Point" or len(coordinates) < 2:
        return {}
    number = str(properties.get("number") or "").strip()
    street = str(properties.get("street") or "").strip()
    if not number or not street:
        return {}

    benchmark_year = 2010 + ((row_number - 1) % 15)
    coordinate_selector = ((row_number * 6_181) % 10_000) / 10_000
    retain_coordinate = coordinate_selector < coordinate_fraction
    location_id = f"OA-US-{row_number:09d}"
    return {
        "entity_id": row_number,
        "location_id": location_id,
        "address_1": f"{number} {street}",
        "address_2": str(properties.get("unit") or "").strip(),
        "city": str(properties.get("city") or properties.get("district") or "").strip(),
        "state": state_code.upper(),
        "zip": str(properties.get("postcode") or "").strip(),
        "latitude": coordinates[1] if retain_coordinate else "",
        "longitude": coordinates[0] if retain_coordinate else "",
        "year": benchmark_year,
        "start_date": f"{benchmark_year}-01-01",
        "end_date": f"{benchmark_year}-12-31",
        "location_source_value": location_id,
        "country_source_value": "UNITED STATES OF AMERICA",
    }


def get_us_address_sources(api_token: str) -> dict[str, list[dict[str, Any]]]:
    """Return downloadable US address jobs grouped by state code."""
    request = urllib.request.Request(
        f"{OPENADDRESSES_API_URL}/data",
        headers={"Authorization": f"Bearer {api_token}", "User-Agent": "EnvironmentalData-benchmark/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            entries = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to retrieve the OpenAddresses US source index: {error}") from error

    sources_by_state: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        source_name = str(entry.get("source") or "")
        source_parts = source_name.split("/")
        output = entry.get("output") or {}
        if len(source_parts) < 3 or source_parts[0] != "us" or entry.get("layer") != "addresses":
            continue
        if not entry.get("job") or not output.get("output"):
            continue
        state_code = source_parts[1].lower()
        if state_code in {"pr", "vi", "as", "gu", "mp"}:
            continue
        sources_by_state.setdefault(state_code, []).append(entry)

    for state_sources in sources_by_state.values():
        state_sources.sort(key=lambda entry: (-int(entry.get("size") or 0), str(entry.get("source"))))
    if not sources_by_state:
        raise RuntimeError("OpenAddresses returned no downloadable US address sources.")
    return sources_by_state


def read_source_features(api_token: str, job_id: int) -> Iterator[dict[str, Any]]:
    """Yield normalized features from one authenticated OpenAddresses job download."""
    url = f"{OPENADDRESSES_API_URL}/job/{job_id}/output/source.geojson.gz"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_token}", "User-Agent": "EnvironmentalData-benchmark/1.0"},
    )
    opener = urllib.request.build_opener(CrossHostRedirectHandler())
    with opener.open(request, timeout=120) as response:
        with gzip.GzipFile(fileobj=response) as compressed_stream:
            for raw_line in compressed_stream:
                if raw_line.strip():
                    yield json.loads(raw_line)


def download_us_encounters(
    output_path: Path,
    record_count: int,
    api_token: str,
    coordinate_fraction: float = 0.2,
) -> Path:
    """Stream a geographically balanced US sample from authenticated OpenAddresses jobs."""
    if record_count <= 0:
        raise ValueError(f"Record count must be positive; received {record_count}.")
    if not 0 <= coordinate_fraction <= 1:
        raise ValueError(f"Coordinate fraction must be between 0 and 1; received {coordinate_fraction}.")

    sources_by_state = get_us_address_sources(api_token=api_token)
    state_codes = sorted(sources_by_state)
    records_per_state = math.ceil(record_count / len(state_codes))
    partial_path = output_path.with_suffix(f"{output_path.suffix}.part")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written_count = 0
    used_sources: list[str] = []

    try:
        with partial_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            for state_number, state_code in enumerate(state_codes, start=1):
                state_target = min(records_per_state, record_count - written_count)
                state_count = 0
                print(
                    f"OpenAddresses download: {state_code.upper()} ({state_number}/{len(state_codes)}), "
                    f"target {state_target:,} records...",
                    flush=True,
                )
                for source in sources_by_state[state_code]:
                    if state_count >= state_target:
                        break
                    try:
                        features = read_source_features(api_token=api_token, job_id=int(source["job"]))
                        source_used = False
                        for feature in features:
                            encounter = normalized_feature_to_encounter(
                                feature=feature,
                                row_number=written_count + 1,
                                state_code=state_code,
                                coordinate_fraction=coordinate_fraction,
                            )
                            if not encounter:
                                continue
                            writer.writerow(encounter)
                            written_count += 1
                            state_count += 1
                            source_used = True
                            if state_count >= state_target or written_count >= record_count:
                                break
                        if source_used:
                            used_sources.append(str(source["source"]))
                    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
                        print(f"Warning: skipping {source['source']}: {error}", flush=True)
                if written_count >= record_count:
                    break

        if written_count != record_count:
            raise RuntimeError(f"OpenAddresses returned only {written_count:,} usable US records; needed {record_count:,}.")
        partial_path.replace(output_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise

    source_metadata = {
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": written_count,
        "coordinate_fraction": coordinate_fraction,
        "source": "OpenAddresses authenticated individual US jobs",
        "states": state_codes,
        "source_count": len(used_sources),
        "sources": used_sources,
        "license_note": "Individual OpenAddresses sources retain their original licenses.",
    }
    output_path.with_suffix(".metadata.json").write_text(json.dumps(source_metadata, indent=2), encoding="utf-8")
    return output_path


def download_encounters(
    output_path: Path,
    record_count: int,
    source_url: str = DEFAULT_SOURCE_URL,
    coordinate_fraction: float = 0.2,
) -> Path:
    """Download address features and write exactly ``record_count`` valid benchmark encounters."""
    if record_count <= 0:
        raise ValueError(f"Record count must be positive; received {record_count}.")
    if not 0 <= coordinate_fraction <= 1:
        raise ValueError(f"Coordinate fraction must be between 0 and 1; received {coordinate_fraction}.")

    metadata = request_json(url=source_url, parameters={"f": "json"})
    object_id_field = metadata.get("objectIdField") or metadata.get("objectIdFieldName")
    if not object_id_field:
        raise RuntimeError("OpenAddresses source metadata does not identify an object ID field.")
    page_size = min(int(metadata.get("maxRecordCount", 2_000)), 2_000)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(f"{output_path.suffix}.part")

    written_count = 0
    offset = 0
    try:
        with partial_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            while written_count < record_count:
                page = request_json(
                    url=f"{source_url}/query",
                    parameters={
                        "f": "json",
                        "where": "1=1",
                        "outFields": ",".join([object_id_field, *SOURCE_FIELDS]),
                        "returnGeometry": "true",
                        "outSR": 4326,
                        "orderByFields": f"{object_id_field} ASC",
                        "resultOffset": offset,
                        "resultRecordCount": page_size,
                    },
                )
                features = page.get("features", [])
                if not features:
                    break
                offset += len(features)
                for feature in features:
                    encounter = feature_to_encounter(
                        feature=feature,
                        row_number=written_count + 1,
                        coordinate_fraction=coordinate_fraction,
                    )
                    if not encounter["address_1"]:
                        continue
                    writer.writerow(encounter)
                    written_count += 1
                    if written_count == record_count:
                        break

        if written_count != record_count:
            raise RuntimeError(f"OpenAddresses source returned only {written_count:,} usable records; needed {record_count:,}.")
        partial_path.replace(output_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise

    source_metadata = {
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": written_count,
        "coordinate_fraction": coordinate_fraction,
        "source_url": source_url,
        "openaddresses_source_definition": SOURCE_DEFINITION_URL,
        "license": "Public domain; attribution not required per the OpenAddresses source definition.",
    }
    output_path.with_suffix(".metadata.json").write_text(json.dumps(source_metadata, indent=2), encoding="utf-8")
    return output_path


def prepare_openaddresses_input(
    record_count: int,
    output_path: Path = DEFAULT_CACHE_PATH,
    source_url: str = DEFAULT_SOURCE_URL,
    coordinate_fraction: float = 0.2,
) -> Path:
    """Return a cached prepared CSV, downloading a replacement when it has too few records."""
    cached_count = count_csv_records(path=output_path)
    if cached_count >= record_count:
        return output_path
    api_token = get_api_token()
    if api_token:
        print(f"Downloading {record_count:,} nationwide OpenAddresses records for the benchmark...", flush=True)
        return download_us_encounters(
            output_path=output_path,
            record_count=record_count,
            api_token=api_token,
            coordinate_fraction=coordinate_fraction,
        )

    print(f"Downloading {record_count:,} public Virginia address records for the benchmark...", flush=True)
    return download_encounters(
        output_path=output_path,
        record_count=record_count,
        source_url=source_url,
        coordinate_fraction=coordinate_fraction,
    )

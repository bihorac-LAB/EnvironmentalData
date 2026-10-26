#!/bin/bash

# Function to check command status
check_status() {
  if [ $? -ne 0 ]; then
    echo "ERROR: $1 failed."
    exit 1
  fi
}

CSV_HEADER_COLS=()

array_contains() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [ "$item" = "$needle" ]; then
      return 0
    fi
  done
  return 1
}

get_csv_header_columns() {
  local csv_path="$1"
  if [ ! -s "$csv_path" ]; then
    echo "ERROR: Required file ${csv_path} is missing or empty."
    exit 1
  fi

  local raw_header
  raw_header=$(head -n 1 "$csv_path" | tr -d '\r')
  if [ -z "$raw_header" ]; then
    echo "ERROR: ${csv_path} does not contain a valid header row."
    exit 1
  fi

  local raw_cols=()
  IFS=',' read -r -a raw_cols <<< "$raw_header"

  CSV_HEADER_COLS=()
  local raw_col
  for raw_col in "${raw_cols[@]}"; do
    local normalized
    normalized=$(echo "$raw_col" \
      | tr '[:upper:]' '[:lower:]' \
      | sed 's/^"//; s/"$//; s/^[[:space:]]*//; s/[[:space:]]*$//')
    if [ -n "$normalized" ]; then
      CSV_HEADER_COLS+=("$normalized")
    fi
  done
}

validate_supported_columns() {
  local file_name="$1"
  shift
  local supported_cols=("$@")
  local unsupported=()
  local col

  for col in "${CSV_HEADER_COLS[@]}"; do
    if ! array_contains "$col" "${supported_cols[@]}"; then
      unsupported+=("$col")
    fi
  done

  if [ "${#unsupported[@]}" -gt 0 ]; then
    echo "ERROR: ${file_name} contains unsupported columns: ${unsupported[*]}"
    exit 1
  fi
}

validate_required_columns() {
  local file_name="$1"
  shift
  local required_cols=("$@")
  local missing=()
  local col

  for col in "${required_cols[@]}"; do
    if ! array_contains "$col" "${CSV_HEADER_COLS[@]}"; then
      missing+=("$col")
    fi
  done

  if [ "${#missing[@]}" -gt 0 ]; then
    echo "ERROR: ${file_name} is missing required columns: ${missing[*]}"
    exit 1
  fi
}

csv_header_sql_list() {
  local IFS=','
  echo "${CSV_HEADER_COLS[*]}"
}

IFS=',' read -r -a DATA_SOURCE_LIST <<< "$DATA_SOURCES"
IFS=',' read -r -a VARIABLE_LIST <<< "$VARIABLES"
GEO_SUBSET_SOURCES=()

# 1. Wait for Postgres to be ready
echo "Waiting for PostgreSQL to start..."
until psql -U postgres -c '\l'; do
  echo "Postgres is unavailable - sleeping"
  sleep 2
done
echo "Postgres is up!"

# 2. Wait for Init Script (Schema Creation)
echo "Waiting for 'working' schema..."
until psql -U postgres -c "SELECT 1 FROM information_schema.schemata WHERE schema_name = 'working'" | grep -q 1; do
  echo "Schema 'working' not ready - sleeping"
  sleep 2
done
echo "Schema 'working' found!"

echo "Loading LOCATION AND LOCATION_HISTORY tables..."

LOCATION_SUPPORTED_COLS=(
  location_id
  address_1
  address_2
  city
  state
  zip
  county
  location_source_value
  country_concept_id
  country_source_value
  latitude
  longitude
  modifier_source_value
)
LOCATION_REQUIRED_COLS=(location_id latitude longitude)

LOCATION_HISTORY_SUPPORTED_COLS=(
  location_id
  relationship_type_concept_id
  domain_id
  entity_id
  start_date
  end_date
)
LOCATION_HISTORY_REQUIRED_COLS=(location_id start_date)

get_csv_header_columns "/source/LOCATION.csv"
validate_supported_columns "LOCATION.csv" "${LOCATION_SUPPORTED_COLS[@]}"
validate_required_columns "LOCATION.csv" "${LOCATION_REQUIRED_COLS[@]}"
LOCATION_COPY_COLS=$(csv_header_sql_list)

psql -v ON_ERROR_STOP=1 -U postgres -c "TRUNCATE TABLE working.location;"
psql -v ON_ERROR_STOP=1 -U postgres -c "DROP TABLE IF EXISTS working.location_raw;"
psql -v ON_ERROR_STOP=1 -U postgres -c "
  CREATE TABLE working.location_raw (
    location_id text,
    address_1 text,
    address_2 text,
    city text,
    state text,
    zip text,
    county text,
    location_source_value text,
    country_concept_id text,
    country_source_value text,
    latitude text,
    longitude text,
    modifier_source_value text
  );"

psql -v ON_ERROR_STOP=1 -U postgres -c "\copy working.location_raw(${LOCATION_COPY_COLS}) FROM '/source/LOCATION.csv' CSV HEADER DELIMITER E','"
check_status "Loading LOCATION.csv"

RAW_LOC_COUNT=$(psql -U postgres -t -c "SELECT count(*) FROM working.location_raw" | xargs)

psql -v ON_ERROR_STOP=1 -U postgres -c "TRUNCATE TABLE working.location;"
psql -v ON_ERROR_STOP=1 -U postgres -c "
INSERT INTO working.location (
  location_id,
  address_1,
  address_2,
  city,
  state,
  zip,
  county,
  location_source_value,
  country_concept_id,
  country_source_value,
  latitude,
  longitude,
  modifier_source_value
)
SELECT
  location_id,
  NULLIF(address_1, ''),
  NULLIF(address_2, ''),
  NULLIF(city, ''),
  NULLIF(state, ''),
  CASE
    WHEN NULLIF(zip, '') IS NULL THEN NULL
    ELSE LEFT(regexp_replace(zip, '[^0-9-]', '', 'g'), 10)
  END,
  NULLIF(county, ''),
  NULLIF(location_source_value, ''),
  country_concept_id,
  NULLIF(country_source_value, ''),
  latitude,
  longitude,
  NULLIF(modifier_source_value, '')
FROM (
  SELECT
    CASE
      WHEN NULLIF(location_id, '') ~ '^[0-9]+$' THEN NULLIF(location_id, '')::integer
      ELSE NULL
    END AS location_id,
    address_1,
    address_2,
    city,
    state,
    zip,
    county,
    location_source_value,
    CASE
      WHEN NULLIF(country_concept_id, '') ~ '^[-]?[0-9]+$' THEN NULLIF(country_concept_id, '')::integer
      ELSE NULL
    END AS country_concept_id,
    country_source_value,
    CASE
      WHEN NULLIF(latitude, '') ~ '^[-+]?[0-9]+([.][0-9]+)?$' THEN NULLIF(latitude, '')::numeric
      ELSE NULL
    END AS latitude,
    CASE
      WHEN NULLIF(longitude, '') ~ '^[-+]?[0-9]+([.][0-9]+)?$' THEN NULLIF(longitude, '')::numeric
      ELSE NULL
    END AS longitude,
    modifier_source_value
  FROM working.location_raw
) cleaned
WHERE location_id IS NOT NULL
  AND latitude IS NOT NULL
  AND longitude IS NOT NULL
  AND latitude BETWEEN -90 AND 90
  AND longitude BETWEEN -180 AND 180;"
check_status "Normalizing LOCATION.csv"

LOC_COUNT=$(psql -U postgres -t -c "SELECT count(*) FROM working.location" | xargs)
LOC_SKIPPED=$((RAW_LOC_COUNT - LOC_COUNT))
echo "Loaded ${LOC_COUNT} valid LOCATION rows; skipped ${LOC_SKIPPED} rows missing required linkage fields or with invalid coordinates."

get_csv_header_columns "/source/LOCATION_HISTORY.csv"
validate_supported_columns "LOCATION_HISTORY.csv" "${LOCATION_HISTORY_SUPPORTED_COLS[@]}"
validate_required_columns "LOCATION_HISTORY.csv" "${LOCATION_HISTORY_REQUIRED_COLS[@]}"
LOCATION_HISTORY_COPY_COLS=$(csv_header_sql_list)

psql -v ON_ERROR_STOP=1 -U postgres -c "TRUNCATE TABLE working.location_history;"
psql -v ON_ERROR_STOP=1 -U postgres -c "DROP TABLE IF EXISTS working.location_history_raw;"
psql -v ON_ERROR_STOP=1 -U postgres -c "
  CREATE TABLE working.location_history_raw (
    location_id text,
    relationship_type_concept_id text,
    domain_id text,
    entity_id text,
    start_date text,
    end_date text
  );"

psql -v ON_ERROR_STOP=1 -U postgres -c "\copy working.location_history_raw(${LOCATION_HISTORY_COPY_COLS}) FROM '/source/LOCATION_HISTORY.csv' CSV HEADER DELIMITER E','"
check_status "Loading LOCATION_HISTORY.csv"

RAW_LOC_HIST_COUNT=$(psql -U postgres -t -c "SELECT count(*) FROM working.location_history_raw" | xargs)

psql -v ON_ERROR_STOP=1 -U postgres -c "
INSERT INTO working.location_history (
  location_id,
  relationship_type_concept_id,
  domain_id,
  entity_id,
  start_date,
  end_date
)
SELECT
  location_id,
  relationship_type_concept_id,
  domain_id,
  entity_id,
  start_date,
  end_date
FROM (
  SELECT
    parsed.location_id,
    parsed.relationship_type_concept_id,
    parsed.domain_id,
    parsed.entity_id,
    parsed.start_date,
    COALESCE(
      CASE
        WHEN lower(NULLIF(parsed.end_date_raw, '')) IN ('na', 'null') THEN NULL
        WHEN NULLIF(parsed.end_date_raw, '') ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' THEN NULLIF(parsed.end_date_raw, '')::date
        WHEN NULLIF(parsed.end_date_raw, '') ~ '^[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}$' THEN to_date(NULLIF(parsed.end_date_raw, ''), 'MM/DD/YYYY')
        ELSE NULL
      END,
      make_date(EXTRACT(YEAR FROM parsed.start_date)::integer, 12, 31)
    ) AS end_date
  FROM (
    SELECT
      CASE
        WHEN NULLIF(location_id, '') ~ '^[0-9]+$' THEN NULLIF(location_id, '')::integer
        ELSE NULL
      END AS location_id,
      CASE
        WHEN NULLIF(relationship_type_concept_id, '') ~ '^[-]?[0-9]+$' THEN NULLIF(relationship_type_concept_id, '')::integer
        ELSE NULL
      END AS relationship_type_concept_id,
      CASE
        WHEN NULLIF(domain_id, '') ~ '^[-]?[0-9]+$' THEN NULLIF(domain_id, '')::integer
        ELSE NULL
      END AS domain_id,
      CASE
        WHEN NULLIF(entity_id, '') ~ '^[-]?[0-9]+$' THEN NULLIF(entity_id, '')::integer
        ELSE NULL
      END AS entity_id,
      CASE
        WHEN NULLIF(start_date, '') ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' THEN NULLIF(start_date, '')::date
        WHEN NULLIF(start_date, '') ~ '^[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}$' THEN to_date(NULLIF(start_date, ''), 'MM/DD/YYYY')
        ELSE NULL
      END AS start_date,
      end_date AS end_date_raw
    FROM working.location_history_raw
  ) parsed
) cleaned
WHERE location_id IS NOT NULL
  AND start_date IS NOT NULL
  AND end_date >= start_date;"
check_status "Normalizing LOCATION_HISTORY.csv"

LOC_HIST_COUNT=$(psql -U postgres -t -c "SELECT count(*) FROM working.location_history" | xargs)
LOC_HIST_SKIPPED=$((RAW_LOC_HIST_COUNT - LOC_HIST_COUNT))
echo "Loaded ${LOC_HIST_COUNT} valid LOCATION_HISTORY rows; skipped ${LOC_HIST_SKIPPED} rows missing required start_date or invalid timing."

# Verify load
if [ "$LOC_COUNT" -eq "0" ]; then
   echo "ERROR: working.location has 0 rows after copy! Check input file '/source/LOCATION.csv'."
   ls -l /source/LOCATION.csv
   head /source/LOCATION.csv
   exit 1
fi
echo "Loaded $LOC_COUNT locations."

if [ "$LOC_HIST_COUNT" -eq "0" ]; then
   echo "ERROR: working.location_history has 0 rows after normalization! Check input file '/source/LOCATION_HISTORY.csv'."
   ls -l /source/LOCATION_HISTORY.csv
   head /source/LOCATION_HISTORY.csv
   exit 1
fi
echo "Loaded $LOC_HIST_COUNT location history rows."

psql -v ON_ERROR_STOP=1 -U postgres -c "DROP TABLE IF EXISTS working.location_merge;"
psql -v ON_ERROR_STOP=1 -U postgres -c "
CREATE TABLE working.location_merge AS
SELECT
  b.location_id,
  b.relationship_type_concept_id,
  b.domain_id,
  b.entity_id,
  GREATEST(b.start_date, gs.year_start::date) AS start_date,
  LEAST(b.end_date, (gs.year_start + interval '1 year - 1 day')::date) AS end_date,
  a.latitude,
  a.longitude,
  a.modifier_source_value
FROM working.location a
INNER JOIN working.location_history b
  ON a.location_id = b.location_id
CROSS JOIN LATERAL generate_series(
  date_trunc('year', b.start_date)::date,
  date_trunc('year', b.end_date)::date,
  interval '1 year'
) AS gs(year_start);"
check_status "Merging location tables"

# * - * - * - * - * - * - * - * - * -
# Linkage settings (all optional)
#   LINKAGE_WORKERS        parallel database sessions and downloads (default: number of CPUs, at most 4)
#   LINKAGE_BATCH_SIZE     locations per checkpointed batch (default 10000)
#   LINKAGE_STATE_DIR      where finished batches are kept so an interrupted run resumes (default /source/.linkage_state)
#   LINKAGE_RESUME         0 discards finished batches and starts over (default 1)
#   LINKAGE_CACHE_DIR      where downloaded source files are kept for later runs (default /source/.linkage_cache)
#   LINKAGE_CACHE          0 downloads every source again and keeps nothing (default 1)
# * - * - * - * - * - * - * - * - * -
CPUS=$(nproc 2>/dev/null || echo 1)
WORKERS=${LINKAGE_WORKERS:-$(( CPUS < 4 ? CPUS : 4 ))}
BATCH_SIZE=${LINKAGE_BATCH_SIZE:-10000}
STATE_DIR=${LINKAGE_STATE_DIR:-/source/.linkage_state}
RESUME=${LINKAGE_RESUME:-1}
CACHE_DIR=${LINKAGE_CACHE_DIR:-/source/.linkage_cache}
[ "${LINKAGE_CACHE:-1}" = "0" ] && CACHE_DIR=""
ASSIGN_CHUNK=25000
# Private scratch folder: under Apptainer /tmp is the host's, shared with any other run on the node.
RUN_TMP=$(mktemp -d /tmp/linkage.XXXXXX)
for setting in WORKERS BATCH_SIZE; do
  if ! [[ "${!setting}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: LINKAGE_${setting} must be a positive integer, got '${!setting}'."
    exit 1
  fi
done

STAGE_START=$EPOCHREALTIME
stage_done() {
  echo "[timing] $1 $(awk "BEGIN{printf \"%.1f\", $EPOCHREALTIME - $STAGE_START}") s"
  STAGE_START=$EPOCHREALTIME
}

# Run at most $WORKERS commands at once. Callers check results through the files each command writes,
# not through exit codes, so a lost status cannot hide a failure.
pool_run() {
  while [ "$(jobs -rp | wc -l)" -ge "$WORKERS" ]; do
    wait -n 2>/dev/null || true
  done
  "$@" &
}

psql -q -v ON_ERROR_STOP=1 -U postgres -f /sql/linkage.sql > /dev/null
check_status "Loading /sql/linkage.sql"

# Handle VARIABLES=ALL — expand to all variable IDs for the requested data sources
if [ "${#VARIABLE_LIST[@]}" -eq 1 ] && [ "${VARIABLE_LIST[0]}" = "ALL" ]; then
  echo "VARIABLES=ALL detected. Fetching variable IDs for loaded data sources..."
  ALL_VARS=$(psql -U postgres -AXqtc "SELECT variable_source_id FROM backbone.variable_source WHERE data_source_uuid IN (${DATA_SOURCES}) ORDER BY variable_source_id")
  VARIABLE_LIST=()
  while IFS= read -r line; do
    [ -n "$line" ] && VARIABLE_LIST+=("$line")
  done <<< "$ALL_VARS"
  echo "Found ${#VARIABLE_LIST[@]} variables to process."
fi
RUN_VARIABLES=$(IFS=','; echo "${VARIABLE_LIST[*]}")

read -r ADDRESS_YEARS POINT_COUNT BATCH_COUNT <<< "$(psql -v ON_ERROR_STOP=1 -U postgres -AXqtc "SELECT working.lk_prepare_points(${BATCH_SIZE})")"
check_status "Preparing address points"
echo "Prepared ${ADDRESS_YEARS} address-years at ${POINT_COUNT} distinct points in ${BATCH_COUNT} batch(es) of up to ${BATCH_SIZE} locations."
stage_done "load_and_merge"

# * - * - * - * - * - * - * - * - * -
# Recovery state: one CSV per finished batch, valid only for the exact inputs, settings and code that made it.
# * - * - * - * - * - * - * - * - * -
FINGERPRINT=$(
  {
    sha256sum /source/LOCATION.csv /source/LOCATION_HISTORY.csv /sql/linkage.sql /sql/link_batch.sql "$0" | cut -d' ' -f1
    echo "VARIABLES=${RUN_VARIABLES}"
    echo "DATA_SOURCES=${DATA_SOURCES}"
    echo "BATCH_SIZE=${BATCH_SIZE}"
    psql -U postgres -AXqtc "SELECT version(), postgis_full_version()"
  } | sha256sum | cut -d' ' -f1
)
if [ -d "$STATE_DIR" ]; then
  if [ "$RESUME" = "0" ]; then
    reset_reason="LINKAGE_RESUME=0"
  elif [ "$(cat "$STATE_DIR/fingerprint" 2>/dev/null)" != "$FINGERPRINT" ]; then
    reset_reason="inputs, settings or code changed since it was written"
  else
    reset_reason=""
  fi
  if [ -n "$reset_reason" ]; then
    echo "Setting aside previous linkage state ($reset_reason): ${STATE_DIR} -> ${STATE_DIR}.previous"
    rm -rf "${STATE_DIR}.previous"
    mv "$STATE_DIR" "${STATE_DIR}.previous"
  fi
fi
mkdir -p "$STATE_DIR"
check_status "Creating ${STATE_DIR}"
echo "$FINGERPRINT" > "$STATE_DIR/fingerprint"
find "$STATE_DIR" -name 'part_*.csv.tmp' -delete

part_file() { printf '%s/part_%06d.csv' "$STATE_DIR" "$1"; }
PENDING=()
for ((b = 0; b < BATCH_COUNT; b++)); do
  [ -f "$(part_file "$b")" ] || PENDING+=("$b")
done
DONE_COUNT=$((BATCH_COUNT - ${#PENDING[@]}))
if [ "$DONE_COUNT" -gt 0 ]; then
  echo "Resuming: ${DONE_COUNT} of ${BATCH_COUNT} batch(es) already finished in ${STATE_DIR}."
fi

# * - * - * - * - * - * - * - * - * -
# Ingest data sources, ${WORKERS} at a time. Downloads are cached in ${CACHE_DIR} when it is set.
# * - * - * - * - * - * - * - * - * -
psql -q -v ON_ERROR_STOP=1 -U postgres -c "CREATE TABLE IF NOT EXISTS working.lk_ingested (data_source_uuid integer PRIMARY KEY, download_url text, table_name text);"

fetch_source() {  # fetch_source <src> <url> <file name> -> prints the local path, or nothing on failure
  local src=$1 url=$2 name=$3
  local dest
  if [ -n "$CACHE_DIR" ]; then
    dest="${CACHE_DIR}/${src}_$(printf '%s' "$url" | sha256sum | cut -c1-12)_${name}"
    if [ -s "$dest" ]; then
      echo "Using cached ${dest}" >&2
      echo "$dest"
      return
    fi
  else
    dest="/shapes/${src}/${name}"
  fi
  local part="${dest}.part.${BASHPID}"
  wget -q -O "$part" "$url"
  if [ $? -ne 0 ] || [ ! -s "$part" ]; then
    rm -f "$part"
    return
  fi
  mv "$part" "$dest"
  echo "$dest"
}

ingest_source() {
  local src=$1
  local download_url file_format shape_name shape_no_ext data_standard srid geom_type
  echo "*********************************************"
  echo "Retrieving metadata for variable ${src}..."
  download_url=$(psql -U postgres -AXqtc "SELECT download_url FROM backbone.data_source WHERE data_source_uuid = ${src}")
  file_format=$(psql -U postgres -AXqtc "SELECT download_subtype FROM backbone.data_source WHERE data_source_uuid = ${src}")
  shape_name=$(psql -U postgres -AXqtc "SELECT download_filename FROM backbone.data_source WHERE data_source_uuid = ${src}")
  shape_no_ext=$(psql -U postgres -AXqtc "SELECT split_part(download_filename, '.', 1) FROM backbone.data_source WHERE data_source_uuid = ${src}")
  data_standard=$(psql -U postgres -AXqtc "SELECT download_data_standard FROM backbone.data_source WHERE data_source_uuid = ${src}")
  srid=$(psql -U postgres -AXqtc "SELECT srid FROM backbone.data_source WHERE data_source_uuid = ${src}")
  geom_type=$(psql -U postgres -AXqtc "SELECT COALESCE(geom_type, 'MISSING') FROM backbone.data_source WHERE data_source_uuid = ${src}")
  echo "Metadata fetched from database!"

  # Already loaded into this database by an earlier run (docker exec again after a failure)
  if psql -U postgres -AXqtc "SELECT 1 FROM working.lk_ingested WHERE data_source_uuid = ${src} AND download_url = '${download_url}' AND to_regclass('public.' || lower(table_name)) IS NOT NULL" | grep -q 1; then
    echo "Source data for ${src} (public.${shape_no_ext}) is already loaded; skipping download and ingest."
    return
  fi

  echo "Fetching source data - ${shape_name} - for variable ${src}..."
  local dir="/shapes/${src}" archive
  mkdir -p "$dir"
  if [ "$file_format" = "zip" ] || [ "$file_format" = "tar.gz" ] || [ "$file_format" = "csv" ]; then
    if [ "$file_format" = "csv" ]; then
      archive=$(fetch_source "$src" "$download_url" "${shape_name}")
    else
      archive=$(fetch_source "$src" "$download_url" "${shape_name}.${file_format}")
    fi
    if [ -z "$archive" ]; then
      echo "ERROR: Download failed for ${shape_name} from ${download_url}. Skipping."
      return
    fi
  else
    echo "Catalog processing for ${file_format} not yet supported!"
    return
  fi
  if [ "$file_format" = "zip" ]; then
    echo "Processing ${archive}..."
    unzip -qq -o -d "$dir" "$archive"
    if [ $? -ne 0 ]; then
      echo "ERROR: Unzip failed for ${shape_name}! Skipping this data source."
      return
    fi
  elif [ "$file_format" = "tar.gz" ]; then
    tar -xzf "$archive" -C "$dir"
  elif [ "$archive" != "${dir}/${shape_name}" ]; then
    ln -sf "$archive" "${dir}/${shape_name}"
  fi
  [ -z "$CACHE_DIR" ] && [ "$file_format" != "csv" ] && rm -f "$archive"

  echo "Ingesting source data - ${shape_name} - for variable ${src}..."
  local TARGET_PATH="${dir}/${shape_name}"
  if [ ! -e "$TARGET_PATH" ]; then
    echo "Target path $TARGET_PATH not found. Searching recursively..."
    local FOUND
    FOUND=$(find "$dir" -name "${shape_name}" -print -quit)
    if [ -n "$FOUND" ]; then
      echo "Found at $FOUND"
      TARGET_PATH="$FOUND"
    else
      echo "ERROR: Could not find ${shape_name} in ${dir}"
      echo "Listing ${dir} contents:"
      ls -R "$dir"
    fi
  fi

  local loaded=1
  if [ "$geom_type" = "polygon" ] && [ "$data_standard" = "shp" ]; then
    psql -q -v ON_ERROR_STOP=1 -U postgres -c "DROP TABLE IF EXISTS public.${shape_no_ext} CASCADE;" > /dev/null 2>&1
    shp2pgsql -c -D -I -s "${srid}" "$TARGET_PATH" "${shape_no_ext}" \
      | psql -q -v ON_ERROR_STOP=1 -U postgres > /dev/null 2>&1 || loaded=0
    psql -q -v ON_ERROR_STOP=1 -U postgres -c "ALTER TABLE public.${shape_no_ext} ADD COLUMN wgs_geom geometry(MULTIPOLYGON, 4326);" > /dev/null || loaded=0
    if [ "$srid" != "4326" ]; then
      psql -q -v ON_ERROR_STOP=1 -U postgres -c "UPDATE public.${shape_no_ext} SET wgs_geom = ST_TRANSFORM(geom, 4326);" > /dev/null || loaded=0
    else
      psql -q -v ON_ERROR_STOP=1 -U postgres -c "UPDATE public.${shape_no_ext} SET wgs_geom = geom;" > /dev/null || loaded=0
    fi
  fi

  if [ "$data_standard" = "gdb" ]; then
    # Use WGS84 (EPSG:4326) and name the geometry column 'wgs_geom' for consistency with shapefile logic
    ogr2ogr \
      -f "PostgreSQL" \
      -overwrite \
      -t_srs EPSG:4326 \
      -lco GEOMETRY_NAME=wgs_geom \
      -nln "${shape_no_ext}" \
      PG:"host='localhost' port='${PGPORT:-5432}' dbname='postgres' user='postgres' password='$PGPASSWORD'" \
      "$TARGET_PATH" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
      echo "ogr2ogr FAILED for ${shape_no_ext}"
      loaded=0
    fi
  fi

  if [ "$data_standard" = "csv" ]; then
    ogr2ogr \
      -f "PostgreSQL" \
      -overwrite \
      -oo HEADERS=YES \
      -oo AUTODETECT_TYPE=NO \
      -nln "${shape_no_ext}" \
      PG:"host='localhost' port='${PGPORT:-5432}' dbname='postgres' user='postgres' password='$PGPASSWORD'" \
      "$TARGET_PATH" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
      echo "ogr2ogr FAILED for ${shape_no_ext}"
      loaded=0
    fi
  fi

  if [ "$loaded" = "1" ]; then
    psql -q -v ON_ERROR_STOP=1 -U postgres -c "INSERT INTO working.lk_ingested VALUES (${src}, '${download_url}', '${shape_no_ext}') ON CONFLICT (data_source_uuid) DO UPDATE SET download_url = EXCLUDED.download_url, table_name = EXCLUDED.table_name;" > /dev/null
  fi
  echo "Source data ingested for public.${shape_no_ext}!"
  echo "*********************************************"
}

if [ "$BATCH_COUNT" -eq 0 ]; then
  echo "No address-years to link (no LOCATION_HISTORY row matches a LOCATION row); skipping ingestion and linkage."
elif [ "${#PENDING[@]}" -eq 0 ]; then
  echo "Every batch is already finished; skipping ingestion and linkage."
else
  echo "LOCATION and LOCATION_HISTORY loaded and merged! Launching ingestion of datasets (${DATA_SOURCES}) with ${WORKERS} worker(s)..."
  rm -rf /shapes
  mkdir /shapes
  [ -n "$CACHE_DIR" ] && mkdir -p "$CACHE_DIR"
  mkdir -p "$RUN_TMP/ingest"
  for src in "${DATA_SOURCE_LIST[@]}"; do
    pool_run ingest_source "$src" > "$RUN_TMP/ingest/${src}.log" 2>&1
  done
  wait
  for src in "${DATA_SOURCE_LIST[@]}"; do
    cat "$RUN_TMP/ingest/${src}.log"
  done
  stage_done "ingest"

  # * - * - * - * - * - * - * - * - * -
  # Link: resolve variables, place points in polygons, read values, then one query per batch.
  # * - * - * - * - * - * - * - * - * -
  echo "Launching spatial join process to combine ${POINT_COUNT} points with ${#VARIABLE_LIST[@]} variable(s) in ${#PENDING[@]} batch(es), ${WORKERS} worker(s)..."
  OK_VARS=$(echo "SELECT working.lk_prepare_run(:'run_variables')" | psql -v ON_ERROR_STOP=1 -U postgres -AXqt -v run_variables="$RUN_VARIABLES")
  check_status "Resolving variables"
  echo "${OK_VARS} of ${#VARIABLE_LIST[@]} variable(s) can be linked."
  stage_done "resolve_variables"

  mkdir -p "$RUN_TMP/assign"
  assign_chunk() {  # assign_chunk <area id> <first pt_id> <last pt_id>
    psql -q -v ON_ERROR_STOP=1 -U postgres -AXtc "SELECT working.lk_assign($1, $2, $3)" > /dev/null \
      && touch "$RUN_TMP/assign/$1_$2.ok"
  }
  AREAS=$(psql -U postgres -AXqtc "SELECT area_id FROM working.lk_area ORDER BY area_id")
  expected=0
  for area in $AREAS; do
    for ((lo = 1; lo <= POINT_COUNT; lo += ASSIGN_CHUNK)); do
      pool_run assign_chunk "$area" "$lo" "$((lo + ASSIGN_CHUNK - 1))"
      expected=$((expected + 1))
    done
  done
  wait
  if [ "$(find "$RUN_TMP/assign" -name '*.ok' | wc -l)" -ne "$expected" ]; then
    echo "ERROR: Placing points in polygons failed for some chunks; see the errors above."
    exit 1
  fi
  stage_done "assign_points"

  VALUE_ROWS=$(psql -v ON_ERROR_STOP=1 -U postgres -AXqtc "SELECT working.lk_build_values()")
  check_status "Reading variable values"
  echo "Read ${VALUE_ROWS} polygon values."
  stage_done "read_values"

  link_batch() {  # link_batch <batch>
    local out
    out=$(part_file "$1")
    if psql -q -X -v ON_ERROR_STOP=1 -U postgres -v batch="$1" -f /sql/link_batch.sql > "${out}.tmp"; then
      mv "${out}.tmp" "$out"
    else
      rm -f "${out}.tmp"
      echo "ERROR: Batch $1 failed; rerun to resume from the finished batches."
    fi
  }
  for b in "${PENDING[@]}"; do
    pool_run link_batch "$b"
  done
  wait
  missing=0
  for ((b = 0; b < BATCH_COUNT; b++)); do
    [ -f "$(part_file "$b")" ] || missing=$((missing + 1))
  done
  if [ "$missing" -gt 0 ]; then
    echo "ERROR: ${missing} of ${BATCH_COUNT} batch(es) did not finish. Finished batches are kept in ${STATE_DIR}; run the script again to resume."
    exit 1
  fi
  stage_done "link_batches"
fi

echo "Data sets and variables processed! Exporting EXTERNAL_EXPOSURE..."

# Ids run 1..N in VARIABLES order, then address-year order, so the file does not depend on batching or workers.
{
  echo "CREATE TEMP TABLE lk_stage (var_ord integer, lm_id bigint, unit text, location_id integer, person_id integer,
    exposure_concept_id integer, exposure_start_date date, exposure_start_datetime timestamp, exposure_end_date date,
    exposure_end_datetime timestamp, exposure_type_concept_id integer, exposure_relationship_concept_id integer,
    exposure_source_concept_id integer, exposure_source_value varchar(50), exposure_relationship_source_value varchar(50),
    dose_unit_source_value varchar(50), quantity integer, modifier_source_value varchar(50), operator_concept_id integer,
    value_as_number float, value_as_concept_id integer, unit_concept_id integer, sdoh_data_year integer,
    sdoh_year_map_status varchar(20));"
  for ((b = 0; b < BATCH_COUNT; b++)); do
    echo "\\copy lk_stage FROM '$(part_file "$b")' CSV"
  done
  echo "TRUNCATE working.external_exposure;"
  echo "INSERT INTO working.external_exposure
    SELECT row_number() OVER (ORDER BY var_ord, lm_id, unit), location_id, person_id, exposure_concept_id,
           exposure_start_date, exposure_start_datetime, exposure_end_date, exposure_end_datetime,
           exposure_type_concept_id, exposure_relationship_concept_id, exposure_source_concept_id, exposure_source_value,
           exposure_relationship_source_value, dose_unit_source_value, quantity, modifier_source_value,
           operator_concept_id, value_as_number, value_as_concept_id, unit_concept_id, sdoh_data_year, sdoh_year_map_status
    FROM lk_stage;"
  echo "\\copy (SELECT * FROM working.external_exposure ORDER BY external_exposure_id) TO '/source/EXTERNAL_EXPOSURE.csv.tmp' CSV HEADER DELIMITER E','"
} > "$RUN_TMP/export.sql"
psql -q -v ON_ERROR_STOP=1 -U postgres -f "$RUN_TMP/export.sql" > /dev/null
check_status "Exporting EXTERNAL_EXPOSURE"
mv /source/EXTERNAL_EXPOSURE.csv.tmp /source/EXTERNAL_EXPOSURE.csv
chmod 777 /source/EXTERNAL_EXPOSURE.csv
ROW_COUNT=$(psql -U postgres -AXqtc "SELECT count(*) FROM working.external_exposure")
stage_done "export"
rm -rf "$RUN_TMP"
echo "Export complete. ${ROW_COUNT} rows written to /source/EXTERNAL_EXPOSURE.csv."

# Exposome Geocoder – Input Preparation and Usage Guide

> **Note:** This toolkit does **not** require or share any Protected Health Information (PHI).

This repository provides a **reproducible workflow** to geocode patient location data into OMOP `LOCATION` / `LOCATION_HISTORY` tables and link them with Exposome datasets for environmental exposure analysis.

> **Demo video** [Watch here](https://drive.google.com/drive/folders/14FdY0lB3WRIiYrgCdeje6NAQI7Q_RB2w?usp=sharing)
---
## 📑 Table of Contents
- [Exposome Geocoder: Input Preparation and Usage Guide](#exposome-geocoder--input-preparation-and-usage-guide)
  - [📑 Table of Contents](#-table-of-contents)
  - [Overview](#overview)
  - [Input Options](#input-options)
    - [Option 1: Address](#option-1-address)
    - [Option 2: Coordinates](#option-2-coordinates)
    - [Option 3: OMOP CDM](#option-3-omop-cdm)
  - [Usage Guide](#usage-guide)
    - [Step 1: Prepare Input Data](#step-1-prepare-input-data)
    - [Step 2: Generate LOCATION Tables](#step-2-generate-location-tables)
    - [Step 3: Output Structure](#step-3-output-structure)
    - [Step 4: GIS Linkage with PostGIS-Exposure Tool](#step-4-gis-linkage-with-postgis-exposure-tool)
      - [Datasets Linked](#datasets-linked)
      - [Prerequisites for GIS Linkage](#prerequisites-for-gis-linkage)
      - [Expected Outputs](#expected-outputs)
      - [GIS Linkage Workflow](#gis-linkage-workflow)
      - [Notes \& Tips](#notes--tips)
    - [Step 5: Validate \& Inspect Outputs](#step-5-validate--inspect-outputs)
    - [Step 6: Site-level Date Shifting (Optional)](#step-6-site-level-date-shifting-optional)
    - [Step 7: Upload \& Centralized De-identification](#step-7-upload--centralized-de-identification)
  - [References \& sample files](#references--sample-files)
  - [Related Office Hours](#related-office-hours)
  - [Appendix A: Geocoding Workflow](#appendix-a-geocoding-workflow)
    - [Method: DeGAUSS Toolkit (Docker-based)](#method-degauss-toolkit-docker-based)
    - [Script Reference](#script-reference)
    - [Environment Variables](#environment-variables)
  - [Appendix B: FIPS Codes (UF Web Platform only)](#appendix-b-fips-codes-uf-web-platform-only)
---

## Overview

This workflow uses **two separate Docker containers** to take patient addresses or coordinates all the way to an analysis-ready exposure file:

1. **Exposome Geocoder Container (`prismaplab/exposome-geocoder:1.0.4`)**  
   Converts addresses or coordinates into OMOP `LOCATION` / `LOCATION_HISTORY` tables with latitude and longitude.

2. **Exposome Linkage Container (`ghcr.io/chorus-ai/chorus-postgis-exposure:main`)**  
   Spatially joins those tables with environmental and social determinant datasets (ADI, SVI, EJI, AHRQ) to produce `EXTERNAL_EXPOSURE.csv`.

The path is the same for every site:

```
Your data  →  Step 2: Address_to_LOCATION.py  →  LOCATION.csv          →  Step 4: linkage  →  EXTERNAL_EXPOSURE.csv
                                                 LOCATION_HISTORY.csv
```

> ⚠️ **Version note:** Use **`1.0.4` or later**. The `Address_to_LOCATION.py` script and the ZIP9/HUD crosswalk reference data it depends on are **not present in `1.0.3` or earlier**.

> **A note on FIPS codes.** Earlier versions of this toolkit centred on generating Census Tract (FIPS 11-digit) codes. **Linkage does not use FIPS.** It joins on latitude and longitude, and `LOCATION.csv` deliberately contains no `FIPS` column. You can ignore FIPS entirely unless you are also uploading to the UF Environmental Data Web Platform, which is internal to UF and covered in [Appendix B](#appendix-b-fips-codes-uf-web-platform-only).

---

## Input Options
You need to prepare **only ONE** of the following data elements per encounter.  

### Option 1: Address
Sample input files [here](https://github.com/bihorac-LAB/EnvironmentalData/tree/main/Tools/demo/address_files/input)

- **Format A: Multi-Column Address**

| location_id | street           | city         | state | zip   | year | entity_id |
|-------------|------------------|--------------|-------|-------|------|-----------|
| 1           | 1250 W 16th St   | Jacksonville | FL    | 32209 | 2019 | 1         |
| 2           | 2001 SW 16th St  | Gainesville  | FL    | 32608 | 2019 | 2         |

> **Tip:** Street **and** ZIP are required. Missing these fields may lead to **imprecise geocoding**.

- **Format B: Single Column Address**

| location_id | address                                      | year | entity_id |
|-------------|----------------------------------------------|------|-----------|
| 1           | 1250 W 16th St Jacksonville FL 32209         | 2019 | 1         |
| 2           | 2001 SW 16th St Gainesville FL 32608         | 2019 | 2         |

> ⚠️ **Required for every row:** a non-blank `location_id` (these are **not** auto-generated, so supply your own site-stable identifiers), and **either** an address **or** a ZIP code. The script exits with an error listing the offending rows if either condition is unmet.

---

### Option 2: Coordinates

Sample input files [here](https://github.com/bihorac-LAB/EnvironmentalData/tree/main/Tools/demo/latlong_files/input)

| location_id | latitude   | longitude | entity_id | year |
|-------------|------------|-----------|-----------|------|
| 1           | 30.353463  | -81.6749  | 1         | 2015 |
| 2           | 29.634219  | -82.3433  | 2         | 2015 |

---

### Option 3: OMOP CDM

If your source data already lives in an OMOP CDM database, extract the following tables first, then run [Step 2](#step-2-generate-location-tables) on the exported CSVs.

| Table              | Required Columns |
|--------------------|------------------------------------------------------|
| person             | person_id                                            |
| visit_occurrence   | visit_occurrence_id, visit_start_date, visit_end_date, person_id |
| location           | location_id, address_1, address_2, city, state, zip, location_source_value, country_concept_id, country_source_value, latitude, longitude |
| location_history   | location_id, relationship_type_concept_id, domain_id, entity_id, start_date, end_date |

The `OMOP_to_FIPS.py` script can perform this extraction for you directly from SQL Server; see [Appendix B](#appendix-b-fips-codes-uf-web-platform-only) for its usage.

---

#### Optional Supporting Files

Including the following optional files will help streamline the **end-to-end workflow** between geocoding and exposome linkage:

- [`LOCATION.csv`](https://github.com/bihorac-LAB/EnvironmentalData/blob/main/Tools/demo/address_files/input/LOCATION.csv)  
- [`LOCATION_HISTORY.csv`](https://github.com/bihorac-LAB/EnvironmentalData/blob/main/Tools/demo/address_files/input/LOCATION_HISTORY.csv)

> **Important**: Do not date-shift your `LOCATION` / `LOCATION_HISTORY` files before linkage. Date shifting, if used, should occur after linkage in [Step 6](#step-6-site-level-date-shifting-optional).

If these files are provided during **geocoding**, the output will automatically include the updated latitude and longitude information required for the linkage container. If they are **not** provided, `Address_to_LOCATION.py` builds both files for you from your input CSV, so no manual step is needed.

##### LOCATION.csv (Follows CDM format)

| location_id | address_1 | address_2 | city | state | zip | county | location_source_value | country_concept_id | country_source_value | latitude | longitude |
|-------------|-----------|-----------|------|-------|-----|--------|----------------------|-------------------|---------------------|----------|-----------|
| 1           | 1248 N Blackstone Ave | | FRESNO | CA | 93703 | | UNITED STATES OF AMERICA | | UNITED STATES OF AMERICA | 36.75891146 | -119.7902719 |

##### LOCATION_HISTORY.csv (Follows CDM format)

| location_id | relationship_type_concept_id | domain_id | entity_id | start_date | end_date |
|-------------|------------------------------|-----------|-----------|------------|----------|
| 1           | 32848                        | 1147314   | 3763      | 2019-01-01 | 2019-12-31 |

> **On dates.** `start_date` is taken from `start_date`, `visit_start_date`, or a 4-digit `year` (as `YYYY-01-01`), in that order. `end_date` is taken only from `end_date` or `visit_end_date`; it is **not** inferred from `year`. When a value cannot be derived it is left **blank** rather than filled with a placeholder, and the run logs a warning naming the affected rows.

---

## Usage Guide

### Step 1: Prepare Input Data
Prepare **only ONE** of the data elements as indicated under the [Input Options](#input-options) per encounter.  
For **Option 1 (Address)** or **Option 2 (Coordinates)**, your data must be in a **CSV file** format. 

#### Folder Structure
- Place the CSV file(s) in a dedicated folder
  - 📂 `input_address/`  *(for address-based data)*  
  - 📂 `input_coordinates/`  *(for coordinate-based data)* 
- Optionally, include:
  -    `LOCATION.csv`
  -   `LOCATION_HISTORY.csv`
    
> ⚠️ Only `.csv` files are supported. Convert `.xlsx` or other formats before running the tool.

---

### Step 2: Generate LOCATION Tables

**Container:** `prismaplab/exposome-geocoder:1.0.4`  
Ensure **Docker Desktop** is running.

This step produces the `LOCATION.csv` and `LOCATION_HISTORY.csv` that the linkage container consumes in [Step 4](#step-4-gis-linkage-with-postgis-exposure-tool).

Coordinates are resolved through a four-tier fallback, stopping at the first tier that succeeds:

| Tier | Source | Default match threshold |
|------|--------|------------------------|
| 1 | Latitude/longitude already supplied | n/a |
| 2 | Full street address | 0.7 |
| 3 | ZIP9 to tract centroid | 0.3 |
| 4 | ZIP5 to tract centroid | 0.1 |

The tier used for each row is recorded in the `modifier_source_value` column of `LOCATION.csv` (for example, `Level 2 | lat/long generated from address`), so every coordinate carries its own provenance.

#### For macOS / Linux / Ubuntu

```bash
docker run -it --rm \
  -v "$(pwd)":/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e HOST_PWD="$(pwd)" \
  -w /workspace \
  prismaplab/exposome-geocoder:1.0.4 \
  /app/code/Address_to_LOCATION.py -i <input_folder_path>
```

#### For Windows
- Open Command Prompt or PowerShell
- Run command `wsl`
- Execute the same command as above inside your WSL terminal.

Example, if your file is inside 📂`input_address/`:

```bash
docker run -it --rm   -v "$(pwd)":/workspace   -v /var/run/docker.sock:/var/run/docker.sock   -e HOST_PWD="$(pwd)"   -w /workspace   prismaplab/exposome-geocoder:1.0.4   /app/code/Address_to_LOCATION.py -i input_address
```

> ℹ️ The script launches the DeGAUSS geocoder in a nested Docker container, which is why the command mounts the Docker socket and passes `HOST_PWD`. All geocoding runs **locally**; no address data leaves your machine.

---

### Step 3: Output Structure

Outputs are written to an `output/` folder created alongside your input folder.

```
output/
├── LOCATION.csv                        # OMOP CDM LOCATION + modifier_source_value
├── LOCATION_HISTORY.csv                # OMOP CDM LOCATION_HISTORY
├── geocoding_summary_<timestamp>.csv   # rows resolved per fallback tier
└── geocode_failures_<timestamp>.csv    # rows that could not be geocoded
```

Sample outputs: [demo/address_files/output](https://github.com/bihorac-LAB/EnvironmentalData/tree/main/Tools/demo/address_files/output)

`LOCATION.csv` carries the 12 standard OMOP CDM `LOCATION` columns plus a 13th, `modifier_source_value`, recording which fallback tier produced each coordinate. **This extra column is expected by the linkage container**, whose `location_raw` table declares it explicitly. Do not strip it.

> ✅ **Before moving on,** open `geocoding_summary_<timestamp>.csv` and check how many rows landed in each tier. A large Tier 4 (ZIP5) count means many coordinates are ZIP-centroid approximations rather than true street matches.
>
> ⚠️ If the log warns that ZIP9 or HUD crosswalk files were not found, Tiers 3 and 4 are **silently disabled** and those rows will appear as failures instead. Confirm you are on `1.0.4`, which ships the crosswalk data.

**Blank values are meaningful.** `entity_id`, `start_date`, and `end_date` are left blank when the source data does not supply them, rather than being filled with placeholder values. The script logs a warning listing affected rows. Review these before linkage.

#### Failure Reasons

Rows that could not be geocoded are listed in `geocode_failures_<timestamp>.csv` with a reason:

- **Hospital address given** – Detected from known hardcoded hospital addresses.  
- **Street missing** – No street info provided.  
- **Blank/Incomplete address** – Address is empty or has missing components.  
- **Zip missing** – ZIP code not provided.  

---

### Step 4: GIS Linkage with PostGIS-Exposure Tool

**Purpose:**  
Spatially joins the latitude and longitude from your `LOCATION` tables with geospatial indices (ADI, SVI, EJI, AHRQ) and produces `EXTERNAL_EXPOSURE.csv`.

---

#### Datasets Linked

Four dataset families are available. All are joined at **Census tract** level, using Census TIGER/Line tract geometry (data source `7700`) as the spatial backbone.

| Dataset | Source | Vintages available | Variables |
|---------|--------|--------------------|-----------|
| **ADI** — Area Deprivation Index | UW-Madison (Zenodo) | 2015, 2020, 2023 | 6 |
| **SVI** — Social Vulnerability Index | CDC/ATSDR | 2010, 2014, 2016, 2018, 2020, 2022 | 651 |
| **EJI** — Environmental Justice Index | CDC/ATSDR | 2022, 2024 | 239 |
| **AHRQ SDOH** — Social Determinants of Health | AHRQ (Zenodo) | 2009–2023, annual | 4,195 |

**How the ID numbers map to datasets.** The two centrally managed CSVs are the lookup tables for the numbers you pass in [Step 1](#gis-linkage-workflow) below:

| File | Column to use | Maps to |
|------|---------------|---------|
| [`VRBL_SRC_SIMPLE.csv`](https://github.com/chorus-ai/chorus-container-apps/blob/main/postgis-exposure/csv/VRBL_SRC_SIMPLE.csv) | `variable_source_id` → `VARIABLES` | One row per variable. `variable_name` is what lands in `exposure_source_value`; `dataset_type` says which family it belongs to (`ADI`, `SVI`, `EJI`, `AHRQ`). |
| [`DATA_SRC_SIMPLE.csv`](https://github.com/chorus-ai/chorus-container-apps/blob/main/postgis-exposure/csv/DATA_SRC_SIMPLE.csv) | `data_source_uuid` → `DATA_SOURCES` | One row per dataset **vintage** (dataset + year), with its download URL and documentation link. |

To confirm what a given ID is before requesting it, look the number up in the relevant file. For example, `variable_source_id` `96` resolves to its `variable_name` and `dataset_type` in `VRBL_SRC_SIMPLE.csv`; `data_source_uuid` `9922` is SVI's 2022 tract release in `DATA_SRC_SIMPLE.csv`.

---

#### Prerequisites for GIS Linkage
- Docker installed.
- Clone the [postgis-exposure repository](https://github.com/chorus-ai/chorus-container-apps/tree/main/postgis-exposure) and run all commands **from the `postgis-exposure` directory**.
- `LOCATION.csv` and `LOCATION_HISTORY.csv` produced in [Step 2](#step-2-generate-location-tables).
- Ensure `DATA_SRC_SIMPLE.csv` and `VRBL_SRC_SIMPLE.csv` files are available (centrally managed; no edits required).
- **Important:** Do **not** date-shift your `LOCATION` / `LOCATION_HISTORY` files before linkage.

Sample `DATA_SRC_SIMPLE.csv` and `VRBL_SRC_SIMPLE.csv`: [here](https://github.com/chorus-ai/chorus-container-apps/tree/main/postgis-exposure/csv)

---

#### Expected Outputs
- `EXTERNAL_EXPOSURE.csv` containing linked indices (ADI, SVI, EJI, AHRQ metrics).

---

#### GIS Linkage Workflow

**Step 0: Stage your input files.** Copy `LOCATION.csv` and `LOCATION_HISTORY.csv` from your geocoder `output/` folder into the `test/` directory of the cloned repo. The container reads them from the mount; there is **no separate ingest command**.

```bash
cp /path/to/output/LOCATION.csv         ./test/
cp /path/to/output/LOCATION_HISTORY.csv ./test/
```

**Step 1: Set the variable and data-source lists.**

```bash
export VARIABLES="96,98,100,102,110,112,116,118,..."
export DATA_SOURCES="7700,9910,9914,9916,9918,..."
```

> The full canonical lists are maintained in the [postgis-exposure README](https://github.com/chorus-ai/chorus-container-apps/blob/main/postgis-exposure/README.md#deploy). Copy them from there rather than from this guide, so you always get the current set. `VARIABLES` IDs come from `VRBL_SRC_SIMPLE.csv`; `DATA_SOURCES` IDs from `DATA_SRC_SIMPLE.csv`.

**Step 2: Start the Postgres/PostGIS container.**

```bash
docker run --rm --name postgis-chorus \
    --env POSTGRES_PASSWORD="dummy" \
    --env VARIABLES="$VARIABLES" \
    --env DATA_SOURCES="$DATA_SOURCES" \
    -v ./test:/source \
    -d ghcr.io/chorus-ai/chorus-postgis-exposure:main
```

> `POSTGRES_PASSWORD=dummy` is safe as-is: the database is local to this throwaway container and is never exposed off-host.

**Step 3: Wait for the database to come up** (10-20 seconds depending on your environment). Confirm with:

```bash
docker logs postgis-chorus
```

Wait until you see **`database is ready to accept connections`** before continuing.

**Step 4: Generate the external exposure file.**

```bash
docker exec postgis-chorus /app/produce_external_exposure.sh
```

**Step 5: Collect the output.** `EXTERNAL_EXPOSURE.csv` will appear in your mounted `./test` directory.

**Step 6: Stop the container.**

```bash
docker stop postgis-chorus
```

#### Notes & Tips
- Run these commands in Terminal (Mac) or WSL/PowerShell/Command Prompt on Windows; WSL is more robust for Docker on Windows.
- Run all commands from the `postgis-exposure` directory; the `-v ./test:/source` mount is relative to it.
- If your site needs more variables, expand `VARIABLES` accordingly.
- **Important**: The container may only run successfully once. To rerun, you may need to delete the container and image, then pull the image again.

---

### Step 5: Validate & Inspect Outputs

`EXTERNAL_EXPOSURE.csv` is in **long format** (one row per location, person, variable, and year), not one row per patient. Sample output: [demo/PostGIS-output](https://github.com/bihorac-LAB/EnvironmentalData/tree/main/Tools/demo/PostGIS-output).

Open `EXTERNAL_EXPOSURE.csv` and confirm:

| Column | What to check |
|--------|---------------|
| `location_id` | Matches the IDs you supplied in `LOCATION.csv` |
| `person_id` | Populated and matches your source data |
| `exposure_source_value` | Holds the variable name (for example `SVI_EP_POV`, `SVI_EP_UNEMP`) |
| `value_as_number` | Holds the measured value; should not be uniformly blank |
| `exposure_start_date` / `exposure_end_date` | Cover the expected period for each row |
| `sdoh_data_year` | The SDoH data year actually matched |
| `sdoh_year_map_status` | `nearest_year` means the exact year was unavailable and the closest was substituted; expected, but worth knowing |

> ℹ️ **There is no latitude, longitude, or FIPS column in this file.** Coordinates are consumed during the spatial join and are not carried through to the output. Their absence is not an error.

- Spot-check a few records for accuracy.
- Confirm the set of distinct `exposure_source_value` entries matches the `VARIABLES` you requested.
- If errors:
  - Ensure `LOCATION.csv` has valid, non-blank latitude and longitude
  - Confirm `VARIABLES` and `DATA_SOURCES` are correct
  - Check mount paths, and that both CSVs were staged in `./test` before the container started

---

### Step 6: Site-level Date Shifting (Optional)
**Purpose:** Anonymize temporal data while preserving relative timelines.

**Guidelines:**
- Apply date shifts locally before upload; do not date-shift prior to GIS linkage.
- Input: `EXTERNAL_EXPOSURE.csv` (from Step 4)
- Output: `EXTERNAL_EXPOSURE_date_shifted.csv`

See [Date Shifting SOP for More Details](https://github.com/chorus-ai/Chorus_SOP/blob/main/sop-website/docs/Privacy/Date-Shifting.mdx).

---

### Step 7: Upload & Centralized De-identification
1. Upload the (optionally date-shifted) `EXTERNAL_EXPOSURE.csv` to the central repository.
2. The central team will apply further de-identification.

---

## References & sample files

#### Geocoding
- Sample files: [Geocoding Demo Files](https://github.com/bihorac-LAB/EnvironmentalData/tree/main/Tools/demo)

#### GIS Linkage
- **Container documentation:** [postgis-exposure README](https://github.com/chorus-ai/chorus-container-apps/blob/main/postgis-exposure/README.md): authoritative source for the linkage container's deploy commands and the current `VARIABLES` / `DATA_SOURCES` lists.
- Sample files: [PostGIS Exposure CSVs](https://github.com/chorus-ai/chorus-container-apps/tree/main/postgis-exposure/csv)
  - **Site-specific:** `LOCATION`, `LOCATION_HISTORY`
  - **Centrally managed:** `DATA_SRC_SIMPLE`, `VRBL_SRC_SIMPLE`
- Sample linkage output: [demo/PostGIS-output](https://github.com/bihorac-LAB/EnvironmentalData/tree/main/Tools/demo/PostGIS-output)

---

## Related Office Hours

The following office hour sessions provide additional context and demonstrations related to this SOP:

- **[08-07-25] Integration of GIS and SDoH data with OMOP**
  - [Video Recording](https://drive.google.com/file/d/1MHx7YWlWIVC2Dggjzw2uczT3MKyppWZ5/view?usp=share_link) | [Transcript](https://docs.google.com/document/d/1v0C3COo1O-KOKpd7haVm5GGmD5gbVR0I/edit?usp=sharing&ouid=104468275537210259794&rtpof=true&sd=true)
  - Comprehensive session on integrating GIS and social determinants of health data

- **[09-18-25] Processing OMOP location_history table into external_exposure table**
  - [Video Recording](https://drive.google.com/file/d/1SWovm3vnf0PVbTC_qS6n3nBuf1dI169L/view?usp=share_link) | [Transcript](https://docs.google.com/document/d/1VII2X_NQhM69ZzUDwB6m4hsqDk6E1VCx/edit?usp=share_link&ouid=104468275537210259794&rtpof=true&sd=true)
  - Technical implementation of location data processing for external exposures

- **[09-25-25] End-to-end demo for capturing GIS data with OMOP**
  - [Video Recording](https://drive.google.com/file/d/118fGVQS0ES0SV4Yu6pxLc4SB0Z08Mq9I/view?usp=share_link) | [Transcript](https://docs.google.com/document/d/1-rPcdQp-7TcEF9-Fgwqi12508Mx2seT9/edit?usp=share_link&ouid=104468275537210259794&rtpof=true&sd=true)
  - Complete workflow demonstration for GIS data capture and processing

- **[10-16-2025] End-to-end demo for capturing GIS data with OMOP or address/latlong**
  - [Video Recording](https://drive.google.com/file/d/1L5OdVWa0AsuLKy-o0Uub8JNb1Wojw7ZO/view?usp=drive_link) | [Transcript](https://drive.google.com/file/d/1-P6edkHBfiAJKSG7ZKlkIj5Ej-u2EZ3G/view?usp=sharing)
  - Complete workflow demonstration for GIS data capture and processing based on updated documentation

---

## Appendix A: Geocoding Workflow

This appendix outlines the scripts and Docker-based DeGAUSS toolkit used internally by [Step 2](#step-2-generate-location-tables).

### Method: DeGAUSS Toolkit (Docker-based)

`Address_to_LOCATION.py` invokes one DeGAUSS container to convert addresses into coordinates:

| Purpose                  | Docker Image                                     |
|--------------------------|--------------------------------------------------|
| Address to Coordinates   | `ghcr.io/degauss-org/geocoder:3.3.0`             |

Executed internally as:

```bash
docker run --rm -v "ABS_OUTPUT_FOLDER:/tmp" \
  ghcr.io/degauss-org/geocoder:3.3.0 \
  /tmp/<your_preprocessed_input.csv> <threshold>
```

**Replace values:**
- `ABS_OUTPUT_FOLDER` → absolute path to your output directory  
- `<threshold>` → numeric value (for example `0.7`)  

---

### Script Reference

#### Address_to_LOCATION.py
This [script](https://github.com/bihorac-LAB/EnvironmentalData/blob/main/Tools/code/Address_to_LOCATION.py) is the LOCATION table producer used by [Step 2](#step-2-generate-location-tables):
- Reads CSV input (address, coordinates, or an existing `LOCATION.csv`)
- Normalizes and parses addresses using `usaddress`, with typo correction
- Resolves coordinates through the four-tier fallback (lat/long, address, ZIP9, ZIP5)
- Records the tier used per row in `modifier_source_value`
- Validates input up front: fails with an error if any row lacks a `location_id`, or lacks both address and ZIP
- Writes `LOCATION.csv`, `LOCATION_HISTORY.csv`, a geocoding summary, and a failure report
- **Does not generate FIPS codes**

---

### Environment Variables

All are optional; defaults are applied when unset.

| Variable | Default | Purpose |
|----------|---------|---------|
| `GEOCODER_THRESHOLD_ADDRESS` | `0.7` | Match threshold for the address tier |
| `GEOCODER_THRESHOLD_ZIP9` | `0.3` | Match threshold for the ZIP9 tier |
| `GEOCODER_THRESHOLD_ZIP5` | `0.1` | Match threshold for the ZIP5 tier |
| `ZIP9_CROSSWALK_DIR` | *(set in image)* | Location of the ZIP9 to FIPS12 crosswalk files |
| `HUD_CROSSWALK_DIR` | *(set in image)* | Location of the HUD `ZIP_TRACT_*.xlsx` crosswalks |
| `HUD_ZIP5_VALIDATE_MODE` / `HUD_ZIP5_LOOKUP_MODE` | *(unset)* | Control HUD ZIP5 validation and lookup behaviour |
| `KEEP_PREPROCESSED` | unset | Retain intermediate preprocessing files for debugging |
| `HOST_PWD` | `$(pwd)` | Host path used when launching the nested DeGAUSS container |
| `ENABLE_GEOPY_PARSE` | `0` (**off**) | See privacy warning below |

> 🔒 **Privacy note on `ENABLE_GEOPY_PARSE`.** This variable is **off by default and should stay off**. When set to `1`, the script sends normalized addresses to **Nominatim, OpenStreetMap's public geocoding web service**, which means address data leaves your environment. This is incompatible with this toolkit's stated guarantee of local-only processing. Do not enable it on patient data.
>
> Note that the unrelated `pgeocode.Nominatim` used internally for ZIP centroid lookups is a **local offline dataset** and performs no network calls. The two share a name but are not the same thing.

---

## Appendix B: FIPS Codes (UF Web Platform only)

> **Most sites can skip this appendix entirely.** FIPS codes are **not** used by the CHoRUS exposome linkage in [Step 4](#step-4-gis-linkage-with-postgis-exposure-tool), which joins on latitude and longitude. This appendix applies only if you are also uploading to the UF Environmental Data Web Platform, which is internal to UF.

`LOCATION.csv` never contains a `FIPS` column. `LOCATION` is a fixed-schema OMOP CDM table and the linkage container does not read FIPS, so adding it has no effect. FIPS codes are written to a separate `<filename>_with_fips.csv` instead. This is intentional; please do not patch FIPS back into `LOCATION.csv`.

### Generating FIPS Codes

FIPS generation adds a second DeGAUSS container that converts coordinates to Census Tract identifiers:

| Purpose                  | Docker Image                                     |
|--------------------------|--------------------------------------------------|
| Coordinates to FIPS      | `ghcr.io/degauss-org/census_block_group:0.6.0`   |

#### For CSV Input (Option 1 & 2)

```bash
docker run -it --rm \
  -v "$(pwd)":/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e HOST_PWD="$(pwd)" \
  -w /workspace \
  prismaplab/exposome-geocoder:1.0.4 \
  /app/code/Address_to_FIPS.py -i <input_folder_path>
```

`Address_to_FIPS.py` reads CSV files, normalizes the address or uses supplied coordinates, runs both DeGAUSS containers, and packages outputs into ZIP archives.

**Files generated.** Each input file produces:
  - `<filename>_with_coordinates.csv`: input + latitude/longitude  
  - `<filename>_with_fips.csv`: input + FIPS codes  

```
output/
├── coordinates_from_address_<timestamp>.zip
├── geocoded_fips_codes_<timestamp>.zip
```

> `<timestamp>` indicates when the script was executed (for example, 20250624_150230).

**Zipped output columns**

| Column           | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `Latitude`       | Latitude returned from the geocoder                                         |
| `Longitude`      | Longitude returned from the geocoder                                        |
| `geocode_result` | Outcome of geocoding: `geocoded` for successful matches, `Imprecise Geocode` if not precise |
| `reason`         | Failure reason if applicable                                                |

> 💡 **Tip:** You can expand hospital detection by adding known addresses to `HOSPITAL_ADDRESSES` in [`Address_to_FIPS.py`](https://github.com/bihorac-LAB/EnvironmentalData/blob/main/Tools/code/Address_to_FIPS.py). Format: single-line string, lowercase letters and numbers only, no commas or special characters, fields separated by single spaces.

#### For OMOP Input (Option 3)

To extract and geocode directly from an OMOP SQL Server database:

```bash
docker run -it --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$(pwd)":/workspace \
  -e HOST_PWD="$(pwd)" \
  -w /workspace \
  prismaplab/exposome-geocoder:1.0.4 \
  /app/code/OMOP_to_FIPS.py \
    --user <your_username> \
    --password <your_password> \
    --server <server_address> \
    --port <port_number> \
    --database <database_name>
```

`OMOP_to_FIPS.py` extracts OMOP CDM data, categorizes records into valid/invalid address or coordinates, runs FIPS generation, and also writes `LOCATION.csv` and `LOCATION_HISTORY.csv`.

**Sample outputs:** [demo/OMOP/output](https://github.com/bihorac-LAB/EnvironmentalData/tree/main/Tools/demo/OMOP/output)

```
OMOP_data/
├── valid_address/               # Records with address, no lat/lon
├── invalid_lat_lon_address/     # Records missing both address and lat/lon
├── valid_lat_long/              # Records with lat/lon

OMOP_FIPS_result/
├── address/
│   ├── address_with_coordinates.zip   # CSVs with lat/lon from address
│   └── address_with_fips.zip          # CSVs with FIPS codes
├── latlong/
│   └── latlong_with_fips.zip          # CSVs with FIPS from coordinates
├── invalid/                           # Usually empty; no usable location data

LOCATION.csv
LOCATION_HISTORY.csv
```

### Uploading to the UF Environmental Data Web Platform

1. Register at [https://exposome.rc.ufl.edu](https://exposome.rc.ufl.edu/)  
2. Upload the `*_with_fips.zip` file generated above
3. Input CSV must contain:  
   - `person_id`  
   - `visit_occurrence_id`  
   - `year`  
   - `FIPS`
4. Select the dataset you want to link it to
5. Download enriched dataset with SDoH variables

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
  - [Appendix: Geocoding Workflow](#appendix-geocoding-workflow)
    - [Method: DeGAUSS Toolkit (Docker-based)](#method-degauss-toolkit-docker-based)
    - [Script Reference](#script-reference)
    - [Environment Variables](#environment-variables)
  - [Appendix: Edge Case Reference](#appendix-edge-case-reference)
    - [1. Silently Wrong Results](#1-silently-wrong-results)
    - [2. Incomplete Addresses That Still Resolve](#2-incomplete-addresses-that-still-resolve)
    - [3. Coordinates Supplied in Your Input](#3-coordinates-supplied-in-your-input)
    - [4. Street Text Normalization](#4-street-text-normalization)
    - [5. ZIP Field](#5-zip-field)
    - [6. Placeholders and Non-US Addresses](#6-placeholders-and-non-us-addresses)
    - [7. LOCATION\_HISTORY](#7-location_history)
    - [8. Rows That Abort the Entire Run](#8-rows-that-abort-the-entire-run)
---

## Overview

This workflow uses **two separate Docker containers** to take patient addresses or coordinates all the way to an analysis-ready exposure file:

1. **Exposome Geocoder Container (`prismaplab/exposome-geocoder:1.0.4`)**  
   Converts addresses or coordinates into OMOP `LOCATION` / `LOCATION_HISTORY` tables with latitude and longitude.

2. **Exposome Linkage Container**, built locally from [`Tools/postgis-exposure`](https://github.com/bihorac-LAB/EnvironmentalData/tree/main/Tools/postgis-exposure) in this repository  
   Spatially joins those tables with environmental and social determinant datasets (ADI, SVI, EJI, AHRQ) to produce `EXTERNAL_EXPOSURE.csv`. Build this image from the cloned repository for now; the published `ghcr.io/chorus-ai/chorus-postgis-exposure:main` tag has not yet been rebuilt against the current code, though it will be shortly.

The path is the same for every site:

```
Your data  →  Step 2: Address_to_LOCATION.py  →  LOCATION.csv          →  Step 4: linkage  →  EXTERNAL_EXPOSURE.csv
                                                 LOCATION_HISTORY.csv
```

> ⚠️ **Version note:** Use **`1.0.4` or later**. The `Address_to_LOCATION.py` script and the ZIP9/HUD crosswalk reference data it depends on are **not present in `1.0.3` or earlier**.

---

## Input Options
You need to prepare **only ONE** of the following data elements per encounter.  

### Option 1: Address
Sample input files [here](https://github.com/bihorac-LAB/EnvironmentalData/tree/main/Tools/demo/geocoding/input)

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

> ⚠️ **Required for every row:** a non-blank `location_id` (these are **not** auto-generated, so supply your own; they should be site specific), and address columns holding **either** a complete address **or**, at minimum, a ZIP code: the `street` / `city` / `state` / `zip` columns in Format A, or the single `address` column in Format B. The script exits with an error listing the offending rows if either condition is unmet.

---

### Option 2: OMOP CDM

If your source data already lives in an OMOP CDM database, extract the following tables first, then run [Step 2](#step-2-generate-location-tables) on the exported CSVs.

| Table              | Required Columns |
|--------------------|------------------------------------------------------|
| location           | location_id, address_1, address_2, city, state, zip, county, location_source_value, country_concept_id, country_source_value, latitude, longitude |
| location_history   | location_id, relationship_type_concept_id, domain_id, entity_id, start_date, end_date |

Sample input files: 
- [`LOCATION.csv`](https://github.com/bihorac-LAB/EnvironmentalData/blob/main/Tools/demo/geocoding/input/LOCATION.csv)  
- [`LOCATION_HISTORY.csv`](https://github.com/bihorac-LAB/EnvironmentalData/blob/main/Tools/demo/geocoding/input/LOCATION_HISTORY.csv)

> **Important**: Do not date-shift your `LOCATION` / `LOCATION_HISTORY` files before linkage. Date shifting, if used, should occur after linkage in [Step 6](#step-6-site-level-date-shifting-optional).

If these files are provided during **geocoding**, the output will automatically include the updated latitude and longitude information required for the linkage container. If they are **not** provided, `Address_to_LOCATION.py` builds both files for you from your input CSV, so no manual step is needed.

> **On dates.** `start_date` is taken from `start_date`, `visit_start_date`, or a 4-digit `year` (as `YYYY-01-01`), in that order. `end_date` is taken only from `end_date` or `visit_end_date`; it is **not** inferred from `year`. When a value cannot be derived it is left **blank** rather than filled with a placeholder, and the run logs a warning naming the affected rows.
---

## Usage Guide

### Step 1: Prepare Input Data
Prepare **only ONE** of the data elements as indicated under the [Input Options](#input-options) per encounter.  
For **Option 1 (Address)** or **Option 2 (OMOP)**, your data must be in a **CSV file** format. 

#### Folder Structure
- Place the CSV file(s) in a dedicated folder
  - 📂 `input_address/` 
> ⚠️ Only `.csv` files are supported. Convert `.xlsx` or other formats before running the tool.
---

### Step 2: Generate LOCATION Tables

**Container:** `prismaplab/exposome-geocoder:1.0.4`  
Ensure **Docker Desktop** is running.

This step produces the `LOCATION.csv` and `LOCATION_HISTORY.csv` that the linkage container consumes in [Step 4](#step-4-gis-linkage-with-postgis-exposure-tool).

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
- Open Command Prompt

Example, if your file is inside 📂`input_address/`:

```bash
docker run -it --rm   -v "%cd%":/workspace   -v /var/run/docker.sock:/var/run/docker.sock   -e HOST_PWD="%cd%"   -w /workspace   prismaplab/exposome-geocoder:1.0.4   /app/code/Address_to_LOCATION.py -i input_address
```

> ℹ️ The script launches the DeGAUSS geocoder in a nested Docker container, which is why the command mounts the Docker socket and passes `HOST_PWD`. All geocoding runs **locally**; no address data leaves your machine.

Coordinates are resolved through a four-tier fallback, stopping at the first tier that succeeds:

| Tier | Information available | Threshold | Interpretation |
|------|-----------------------|-----------|----------------|
| 1 | Latitude/longitude already supplied | n/a | No matching needed |
| 2 | Full street address | 0.7 | Requires a relatively strong address match |
| 3 | ZIP9 with block-level crosswalk | 0.3 | Lower-confidence geographic match |
| 4 | ZIP5 with ZIP-to-tract crosswalk | 0.1 | Least precise |

**What the threshold means.** The threshold is a similarity cutoff for deciding whether a geographic match is good enough to accept. It is passed internally to DeGAUSS, which scores each candidate it finds from 0 to 1 on how closely that candidate matches the information submitted, and returns nothing for a candidate scoring below the tier's cutoff, so the row falls through to the next tier. The cutoff drops as the information available gets coarser, because a ZIP-only query cannot match as precisely as a full street address. Override the defaults with `GEOCODER_THRESHOLD_ADDRESS`, `GEOCODER_THRESHOLD_ZIP9`, and `GEOCODER_THRESHOLD_ZIP5` (see [Environment Variables](#environment-variables)).

**What Tiers 3 and 4 do.** When no exact address is available, the script uses the ZIP information and a block-level geographic lookup to identify the most appropriate Census tract, then represents that tract with a point. The ZIP is checked against a crosswalk first, ZIP9 against the ZIP9 to FIPS12 crosswalk and ZIP5 against the HUD crosswalk, and the row is skipped if it does not validate.

The tier used for each row is recorded in the `modifier_source_value` column of `LOCATION.csv` (for example, `Level 2 | lat/long generated from address`), so every coordinate carries its own provenance.

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

Sample outputs: [demo/geocoding/output](https://github.com/bihorac-LAB/EnvironmentalData/tree/main/Tools/demo/geocoding/output)

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
| **ADI**: Area Deprivation Index | [UW-Madison (Zenodo)](https://doi.org/10.5281/zenodo.19475818) | 2015, 2020, 2023 | 6 |
| **SVI**: Social Vulnerability Index | [CDC/ATSDR](https://www.atsdr.cdc.gov/placeandhealth/svi/index.html) | 2010, 2014, 2016, 2018, 2020, 2022 | 651 |
| **EJI**: Environmental Justice Index | [CDC/ATSDR](https://www.atsdr.cdc.gov/place-health/php/eji/index.html) | 2022, 2024 | 239 |
| **AHRQ SDOH**: Social Determinants of Health | [AHRQ (Zenodo)](https://doi.org/10.5281/zenodo.19475914) | 2009–2023, annual | 4,195 |

---

#### Prerequisites for GIS Linkage
- Docker installed.
- A clone of this repository. All linkage commands run from the [`Tools/postgis-exposure`](https://github.com/bihorac-LAB/EnvironmentalData/tree/main/Tools/postgis-exposure) directory; [Step 0](#gis-linkage-workflow) below has the commands.
- `LOCATION.csv` and `LOCATION_HISTORY.csv` produced in [Step 2](#step-2-generate-location-tables).
- Ensure `DATA_SRC_SIMPLE.csv` and `VRBL_SRC_SIMPLE.csv` files are available (centrally managed; no edits required).
- **Important:** Do **not** date-shift your `LOCATION` / `LOCATION_HISTORY` files before linkage.

Sample `DATA_SRC_SIMPLE.csv` and `VRBL_SRC_SIMPLE.csv`: [here](https://github.com/bihorac-LAB/EnvironmentalData/tree/main/Tools/postgis-exposure/csv)

**How the ID numbers map to datasets.** The two centrally managed CSVs are the lookup tables for the numbers you pass in [Step 1](#gis-linkage-workflow) below:

| File | Column to use | Maps to |
|------|---------------|---------|
| [`VRBL_SRC_SIMPLE.csv`](https://github.com/bihorac-LAB/EnvironmentalData/blob/main/Tools/postgis-exposure/csv/VRBL_SRC_SIMPLE.csv) | `variable_source_id` → `VARIABLES` | One row per variable. `variable_name` is what lands in `exposure_source_value`; `dataset_type` says which family it belongs to (`ADI`, `SVI`, `EJI`, `AHRQ`). |
| [`DATA_SRC_SIMPLE.csv`](https://github.com/bihorac-LAB/EnvironmentalData/blob/main/Tools/postgis-exposure/csv/DATA_SRC_SIMPLE.csv) | `data_source_uuid` → `DATA_SOURCES` | One row per dataset **vintage** (dataset + year), with its download URL and documentation link. |

To confirm what a given ID is before requesting it, look the number up in the relevant file. For example, `variable_source_id` `96` resolves to its `variable_name` and `dataset_type` in `VRBL_SRC_SIMPLE.csv`; `data_source_uuid` `9922` is SVI's 2022 tract release in `DATA_SRC_SIMPLE.csv`.

---

#### Expected Outputs
- `EXTERNAL_EXPOSURE.csv` containing linked indices (ADI, SVI, EJI, AHRQ metrics).

Sample file [here](https://github.com/bihorac-LAB/EnvironmentalData/blob/main/Tools/demo/GIS-linkage/EXTERNAL_EXPOSURE.csv)

---

#### GIS Linkage Workflow

**Step 0: Get the code and stage your input files.** Clone this repository and change into the linkage directory:

```bash
git clone https://github.com/bihorac-LAB/EnvironmentalData.git
cd EnvironmentalData/Tools/postgis-exposure
```

Then copy `LOCATION.csv` and `LOCATION_HISTORY.csv` from your geocoder `output/` folder into `test/`. The container reads them from the mount; there is **no separate ingest command**.

```bash
cp /path/to/output/LOCATION.csv         ./test/
cp /path/to/output/LOCATION_HISTORY.csv ./test/
```

> ⚠️ **Every remaining command in this step runs from `Tools/postgis-exposure`.** The `docker build .` in Step 2 and the `-v ./test:/source` mount are both relative to it, so they fail or mount the wrong folder if you have changed directory.

**Step 1: Set the variable and data-source lists.**

```bash
export VARIABLES="96,98,100,102,110,112,116,118,120,122,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,156,157,159,197,199,201,203,211,213,217,254,256,258,260,268,270,274,276,278,280,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,314,315,317,319,334,360,362,364,366,374,378,380,382,384,386,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,422,423,470,496,498,500,502,510,514,516,518,520,522,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,558,559,592,593,594,595,601,603,611,612,613,614,615,616,617,618,622,623,632,633,674,675,676,677,678,681,685,687,689,691,692,693,694,695,696,717,718,719,720,721,722,723,724,729,730,739,740,831,832,833,834,835,836,1795,1796,1797,1798,1799,1800,1801,1802,1803,1804,1805,1806,1807,1808,1809,1810,1811,1812,1813,1814,1815,1816,1817,1818,1819,1820,1821,1822,1823,1824,1825,1826,1827,1828,1829,1830,1831,1832,1833,1834,1835,1836,1837,1838,1839,1840,1841,1842,1843,1844,1845,1846,1847,1848,1849,1850,1851,1852,1853,1854,2322,2323,2324,2325,2326,2327,2328,2329,2330,2331,2332,2333,2334,2335,2336,2421,2422,2423,2424,2425,2426,2427,2428,2429,2430,2431,2432,2433,2434,2435,2436,2437,2438,2439,2440,2441,2442,2443,2444,2445,2446,2447,2448,2449,2583,2584,2585,2586,2587,2588,2589,2590,2591,2592,2593,2594,2595,2596,2597,3214,3215,3216,3217,3218,3219,3220,3221,3222,3223,3224,3225,3226,3227,3228,3229,3230,3231,3232,3233,3234,3235,3831,3832,3833,3834,3835,3836,3837,3838,3839,3840,3841,3842,3843,3844,3845,3846,3847,3848,3849,3850,3851,3852,3853,3854,3855,3856,3857,3858,3859,4179,4180,4181,4182,4183,4184,4185,4186,4187,4188,4189,4190,4191,4192,4221,4222,4223,4224,4225,4226,4227,4228,4229,4230,4231,4232,4233,4234,4235,4236,4237,4238,4239,4240,4241,4242,4243,4244,4245,4246,4247,4248,4249,4315,4316,4317,4318,4319,4320,4321,4322,4323,4324,4325,4326,4327,4328,4329,4352,4353,4354,4355,4356,4357,4358,4359,4360,4361,4362,4363,4364,4365,4366,4367,4368,4369,4370,4371,4372,4373,4374,4375,4376,4377,4378,4379,4380,4381,4382,4383,4384,4385,4386,4387,4388,4957,4958,4959,4960,4961,4962,4963,4964,4965,4966,4967,4968,4969,4970,4971,5032,5033,5034,5035,5036,5037,5038,5039,5040,5041,5042,5043,5044,5045,5046,5047,5048,5049,5050,5051,5052,5053,5054,5055,5056,5057,5058,5059,5060,5061,5062,5063,5064,5065,5066,5067,5068,5069,5070,5071,5072,5073,5074,5075,5076,5077,5078,5079,5080,5081,5082,5083,5084,5085,5086,5087,5088,5089,5090,5091"
export DATA_SOURCES="7700,9910,9914,9916,9918,9920,9922,8822,8824,10515,10520,10523,11209,11210,11211,11212,11213,11214,11215,11216,11217,11218,11219,11220,11221,11222,11223"
```

> These are the full canonical lists, reproduced verbatim from the [postgis-exposure README](https://github.com/bihorac-LAB/EnvironmentalData/blob/main/Tools/postgis-exposure/README.md#deploy), which remains the authoritative source. If you are re-running after a while, check there for the current set. `VARIABLES` IDs come from `VRBL_SRC_SIMPLE.csv`; `DATA_SOURCES` IDs from `DATA_SRC_SIMPLE.csv`.

**Step 2: Build the image and start the Postgres/PostGIS container.**

Build the image, then start the container. Both commands run from `Tools/postgis-exposure` and reuse the `VARIABLES` and `DATA_SOURCES` exports from Step 1.

```bash
docker build -t chorus-postgis-exposure-local .

docker run --rm --name postgis-chorus \
    --env POSTGRES_PASSWORD="dummy" \
    --env VARIABLES="$VARIABLES" \
    --env DATA_SOURCES="$DATA_SOURCES" \
    -v ./test:/source \
    -d chorus-postgis-exposure-local:latest
```

> ℹ️ **Build locally rather than pulling the published image.** `ghcr.io/chorus-ai/chorus-postgis-exposure:main` has not yet been rebuilt against the current code, so it may not match what this guide describes. A refreshed image is expected soon; once it lands you can swap `-d chorus-postgis-exposure-local:latest` for `-d ghcr.io/chorus-ai/chorus-postgis-exposure:main` and drop the `docker build` line. Every later step is identical either way.

> `POSTGRES_PASSWORD=dummy` is safe as-is: the database is local to this throwaway container and is never exposed off-host.

This brings up a Docker container locally with all dependencies needed to run the dataset retrieval and spatial joining processes.

**Step 3: Wait for the database to come up** (10-20 seconds depending on your environment). Confirm with:

```bash
docker logs postgis-chorus
```

Wait until you see **`database is ready to accept connections`** before continuing.

**Step 4: Generate the external exposure file.**

```bash
docker exec postgis-chorus /app/produce_external_exposure.sh
```

This retrieves the data sources and combines them with your data to produce the external exposure table.

**Step 5: Collect the output.** `EXTERNAL_EXPOSURE.csv` will appear in your mounted `./test` directory.

**Step 6: Stop the container.**

```bash
docker stop postgis-chorus
```

#### Notes & Tips
- Run these commands in Terminal (Mac) or WSL/PowerShell/Command Prompt on Windows; WSL is more robust for Docker on Windows.
- Run all commands from the `Tools/postgis-exposure` directory; the `-v ./test:/source` mount is relative to it.
- If your site needs more variables, expand `VARIABLES` accordingly.
- **Important**: The container may only run successfully once. To rerun, you may need to delete the container and image, then rebuild.

---

### Step 5: Validate & Inspect Outputs

`EXTERNAL_EXPOSURE.csv` is in **long format** (one row per location, person, variable, and year), not one row per patient. Sample output: [demo/GIS-linkage](https://github.com/bihorac-LAB/EnvironmentalData/tree/main/Tools/demo/GIS-linkage).

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
- **Container documentation:** [postgis-exposure README](https://github.com/bihorac-LAB/EnvironmentalData/blob/main/Tools/postgis-exposure/README.md): authoritative source for the linkage container's deploy commands and the current `VARIABLES` / `DATA_SOURCES` lists.
- Sample files: [PostGIS Exposure CSVs](https://github.com/bihorac-LAB/EnvironmentalData/tree/main/Tools/postgis-exposure/csv)
  - **Site-specific:** `LOCATION`, `LOCATION_HISTORY`
  - **Centrally managed:** `DATA_SRC_SIMPLE`, `VRBL_SRC_SIMPLE`
- Sample linkage output: [demo/GIS-linkage](https://github.com/bihorac-LAB/EnvironmentalData/tree/main/Tools/demo/GIS-linkage)

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

## Appendix: Geocoding Workflow

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

## Appendix: Edge Case Reference

How `Address_to_LOCATION.py` (`1.0.4`) behaves on real-world input.

### 1. Silently Wrong Results

⚠️ Every case below is **reported as a successful geocode**. Nothing downstream flags them.

| Case | Example | What actually happens |
|------|---------|-----------------------|
| `0, 0` coordinates | `latitude = 0, longitude = 0` | Accepted as **Level 1**. |
| Latitude and longitude swapped | `latitude = -82.3441, longitude = 29.6392` | Accepted as **Level 1** |
| Impossible latitude | `latitude = 95.1234, longitude = -82.6403` | Rejected at first, then recycled into its ZIP's average and re-emitted as **Level 4 with `95.1234` still attached**. |
| Full state name instead of code | `state = "Georgia"` → `GE`; `"Mississippi"` → `MI` | The first two letters are kept blindly. `GE` is not a state, so the ZIP+4 lookup fails for that row; `MI` is Michigan. **Send two-letter codes.** |
| Mixed date formats in one column | `2018-03-15` and `03/15/2018` | The format of the **first row** wins; every date not matching it is blanked (logged). Which rows survive depends on row order, so re-sorting the file changes the result. **Use `YYYY-MM-DD` throughout.** |
| `end_date` before `start_date` | `start_date = 2021-01-01`, `end_date = 2019-01-01` | Accepted silently; order is never checked. Any exposure window built from that row is empty. |
| Orphan `location_id` | `9999` present in `LOCATION_HISTORY.csv` only | Accepted and written straight to output. Referential integrity against `LOCATION.csv` is not checked. |
| Spelled-out or misspelled street suffix | `1250 West 16th Streett` | The address parser runs **before** the spelling fixes, so the suffix lands in the wrong field: street → `1250 W 16th`, city → `Street Jacksonville`. `12345 State Highway 26` → street `12345 26`. Coordinates usually still arrive via ZIP, but the cleaned address in the output is wrong. Sending `St` / `Ave` / `Blvd` / `Hwy` avoids this entirely. |

### 2. Incomplete Addresses That Still Resolve

The tool tries the full address first, then falls back to coarser location information. An incomplete address is usually not fatal.

| Case | Example | Outcome |
|------|---------|---------|
| ZIP only, no street or city | `state = "FL", zip = "33414"` | Accepted → ZIP centroid, labelled `Level 4 \| lat/long generated from zip5`. |
| Street present, city missing | `address_1 = "3415 sw 39th blvd", state = "FL", zip = "32608"` | Accepted. Street normalized to `3415 SW 39th Blvd`. |
| Street with no house number | `"Millhopper Rd", Gainesville, 32653` | Imprecise match, so the row drops to ZIP level rather than failing. |
| PO Box | `"PO Box 1234"` | Cannot match a building → falls back to ZIP level. |
| Intersection | `"Main St & 1st Ave"` | Splits badly (`Ave Jacksonville` ends up in city). ZIP-level result. |

### 3. Coordinates Supplied in Your Input

| Case | Example | Outcome |
|------|---------|---------|
| Non-numeric values | `latitude = "N/A", longitude = "unknown"` | Both ignored; row falls through to normal address geocoding. Correct behaviour. |
| Only one of the pair | `latitude = 25.7617, longitude = ""` | Ignored, falls through. |

### 4. Street Text Normalization

Works as intended. ALL CAPS, all lowercase, spelled-out suffixes, and common typos are corrected.

| Input | Becomes | Note |
|-------|---------|------|
| `8441 HELEN TERRACE` | `8441 Helen Ter` | Case and suffix normalized |
| `2060 Continental Ave, Sutie 200` | `… Ste 200` | Common misspellings corrected |
| `1200 O'Brien Rd` | `1200 O Brien Rd` | Apostrophes and periods removed; fine for geocoding, but affects later exact-text matching |
| `1234 Ponce de León Blvd` | `1234 Ponce De León Blvd` | Accented characters preserved |

### 5. ZIP Field

| Case | Example | Outcome |
|------|---------|---------|
| ZIP+4 (best ZIP-level result) | `zip = "32608-1234"` | Split into ZIP5 `32608` and ZIP9 `326081234`. If present in the reference data the row resolves at **Level 3**, more precise than a plain ZIP. Hyphen optional. |
| ⚠️ Leading zero lost in Excel | `zip = "2134"` (should be `02134`, Boston) | At least five digits are required and **no padding is applied**, so the row **fails completely**. The single most common ZIP problem. **Export ZIP as text, not as a number**. |
| Exported as a number with a decimal | `zip = "32608.0"` | Recovered correctly to `32608`, though a stray `0` ends up in `address_2`. |
| Well-formed but nonexistent | `zip = "00000"` or `"99999"` | Passes the five-digit format check but is absent from the reference data → failure report reason `ZIP5 not found in HUD crosswalk`. |

### 6. Placeholders and Non-US Addresses

| Case | Example | Outcome |
|------|---------|---------|
| Placeholder words instead of values | `address_1 = "Not Stated"`, `city = "UNKNOWN"`, `city = "N/A"` | `Not Stated`, `Unknown`, `N/A`, `None`, `Null`, `Missing` and similar are recognized as placeholders when explaining a failure. The row **can still receive a ZIP-level position**, so a placeholder address does not necessarily mean a failed row. A genuinely blank field is cleaner than writing `Unknown` into it. |
| Address outside the United States | `"483 Bay St, Toronto, ON, M5V 3L9"` | The postal code contains only three digits, so it is discarded, and the geocoder covers the US only. Row fails. Non-US addresses are out of scope today. |

### 7. LOCATION_HISTORY

| Case | Example | Outcome |
|------|---------|---------|
| One person, several addresses over time | `entity_id 5` at location `5` (`2015-01-01` → `2018-01-01`), then location `14` (`2018-01-01` → blank) | Works as intended; this is the table's primary purpose. Leave `end_date` blank for the current address. |
| Impossible or missing date | `start_date = "2021-13-45"` or `""` | Both end up blank in the output; the log warns and gives the row numbers. |
| Blank or junk concept ids | `relationship_type_concept_id = ""` or `"abc"`; `domain_id = ""` | Filled with the defaults `32848` and `1147314`, with no warning. Safe to leave blank if you do not have them. |

### 8. Rows That Abort the Entire Run

Two checks run **before** any geocoding. They abort the whole job rather than skipping the offending row, so one bad row means **no output at all**.

| Case | Example | Outcome |
|------|---------|---------|
| Blank `location_id` | `location_id = ""` (address and ZIP fine) | Run stops: `ALERT: Input must include non-blank location_id for every row. Row numbers: 2`. IDs need not be numeric. |
| No address **and** no ZIP on the same row | `address_1 = ""` and `zip = ""` | Run stops: `ALERT: Each input row must include either address or zip for LOCATION output generation. Row numbers: 2`. Either field alone is enough. |
| No `location_id` **column** at all | a raw extract whose header is `street,city,state,zip,year,entity_id` | Every row reads as blank `location_id`, so the first check stops the run and nothing is produced. Add the column before running. The demo inputs [`multi_column_address_data.csv`](https://github.com/bihorac-LAB/EnvironmentalData/blob/main/Tools/demo/geocoding/input/multi_column_address_data.csv) and [`single_column_address_data.csv`](https://github.com/bihorac-LAB/EnvironmentalData/blob/main/Tools/demo/geocoding/input/single_column_address_data.csv) each carry `location_id` as their first column for exactly this reason. |

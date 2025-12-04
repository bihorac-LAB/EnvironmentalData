# Exposome Geocoder – Input Preparation and Usage Guide

> **Note:** This toolkit does **not** require or share any Protected Health Information (PHI).

This repository provides a **reproducible workflow** to geocode patient location data and link the resulting Census Tract (FIPS 11-digit) identifiers with Exposome datasets for environmental exposure analysis.

> **Demo video** [Watch here](https://drive.google.com/drive/folders/14FdY0lB3WRIiYrgCdeje6NAQI7Q_RB2w?usp=sharing)
---
## 📑 Table of Contents
- [Geocoding Patient Data for Exposome Linkage](#geocoding-patient-data-for-exposome-linkage)
  - [📑 Table of Contents](#-table-of-contents)
  - [Overview](#overview)
  - [Input Options](#input-options)
    - [Option 1: Address](#option-1-address)
    - [Option 2: Coordinates](#option-2-coordinates)
    - [Option 3: OMOP CDM](#option-3-omop-cdm)
  - [Usage Guide](#UsageGuideForGeocoding)
    - [Step 1: Prepare Input Data](#step-1-prepare-input-data)
    - [Step 2: Generate FIPS Codes](#step-2-generate-fips-codes)
      - [For CSV Input (Option 1 & 2)](#for-csv-input-option-1--2)
      - [For OMOP Input (Option 3)](#for-omop-input-option-3)
    - [Step 3: Output Structure](#step-3-output-structure)
      - [CSV Input (Option 1 & 2)](#csv-input-option-1--2)
      - [OMOP Input (Option 3)](#omop-input-option-3)
    - [Step 4: GIS Linkage with PostGIS-Exposure Tool](#step-4-gis-linkage-with-postgis-exposure-tool)
      - [Prerequisites for GIS Linkage](#prerequisites-for-gis-linkage)
      - [Expected Outputs](#expected-outputs)
      - [GIS Linkage Workflow](#gis-linkage-workflow)
      - [Notes \& Tips](#notes--tips)
    - [Step 5: Validate \& Inspect Outputs](#step-5-validate--inspect-outputs)
    - [Step 6: Optional - Site-level Date Shifting](#step-6-optional---site-level-date-shifting)
    - [Step 7: Upload \& Centralized De-identification](#step-7-upload--centralized-de-identification)
    - [Step 8: Link with Exposome Web Platform](#step-8-link-with-exposome-web-platform)
  - [Appendix](#appendix)
    - [Geocoding Workflow](#geocoding-workflow)
      - [Method: DeGAUSS Toolkit (Docker-based)](#method-degauss-toolkit-docker-based)
      - [DeGAUSS Docker Commands (Executed Internally)](#degauss-docker-commands-executed-internally)
      - [Script Highlights](#script-highlights)
        - [Address\_to\_FIPS.py Logic](#address_to_fipspy-logic)
        - [OMOP\_to\_FIPS.py Logic](#omop_to_fipspy-logic)
---

## Overview
This workflow uses **two separate Docker containers** to support end-to-end geocoding and data linkage:

1. **Exposome Geocoder Container (`prismaplab/exposome-geocoder:1.0.3`)**  
   Performs address or coordinate-based geocoding to generate Census Tract (FIPS 11-digit) codes using [DeGAUSS](https://degauss.org) backend tools.

2. **Exposome Linkage Container (`ghcr.io/chorus-ai/chorus-postgis-sdoh:main`)**  
   Integrates the geocoded outputs with relevant environmental and social determinant datasets to produce analysis-ready files.

Together, these containers enable:  
- Address and latitude/longitude-based geocoding  
- OMOP CDM geocoding extraction and processing  
- GIS linkage with PostGIS-SDoH indices (ADI, SVI, AHRQ)

---

## Input Options
You need to prepare **only ONE** of the following data elements per encounter.  

### Option 1: Address
Sample input files [here](https://github.com/bihorac-LAB/EnvironmentalData/tree/main/Tools/demo/address_files/input)

- **Format A: Multi-Column Address**

| street           | city         | state | zip   | year | entity_id |
|------------------|--------------|-------|-------|------|----------------|
| 1250 W 16th St   | Jacksonville | FL    | 32209 | 2019 | 1              |
| 2001 SW 16th St  | Gainesville  | FL    | 32608 | 2019 | 2              |

> **Tip:** Street **and** ZIP are required. Missing these fields may lead to **imprecise geocoding**.

- **Format B: Single Column Address**

| address                                      | year | entity_id |
|----------------------------------------------|------|----------------|
| 1250 W 16th St Jacksonville FL 32209         | 2019 | 1              |
| 2001 SW 16th St Gainesville FL 32608         | 2019 | 2              |

---

#### Optional Supporting Files

Including the following optional files will help streamline the **end-to-end workflow** between geocoding and exposome linkage:
- **Important**: Do not date-shift your LOCATION/LOCATION_HISTORY files before linkage. Date shifting (if used) should occur post linkage in Step 4.
- 
- [`LOCATION.csv`](https://github.com/bihorac-LAB/EnvironmentalData/blob/main/Tools/demo/address_files/input/LOCATION.csv)  
- [`LOCATION_HISTORY.csv`](https://github.com/bihorac-LAB/EnvironmentalData/blob/main/Tools/demo/address_files/input/LOCATION_HISTORY.csv)

If these files are provided during **geocoding**, the output will automatically include the updated latitude and longitude information required for the **postgis linkage container**.  

If they are **not provided**, users will need to **manually update their LOCATION files** with the geocoded latitude/longitude before executing the commands for linkage.

##### LOCATION.csv (Follows CDM format)

| location_id | address_1 | address_2 | city | state | zip | county | location_source_value | country_concept_id | country_source_value | latitude | longitude |
|-------------|-----------|-----------|------|-------|-----|--------|----------------------|-------------------|---------------------|----------|-----------|
| 1           | 1248 N Blackstone Ave | | FRESNO | CA | 93703 | | UNITED STATES OF AMERICA | | UNITED STATES OF AMERICA | 36.75891146 | -119.7902719 |

##### LOCATION_HISTORY.csv (Follows CDM format)

| location_id | relationship_type_concept_id | domain_id | entity_id | start_date | end_date |
|-------------|------------------------------|-----------|-----------|------------|----------|
| 1           | 32848                        | 1147314   | 3763      | 1998-01-01 | 2020-01-01 |

---

### Option 2: Coordinates

Sample input files [here](https://github.com/bihorac-LAB/EnvironmentalData/tree/main/Tools/demo/latlong_files/input)

| latitude   | longitude | entity_id | year
|------------|-----------|-----------|-------
| 30.353463  | -81.6749  | 1         | 2015
| 29.634219  | -82.3433  | 2         | 2015

As with address-based input, including `LOCATION.csv` and `LOCATION_HISTORY.csv` enables seamless downstream processing with the linkage container.

---

### Option 3: OMOP CDM

| Table              | Required Columns |
|--------------------|------------------------------------------------------|
| person             | person_id                                            |
| visit_occurrence   | visit_occurrence_id, visit_start_date, visit_end_date, person_id |
| location           | location_id, address_1, address_2, city, state, zip, location_source_value, country_concept_id, country_source_value, latitude, longitude |
| location_history   | location_id, relationship_type_concept_id, domain_id, entity_id, start_date, end_date |

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

### Step 2: Generate FIPS Codes

**Container:** `prismaplab/exposome-geocoder:1.0.3`  
Ensure **Docker Desktop** is running.  

This step uses the Exposome Geocoder container to:
- Convert addresses or coordinates into latitude/longitude
- Assign 11-digit Census Tract (FIPS) codes

#### For CSV Input (Option 1 & 2)

##### For macOS / Linux / Ubuntu

```bash
docker run -it --rm \
  -v "$(pwd)":/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e HOST_PWD="$(pwd)" \
  -w /workspace \
  prismaplab/exposome-geocoder:1.0.3 \
  /app/code/Address_to_FIPS.py -i <input_folder_path>
```

##### For Windows
- Open Command Prompt or powershell
- Run command `wsl`
- Execute the same command as above inside your WSL terminal.

Example:

If your file is named patients_address.csv inside 📂`input_address/`, run:

```bash
docker run -it --rm   -v "$(pwd)":/workspace   -v /var/run/docker.sock:/var/run/docker.sock   -e HOST_PWD="$(pwd)"   -w /workspace   prismaplab/exposome-geocoder:1.0.3   /app/code/Address_to_FIPS.py -i input_address
```

#### For OMOP Input (Option 3)
To extract and geocode directly from an OMOP database:

```bash
docker run -it --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$(pwd)":/workspace \
  -e HOST_PWD="$(pwd)" \
  -w /workspace \
  prismaplab/exposome-geocoder:1.0.3 \
  /app/code/OMOP_to_FIPS.py \
    --user <your_username> \
    --password <your_password> \
    --server <server_address> \
    --port <port_number> \
    --database <database_name>
```
---

### Step 3: Output Structure
After running the geocoder container (for Option 1, 2, or 3), the tool generates output files in the `output/` folder.

#### CSV Input (Option 1 & 2)
Sample outputs [demo/address_files/output](https://github.com/bihorac-LAB/EnvironmentalData/tree/main/Tools/demo/address_files/output)

**Files Generated**
Each input file produces:
  - `<filename>_with_coordinates.csv` — input + latitude/longitude  
  - `<filename>_with_fips.csv` — input + FIPS codes  

#### **Output Folder Example**
  ```
  output/
  ├── coordinates_from_address_<timestamp>.zip
  ├── geocoded_fips_codes_<timestamp>.zip
  ```
> `<timestamp>` indicates when the script was executed (e.g., 20250624_150230).

If `LOCATION.csv` and `LOCATION_HISTORY.csv` were included, they are copied to `output/` but not zipped.

**Zipped Output Columns Description**

| Column           | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `Latitude`       | Latitude returned from the geocoder                                         |
| `Longitude`      | Longitude returned from the geocoder                                        |
| `geocode_result` | Outcome of geocoding — `geocoded` for successful matches, `Imprecise Geocode` if not precise |
| `reason`         | Failure reason if applicable (see [Reason Column Values](#reason-column-values)) |

##### Reason Column Values
Used when geocoding fails or is imprecise. Possible values include:

- **Hospital address given** – Detected from known hardcoded hospital addresses.  
- **Street missing** – No street info provided.  
- **Blank/Incomplete address** – Address is empty or has missing components.  
- **Zip missing** – ZIP code not provided.  

> 💡 **Tip:** You can expand hospital detection by adding known addresses to `HOSPITAL_ADDRESSES` in [`Address_to_FIPS.py`](https://github.com/bihorac-LAB/EnvironmentalData/blob/main/Tools/code/Address_to_FIPS.py).

**Formatting Note for `HOSPITAL_ADDRESSES`:**
  - Single-line string  
  - Lowercase letters and numbers only  
  - No commas or special characters  
  - Fields separated by single spaces  
  
---

#### OMOP Input (Option 3)
**Sample outputs:** [demo/OMOP/output](https://github.com/bihorac-LAB/EnvironmentalData/tree/main/Tools/demo/OMOP/output)

#### **Folder Structure**

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
---

### Step 4: GIS Linkage with PostGIS-Exposure Tool

**Purpose:**  
Spatially joins the lat/lon (and FIPS) from geocoding with geospatial indices (ADI, SVI, AHRQ) and produces `EXTERNAL_EXPOSURE.csv`.

---

#### Prerequisites for GIS Linkage
- Docker installed.
- Clone [postgis-exposure repository](https://github.com/chorus-ai/chorus-container-apps/tree/main/postgis-exposure)
- Updated `LOCATION`, `LOCATION_HISTORY` files to include the geocoded lat/lon from Step 2. Not needed if you included these during the geocoding step
- Ensure `DATA_SRC_SIMPLE.csv` and `VRBL_SRC_SIMPLE.csv` files are available (centrally managed; no edits required).
- **Important:** Do **not** date-shift your `LOCATION`/`LOCATION_HISTORY` files before linkage. Date shifting (if used) should occur following this step.

Sample `DATA_SRC_SIMPLE.csv` and `VRBL_SRC_SIMPLE.csv`: [here](https://github.com/chorus-ai/chorus-container-apps/tree/main/postgis-exposure/csv)

---

#### Expected Outputs
- `EXTERNAL_EXPOSURE.csv` containing linked indices (ADI, SVI, AHRQ metrics).

---

#### GIS Linkage Workflow

1. **Start Postgres/PostGIS container** following the instructions in the [postgis-exposure repository](https://github.com/chorus-ai/chorus-container-apps/tree/main/postgis-exposure).
   Container sequence: start/load database → ingest location tables → run the produce script.
   **First Docker command (prepares the database):**

   ```bash
   docker run --rm --name postgis-chorus \
     --env POSTGRES_PASSWORD=dummy \
     --env VARIABLES=134,135,136 \
     --env DATA_SOURCES=1234,5150,9999 \
     -v $(pwd)/test/source:/source \
     -d ghcr.io/chorus-ai/chorus-postgis-sdoh:main
   ```
   - Replace `VARIABLES` with the comma-separated list of variable IDs you need from `VRBL_SRC_SIMPLE.csv`.
   - Replace `DATA_SOURCES` with the relevant data source IDs (from `DATA_SRC_SIMPLE.csv`).

2. ** Generate the external exposure file:**

   ```bash
   docker exec postgis-chorus /app/produce_external_exposure.sh
   ```

3. **Output:** `EXTERNAL_EXPOSURE.csv` will appear in your mounted directory (e.g., `./test/source`).

#### Notes & Tips
- Run these commands in Terminal (Mac) or WSL/PowerShell/Command Prompt on Windows; WSL is more robust for Docker on Windows.
- If your site needs more variables, expand `VARIABLES` accordingly.
- **Important**: The container may only run successfully once. To rerun, you may need to delete the container and image, then pull the image again.

---

### Step 5: Validate & Inspect Outputs
- Open `EXTERNAL_EXPOSURE.csv`. Confirm:
  - Patient ID, lat, lon, FIPS
  - ADI, SVI, AHRQ, and VRBL-coded fields
- Spot-check a few records for accuracy.
- If errors:
  - Ensure `LOCATION` has valid lat/lon/FIPS
  - Confirm `VARIABLES` and `DATA_SOURCES` are correct
  - Check mount paths

---

### Step 6: Optional - Site-level Date Shifting
**Purpose:** Anonymize temporal data while preserving relative timelines.

**Guidelines:**
- Apply date shifts locally before upload — do not date-shift prior to GIS linkage.
- Input: `EXTERNAL_EXPOSURE.csv` (from Step 4)
- Output: `EXTERNAL_EXPOSURE_date_shifted.csv`

See [Date Shifting SOP for More Details](https://github.com/chorus-ai/Chorus_SOP/blob/main/sop-website/docs/Privacy/Date-Shifting.mdx).

---

### Step 7: Upload & Centralized De-identification
1. Upload the (optionally date-shifted) `EXTERNAL_EXPOSURE.csv` to the central repository.
2. The central team will apply further de-identification.

---

### Step 8: Link with Exposome Web Platform (internal to UF)
1. Register at [https://exposome.rc.ufl.edu](https://exposome.rc.ufl.edu/)  
2. Upload `*_with_fips.zip` file obtained from Step 3 
3. Input CSV must contain:  
   - `person_id`  
   - `visit_occurrence_id`  
   - `year`  
   - `FIPS`
4. Select the dataset you want to link it to
5. Download enriched dataset with SDoH variables

---
### References & sample files
#### Geocoding
- Sample files: [Geocoding Demo Files](https://github.com/bihorac-LAB/EnvironmentalData/tree/main/Tools/demo)

#### GIS Linkage
- Sample files: [PostGIS Exposure CSVs](https://github.com/chorus-ai/chorus-container-apps/tree/main/postgis-exposure/csv)
  - **Site-specific:** `LOCATION`, `LOCATION_HISTORY`
  - **Centrally managed:** `DATA_SRC_SIMPLE`, `VRBL_SRC_SIMPLE`
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

## Appendix

### Geocoding Workflow
This guide outlines the scripts, workflows, and Docker-based DeGAUSS toolkit used for generating Census Tract (FIPS) information from patient data.
To convert patient location data into Census Tract identifiers (**FIPS11**), we use a two-step geocoding process powered by [DeGAUSS](https://degauss.org), executed locally via Docker containers.

#### Method: DeGAUSS Toolkit (Docker-based)

DeGAUSS consists of two Docker containers:

1. **Geocoder (3.3.0)** — Converts address to latitude/longitude  
2. **Census Block Group (0.6.0)** — Converts latitude/longitude to Census Tract FIPS codes  

| Step | Purpose                  | Docker Image                                     |
|------|--------------------------|--------------------------------------------------|
| 1    | Address → Coordinates    | `ghcr.io/degauss-org/geocoder:3.3.0`             |
| 2    | Coordinates → FIPS       | `ghcr.io/degauss-org/census_block_group:0.6.0`   |

---

#### DeGAUSS Docker Commands (Executed Internally)

```bash
# Step 1: Get Coordinates from Address
docker run --rm -v "ABS_OUTPUT_FOLDER:/tmp" \
  ghcr.io/degauss-org/geocoder:3.3.0 \
  /tmp/<your_preprocessed_input.csv> <threshold>

# Step 2: Get FIPS from Coordinates
docker run --rm -v "ABS_OUTPUT_FOLDER:/tmp" \
  ghcr.io/degauss-org/census_block_group:0.6.0 \
  /tmp/<your_coordinate_output.csv> <year>
```

**Replace values:**
- `ABS_OUTPUT_FOLDER` → absolute path to your output directory  
- `<threshold>` → numeric value (e.g., `0.7`)  
- `<year>` → either `2010` or `2020`  

--- 

#### Script Highlights

##### Address_to_FIPS.py Logic
This [script](https://github.com/bihorac-LAB/EnvironmentalData/blob/main/Tools/code/Address_to_FIPS.py) handles CSV-based input:
- Reads CSV files
- Normalizes address or uses lat/lon
- Runs DeGAUSS Docker container to generate:
     - Latitude/Longitude (via `ghcr.io/degauss-org/geocoder`)
     - FIPS codes(via `ghcr.io/degauss-org/census_block_group`)
- Packages outputs into ZIP

##### OMOP_to_FIPS.py Logic
This [script](https://github.com/bihorac-LAB/EnvironmentalData/blob/main/Tools/code/OMOP_to_FIPS.py) integrates directly with **OMOP CDM**: 
- Extracts OMOP CDM data
- Categorizes into valid/invalid address or coordinates
- Executes FIPS generation (same as CSV workflow) 
- Packages outputs into ZIP
---

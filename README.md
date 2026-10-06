# Environmental Data Linkage Toolkit

A privacy-preserving, reproducible pipeline to geocode patient location data into OMOP CDM `LOCATION` / `LOCATION_HISTORY` formats, spatially link them with environmental and Social Determinants of Health (SDoH) exposure datasets, and validate geocoding accuracy against national benchmarks.

Developed by the **Bihorac Lab / PRISMAp** at the University of Florida.

---

## Architecture Overview

Clinical studies increasingly rely on environmental exposures (such as air quality, temperature, walkability) and Social Determinants of Health (such as ADI, SVI, EJI, AHRQ). Because patient address data constitutes Protected Health Information (PHI), all geocoding and spatial processing in this toolkit executes **entirely on your local infrastructure**. No addresses or patient identifiers are ever sent to external cloud APIs during production runs.

```
+-------------------------------------------------------------------------------------------------------+
|                                        DATA & EXECUTION LINEAGE                                       |
+-------------------------------------------------------------------------------------------------------+

  [ Raw Input Addresses ]  ---> Placed in input_address/ (or extracted from OMOP CDM)
          |
          v
  [ Step 1: Geocoding Engine ]  <--- Tools/code/Address_to_LOCATION.py
   * Runs inside Docker: prismaplab/exposome-geocoder:1.0.4
   * Calls local DeGAUSS 3.3.0 container via Docker socket
   * Resolves: Street-Level -> City/ZIP -> ZIP Centroid Fallback
          |
          v
  [ Standardized Location Tables ]  ---> Written to output/
   * LOCATION.csv (OMOP CDM v5.4 compatible: location_id, latitude, longitude, county)
   * LOCATION_HISTORY.csv (Temporal linkage: entity_id, start_date, end_date)
   * log/ (Full runtime and container execution logs)
          |
          v
  [ Step 2: PostGIS Exposure Linkage ]  <--- Tools/postgis-exposure/
   * Loads shapefiles & SDoH databases (ADI, SVI, EJI, AHRQ, USDA)
   * Executes spatial point-in-polygon joins
          |
          v
  [ Final Research Output ]
   * EXTERNAL_EXPOSURE.csv (OMOP-linked environmental covariates)
          |
          +--------------------------------------------------------------------------+
          | Validation Benchmarking Pipeline                                         |
          | Script: geocoding_comparison.py                                          |
          | Benchmarks local Exposome results against U.S. Census Bureau API        |
          | Input:  nad_encounters_sample.csv / input_address/                       |
          | Chunks: census_output/ (Census API batch responses)                      |
          | Tracts: exposome_output/ (Exposome coordinates resolved to tracts)      |
          | Report: results/ (comparison_results.csv, summary.json, report.html)   |
          +--------------------------------------------------------------------------+
```

---

## Repository Directory Structure & File Lineage

The tree below details every directory and key file across the repository.

```
EnvironmentalData/
├── .gitignore                          # Industry-standard git ignore rules (Python, OS, caches)
├── LICENSE                             # MIT License
├── README.md                           # Master repository documentation (this file)
├── geocoding_comparison.py             # Validation benchmark script (Census API vs Exposome container)
├── nad_encounters_sample.csv           # 2,000 reference addresses for benchmarking
├── requirements.txt                    # Python dependencies for validation and utilities
├── assets/                             # Architecture diagrams and system workflows
│   ├── Database Design.jpeg            # Schema model for environmental exposure database
│   └── External Investigator Workflow.png # End-to-end investigator process diagram
├── census_output/                      # U.S. Census Bureau API benchmark outputs and chunks
│   ├── census_results.csv              # Batch geocoded results from Census API
│   └── nad_census_batch_input.csv      # Formatted batch input sent to Census API
├── exposome_output/                    # Exposome coordinates resolved to Census tracts
│   ├── LOCATION.csv                    # Exposome-geocoded coordinates
│   ├── exposome_with_tracts.csv        # Evaluated tracts mapped from Exposome coordinates
│   └── coord_tract_cache.json          # Cached coordinate-to-tract lookups
├── input_address/                      # Working input folder for Exposome container runs
│   └── nad_exposome_input.csv          # Standardized address input for geocoder container
├── output/                             # Production outputs from Address_to_LOCATION.py
│   ├── LOCATION.csv                    # OMOP CDM Location table (latitude, longitude, county)
│   ├── LOCATION_HISTORY.csv            # OMOP CDM Location History table
│   ├── geocoding_summary_*.csv         # Match rate metrics and resolution tier breakdowns
│   └── log/                            # Timestamped container execution logs
│       ├── address_to_location_*.log
├── results/                            # Benchmark evaluation results and reports
│   ├── comparison_results.csv          # Row-by-row match results across 2,000 addresses
│   ├── summary.json                    # Match rates and tier breakdown metrics
│   ├── report.html                     # Self-contained visual HTML report
│   └── run_*.log                       # Benchmark execution logs
└── Tools/                              # Core engineering toolkits and container recipes
    ├── Dockerfile                      # Exposome geocoder container recipe
    ├── Dockerfile.crosswalks           # Offline geocoder recipe with embedded crosswalks
    ├── code/                           # Core geocoding production scripts
    │   ├── Address_to_LOCATION.py      # Main pipeline script: addresses to OMOP tables
    │   ├── Address_to_FIPS.py          # Legacy address to FIPS script
    │   ├── OMOP_to_FIPS.py             # OMOP CDM database extraction to FIPS
    │   └── requirements.txt            # Python dependencies for geocoding code
    ├── demo/                           # Reference test inputs and expected outputs
    │   ├── GIS-linkage/
    │   │   └── EXTERNAL_EXPOSURE.csv   # Reference sample of final spatial linkage output
    │   └── geocoding/
    │       ├── input/                  # Validated input templates (single- and multi-column)
    │       └── output/                 # Reference sample output from demo runs
    ├── doc/
    │   └── UserManual.md               # Comprehensive user manual and edge case guide
    └── postgis-exposure/               # Spatial join engine for SDoH linkage
        ├── Dockerfile                  # PostGIS container recipe with GDAL and shp2pgsql
        ├── LICENSE                     # PostGIS component license
        ├── produce_external_exposure.sh# Ingestion and spatial join orchestrator
        ├── csv/                        # Metadata dictionaries (DATA_SRC_SIMPLE, VRBL_SRC_SIMPLE)
        ├── sql/                        # SQL join templates (init, 1-point and 2-point joins)
        └── test/                       # Verification fixtures
```

### Lineage Guide: Where to Go for What

| Task | Primary Directory / File | Description |
| :--- | :--- | :--- |
| **Prepare input addresses** | `input_address/` or `Tools/demo/geocoding/input/` | Place your address CSV in `input_address/` using demo formats as templates |
| **Run local address geocoding** | `Tools/code/Address_to_LOCATION.py` | Executed via Docker container to generate `output/LOCATION.csv` |
| **Inspect geocoding results & logs**| `output/` and `output/log/` | Review `LOCATION.csv`, summary CSVs, and container run logs |
| **Run environmental spatial join** | `Tools/postgis-exposure/` | Spatially join coordinates against SDoH shapefiles to build `EXTERNAL_EXPOSURE.csv` |
| **Run accuracy benchmark** | `geocoding_comparison.py` | Benchmarks accuracy against U.S. Census API, writing to `census_output/`, `exposome_output/`, `results/` |
| **Inspect benchmark reports** | `results/` | Open `results/report.html` or inspect `results/summary.json` |
| **Read detailed technical manual** | `Tools/doc/UserManual.md` | Full specifications, Docker socket setup, and edge case handling |

---

## Prerequisites

1. **Docker Desktop** (or Docker Engine on Linux) with resource allocations:
   - Minimum 4 CPU cores
   - Minimum 8 GB RAM
2. **Docker Images**:
   Pull the production images before running the pipeline:
   ```bash
   # Primary geocoding wrapper
   docker pull prismaplab/exposome-geocoder:1.0.4

   # Underlying street-level geocoder engine (approx. 4.4 GB download, 8.8 GB unpacked)
   docker pull ghcr.io/degauss-org/geocoder:3.3.0
   ```
3. **Python 3.10+** (for running `geocoding_comparison.py` or local post-processing scripts).

---

## Input Data Formats

Place your input file inside `input_address/`. Choose **one** of the two supported formats:

### Format A: Multi-Column Address (Recommended)

| location_id | street | city | state | zip | year | entity_id |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 1250 W 16th St | Jacksonville | FL | 32209 | 2019 | 1001 |
| 2 | 2001 SW 16th St | Gainesville | FL | 32608 | 2019 | 1002 |

- `street` and `zip` are required for high-accuracy street geocoding.
- `city` may be omitted if unknown (leave empty).
- `year` and `entity_id` populate `LOCATION_HISTORY.csv`.

### Format B: Single-Column Address

| location_id | address | year | entity_id |
| :--- | :--- | :--- | :--- |
| 1 | 1250 W 16th St, Jacksonville, FL 32209 | 2019 | 1001 |
| 2 | 2001 SW 16th St, Gainesville, FL 32608 | 2019 | 1002 |

---

## Step-by-Step Launch Sequence

### Step 1: Geocoding via Exposome Container

The geocoding wrapper container spawns an internal DeGAUSS container to geocode addresses offline. Because it orchestrates a child container, **the host Docker socket must be mounted**.

#### Windows (Command Prompt - CMD)

```cmd
docker run -it --rm ^
  -v "%cd%":/workspace ^
  -v /var/run/docker.sock:/var/run/docker.sock ^
  -e HOST_PWD="%cd%" ^
  -w /workspace ^
  prismaplab/exposome-geocoder:1.0.4 ^
  /app/code/Address_to_LOCATION.py -i input_address
```

#### Windows (PowerShell)

```powershell
docker run -it --rm `
  -v "${PWD}:/workspace" `
  -v //var/run/docker.sock:/var/run/docker.sock `
  -e "HOST_PWD=${PWD}" `
  -w /workspace `
  prismaplab/exposome-geocoder:1.0.4 `
  /app/code/Address_to_LOCATION.py -i input_address
```

> **Important Windows Socket Note:** In PowerShell or Git Bash, mount the Docker socket using double slashes: `-v //var/run/docker.sock:/var/run/docker.sock`. Do not use Windows named pipes (`//./pipe/docker_engine`), because the inner container requires a POSIX socket.

#### Linux / macOS (Bash)

```bash
docker run -it --rm \
  -v "$(pwd)":/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e HOST_PWD="$(pwd)" \
  -w /workspace \
  prismaplab/exposome-geocoder:1.0.4 \
  /app/code/Address_to_LOCATION.py -i input_address
```

#### Expected Outputs in `output/`

- `output/LOCATION.csv`: OMOP CDM v5.4 compatible table with geocoded latitude, longitude, and county.
- `output/LOCATION_HISTORY.csv`: Tracks patient encounters over time with start and end dates.
- `output/geocoding_summary_<timestamp>.csv`: Summary metrics (success counts, failure counts, tier breakdown).
- `output/geocode_failures_<timestamp>.csv`: Offending records that failed parsing (e.g. missing both street and ZIP).
- `output/log/address_to_location_<timestamp>.log`: Complete runtime execution log.

---

### Step 2: Environmental Spatial Linkage (PostGIS-Exposure)

Once coordinates are produced in `output/LOCATION.csv`, use `Tools/postgis-exposure/` to link coordinates with environmental datasets (such as ADI, SVI, EJI, AHRQ).

1. Build the local linkage image:
   ```bash
   cd Tools/postgis-exposure
   docker build -t chorus-postgis-exposure:local .
   cd ../..
   ```

2. Execute spatial linkage:
   ```bash
   docker run --rm \
     -v "$(pwd)/output":/source \
     -v "/path/to/spatial/shapes":/shapes \
     -e DATA_SOURCES="1,2,3" \
     -e VARIABLES="ALL" \
     chorus-postgis-exposure:local \
     /produce_external_exposure.sh
   ```

3. Output produced:
   - `output/EXTERNAL_EXPOSURE.csv`: Analysis-ready exposure table.

---

### Step 3: Quality Validation & Accuracy Benchmarking

To verify that local geocoding matches national standards, use the automated benchmark pipeline `geocoding_comparison.py`. It runs a parallel comparison between the local container and the official U.S. Census Bureau Geocoding API.

1. Install validation dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the benchmark:
   ```bash
   python geocoding_comparison.py
   ```

#### Benchmark Results (Validated on 2,000 Sample Addresses)

| Metric | Count | Percentage |
| :--- | :--- | :--- |
| **Total Evaluated Addresses** | 2,000 | 100.0% |
| **Both Resolved to Census Tract** | 1,711 | 85.55% |
| **Matching Census Tract** | 1,579 | **92.29%** |
| **Tract Discrepancies** | 132 | 7.71% |
| **U.S. Census API Unresolved** | 289 | 14.45% |
| **Exposome Unresolved** | 0 | 0.00% |

#### Performance by Resolution Tier

| Geocoding Tier / Modifier | Evaluated | Tract Matches | Match Rate |
| :--- | :--- | :--- | :--- |
| **Street-Level (`address_full_address`)** | 1,538 | 1,502 | **97.66%** |
| **City/State/ZIP (`zip5_city_state_zip5`)** | 26 | 13 | 50.00% |
| **ZIP-Only Fallback (`zip5_zip5_only`)** | 147 | 64 | 43.54% |

- Open `results/report.html` in your browser for the full interactive dashboard.

---

## Troubleshooting & Best Practices

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| `Cannot connect to the Docker daemon at unix:///var/run/docker.sock` | Docker socket not mounted into container, or named pipe used on Windows | Ensure `-v //var/run/docker.sock:/var/run/docker.sock` (PowerShell) or `-v /var/run/docker.sock:/var/run/docker.sock` (CMD/Linux) is included. |
| Container takes over 30 minutes on first run | DeGAUSS 3.3.0 image is downloading in the background (approx. 4.4 GB) | Pre-pull the image before launching: `docker pull ghcr.io/degauss-org/geocoder:3.3.0`. |
| Low tract match rate (around 30%) | DeGAUSS was skipped due to missing socket, falling back to ZIP5 centroids | Verify Docker socket permissions and check container logs in `output/log/`. |
| Address failed validation | Missing `location_id`, or missing both `street` and `zip` | Provide unique `location_id` per row and ensure at least street or 5-digit ZIP is populated. |

---

## Git Workflow: Branching and Updates

To push these updates to the GitHub repository:

```bash
# 1. Create and switch to a feature branch
git checkout -b feature/geocoding-validation-toolkit

# 2. Stage all updated files and new folders
git add .

# 3. Commit the changes
git commit -m "feat: add geocoding validation suite, benchmark results, run logs, and updated lineage docs"

# 4. Push to remote
git push origin feature/geocoding-validation-toolkit
```

---

## License

This repository is licensed under the [MIT License](LICENSE).
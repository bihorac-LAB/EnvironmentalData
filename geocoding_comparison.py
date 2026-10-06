"""
geocoding_comparison.py
========================
Bihorac Lab - PRISMAp

Compares geocoding accuracy between:
  1. U.S. Census Bureau Geocoding API  (trusted reference)
  2. Exposome geocoder Docker container (prismaplab/exposome-geocoder:1.0.4)

End goal: Report % of addresses that resolve to the same census tract.

Usage:
    python geocoding_comparison.py [--input <csv>] [--workers <n>] [--chunk <n>] [--skip-docker] [--no-cache]

Outputs:
    results/comparison_results.csv      - full row-by-row comparison
    results/summary.json                - match rate summary
    results/report.html                 - human-readable HTML report
"""

import argparse
import csv
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Force UTF-8 output on Windows so unicode chars do not crash the logger
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config  (all tuneable via CLI args or env vars -- nothing hardcoded)
# ---------------------------------------------------------------------------
DEFAULTS = {
    "input_file":           "nad_encounters_sample.csv",
    "census_batch_url":     "https://geocoding.geo.census.gov/geocoder/geographies/addressbatch",
    "census_coord_url":     "https://geocoding.geo.census.gov/geocoder/geographies/coordinates",
    "census_benchmark":     "Public_AR_Current",
    "census_vintage":       "Current_Current",
    "exposome_image":       "prismaplab/exposome-geocoder:1.0.4",
    "exposome_input_dir":   "input_address",
    "exposome_input_file":  "nad_exposome_input.csv",
    "census_output_dir":    "census_output",
    "exposome_output_dir":  "exposome_output",
    "results_dir":          "results",
    "chunk_size":           9000,    # Census API hard limit is 10,000
    "coord_workers":        5,       # parallel threads for coord lookups
    "api_retries":          3,
    "retry_delay_s":        2,
    "coord_delay_s":        0.2,     # polite delay per worker thread
    "use_cache":            True,
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(results_dir: Path) -> logging.Logger:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    log = logging.getLogger("geocoding_comparison")
    log.info(f"Log file: {log_path}")
    return log


def retry(fn, retries: int, delay: float, log: logging.Logger):
    """Call fn(), retrying on exception up to `retries` times."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            log.warning(f"  Attempt {attempt}/{retries} failed ({exc}). Retrying in {delay}s...")
            time.sleep(delay)
    raise last_exc


# ---------------------------------------------------------------------------
# Step 1 â€” Build input CSVs
# ---------------------------------------------------------------------------
def build_exposome_input(df: pd.DataFrame, out_path: Path, log: logging.Logger) -> Path:
    """
    Produce the Exposome multi-column CSV with lat/lon stripped so the
    container geocodes from the address (manager instruction).
    Replaces 'Not stated' city with blank; trims ZIP+4 to ZIP5.
    """
    out = pd.DataFrame({
        "location_id": df["location_id"].astype(str).str.strip(),
        "street":      df["address_1"].astype(str).str.strip(),
        "city":        df["city"].astype(str).str.strip().replace("Not stated", ""),
        "state":       df["state"].astype(str).str.strip(),
        "zip":         df["zip"].astype(str).str.strip().str[:5],
        "year":        df["year"].astype(str).str.strip(),
        "entity_id":   df["entity_id"].astype(str).str.strip(),
    })
    out.to_csv(out_path, index=False)
    log.info(f"  Exposome input written: {out_path}  ({len(out)} rows)")
    return out_path


def build_census_batch_input(df: pd.DataFrame, out_path: Path, log: logging.Logger) -> Path:
    """
    Produce Census batch CSV: no header, columns = ID, Street, City, State, ZIP.
    """
    out = pd.DataFrame({
        "location_id": df["location_id"].astype(str).str.strip(),
        "street":      df["address_1"].astype(str).str.strip(),
        "city":        df["city"].astype(str).str.strip().replace("Not stated", ""),
        "state":       df["state"].astype(str).str.strip(),
        "zip":         df["zip"].astype(str).str.strip().str[:5],
    })
    out.to_csv(out_path, index=False, header=False)
    log.info(f"  Census batch input written: {out_path}  ({len(out)} rows)")
    return out_path


# ---------------------------------------------------------------------------
# Step 2 â€” Census batch geocoding (chunked with caching & csv.reader)
# ---------------------------------------------------------------------------
def census_batch_geocode(csv_path: Path, cfg: dict, log: logging.Logger) -> pd.DataFrame:
    """
    Submit addresses in chunks to the Census batch API or load from cache.
    Returns DataFrame: location_id, match_status_census, lat_census, lon_census, census_tract_census
    """
    df = pd.read_csv(csv_path, header=None,
                     names=["location_id", "street", "city", "state", "zip"],
                     dtype=str)
    chunk_size = cfg["chunk_size"]
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    log.info(f"  Census API: {len(df)} rows split into {len(chunks)} chunk(s) of <={chunk_size}")

    all_records = []
    for idx, chunk in enumerate(chunks, 1):
        raw_path = Path(cfg["census_output_dir"]) / f"census_raw_chunk_{idx}.csv"
        raw_text = None

        if cfg.get("use_cache", True) and raw_path.exists():
            log.info(f"  Chunk {idx}/{len(chunks)}: using cached Census API response: {raw_path}")
            raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
        else:
            log.info(f"  Submitting chunk {idx}/{len(chunks)} ({len(chunk)} rows) ...")
            buf = io.StringIO()
            chunk.to_csv(buf, index=False, header=False)
            buf.seek(0)

            def post():
                resp = requests.post(
                    cfg["census_batch_url"],
                    data={"benchmark": cfg["census_benchmark"], "vintage": cfg["census_vintage"]},
                    files={"addressFile": (f"chunk_{idx}.csv", buf.read().encode("utf-8"), "text/csv")},
                    timeout=300,
                )
                resp.raise_for_status()
                return resp

            response = retry(lambda: post(), cfg["api_retries"], cfg["retry_delay_s"], log)
            raw_text = response.text
            raw_path.write_text(raw_text, encoding="utf-8")

        # Parse raw response with csv.reader to handle addresses with commas
        reader = csv.reader(io.StringIO(raw_text))
        for row in reader:
            if not row:
                continue
            location_id  = row[0].strip().strip('"')
            match_status = row[2].strip() if len(row) > 2 else "No_Match"

            lat_c = lon_c = full_tract = None
            if match_status == "Match" and len(row) >= 11:
                coords = row[5].strip() if len(row) > 5 else ""
                if "," in coords:
                    cp = coords.split(",")
                    if len(cp) == 2:
                        try:
                            lon_c, lat_c = float(cp[0]), float(cp[1])
                        except ValueError:
                            pass

                state_fips  = row[8].strip()  if len(row) > 8  else ""
                county_fips = row[9].strip()  if len(row) > 9  else ""
                tract_code  = row[10].strip() if len(row) > 10 else ""

                if state_fips and county_fips and tract_code:
                    full_tract = f"{state_fips.zfill(2)}{county_fips.zfill(3)}{tract_code.zfill(6)}"

            all_records.append({
                "location_id":         location_id,
                "match_status_census": match_status,
                "lat_census":          lat_c,
                "lon_census":          lon_c,
                "census_tract_census": full_tract,
            })

    result = pd.DataFrame(all_records)
    log.info(f"  Census results: {len(result)} rows | statuses: {result['match_status_census'].value_counts().to_dict()}")
    return result


# ---------------------------------------------------------------------------
# Step 3 â€” Exposome Docker container
# ---------------------------------------------------------------------------
def run_exposome_container(cfg: dict, log: logging.Logger) -> bool:
    """
    Pull and run the Exposome geocoder container on Windows Docker Desktop.
    Input folder is mounted as /workspace; the input CSV is inside it.
    """
    image = cfg["exposome_image"]
    input_dir = Path(cfg["exposome_input_dir"]).resolve()
    workspace = str(input_dir.parent)

    log.info(f"  Pulling image: {image} ...")
    pull = subprocess.run(["docker", "pull", image], capture_output=True, text=True)
    if pull.returncode != 0:
        log.error(f"  Pull failed:\n{pull.stderr}")
        return False
    log.info("  Image ready.")

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{workspace}:/workspace",
        "-v", ("//var/run/docker.sock:/var/run/docker.sock" if sys.platform == "win32" else "/var/run/docker.sock:/var/run/docker.sock"),
        "-e", f"HOST_PWD={workspace}",
        "-w", "/workspace",
        image,
        "/app/code/Address_to_LOCATION.py",
        "-i", cfg["exposome_input_dir"],
    ]
    log.info(f"  Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.stdout:
        log.info(f"  Container stdout:\n{result.stdout[:4000]}")
    if result.stderr:
        log.warning(f"  Container stderr:\n{result.stderr[:4000]}")

    if result.returncode != 0:
        log.error(f"  Container exited with code {result.returncode}")
        return False

    log.info("  Exposome container finished successfully.")
    return True


def find_location_csv(cfg: dict, log: logging.Logger) -> Path | None:
    """Search for LOCATION.csv produced by the Exposome container."""
    candidates = [
        Path(cfg["exposome_input_dir"]).parent / "output" / "LOCATION.csv",
        Path(cfg["exposome_input_dir"]) / "output" / "LOCATION.csv",
        Path("output") / "LOCATION.csv",
    ]
    for p in candidates:
        if p.exists():
            log.info(f"  Found LOCATION.csv: {p}")
            return p
    log.error(f"  LOCATION.csv not found in any expected location: {candidates}")
    return None


# ---------------------------------------------------------------------------
# Step 4 â€” Coordinate -> Census Tract via Census API (with caching)
# ---------------------------------------------------------------------------
def _lookup_one(args):
    loc_id, lat, lon, cfg = args
    time.sleep(cfg["coord_delay_s"])
    try:
        resp = requests.get(
            cfg["census_coord_url"],
            params={
                "x": lon, "y": lat,
                "benchmark": cfg["census_benchmark"],
                "vintage":   cfg["census_vintage"],
                "layers":    "Census Tracts",
                "format":    "json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        tracts = resp.json().get("result", {}).get("geographies", {}).get("Census Tracts", [])
        if tracts:
            t = tracts[0]
            return loc_id, lat, lon, (
                f"{t.get('STATE','').zfill(2)}"
                f"{t.get('COUNTY','').zfill(3)}"
                f"{t.get('TRACT','').zfill(6)}"
            )
    except Exception:
        pass
    return loc_id, lat, lon, None


def lookup_exposome_tracts(location_csv: Path, cfg: dict, log: logging.Logger) -> pd.DataFrame:
    """
    Read Exposome LOCATION.csv, look up census tract for each coordinate
    using concurrent threads, with local cache to avoid re-querying unchanged coords.
    """
    df = pd.read_csv(location_csv, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    log.info(f"  Exposome LOCATION.csv: {len(df)} rows | cols: {list(df.columns)}")

    cache_file = Path(cfg["exposome_output_dir"]) / "coord_tract_cache.json"
    cache = {}
    if cfg.get("use_cache", True) and cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
            log.info(f"  Loaded {len(cache)} cached coordinate tract lookups.")
        except Exception:
            cache = {}

    tasks, cached_results, no_coord = [], [], []
    for _, row in df.iterrows():
        loc_id = str(row.get("location_id", "")).strip().strip('"')
        lat_s  = str(row.get("latitude",  row.get("lat",  ""))).strip()
        lon_s  = str(row.get("longitude", row.get("lon",  row.get("long", "")))).strip()
        modifier = row.get("modifier_source_value", "")
        try:
            lat, lon = float(lat_s), float(lon_s)
            coord_key = f"{lat:.6f},{lon:.6f}"
            if coord_key in cache:
                cached_results.append({
                    "location_id":           loc_id,
                    "lat_exposome":          lat,
                    "lon_exposome":          lon,
                    "census_tract_exposome": cache[coord_key],
                    "modifier_source_value": modifier,
                })
            else:
                tasks.append((loc_id, lat, lon, cfg))
        except ValueError:
            no_coord.append({
                "location_id":           loc_id,
                "lat_exposome":          None,
                "lon_exposome":          None,
                "census_tract_exposome": None,
                "modifier_source_value": modifier,
            })

    log.info(f"  Coordinate lookups needed: {len(tasks)} (cached: {len(cached_results)}) | no-coord rows: {len(no_coord)}")

    new_results = []
    if tasks:
        workers = min(cfg["coord_workers"], len(tasks))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_lookup_one, t): t for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="  Tract lookups", unit="addr"):
                loc_id, lat, lon, tract = fut.result()
                orig_match = df[df["location_id"].astype(str).str.strip().str.strip('"') == loc_id]
                modifier = orig_match.iloc[0].get("modifier_source_value", "") if not orig_match.empty else ""
                coord_key = f"{lat:.6f},{lon:.6f}"
                cache[coord_key] = tract
                new_results.append({
                    "location_id":           loc_id,
                    "lat_exposome":          lat,
                    "lon_exposome":          lon,
                    "census_tract_exposome": tract,
                    "modifier_source_value": modifier,
                })
        # Save updated cache
        try:
            cache_file.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        except Exception:
            pass

    out_df = pd.DataFrame(cached_results + new_results + no_coord)
    return out_df


# ---------------------------------------------------------------------------
# Step 5 â€” Compare & report
# ---------------------------------------------------------------------------
def compare(census_df: pd.DataFrame, exposome_df: pd.DataFrame,
            log: logging.Logger) -> pd.DataFrame:
    census_df["location_id"]   = census_df["location_id"].astype(str).str.strip().str.strip('"')
    exposome_df["location_id"] = exposome_df["location_id"].astype(str).str.strip().str.strip('"')
    merged = pd.merge(census_df, exposome_df, on="location_id", how="outer")

    def normalize_tract(v):
        if pd.isna(v):
            return None
        s = str(v).strip().strip('"')
        return s if s and s not in ("", "None", "nan") else None

    merged["census_tract_census"]   = merged["census_tract_census"].apply(normalize_tract)
    merged["census_tract_exposome"] = merged["census_tract_exposome"].apply(normalize_tract)

    merged["both_have_tract"] = (
        merged["census_tract_census"].notna() & merged["census_tract_exposome"].notna()
    )
    merged["tract_match"] = merged.apply(
        lambda r: r["census_tract_census"] == r["census_tract_exposome"]
                  if r["both_have_tract"] else False, axis=1
    )
    return merged


def print_summary(merged: pd.DataFrame, log: logging.Logger) -> dict:
    total      = len(merged)
    both       = int(merged["both_have_tract"].sum())
    matches    = int(merged["tract_match"].sum())
    no_census  = int(merged["census_tract_census"].isna().sum())
    no_exposome= int(merged["census_tract_exposome"].isna().sum())
    pct        = round(matches / both * 100, 2) if both > 0 else 0.0

    log.info("\n" + "="*60)
    log.info("RESULTS")
    log.info("="*60)
    log.info(f"  Total addresses:              {total}")
    log.info(f"  Both returned a tract:        {both}")
    log.info(f"  Matching census tract:        {matches}")
    log.info(f"  Non-matching:                 {both - matches}")
    log.info(f"  Census API no-match:          {no_census}")
    log.info(f"  Exposome no-coord:            {no_exposome}")
    log.info(f"\n  MATCH RATE: {matches}/{both} = {pct}%\n")
    log.info("="*60)

    return {
        "run_timestamp":       datetime.now().isoformat(),
        "total_addresses":     total,
        "both_returned_tract": both,
        "matching_tract":      matches,
        "non_matching_tract":  both - matches,
        "census_no_match":     no_census,
        "exposome_no_coord":   no_exposome,
        "match_rate_pct":      pct,
    }


def write_html_report(merged: pd.DataFrame, summary: dict, out_path: Path):
    """Generate a self-contained, modern HTML report."""
    ts       = datetime.now().strftime("%B %d, %Y - %H:%M:%S")
    total    = summary["total_addresses"]
    both     = summary["both_returned_tract"]
    matches  = summary["matching_tract"]
    non_match= summary["non_matching_tract"]
    pct      = summary["match_rate_pct"]
    color    = "#22c55e" if pct >= 80 else "#f59e0b" if pct >= 60 else "#ef4444"

    # Breakdown by Exposome modifier / tier
    tier_counts = {}
    for _, r in merged.iterrows():
        t = str(r.get("modifier_source_value", "Unknown"))
        if not t or t == "nan":
            t = "Unknown"
        is_m = r.get("tract_match", False)
        b_h  = r.get("both_have_tract", False)
        if t not in tier_counts:
            tier_counts[t] = {"total": 0, "both": 0, "match": 0}
        tier_counts[t]["total"] += 1
        if b_h:
            tier_counts[t]["both"] += 1
        if is_m:
            tier_counts[t]["match"] += 1

    tier_rows_html = ""
    for tier_name, stat in sorted(tier_counts.items(), key=lambda x: -x[1]["total"]):
        t_tot = stat["total"]
        t_both = stat["both"]
        t_match = stat["match"]
        t_pct = f"{round(t_match / t_both * 100, 1)}%" if t_both > 0 else "N/A"
        tier_rows_html += f"<tr><td><strong>{tier_name}</strong></td><td>{t_tot}</td><td>{t_both}</td><td>{t_match}</td><td><span class='badge'>{t_pct}</span></td></tr>"

    rows_html = ""
    for _, r in merged.head(1000).iterrows():
        ct_c  = r.get("census_tract_census")  or "â€”"
        ct_e  = r.get("census_tract_exposome") or "â€”"
        match = r.get("tract_match", False)
        badge = '<span class="match">MATCH</span>' if match else '<span class="nomatch">NO MATCH</span>'
        tier  = str(r.get("modifier_source_value", "") or "")[:50]
        rows_html += f"<tr><td>{r['location_id']}</td><td><code>{ct_c}</code></td><td><code>{ct_e}</code></td><td>{badge}</td><td style='font-size:0.75rem;color:#64748b'>{tier}</td></tr>\n"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Geocoding Comparison Report | Bihorac Lab - PRISMAp</title>
<style>
  :root {{
    --primary: #1e3a5f;
    --primary-dark: #0f172a;
    --accent: {color};
    --bg: #f8fafc;
    --card: #ffffff;
    --border: #e2e8f0;
    --text: #1e293b;
    --text-muted: #64748b;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: var(--bg); color: var(--text); line-height: 1.5; }}
  header {{ background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%); color: #fff; padding: 36px 48px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
  header h1 {{ font-size: 1.85rem; font-weight: 700; letter-spacing: -0.02em; }}
  header p {{ margin-top: 6px; opacity: 0.85; font-size: 0.92rem; }}
  .container {{ max-width: 1300px; margin: 0 auto; padding: 32px 40px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 20px; margin-bottom: 32px; }}
  .card {{ background: var(--card); border-radius: 14px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); border: 1px solid var(--border); transition: transform 0.15s ease; }}
  .card:hover {{ transform: translateY(-2px); }}
  .card .num {{ font-size: 2.3rem; font-weight: 800; color: var(--primary); line-height: 1.1; }}
  .card .lbl {{ font-size: 0.82rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 8px; }}
  .card.highlight {{ border-top: 4px solid var(--accent); }}
  .card.highlight .num {{ color: var(--accent); }}
  .section-title {{ font-size: 1.25rem; font-weight: 700; color: var(--primary-dark); margin: 36px 0 16px; display: flex; align-items: center; gap: 8px; }}
  .table-card {{ background: var(--card); border-radius: 14px; border: 1px solid var(--border); overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.04); margin-bottom: 32px; }}
  table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 0.85rem; }}
  th {{ background: #f1f5f9; color: var(--primary-dark); padding: 14px 18px; font-weight: 600; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.03em; border-bottom: 1px solid var(--border); }}
  td {{ padding: 12px 18px; border-bottom: 1px solid var(--border); }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f8fafc; }}
  code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-family: ui-monospace, SFMono-Regular, monospace; font-size: 0.82rem; color: #0f172a; }}
  .match {{ background: #dcfce7; color: #15803d; padding: 4px 10px; border-radius: 9999px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em; display: inline-block; }}
  .nomatch {{ background: #fee2e2; color: #b91c1c; padding: 4px 10px; border-radius: 9999px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em; display: inline-block; }}
  .badge {{ background: #e0f2fe; color: #0369a1; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 0.78rem; }}
  footer {{ text-align: center; padding: 36px 20px; color: var(--text-muted); font-size: 0.82rem; border-top: 1px solid var(--border); background: #fff; margin-top: 40px; }}
</style>
</head>
<body>
<header>
  <h1>Geocoding Comparison Report</h1>
  <p>Bihorac Lab / PRISMAp &nbsp;â€¢&nbsp; Reference: U.S. Census Bureau API vs Evaluated: Exposome Container &nbsp;â€¢&nbsp; {ts}</p>
</header>

<div class="container">
  <div class="cards">
    <div class="card highlight">
      <div class="num">{pct}%</div>
      <div class="lbl">Census Tract Match Rate</div>
    </div>
    <div class="card">
      <div class="num">{total:,}</div>
      <div class="lbl">Total Records</div>
    </div>
    <div class="card">
      <div class="num">{both:,}</div>
      <div class="lbl">Both Geocoded</div>
    </div>
    <div class="card">
      <div class="num">{matches:,}</div>
      <div class="lbl">Matching Tracts</div>
    </div>
    <div class="card">
      <div class="num">{non_match:,}</div>
      <div class="lbl">Tract Discrepancies</div>
    </div>
  </div>

  <div class="section-title">ðŸ“Š Performance by Exposome Geocoding Tier</div>
  <div class="table-card">
    <table>
      <thead>
        <tr>
          <th>Geocoding Tier / Modifier</th>
          <th>Total Input</th>
          <th>Both Evaluated</th>
          <th>Tract Matches</th>
          <th>Match Rate</th>
        </tr>
      </thead>
      <tbody>
        {tier_rows_html}
      </tbody>
    </table>
  </div>

  <div class="section-title">ðŸ” Address Level Comparison (First 1,000 Rows)</div>
  <div class="table-card">
    <table>
      <thead>
        <tr>
          <th style="width: 100px;">Location ID</th>
          <th>Reference Tract (Census API)</th>
          <th>Exposome Tract (Container)</th>
          <th style="width: 120px;">Result</th>
          <th>Exposome Tier</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>
</div>

<footer>
  Generated automatically by <code>geocoding_comparison.py</code> &nbsp;â€¢&nbsp; Bihorac Lab / PRISMAp Core Pipeline
</footer>
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Geocoding comparison: Census API vs Exposome container")
    p.add_argument("--input",        default=DEFAULTS["input_file"],        help="Input CSV path")
    p.add_argument("--workers",      type=int, default=DEFAULTS["coord_workers"],  help="Parallel coord-lookup threads")
    p.add_argument("--chunk",        type=int, default=DEFAULTS["chunk_size"],     help="Census batch chunk size")
    p.add_argument("--retries",      type=int, default=DEFAULTS["api_retries"],    help="API retry attempts")
    p.add_argument("--skip-docker",  action="store_true", help="Skip running Exposome container if LOCATION.csv exists")
    p.add_argument("--force-docker", action="store_true", help="Force re-running Exposome container")
    p.add_argument("--no-cache",     action="store_true", help="Ignore cached API responses")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = {
        **DEFAULTS,
        "input_file":    args.input,
        "coord_workers": args.workers,
        "chunk_size":    args.chunk,
        "api_retries":   args.retries,
        "use_cache":     not args.no_cache,
    }

    # Resolve input file
    input_p = Path(cfg["input_file"])
    if not input_p.exists():
        fallback = Path("nad_encounters_sample.csv")
        if fallback.exists():
            cfg["input_file"] = str(fallback)
        else:
            fallback2 = Path("input_address") / "nad_encounters_sample.csv"
            if fallback2.exists():
                cfg["input_file"] = str(fallback2)

    for d in [cfg["census_output_dir"], cfg["exposome_output_dir"], cfg["results_dir"]]:
        Path(d).mkdir(parents=True, exist_ok=True)

    log = setup_logging(Path(cfg["results_dir"]))
    log.info(f"Input file : {cfg['input_file']}")
    log.info(f"Workers    : {cfg['coord_workers']}")
    log.info(f"Chunk size : {cfg['chunk_size']}")
    log.info(f"Use cache  : {cfg['use_cache']}")

    # Step 1 â€” Load
    log.info("\nStep 1: Loading input ...")
    df = pd.read_csv(cfg["input_file"], dtype=str)
    log.info(f"  Rows: {len(df)} | Columns: {list(df.columns)}")

    # Step 2 â€” Census Batch Geocoding
    log.info("\nStep 2: Census batch geocoding ...")
    census_in = Path(cfg["census_output_dir"]) / "nad_census_batch_input.csv"
    build_census_batch_input(df, census_in, log)
    census_df = census_batch_geocode(census_in, cfg, log)
    census_df.to_csv(Path(cfg["census_output_dir"]) / "census_results.csv", index=False)

    # Step 3 â€” Exposome Docker Container
    log.info("\nStep 3: Exposome Docker container ...")
    exposome_in = Path(cfg["exposome_input_dir"]) / cfg["exposome_input_file"]
    build_exposome_input(df, exposome_in, log)

    location_csv = find_location_csv(cfg, log)
    run_needed = args.force_docker or (location_csv is None and not args.skip_docker)

    if run_needed:
        if not run_exposome_container(cfg, log):
            log.error("Exposome container failed. Saving Census-only results and exiting.")
            census_df.to_csv(Path(cfg["results_dir"]) / "census_only_results.csv", index=False)
            sys.exit(1)
        location_csv = find_location_csv(cfg, log)
    else:
        log.info(f"  Skipping Docker container run (using existing: {location_csv})")

    if not location_csv:
        log.error("No LOCATION.csv available.")
        sys.exit(1)

    # Step 4 â€” Coordinate -> Census Tract
    log.info("\nStep 4: Deriving census tracts from Exposome coordinates ...")
    shutil.copy(location_csv, Path(cfg["exposome_output_dir"]) / "LOCATION.csv")
    exposome_df = lookup_exposome_tracts(location_csv, cfg, log)
    exposome_df.to_csv(Path(cfg["exposome_output_dir"]) / "exposome_with_tracts.csv", index=False)

    # Step 5 â€” Compare & Report
    log.info("\nStep 5: Comparing and generating report ...")
    merged = compare(census_df, exposome_df, log)
    summary = print_summary(merged, log)

    results_csv = Path(cfg["results_dir"]) / "comparison_results.csv"
    merged.to_csv(results_csv, index=False)

    summary_json = Path(cfg["results_dir"]) / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_html = Path(cfg["results_dir"]) / "report.html"
    write_html_report(merged, summary, report_html)

    log.info(f"\n  Outputs:")
    log.info(f"    {results_csv}")
    log.info(f"    {summary_json}")
    log.info(f"    {report_html}  <-- open this in your browser\n")


if __name__ == "__main__":
    main()
"""
Generate performance metrics on 1k, 10k, and 100k fake records.
 
Track time spent, computer used, and count + % for Levels ['1', '2', '3', '4', 'Failed'].
 
Metrics are generated for two containers: One that generates lat long and another that does linkage.
"""
 
import argparse
import csv
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
 
try:
    from .openaddresses_data import DEFAULT_SOURCE_URL, get_api_token, load_env_file, prepare_openaddresses_input
except ImportError:
    from openaddresses_data import DEFAULT_SOURCE_URL, get_api_token, load_env_file, prepare_openaddresses_input
 
 
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
GEOCODER_IMAGE = "prismaplab/exposome-geocoder:1.0.4"
LINKAGE_IMAGE = "chorus-postgis-exposure-local:latest"
LEVEL_PREFIXES = {
    "level_1": "Level 1",
    "level_2": "Level 2",
    "level_3": "Level 3",
    "level_4": "Level 4",
    "failed": "failed",
}
DATA_SOURCE_ALIASES = {
    "SVI": ["9910", "9914", "9916", "9918", "9920", "9922"],
    "EJI": ["8822", "8824"],
    "ADI": ["10515", "10520", "10523"],
    "AHRQ": [str(source_id) for source_id in range(11209, 11224)],
}
 
 
 
def get_default_data_sources() -> str:
    """Return every linkage data-source ID shipped in the repository."""
    data_source_path = REPOSITORY_ROOT / "Tools" / "postgis-exposure" / "csv" / "DATA_SRC_SIMPLE.csv"
    with data_source_path.open("r", encoding="utf-8-sig", newline="") as data_source_file:
        return ",".join(row["data_source_uuid"] for row in csv.DictReader(data_source_file))
 
 
def resolve_data_sources(value: str) -> str:
    """Expand friendly dataset-family names into the source IDs expected by the container."""
    resolved: list[str] = []
    for item in value.split(","):
        name = item.strip()
        if not name:
            continue
        resolved.extend(DATA_SOURCE_ALIASES.get(name.upper(), [name]))
    if not resolved:
        raise ValueError("At least one data source is required.")
    return ",".join(dict.fromkeys(resolved))
 
 
def run_command(
    command: list[str],
    cwd: Path,
    log_path: Path,
    output_handler: Optional[Callable[[str], None]] = None,
) -> float:
    """Run a command, save its combined output, and return elapsed seconds."""
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if process.stdout is None:
            raise RuntimeError("Unable to capture command output.")
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            if output_handler is not None:
                output_handler(line.rstrip())
        return_code = process.wait()
    if return_code != 0:
        log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        error_excerpt = "\n".join(log_lines[-20:])
        raise RuntimeError(
            f"Command failed with exit code {return_code}. Full output: {log_path}\n"
            f"Last output lines:\n{error_excerpt}"
        )
    return time.perf_counter() - started
 
 
def create_linkage_progress_handler() -> Callable[[str], None]:
    """Create a line handler that reports linkage stages and spatial-join percentages."""
    total_variables: Optional[int] = None
    completed_variables = 0
    last_reported_percent = -1
 
    def report_progress(line: str) -> None:
        """Print selected linkage output lines and suppress repetitive container details."""
        nonlocal total_variables, completed_variables, last_reported_percent
        total_match = re.search(r"Found ([\d,]+) variables to process", line)
        if total_match:
            total_variables = int(total_match.group(1).replace(",", ""))
            print(f"Linkage spatial joins: 0/{total_variables:,} (0%)", flush=True)
            return
 
        variable_match = re.search(r"combine locations with variable (\d+)", line)
        if variable_match and total_variables:
            completed_variables += 1
            percent_complete = min(int(completed_variables / total_variables * 100), 100)
            if completed_variables <= 10 or percent_complete > last_reported_percent:
                print(
                    f"Linkage spatial joins: {completed_variables:,}/{total_variables:,} "
                    f"({completed_variables / total_variables * 100:.1f}%)",
                    flush=True,
                )
                last_reported_percent = percent_complete
            return
 
        if line.startswith("Fetching source data") or "Exporting EXTERNAL_EXPOSURE" in line:
            print(line, flush=True)
 
    return report_progress
 
 
def create_input_sample(source_path: Path, destination_path: Path, record_count: int) -> None:
    """Copy the CSV header and requested number of data records into a benchmark input file."""
    if record_count <= 0:
        raise ValueError(f"Record counts must be positive; received {record_count}.")
 
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("r", encoding="utf-8-sig", newline="") as source_file:
        reader = csv.reader(source_file)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Input CSV is empty: {source_path}")
        with destination_path.open("w", encoding="utf-8", newline="") as destination_file:
            writer = csv.writer(destination_file)
            writer.writerow(header)
            copied_count = 0
            for row in reader:
                if copied_count == record_count:
                    break
                writer.writerow(row)
                copied_count += 1
 
    if copied_count != record_count:
        raise ValueError(f"Requested {record_count:,} records, but {source_path} contains {copied_count:,}.")
 
 
def get_machine_metadata() -> dict[str, Any]:
    """Return stable machine details useful when comparing benchmark runs."""
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "python_version": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
    }
 
 
def validate_docker_environment(linkage_image: str, skip_linkage: bool) -> None:
    """Verify Docker is reachable and the locally built linkage image is available when required."""
    docker_info = subprocess.run(["docker", "info"], capture_output=True, text=True, check=False)
    if docker_info.returncode != 0:
        docker_error = docker_info.stderr.strip() or docker_info.stdout.strip()
        raise RuntimeError(f"Docker is not available. Start Docker Desktop and try again.\n{docker_error}")
 
    if skip_linkage:
        return
 
    image_check = subprocess.run(["docker", "image", "inspect", linkage_image], capture_output=True, text=True, check=False)
    if image_check.returncode != 0:
        linkage_directory = REPOSITORY_ROOT / "Tools" / "postgis-exposure"
        raise RuntimeError(
            f"Required linkage image '{linkage_image}' is not available locally. Build it before running the full "
            f"benchmark:\n  docker build -t {linkage_image} \"{linkage_directory}\"\n"
            "To benchmark only geocoding, pass --skip-linkage."
        )
 
 
def run_geocoder(input_dir: Path, image: str, log_path: Path) -> float:
    """Run the geocoder container against a prepared input directory."""
    repository_mount = str(REPOSITORY_ROOT.resolve())
    relative_input = input_dir.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    command = [
        "docker", "run", "--rm", "-v", f"{repository_mount}:/workspace",
        "-v", "/var/run/docker.sock:/var/run/docker.sock", "-e", f"HOST_PWD={repository_mount}",
        "-w", "/workspace", image, "/app/code/Address_to_LOCATION.py", "-i", f"/workspace/{relative_input}",
    ]
    return run_command(command=command, cwd=REPOSITORY_ROOT, log_path=log_path)
 
 
def wait_for_postgres(container_name: str, timeout_seconds: float, log_path: Path) -> None:
    """Wait until the linkage container reports that Postgres accepts connections."""
    deadline = time.monotonic() + timeout_seconds
    last_logs = ""
    while time.monotonic() < deadline:
        result = subprocess.run(["docker", "logs", container_name], capture_output=True, text=True, check=False)
        last_logs = f"{result.stdout}\n{result.stderr}"
        if "database system is ready to accept connections" in last_logs:
            log_path.write_text(last_logs, encoding="utf-8")
            return
        time.sleep(1)
    log_path.write_text(last_logs, encoding="utf-8")
    raise TimeoutError(f"Postgres did not become ready within {timeout_seconds:g} seconds. See {log_path}.")
 
 
def run_linkage(
    source_dir: Path,
    image: str,
    variables: str,
    data_sources: str,
    startup_timeout: float,
    log_path: Path,
) -> float:
    """Run the linkage container and return startup plus processing time."""
    container_name = f"environmental-benchmark-{uuid.uuid4().hex[:10]}"
    command = [
        "docker", "run", "--rm", "--name", container_name, "--env", "POSTGRES_PASSWORD=dummy",
        "--env", f"VARIABLES={variables}", "--env", f"DATA_SOURCES={data_sources}",
        "-v", f"{source_dir.resolve()}:/source", "-d", image,
    ]
    started = time.perf_counter()
    container_started = False
    try:
        result = subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            docker_error = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"Unable to start linkage container '{image}':\n{docker_error}")
        container_started = True
        wait_for_postgres(
            container_name=container_name,
            timeout_seconds=startup_timeout,
            log_path=log_path.with_name("linkage_startup.log"),
        )
        run_command(
            command=["docker", "exec", container_name, "/app/produce_external_exposure.sh"],
            cwd=REPOSITORY_ROOT,
            log_path=log_path,
            output_handler=create_linkage_progress_handler(),
        )
    finally:
        if container_started:
            subprocess.run(["docker", "stop", container_name], capture_output=True, text=True, check=False)
    return time.perf_counter() - started
 
 
def collect_level_metrics(location_path: Path, record_count: int) -> dict[str, int | float]:
    """Count geocoding levels from LOCATION.modifier_source_value and calculate percentages."""
    counts = {metric_name: 0 for metric_name in LEVEL_PREFIXES}
    with location_path.open("r", encoding="utf-8-sig", newline="") as location_file:
        for row in csv.DictReader(location_file):
            modifier = row.get("modifier_source_value", "").strip()
            matching_metrics = [name for name, prefix in LEVEL_PREFIXES.items() if modifier.lower().startswith(prefix.lower())]
            if len(matching_metrics) != 1:
                raise ValueError(f"Unrecognized modifier_source_value: {modifier!r}")
            counts[matching_metrics[0]] += 1
 
    observed_count = sum(counts.values())
    if observed_count != record_count:
        raise ValueError(f"Expected {record_count:,} LOCATION rows, but found {observed_count:,}.")
 
    metrics: dict[str, int | float] = {}
    for metric_name, count in counts.items():
        metrics[f"{metric_name}_count"] = count
        metrics[f"{metric_name}_percent"] = round(count / record_count * 100, 2)
    return metrics
 
 
def write_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write benchmark result rows to CSV."""
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
 
 
def benchmark_count(
    input_path: Path,
    run_dir: Path,
    record_count: int,
    machine: dict[str, Any],
    geocoder_image: str,
    linkage_image: str,
    skip_linkage: bool,
    variables: str,
    data_sources: str,
    startup_timeout: float,
) -> dict[str, Any]:
    """Benchmark both workflow stages for one record count and return a result row."""
    count_dir = run_dir / f"records_{record_count}"
    input_dir = count_dir / "geocoder" / "input"
    sample_path = input_dir / "encounters.csv"
    create_input_sample(source_path=input_path, destination_path=sample_path, record_count=record_count)
 
    print("Starting geocoder stage...", flush=True)
    geocoder_seconds = run_geocoder(input_dir=input_dir, image=geocoder_image, log_path=count_dir / "geocoder.log")
    print(f"Geocoder stage completed in {geocoder_seconds:.1f} seconds.", flush=True)
    geocoder_output = input_dir.parent / "output"
    location_path = geocoder_output / "LOCATION.csv"
    history_path = geocoder_output / "LOCATION_HISTORY.csv"
    if not location_path.is_file() or not history_path.is_file():
        raise FileNotFoundError(f"Geocoder did not produce LOCATION.csv and LOCATION_HISTORY.csv in {geocoder_output}.")
 
    linkage_seconds: Optional[float] = None
    if not skip_linkage:
        print("Starting environmental linkage stage...", flush=True)
        linkage_source = count_dir / "linkage" / "source"
        linkage_source.mkdir(parents=True, exist_ok=True)
        shutil.copy2(location_path, linkage_source / location_path.name)
        shutil.copy2(history_path, linkage_source / history_path.name)
        linkage_seconds = run_linkage(
            source_dir=linkage_source,
            image=linkage_image,
            variables=variables,
            data_sources=data_sources,
            startup_timeout=startup_timeout,
            log_path=count_dir / "linkage.log",
        )
        print(f"Linkage stage completed in {linkage_seconds:.1f} seconds.", flush=True)
        if not (linkage_source / "EXTERNAL_EXPOSURE.csv").is_file():
            raise FileNotFoundError("Linkage container completed without producing EXTERNAL_EXPOSURE.csv.")
 
    metrics = collect_level_metrics(location_path=location_path, record_count=record_count)
    return {
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": record_count,
        "geocoder_seconds": round(geocoder_seconds, 3),
        "linkage_seconds": "" if linkage_seconds is None else round(linkage_seconds, 3),
        "total_seconds": round(geocoder_seconds + (linkage_seconds or 0), 3),
        "geocoder_image": geocoder_image,
        "linkage_image": "" if skip_linkage else linkage_image,
        **machine,
        **metrics,
    }
 
 
def run_benchmarks(
    input_path: Optional[Path],
    counts: list[int],
    results_dir: Path = DEFAULT_RESULTS_DIR,
    geocoder_image: str = GEOCODER_IMAGE,
    linkage_image: str = LINKAGE_IMAGE,
    skip_linkage: bool = False,
    variables: str = "ALL",
    data_sources: Optional[str] = None,
    startup_timeout: float = 120.0,
    openaddresses_source_url: str = DEFAULT_SOURCE_URL,
    coordinate_fraction: float = 0.2,
    repeats: int = 1,
) -> Path:
    """Run configured benchmarks, persist their metrics, and return the result CSV path."""
    load_env_file(env_path=REPOSITORY_ROOT / ".env")
    results_dir = results_dir.resolve()
    data_sources = resolve_data_sources(data_sources or get_default_data_sources())
    if not counts:
        raise ValueError("At least one record count is required.")
    if repeats <= 0:
        raise ValueError("Repeats must be positive.")
    if len(set(counts)) != len(counts):
        raise ValueError("Record counts must not contain duplicates.")
    validate_docker_environment(linkage_image=linkage_image, skip_linkage=skip_linkage)
    if input_path is None:
        get_api_token(warn_if_missing=True)
        input_path = prepare_openaddresses_input(
            record_count=max(counts),
            source_url=openaddresses_source_url,
            coordinate_fraction=coordinate_fraction,
        )
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
 
    run_token = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = results_dir / run_token
    experiment_dir.mkdir(parents=True)
    machine = get_machine_metadata()
    rows = []
    for repeat_number in range(1, repeats + 1):
        run_dir = experiment_dir if repeats == 1 else experiment_dir / f"repeat_{repeat_number:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "machine.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")
        for record_count in counts:
            print(f"Benchmarking {record_count:,} records (experiment {repeat_number}/{repeats})...", flush=True)
            result = benchmark_count(
                    input_path=input_path,
                    run_dir=run_dir,
                    record_count=record_count,
                    machine=machine,
                    geocoder_image=geocoder_image,
                    linkage_image=linkage_image,
                    skip_linkage=skip_linkage,
                    variables=variables,
                    data_sources=data_sources,
                    startup_timeout=startup_timeout,
            )
            result["experiment_number"] = repeat_number
            result["experiment_count"] = repeats
            rows.append(result)
            results_path = run_dir / "benchmark_results.csv"
            write_results(rows=[result], output_path=results_path)
    results_path = experiment_dir / "benchmark_results.csv"
    write_results(rows=rows, output_path=results_path)
    print(f"Benchmark results: {results_path}")
    return results_path
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark geocoding and linkage for one or more record counts.")
    parser.add_argument(
        "--input",
        type=Path,
        help="Existing encounter CSV. When omitted, public address data is downloaded and cached automatically.",
    )
    parser.add_argument("--counts", type=int, nargs="+", default=[1_000], help="Record counts to benchmark.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory for benchmark artifacts.")
    parser.add_argument("--geocoder-image", default=GEOCODER_IMAGE, help="Geocoder container image.")
    parser.add_argument("--linkage-image", default=LINKAGE_IMAGE, help="Linkage container image.")
    parser.add_argument("--skip-linkage", action="store_true", help="Run only the geocoder benchmark.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of times to repeat the experiment (default: 1).")
    parser.add_argument("--variables", default="ALL", help="Comma-separated linkage variable IDs, or ALL.")
    parser.add_argument(
        "--data-sources",
        default=get_default_data_sources(),
        help="Comma-separated source IDs or dataset names (SVI, EJI, ADI, AHRQ). Defaults to every source.",
    )
    parser.add_argument("--startup-timeout", type=float, default=120.0, help="Seconds to wait for Postgres startup.")
    parser.add_argument("--openaddresses-source-url", default=DEFAULT_SOURCE_URL, help="ArcGIS address source URL.")
    parser.add_argument(
        "--coordinate-fraction",
        type=float,
        default=0.2,
        help="Fraction of generated rows retaining source coordinates (default: 0.2).",
    )
    args = parser.parse_args()
    try:
        run_benchmarks(
            input_path=args.input,
            counts=args.counts,
            results_dir=args.results_dir,
            geocoder_image=args.geocoder_image,
            linkage_image=args.linkage_image,
            skip_linkage=args.skip_linkage,
            variables=args.variables,
            data_sources=args.data_sources,
            startup_timeout=args.startup_timeout,
            openaddresses_source_url=args.openaddresses_source_url,
            coordinate_fraction=args.coordinate_fraction,
            repeats=args.repeats,
        )
    except RuntimeError as error:
        parser.error(str(error))
 
 
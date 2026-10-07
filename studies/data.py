"""Explicit public-data acquisition and offline, checksum-verified loading."""

import argparse
import calendar
from datetime import date, datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "snapshots"
UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
NOAA = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/"
STATION = "USW00023183"


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def download(url, limit=20_000_000):
    request = Request(url, headers={"User-Agent": "Curious-Coder-educational-data-study/1.0"})
    with urlopen(request, timeout=90) as response:
        payload = response.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(f"Source exceeds the {limit}-byte acquisition limit: {url}")
    return payload


def parse_daily(text, station=STATION, years=(2023, 2024)):
    records = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if len(line) < 269:
            raise ValueError("Truncated GHCN daily record.")
        year, month, element = int(line[11:15]), int(line[15:17]), line[17:21]
        if line[:11] != station or year not in years or element not in {"TMIN", "TMAX"}:
            continue
        for day in range(1, calendar.monthrange(year, month)[1] + 1):
            offset = 21 + (day - 1) * 8
            raw = int(line[offset:offset + 5])
            mflag, qflag, sflag = line[offset + 5:offset + 8]
            records.append({"date": date(year, month, day).isoformat(), "element": element,
                            "value_c": raw / 10 if raw != -9999 and qflag == " " else np.nan,
                            "raw_tenths_c": raw, "measurement_flag": mflag.strip(),
                            "quality_flag": qflag.strip(), "source_flag": sflag.strip()})
    result = pd.DataFrame(records)
    if result.empty or result.duplicated(["date", "element"]).any():
        raise ValueError("Temperature records are absent or duplicated.")
    return result.sort_values(["date", "element"]).reset_index(drop=True)


def parse_wdbc(payload):
    names = [str(name) for name in load_breast_cancer().feature_names]
    frame = pd.read_csv(io.BytesIO(payload), header=None, names=["record_id", "diagnosis", *names])
    if len(frame) != 569 or frame.record_id.duplicated().any() or set(frame.diagnosis) != {"B", "M"}:
        raise ValueError("Unexpected WDBC record count, identifiers, or diagnosis labels.")
    if not np.isfinite(frame[names].to_numpy(dtype=float)).all():
        raise ValueError("WDBC features contain missing or nonfinite measurements.")
    return frame


def acquire(destination=DATA):
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Snapshot already exists. Verify it offline, or acquire into a new --output directory.")
    uci = download(UCI)
    parse_wdbc(uci)
    climate = download(NOAA + f"all/{STATION}.dly")
    station_catalog = download(NOAA + "ghcnd-stations.txt")
    version = download(NOAA + "ghcnd-version.txt")
    lines = [line for line in climate.decode("ascii").splitlines()
             if line[:11] == STATION and line[11:15] in {"2023", "2024"} and line[17:21] in {"TMIN", "TMAX"}]
    subset = ("\n".join(lines) + "\n").encode("ascii")
    daily = parse_daily(subset.decode("ascii"))
    station_line = next(line for line in station_catalog.splitlines() if line.startswith(STATION.encode("ascii"))).decode("ascii")
    station = {"id": STATION, "latitude": float(station_line[12:20]), "longitude": float(station_line[21:30]),
               "elevation_m": float(station_line[31:37]), "name": station_line[41:71].strip()}
    files = {"wdbc.data": uci, "phoenix_2023_2024.dly": subset,
             "phoenix_daily.csv": daily.to_csv(index=False, lineterminator="\n").encode(),
             "station.json": (json.dumps(station, indent=2, sort_keys=True) + "\n").encode(),
             "ghcn_version.txt": version}
    manifest = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "files": {name: {"sha256": digest(payload), "bytes": len(payload)} for name, payload in files.items()},
        "sources": {
            "wdbc": {"url": UCI, "sha256": digest(uci), "doi": "10.24432/C5DW2B", "license": "CC BY 4.0"},
            "ghcn_station": {"url": NOAA + f"all/{STATION}.dly", "sha256_full_download": digest(climate),
                             "doi": "10.7289/V5D21VHZ", "subset": "2023-2024 TMIN/TMAX; USW00023183",
                             "reuse": "NOAA public-access US station observations; cite NOAA/NCEI; no endorsement"},
            "station_catalog": {"url": NOAA + "ghcnd-stations.txt", "sha256_full_download": digest(station_catalog)},
        },
        "transformations": ["WDBC bytes unchanged; ID excluded from predictors; malignant becomes positive class 1 at load time.",
                            "GHCN original monthly records subset to one station, two years, TMIN/TMAX.",
                            "Daily CSV preserves raw values and all three flags; -9999 or nonblank quality flag becomes missing.",
                            "Temperature converted from tenths of degrees Celsius to degrees Celsius; no imputation."],
    }
    destination.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        (destination / name).write_bytes(payload)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def verify_snapshot(directory=DATA):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("Snapshot filenames must not contain directory components.")
        payload = (directory / name).read_bytes()
        if digest(payload) != expected["sha256"] or len(payload) != expected["bytes"]:
            raise ValueError(f"Snapshot integrity mismatch: {name}")
    return manifest


def clinical_data(directory=DATA):
    verify_snapshot(directory)
    frame = parse_wdbc((Path(directory) / "wdbc.data").read_bytes())
    return frame.drop(columns=["record_id", "diagnosis"]), (frame.diagnosis == "M").astype(int).to_numpy()


def climate_data(directory=DATA):
    verify_snapshot(directory)
    frame = pd.read_csv(Path(directory) / "phoenix_daily.csv", parse_dates=["date"])
    complete = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    temperatures = frame.pivot(index="date", columns="element", values="value_c").reindex(complete)
    temperatures.index.name = "date"
    return temperatures, json.loads((Path(directory) / "station.json").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync", action="store_true", help="Acquire public sources; never overwrite a snapshot.")
    parser.add_argument("--output", type=Path, default=DATA)
    args = parser.parse_args()
    result = acquire(args.output) if args.sync else verify_snapshot(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))

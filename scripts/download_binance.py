"""Download daily files from Binance's public data site (data.binance.vision).

Each daily file is a zip with a sibling ``.CHECKSUM`` file holding its SHA-256.
For every requested day we:

1. skip it if the unzipped CSV is already in ``data/raw/...``;
2. download the zip and its .CHECKSUM;
3. verify the SHA-256 (a mismatch is an error, never ignored);
4. extract the CSV into ``data/raw/<market>/<data_type>/<symbol>/``;
5. sleep politely before the next request.

A missing file (HTTP 404) is reported and logged, not silently skipped over:
the function returns which days were missing so the caller can decide.

Examples (run from the repo root)::

    python scripts/download_binance.py --market futures/um --data-type bookTicker \
        --symbol BTCUSDT --start 2024-03-01 --end 2024-03-02
    python scripts/download_binance.py --market spot --data-type trades \
        --symbol BTCUSDT --start 2026-11-24 --end 2026-11-24
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import logging
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import requests

BASE_URL = "https://data.binance.vision/data"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"

log = logging.getLogger("download_binance")


@dataclass
class DownloadReport:
    """What happened for each requested day."""

    downloaded: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # dates with HTTP 404


def daily_url(market: str, data_type: str, symbol: str, day: dt.date) -> str:
    """URL of one daily zip, e.g. .../futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-03-01.zip."""
    name = f"{symbol}-{data_type}-{day.isoformat()}.zip"
    return f"{BASE_URL}/{market}/daily/{data_type}/{symbol}/{name}"


def file_exists(session: requests.Session, url: str, timeout: float = 30) -> bool:
    """HEAD request: True if the file is published, False on 404."""
    r = session.head(url, timeout=timeout)
    if r.status_code == 404:
        return False
    r.raise_for_status()
    return True


def _safe_extract_csv(zip_path: Path, out_dir: Path) -> Path:
    """Extract the single CSV from a Binance zip, refusing odd member names."""
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"{zip_path.name}: expected exactly one CSV, found {zf.namelist()}")
        member = members[0]
        if Path(member).name != member:  # no directories / path traversal
            raise ValueError(f"{zip_path.name}: unexpected member path {member!r}")
        target = out_dir / member
        with zf.open(member) as src, open(target, "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
    return target


def download_verified_zip(
    session: requests.Session,
    market: str,
    data_type: str,
    symbol: str,
    day: dt.date,
    out_dir: Path,
    timeout: float = 120,
) -> Path | None:
    """Download one daily zip into out_dir and verify its SHA-256 against the .CHECKSUM file.

    Returns the zip path, or None if the day is not published (HTTP 404).
    A checksum mismatch deletes the zip and raises.
    """
    url = daily_url(market, data_type, symbol, day)
    return fetch_verified(session, url, out_dir / f"{symbol}-{data_type}-{day.isoformat()}.zip", timeout)


def monthly_url(market: str, data_type: str, symbol: str, month: str, interval: str | None = None) -> str:
    """URL of one monthly zip, e.g. .../futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2023-05.zip."""
    if interval:
        return f"{BASE_URL}/{market}/monthly/{data_type}/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    return f"{BASE_URL}/{market}/monthly/{data_type}/{symbol}/{symbol}-{data_type}-{month}.zip"


def fetch_verified(session: requests.Session, url: str, zip_path: Path, timeout: float = 120) -> Path | None:
    """Download url to zip_path and check its SHA-256 against url + '.CHECKSUM'.

    Returns zip_path, or None on HTTP 404. A mismatch deletes the file and raises.
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    chk = session.get(url + ".CHECKSUM", timeout=timeout)
    if chk.status_code == 404:
        log.warning("not published: %s", url)
        return None
    chk.raise_for_status()
    expected_sha = chk.text.split()[0].strip().lower()
    sha = hashlib.sha256()
    with session.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                sha.update(chunk)
    if sha.hexdigest() != expected_sha:
        zip_path.unlink()
        raise RuntimeError(f"checksum mismatch for {url}: got {sha.hexdigest()}, expected {expected_sha}")
    log.info("checksum ok: %s", zip_path.name)
    return zip_path


def download_day(
    session: requests.Session,
    market: str,
    data_type: str,
    symbol: str,
    day: dt.date,
    raw_dir: Path = DEFAULT_RAW_DIR,
    timeout: float = 120,
) -> Path | None:
    """Download, verify and unzip one day. Returns the CSV path, or None if not published (404)."""
    out_dir = raw_dir / Path(*market.split("/")) / data_type / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{symbol}-{data_type}-{day.isoformat()}.csv"
    if csv_path.exists():
        log.info("skip (present): %s", csv_path.name)
        return csv_path

    zip_path = download_verified_zip(session, market, data_type, symbol, day, out_dir, timeout)
    if zip_path is None:
        return None

    extracted = _safe_extract_csv(zip_path, out_dir)
    if extracted != csv_path:
        extracted.rename(csv_path)
    zip_path.unlink()
    return csv_path


def download_range(
    market: str,
    data_type: str,
    symbol: str,
    start: dt.date,
    end: dt.date,
    raw_dir: Path = DEFAULT_RAW_DIR,
    sleep_s: float = 1.0,
) -> DownloadReport:
    """Download every day in [start, end] inclusive."""
    report = DownloadReport()
    with requests.Session() as session:
        session.headers["User-Agent"] = "market-making-sim research downloader"
        day = start
        while day <= end:
            out = raw_dir / Path(*market.split("/")) / data_type / symbol / f"{symbol}-{data_type}-{day.isoformat()}.csv"
            if out.exists():
                report.skipped.append(out)
            else:
                path = download_day(session, market, data_type, symbol, day, raw_dir)
                if path is None:
                    report.missing.append(day.isoformat())
                else:
                    report.downloaded.append(path)
                time.sleep(sleep_s)
            day += dt.timedelta(days=1)
    if report.missing:
        log.warning("%d day(s) not published for %s %s: %s", len(report.missing), market, data_type, report.missing)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--market", required=True, help="e.g. futures/um or spot")
    p.add_argument("--data-type", required=True, help="e.g. bookTicker, aggTrades, trades")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", required=True, type=dt.date.fromisoformat)
    p.add_argument("--end", required=True, type=dt.date.fromisoformat)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--sleep", type=float, default=1.0, help="seconds between requests")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rep = download_range(args.market, args.data_type, args.symbol, args.start, args.end, args.raw_dir, args.sleep)
    log.info("downloaded=%d skipped=%d missing=%d", len(rep.downloaded), len(rep.skipped), len(rep.missing))


if __name__ == "__main__":
    main()

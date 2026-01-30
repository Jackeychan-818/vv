#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE = "https://www.ncei.noaa.gov/data/international-comprehensive-ocean-atmosphere/v3/archive/nrt/daily/"
YEARS = range(2018, 2025)  # 2018-2024 inclusive
MONTHS = [f"{m:02d}" for m in range(1, 13)]

# ✅ FIX: make it a Path
OUT_DIR = Path("/Users/jackey/NUS Dropbox/Zeqi Chen/vv/raw_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SLEEP_BETWEEN_DIRS_SEC = 0.2

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; icoads-nc-downloader/1.0)"
}

TIMEOUT = 60
MAX_RETRIES = 5


def list_nc_links(session: requests.Session, month_url: str) -> list[str]:
    r = session.get(month_url, timeout=TIMEOUT)
    if r.status_code == 404:
        return []
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    links = []
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if href.endswith(".nc"):
            links.append(urljoin(month_url, href))
    return sorted(set(links))


def filename_from_url(url: str) -> str:
    p = urlparse(url)
    name = Path(p.path).name
    return name or "download.nc"


def head_content_length(session: requests.Session, url: str) -> int | None:
    try:
        r = session.head(url, allow_redirects=True, timeout=TIMEOUT)
        if r.status_code >= 400:
            return None
        cl = r.headers.get("Content-Length")
        return int(cl) if cl else None
    except Exception:
        return None


def download_with_resume(session: requests.Session, url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    remote_size = head_content_length(session, url)

    # If file already complete, skip
    if out_path.exists() and out_path.stat().st_size > 0:
        if remote_size is None or out_path.stat().st_size == remote_size:
            return

    resume_from = tmp_path.stat().st_size if tmp_path.exists() else 0
    headers = dict(HEADERS)
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"

    with session.get(url, stream=True, headers=headers, timeout=TIMEOUT) as r:
        r.raise_for_status()
        total_remaining = int(r.headers.get("Content-Length", 0) or 0)

        mode = "ab" if resume_from > 0 else "wb"
        with open(tmp_path, mode) as f, tqdm(
            total=total_remaining,
            unit="B",
            unit_scale=True,
            desc=out_path.name,
            leave=False,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    final_size = tmp_path.stat().st_size
    if remote_size is not None and final_size != remote_size:
        raise RuntimeError(f"Size mismatch for {out_path.name}: got {final_size}, expected {remote_size}")

    tmp_path.replace(out_path)


def download_with_retries(session: requests.Session, url: str, out_path: Path) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            download_with_resume(session, url, out_path)
            return True
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"[FAIL] {url} -> {e}")
                return False
            sleep = min(2 ** attempt, 30)
            print(f"[Retry {attempt}/{MAX_RETRIES}] {out_path.name}: {e} (sleep {sleep}s)")
            time.sleep(sleep)
    return False


def main():
    with requests.Session() as s:
        s.headers.update(HEADERS)

        all_targets: list[tuple[str, Path]] = []

        for y in YEARS:
            for m in MONTHS:
                month_url = f"{BASE}{y}/{m}/"
                links = list_nc_links(s, month_url)
                for url in links:
                    fn = filename_from_url(url)
                    out_path = OUT_DIR / str(y) / m / fn
                    all_targets.append((url, out_path))

                time.sleep(SLEEP_BETWEEN_DIRS_SEC)

        print(f"Found {len(all_targets)} .nc files total.")

        ok = 0
        for url, out_path in tqdm(all_targets, desc="Downloading", unit="file"):
            if download_with_retries(s, url, out_path):
                ok += 1

        print(f"Done. Successfully downloaded {ok}/{len(all_targets)} files.")
        print(f"Saved under: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
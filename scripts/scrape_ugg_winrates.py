"""Scrape U.GG Master+ tier list data from rendered HTML.

Dependencies:
- selenium
- webdriver-manager
- beautifulsoup4

This script relies on DOM inspection only (no network interception).
It is a one-time batch job for demo prep.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


ROLE_URLS = {
    "TOP": "https://u.gg/lol/top-lane-tier-list?rank=master_plus",
    "JUNGLE": "https://u.gg/lol/jungle-tier-list?rank=master_plus",
    "MID": "https://u.gg/lol/mid-lane-tier-list?rank=master_plus",
    "ADC": "https://u.gg/lol/adc-tier-list?rank=master_plus",
    "SUPPORT": "https://u.gg/lol/support-tier-list?rank=master_plus",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape U.GG winrates for demo use.")
    parser.add_argument("--patch", required=True, help="Patch string (e.g., 26.2)")
    parser.add_argument(
        "--output",
        default="data/meta/u_gg_winrates.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run Chrome with a visible window (useful for SSL issues).",
    )
    args = parser.parse_args()

    rows = scrape_ugg_rows(headed=args.headed)
    cleaned = [normalize_row(row, args.patch) for row in rows]
    write_csv(args.output, cleaned)


def scrape_ugg_rows(*, headed: bool) -> list[dict[str, str]]:
    """Load each role page, wait for DOM readiness, and extract row data."""
    options = Options()
    if not headed:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1400,900")
    # Mitigate occasional SSL handshake failures in headless Chrome on Windows.
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--allow-insecure-localhost")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    try:
        all_rows: list[dict[str, str]] = []
        for role, url in ROLE_URLS.items():
            driver.get(url)

            # Wait for tier list rows to render. The selector is intentionally broad
            # to be resilient to minor layout changes.
            wait = WebDriverWait(driver, 20)
            wait.until(
                ec.presence_of_element_located(
                    (By.XPATH, "//*[contains(text(), '%')]")
                )
            )
            _scroll_to_load_all(driver)

            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            all_rows.extend(extract_rows_from_dom(soup, role=role))
        return all_rows
    except TimeoutException:
        raise RuntimeError("U.GG page did not load tier list data in time.")
    finally:
        driver.quit()


def extract_rows_from_dom(soup: BeautifulSoup, *, role: str) -> list[dict[str, str]]:
    """Extract champion stats using DOM heuristics.

    This is intentionally tolerant: we look for rows that contain a champion
    name and a winrate percentage in the same row container.
    """
    rows: list[dict[str, str]] = []
    candidate_rows = soup.select("[role='row'], .rt-tr, .table-row, .tier-list-row")
    for row in candidate_rows:
        text = " ".join(row.stripped_strings)
        if "%" not in text:
            continue

        champion = _extract_champion_name(row)
        percents = _extract_percents(text)
        if not champion or not percents:
            continue

        winrate = percents[0] if len(percents) >= 1 else None
        pickrate = percents[1] if len(percents) >= 2 else None
        banrate = percents[2] if len(percents) >= 3 else None

        rows.append(
            {
                "champion": champion,
                "role": role,
                "winrate": winrate,
                "pickrate": pickrate,
                "banrate": banrate,
            }
        )
    return rows


def _extract_champion_name(row) -> str:
    """Find a champion name within a row using common DOM patterns."""
    name_node = row.select_one("[data-champion-name], .champion-name, .champion")
    if name_node:
        return name_node.get_text(strip=True)
    # Fallback: use first capitalized token sequence.
    tokens = list(row.stripped_strings)
    for token in tokens:
        if token and token[0].isupper() and "%" not in token:
            return token.strip()
    return ""


def _extract_percents(text: str) -> list[float]:
    """Extract all percentage values from row text."""
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)
    return [float(value) for value in matches]


def normalize_row(row: dict[str, str], patch: str) -> dict[str, str]:
    """Normalize and clean row values."""
    return {
        "patch": patch.strip(),
        "champion": row["champion"].strip(),
        "role": row.get("role", "").strip(),
        "winrate": f"{row['winrate']:.2f}" if row.get("winrate") is not None else "",
        "pickrate": f"{row['pickrate']:.2f}" if row.get("pickrate") is not None else "",
        "banrate": f"{row['banrate']:.2f}" if row.get("banrate") is not None else "",
    }


def write_csv(path: str, rows: Iterable[dict[str, str]]) -> None:
    """Write rows to CSV with a fixed schema."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["patch", "champion", "role", "winrate", "pickrate", "banrate"]
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _scroll_to_load_all(driver: webdriver.Chrome) -> None:
    """Scroll to the bottom to trigger lazy-loaded rows."""
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(30):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


if __name__ == "__main__":
    main()

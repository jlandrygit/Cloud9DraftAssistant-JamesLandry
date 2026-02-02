"""Scrape U.GG duo data for champions by role.

Dependencies:
- selenium
- webdriver-manager
- beautifulsoup4
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from collections import defaultdict
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


ROLE_SLUGS = {
    "TOP": "top",
    "JUNGLE": "jungle",
    "MID": "mid",
    "ADC": "adc",
    "SUPPORT": "support",
}

SLUG_OVERRIDES = {
    "Renata Glasc": "renata",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape U.GG duos by role.")
    parser.add_argument(
        "--roles",
        default="data/meta/u_gg_roles.csv",
        help="Path to u_gg_roles.csv",
    )
    parser.add_argument(
        "--output",
        default="data/meta/u_gg_duos.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run Chrome with a visible window (useful for SSL issues).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay (seconds) between page loads to reduce throttling.",
    )
    args = parser.parse_args()

    roles_map = load_roles_csv(args.roles)
    rows = scrape_all_duos(roles_map, headed=args.headed, delay=args.delay)
    write_csv(args.output, rows)


def load_roles_csv(path: str) -> dict[str, set[str]]:
    """Load champion roles from u_gg_roles.csv."""
    roles_map: dict[str, set[str]] = defaultdict(set)
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            champion = str(row.get("champion", "")).strip()
            role = str(row.get("role", "")).strip().upper()
            if not champion or not role:
                continue
            roles_map[champion].add(role)
    return dict(roles_map)


def scrape_all_duos(
    roles_map: dict[str, set[str]], *, headed: bool, delay: float
) -> list[dict[str, str]]:
    """Scrape duos for every champion-role pair."""
    options = Options()
    if not headed:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1400,900")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--allow-insecure-localhost")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    try:
        all_rows: list[dict[str, str]] = []
        for champion, roles in sorted(roles_map.items()):
            for role in sorted(roles):
                role_slug = ROLE_SLUGS.get(role)
                if not role_slug:
                    continue
                print(f"Fetching duos for {champion} {role}")
                url = build_duo_url(champion, role_slug)
                pairs = scrape_duo_page(driver, url, champion=champion, role=role)
                for duo_champion, winrate, matches in pairs:
                    if winrate < 50.0:
                        continue
                    duo_roles = roles_map.get(duo_champion, set()) or {"UNKNOWN"}
                    for duo_role in sorted(duo_roles):
                        all_rows.append(
                            {
                                "champion": champion,
                                "role": role,
                                "duo_champion": duo_champion,
                                "duo_role": duo_role,
                                "winrate": f"{winrate:.2f}",
                                "matches": str(matches),
                            }
                        )
                time.sleep(max(0.0, delay))
        return all_rows
    finally:
        driver.quit()


def build_duo_url(champion: str, role_slug: str) -> str:
    """Build U.GG duo URL for a champion and role."""
    slug = _slugify_champion(champion)
    return (
        "https://u.gg/lol/champions/"
        + slug
        + "/duos?rank=master_plus&role="
        + role_slug
    )


def scrape_duo_page(
    driver: webdriver.Chrome, url: str, *, champion: str, role: str
) -> list[tuple[str, float, int]]:
    """Load a duo page and extract duo winrates."""
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 20)
        wait.until(
            ec.presence_of_element_located(
                (By.CSS_SELECTOR, "div.duos-list-table, [role='table']")
            )
        )
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        rows = extract_best_duos(soup)
        print(f"Found {len(rows)} duos for {champion} {role}")
        if not rows:
            _write_debug_artifacts(champion, role, html, driver)
        return rows
    except TimeoutException:
        try:
            _write_debug_artifacts(champion, role, driver.page_source, driver)
        except Exception:
            pass
        return []


def extract_best_duos(soup: BeautifulSoup) -> list[tuple[str, float, int]]:
    """Extract duo champions, winrates, and matches from a rendered duos page."""
    table = _find_duos_table(soup)
    if table is None:
        return []
    rows: dict[str, tuple[float, int]] = {}  # champion -> (winrate, matches)
    for row in table.select("[role='row']"):
        champion = _extract_champion_from_row(row)
        if not champion:
            continue
        winrate = _extract_winrate(row.get_text(" ", strip=True))
        if winrate is None:
            continue
        matches = _extract_matches_from_row(row)
        # Keep the entry with highest winrate if duplicate champion
        if champion not in rows or winrate > rows[champion][0]:
            rows[champion] = (winrate, matches)
    # Return as (champion, winrate, matches) tuples
    return [(champ, winrate, matches) for champ, (winrate, matches) in sorted(rows.items(), key=lambda item: (-item[1][0], item[0]))]


def _find_duos_table(soup: BeautifulSoup):
    """Locate the duos table container."""
    table = soup.find("div", class_=re.compile(r"duos-list-table"))
    if table:
        return table
    return soup.find("div", attrs={"role": "table"})


def _extract_winrate(text: str) -> float | None:
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*%", text, re.I)
    if not matches:
        return None
    for value in matches:
        try:
            winrate = float(value)
        except ValueError:
            continue
        if winrate >= 100.0:
            continue
        return winrate
    return None


def _extract_champion_from_row(row) -> str:
    name_node = row.select_one("strong.champion-name")
    if name_node:
        return name_node.get_text(strip=True)
    anchor = row.select_one("a[href^='/lol/champions/']")
    if anchor and anchor.get("href"):
        slug = anchor.get("href").split("/lol/champions/")[1].split("/")[0]
        return _unslugify_champion(slug)
    return ""


def _extract_matches_from_row(row) -> int:
    """Extract matches/games count from the duos table row.
    
    Handles comma-separated numbers like "3,983" or "3,983 games".
    """
    # Look for the matches cell (typically has class "matches")
    matches_cell = row.select_one("div.matches, [class*='matches']")
    if matches_cell:
        text = matches_cell.get_text(strip=True)
        # Extract number with optional commas (e.g., "3,983" or "3983")
        match = re.search(r"(\d{1,3}(?:,\d{3})*)", text)
        if match:
            try:
                # Remove commas before converting to int
                num_str = match.group(1).replace(",", "")
                return int(num_str)
            except ValueError:
                pass
    
    # Fallback: search for numbers in the row text
    row_text = row.get_text(" ", strip=True)
    # Look for pattern like "3,983 games" or "3983 games" or just numbers
    match = re.search(r"(\d{1,3}(?:,\d{3})*)\s*games?", row_text, re.I)
    if match:
        try:
            # Remove commas before converting to int
            num_str = match.group(1).replace(",", "")
            return int(num_str)
        except ValueError:
            pass
    
    # Final fallback: try to find any number with commas in the row
    match = re.search(r"(\d{1,3}(?:,\d{3})+)", row_text)
    if match:
        try:
            num_str = match.group(1).replace(",", "")
            return int(num_str)
        except ValueError:
            pass
    
    return 0


def _slugify_champion(name: str) -> str:
    """Normalize champion name for U.GG URLs."""
    if name in SLUG_OVERRIDES:
        return SLUG_OVERRIDES[name]
    slug = name.strip().lower()
    slug = slug.replace("'", "")
    slug = slug.replace(".", "")
    slug = slug.replace("&", "and")
    slug = re.sub(r"\s+", "-", slug)
    return slug


def _unslugify_champion(slug: str) -> str:
    """Convert URL slug to a display name."""
    name = slug.replace("-", " ").strip()
    name = name.replace("and", "&") if name == "nunu & willump" else name
    return " ".join(part.capitalize() if part else part for part in name.split())


def _write_debug_artifacts(
    champion: str, role: str, html: str, driver: webdriver.Chrome
) -> None:
    """Persist HTML/screenshot to help debug empty parses."""
    debug_dir = Path("data/meta/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    safe_champion = re.sub(r"[^a-zA-Z0-9_-]+", "_", champion)
    safe_role = re.sub(r"[^a-zA-Z0-9_-]+", "_", role)
    html_path = debug_dir / f"ugg_duos_{safe_champion}_{safe_role}.html"
    html_path.write_text(html, encoding="utf-8")
    screenshot_path = debug_dir / f"ugg_duos_{safe_champion}_{safe_role}.png"
    try:
        driver.save_screenshot(str(screenshot_path))
    except Exception:
        return


def write_csv(path: str, rows: Iterable[dict[str, str]]) -> None:
    """Write duo rows to CSV with a fixed schema."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["champion", "role", "duo_champion", "duo_role", "winrate", "matches"]
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()

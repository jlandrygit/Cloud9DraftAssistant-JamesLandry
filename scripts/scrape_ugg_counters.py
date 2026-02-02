"""Scrape U.GG counter data for champions by role.

Dependencies:
- selenium
- webdriver-manager
- beautifulsoup4

This script loads the counter pages for each champion-role pair and extracts
the "Best Picks vs" winrates (filtered to > 50%).
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
    "Nunu & Willump": "nunu",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape U.GG counters by role.")
    parser.add_argument(
        "--roles",
        default="data/meta/u_gg_roles.csv",
        help="Path to u_gg_roles.csv",
    )
    parser.add_argument(
        "--output",
        default="data/meta/u_gg_counters.csv",
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
    rows = scrape_all_counters(roles_map, headed=args.headed, delay=args.delay)
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


def scrape_all_counters(
    roles_map: dict[str, set[str]], *, headed: bool, delay: float
) -> list[dict[str, str]]:
    """Scrape counters for every champion-role pair."""
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
                print(f"Fetching counters for {champion} {role}")
                url = build_counter_url(champion, role_slug)
                rows = scrape_counter_page(driver, url, champion=champion, role=role)
                for counter, winrate, matches in rows:
                    if winrate <= 50.0:
                        continue
                    all_rows.append(
                        {
                            "champion": champion,
                            "role": role,
                            "counter_champion": counter,
                            "winrate": f"{winrate:.2f}",
                            "matches": str(matches),
                        }
                    )
                time.sleep(max(0.0, delay))
        return all_rows
    finally:
        driver.quit()


def build_counter_url(champion: str, role_slug: str) -> str:
    """Build U.GG counter URL for a champion and role."""
    slug = _slugify_champion(champion)
    return (
        "https://u.gg/lol/champions/"
        + slug
        + "/counter?rank=master_plus&role="
        + role_slug
    )


def scrape_counter_page(
    driver: webdriver.Chrome, url: str, *, champion: str, role: str
) -> list[tuple[str, float, int]]:
    """Load a counter page and extract counter winrates."""
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 20)
        wait.until(
            ec.presence_of_element_located(
                (By.XPATH, "//*[contains(text(), 'WR')]")
            )
        )
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        rows = extract_best_counters(soup)
        print(f"Found {len(rows)} counters for {champion} {role}")
        if not rows:
            _write_debug_artifacts(champion, role, html, driver)
        return rows
    except TimeoutException:
        return []


def extract_best_counters(soup: BeautifulSoup) -> list[tuple[str, float, int]]:
    """Extract counter champions, winrates, and matches from a rendered counter page.
    
    Each counter is in an <a> tag, and the matches are in a nested div with
    classes: mt-[2px] text-accent-gray-100 text-[11px] font-normal whitespace-nowrap
    """
    list_container = _find_best_picks_list(soup) or soup
    rows: dict[str, tuple[float, int]] = {}  # champion -> (winrate, matches)
    
    # Process each <a> tag individually - each <a> is a counter entry
    for anchor in list_container.select('a[href^="/lol/champions/"]'):
        text = anchor.get_text(" ", strip=True)
        winrate = _extract_winrate(text)
        if winrate is None:
            continue
        
        champion = _extract_champion_from_anchor(anchor)
        if not champion:
            continue
        
        # Extract matches from THIS specific anchor's nested div structure
        # The matches are in a div with classes: mt-[2px] text-accent-gray-100 text-[11px] font-normal whitespace-nowrap
        matches = _extract_matches_from_anchor(anchor)
        
        # Keep the entry with highest winrate if duplicate champion
        if champion not in rows or winrate > rows[champion][0]:
            rows[champion] = (winrate, matches)
    
    # Return as (champion, winrate, matches) tuples
    return [(champ, winrate, matches) for champ, (winrate, matches) in sorted(rows.items(), key=lambda item: (-item[1][0], item[0]))]


def _find_best_picks_list(soup: BeautifulSoup):
    """Locate the DOM list that holds the 'Best Picks vs' rows."""
    header = soup.find(string=re.compile(r"Best Picks vs", re.I))
    if header is None:
        return None
    card = header.find_parent("div")
    if card is None:
        return None
    # The list container is typically the next sibling with overflow scrolling.
    sibling = card.find_next_sibling("div")
    if sibling and "overflow" in " ".join(sibling.get("class", [])):
        return sibling
    # Fallback: search up the tree and locate any overflow list nearby.
    ancestor = card
    for _ in range(5):
        ancestor = ancestor.find_parent("div")
        if ancestor is None:
            break
        candidate = ancestor.find("div", class_=re.compile(r"overflow-(auto|scroll)"))
        if candidate:
            return candidate
    return None


def _extract_winrate(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*WR", text, re.I)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _extract_champion_from_anchor(anchor) -> str:
    name_node = anchor.select_one("div.text-white.font-bold") or anchor.select_one(
        "div.font-bold.truncate"
    )
    if name_node:
        return name_node.get_text(strip=True)
    href = anchor.get("href", "")
    if "/lol/champions/" in href:
        slug = href.split("/lol/champions/")[1].split("/")[0]
        return _unslugify_champion(slug)
    return ""


def _extract_matches_from_row(row) -> int:
    """Extract matches/games count from a specific row element.
    
    This function searches within a specific row to find the matches count
    associated with that row's counter champion.
    
    Handles comma-separated numbers like "3,983" or "3,983 games".
    """
    # Strategy 1: Look for divs with specific classes that match the matches cell
    # The matches div has classes: mt-[2px], text-accent-gray-100, text-[11px], font-normal, whitespace-nowrap
    for div in row.find_all("div"):
        classes = " ".join(div.get("class", []))
        # Check if this div has the characteristic classes of a matches cell
        if ("text-[" in classes or "text-accent" in classes) and ("whitespace-nowrap" in classes or "font-normal" in classes):
            text = div.get_text(strip=True)
            # Pattern: "118 games" or "3,983 games" - handle comma-separated numbers
            match = re.search(r"(\d{1,3}(?:,\d{3})*)\s*games?", text, re.I)
            if match:
                try:
                    # Remove commas before converting to int
                    num_str = match.group(1).replace(",", "")
                    num = int(num_str)
                    if 1 <= num <= 100000:  # Reasonable range for game counts
                        return num
                except ValueError:
                    pass
    
    # Strategy 2: Look for text containing "games" in this specific row
    text = row.get_text(" ", strip=True)
    # Find all "XXX games" or "X,XXX games" patterns in this row - handle comma-separated numbers
    matches = re.finditer(r"(\d{1,3}(?:,\d{3})*)\s*games?", text, re.I)
    for match_obj in matches:
        try:
            # Remove commas before converting to int
            num_str = match_obj.group(1).replace(",", "")
            num = int(num_str)
            if 1 <= num <= 100000:
                # Check context - make sure it's not part of a winrate percentage
                start = max(0, match_obj.start() - 10)
                end = min(len(text), match_obj.end() + 10)
                context = text[start:end]
                # If we see "WR" or "%" very close, this might be a winrate, skip it
                if "WR" not in context[:5] and "%" not in context[:3]:
                    return num
        except ValueError:
            continue
    
    # Strategy 3: Look for cells (rt-td) in this row that might contain matches
    for cell in row.select("div.rt-td, [role='cell']"):
        cell_text = cell.get_text(strip=True)
        match = re.search(r"(\d+)\s*games?", cell_text, re.I)
        if match:
            try:
                num = int(match.group(1))
                if 1 <= num <= 100000:
                    return num
            except ValueError:
                pass
    
    return 0


def _extract_matches_from_anchor(anchor) -> int:
    """Extract matches/games count from within the anchor's nested div structure.
    
    Based on U.GG HTML structure:
    - Each counter is in an <a> tag
    - The matches are in a div with classes: mt-[2px] text-accent-gray-100 text-[11px] font-normal whitespace-nowrap
    - The div contains text like "206" or "3,983" and "games" (can be separate text nodes)
    
    Handles comma-separated numbers like "3,983" or "3,983 games".
    """
    # Look for the div with the specific classes that contain matches
    # The div has classes: mt-[2px] text-accent-gray-100 text-[11px] font-normal whitespace-nowrap
    # Search within the anchor for divs with these classes
    for div in anchor.find_all("div"):
        classes = " ".join(div.get("class", []))
        # Check if this div has the characteristic classes of the matches cell
        # Looking for: mt-[2px], text-accent-gray-100, text-[11px], font-normal, whitespace-nowrap
        has_mt = "mt-[" in classes or "mt-" in classes
        has_text_accent = "text-accent-gray" in classes or "text-accent" in classes
        has_text_size = "text-[11px]" in classes or "text-11px" in classes
        has_whitespace = "whitespace-nowrap" in classes
        
        # If it has most of these characteristics, check if it contains "games"
        if (has_text_accent or has_text_size) and has_whitespace:
            text = div.get_text(" ", strip=True)
            # Pattern: "206 games" or "3,983 games" or "3,983\ngames" - handle comma-separated numbers
            match = re.search(r"(\d{1,3}(?:,\d{3})*)\s*games?", text, re.I)
            if match:
                try:
                    # Remove commas before converting to int
                    num_str = match.group(1).replace(",", "")
                    num = int(num_str)
                    if 1 <= num <= 100000:  # Reasonable range for game counts
                        return num
                except ValueError:
                    pass
    
    # Fallback: search all divs within the anchor for one containing "games"
    for div in anchor.find_all("div"):
        text = div.get_text(" ", strip=True)
        # Look for pattern with "games" keyword - handle comma-separated numbers
        match = re.search(r"(\d{1,3}(?:,\d{3})*)\s*games?", text, re.I)
        if match:
            try:
                # Remove commas before converting to int
                num_str = match.group(1).replace(",", "")
                num = int(num_str)
                if 1 <= num <= 100000:
                    # Make sure this isn't part of a winrate (check for WR or % nearby)
                    if "WR" not in text and "%" not in text:
                        return num
            except ValueError:
                pass
    
    return 0


def _slugify_champion(name: str) -> str:
    """Normalize champion name for U.GG URLs."""
    if name in SLUG_OVERRIDES:
        return SLUG_OVERRIDES[name]
    slug = name.strip().lower()
    # Remove apostrophes per request (e.g. K'Sante -> ksante).
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
    html_path = debug_dir / f"ugg_counters_{safe_champion}_{safe_role}.html"
    html_path.write_text(html, encoding="utf-8")
    screenshot_path = debug_dir / f"ugg_counters_{safe_champion}_{safe_role}.png"
    try:
        driver.save_screenshot(str(screenshot_path))
    except Exception:
        return


def write_csv(path: str, rows: Iterable[dict[str, str]]) -> None:
    """Write counter rows to CSV with a fixed schema."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["champion", "role", "counter_champion", "winrate", "matches"]
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()

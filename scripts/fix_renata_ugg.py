"""Fetch U.GG counters/duos for Renata Glasc using the renata slug."""

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

RENATA_NAME = "Renata Glasc"
RENATA_SLUG = "renata"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append U.GG counters/duos for Renata Glasc."
    )
    parser.add_argument(
        "--roles",
        default="data/meta/u_gg_roles.csv",
        help="Path to u_gg_roles.csv",
    )
    parser.add_argument(
        "--counters",
        default="data/meta/u_gg_counters.csv",
        help="Counters CSV path",
    )
    parser.add_argument(
        "--duos",
        default="data/meta/u_gg_duos.csv",
        help="Duos CSV path",
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
    roles = sorted(roles_map.get(RENATA_NAME, set()))
    if not roles:
        raise ValueError(f"No roles found for {RENATA_NAME} in {args.roles}.")

    driver = _build_driver(headed=args.headed)
    try:
        counter_rows = fetch_counters(driver, roles, delay=args.delay)
        duo_rows = fetch_duos(driver, roles, roles_map, delay=args.delay)
    finally:
        driver.quit()

    append_rows(
        args.counters,
        counter_rows,
        key_fields=("champion", "role", "counter_champion"),
    )
    append_rows(
        args.duos,
        duo_rows,
        key_fields=("champion", "role", "duo_champion", "duo_role"),
    )


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


def fetch_counters(
    driver: webdriver.Chrome, roles: list[str], *, delay: float
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for role in roles:
        role_slug = ROLE_SLUGS.get(role)
        if not role_slug:
            continue
        print(f"Fetching counters for {RENATA_NAME} {role}")
        url = f"https://u.gg/lol/champions/{RENATA_SLUG}/counter?rank=master_plus&role={role_slug}"
        pairs = _scrape_counters(driver, url)
        print(f"Found {len(pairs)} counters for {RENATA_NAME} {role}")
        for counter, winrate in pairs:
            if winrate <= 50.0:
                continue
            rows.append(
                {
                    "champion": RENATA_NAME,
                    "role": role,
                    "counter_champion": counter,
                    "winrate": f"{winrate:.2f}",
                }
            )
        time.sleep(max(0.0, delay))
    return rows


def fetch_duos(
    driver: webdriver.Chrome,
    roles: list[str],
    roles_map: dict[str, set[str]],
    *,
    delay: float,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for role in roles:
        role_slug = ROLE_SLUGS.get(role)
        if not role_slug:
            continue
        print(f"Fetching duos for {RENATA_NAME} {role}")
        url = f"https://u.gg/lol/champions/{RENATA_SLUG}/duos?rank=master_plus&role={role_slug}"
        pairs = _scrape_duos(driver, url)
        print(f"Found {len(pairs)} duos for {RENATA_NAME} {role}")
        for duo, winrate in pairs:
            if winrate < 50.0:
                continue
            duo_roles = roles_map.get(duo, set()) or {"UNKNOWN"}
            for duo_role in sorted(duo_roles):
                rows.append(
                    {
                        "champion": RENATA_NAME,
                        "role": role,
                        "duo_champion": duo,
                        "duo_role": duo_role,
                        "winrate": f"{winrate:.2f}",
                    }
                )
        time.sleep(max(0.0, delay))
    return rows


def _scrape_counters(driver: webdriver.Chrome, url: str) -> list[tuple[str, float]]:
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 20)
        wait.until(ec.presence_of_element_located((By.XPATH, "//*[contains(text(), 'WR')]")))
        soup = BeautifulSoup(driver.page_source, "html.parser")
        container = _find_best_picks_list(soup) or soup
        rows: dict[str, float] = {}
        for anchor in container.select('a[href^="/lol/champions/"]'):
            text = anchor.get_text(" ", strip=True)
            winrate = _extract_wr(text, require_wr=True)
            if winrate is None:
                continue
            champion = _extract_champion_from_anchor(anchor)
            if not champion:
                continue
            rows[champion] = max(rows.get(champion, 0.0), winrate)
        return sorted(rows.items(), key=lambda item: (-item[1], item[0]))
    except TimeoutException:
        return []


def _scrape_duos(driver: webdriver.Chrome, url: str) -> list[tuple[str, float]]:
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 20)
        wait.until(
            ec.presence_of_element_located(
                (By.CSS_SELECTOR, "div.duos-list-table, [role='table']")
            )
        )
        soup = BeautifulSoup(driver.page_source, "html.parser")
        table = _find_duos_table(soup)
        if table is None:
            return []
        rows: dict[str, float] = {}
        for row in table.select("[role='row']"):
            champion = _extract_champion_from_row(row)
            if not champion:
                continue
            winrate = _extract_wr(row.get_text(" ", strip=True), require_wr=False)
            if winrate is None or winrate >= 100.0:
                continue
            rows[champion] = max(rows.get(champion, 0.0), winrate)
        return sorted(rows.items(), key=lambda item: (-item[1], item[0]))
    except TimeoutException:
        return []


def _find_best_picks_list(soup: BeautifulSoup):
    header = soup.find(string=re.compile(r"Best Picks vs", re.I))
    if header is None:
        return None
    card = header.find_parent("div")
    if card is None:
        return None
    sibling = card.find_next_sibling("div")
    if sibling and "overflow" in " ".join(sibling.get("class", [])):
        return sibling
    ancestor = card
    for _ in range(5):
        ancestor = ancestor.find_parent("div")
        if ancestor is None:
            break
        candidate = ancestor.find("div", class_=re.compile(r"overflow-(auto|scroll)"))
        if candidate:
            return candidate
    return None


def _find_duos_table(soup: BeautifulSoup):
    table = soup.find("div", class_=re.compile(r"duos-list-table"))
    if table:
        return table
    return soup.find("div", attrs={"role": "table"})


def _extract_wr(text: str, *, require_wr: bool) -> float | None:
    pattern = r"(\d+(?:\.\d+)?)\s*%\s*WR" if require_wr else r"(\d+(?:\.\d+)?)\s*%"
    match = re.search(pattern, text, re.I)
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


def _extract_champion_from_row(row) -> str:
    name_node = row.select_one("strong.champion-name")
    if name_node:
        return name_node.get_text(strip=True)
    anchor = row.select_one("a[href^='/lol/champions/']")
    if anchor and anchor.get("href"):
        slug = anchor.get("href").split("/lol/champions/")[1].split("/")[0]
        return _unslugify_champion(slug)
    return ""


def _unslugify_champion(slug: str) -> str:
    name = slug.replace("-", " ").strip()
    name = name.replace("and", "&") if name == "nunu & willump" else name
    return " ".join(part.capitalize() if part else part for part in name.split())


def _build_driver(*, headed: bool) -> webdriver.Chrome:
    options = Options()
    if not headed:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1400,900")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--allow-insecure-localhost")
    options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def append_rows(
    path: str,
    rows: Iterable[dict[str, str]],
    *,
    key_fields: tuple[str, ...],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing_keys(target, key_fields)
    rows = list(rows)
    if not rows:
        return
    write_header = not target.exists() or target.stat().st_size == 0
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        for row in rows:
            key = tuple(row.get(field, "") for field in key_fields)
            if key in existing:
                continue
            writer.writerow(row)
            existing.add(key)


def _load_existing_keys(path: Path, key_fields: tuple[str, ...]) -> set[tuple[str, ...]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    keys: set[tuple[str, ...]] = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            keys.add(tuple(row.get(field, "") for field in key_fields))
    return keys


if __name__ == "__main__":
    main()

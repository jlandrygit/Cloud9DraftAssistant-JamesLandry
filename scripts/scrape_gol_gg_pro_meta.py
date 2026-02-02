"""Scrape professional League of Legends pick/ban data from gol.gg tournament pages.

This script scrapes pick and ban statistics from 4 major professional tournaments:
- LPL 2026 Split 1
- LCK Cup 2026
- LEC 2026 Versus Season
- LCS 2026 Lock-In

The data is stored in a CSV with columns: champion, tournament, picks, bans, total_games
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


TOURNAMENT_URLS = {
    "LPL 2026 Split 1": "https://gol.gg/tournament/tournament-picksandbans/LPL%202026%20Split%201/",
    "LCK Cup 2026": "https://gol.gg/tournament/tournament-picksandbans/LCK%20Cup%202026/",
    "LEC 2026 Versus Season": "https://gol.gg/tournament/tournament-picksandbans/LEC%202026%20Versus%20Season/",
    "LCS 2026 Lock-In": "https://gol.gg/tournament/tournament-picksandbans/LCS%202026%20Lock-In/",
}


def setup_driver(headed: bool = False) -> webdriver.Chrome:
    """Set up and return a Chrome WebDriver instance."""
    options = Options()
    if not headed:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    
    driver = webdriver.Chrome(options=options)
    return driver


def extract_champion_data_from_table(soup: BeautifulSoup, tournament: str) -> list[dict[str, Any]]:
    """Extract pick/ban/winrate data from the tournament picks & bans table.
    
    The table has two structures:
    1. Links with span containing "Picks : X, Bans : Y" (total picks/bans)
    2. Links with green bar divs showing winrate (role-specific, with picks count after bar)
    
    We extract picks/bans from structure 1, and winrate from structure 2, then match them by champion.
    """
    """Extract pick/ban data from the tournament picks & bans table.
    
    HTML structure:
    <a class="tltip" href="../champion/champion-stats/36/...">
      <img alt="" class="champion_icon" src="../_img/champions_icon/Jayce.png"/>
      <span>
        Bans : 87
        <br/>
        Picks : 3
        <br/>
        Winrate : 66.67%
      </span>
    </a>
    
    Returns a list of dicts with keys: champion, tournament, picks, bans, total_games
    """
    results = []
    
    # Find the table with class "table_list"
    table = soup.find("table", class_="table_list")
    if not table:
        print(f"  Warning: Could not find table_list for {tournament}")
        return results
    
    # Find all table rows in tbody
    tbody = table.find("tbody")
    if not tbody:
        print(f"  Warning: Could not find tbody for {tournament}")
        return results
    
    rows = tbody.find_all("tr")
    
    # First pass: Extract picks/bans from span structure
    picks_bans_data: dict[str, dict[str, int]] = {}  # champion -> {picks, bans}
    
    # Second pass: Extract winrate from green bar structure
    winrate_data: dict[str, float] = {}  # champion -> winrate_pct
    
    # First pass: Extract picks and bans
    for row_idx, row in enumerate(rows):
        cells = row.find_all("td")
        
        for cell_idx, cell in enumerate(cells):
            # Look for champion links with span structure (picks/bans)
            champion_links = cell.find_all("a", href=re.compile(r"/champion/champion-stats/"))
            
            for link in champion_links:
                # Check if this link has the span structure (picks/bans)
                span = link.find("span")
                if not span:
                    continue
                
                # Extract champion name
                champion_name = None
                img = link.find("img", class_="champion_icon")
                if img:
                    img_src = img.get("src", "")
                    match = re.search(r"champions_icon/([^/]+)\.png", img_src)
                    if match:
                        champion_name = match.group(1)
                
                if not champion_name:
                    continue
                
                # Extract picks and bans from span
                picks = 0
                bans = 0
                span_text = span.get_text(separator="\n", strip=True)
                lines = span_text.split("\n")
                
                for line in lines:
                    line = line.strip()
                    if line.startswith("Bans :"):
                        try:
                            bans = int(re.search(r"Bans\s*:\s*(\d+)", line).group(1))
                        except (AttributeError, ValueError):
                            pass
                    elif line.startswith("Picks :"):
                        try:
                            picks = int(re.search(r"Picks\s*:\s*(\d+)", line).group(1))
                        except (AttributeError, ValueError):
                            pass
                
                if picks > 0 or bans > 0:
                    picks_bans_data[champion_name] = {"picks": picks, "bans": bans}
    
    # Second pass: Extract winrates and role-specific picks from green bar structure
    # winrate_data will store: champion -> list of (picks_in_role, winrate_pct) tuples
    for row_idx, row in enumerate(rows):
        cells = row.find_all("td")
        
        for cell_idx, cell in enumerate(cells):
            # Look for green bar divs
            bar_divs = cell.find_all("div", style=re.compile(r"width\s*:\s*25px"))
            
            for div in bar_divs:
                div_style = div.get("style", "")
                if ("height:3px" in div_style or "height: 3px" in div_style) and \
                   ("#BC4247" in div_style or "background-color:#BC4247" in div_style):
                    # Find the green span inside
                    green_span = div.find("span", style=re.compile(r"background-color\s*:\s*#00920E"))
                    if green_span:
                        # Find the champion link in the parent div
                        parent_div = div.find_parent("div", style=re.compile(r"text-align\s*:\s*center"))
                        if parent_div:
                            link = parent_div.find("a", href=re.compile(r"/champion/champion-stats/"))
                            if link:
                                # Extract champion name
                                champion_name = None
                                img = link.find("img", class_=re.compile(r"champion_icon"))
                                if img:
                                    img_src = img.get("src", "")
                                    match = re.search(r"champions_icon/([^/]+)\.png", img_src)
                                    if match:
                                        champion_name = match.group(1)
                                
                                if champion_name:
                                    # Extract winrate from green bar width
                                    span_style = green_span.get("style", "")
                                    width_match = re.search(r"width\s*:\s*(\d+)px", span_style)
                                    if width_match:
                                        width_px = int(width_match.group(1))
                                        winrate_pct = (width_px / 24.0) * 100.0
                                        
                                        # Extract picks count from number after the bar div
                                        picks_in_role = 0
                                        next_text = div.next_sibling
                                        while next_text:
                                            if hasattr(next_text, 'string') and next_text.string:
                                                text = next_text.string.strip()
                                                if text.isdigit():
                                                    picks_in_role = int(text)
                                                    break
                                            next_text = next_text.next_sibling
                                        
                                        # Store (picks_in_role, winrate_pct) tuple
                                        if champion_name not in winrate_data:
                                            winrate_data[champion_name] = []
                                        winrate_data[champion_name].append((picks_in_role, winrate_pct))
    
    # Combine picks/bans with winrates
    results = []
    for champion_name, pb_data in picks_bans_data.items():
        # Calculate total wins from role-specific picks and winrates
        winrate_pct = None
        total_wins = 0.0
        
        if champion_name in winrate_data:
            role_data = winrate_data[champion_name]  # List of (picks_in_role, winrate_pct) tuples
            
            for picks_in_role, wr_pct in role_data:
                if picks_in_role > 0 and wr_pct is not None:
                    wins_in_role = picks_in_role * (wr_pct / 100.0)
                    total_wins += wins_in_role
            
            # Calculate overall winrate: total_wins / total_picks (use total picks from span structure)
            total_picks = pb_data["picks"]
            if total_picks > 0:
                winrate_pct = (total_wins / total_picks) * 100.0
        
        results.append({
            "champion": champion_name,
            "tournament": tournament,
            "picks": pb_data["picks"],
            "bans": pb_data["bans"],
            "total_games": pb_data["picks"] + pb_data["bans"],
            "winrate_pct": winrate_pct,
            "wins": total_wins if winrate_pct is not None else None,
        })
    
    return results


def scrape_tournament(driver: webdriver.Chrome, tournament: str, url: str, debug: bool = False) -> list[dict[str, Any]]:
    """Scrape pick/ban data from a single tournament page."""
    print(f"Fetching data for {tournament}...")
    print(f"  URL: {url}")
    
    try:
        driver.get(url)
        # Wait for the table to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "table_list"))
        )
        
        # Give it a moment for any dynamic content
        time.sleep(2)
        
        # Get page source and parse with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        # Save debug HTML if requested
        if debug:
            debug_dir = Path("data/meta/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_file = debug_dir / f"gol_gg_{tournament.replace(' ', '_').replace('/', '_')}.html"
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(soup.prettify())
            print(f"  Saved debug HTML to {debug_file}")
        
        # Extract data
        data = extract_champion_data_from_table(soup, tournament)
        
        print(f"  Found {len(data)} champion entries")
        if data:
            sample = data[0]
            print(f"  Sample entry: {sample['champion']} - Picks: {sample['picks']}, Bans: {sample['bans']}")
        return data
        
    except Exception as e:
        print(f"  Error scraping {tournament}: {e}")
        import traceback
        traceback.print_exc()
        return []


def aggregate_data(all_data: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate pick/ban/winrate data across all tournaments by champion.
    
    Calculates:
    - Total picks and bans
    - Total wins (from winrate * picks for each tournament)
    - Overall winrate (total_wins / total_picks)
    
    Returns a dict: {champion: {picks: int, bans: int, total_games: int, wins: float, winrate: float}}
    """
    aggregated: dict[str, dict[str, Any]] = {}
    
    for entry in all_data:
        champion = entry["champion"]
        if champion not in aggregated:
            aggregated[champion] = {
                "picks": 0,
                "bans": 0,
                "total_games": 0,
                "wins": 0.0,
            }
        
        picks = entry["picks"]
        bans = entry["bans"]
        winrate_pct = entry.get("winrate_pct")
        wins = entry.get("wins")
        
        aggregated[champion]["picks"] += picks
        aggregated[champion]["bans"] += bans
        aggregated[champion]["total_games"] += picks + bans
        
        # Use wins from entry if available, otherwise calculate from winrate
        if wins is not None:
            aggregated[champion]["wins"] += wins
        elif winrate_pct is not None and picks > 0:
            wins = picks * (winrate_pct / 100.0)
            aggregated[champion]["wins"] += wins
    
    # Calculate overall winrate for each champion
    for champion, stats in aggregated.items():
        total_picks = stats["picks"]
        total_wins = stats["wins"]
        
        if total_picks > 0:
            stats["winrate"] = (total_wins / total_picks) * 100.0
        else:
            stats["winrate"] = 0.0
    
    return aggregated


def save_to_csv(data: list[dict[str, Any]], output_path: Path) -> None:
    """Save scraped data to CSV file with winrate."""
    if not data:
        print("No data to save.")
        return
    
    # Aggregate by champion
    aggregated = aggregate_data(data)
    
    # Write aggregated data with winrate
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["champion", "picks", "bans", "total_games", "wins", "winrate"],
        )
        writer.writeheader()
        
        for champion, stats in sorted(aggregated.items()):
            writer.writerow({
                "champion": champion,
                "picks": stats["picks"],
                "bans": stats["bans"],
                "total_games": stats["total_games"],
                "wins": round(stats["wins"], 2),
                "winrate": round(stats["winrate"], 2),
            })
    
    print(f"\nSaved aggregated data to {output_path}")
    print(f"Total champions: {len(aggregated)}")
    print(f"Total picks: {sum(s['picks'] for s in aggregated.values())}")
    print(f"Total bans: {sum(s['bans'] for s in aggregated.values())}")
    print(f"Champions with winrate data: {sum(1 for s in aggregated.values() if s.get('wins', 0) > 0)}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Scrape professional LoL pick/ban data from gol.gg"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/meta/gol_gg_pro_meta.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run browser in headed mode (visible)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save debug HTML files",
    )
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    driver = setup_driver(headed=args.headed)
    all_data = []
    
    try:
        for tournament, url in TOURNAMENT_URLS.items():
            data = scrape_tournament(driver, tournament, url, debug=args.debug)
            all_data.extend(data)
            time.sleep(1)  # Be polite with requests
        
        if all_data:
            save_to_csv(all_data, output_path)
        else:
            print("No data was scraped. Check the HTML structure or network issues.")
            
    finally:
        driver.quit()


if __name__ == "__main__":
    main()

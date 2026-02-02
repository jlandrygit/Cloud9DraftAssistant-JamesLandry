"""Scrape draft data from gol.gg game pages.

Extracts:
- Patch version (converted from v16.1 to 26.1 format)
- League
- First Pick team
- Last Pick team
- Winner
- Bans in order (BB1, RB1, BB2, RB2, BB3, RB3, RB4, BB4, RB5, BB5)
- Picks in order (BP1, RP1, RP2, BP2, BP3, RP3, RP4, BP4, BP5, RP5)
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def convert_patch_version(patch_str: str) -> str:
    """Convert patch version from v16.1 to 26.1 format.
    
    Args:
        patch_str: Version string like "v16.1" or "v16.2"
    
    Returns:
        Converted version like "26.1" or "26.2"
    """
    match = re.search(r"v(\d+)\.(\d+)", patch_str)
    if match:
        major = int(match.group(1))
        minor = match.group(2)
        # Convert 16 -> 26, 17 -> 27, etc.
        new_major = major + 10
        return f"{new_major}.{minor}"
    return patch_str


def extract_champion_name(element) -> str | None:
    """Extract champion name from an <a> tag.
    
    Tries multiple methods:
    1. title attribute (e.g., "Jayce stats")
    2. img src attribute (e.g., "champions_icon/jayce.png")
    3. Link text
    
    Args:
        element: BeautifulSoup element (typically an <a> tag)
    
    Returns:
        Champion name or None
    """
    # Try title attribute first
    title = element.get("title", "")
    if title:
        # Extract champion name from "ChampionName stats"
        match = re.match(r"^([^ ]+) stats", title)
        if match:
            return match.group(1)
    
    # Try img src attribute
    img = element.find("img")
    if img:
        src = img.get("src", "")
        match = re.search(r"champions_icon/([^/]+)\.png", src)
        if match:
            return match.group(1)
    
    # Try link text
    text = element.get_text(strip=True)
    if text and text not in ["***", "...", ""]:
        return text
    
    return None


def extract_team_data(soup: BeautifulSoup, team_class: str) -> dict[str, any]:
    """Extract team name, win/loss, bans, and picks.
    
    Args:
        soup: BeautifulSoup object
        team_class: Either "blue-line-header" or "red-line-header"
    
    Returns:
        dict with keys: name, won, first_pick, bans, picks
    """
    # Find team header
    header = soup.find("div", class_=team_class)
    if not header:
        return {"name": None, "won": False, "first_pick": False, "bans": [], "picks": []}
    
    # Extract team name
    team_link = header.find("a")
    team_name = team_link.get_text(strip=True) if team_link else None
    
    # Check if team won
    header_text = header.get_text()
    won = "WIN" in header_text
    
    # Find the parent container for this team
    # The structure is: div.col-12.col-sm-6 > div.blue-line-header/red-line-header > ... > div.row (bans/picks)
    team_container = header.find_parent("div", class_=re.compile("col-12"))
    if not team_container:
        team_container = header.find_parent("div", class_=re.compile("col-sm-6"))
    
    # Check for first pick indicator
    first_pick = False
    if team_container:
        first_pick_img = team_container.find("img", alt="First Pick")
        first_pick = first_pick_img is not None
    
    # Find bans and picks sections within this team's container
    bans = []
    picks = []
    
    if team_container:
        # Find all rows within this team's container
        rows = team_container.find_all("div", class_="row")
        for row in rows:
            cols = row.find_all("div", recursive=False)
            if len(cols) >= 2:
                label_col = cols[0]
                champions_col = cols[1]
                
                label_text = label_col.get_text(strip=True)
                
                if label_text == "Bans":
                    # Extract champion links - try multiple selectors
                    champion_links = champions_col.find_all("a", class_="black_link")
                    if not champion_links:
                        # Try without class filter
                        champion_links = champions_col.find_all("a", href=re.compile(r"/champion/"))
                    
                    for link in champion_links:
                        champ = extract_champion_name(link)
                        if champ:
                            bans.append(champ)
                
                elif label_text == "Picks":
                    # Extract champion links - try multiple selectors
                    champion_links = champions_col.find_all("a", class_="black_link")
                    if not champion_links:
                        # Try without class filter
                        champion_links = champions_col.find_all("a", href=re.compile(r"/champion/"))
                    
                    for link in champion_links:
                        champ = extract_champion_name(link)
                        if champ:
                            picks.append(champ)
    
    return {
        "name": team_name,
        "won": won,
        "first_pick": first_pick,
        "bans": bans,
        "picks": picks,
    }


def scrape_game_page(url: str, driver: webdriver.Chrome, debug: bool = False) -> dict[str, any]:
    """Scrape draft data from a single gol.gg game page.
    
    Assumes the page has already been loaded in the driver.
    
    Args:
        url: URL of the game page (for reference)
        driver: Selenium WebDriver instance (page should already be loaded)
        debug: If True, save HTML for debugging
    
    Returns:
        dict with all extracted draft data
    
    Raises:
        Exception: If page cannot be scraped
    """
    # Get page source
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    
    if debug:
        # Save HTML for debugging
        debug_path = Path("data/meta/debug/gol_gg_draft_debug.html")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Saved debug HTML to {debug_path}")
    
    # Extract patch version
    patch = None
    patch_elements = soup.find_all("div", class_=re.compile("col-3"))
    for elem in patch_elements:
        text = elem.get_text(strip=True)
        if text.startswith("v"):
            patch = convert_patch_version(text)
            break
    
    # Extract league
    league = None
    league_links = soup.find_all("a", href=re.compile(r"/tournament/"))
    for link in league_links:
        text = link.get_text(strip=True)
        if text and "2026" in text:
            league = text
            break
    
    # Extract team data
    blue_team = extract_team_data(soup, "blue-line-header")
    red_team = extract_team_data(soup, "red-line-header")
    
    # Determine first and last pick teams
    first_pick_team_data = blue_team if blue_team["first_pick"] else red_team
    last_pick_team_data = red_team if blue_team["first_pick"] else blue_team
    
    first_pick_team = first_pick_team_data["name"]
    last_pick_team = last_pick_team_data["name"]
    
    # Determine winner
    winner = blue_team["name"] if blue_team["won"] else red_team["name"]
    
    # Get bans and picks for first pick team and last pick team
    # BB = First Pick Team Ban, RB = Last Pick Team Ban
    # BP = First Pick Team Pick, RP = Last Pick Team Pick
    first_pick_bans = first_pick_team_data["bans"][:5]  # Take first 5
    last_pick_bans = last_pick_team_data["bans"][:5]  # Take first 5
    first_pick_picks = first_pick_team_data["picks"][:5]  # Take first 5
    last_pick_picks = last_pick_team_data["picks"][:5]  # Take first 5
    
    # Pad with empty strings if needed
    while len(first_pick_bans) < 5:
        first_pick_bans.append("")
    while len(last_pick_bans) < 5:
        last_pick_bans.append("")
    while len(first_pick_picks) < 5:
        first_pick_picks.append("")
    while len(last_pick_picks) < 5:
        last_pick_picks.append("")
    
    # Draft order: BB1, RB1, BB2, RB2, BB3, RB3, BP1, RP1, RP2, BP2, BP3, RP3, RB4, BB4, RB5, BB5, RP4, BP4, BP5, RP5
    # Where:
    # BB = First Pick Team Ban
    # RB = Last Pick Team Ban
    # BP = First Pick Team Pick
    # RP = Last Pick Team Pick
    
    # Build result dict
    result = {
        "Patch": patch or "",
        "League": league or "",
        "First Pick": first_pick_team or "",
        "Last Pick": last_pick_team or "",
        "Winner": winner or "",
        "BB1": first_pick_bans[0],  # First Pick Team Ban 1
        "RB1": last_pick_bans[0],   # Last Pick Team Ban 1
        "BB2": first_pick_bans[1],  # First Pick Team Ban 2
        "RB2": last_pick_bans[1],   # Last Pick Team Ban 2
        "BB3": first_pick_bans[2],  # First Pick Team Ban 3
        "RB3": last_pick_bans[2],   # Last Pick Team Ban 3
        "BP1": first_pick_picks[0], # First Pick Team Pick 1
        "RP1": last_pick_picks[0],  # Last Pick Team Pick 1
        "RP2": last_pick_picks[1],  # Last Pick Team Pick 2
        "BP2": first_pick_picks[1], # First Pick Team Pick 2
        "BP3": first_pick_picks[2], # First Pick Team Pick 3
        "RP3": last_pick_picks[2],  # Last Pick Team Pick 3
        "RB4": last_pick_bans[3],   # Last Pick Team Ban 4
        "BB4": first_pick_bans[3],  # First Pick Team Ban 4
        "RB5": last_pick_bans[4],   # Last Pick Team Ban 5
        "BB5": first_pick_bans[4],  # First Pick Team Ban 5
        "RP4": last_pick_picks[3],  # Last Pick Team Pick 4
        "BP4": first_pick_picks[3], # First Pick Team Pick 4
        "BP5": first_pick_picks[4], # First Pick Team Pick 5
        "RP5": last_pick_picks[4],  # Last Pick Team Pick 5
    }
    
    return result


def scrape_game_by_id(game_id: int, driver: webdriver.Chrome, debug: bool = False) -> dict[str, any] | None:
    """Scrape a game page by game ID.
    
    Args:
        game_id: Game ID number
        driver: Selenium WebDriver instance
        debug: If True, save HTML for debugging
    
    Returns:
        dict with extracted draft data, or None if page not found
    """
    url = f"https://gol.gg/game/stats/{game_id}/page-game/"
    try:
        driver.get(url)
        time.sleep(2)
        
        # Check if page exists by looking for key elements
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        
        # Check for 404 indicators
        page_title = driver.title.lower()
        if "404" in page_title or "not found" in page_title:
            print(f"  Game {game_id} not found (404 in title), skipping...")
            return None
        
        # Check if we can find team headers (indicates valid game page)
        blue_header = soup.find("div", class_="blue-line-header")
        red_header = soup.find("div", class_="red-line-header")
        
        if not blue_header and not red_header:
            # Might be a 404 page or invalid game
            # Check for common error indicators
            if "404" in html.lower()[:1000] or "page not found" in html.lower()[:1000]:
                print(f"  Game {game_id} not found (no team data), skipping...")
                return None
        
        # Try to scrape
        return scrape_game_page(url, driver, debug=debug)
    except Exception as e:
        # For any error, print and continue
        error_msg = str(e).lower()
        if "404" in error_msg or "not found" in error_msg:
            print(f"  Game {game_id} not found, skipping...")
        else:
            print(f"  Error scraping game {game_id}: {e}")
        return None


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Scrape draft data from gol.gg game pages")
    parser.add_argument(
        "--start-id",
        type=int,
        default=73116,
        help="Starting game ID",
    )
    parser.add_argument(
        "--end-id",
        type=int,
        default=73776,
        help="Ending game ID (inclusive)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Single URL to scrape (overrides start-id and end-id)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gol_pro_draft_data.xlsx",
        help="Output Excel file path",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run browser in headed mode (visible)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save debug HTML",
    )
    args = parser.parse_args()
    
    # Setup Chrome driver
    chrome_options = Options()
    if not args.headed:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    driver = webdriver.Chrome(options=chrome_options)
    
    all_game_data = []
    
    try:
        if args.url:
            # Scrape single URL
            print(f"Scraping single URL: {args.url}")
            game_data = scrape_game_page(args.url, driver, debug=args.debug)
            if game_data:
                all_game_data.append(game_data)
        else:
            # Scrape range of game IDs
            print(f"Scraping games {args.start_id} to {args.end_id}...")
            total_games = args.end_id - args.start_id + 1
            found_count = 0
            
            for game_id in range(args.start_id, args.end_id + 1):
                print(f"Processing game {game_id} ({game_id - args.start_id + 1}/{total_games})...")
                try:
                    game_data = scrape_game_by_id(game_id, driver, debug=args.debug)
                    if game_data:
                        all_game_data.append(game_data)
                        found_count += 1
                        print(f"  [OK] Successfully scraped game {game_id}")
                except Exception as e:
                    print(f"  [ERROR] Error scraping game {game_id}: {e}")
                    # Continue to next game
                    continue
            
            print(f"\nScraped {found_count} games out of {total_games} attempted")
        
        if not all_game_data:
            print("No game data collected. Exiting.")
            return
        
        # Create DataFrame
        df = pd.DataFrame(all_game_data)
        
        # Save to Excel
        output_path = Path(args.output)
        df.to_excel(output_path, index=False, engine="openpyxl")
        
        print(f"\nSaved {len(all_game_data)} games to {output_path}")
        print(f"\nSample data (first game):")
        if all_game_data:
            for key, value in all_game_data[0].items():
                print(f"  {key}: {value}")
    
    finally:
        driver.quit()


if __name__ == "__main__":
    main()

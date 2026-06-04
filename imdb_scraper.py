"""
IMDb Top 250 Movie Rating Scraper
----------------------------------
Uses Selenium to scrape dynamic IMDb content.
Saves results to CSV with timestamp for trend tracking.
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import requests
import time
import os
import re
from datetime import datetime


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
IMDB_URL = "https://www.imdb.com/chart/top/"
OUTPUT_DIR = "data"
DATASET_URL_BASE = "https://datasets.imdbws.com"
DATASET_CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
HEADLESS = True          # Set False if you want to see the browser
DELAY = 2                # Seconds to wait for page load


def format_votes(votes):
    if pd.isna(votes):
        return "N/A"
    votes = int(votes)
    if votes >= 1_000_000:
        return f"{votes / 1_000_000:.1f}M"
    if votes >= 1_000:
        return f"{votes / 1_000:.1f}K"
    return str(votes)


def download_dataset_file(filename):
    os.makedirs(DATASET_CACHE_DIR, exist_ok=True)
    path = os.path.join(DATASET_CACHE_DIR, filename)
    if os.path.exists(path):
        return path

    url = f"{DATASET_URL_BASE}/{filename}"
    print(f"[INFO] Downloading IMDb dataset {filename}...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return path


def scrape_imdb_top250_from_datasets(limit=250, min_votes=25000):
    print("[INFO] Falling back to IMDb public datasets...")
    ratings_path = download_dataset_file("title.ratings.tsv.gz")
    df_ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        compression="gzip",
        usecols=["tconst", "averageRating", "numVotes"],
        dtype={"tconst": str, "averageRating": float, "numVotes": int},
        na_values="\\N"
    )
    df_ratings = df_ratings[df_ratings["numVotes"] >= min_votes]
    candidate_count = max(limit * 20, limit + 100)
    df_candidates = df_ratings.sort_values(["averageRating", "numVotes"], ascending=[False, False]).head(candidate_count)

    basics_path = download_dataset_file("title.basics.tsv.gz")
    needed_ids = set(df_candidates["tconst"].tolist())
    basic_chunks = []
    for chunk in pd.read_csv(
        basics_path,
        sep="\t",
        compression="gzip",
        usecols=["tconst", "primaryTitle", "startYear", "runtimeMinutes", "titleType", "isAdult"],
        dtype=str,
        na_values="\\N",
        chunksize=500000,
    ):
        matched = chunk[
            chunk["tconst"].isin(needed_ids) &
            (chunk["titleType"] == "movie") &
            (chunk["isAdult"] == "0")
        ]
        if not matched.empty:
            basic_chunks.append(matched)
    df_basics = pd.concat(basic_chunks, ignore_index=True) if basic_chunks else pd.DataFrame(
        columns=["tconst", "primaryTitle", "startYear", "runtimeMinutes", "titleType", "isAdult"]
    )

    df = pd.merge(df_candidates, df_basics, on="tconst", how="inner")
    df = df.sort_values(["averageRating", "numVotes"], ascending=[False, False]).head(limit)
    df["title"] = df["primaryTitle"].fillna("Unknown")
    df["year"] = df["startYear"].fillna("N/A").replace("\\N", "N/A")
    df["duration"] = df["runtimeMinutes"].fillna("N/A").replace("\\N", "N/A")
    df["duration"] = df["duration"].apply(lambda x: f"{x} min" if x not in ["N/A", None, ""] else "N/A")
    df["votes"] = df["numVotes"].apply(format_votes)
    df["rank"] = range(1, len(df) + 1)
    df["director"] = "Unknown"
    df["movie_url"] = "N/A"
    df = df[["rank", "title", "year", "duration", "averageRating", "votes", "director", "movie_url"]]
    df = df.rename(columns={"averageRating": "rating"})
    return df


def scrape_director_from_movie_page(movie_url, driver):
    """Scrapes director info from individual IMDb movie page."""
    try:
        driver.get(movie_url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='title-pc-headline']"))
        )
        time.sleep(0.5)
        
        # Try to find director link - usually in the header credits
        director = "Unknown"
        try:
            # Look for director in the credits section
            director_link = driver.find_element(By.CSS_SELECTOR, "a[href*='/name/'] span")
            if director_link:
                director = director_link.text.strip()
        except:
            try:
                # Fallback: look for director text
                credits = driver.find_element(By.CSS_SELECTOR, "[data-testid='title-pc-principal-credit']")
                director_elem = credits.find_element(By.TAG_NAME, "a")
                director = director_elem.text.strip()
            except:
                pass
        
        return director
    except Exception as exc:
        print(f"  [WARN] Could not scrape director: {exc}")
        return "Unknown"


def scrape_imdb_top250_page(headless=True, scrape_directors=True):
    driver = create_driver(headless)
    movies = []
    try:
        print("[INFO] Opening IMDb Top 250 page...")
        driver.get(IMDB_URL)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "tbody.lister-list tr"))
        )
        time.sleep(DELAY)

        rows = driver.find_elements(By.CSS_SELECTOR, "tbody.lister-list tr")
        print(f"[INFO] Extracting {len(rows)} rows...")
        for idx, row in enumerate(rows):
            try:
                title_cell = row.find_element(By.CSS_SELECTOR, "td.titleColumn")
                title_link = title_cell.find_element(By.TAG_NAME, "a")
                title = title_link.text.strip()
                movie_url = title_link.get_attribute("href")
                if not movie_url.startswith("http"):
                    movie_url = "https://www.imdb.com" + movie_url
                
                rank = int(title_cell.text.split(".")[0].strip())
                year = title_cell.find_element(By.CSS_SELECTOR, "span.secondaryInfo").text.strip("()")

                rating_element = row.find_element(By.CSS_SELECTOR, "td.ratingColumn.imdbRating strong")
                rating = float(rating_element.text.strip())
                title_attr = rating_element.get_attribute("title") or ""
                votes_match = re.search(r"based on ([\d,]+) user ratings", title_attr)
                votes = format_votes(int(votes_match.group(1).replace(",", ""))) if votes_match else "N/A"

                # Scrape director info
                director = "Unknown"
                if scrape_directors:
                    print(f"  [INFO] Scraping director for {title}... ({idx+1}/{len(rows)})")
                    director = scrape_director_from_movie_page(movie_url, driver)

                movies.append({
                    "rank": rank,
                    "title": title,
                    "year": year or "N/A",
                    "duration": "N/A",
                    "rating": rating,
                    "votes": votes,
                    "director": director,
                    "movie_url": movie_url,
                })
            except Exception as exc:
                print(f"  [WARN] Skipped row: {exc}")
        if len(movies) >= 200:
            return pd.DataFrame(movies)
        raise RuntimeError("IMDb page scraper returned too few rows")
    finally:
        driver.quit()


# ──────────────────────────────────────────────
# SETUP DRIVER
# ──────────────────────────────────────────────
def create_driver(headless=True):
    """Creates and returns a configured Chrome WebDriver."""
    options = Options()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    # Spoof as a real browser to avoid bot detection
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver


# ──────────────────────────────────────────────
# SCRAPER
# ──────────────────────────────────────────────
def scrape_imdb_top250(headless=True, scrape_directors=False):
    """
    Scrapes IMDb Top 250 movies.

    Args:
        headless: Run browser in headless mode
        scrape_directors: Scrape individual director info (slower but more complete)

    Returns:
        pd.DataFrame with columns:
        rank, title, year, duration, rating, votes, director, movie_url
    """
    try:
        return scrape_imdb_top250_page(headless=headless, scrape_directors=scrape_directors)
    except Exception as exc:
        print(f"[WARN] Page scraping failed: {exc}")
        print("[INFO] Falling back to IMDb public datasets")
        return scrape_imdb_top250_from_datasets()


# ──────────────────────────────────────────────
# SAVE WITH TIMESTAMP (for trend tracking)
# ──────────────────────────────────────────────
def save_data(df: pd.DataFrame):
    """
    Saves the scraped data to two CSVs:
      1. imdb_top250_latest.csv  → always the freshest snapshot
      2. imdb_top250_YYYYMMDD.csv → archived snapshot for trend analysis
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    today = datetime.now().strftime("%Y%m%d")
    df["scraped_date"] = today

    # Latest snapshot (overwritten each run)
    latest_path = os.path.join(OUTPUT_DIR, "imdb_top250_latest.csv")
    df.to_csv(latest_path, index=False)
    print(f"💾 Saved → {latest_path}")

    # Archived snapshot
    archive_path = os.path.join(OUTPUT_DIR, f"imdb_top250_{today}.csv")
    if not os.path.exists(archive_path):
        df.to_csv(archive_path, index=False)
        print(f"📁 Archived → {archive_path}")
    else:
        print(f"📁 Archive for today already exists: {archive_path}")

    return latest_path


# ──────────────────────────────────────────────
# TREND COMPARISON (Unique Feature #1)
# ──────────────────────────────────────────────
def compare_snapshots(file1: str, file2: str):
    """
    Compares two dated CSV snapshots and highlights:
    - Movies that moved UP in rank
    - Movies that moved DOWN in rank
    - New entrants (entered Top 250)
    - Drop-outs (left Top 250)
    - Rating changes

    Args:
        file1: path to older snapshot CSV
        file2: path to newer snapshot CSV
    Returns:
        dict with keys: moved_up, moved_down, new_entrants, drop_outs, rating_changes
    """
    df1 = pd.read_csv(file1).set_index("title")
    df2 = pd.read_csv(file2).set_index("title")

    common = df1.index.intersection(df2.index)
    rank_change = (df1.loc[common, "rank"] - df2.loc[common, "rank"]).sort_values(ascending=False)

    moved_up   = rank_change[rank_change > 0].head(10)
    moved_down = rank_change[rank_change < 0].tail(10)

    new_entrants = df2.index.difference(df1.index).tolist()
    drop_outs    = df1.index.difference(df2.index).tolist()

    rating_changes = (df2.loc[common, "rating"] - df1.loc[common, "rating"])
    rating_changes = rating_changes[rating_changes != 0].sort_values(ascending=False)

    report = {
        "moved_up":       moved_up,
        "moved_down":     moved_down,
        "new_entrants":   new_entrants,
        "drop_outs":      drop_outs,
        "rating_changes": rating_changes,
    }

    print("\nTREND REPORT")
    print("=" * 40)
    print("Most improved ranks:\n", moved_up.to_string())
    print("\nMost dropped ranks:\n", moved_down.to_string())
    print(f"\nNew entrants ({len(new_entrants)}): {new_entrants[:5]}")
    print(f"\nDrop-outs ({len(drop_outs)}): {drop_outs[:5]}")
    print("\nRating changes:\n", rating_changes.head(10).to_string())

    return report


# ──────────────────────────────────────────────
# QUICK STATS (Unique Feature #2)
# ──────────────────────────────────────────────
def generate_stats(df: pd.DataFrame):
    """Prints summary statistics about the scraped dataset."""
    df["year_num"] = pd.to_numeric(df["year"].str.extract(r"(\d{4})")[0], errors="coerce")

    print("\nQUICK STATS")
    print("=" * 40)
    print(f"Total movies: {len(df)}")
    print(f"Average rating: {df['rating'].mean():.2f}")
    if len(df) > 0:
        print(f"Highest rated: {df.iloc[0]['title']} ({df.iloc[0]['rating']})")
    else:
        print("Highest rated: N/A (no data)")
    decade_series = df['year_num'].dropna().apply(lambda y: (int(y)//10)*10).value_counts()
    top_decade = f"{int(decade_series.idxmax())}s" if not decade_series.empty else "N/A"
    print(f"Top decade: {top_decade}")

    year_valid = df['year_num'].dropna()
    if len(year_valid) > 0:
        oldest_idx = year_valid.idxmin()
        newest_idx = year_valid.idxmax()
        print(f"Oldest movie: {df.loc[oldest_idx, 'title']} ({int(df.loc[oldest_idx, 'year_num'])})")
        print(f"Newest movie: {df.loc[newest_idx, 'title']} ({int(df.loc[newest_idx, 'year_num'])})")
    else:
        print("Oldest movie: N/A (no year data)")
        print("Newest movie: N/A (no year data)")

    decade_counts = df["year_num"].dropna().apply(lambda y: (int(y)//10)*10).value_counts().sort_index()
    if len(decade_counts) > 0:
        decade_labels = {int(k): f"{int(k)}s" for k in decade_counts.index}
        decade_counts.index = decade_counts.index.map(decade_labels)
        print("\nMovies per decade:")
        print(decade_counts.to_string())
    else:
        print("\nMovies per decade: N/A (no year data)")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    df = scrape_imdb_top250(headless=HEADLESS, scrape_directors=True)
    save_data(df)
    generate_stats(df)
    print("\nDone! Open data/imdb_top250_latest.csv to view results.")

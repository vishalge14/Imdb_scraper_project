"""
Utility functions for IMDb dashboard:
- Movie recommendation engine
- Director stats and analytics
- Trend analysis
"""

import pandas as pd
import os
import requests
import warnings
from datetime import datetime

DATASET_URL_BASE = "https://datasets.imdbws.com"
DATA_CACHE_DIR = "data/cache"


def _download_dataset_file(filename: str, cache_dir: str = DATA_CACHE_DIR) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, filename)
    if os.path.exists(path):
        return path
    url = f"{DATASET_URL_BASE}/{filename}"
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return path


def _load_title_basics(cache_dir: str = DATA_CACHE_DIR) -> pd.DataFrame:
    path = _download_dataset_file("title.basics.tsv.gz", cache_dir)
    return pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        usecols=["tconst", "primaryTitle", "startYear", "titleType", "isAdult"],
        dtype=str,
        na_values="\\N",
    )


def _load_title_crew(cache_dir: str = DATA_CACHE_DIR) -> pd.DataFrame:
    path = _download_dataset_file("title.crew.tsv.gz", cache_dir)
    return pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        usecols=["tconst", "directors"],
        dtype=str,
        na_values="\\N",
    )


def _load_names_for_ids(nconst_ids: set[str], cache_dir: str = DATA_CACHE_DIR) -> dict[str, str]:
    path = _download_dataset_file("name.basics.tsv.gz", cache_dir)
    result: dict[str, str] = {}
    for chunk in pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        usecols=["nconst", "primaryName"],
        dtype=str,
        na_values="\\N",
        chunksize=500000,
    ):
        subset = chunk[chunk["nconst"].isin(nconst_ids)]
        if not subset.empty:
            result.update(subset.set_index("nconst")["primaryName"].to_dict())
        if len(result) >= len(nconst_ids):
            break
    return result


def _build_recommendation(df: pd.DataFrame, mood: str, max_results: int = 5, reason: str | None = None) -> str:
    df = df.copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
    df["year_num"] = pd.to_numeric(df["year"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce").fillna(0)

    mood_lower = mood.strip().lower()
    theme = "great IMDb picks"
    score = df["rating"] * 10.0

    if any(token in mood_lower for token in ["sci-fi", "science fiction", "space", "future", "alien", "robot"]):
        theme = "sci-fi and futuristic energy"
        score += df["title"].astype(str).str.lower().str.contains(
            "alien|space|star|future|robot|matrix|planet|galaxy|war|mission",
            regex=True,
            na=False,
        ).astype(float) * 1.5
    elif any(token in mood_lower for token in ["action", "thrill", "adrenaline", "fight", "chase", "battle", "explosive", "intense"]):
        theme = "fast-paced action and excitement"
        score += df["title"].astype(str).str.lower().str.contains(
            "war|battle|fight|run|mission|heat|speed|force|dark",
            regex=True,
            na=False,
        ).astype(float) * 1.5
    elif any(token in mood_lower for token in ["comedy", "funny", "laugh", "humor"]):
        theme = "light-hearted comedy and charm"
        score += df["title"].astype(str).str.lower().str.contains(
            "love|life|day|crazy|fun|laugh|family|hotel|mask",
            regex=True,
            na=False,
        ).astype(float) * 1.5
    elif any(token in mood_lower for token in ["drama", "emotional", "heart", "feel", "tears"]):
        theme = "emotional drama and depth"
        score += df["title"].astype(str).str.lower().str.contains(
            "woman|man|love|story|life|secret|family|king|queen|city",
            regex=True,
            na=False,
        ).astype(float) * 1.5
    elif any(token in mood_lower for token in ["romance", "love", "heartwarming", "romantic"]):
        theme = "romantic storytelling"
        score += df["title"].astype(str).str.lower().str.contains(
            "love|heart|romeo|juliet|affair|before|after|bridge",
            regex=True,
            na=False,
        ).astype(float) * 1.5
    elif any(token in mood_lower for token in ["horror", "scary", "terrifying", "spooky", "monster"]):
        theme = "thrilling horror and tension"
        score += df["title"].astype(str).str.lower().str.contains(
            "night|dark|silence|house|nightmare|halloween|devil|ghost|shutter",
            regex=True,
            na=False,
        ).astype(float) * 1.5

    if any(token in mood_lower for token in ["classic", "timeless", "vintage", "old school", "golden age"]):
        score += ((1950 - df["year_num"]).clip(lower=0) / 50.0)
    elif any(token in mood_lower for token in ["modern", "new", "recent", "fresh", "now"]):
        score += ((df["year_num"] - 2000).clip(lower=0) / 50.0)

    df["score"] = score
    top_movies = df.sort_values(["score", "rating", "year_num"], ascending=[False, False, False]).head(max_results)

    lines = ["🎬 Local recommendation engine"]
    if reason:
        lines.append(f"Reason: {reason}")
    lines.append(f"Mood: {mood}")
    lines.append("")

    if top_movies.empty:
        lines.append("No recommendations could be generated from the current dataset.")
        return "\n".join(lines)

    for rank, row in enumerate(top_movies.itertuples(index=False), start=1):
        director = getattr(row, "director", "Unknown")
        title = getattr(row, "title", "Unknown")
        year = getattr(row, "year", "Unknown")
        rating = getattr(row, "rating", 0.0)
        lines.append(f"{rank}. {title} ({year}) — ⭐ {rating}/10 — Director: {director}")
        lines.append(f"   Why: This movie is a strong match for the mood you described and reflects the dataset's top IMDb storytelling.")

    return "\n".join(lines)


def get_ai_recommendation(df: pd.DataFrame, mood: str, max_results: int = 5) -> str:
    """
    Generates movie recommendations locally from the IMDb Top 250 dataset.

    Args:
        df: DataFrame with movies (must have: title, rating, year, director)
        mood: User's desired mood/preference
        max_results: Number of recommendations to return

    Returns:
        A simple dataset-driven recommendation list.
    """
    return _build_recommendation(
        df,
        mood,
        max_results,
        reason="Local recommendations generated from IMDb Top 250 data. No external AI service is required.",
    )


# ──────────────────────────────────────────────
# DIRECTOR HALL OF FAME
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# DIRECTOR HALL OF FAME
# ──────────────────────────────────────────────

def enrich_director_data(df: pd.DataFrame, cache_dir: str = DATA_CACHE_DIR) -> pd.DataFrame:
    if "director" not in df.columns:
        return df

    unknown_mask = df["director"].isna() | (df["director"] == "Unknown")
    if not unknown_mask.any():
        return df

    try:
        title_basics = _load_title_basics(cache_dir)
        title_basics = title_basics[
            (title_basics["titleType"] == "movie") &
            (title_basics["isAdult"] == "0")
        ][["tconst", "primaryTitle", "startYear"]]
    except Exception:
        return df

    df_copy = df.copy()
    df_copy["year"] = df_copy["year"].astype(str).str.extract(r"(\d{4})")[0]
    title_basics = title_basics.rename(columns={"primaryTitle": "title", "startYear": "year"})

    merged = df_copy.merge(title_basics, on=["title", "year"], how="left")
    if merged["tconst"].isna().all():
        return df

    try:
        title_crew = _load_title_crew(cache_dir)
    except Exception:
        return df

    merged = merged.merge(title_crew, on="tconst", how="left")
    merged["director_id"] = merged["directors"].astype(str).str.split(",").str[0]
    director_ids = set(merged["director_id"].dropna().unique())
    if not director_ids:
        return df

    try:
        name_map = _load_names_for_ids(director_ids, cache_dir)
    except Exception:
        return df

    merged["director_name"] = merged["director_id"].map(name_map)
    merged["director"] = merged["director"].where(
        merged["director"].notna() & (merged["director"] != "Unknown"),
        merged["director_name"]
    )

    return merged[df.columns]


def get_director_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes director stats: number of films, avg rating, etc.
    
    Args:
        df: DataFrame with movies (must have: director, rating)
    
    Returns:
        DataFrame with director statistics
    """
    if "director" not in df.columns:
        return pd.DataFrame()
    
    # Filter out "Unknown" directors
    df_with_directors = df[df["director"].notna() & (df["director"] != "Unknown")].copy()
    
    if len(df_with_directors) == 0:
        return pd.DataFrame()
    
    director_stats = df_with_directors.groupby("director").agg(
        films=("title", "count"),
        avg_rating=("rating", "mean"),
        best_rating=("rating", "max"),
        total_votes=("votes", "sum"),  # This is formatted, so it's symbolic
    ).reset_index()
    
    director_stats["avg_rating"] = director_stats["avg_rating"].round(2)
    director_stats = director_stats.sort_values("avg_rating", ascending=False)
    
    return director_stats


def get_top_directors(df: pd.DataFrame, top_n: int = 10, min_films: int = 1) -> pd.DataFrame:
    """
    Get top directors by average rating (with min films filter).
    
    Args:
        df: Movie DataFrame
        top_n: Number of directors to return
        min_films: Minimum number of films to qualify
    
    Returns:
        Top N directors with highest avg rating
    """
    stats = get_director_stats(df)
    if len(stats) == 0:
        return pd.DataFrame()
    
    stats = stats[stats["films"] >= min_films]
    return stats.head(top_n)


# ──────────────────────────────────────────────
# LIVE RATING TRENDS
# ──────────────────────────────────────────────

def load_all_snapshots(data_dir: str = "data") -> pd.DataFrame:
    """
    Loads all dated snapshots and returns combined time-series data.

    Args:
        data_dir: Directory containing CSV snapshots

    Returns:
        DataFrame with columns: title, snapshot_date, rating, rank
    """
    import glob

    base_dir = os.path.dirname(__file__)
    data_dir_path = data_dir if os.path.isabs(data_dir) else os.path.join(base_dir, data_dir)
    snapshots = sorted(glob.glob(os.path.join(data_dir_path, "imdb_top250_20??????.csv")))

    if not snapshots and os.path.exists(os.path.join(data_dir_path, "imdb_top250_latest.csv")):
        snapshots = [os.path.join(data_dir_path, "imdb_top250_latest.csv")]

    if not snapshots:
        return pd.DataFrame()

    all_data = []
    for snapshot_path in snapshots:
        try:
            filename = os.path.basename(snapshot_path)
            date_str = filename.replace("imdb_top250_", "").replace(".csv", "")

            df = pd.read_csv(snapshot_path)
            if date_str == "latest":
                if "scraped_date" in df.columns:
                    df["snapshot_date"] = pd.to_datetime(df["scraped_date"].astype(str), format="%Y%m%d", errors="coerce")
                else:
                    df["snapshot_date"] = pd.to_datetime("today")
            else:
                df["snapshot_date"] = pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")

            all_data.append(df[["title", "snapshot_date", "rating", "rank"]])
        except Exception as e:
            print(f"[WARN] Could not load {snapshot_path}: {e}")

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


def get_rating_trend_for_movie(movie_title: str, data_dir: str = "data") -> pd.DataFrame:
    """
    Gets rating trend for a specific movie across snapshots.
    
    Args:
        movie_title: Title of the movie to track
        data_dir: Directory with snapshots
    
    Returns:
        DataFrame with date and rating columns
    """
    all_data = load_all_snapshots(data_dir)
    
    if all_data.empty:
        return pd.DataFrame()
    
    movie_data = all_data[all_data["title"].str.contains(movie_title, case=False, na=False)]
    
    if movie_data.empty:
        return pd.DataFrame()
    
    return movie_data[["snapshot_date", "rating"]].sort_values("snapshot_date").drop_duplicates()


def get_overall_rating_trend(data_dir: str = "data") -> pd.DataFrame:
    """
    Gets overall Top 250 rating trend (average rating over time).
    
    Args:
        data_dir: Directory with snapshots
    
    Returns:
        DataFrame with date and avg_rating columns
    """
    all_data = load_all_snapshots(data_dir)
    
    if all_data.empty:
        return pd.DataFrame()
    
    trend = all_data.groupby("snapshot_date")["rating"].agg(["mean", "median", "std"]).reset_index()
    trend.columns = ["date", "avg_rating", "median_rating", "std_rating"]
    
    return trend.sort_values("date")


# ──────────────────────────────────────────────
# UI HELPER FUNCTIONS
# ──────────────────────────────────────────────

def format_director_name(name: str) -> str:
    """Format director name for display."""
    if name in ["Unknown", "N/A", None]:
        return "Unknown"
    return name.title()


def get_color_for_rating(rating: float) -> str:
    """Get color hex code based on rating."""
    if rating >= 9.0:
        return "#ffd700"  # Gold
    elif rating >= 8.7:
        return "#ff6b6b"  # Red
    elif rating >= 8.5:
        return "#ff8c42"  # Orange
    elif rating >= 8.3:
        return "#6bcf7f"  # Green
    else:
        return "#4ecdc4"  # Cyan

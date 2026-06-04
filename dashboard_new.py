"""
IMDb Top 250 – Advanced Analytics Dashboard
--------------------------------------------
Run with: streamlit run dashboard_new.py

Features:
✨ What to Watch — Local dataset recommendations
📈 Live Rating Trend - Track rating changes over time  
🏆 Director Hall of Fame - Discover top-rated directors
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
from utils import (
    get_ai_recommendation, 
    get_director_stats, 
    get_top_directors,
    get_overall_rating_trend,
    get_rating_trend_for_movie,
    enrich_director_data,
    format_director_name,
    get_color_for_rating,
    load_all_snapshots
)

BASE_DIR = os.path.dirname(__file__)


def format_date_for_display(date_value):
    if pd.isna(date_value):
        return "Unknown"

    date_str = str(date_value).strip()
    if not date_str:
        return "Unknown"

    parsed = pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(parsed):
        return date_str

    return parsed.strftime("%B %d, %Y")

# ──────────────────────────────────────────────
# PAGE CONFIG & CUSTOM CSS
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="IMDb Top 250 Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Main background and text colors */
    :root {
        --primary-color: #f5c518;
        --secondary-color: #1f77b4;
        --accent-color: #ff6b6b;
    }
    
    /* Custom card styling */
    .metric-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #f5c518;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Feature section headers */
    .feature-header {
        font-size: 28px;
        font-weight: bold;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #f5c518;
    }
    
    /* Recommendation card */
    .rec-card {
        background: linear-gradient(135deg, #2d2d2d 0%, #1e1e1e 100%);
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #ff6b6b;
        margin: 10px 0;
    }
    
    /* Director card */
    .director-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border-top: 3px solid #f5c518;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
DATA_PATH = os.path.join(BASE_DIR, "data", "imdb_top250_latest.csv")

@st.cache_data
def load_and_enrich_data(path):
    df = pd.read_csv(path)
    df["year_num"] = pd.to_numeric(df["year"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
    df["decade"] = df["year_num"].apply(lambda y: f"{int(y)//10*10}s" if pd.notna(y) else None)
    return enrich_director_data(df)


@st.cache_data
def load_snapshot_data():
    return load_all_snapshots("data")


@st.cache_data
def load_overall_trend():
    all_data = load_snapshot_data()
    if all_data.empty:
        return pd.DataFrame()

    trend = all_data.groupby("snapshot_date")["rating"].agg(["mean", "median", "std"]).reset_index()
    trend.columns = ["date", "avg_rating", "median_rating", "std_rating"]
    return trend.sort_values("date")


@st.cache_data
def load_movie_trend(movie_title: str):
    all_data = load_snapshot_data()
    if all_data.empty:
        return pd.DataFrame()

    movie_data = all_data[all_data["title"].str.contains(movie_title, case=False, na=False)]
    if movie_data.empty:
        return pd.DataFrame()

    return movie_data[["snapshot_date", "rating"]].sort_values("snapshot_date").drop_duplicates()


if not os.path.exists(DATA_PATH):
    st.error("⚠️ No data found. Please run `python imdb_scraper.py` first.")
    st.stop()

df = load_and_enrich_data(DATA_PATH)
scraped_date = "Unknown"
if "scraped_date" in df.columns:
    scraped_date = format_date_for_display(df["scraped_date"].iloc[0])

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("# 🎬 IMDb Top 250 — Pro Analytics")
st.markdown("""
**Your smart movie discovery platform** | Live trends • Director insights • Smart recommendations
""")

col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("📅 Last Updated", scraped_date)
with col_info2:
    st.metric("🎥 Movies Loaded", len(df))
with col_info3:
    st.metric("⭐ Avg Rating", f"{df['rating'].mean():.2f}")

st.divider()

# ──────────────────────────────────────────────
# SIDEBAR FILTERS
# ──────────────────────────────────────────────
st.sidebar.header("🔍 Dashboard Filters")

min_rating = st.sidebar.slider("Minimum Rating", 8.0, 9.5, 8.0, 0.1)

year_valid = df["year_num"].dropna()
if len(year_valid) > 0:
    year_min = int(year_valid.min())
    year_max = int(year_valid.max())
else:
    year_min = 1900
    year_max = 2030

year_range = st.sidebar.slider(
    "Year Range",
    year_min, year_max,
    (1950, year_max)
)

top_n = st.sidebar.selectbox(
    "Show Top N Movies",
    [10, 25, 50, 100, 250],
    index=1,
    format_func=lambda x: f"Top {x} Movies"
)

search_query = st.sidebar.text_input("🔎 Search Movie Title")

# Apply filters
filtered = df[
    (df["rating"] >= min_rating) &
    (df["year_num"] >= year_range[0]) &
    (df["year_num"] <= year_range[1])
].head(top_n)

if search_query:
    filtered = filtered[filtered["title"].str.contains(search_query, case=False, na=False)]

st.sidebar.divider()
st.sidebar.caption("💡 Adjust filters to explore different eras and ratings!")

# ──────────────────────────────────────────────
# TAB NAVIGATION
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "🤖 What to Watch AI",
    "📈 Rating Trends",
    "🏆 Director Hall of Fame",
    "📊 Data Explorer"
])

# ──────────────────────────────────────────────
# TAB 1: OVERVIEW
# ──────────────────────────────────────────────
with tab1:
    st.markdown("### 📊 Quick Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("### 🎥")
        st.metric("Total Movies", len(filtered))
    with col2:
        st.markdown("### ⭐")
        avg_rating = f"{filtered['rating'].mean():.2f}" if len(filtered) > 0 else "N/A"
        st.metric("Avg Rating", avg_rating)
    with col3:
        st.markdown("### 🏅")
        top_movie = filtered.iloc[0]["title"] if len(filtered) > 0 else "N/A"
        st.metric("Highest Rated", top_movie, delta=f"{filtered.iloc[0]['rating']}/10" if len(filtered) > 0 else None)
    with col4:
        st.markdown("### 📅")
        if "decade" in filtered.columns:
            decade_counts = filtered["decade"].value_counts()
            top_decade = decade_counts.idxmax() if not decade_counts.empty else "N/A"
        else:
            top_decade = "N/A"
        st.metric("Top Decade", top_decade)
    
    st.divider()
    
    # Top movies bar chart
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📽️ Top Movies by Rating")
        top20 = filtered.head(20).sort_values("rating")
        if len(top20) > 0:
            fig = px.bar(
                top20, x="rating", y="title", orientation="h",
                color="rating", color_continuous_scale="Viridis",
                labels={"rating": "IMDb Rating", "title": "Movie"},
            )
            fig.update_layout(
                height=500, 
                showlegend=False, 
                yaxis_title="",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No movies to display")
    
    with col_right:
        st.subheader("🎞️ Movies per Decade")
        decade_data = filtered["decade"].dropna().value_counts().sort_index().reset_index()
        if "count" in decade_data.columns:
            decade_data.columns = ["Decade", "Count"]
        elif "decade" in decade_data.columns:
            decade_data = decade_data.rename(columns={list(decade_data.columns)[1]: "Count"})
        
        if len(decade_data) > 0:
            fig2 = px.pie(
                decade_data, values="Count", names="Decade",
                hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu
            )
            fig2.update_layout(
                height=500,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No decade data available")
    
    st.divider()
    
    # Rating distribution and scatter
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("📊 Rating Distribution")
        fig3 = px.histogram(
            filtered, x="rating", nbins=20,
            color_discrete_sequence=["#f5c518"],
            labels={"rating": "IMDb Rating", "count": "Number of Movies"}
        )
        fig3.update_layout(
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        st.subheader("🎯 Rank vs Rating")
        fig4 = px.scatter(
            filtered, x="rank", y="rating",
            hover_name="title", hover_data=["year"],
            color="rating", color_continuous_scale="Cividis",
            labels={"rank": "IMDb Rank", "rating": "IMDb Rating"}
        )
        fig4.update_layout(
            height=400,
            xaxis_autorange="reversed",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig4, use_container_width=True)

# ──────────────────────────────────────────────
# TAB 2: WHAT TO WATCH AI
# ──────────────────────────────────────────────
with tab2:
    st.markdown("### 🤖 What to Watch — Recommendations")
    st.markdown("""
    Tell the dashboard your mood, and get personalized movie recommendations from the IMDb Top 250.
    Recommendations are generated locally from the dataset; no external AI service is required.
    """)
    st.info("Local recommendations are generated from the dataset and do not require any external AI service.")

    col_ai1, col_ai2 = st.columns([2, 1])
    
    with col_ai1:
        mood_input = st.text_input(
            "📝 Describe your mood or what you're looking for:",
            placeholder="e.g., 'mind-bending sci-fi with twists', 'heartwarming drama', 'intense action thriller'",
            help="Be specific for better recommendations!"
        )
    
    with col_ai2:
        num_recs = st.selectbox("# of Recommendations", [3, 5, 7, 10], index=1)
    
    if st.button("🎬 Get Recommendations", type="primary", use_container_width=True):
        if mood_input.strip():
            with st.spinner("Generating recommendations..."):
                recommendation = get_ai_recommendation(
                    filtered if len(filtered) > 0 else df,
                    mood_input,
                    max_results=num_recs
                )
            
            st.markdown("### 🎯 Your Personalized Recommendations:")
            st.markdown(recommendation, unsafe_allow_html=True)
            
            # Add download button
            st.download_button(
                "📋 Download Recommendations",
                data=recommendation,
                file_name=f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
        else:
            st.warning("Please describe your mood or preference!")
    
    st.divider()
    
    st.markdown("### 💡 Quick Mood Suggestions")
    cols = st.columns(3)
    moods = [
        ("🧠 Mind-Bending", "sci-fi with plot twists and complexity"),
        ("😂 Laugh Out Loud", "hilarious comedy that's entertaining"),
        ("😭 Emotional", "touching drama that hits you in the feels"),
        ("⚡ High-Energy", "intense action and adventure"),
        ("🎨 Artistic", "visually stunning films with deep meaning"),
        ("🌍 Epic", "grand sweeping stories with huge scale"),
    ]
    
    for idx, (emoji, desc) in enumerate(moods):
        with cols[idx % 3]:
            if st.button(f"{emoji}\n{desc}", use_container_width=True, key=f"mood_{idx}"):
                with st.spinner("Generating recommendations..."):
                    recommendation = get_ai_recommendation(
                        filtered if len(filtered) > 0 else df,
                        desc,
                        max_results=5
                    )
                st.markdown(recommendation, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# TAB 3: RATING COMPARISON (replaces trends)
# ──────────────────────────────────────────────
with tab3:
    st.markdown("### 📊 Rating Comparison")
    st.markdown("Compare current IMDb ratings across multiple movies. Select titles to view side-by-side ratings and recent changes (if snapshot data is available).")

    # Movie selection
    movie_options = (filtered["title"].dropna().unique().tolist() if len(filtered) > 0 else df["title"].dropna().unique().tolist())
    default_selection = movie_options[:5]
    selected_movies = st.multiselect("Select movies to compare (max 10):", movie_options, default=default_selection)

    if not selected_movies:
        st.info("Select one or more movies from the list to compare their ratings.")
    else:
        # Current ratings from the main dataset
        comp_df = df[df["title"].isin(selected_movies)][["title", "rating", "year", "rank"]].drop_duplicates().set_index("title")
        comp_df = comp_df.reindex(selected_movies).reset_index()

        # Bar chart comparison
        fig_comp = px.bar(
            comp_df,
            x="rating",
            y="title",
            orientation="h",
            color="rating",
            color_continuous_scale="Viridis",
            labels={"rating": "IMDb Rating", "title": "Movie"}
        )
        fig_comp.update_layout(
            height=400,
            yaxis_title="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # If snapshot data exists, compute change vs previous snapshot
        all_snap = load_snapshot_data()
        if not all_snap.empty:
            try:
                dates = sorted(pd.to_datetime(all_snap["snapshot_date"]).dropna().unique())
            except Exception:
                dates = []

            if len(dates) >= 2:
                latest = dates[-1]
                prev = dates[-2]

                pivot = all_snap[all_snap["title"].isin(selected_movies)].pivot_table(
                    index="title", columns="snapshot_date", values="rating", aggfunc="last"
                )
                pivot = pivot.reindex(selected_movies)

                latest_vals = pivot[latest] if latest in pivot.columns else pd.Series([None] * len(pivot), index=pivot.index)
                prev_vals = pivot[prev] if prev in pivot.columns else pd.Series([None] * len(pivot), index=pivot.index)
                delta = latest_vals - prev_vals

                compare_table = pd.DataFrame({
                    "Title": pivot.index,
                    "Current Rating": [f"{r:.2f}" if pd.notna(r) else "N/A" for r in comp_df["rating"]],
                    f"Latest ({latest.date()})": [f"{v:.2f}" if pd.notna(v) else "N/A" for v in latest_vals.values],
                    f"Previous ({prev.date()})": [f"{v:.2f}" if pd.notna(v) else "N/A" for v in prev_vals.values],
                    "Delta": [f"{d:+.2f}" if pd.notna(d) else "N/A" for d in delta.values]
                })

                st.subheader("🔁 Recent Snapshot Comparison")
                st.dataframe(compare_table, use_container_width=True)
            else:
                st.info("Not enough dated snapshots to compute deltas. Add dated CSVs in the `data/` folder to enable snapshot comparisons.")
        else:
            st.info("No historical snapshot data available. Add dated CSVs in the `data/` folder to enable snapshot comparisons.")

# ──────────────────────────────────────────────
# TAB 4: DIRECTOR HALL OF FAME
# ──────────────────────────────────────────────
with tab4:
    st.markdown("### 🏆 Director Hall of Fame")
    st.markdown("Discover the most acclaimed directors in the Top 250!")
    
    # Get director stats
    top_dirs = get_top_directors(filtered if len(filtered) > 0 else df, top_n=15, min_films=1)
    
    if not top_dirs.empty:
        col_dir1, col_dir2 = st.columns([2, 1])
        
        with col_dir1:
            st.subheader("🎥 Top Directors by Average Rating")
            
            fig_dirs = px.bar(
                top_dirs.head(10),
                x="avg_rating",
                y="director",
                orientation="h",
                color="avg_rating",
                color_continuous_scale="Reds",
                labels={"avg_rating": "Avg Rating", "director": "Director"},
                hover_data={"films": True}
            )
            fig_dirs.update_layout(
                height=500,
                yaxis_title="",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False
            )
            st.plotly_chart(fig_dirs, use_container_width=True)
        
        with col_dir2:
            st.subheader("📊 Top Directors")
            display_df = top_dirs.head(10)[["director", "films", "avg_rating"]].reset_index(drop=True)
            display_df.columns = ["Director", "# Films", "Avg ⭐"]
            display_df.index = display_df.index + 1
            st.dataframe(display_df, use_container_width=True, hide_index=False)
        
        st.divider()
        
        # Director spotlight
        st.subheader("🌟 Director Spotlight")
        
        if len(top_dirs) > 0:
            selected_director = st.selectbox(
                "Select a director to explore:",
                top_dirs["director"].head(20).unique()
            )
            
            director_movies = filtered[
                (filtered["director"].notna()) & 
                (filtered["director"].str.contains(selected_director, case=False, na=False))
            ] if "director" in filtered.columns else pd.DataFrame()
            
            if not director_movies.empty:
                col_spot1, col_spot2, col_spot3 = st.columns(3)
                
                with col_spot1:
                    st.metric("🎬 Films", len(director_movies))
                with col_spot2:
                    st.metric("⭐ Avg Rating", f"{director_movies['rating'].mean():.2f}")
                with col_spot3:
                    st.metric("🏅 Best Rating", f"{director_movies['rating'].max():.2f}")
                
                st.markdown("**🎞️ Filmography:**")
                for _, movie in director_movies.sort_values("rating", ascending=False).iterrows():
                    st.markdown(f"""
                    **{movie['title']}** ({movie['year']}) · ⭐ {movie['rating']}/10
                    """)
            else:
                st.info(f"No films found for {selected_director} in current filters.")
    else:
        st.info("💡 Director data not available. Run scraper with director scraping enabled for this feature.")

# ──────────────────────────────────────────────
# TAB 5: DATA EXPLORER
# ──────────────────────────────────────────────
with tab5:
    st.markdown("### 📋 Full Data Explorer")
    
    # Data view options
    col_opt1, col_opt2 = st.columns(2)
    
    with col_opt1:
        sort_by = st.selectbox(
            "Sort by:",
            ["Rank", "Rating", "Year", "Votes"],
            format_func=lambda x: {"Rank": "🔢 Rank", "Rating": "⭐ Rating", "Year": "📅 Year", "Votes": "👥 Votes"}.get(x)
        )
    
    with col_opt2:
        sort_order = st.radio("Order:", ["Ascending ↑", "Descending ↓"], horizontal=True)
    
    # Sort data
    sort_col_map = {"Rank": "rank", "Rating": "rating", "Year": "year_num", "Votes": "votes"}
    sort_col = sort_col_map[sort_by]
    ascending = sort_order == "Ascending ↑"
    
    sorted_filtered = filtered.sort_values(sort_col, ascending=ascending, na_position="last")
    
    # Display columns
    display_cols = ["rank", "title", "year", "rating"]
    if "director" in filtered.columns:
        display_cols.append("director")
    display_cols.extend(["votes", "duration"])
    
    available_cols = [c for c in display_cols if c in sorted_filtered.columns]
    
    st.dataframe(
        sorted_filtered[available_cols].reset_index(drop=True),
        use_container_width=True,
        height=500
    )
    
    st.divider()
    
    # Download button
    st.subheader("⬇️ Export Data")
    csv = sorted_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered CSV",
        data=csv,
        file_name=f"imdb_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.divider()
st.caption("""
🎬 **IMDb Top 250 Pro** | Built with Selenium • pandas • Streamlit • Plotly
📊 Data is live-scraped from IMDb and refreshed daily
🤖 Movie recommendations are generated locally from the IMDb dataset
""")

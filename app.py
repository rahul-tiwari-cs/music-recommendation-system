"""
Music Recommendation System — Streamlit App
--------------------------------------------
Loads pre-trained artifacts (metadata, sparse feature matrix, preprocessor)
and serves content-based recommendations via cosine similarity.

Run with:  streamlit run app.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# File paths (adjust here if you relocate artifacts)
# --------------------------------------------------------------------------
METADATA_PATH = "music_metadata.pkl"
MATRIX_PATH = "music_matrix.npz"
PREPROCESSOR_PATH = "preprocessor.pkl"
RAW_DATA_PATH = "data.csv"  # kept for reference / optional use


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading metadata...")
def load_metadata(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at '{path}'.")
    df = joblib.load(path)
    df = df.reset_index(drop=True)
    return df


@st.cache_resource(show_spinner="Loading feature matrix...")
def load_matrix(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature matrix file not found at '{path}'.")
    matrix = sparse.load_npz(path)
    if not sparse.isspmatrix_csr(matrix):
        matrix = matrix.tocsr()
    return matrix


@st.cache_resource(show_spinner="Loading preprocessor...")
def load_preprocessor(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessor file not found at '{path}'.")
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def build_track_index(_df: pd.DataFrame):
    """
    Build a fast, native-Python lookup structure so the selectbox never has
    to touch pandas on every keystroke.

    Returns:
        display_list: list[str] — "Song Name — Artist" labels for the UI
        label_to_idx: dict[str, int] — label -> row index in metadata/matrix
    """
    names = _df["name"].astype(str).tolist()
    artists = _df["artists"].astype(str).tolist()

    display_list = []
    label_to_idx = {}
    for i, (n, a) in enumerate(zip(names, artists)):
        # artists column is often stored as a stringified list, e.g. "['Adele']"
        clean_artist = a.strip("[]'\"")
        label = f"{n} — {clean_artist}"
        display_list.append(label)
        label_to_idx[label] = i

    display_list.sort()
    return display_list, label_to_idx


# --------------------------------------------------------------------------
# Recommendation logic
# --------------------------------------------------------------------------
def get_recommendations(
    query_idx: int,
    matrix,
    metadata: pd.DataFrame,
    top_n: int,
    year_range: tuple,
    min_popularity: int,
) -> pd.DataFrame:
    """Compute cosine similarity of the query track against all others,
    apply sidebar filters, and return the top-N matches."""

    query_vector = matrix[query_idx]
    sims = cosine_similarity(query_vector, matrix).flatten()

    # Exclude the query track itself from its own recommendations
    sims[query_idx] = -1.0

    result = metadata.copy()
    result["similarity"] = sims

    # Apply sidebar filters
    if "year" in result.columns:
        result = result[
            (result["year"] >= year_range[0]) & (result["year"] <= year_range[1])
        ]
    if "popularity" in result.columns:
        result = result[result["popularity"] >= min_popularity]

    # Drop the query track just in case filters somehow keep a duplicate row
    result = result[result["similarity"] > -1.0]

    result = result.sort_values("similarity", ascending=False).head(top_n)
    result["Match Score (%)"] = (result["similarity"] * 100).round(2)

    display_cols = {
        "name": "Track Name",
        "artists": "Artist",
        "year": "Year",
        "Match Score (%)": "Match Score (%)",
    }
    available_cols = [c for c in display_cols if c in result.columns]
    result = result[available_cols].rename(columns=display_cols)

    if "Artist" in result.columns:
        result["Artist"] = result["Artist"].astype(str).str.strip("[]'\"")

    return result.reset_index(drop=True)


# --------------------------------------------------------------------------
# Load artifacts (errors are caught and shown cleanly, app halts gracefully)
# --------------------------------------------------------------------------
try:
    metadata_df = load_metadata(METADATA_PATH)
    feature_matrix = load_matrix(MATRIX_PATH)
    preprocessor = load_preprocessor(PREPROCESSOR_PATH)
except FileNotFoundError as e:
    st.error(f"❌ Failed to load required artifact: {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Unexpected error while loading artifacts: {e}")
    st.stop()

if metadata_df.shape[0] != feature_matrix.shape[0]:
    st.error(
        "❌ Metadata and feature matrix row counts don't match "
        f"({metadata_df.shape[0]} vs {feature_matrix.shape[0]}). "
        "Please re-export matching artifacts."
    )
    st.stop()

track_display_list, track_label_to_idx = build_track_index(metadata_df)

# --------------------------------------------------------------------------
# Sidebar — controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    top_n = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=50,
        value=10,
        step=1,
    )

    st.subheader("Filters")

    if "year" in metadata_df.columns:
        year_min = int(metadata_df["year"].min())
        year_max = int(metadata_df["year"].max())
        year_range = st.slider(
            "Release year range",
            min_value=year_min,
            max_value=year_max,
            value=(year_min, year_max),
        )
    else:
        year_range = (0, 9999)

    if "popularity" in metadata_df.columns:
        min_popularity = st.slider(
            "Minimum popularity",
            min_value=0,
            max_value=100,
            value=0,
        )
    else:
        min_popularity = 0

    st.markdown("---")
    st.caption(f"Dataset size: {metadata_df.shape[0]:,} tracks")

# --------------------------------------------------------------------------
# Main page
# --------------------------------------------------------------------------
st.title("🎵 Music Recommendation System")
st.markdown(
    "Discover songs similar to your favorites using a content-based "
    "recommender trained on audio features. Search a track below, tune the "
    "filters in the sidebar, and hit **Get Recommendations**."
)
st.markdown("---")

MAX_SEARCH_RESULTS = 50

search_query = st.text_input(
    "🔍 Search for a song",
    placeholder="Start typing a song name or artist...",
)

if search_query:
    query_lower = search_query.lower()
    filtered_options = [
        label for label in track_display_list if query_lower in label.lower()
    ][:MAX_SEARCH_RESULTS]
else:
    filtered_options = track_display_list[:MAX_SEARCH_RESULTS]

if search_query and not filtered_options:
    st.caption("No matches — try a different spelling or fewer words.")

selected_label = st.selectbox(
    f"Matches ({len(filtered_options)} shown"
    + (f" of {len(track_display_list):,}" if not search_query else "")
    + ")",
    options=filtered_options,
    index=None,
    placeholder="Pick a track from the matches above...",
)

get_recs_clicked = st.button("🎧 Get Recommendations", type="primary")

if get_recs_clicked:
    if not selected_label:
        st.warning("⚠️ Please select a song before requesting recommendations.")
    else:
        query_idx = track_label_to_idx.get(selected_label)
        if query_idx is None:
            st.error("❌ Selected song could not be matched to the dataset. Try again.")
        else:
            with st.spinner("Finding tracks with similar sound..."):
                try:
                    recs_df = get_recommendations(
                        query_idx=query_idx,
                        matrix=feature_matrix,
                        metadata=metadata_df,
                        top_n=top_n,
                        year_range=year_range,
                        min_popularity=min_popularity,
                    )
                except Exception as e:
                    st.error(f"❌ Failed to compute recommendations: {e}")
                    recs_df = None

            if recs_df is not None:
                if recs_df.empty:
                    st.info(
                        "No matches found within the current filters. "
                        "Try widening the year range or lowering minimum popularity."
                    )
                else:
                    st.success(f"Top {len(recs_df)} tracks similar to **{selected_label}**")

                    st.dataframe(
                        recs_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Match Score (%)": st.column_config.ProgressColumn(
                                "Match Score (%)",
                                format="%.2f%%",
                                min_value=0,
                                max_value=100,
                            ),
                        },
                    )

                    with st.expander("🧾 View as cards"):
                        cols_per_row = 3
                        rows = [
                            recs_df.iloc[i : i + cols_per_row]
                            for i in range(0, len(recs_df), cols_per_row)
                        ]
                        for row_chunk in rows:
                            cols = st.columns(cols_per_row)
                            for col, (_, track) in zip(cols, row_chunk.iterrows()):
                                with col:
                                    st.markdown(
                                        f"""
                                        **{track.get('Track Name', 'N/A')}**
                                        {track.get('Artist', 'N/A')}
                                        `{track.get('Year', 'N/A')}` · {track.get('Match Score (%)', 0)}% match
                                        """
                                    )
else:
    st.info("👈 Select a song and click **Get Recommendations** to see results.")
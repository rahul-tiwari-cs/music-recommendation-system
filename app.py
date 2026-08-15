import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Configuration
st.set_page_config(
    page_title="Music Recommender", page_icon="🎵", layout="wide"
)


# 2. Fast Cached Data & Model Loading
@st.cache_data
def load_data():
  return joblib.load("music_metadata.pkl")


@st.cache_resource
def load_features():
  return sp.load_npz("music_matrix.npz")


@st.cache_data
def get_song_list(_df):
  # Cache the list as a Python list so Streamlit doesn't recalculate on every keystroke
  return _df["name"].drop_duplicates().sort_values().tolist()


df = load_data()
music_matrix = load_features()
song_list = get_song_list(df)


# 3. Fast Recommendation Engine Logic
def recommend(song_name, top_n=5):
  matches = df[df["name"] == song_name]
  if matches.empty:
    return None

  idx = matches.index[0]
  target_vec = music_matrix[idx : idx + 1]

  scores = cosine_similarity(target_vec, music_matrix).flatten()
  scores[idx] = -1.0  # Exclude target song itself

  top_indices = np.argsort(scores)[::-1][:top_n]
  results = df.iloc[top_indices].copy()
  results["similarity"] = np.round(scores[top_indices] * 100, 2)
  return results


# 4. Streamlit Web Interface
st.title("🎵 Music Recommendation Engine")

selected_song = st.selectbox(
    "Search or select a track:", options=song_list, index=0
)

num_recs = st.slider("Number of recommendations:", 1, 15, 5)

if st.button("Get Recommendations", type="primary"):
  with st.spinner("Analyzing audio features..."):
    recs = recommend(selected_song, num_recs)

  if recs is not None:
    st.success(f"Recommendations for **{selected_song}**:")
    st.dataframe(
        recs[["name", "artists", "similarity"]].rename(
            columns={
                "name": "Track Name",
                "artists": "Artist",
                "similarity": "Match Score (%)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
  else:
    st.error("Song not found.")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from collections import Counter

# --- CONFIGURATION ---
MODEL_PATH = "kmeans_model.pkl"
EMBEDDINGS_CSV = "data/music_embeds.csv"
EMBEDDING_COL_PREFIX = "dim_"

# --- PRESET COHORT DEFINITIONS ---
COHORT_SONGS = {
    "Cohort A": [
        "For my Hand","Abracadabra","Freefall","Freedom Time","Belong to the World",
        "The Line","Inochi No Tabekata","Tadow","Niagra Falls","Diamonds",
        "Kalidoscope","Only in my Dreams","My Favorite Things","Left Behind","Prove",
    ],
    "Cohort B": [
        "Blue Bayou","Misty","Make Yours a Happy Home","I'm Gonna Be","Frederick",
        "Wake Up","i am enough","Dancer","Promises (feat. Joe L Barnes)","Do 4 Love",
        "Promises","Valentina","Honey","The First Time Ever I Saw Your Face",
    ],
}


# --- LOAD ASSETS ---
@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_resource
def load_data(path):
    df = pd.read_csv(path)
    emb_cols = [c for c in df.columns if c.startswith(EMBEDDING_COL_PREFIX)]
    return df, emb_cols


model = load_model(MODEL_PATH)
df, emb_cols = load_data(EMBEDDINGS_CSV)

st.title("🎧 Mini-Lab: Cluster-Based Song Recommendations")

# --- 1. Cohort Presets ---
st.markdown("### 1. Select a preset cohort (optional)")
chosen_cohorts = st.multiselect(
    "Pick one or both presets:",
    options=list(COHORT_SONGS.keys())
)

# build initial history from cohorts
history_songs = []
for cohort in chosen_cohorts:
    history_songs.extend(COHORT_SONGS[cohort])

# --- 2. Manual History Selection ---
st.markdown("### 2. Add any additional songs you’ve listened to")
manual = st.multiselect(
    "Or pick individual songs:",
    options=df["song"].tolist(),
    default=history_songs  # pre-fills with cohort songs
)

# final history is the union of both
history = list(dict.fromkeys(manual))  # preserves order, dedup

if not history:
    st.info("Select at least one song (either via cohort or manually) to get a recommendation.")
    st.stop()

# --- RECOMMENDATION LOGIC ---
hist_df = df[df["song"].isin(history)]
hist_embeddings = hist_df[emb_cols].values
hist_clusters = model.predict(hist_embeddings)
fav_cluster = Counter(hist_clusters).most_common(1)[0][0]

df["cluster"] = model.labels_
candidates = df[(df["cluster"] == fav_cluster) & (~df["song"].isin(history))]

if candidates.empty:
    st.warning("No new songs left in your favorite cluster—try selecting a different cohort or adding new songs.")
    st.stop()

centroid = model.cluster_centers_[fav_cluster]
cand_embeddings = candidates[emb_cols].values
dists = np.linalg.norm(cand_embeddings - centroid, axis=1)

# choose one of the top 5 songs
N = 5
top_idxs = np.argsort(dists)[:N]
chosen_idx = np.random.choice(top_idxs)
recommended = candidates.iloc[chosen_idx]["song"]

# --- OUTPUT ---
st.markdown("### 🎯 Your next recommendation:")
st.success(f"**{recommended}** (from cluster {fav_cluster})")

st.markdown("---")
st.markdown("**How it works:**")
st.markdown("""
1. You optionally pick a preset “Cohort A/B” to autofill a set of songs.  
2. You can also add any extra songs from the full list.  
3. We infer each selected song’s cluster and find your most-listened cluster.  
4. We recommend the song in that cluster closest to its centroid that you haven’t yet heard.  
""")

# Cross-Cohort Music Recommendation — EDA & Unsupervised Clustering

**This repo is a mini-project demonstrating the design of an unsupervised music recommender.**  
Goal: explore “waiting-room” music’s impact on student attentiveness by clustering songs via embeddings and recommending similar tracks.

Repo: `music-recommendation-mini-tlab-7`  
Data: 61 songs with **music2vec** embeddings (vector features)

---

## 🔍 Problem Context

You’re a data scientist at a remote education company. Your manager wants to test whether curated “waiting-room” music improves attention during a 4-hour class. You’ll:

- Do **EDA** on provided song embeddings.  
- Use **PCA** to reduce dimensionality.  
- Use **KMeans** to cluster songs.  
- Pick **K** via **Silhouette** and **Elbow (“shoulder”)** methods.  
- Serialize the pipeline and load it into a **Streamlit** dashboard for recommendations.

---

## 🧱 Project Structure

```text
.
├── data/
│   └── songs.csv                # 61 songs + embeddings (not included if private)
├── notebooks/
│   └── clustering.ipynb         # EDA + PCA + KMeans + model selection
├── src/                         # optional (scripts, helpers)
├── models/                      # saved artifacts
│   ├── pca.pkl                  # saved PCA
│   ├── kmeans.pkl               # saved KMeans
│   └── scaler.pkl               # saved StandardScaler
├── app/
│   └── recommender.py           # Streamlit app
├── requirements.txt
└── README.md
```

> Only using the notebook + Streamlit? Keep `notebooks/` and `app/`, then create `models/` when **saving** artifacts.

---

## ⚙️ Environment

```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS/Linux (bash/zsh)
source .venv/bin/activate

pip install -r requirements.txt
```

**requirements.txt (baseline)**
```
pandas
numpy
scikit-learn
matplotlib
plotly
seaborn
streamlit
joblib
jupyter
```

---

## 📊 Workflow

1) **Load & inspect**  
   - Summary stats, missingness, outliers  
   - Pairwise similarity (cosine/Euclidean) checks  
2) **Preprocess**  
   - `StandardScaler` on embedding columns  
3) **Dimensionality reduction (PCA)**  
   - Keep components explaining ~90–95% variance (tune based on scree)  
4) **Cluster (KMeans)**  
   - Search K in 2…10 (or as appropriate)  
   - Select via **Silhouette** (maximize) and **Elbow** (first strong bend)  
5) **Persist artifacts**  
   - Save `scaler.pkl`, `pca.pkl`, and `kmeans.pkl` with `joblib`  
6) **App**  
   - Load artifacts, take a user vector / selected song, recommend nearest neighbors within cluster

---

## 🧪 Model Selection Snippets

**Silhouette & Elbow (“shoulder”) search**
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RNG = 42
X = pd.read_csv("data/songs.csv").filter(regex="^embed_")  # adjust columns

# scale
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# PCA (tune n_components)
pca = PCA(n_components=0.95, random_state=RNG)  # keep 95% variance
Xp = pca.fit_transform(Xs)

k_range = range(2, 11)
sil_scores, sse = [], []

for k in k_range:
    km = KMeans(n_clusters=k, n_init="auto", random_state=RNG)
    labels = km.fit_predict(Xp)
    sil_scores.append(silhouette_score(Xp, labels))
    sse.append(km.inertia_)

best_k = k_range[int(np.argmax(sil_scores))]
print("Best K by silhouette:", best_k)
```

> **Choose K** where **silhouette** peaks and **SSE** (inertia) shows the first strong bend. If they disagree, prefer interpretability and stability (re-run with different seeds).

**Persist models**
```python
import joblib
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(pca, "models/pca.pkl")
joblib.dump(KMeans(n_clusters=best_k, n_init="auto", random_state=RNG).fit(Xp), "models/kmeans.pkl")
```

---

## ▶️ Run the Recommender

Once you’ve saved your artifacts to `models/`:

```bash
streamlit run app/recommender.py
```

The app should:
- Load `scaler.pkl`, `pca.pkl`, `kmeans.pkl`  
- Let you pick a seed song or paste a user embedding  
- Map to a cluster and return **top-N nearest** songs (cosine similarity within the cluster)

---

## 📈 What to Report (fill these in)

- **PCA**: number of components, cumulative variance explained  
- **K selection**: silhouette values by K + elbow plot; final K with rationale  
- **Cluster insight**: quick descriptors (e.g., tempo/energy proxies if available)  
- **Examples**: 2–3 representative songs per cluster  
- **Limitations**: tiny dataset (61 songs), embedding provenance, cold-start users

---

## ✅ Repro

```bash
# 1) Setup
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt

# 2) EDA + training (notebook)
jupyter notebook notebooks/clustering.ipynb
# save artifacts to models/

# 3) App
streamlit run app/recommender.py
```

---

## 🔮 Next Steps

- Swap in a larger embeddings dataset.  
- Try **HDBSCAN** for non-spherical clusters.  
- Add **UMAP** for neighborhood-preserving projections before KMeans.  
- Add a simple user feedback loop to refine recommendations.

---

## 📜 License & Credits

- License: MIT  
- Data: internal sample of 61 songs with music2vec embeddings.  
- Inspired by classical NLP/ML pipelines (PCA + KMeans) and modern recsys heuristics.

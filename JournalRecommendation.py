"""Journal recommendation system built on Scopus papers."""

import json
import sqlite3
from typing import Any, Dict, Optional, cast

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Journal recommendation system",
    layout="wide",
)

st.title("Journal Recommendation System for New Manuscript")
st.markdown(
    "Given a paper title and abstract (plus optional keywords), this app"
    " suggests **suitable journals** based on past Scopus articles."
)


def parse_keywords(raw) -> str:
    """Normalize keyword representations to a single string."""
    if raw is None or raw == "" or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    if isinstance(raw, (list, tuple, set)):
        return " ".join(str(v) for v in raw if v)
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except Exception:
            return raw
        if isinstance(data, dict):
            return " ".join(str(v) for v in data.values() if v)
        if isinstance(data, list):
            return " ".join(str(v) for v in data if v)
    return str(raw)


@st.cache_data(show_spinner="Loading papers from scopus.db ...")
def load_data() -> pd.DataFrame:
    query = """
        SELECT file_id, citation_title, abstracts, keywords, sourcetitle
        FROM research_papers
    """
    with sqlite3.connect("scopus.db") as conn:
        df = pd.read_sql_query(query, conn)

    for col in ["citation_title", "abstracts", "sourcetitle"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    if "keywords" in df.columns:
        df["keywords_text"] = df["keywords"].fillna("").apply(parse_keywords)
    else:
        df["keywords_text"] = ""

    df["text"] = (
        df["citation_title"].str.strip()
        + " "
        + df["abstracts"].str.strip()
        + " "
        + df["keywords_text"].str.strip()
    ).str.replace(r"\s+", " ", regex=True)

    df = df[(df["text"].str.strip() != "") & (df["sourcetitle"].str.strip() != "")]
    return df

df = load_data()

st.caption(
    f"Loaded **{len(df):,}** papers across **{df['sourcetitle'].nunique():,}** journals."
)

# ------------------------------------------------------------
# 2) Training function for the journal classifier
# ------------------------------------------------------------

@st.cache_resource(show_spinner="Training journal recommendation model ...")
def train_journal_model(
    data: pd.DataFrame,
    max_features: int,
    use_clustering: bool,
    cluster_k: int,
    filter_mode: str,
    top_n_journals: Optional[int],
    min_papers: int,
) -> Dict[str, Any]:
    """Train TF-IDF + LogisticRegression with optional KMeans clusters."""

    counts = data["sourcetitle"].value_counts()

    if filter_mode == "top":
        limit = top_n_journals or len(counts)
        selected = counts.head(limit).index.tolist()
    elif filter_mode == "min":
        selected = counts[counts >= max(1, min_papers)].index.tolist()
    else:
        selected = counts.index.tolist()

    if not selected:
        selected = counts.index.tolist()

    sub = data[data["sourcetitle"].isin(selected)].copy()

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        stop_words="english",
    )
    X = vectorizer.fit_transform(sub["text"].values)

    le_global = LabelEncoder()
    y = le_global.fit_transform(sub["sourcetitle"].astype(str))

    clf_global = LogisticRegression(
        max_iter=2000,
        multi_class="auto",
        solver="lbfgs",
    )
    clf_global.fit(X, y)

    cluster_model: Optional[KMeans] = None
    cluster_classifiers: Dict[int, Dict[str, object]] = {}

    if use_clustering:
        cluster_model = KMeans(n_clusters=cluster_k, n_init=10, random_state=42)
        cluster_labels = cluster_model.fit_predict(X)
        sub["cluster_id"] = cluster_labels

        for cid in np.unique(cluster_labels):
            mask = cluster_labels == cid
            cluster_subset = sub.loc[mask]
            if cluster_subset["sourcetitle"].nunique() < 2:
                continue
            le_cluster = LabelEncoder()
            y_cluster = le_cluster.fit_transform(cluster_subset["sourcetitle"])
            clf_cluster = LogisticRegression(
                max_iter=2000,
                multi_class="auto",
                solver="lbfgs",
            )
            row_idx = np.where(mask)[0]
            X_cluster = X[row_idx]  # type: ignore[index]
            clf_cluster.fit(X_cluster, y_cluster)
            cluster_classifiers[int(cid)] = {
                "clf": clf_cluster,
                "label_encoder": le_cluster,
                "count": int(mask.sum()),
            }

    return {
        "vectorizer": vectorizer,
        "clf_global": clf_global,
        "label_encoder": le_global,
        "training_subset": sub,
        "class_counts": counts.loc[selected],
        "use_clustering": use_clustering and bool(cluster_classifiers),
        "cluster_model": cluster_model,
        "cluster_classifiers": cluster_classifiers,
    }

# ------------------------------------------------------------
# 3) Sidebar controls
# ------------------------------------------------------------

st.sidebar.header("Model settings")

filter_mode_label = st.sidebar.radio(
    "Journal selection mode",
    ("All journals", "Top N by frequency", "Minimum paper count"),
    index=0,
)
if filter_mode_label.startswith("All"):
    filter_mode = "all"
elif filter_mode_label.startswith("Top"):
    filter_mode = "top"
else:
    filter_mode = "min"

unique_journals = df["sourcetitle"].nunique()
top_n = None
min_papers = 1

if filter_mode == "top":
    slider_max = int(min(max(unique_journals, 5), 300))
    top_n = st.sidebar.slider(
        "Number of journals to include (top N by frequency)",
        min_value=5,
        max_value=slider_max,
        value=min(50, slider_max),
        step=1,
    )
elif filter_mode == "min":
    max_count = int(df.groupby("sourcetitle").size().max())
    min_papers = st.sidebar.slider(
        "Minimum papers per journal",
        min_value=1,
        max_value=max(5, min(200, max_count)),
        value=min(25, max(5, max_count)),
        step=1,
    )

max_feat = st.sidebar.slider(
    "Max TF-IDF features",
    min_value=5000,
    max_value=50000,
    value=20000,
    step=5000,
)

k_recommend = st.sidebar.slider(
    "Number of journals recommendations",
    min_value=3,
    max_value=10,
    value=5,
    step=1,
)

use_clusters = st.sidebar.checkbox("Use topic clusters (KMeans)", value=False)
cluster_k = st.sidebar.slider(
    "Number of clusters", min_value=3, max_value=20, value=8, step=1
)

model_bundle: Dict[str, Any] = train_journal_model(
    df,
    max_features=max_feat,
    use_clustering=use_clusters,
    cluster_k=cluster_k,
    filter_mode=filter_mode,
    top_n_journals=top_n,
    min_papers=min_papers,
)
vec = cast(TfidfVectorizer, model_bundle["vectorizer"])
clf = cast(LogisticRegression, model_bundle["clf_global"])
le = cast(LabelEncoder, model_bundle["label_encoder"])
train_sub = cast(pd.DataFrame, model_bundle["training_subset"])
class_counts = cast(pd.Series, model_bundle["class_counts"])
cluster_enabled = cast(bool, model_bundle["use_clustering"])
cluster_model = cast(Optional[KMeans], model_bundle["cluster_model"])
cluster_classifiers = cast(Dict[int, Dict[str, Any]], model_bundle["cluster_classifiers"])

st.sidebar.success(
    f"Model trained on {len(train_sub):,} papers "
    f"from {len(class_counts):,} journals."
)

if filter_mode == "top":
    coverage_desc = f"Top {top_n or len(class_counts)} journals by publication volume."
elif filter_mode == "min":
    coverage_desc = (
        f"All journals with at least {min_papers} papers (total {len(class_counts)})."
    )
else:
    coverage_desc = f"All {len(class_counts)} journals present in the dataset."
st.sidebar.caption(coverage_desc)

if use_clusters and not cluster_enabled:
    st.sidebar.warning(
        "Clustering requested but not enough data per cluster. Using the global"
        " classifier instead."
    )

if cluster_enabled and cluster_classifiers:
    cluster_summary = pd.DataFrame(
        {
            "cluster": list(cluster_classifiers.keys()),
            "papers": [info.get("count", 0) for info in cluster_classifiers.values()],
            "journals": [
                len(cast(LabelEncoder, info["label_encoder"]).classes_)
                for info in cluster_classifiers.values()
            ],
        }
    ).sort_values("papers", ascending=False)
    st.sidebar.markdown("**Cluster coverage**")
    st.sidebar.dataframe(cluster_summary, hide_index=True, use_container_width=True)

# ------------------------------------------------------------
# 4) Input area for new manuscript
# ------------------------------------------------------------

st.markdown("---")
st.header("Try it on a new manuscript")

col1, col2 = st.columns([2, 1])

with col1:
    user_title = st.text_input(
        "Paper title",
        value="",
        placeholder="Enter your manuscript title here...",
    )
    user_abs = st.text_area(
        "Abstract",
        value="",
        height=200,
        placeholder="Paste your abstract here...",
    )
    user_kw = st.text_input(
        "Keywords (comma-separated)",
        value="",
        placeholder="e.g. machine learning, climate change, satellite",
    )

with col2:
    st.markdown("**Current training journals (top N)**")
    st.dataframe(
        class_counts.rename("paper_count").reset_index().rename(
            columns={"sourcetitle": "journal"}
        ),
        use_container_width=True,
        height=250,
    )

# ------------------------------------------------------------
# 5) Make recommendation
# ------------------------------------------------------------

def recommend_journals(title: str, abstract: str, keywords: str, top_k: int = 5):
    segments = [title or "", abstract or "", keywords or ""]
    text = " ".join(seg.strip() for seg in segments if seg and seg.strip())
    if not text:
        return None

    X_new = vec.transform([text])
    use_cluster_model = cluster_enabled and cluster_model is not None
    chosen_clf = clf
    chosen_le = le

    if use_cluster_model:
        assert cluster_model is not None
        cluster_id = int(cluster_model.predict(X_new)[0])
        cluster_pack = cluster_classifiers.get(cluster_id)
        if cluster_pack:
            chosen_clf = cluster_pack["clf"]  # type: ignore[assignment]
            chosen_le = cluster_pack["label_encoder"]  # type: ignore[assignment]
        else:
            st.info(
                "Cluster did not have enough training samples. Falling back to"
                " global model."
            )

    if hasattr(chosen_clf, "predict_proba"):
        proba = chosen_clf.predict_proba(X_new)[0]
    else:
        scores = chosen_clf.decision_function(X_new)[0]
        exp_scores = np.exp(scores - scores.max())
        proba = exp_scores / exp_scores.sum()

    order = np.argsort(proba)[::-1][:top_k]
    journals = chosen_le.inverse_transform(order)
    scores = proba[order]

    rec_df = pd.DataFrame(
        {
            "rank": np.arange(1, len(journals) + 1),
            "journal": journals,
            "probability": np.round(scores, 4),
        }
    )
    return rec_df

st.markdown("")
if st.button("Recommend journals"):
    result = recommend_journals(user_title, user_abs, user_kw, k_recommend)
    if result is None:
        st.warning("Please enter a title or abstract first.")
    else:
        st.subheader("Suggested journals")
        st.table(result)

# ------------------------------------------------------------
# 6) Small analysis plot: class distribution
# ------------------------------------------------------------
st.markdown("---")
st.subheader("Journal distribution used for training (top N)")

dist_df = class_counts.reset_index()
dist_df.columns = ["journal", "paper_count"]

bar_fig = {
    "data": [
        {
            "x": dist_df["journal"],
            "y": dist_df["paper_count"],
            "type": "bar",
        }
    ],
    "layout": {
        "xaxis": {"title": "Journal", "tickangle": -45},
        "yaxis": {"title": "Number of papers"},
        "margin": {"b": 120},
        "height": 450,
    },
}
st.plotly_chart(bar_fig, use_container_width=True)

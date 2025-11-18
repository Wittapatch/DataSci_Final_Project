# 03_journal_recommender.py
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Journal recommendation system",
    layout="wide",
)

st.title("Journal Recommendation System for New Manuscript")
st.markdown(
    "Given a paper title and abstract, this app suggests **suitable journals** "
    "based on past Scopus articles."
)

SQL = r"""
SELECT
    file_id,
    year,
    json_extract(
        raw_json,
        '$."abstracts-retrieval-response".item.bibrecord.head."citation-title"'
    ) AS citation_title,
    json_extract(
        raw_json,
        '$."abstracts-retrieval-response".item.bibrecord.head.abstracts'
    ) AS abstracts,
    json_extract(
        raw_json,
        '$."abstracts-retrieval-response".item.bibrecord.head.source.sourcetitle'
    ) AS sourcetitle
FROM papers_raw;
"""

@st.cache_data(show_spinner="Loading papers from scopus.db ...")
def load_data():
    conn = sqlite3.connect("scopus.db")
    df = pd.read_sql_query(SQL, conn)
    conn.close()

    # basic cleaning
    df["citation_title"] = df["citation_title"].fillna("").astype(str)
    df["abstracts"] = df["abstracts"].fillna("").astype(str)
    df["sourcetitle"] = df["sourcetitle"].fillna("").astype(str)

    # combine title + abstract as text features
    df["text"] = (
        df["citation_title"].str.strip()
        + " "
        + df["abstracts"].str.strip()
    ).str.replace(r"\s+", " ", regex=True)

    # drop completely empty rows or missing journal
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
    top_n_journals: int = 20,
    max_features: int = 20000,
):
    """
    Train TF-IDF + LogisticRegression classifier.
    Only uses the top_n_journals most frequent sourcetitle labels.
    """
    # pick top-N most frequent journals
    counts = data["sourcetitle"].value_counts()
    top_journals = counts.head(top_n_journals).index.tolist()
    sub = data[data["sourcetitle"].isin(top_journals)].copy()

    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(sub["sourcetitle"].values)

    # TF-IDF features from title + abstract
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        stop_words="english",
    )
    X = vectorizer.fit_transform(sub["text"].values)

    # multi-class logistic regression
    clf = LogisticRegression(
        max_iter=2000,
        multi_class="auto",
        solver="lbfgs",
    )
    clf.fit(X, y)

    return {
        "vectorizer": vectorizer,
        "clf": clf,
        "label_encoder": le,
        "training_subset": sub,
        "class_counts": counts.loc[top_journals],
    }

# ------------------------------------------------------------
# 3) Sidebar controls
# ------------------------------------------------------------

st.sidebar.header("Model settings")

top_n = st.sidebar.slider(
    "Number of journals to include (top N by frequency)",
    min_value=5,
    max_value=50,
    value=20,
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

model_bundle = train_journal_model(df, top_n_journals=top_n, max_features=max_feat)
vec = model_bundle["vectorizer"]
clf = model_bundle["clf"]
le = model_bundle["label_encoder"]
train_sub = model_bundle["training_subset"]
class_counts = model_bundle["class_counts"]

st.sidebar.success(
    f"Model trained on {len(train_sub):,} papers "
    f"from {len(class_counts):,} journals."
)

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

def recommend_journals(title: str, abstract: str, top_k: int = 5):
    text = (title or "").strip() + " " + (abstract or "").strip()
    if not text.strip():
        return None

    X_new = vec.transform([text])
    # probabilities for each journal
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_new)[0]
    else:
        # fallback: use decision_function and softmax-ish scaling
        scores = clf.decision_function(X_new)[0]
        exp_scores = np.exp(scores - scores.max())
        proba = exp_scores / exp_scores.sum()

    order = np.argsort(proba)[::-1][:top_k]
    journals = le.inverse_transform(order)
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
    result = recommend_journals(user_title, user_abs, k_recommend)
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

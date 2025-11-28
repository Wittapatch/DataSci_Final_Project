import streamlit as st
import pandas as pd
import numpy as np
import ast
import sqlite3

from math import inf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
import plotly.express as px
import plotly.figure_factory as ff


def configure_page():
    st.set_page_config(
        page_title="Scopus Citation Classifier",
        layout="wide",
    )


@st.cache_data(show_spinner="Loading Scopus data …")
def load_data(db_path="scopus.db"):
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query("SELECT * FROM research_papers", con)


def parse_keywords(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    return ast.literal_eval(x)


@st.cache_resource(show_spinner="Training model with current parameters ...")
def run_full_pipeline(
    df: pd.DataFrame,
    tf_max_features: int,
    ngram_max: int,
    k_min: int,
    k_max: int,
    k_step: int,
    depth_opts,
    nest_opts,
    lr_opts,
    cv_folds: int,
):
    if not depth_opts:
        depth_opts = [3, 5]
    if not nest_opts:
        nest_opts = [200]
    if not lr_opts:
        lr_opts = [0.1]

    df_clean = df.drop(columns=["file_id", "document_classification_codes"], errors="ignore")

    if "publication_date" in df_clean.columns:
        df_clean["publication_date"] = pd.to_datetime(
            df_clean["publication_date"],
            format="%d/%m/%Y",
            errors="coerce",
        )
        latest_date = df_clean["publication_date"].max()
        df_clean["days_from_latest_date"] = (latest_date - df_clean["publication_date"]).dt.days
        df_clean.drop(columns=["publication_date"], inplace=True)

    df_clean["refcount"] = pd.to_numeric(df_clean["refcount"], errors="coerce").astype("Int64")

    df_clean["citedbycount"] = df_clean["citedbycount"].astype("Int64")
    df_clean = df_clean[pd.notna(df_clean["citedbycount"])]

    y = (df_clean["citedbycount"] > df_clean["citedbycount"].quantile(0.8)).astype(int)

    df_clean["abstracts"] = df_clean["abstracts"].apply(lambda x: "" if pd.isna(x) else x)
    df_clean["comb_keywords"] = df_clean["keywords"].apply(
        lambda x: " ".join(parse_keywords(x))
    )
    df_clean["combined_text"] = (
        df_clean["citation_title"].fillna("")
        + " "
        + df_clean["sourcetitle"].fillna("")
        + " "
        + df_clean["comb_keywords"]
        + " "
        + df_clean["abstracts"]
    )

    tfidf = TfidfVectorizer(
        max_features=tf_max_features,
        stop_words="english",
        ngram_range=(1, ngram_max),
    )
    text_matrix = tfidf.fit_transform(df_clean["combined_text"])
    text_shape = text_matrix.shape

    best_score = -inf
    best_k = None
    best_model = None
    sil_scores = {}
    k_values = list(range(k_min, k_max + 1, k_step))

    for k in k_values:
        km = KMeans(k, random_state=42, n_init=10)
        lab = km.fit_predict(text_matrix)
        score = silhouette_score(text_matrix, lab)
        sil_scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k
            best_model = km

    if best_model is None:
        df_clean["topic_cluster"] = -1
    else:
        df_clean["topic_cluster"] = best_model.predict(text_matrix)

    drop_text_cols = [
        "citation_title",
        "sourcetitle",
        "keywords",
        "comb_keywords",
        "combined_text",
        "allauthors_name",
        "categories",
        "creator",
        "publishername",
        "creator_degree",
        "abstracts",
    ]
    X = df_clean.drop(columns=drop_text_cols, errors="ignore")
    X = X.drop(columns=["citedbycount"], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    param_grid = {
        "max_depth": depth_opts,
        "n_estimators": nest_opts,
        "learning_rate": lr_opts,
    }

    xgb_base = XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )

    grid = GridSearchCV(
        xgb_base,
        param_grid=param_grid,
        n_jobs=-1,
        cv=cv_folds,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    best_xgb = grid.best_estimator_

    y_pred = best_xgb.predict(X_test)
    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred)

    fi = best_xgb.feature_importances_
    fi_df = pd.DataFrame(
        {"feature": feature_names, "importance": fi}
    ).sort_values("importance", ascending=False)

    return {
        "df_clean": df_clean,
        "y": y,
        "text_shape": text_shape,
        "sil_scores": sil_scores,
        "best_k": best_k,
        "best_sil": best_score,
        "best_model": best_model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "report_dict": report_dict,
        "cm": cm,
        "feature_importance": fi_df,
        "best_xgb": best_xgb,
        "tfidf_params": {
            "max_features": tf_max_features,
            "ngram_max": ngram_max,
        },
        "xgb_param_grid": param_grid,
        "k_values": k_values,
        "cv_folds": cv_folds,
    }


def run_models_page():
    df = load_data()

    st.title("Scopus High-Impact Citation Classifier")
    st.caption("TF-IDF  →  KMeans topics  →  XGBoost (Top-20% cited label)")

    st.sidebar.header("Training parameters")

    tf_max_features = st.sidebar.slider(
        "TF-IDF max_features",
        min_value=1000,
        max_value=20000,
        value=5000,
        step=1000,
    )
    ngram_max = st.sidebar.selectbox(
        "TF-IDF max n-gram",
        options=[1, 2, 3],
        index=2,
    )

    k_min, k_max = st.sidebar.slider(
        "KMeans k range",
        min_value=2,
        max_value=40,
        value=(10, 20),
    )
    k_step = st.sidebar.selectbox(
        "KMeans k step",
        options=[1, 2, 3, 4, 5],
        index=2,
    )

    depth_opts = st.sidebar.multiselect(
        "XGB max_depth candidates",
        options=[3, 4, 5, 6, 7, 8, 9],
        default=[3, 5, 7, 9],
    )
    nest_opts = st.sidebar.multiselect(
        "XGB n_estimators candidates",
        options=[100, 200, 400, 800, 1200],
        default=[200, 400, 800],
    )
    lr_opts = st.sidebar.multiselect(
        "XGB learning_rate candidates",
        options=[0.01, 0.05, 0.1, 0.2],
        default=[0.01, 0.05, 0.1],
    )

    cv_folds = st.sidebar.slider(
        "Cross-validation folds (cv)",
        min_value=3,
        max_value=10,
        value=5,
    )

    st.sidebar.markdown("---")
    run_btn = st.sidebar.button("Train")

    st.markdown("---")
    st.subheader("1. Data loading")
    st.write("Current dataframe shape:", df.shape)

    if not run_btn:
        st.info("Set parameters in the sidebar, then click **Run training / re-train**.")
        st.stop()

    results = run_full_pipeline(
        df,
        tf_max_features,
        ngram_max,
        k_min,
        k_max,
        k_step,
        depth_opts,
        nest_opts,
        lr_opts,
        cv_folds,
    )

    st.success("Training finished with current parameters.")

    st.markdown("### 2. Current configuration")
    st.write("**TF-IDF**:", results["tfidf_params"])
    st.write("**KMeans k candidates**:", results["k_values"])
    st.write(
        "**Best k**:",
        results["best_k"],
        "with silhouette ≈",
        round(results["best_sil"], 3),
    )
    st.write("**XGBoost param grid**:", results["xgb_param_grid"])
    st.write("**CV folds**:", results["cv_folds"])

    sil_df = pd.DataFrame(
        {"k_clusters": list(results["sil_scores"].keys()),
         "silhouette": list(results["sil_scores"].values())}
    )
    fig_sil = px.line(
        sil_df,
        x="k_clusters",
        y="silhouette",
        markers=True,
        title="Silhouette score vs number of clusters",
    )
    fig_sil.update_layout(xaxis_title="k", yaxis_title="Silhouette score")
    st.plotly_chart(fig_sil, use_container_width=True)

    st.markdown("---")
    st.subheader("3. XGBoost performance")

    report_df = pd.DataFrame(results["report_dict"]).T
    report_df.rename(columns={"f1-score": "f1_score"}, inplace=True)

    st.markdown(
        """
        <style>
        .metric-card {
            background-color: #8FABD4;
            border-radius: 26px;
            padding: 1.1rem 1rem;
            color: #0b1b33;
            box-shadow: 0 10px 20px rgba(0,0,0,0.08);
            margin: 0 auto 1rem auto;
            max-width: 260px;
        }
        .metric-card h4 {
            font-size: 1.1rem;
            margin: 0 0 0.85rem 0;
            font-weight: 700;
            color: #ffffff;
        }
        .metric-row {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            padding: 0.3rem 0;
        }
        .metric-label {
            font-size: 0.95rem;
            color: rgba(255,255,255,0.96);
            font-weight: 700;
        }
        .metric-value {
            font-size: 1.05rem;
            font-weight: 800;
            color: #0b1b33;
            background-color: #E8F9FF;
            padding: 0.1rem 0.6rem;
            border-radius: 999px;
            display: inline-block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    metric_map = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1 score"),
        ("support", "Support"),
    ]

    metric_cols = st.columns(len(metric_map))
    for (metric_key, display), col in zip(metric_map, metric_cols):
        with col:
            col_df = (
                report_df[[metric_key]]
                .reset_index()
                .rename(columns={"index": "label", metric_key: display})
            )

            if metric_key != "support":
                col_df[display] = col_df[display].apply(lambda x: "-" if pd.isna(x) else f"{x:.3f}")
            else:
                col_df[display] = col_df[display].apply(lambda x: "-" if pd.isna(x) else f"{int(x)}")

            rows_html = "".join(
                f"<div class='metric-row'><span class='metric-label'>{row['label']}</span>"
                f"<span class='metric-value'>{row[display]}</span></div>"
                for _, row in col_df.iterrows()
            )

            st.markdown(
                f"<div class='metric-card'><h4>{display}</h4>{rows_html}</div>",
                unsafe_allow_html=True,
            )

    cm = results["cm"]
    fig_cm = ff.create_annotated_heatmap(
        z=cm.astype(float),
        x=["Pred 0", "Pred 1"],
        y=["True 0", "True 1"],
        colorscale="Blues",
        showscale=True,
    )
    fig_cm.update_layout(
        title="Confusion matrix",
        xaxis_title="Predicted label",
        yaxis_title="True label",
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    st.subheader("4. Feature importance")

    fi_df = results["feature_importance"]

    st.markdown(
        """
        <style>
        .fi-card {
            background-color: #8FABD4;
            border-radius: 22px;
            padding: 1.25rem;
            color: #ffffff;
            box-shadow: 0 10px 20px rgba(0,0,0,0.08);
            margin-bottom: 1rem;
        }
        .fi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.75rem 1rem;
        }
        .fi-item {
            display: flex;
            flex-direction: column;
        }
        .fi-name {
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .fi-score {
            font-size: 1.05rem;
            font-weight: 800;
            color: #0b1b33;
            background-color: #E8F9FF;
            padding: 0.1rem 0.5rem;
            border-radius: 999px;
            display: inline-block;
            align-self: flex-start;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    top_fi = fi_df.head(12).reset_index(drop=True)
    fi_cards = "".join(
        f"<div class='fi-item'><div class='fi-name'>{row['feature']}</div>"
        f"<div class='fi-score'>{row['importance']:.4f}</div></div>"
        for _, row in top_fi.iterrows()
    )
    st.markdown(f"<div class='fi-card'><div class='fi-grid'>{fi_cards}</div></div>", unsafe_allow_html=True)

    fig_fi = px.bar(
        fi_df.head(20),
        x="importance",
        y="feature",
        orientation="h",
        title="Top features by importance",
    )
    fig_fi.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=500,
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")
    st.subheader("5. Preview of cleaned data (df_clean)")
    st.dataframe(results["df_clean"].head(20), use_container_width=True)


def main():
    configure_page()
    run_models_page()


if __name__ == "__main__":
    main()

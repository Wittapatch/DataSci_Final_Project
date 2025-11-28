import sqlite3
import json
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

@st.cache_data(show_spinner="Loading Scopus data …")
def load_df(db_path="scopus.db"):
  with sqlite3.connect(db_path) as con:
    df = pd.read_sql_query("SELECT * FROM research_papers", con)
  if "year" in df.columns:
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
  return df


def parse_json_list(s):
  if pd.isna(s):
    return []
  if isinstance(s, (list, tuple)):
    return list(s)
  try:
    v = json.loads(s)
    return v if isinstance(v, list) else []
  except Exception:
    return []


def categories_to_subjects(s):
  subjects = []
  for item in parse_json_list(s):
    if isinstance(item, dict):
      for k in item.keys():
        subjects.append(k)
  return subjects


def run_eda_page(db_path: str = "scopus.db") -> None:
  df = load_df(db_path)

  df = df.copy()
  author_col = "authors_deg_name_json" if "authors_deg_name_json" in df.columns else "authors"
  if author_col in df.columns:
    df["_authors_list"] = df[author_col].apply(parse_json_list)
  else:
    df["_authors_list"] = [[] for _ in range(len(df))]
  if "authors_count" in df.columns:
    df["authors_count"] = pd.to_numeric(df["authors_count"], errors="coerce").fillna(0).astype(int)
  else:
    df["authors_count"] = df["_authors_list"].apply(len)
  if "keywords" in df.columns:
    df["_keywords_list"] = df["keywords"].apply(parse_json_list)
  else:
    df["_keywords_list"] = [[] for _ in range(len(df))]
  if "categories" in df.columns:
    df["_subjects_list"] = df["categories"].apply(categories_to_subjects)
  else:
    df["_subjects_list"] = [[] for _ in range(len(df))]

  st.title("📚 Scopus EDA Dashboard")
  st.caption("Reads from scopus.db → research_papers. No rows dropped; missing values preserved.")

  st.sidebar.header("EDA Filters")

  years = sorted([int(y) for y in df["year"].dropna().unique()]) if "year" in df.columns else []
  year_sel = st.sidebar.multiselect("Year", years, default=years, key="eda_year_sel")

  pubs = (
    df["publishername"].fillna("—")
      .value_counts()
      .head(30)
      .index.tolist()
  ) if "publishername" in df.columns else []
  pub_sel = st.sidebar.multiselect("Publisher (top 30)", pubs, default=pubs, key="eda_pub_sel")

  source_sel = st.sidebar.text_input("Source title contains", "", key="eda_source")
  q_text = st.sidebar.text_input("Search in title / abstract / keywords", "", key="eda_query")

  cmin = int(pd.to_numeric(df["citedbycount"], errors="coerce").min(skipna=True) or 0)
  cmax = int(pd.to_numeric(df["citedbycount"], errors="coerce").max(skipna=True) or 0)
  cmin_sel, cmax_sel = st.sidebar.slider("Cited-by range", cmin, cmax, (cmin, cmax), key="eda_cited_slider")

  mask = pd.Series(True, index=df.index)

  if year_sel and years and len(year_sel) != len(years):
    mask &= df["year"].isin(year_sel)

  if pub_sel and pubs and len(pub_sel) != len(pubs):
    mask &= df["publishername"].fillna("—").isin(pub_sel)

  if source_sel.strip():
    s = source_sel.strip().lower()
    mask &= df["sourcetitle"].fillna("").str.lower().str.contains(s)

  if q_text.strip():
    q = q_text.strip().lower()
    in_title = df["citation_title"].fillna("").str.lower().str.contains(q)
    in_abs = df["abstracts"].fillna("").str.lower().str.contains(q)
    in_kw = df["_keywords_list"].apply(lambda lst: any(q in str(x).lower() for x in lst))
    mask &= (in_title | in_abs | in_kw)

  cb = pd.to_numeric(df["citedbycount"], errors="coerce")
  mask &= (cb.between(cmin_sel, cmax_sel, inclusive="both") | cb.isna())

  dff = df.loc[mask].copy()

  c1, c2, c3, c4 = st.columns(4)
  with c1:
    st.metric("Papers", len(dff))
  with c2:
    st.metric("Years covered", dff["year"].nunique())
  with c3:
    st.metric("Authors (sum)", int(dff["authors_count"].sum()))
  with c4:
    st.metric("Cited-by (sum)", int(pd.to_numeric(dff["citedbycount"], errors="coerce").sum(skipna=True)))

  st.subheader("Missing values by column (filtered data)")
  miss_df = pd.DataFrame({
    "column": dff.columns,
    "missing": dff.isna().sum().values,
  })
  miss_df["missing_pct"] = (miss_df["missing"] / len(dff)).round(4) * 100
  miss_df = miss_df.sort_values("missing", ascending=False)
  st.dataframe(miss_df, hide_index=True, use_container_width=True)

  st.divider()

  papers_per_year = (
    dff.groupby("year", dropna=False)
       .size()
       .reset_index(name="papers")
       .sort_values("year")
  )
  st.plotly_chart(
    px.bar(papers_per_year, x="year", y="papers", title="Papers per Year").update_layout(xaxis=dict(dtick=1)),
    use_container_width=True,
  )

  authors_per_year = (
    dff.groupby("year", dropna=False)["authors_count"]
       .sum(min_count=1)
       .reset_index(name="authors_sum")
       .sort_values("year")
  )
  st.plotly_chart(
    px.bar(authors_per_year, x="year", y="authors_sum", title="Authors (sum) per Year").update_layout(xaxis=dict(dtick=1)),
    use_container_width=True,
  )

  cited_by_year = (
    dff.assign(citedbycount_num=pd.to_numeric(dff["citedbycount"], errors="coerce"))
       .groupby("year", dropna=False)["citedbycount_num"]
       .sum(min_count=1)
       .reset_index(name="citedby_sum")
       .sort_values("year")
  )
  st.plotly_chart(
    px.bar(cited_by_year, x="year", y="citedby_sum", title="Cited-by (sum) per Year").update_layout(xaxis=dict(dtick=1)),
    use_container_width=True,
  )

  ref_per_year = (
    dff.assign(refcount_num=pd.to_numeric(dff["refcount"], errors="coerce"))
       .groupby("year", dropna=False)["refcount_num"]
       .sum(min_count=1)
       .reset_index(name="refcount_sum")
       .sort_values("year")
  )
  st.plotly_chart(
    px.bar(ref_per_year, x="year", y="refcount_sum", title="Reference Count (sum) per Year").update_layout(xaxis=dict(dtick=1)),
    use_container_width=True,
  )

  subjects_long = (
    dff.loc[:, ["year", "_subjects_list"]]
    .explode("_subjects_list")
    .rename(columns={"_subjects_list": "subject"})
  )
  subjects_long = subjects_long[subjects_long["subject"].notna() & (subjects_long["subject"] != "")]
  if not subjects_long.empty:
    top_subjects = subjects_long["subject"].value_counts().head(20).reset_index()
    top_subjects.columns = ["subject", "count"]
    st.plotly_chart(
      px.bar(top_subjects, x="subject", y="count", title="Top 20 Subjects").update_layout(xaxis_tickangle=-45),
      use_container_width=True,
    )

    sub_year = (
      subjects_long.groupby(["year", "subject"])
      .size()
      .reset_index(name="count")
      .sort_values(["year", "count"], ascending=[True, False])
    )
    st.plotly_chart(
      px.bar(
        sub_year,
        x="year",
        y="count",
        color="subject",
        barmode="stack",
        title="Subjects per Year (stacked)",
      ).update_layout(xaxis=dict(dtick=1)),
      use_container_width=True,
    )

  kw_long = (
    dff.loc[:, ["year", "_keywords_list"]]
    .explode("_keywords_list")
    .rename(columns={"_keywords_list": "keyword"})
  )
  kw_long = kw_long[kw_long["keyword"].notna() & (kw_long["keyword"] != "")]
  if not kw_long.empty:
    top_kw = kw_long["keyword"].value_counts().head(30).reset_index()
    top_kw.columns = ["keyword", "count"]
    st.plotly_chart(
      px.bar(top_kw, x="keyword", y="count", title="Top 30 Author Keywords").update_layout(xaxis_tickangle=-45),
      use_container_width=True,
    )

  st.divider()

  st.subheader("Filtered rows (first 200)")
  show_cols = [
    "year",
    "publication_date",
    "citation_title",
    "publishername",
    "sourcetitle",
    "creator",
    "document_classification_codes",
    "refcount",
    "citedbycount",
    "authors_deg_name_json",
    "authors_count",
    "categories",
    "keywords",
  ]
  present_cols = [c for c in show_cols if c in dff.columns]
  st.dataframe(dff[present_cols].head(200), use_container_width=True)

  st.download_button(
    "Download filtered CSV",
    data=dff.to_csv(index=False),
    file_name="scopus_filtered.csv",
    mime="text/csv",
  )

  avg_authors_year = (
    dff.assign(authors_count_num=pd.to_numeric(dff["authors_count"], errors="coerce"))
       .groupby("year", dropna=False)["authors_count_num"]
       .mean()
       .reset_index(name="avg_authors_per_paper")
       .sort_values("year")
  )

  fig_avg = px.line(
    avg_authors_year,
    x="year",
    y="avg_authors_per_paper",
    markers=True,
    title="Average Authors per Paper by Year",
  ).update_layout(
    xaxis=dict(dtick=1),
    yaxis=dict(title="Avg authors / paper", tickformat=".2f"),
  )

  st.plotly_chart(fig_avg, use_container_width=True)


if __name__ == "__main__":
  st.set_page_config(page_title="Scopus EDA", layout="wide")
  run_eda_page()

# app.py
import sqlite3
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go  # Used for the cumulative chart
import streamlit as st

st.set_page_config(page_title="Scopus EDA", layout="wide")

# --- SQL (Modified to only pull columns you listed) ---
# This query is based on your 'eda.py' and matches your new column list.
# It excludes the complex parsing for affiliations and funding.
SQL = """
SELECT
  file_id,
  year,

  -- basics
  json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head."citation-title"') AS citation_title,
  json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head.abstracts')        AS abstracts,
  json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head.source.publisher.publishername') AS publishername,
  json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head.source.sourcetitle')             AS sourcetitle,

  /* publication_date: DD/MM/YYYY */
  CASE
    WHEN json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head.source.publicationdate.day')   IS NOT NULL
     AND json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head.source.publicationdate.month') IS NOT NULL
     AND json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head.source.publicationdate.year')  IS NOT NULL
    THEN printf('%02d/%02d/%04d',
                CAST(json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head.source.publicationdate.day')   AS INTEGER),
                CAST(json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head.source.publicationdate.month') AS INTEGER),
                CAST(json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head.source.publicationdate.year')  AS INTEGER))
    ELSE NULL
  END AS publication_date,

  /* ce:doi */
  COALESCE(
    json_extract(raw_json,'$."abstracts-retrieval-response".item."item-info"."itemidlist"."ce:doi"'),
    (SELECT t.value
     FROM json_tree(raw_json, '$."abstracts-retrieval-response"') AS t
     WHERE t.key = 'ce:doi'
     LIMIT 1)
  ) AS document_classification_codes,

  -- counts
  json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.tail.bibliography."@refcount"') AS refcount,
  CAST(
    COALESCE(
      json_extract(raw_json,'$."abstracts-retrieval-response".coredata."citedby-count"'),
      json_extract(raw_json,'$."abstracts-retrieval-response".item.coredata."citedby-count"'),
      (SELECT t.value
       FROM json_tree(raw_json, '$."abstracts-retrieval-response"') AS t
       WHERE t.key = 'citedby-count'
       LIMIT 1)
    ) AS INTEGER
  ) AS citedbycount,

  /* authors (degree + given + surname) as JSON array of strings */
  (
    SELECT json_group_array(name_str)
    FROM (
      SELECT
        TRIM(
          COALESCE(json_extract(a.value,'$."ce:degrees"') || ' ', '') ||
          TRIM(
            COALESCE(json_extract(a.value,'$."preferred-name"."ce:given-name"'),
                     json_extract(a.value,'$."ce:given-name"'), '') || ' ' ||
            COALESCE(json_extract(a.value,'$."preferred-name"."ce:surname"'),
                     json_extract(a.value,'$."ce:surname"'), '')
          )
        ) AS name_str
      FROM json_each(
             COALESCE(
               json_extract(raw_json,'$."abstracts-retrieval-response".authors.author'),
               json_array()
             )
           ) AS a
      WHERE TRIM(
              COALESCE(json_extract(a.value,'$."preferred-name"."ce:given-name"'),
                       json_extract(a.value,'$."ce:given-name"'), '') || ' ' ||
              COALESCE(json_extract(a.value,'$."preferred-name"."ce:surname"'),
                       json_extract(a.value,'$."ce:surname"'), '')
            ) <> ''
    )
  ) AS authors_deg_name_json,

  /* categories (subject → abbrev) as JSON array of {"<name>":"<abbrev>"} */
  (
    SELECT json_group_array(json_object(subject, abbrev))
    FROM (
      SELECT DISTINCT
        json_extract(sa.value,'$."$"')        AS subject,
        json_extract(sa.value,'$."@abbrev"')  AS abbrev
      FROM (
        -- primary
        SELECT * FROM json_each(
          CASE json_type(json_extract(raw_json,'$."abstracts-retrieval-response"."subject-areas"."subject-area"'))
            WHEN 'array'  THEN json_extract(raw_json,'$."abstracts-retrieval-response"."subject-areas"."subject-area"')
            WHEN 'object' THEN json_array(json_extract(raw_json,'$."abstracts-retrieval-response"."subject-areas"."subject-area"'))
            ELSE json_array()
          END
        )
        UNION ALL
        -- fallback A
        SELECT * FROM json_each(
          CASE json_type(json_extract(raw_json,'$."abstracts-retrieval-response".item.coredata."subject-areas"."subject-area"'))
            WHEN 'array'  THEN json_extract(raw_json,'$."abstracts-retrieval-response".item.coredata."subject-areas"."subject-area"')
            WHEN 'object' THEN json_array(json_extract(raw_json,'$."abstracts-retrieval-response".item.coredata."subject-areas"."subject-area"'))
            ELSE json_array()
          END
        )
        UNION ALL
        -- fallback B
        SELECT * FROM json_each(
          CASE json_type(json_extract(raw_json,'$."abstracts-retrieval-response".coredata."subject-areas"."subject-area"'))
            WHEN 'array'  THEN json_extract(raw_json,'$."abstracts-retrieval-response".coredata."subject-areas"."subject-area"')
            WHEN 'object' THEN json_array(json_extract(raw_json,'$."abstracts-retrieval-response".coredata."subject-areas"."subject-area"'))
            ELSE json_array()
          END
        )
      ) AS sa
      WHERE subject IS NOT NULL AND abbrev IS NOT NULL
    )
  ) AS categories,

  /* creator = "<given> <surname>" (first creator found) */
  (
    SELECT name_full
    FROM (
      SELECT TRIM(
               COALESCE(json_extract(a.value,'$."preferred-name"."ce:given-name"'),
                        json_extract(a.value,'$."ce:given-name"'), '') || ' ' ||
               COALESCE(json_extract(a.value,'$."preferred-name"."ce:surname"'),
                        json_extract(a.value,'$."ce:surname"'), '')
             ) AS name_full,
             json_extract(a.value,'$."ce:degrees"') AS degree
      FROM (
        SELECT * FROM json_each(
          CASE json_type(json_extract(raw_json,'$."abstracts-retrieval-response".item.coredata."dc:creator".author'))
            WHEN 'array'  THEN json_extract(raw_json,'$."abstracts-retrieval-response".item.coredata."dc:creator".author')
            WHEN 'object' THEN json_array(json_extract(raw_json,'$."abstracts-retrieval-response".item.coredata."dc:creator".author'))
            ELSE json_array()
          END
        )
        UNION ALL
        SELECT * FROM json_each(
          CASE json_type(json_extract(raw_json,'$."abstracts-retrieval-response".coredata."dc:creator".author'))
            WHEN 'array'  THEN json_extract(raw_json,'$."abstracts-retrieval-response".coredata."dc:creator".author')
            WHEN 'object' THEN json_array(json_extract(raw_json,'$."abstracts-retrieval-response".coredata."dc:creator".author'))
            ELSE json_array()
          END
        )
      ) AS a
      WHERE name_full <> ''
      LIMIT 1
    )
  ) AS creator,
  
    /* creator_degree (first creator found) */
  (
    SELECT degree
    FROM (
      SELECT json_extract(a.value,'$."ce:degrees"') AS degree,
             TRIM(
               COALESCE(json_extract(a.value,'$."preferred-name"."ce:given-name"'),
                        json_extract(a.value,'$."ce:given-name"'), '') || ' ' ||
               COALESCE(json_extract(a.value,'$."preferred-name"."ce:surname"'),
                        json_extract(a.value,'$."ce:surname"'), '')
             ) AS name_full
      FROM (
        SELECT * FROM json_each(
          CASE json_type(json_extract(raw_json,'$."abstracts-retrieval-response".item.coredata."dc:creator".author'))
            WHEN 'array'  THEN json_extract(raw_json,'$."abstracts-retrieval-response".item.coredata."dc:creator".author')
            WHEN 'object' THEN json_array(json_extract(raw_json,'$."abstracts-retrieval-response".item.coredata."dc:creator".author'))
            ELSE json_array()
          END
        )
        UNION ALL
        SELECT * FROM json_each(
          CASE json_type(json_extract(raw_json,'$."abstracts-retrieval-response".coredata."dc:creator".author'))
            WHEN 'array'  THEN json_extract(raw_json,'$."abstracts-retrieval-response".coredata."dc:creator".author')
            WHEN 'object' THEN json_array(json_extract(raw_json,'$."abstracts-retrieval-response".coredata."dc:creator".author'))
            ELSE json_array()
          END
        )
      ) AS a
      WHERE name_full <> ''
      LIMIT 1
    )
  ) AS creator_degree,


  /* keywords (JSON array of "$" strings) */
  (
    SELECT json_group_array(kw_src.kw)
    FROM (
      -- Path 1: head → citation-info → author-keywords
      SELECT json_extract(k.value,'$."$"') AS kw
      FROM json_each(
        CASE json_type(json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head."citation-info"."author-keywords"."author-keyword"'))
          WHEN 'array'  THEN json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head."citation-info"."author-keywords"."author-keyword"')
          WHEN 'object' THEN json_array(json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head."citation-info"."author-keywords"."author-keyword"'))
          ELSE json_array()
        END
      ) AS k
      UNION ALL
      -- Path 2: top-level authkeywords
      SELECT json_extract(k2.value,'$."$"') AS kw
      FROM json_each(
        CASE json_type(json_extract(raw_json,'$."abstracts-retrieval-response"."authkeywords"."author-keyword"'))
          WHEN 'array'  THEN json_extract(raw_json,'$."abstracts-retrieval-response"."authkeywords"."author-keyword"')
          WHEN 'object' THEN json_array(json_extract(raw_json,'$."abstracts-retrieval-response"."authkeywords"."author-keyword"'))
          ELSE json_array()
        END
      ) AS k2
    ) AS kw_src
    WHERE kw_src.kw IS NOT NULL
  ) AS keywords

FROM papers_raw
ORDER BY year, file_id;
"""

# --- Helper functions ---
def parse_json_list(s):
    """Parses a JSON string representing a list into a Python list."""
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
    """Converts the 'categories' JSON into a simple list of subject names."""
    subjects = []
    for item in parse_json_list(s):
        if isinstance(item, dict):
            for k in item.keys():
                subjects.append(k)
    return subjects

# --- Load data (cached) ---
@st.cache_data(show_spinner=False)
def load_df(db_path="scopus.db"):
    """Loads the data from SQLite, parses JSON, and creates derived columns."""
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(SQL, con)
    finally:
        con.close()
    
    # --- Base Column Type Conversion ---
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["citedbycount_num"] = pd.to_numeric(df["citedbycount"], errors="coerce").fillna(0).astype(int)
    df["refcount_num"] = pd.to_numeric(df["refcount"], errors="coerce").fillna(0).astype(int)

    # --- Derived Columns (from JSON) ---
    df = df.copy()
    
    # Authors
    df["_authors_list"] = df["authors_deg_name_json"].apply(parse_json_list)
    df["authors_count"] = df["_authors_list"].apply(len)
    
    # Keywords
    df["_keywords_list"] = df["keywords"].apply(parse_json_list)
    
    # Subjects
    df["_subjects_list"] = df["categories"].apply(categories_to_subjects)

    # --- Add datetime column for monthly grouping ---
    df["publication_date_dt"] = pd.to_datetime(df["publication_date"], format='%d/%m/%Y', errors='coerce')

    return df

with st.spinner("Loading and processing data from scopus.db..."):
    df = load_df()

st.title("📚 Scopus EDA Dashboard")
st.caption("Dashboard analyzing publications, authors, citations, subjects, and keywords.")

# --- Sidebar filters ---
st.sidebar.header("Filters")

years = sorted([int(y) for y in df["year"].dropna().unique()])
year_sel = st.sidebar.multiselect("Year", years, default=years)

pubs = (
    df["publishername"].fillna("—")
      .value_counts()
      .head(30)
      .index.tolist()
)
pub_sel = st.sidebar.multiselect("Publisher (top 30)", pubs, default=pubs)

source_sel = st.sidebar.text_input("Source title contains", "")
q_text = st.sidebar.text_input("Search in title / abstract / keywords", "")

cmin = int(df["citedbycount_num"].min())
cmax = int(df["citedbycount_num"].max())
cmin_sel, cmax_sel = st.sidebar.slider("Cited-by range", cmin, cmax, (cmin, cmax))

# --- Apply filters ---
mask = pd.Series(True, index=df.index)

if year_sel:
    mask &= df["year"].isin(year_sel)

if pub_sel:
    mask &= df["publishername"].fillna("—").isin(pub_sel)

if source_sel.strip():
    s = source_sel.strip().lower()
    mask &= df["sourcetitle"].fillna("").str.lower().str.contains(s)

if q_text.strip():
    q = q_text.strip().lower()
    in_title = df["citation_title"].fillna("").str.lower().str.contains(q)
    in_abs   = df["abstracts"].fillna("").str.lower().str.contains(q)
    in_kw    = df["_keywords_list"].apply(lambda lst: any(q in str(x).lower() for x in lst))
    mask &= (in_title | in_abs | in_kw)

mask &= (df["citedbycount_num"].between(cmin_sel, cmax_sel, inclusive="both"))

dff = df.loc[mask].copy()

# --- Pre-calculate exploded DataFrames for charts ---
try:
    # Explode subjects
    dff_subjects_long = dff.loc[:, ["file_id", "year", "_subjects_list", "citedbycount_num"]].explode("_subjects_list")
    dff_subjects_long = dff_subjects_long.dropna(subset=["_subjects_list"])
    dff_subjects_long = dff_subjects_long[dff_subjects_long["_subjects_list"] != ""]
    
    # Explode keywords
    dff_kw_long = dff.loc[:, ["file_id", "year", "_keywords_list"]].explode("_keywords_list")
    dff_kw_long = dff_kw_long.dropna(subset=["_keywords_list"])
    dff_kw_long = dff_kw_long[dff_kw_long["_keywords_list"] != ""]

except Exception as e:
    st.error(f"An error occurred during data processing: {e}")
    # Create empty DataFrames to avoid crashing the app
    dff_subjects_long = pd.DataFrame(columns=["file_id", "year", "_subjects_list", "citedbycount_num"])
    dff_kw_long = pd.DataFrame(columns=["file_id", "year", "_keywords_list"])


# --- KPI cards ---
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Papers", len(dff))
with c2:
    st.metric("Years covered", dff["year"].nunique())
with c3:
    st.metric("Authors (sum)", int(dff["authors_count"].sum()))
with c4:
    st.metric("Cited-by (sum)", int(dff["citedbycount_num"].sum()))

st.divider()

# --- Publication Trends (per Month) ---
st.subheader("Publication Trends")

# Chart: Papers per Month (with cumulative)
# Use the pre-converted 'publication_date_dt' column
monthly_data = dff.dropna(subset=["publication_date_dt"]).copy()
monthly_data["pub_month_year"] = monthly_data["publication_date_dt"].dt.strftime('%Y-%m')

papers_per_month = (
    monthly_data.groupby("pub_month_year")
                .size()
                .reset_index(name="papers")
                .sort_values("pub_month_year")
)

if not papers_per_month.empty:
    papers_per_month["cumulative_papers"] = papers_per_month["papers"].cumsum()
    
    show_cumulative = st.toggle("Show cumulative paper count", value=False)

    fig_papers_month = px.bar(papers_per_month, x="pub_month_year", y="papers", title="Papers per Month")
    fig_papers_month.update_xaxes(title_text="Month")
    
    if show_cumulative:
        fig_papers_month.add_trace(go.Scatter(
            x=papers_per_month["pub_month_year"],
            y=papers_per_month["cumulative_papers"],
            name="Cumulative",
            yaxis="y2"
        ))
    
    fig_papers_month.update_layout(
        yaxis_title="Papers",
        yaxis2=dict(title="Cumulative Papers", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_papers_month, use_container_width=True)
else:
    st.write("No valid publication date data found for the filtered selection to display monthly trends.")


# --- Popularity & Impact Charts ---
st.subheader("Popularity & Impact Analysis")

# Chart: Avg Cited-by and Avg Authors per Year
c1, c2 = st.columns(2)
with c1:
    avg_cited_by_year = (
        dff.groupby("year")["citedbycount_num"]
           .mean()
           .reset_index(name="avg_cited_by")
           .sort_values("year")
    )
    st.plotly_chart(px.bar(avg_cited_by_year, x="year", y="avg_cited_by", title="Average Cited-by Count per Year").update_layout(xaxis=dict(dtick=1)), use_container_width=True)

with c2:
    avg_authors_year = (
        dff.groupby("year")["authors_count"]
           .mean()
           .reset_index(name="avg_authors")
           .sort_values("year")
    )
    st.plotly_chart(px.bar(avg_authors_year, x="year", y="avg_authors", title="Average Authors per Paper, per Year").update_layout(xaxis=dict(dtick=1)), use_container_width=True)

# Chart: Popularity by Subject
if not dff_subjects_long.empty:
    st.markdown("### Subject Popularity")
    subject_stats = (
        dff_subjects_long.groupby("_subjects_list")
                         .agg(
                             papers_count=("_subjects_list", "size"),
                             avg_cited_by=("citedbycount_num", "mean")
                         )
                         .reset_index()
                         .rename(columns={"_subjects_list": "subject"})
    )
    
    metric_sel = st.radio(
        "Measure subject popularity by:",
        ("Paper Count", "Average Cited-by"),
        horizontal=True,
        key="subject_metric"
    )
    
    if metric_sel == "Paper Count":
        sort_by, y_col, title = "papers_count", "papers_count", "Top 20 Subjects by Paper Count"
    else:
        sort_by, y_col, title = "avg_cited_by", "avg_cited_by", "Top 20 Subjects by Average Cited-by"
        
    top_subjects = subject_stats.sort_values(sort_by, ascending=False).head(20)
    st.plotly_chart(px.bar(top_subjects, x="subject", y=y_col, title=title).update_layout(xaxis_tickangle=-45), use_container_width=True)

# Interactive: Top Papers by Subject
if not dff_subjects_long.empty:
    st.markdown("### Most Popular Papers by Subject")
    all_subjects = sorted(list(dff_subjects_long["_subjects_list"].unique()))
    subject_select = st.selectbox("Select Subject to see top papers", all_subjects)
    
    if subject_select:
        # Find all papers that have this subject
        paper_ids_in_subject = dff_subjects_long[dff_subjects_long["_subjects_list"] == subject_select]["file_id"].unique()
        
        # Filter the main DataFrame for these papers
        top_papers_in_subject = (
            dff[dff["file_id"].isin(paper_ids_in_subject)]
            .sort_values("citedbycount_num", ascending=False)
            .head(10)
        )
        st.dataframe(
            top_papers_in_subject[["citation_title", "citedbycount_num", "year", "sourcetitle"]],
            use_container_width=True
        )

# Chart: Reference Ratio Histogram
st.markdown("### Reference Analysis")
# Modified to only divide by authors_count, since funding_count is no longer available
dff["ref_ratio"] = dff["refcount_num"] / dff["authors_count"].replace(0, 1)
st.plotly_chart(
    px.histogram(dff, x="ref_ratio", title="Reference Ratio (References / Authors)"),
    use_container_width=True
)

st.divider()

# --- MODIFIED: Subject & Keyword Charts ---
st.subheader("Subjects & Keywords")

c1, c2 = st.columns(2)

with c1:
    # MODIFIED: Pie Chart: Top Subjects
    if not dff_subjects_long.empty:
        subject_counts = dff_subjects_long["_subjects_list"].value_counts()
        fig_pie_subject = px.pie(
            subject_counts.head(10),
            values=subject_counts.head(10).values,
            names=subject_counts.head(10).index,
            title="Top 10 Subjects"
        )
        fig_pie_subject.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie_subject, use_container_width=True)
    else:
        st.write("No subject data to display.")

with c2:
    # Bar Chart: Top Keywords (Unchanged)
    if not dff_kw_long.empty:
        top_kw = dff_kw_long["_keywords_list"].value_counts().head(20).reset_index()
        top_kw.columns = ["keyword", "count"]
        
        fig_bar_kw = px.bar(
            top_kw.sort_values("count"), # Sort ascending for Plotly horizontal bar
            x="count",
            y="keyword",
            orientation='h',
            title="Top 20 Author Keywords"
        )
        st.plotly_chart(fig_bar_kw, use_container_width=True)
    else:
        st.write("No keyword data to display.")

st.divider()

# --- Preview + Download ---
st.subheader(f"Filtered rows ({len(dff)}) - preview of first 200")
# Modified column list to match your provided list
show_cols = [
    "year", "publication_date", "citation_title", "publishername", "sourcetitle",
    "creator", "creator_degree", "citedbycount", "refcount", "authors_count",
    "authors_deg_name_json", "categories", "keywords", "document_classification_codes"
]
present_cols = [c for c in show_cols if c in dff.columns]
st.dataframe(dff[present_cols].head(200), use_container_width=True)

@st.cache_data
def convert_df_to_csv(df_to_convert):
    # Use only the columns that are allowed
    present_cols_in_df = [c for c in show_cols if c in df_to_convert.columns]
    return df_to_convert[present_cols_in_df].to_csv(index=False).encode('utf-8')

st.download_button(
    "Download filtered CSV",
    data=convert_df_to_csv(dff),
    file_name="scopus_filtered.csv",
    mime="text/csv"
)
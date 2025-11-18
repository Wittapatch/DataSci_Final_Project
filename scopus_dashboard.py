# scopus_dashboard.py

import sqlite3
import json
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Streamlit config
# ------------------------------------------------------------
st.set_page_config(page_title="Scopus Dashboard", layout="wide")

# ------------------------------------------------------------
# SQL query
# ------------------------------------------------------------
SQL = r"""
WITH base AS (
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

    /* ce:doi -> document_classification_codes */
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

    /* authors -> JSON array of "<given> <surname>" */
    (
      SELECT COALESCE(json_group_array(name_str), json_array())
      FROM (
        SELECT
          TRIM(
            TRIM(COALESCE(json_extract(a.value,'$."preferred-name"."ce:given-name"'),
                          json_extract(a.value,'$."ce:given-name"'), ''))
            || ' ' ||
            TRIM(COALESCE(json_extract(a.value,'$."preferred-name"."ce:surname"'),
                          json_extract(a.value,'$."ce:surname"'), ''))
          ) AS name_str
        FROM json_each(
               CASE json_type(json_extract(raw_json,'$."abstracts-retrieval-response".authors.author'))
                 WHEN 'array'  THEN json_extract(raw_json,'$."abstracts-retrieval-response".authors.author')
                 WHEN 'object' THEN json_array(json_extract(raw_json,'$."abstracts-retrieval-response".authors.author'))
                 ELSE json_array()
               END
             ) AS a
        WHERE TRIM(
                COALESCE(json_extract(a.value,'$."preferred-name"."ce:given-name"'),
                         json_extract(a.value,'$."ce:given-name"'), '') || ' ' ||
                COALESCE(json_extract(a.value,'$."preferred-name"."ce:surname"'),
                         json_extract(a.value,'$."ce:surname"'), '')
              ) <> ''
      )
    ) AS authors,

    /* categories (subject → abbrev) */
    (
      SELECT json_group_array(json_object(subject, abbrev))
      FROM (
        SELECT DISTINCT
          json_extract(sa.value,'$."$"')        AS subject,
          json_extract(sa.value,'$."@abbrev"')  AS abbrev
        FROM (
          SELECT * FROM json_each(
            CASE json_type(json_extract(raw_json,'$."abstracts-retrieval-response"."subject-areas"."subject-area"'))
              WHEN 'array'  THEN json_extract(raw_json,'$."abstracts-retrieval-response"."subject-areas"."subject-area"')
              WHEN 'object' THEN json_array(json_extract(raw_json,'$."abstracts-retrieval-response"."subject-areas"."subject-area"'))
              ELSE json_array()
            END
          )
          UNION ALL
          SELECT * FROM json_each(
            CASE json_type(json_extract(raw_json,'$."abstracts-retrieval-response".item.coredata."subject-areas"."subject-area"'))
              WHEN 'array'  THEN json_extract(raw_json,'$."abstracts-retrieval-response".item.coredata."subject-areas"."subject-area"')
              WHEN 'object' THEN json_array(json_extract(raw_json,'$."abstracts-retrieval-response".item.coredata."subject-areas"."subject-area"'))
              ELSE json_array()
            END
          )
          UNION ALL
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

    /* creator = "<given> <surname>" */
    (
      SELECT name_full
      FROM (
        SELECT TRIM(
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
    ) AS creator,

    /* keywords */
    (
      SELECT json_group_array(kw_src.kw)
      FROM (
        SELECT json_extract(k.value,'$."$"') AS kw
        FROM json_each(
          CASE json_type(json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head."citation-info"."author-keywords"."author-keyword"'))
            WHEN 'array'  THEN json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head."citation-info"."author-keywords"."author-keyword"')
            WHEN 'object' THEN json_array(json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head."citation-info"."author-keywords"."author-keyword"'))
            ELSE json_array()
          END
        ) AS k
        UNION ALL
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
    ) AS keywords,

    /* funding (array of agency names) */
    COALESCE((
      SELECT json_group_array(f_src.name)
      FROM (
        SELECT json_extract(f.value, '$."xocs:funding-agency-matched-string"') AS name
        FROM json_each(
          CASE json_type(json_extract(raw_json,
            '$."abstracts-retrieval-response".item."xocs:meta"."xocs:funding-list"."xocs:funding"'))
            WHEN 'array'  THEN json_extract(raw_json,
              '$."abstracts-retrieval-response".item."xocs:meta"."xocs:funding-list"."xocs:funding"')
            WHEN 'object' THEN json_array(json_extract(raw_json,
              '$."abstracts-retrieval-response".item."xocs:meta"."xocs:funding-list"."xocs:funding"'))
            ELSE json_array()
          END
        ) AS f
        UNION ALL
        SELECT json_extract(f2.value, '$."xocs:funding-agency-matched-string"')
        FROM json_each(
          CASE json_type(json_extract(raw_json,
            '$."abstracts-retrieval-response"."xocs:meta"."xocs:funding-list"."xocs:funding"'))
            WHEN 'array'  THEN json_extract(raw_json,
              '$."abstracts-retrieval-response"."xocs:meta"."xocs:funding-list"."xocs:funding"')
            WHEN 'object' THEN json_array(json_extract(raw_json,
              '$."abstracts-retrieval-response"."xocs:meta"."xocs:funding-list"."xocs:funding"'))
            ELSE json_array()
          END
        ) AS f2
        UNION ALL
        SELECT json_extract(f3.value, '$."xocs:funding-agency-matched-string"')
        FROM json_each(
          CASE json_type(json_extract(raw_json,
            '$."abstracts-retrieval-response"."xocs:funding-list"."xocs:funding"'))
            WHEN 'array'  THEN json_extract(raw_json,
              '$."abstracts-retrieval-response"."xocs:funding-list"."xocs:funding"')
            WHEN 'object' THEN json_array(json_extract(raw_json,
              '$."abstracts-retrieval-response"."xocs:funding-list"."xocs:funding"'))
            ELSE json_array()
          END
        ) AS f3
        UNION ALL
        SELECT json_extract(f4.value, '$."xocs:funding-agency-matched-string"')
        FROM json_each(
          CASE json_type(json_extract(raw_json,
            '$."abstracts-retrieval-response".coredata."xocs:funding-list"."xocs:funding"'))
            WHEN 'array'  THEN json_extract(raw_json,
              '$."abstracts-retrieval-response".coredata."xocs:funding-list"."xocs:funding"')
            WHEN 'object' THEN json_array(json_extract(raw_json,
              '$."abstracts-retrieval-response".coredata."xocs:funding-list"."xocs:funding"'))
            ELSE json_array()
          END
        ) AS f4
      ) AS f_src
      WHERE f_src.name IS NOT NULL AND TRIM(f_src.name) <> ''
    ), json_array()) AS funding,

    /* ref_list: JSON array of ref-titletext */
    (
      SELECT COALESCE(json_group_array(title), json_array())
      FROM (
        SELECT
          NULLIF(TRIM(
            COALESCE(
              json_extract(r.value,'$."ref-info"."ref-title"."ref-titletext"'),
              json_extract(r.value,'$."ref-info"."ref-titletext"'),
              json_extract(r.value,'$."ref-title"."ref-titletext"'),
              json_extract(r.value,'$."ref-titletext"')
            )
          ), '') AS title
        FROM json_each(
          CASE json_type(json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.tail.bibliography.reference'))
            WHEN 'array'  THEN json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.tail.bibliography.reference')
            WHEN 'object' THEN json_array(json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.tail.bibliography.reference'))
            ELSE json_array()
          END
        ) AS r
      )
      WHERE title IS NOT NULL
    ) AS ref_list

  FROM papers_raw
)
SELECT
  base.*,
  COALESCE(json_array_length(base.funding), 0) AS funding_count,
  COALESCE(json_array_length(base.authors), 0) AS authors_count
FROM base
ORDER BY year, file_id;
"""

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def parse_json_list(s):
    if pd.isna(s):
        return []
    if isinstance(s, list):
        return s
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else []
    except Exception:
        return []


def categories_to_subjects(cat_str):
    subjects = []
    for item in parse_json_list(cat_str):
        if isinstance(item, dict) and item:
            subjects.extend(list(item.keys()))
    return subjects


def build_keyword_network(keyword_lists):
    freq = Counter()
    pair_counts = Counter()
    for kws in keyword_lists:
        if not isinstance(kws, list):
            continue
        clean = sorted(set(str(k).strip() for k in kws if str(k).strip()))
        freq.update(clean)
        for i in range(len(clean)):
            for j in range(i + 1, len(clean)):
                pair_counts[(clean[i], clean[j])] += 1
    return freq, pair_counts


def plot_top_keyword_network(freq, pair_counts, top_n=25, min_coocc=2):
    G = nx.Graph()
    top_nodes = [kw for kw, _ in freq.most_common(top_n)]
    for kw in top_nodes:
        G.add_node(kw, size=freq[kw])

    for (a, b), w in pair_counts.items():
        if a in top_nodes and b in top_nodes and w >= min_coocc:
            G.add_edge(a, b, weight=w)

    if len(G.nodes) == 0:
        return go.Figure()

    pos = nx.spring_layout(G, k=0.6, iterations=60, seed=42)

    edge_x, edge_y = [], []
    for a, b in G.edges():
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="rgba(200,200,200,0.6)"),
        hoverinfo="none",
    )

    node_x, node_y, hovertext, sizes = [], [], [], []
    labels = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        labels.append(node)
        hovertext.append(f"{node} (freq={freq[node]})")
        sizes.append(12 + 5 * np.log1p(freq[node]))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=labels,
        textposition="top center",
        hovertext=hovertext,
        hoverinfo="text",
        marker=dict(
            size=sizes,
            color=np.linspace(0, 1, len(labels)),
            colorscale="Viridis",
            showscale=False,
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Network Graph of Top Keywords",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plot_query_keyword_network(freq, pair_counts, query, top_n=30):
    query = query.strip()
    if not query:
        return None

    neighbors = []
    for (a, b), w in pair_counts.items():
        if a == query:
            neighbors.append((b, w))
        elif b == query:
            neighbors.append((a, w))

    if not neighbors:
        return None

    neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)[:top_n]

    G = nx.Graph()
    G.add_node(query, size=freq.get(query, 1))
    for other, w in neighbors:
        G.add_node(other, size=freq.get(other, 1))
        G.add_edge(query, other, weight=w)

    pos = nx.spring_layout(G, k=0.9, iterations=60, seed=42)

    edge_x, edge_y = [], []
    for a, b in G.edges():
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.7, color="rgba(220,220,220,0.7)"),
        hoverinfo="none",
    )

    node_x, node_y, hovertext, sizes, colors = [], [], [], [], []
    labels = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        labels.append(node)
        hovertext.append(f"{node} (freq={freq.get(node, 0)})")
        sizes.append(12 + 5 * np.log1p(freq.get(node, 1)))
        colors.append("red" if node == query else "dodgerblue")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=labels,
        textposition="top center",
        hovertext=hovertext,
        hoverinfo="text",
        marker=dict(size=sizes, color=colors),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"Network Graph of Closest Keywords to '{query}'",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ------------------------------------------------------------
# Load data with loading spinner
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_df(db_path="scopus.db"):
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(SQL, con)
    finally:
        con.close()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    for col in ["refcount", "citedbycount", "funding_count", "authors_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


with st.spinner("Loading Scopus data..."):
    df = load_df()

# ------------------------------------------------------------
# Pre-processing
# ------------------------------------------------------------
df = df.copy()
df["pub_date_dt"] = pd.to_datetime(
    df["publication_date"], format="%d/%m/%Y", errors="coerce"
)
df["_keywords_list"] = df["keywords"].apply(parse_json_list)
df["_subjects_list"] = df["categories"].apply(categories_to_subjects)
df["_funding_list"] = df["funding"].apply(parse_json_list)
df["_papers"] = 1

# ------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------
st.sidebar.header("Filters")

years = sorted(df["year"].dropna().unique().astype(int))
year_sel = st.sidebar.multiselect("Year", years, default=years)

mask = pd.Series(True, index=df.index)

if year_sel:
    mask &= df["year"].isin(year_sel)

dff = df.loc[mask].copy()

st.title("Scopus Literature Dashboard")

if dff.empty:
    st.warning("No rows match current filters. Try selecting different years or query.")
    st.stop()

# numeric helpers
dff["refcount_num"] = pd.to_numeric(dff["refcount"], errors="coerce")
dff["citedby_num"] = pd.to_numeric(dff["citedbycount"], errors="coerce")
dff["funding_count"] = pd.to_numeric(dff["funding_count"], errors="coerce")
dff["authors_count"] = pd.to_numeric(dff["authors_count"], errors="coerce")

# ------------------------------------------------------------
# KPI cards
# ------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Number of papers", len(dff))
with c2:
    st.metric("Year span", f"{int(dff['year'].min())}–{int(dff['year'].max())}")
with c3:
    st.metric("Total citations", int(dff["citedby_num"].sum(skipna=True)))
with c4:
    st.metric("Total authors", int(dff["authors_count"].sum(skipna=True)))

st.markdown("---")

# ------------------------------------------------------------
# Literature count (with in-chart toggle)
# ------------------------------------------------------------
st.subheader("Literature Count over Time")

mode = st.radio(
    "Display mode", ["Yearly", "Cumulative"], horizontal=True, key="lit_mode"
)

papers_per_year = (
    dff.groupby("year")["_papers"]
    .sum()
    .reset_index(name="paper_count")
    .sort_values("year")
)

if mode == "Cumulative":
    papers_per_year["paper_count"] = papers_per_year["paper_count"].cumsum()
    title_lit = "Cumulative Literature Count per Year"
else:
    title_lit = "Literature Count per Year"

fig_lit = px.line(
    papers_per_year,
    x="year",
    y="paper_count",
    markers=True,
    title=title_lit,
    color_discrete_sequence=["#7695FF"],
)
fig_lit.update_layout(xaxis=dict(dtick=1))
fig_lit.update_traces(line=dict(width=3))
fig_lit.update_yaxes(rangemode="tozero")
st.plotly_chart(fig_lit, use_container_width=True)

st.markdown("---")

# ------------------------------------------------------------
# Normalized references per year
# ------------------------------------------------------------
st.subheader("Normalized Number of References per Year")

denom = dff["authors_count"].fillna(0) + dff["funding_count"].fillna(0)
denom = denom.replace(0, np.nan)
dff["ref_norm"] = dff["refcount_num"] / denom

norm_year = (
    dff.groupby("year")["ref_norm"]
    .mean()
    .reset_index(name="avg_normalized_refs")
    .sort_values("year")
)

fig_norm = px.line(
    norm_year,
    x="year",
    y="avg_normalized_refs",
    markers=True,
    title="Average Normalized References per Year",
    color_discrete_sequence=["#7695FF"],
)
fig_norm.update_layout(xaxis=dict(dtick=1))
fig_norm.update_traces(line=dict(width=3))
fig_norm.update_yaxes(rangemode="tozero")
st.plotly_chart(fig_norm, use_container_width=True)
st.caption(
    "Normalization formula: references divided by (authors + funders) per paper, then averaged per year."
)

st.markdown("---")

# ------------------------------------------------------------
# Subject Area metrics (Keywords / Papers / References)
# ------------------------------------------------------------
st.subheader("Subject Area Metrics")

sub_df = dff[["year", "_subjects_list", "_keywords_list", "refcount_num"]].explode(
    "_subjects_list"
)
sub_df = sub_df.rename(columns={"_subjects_list": "subject"})
sub_df = sub_df[sub_df["subject"].notna() & (sub_df["subject"] != "")]
sub_df["paper"] = 1
sub_df["keyword_count"] = sub_df["_keywords_list"].apply(
    lambda lst: len(lst) if isinstance(lst, list) else 0
)

metric_choice = st.selectbox(
    "Choose data type",
    ["Keywords per Subject Area", "Paper count per Subject Area", "Reference count per Subject Area"],
)

if metric_choice == "Keywords per Subject Area":
    agg = (
        sub_df.groupby("subject")["keyword_count"]
        .sum()
        .reset_index(name="value")
    )
    y_label = "Total keywords"
elif metric_choice == "Paper count per Subject Area":
    agg = sub_df.groupby("subject")["paper"].sum().reset_index(name="value")
    y_label = "Number of papers"
else:
    agg = (
        sub_df.groupby("subject")["refcount_num"]
        .sum()
        .reset_index(name="value")
    )
    y_label = "Total references"

agg = agg.sort_values("value", ascending=False).head(20)

fig_subj = px.bar(
    agg,
    x="subject",
    y="value",
    title=f"{metric_choice}",
    color_discrete_sequence=["#FF9874"],
)
fig_subj.update_layout(
    xaxis_tickangle=-45,
    yaxis_title=y_label,
    showlegend=False,
)
st.plotly_chart(fig_subj, use_container_width=True)

st.markdown("---")

# ------------------------------------------------------------
# Funded Papers by Different Affiliations (funding agencies)
# ------------------------------------------------------------
st.subheader("Funded Papers by Different Affiliations (Funding Agencies)")

funded = dff[dff["funding_count"] > 0].copy()
funded_ex = funded.explode("_funding_list")
funded_ex = funded_ex[
    funded_ex["_funding_list"].notna() & (funded_ex["_funding_list"] != "")
]

if funded_ex.empty:
    st.info("No funded papers found for current filters.")
else:
    top_funding = (
        funded_ex.groupby("_funding_list")["_papers"]
        .sum()
        .reset_index(name="funded_papers")
        .sort_values("funded_papers", ascending=False)
        .head(20)
    )
    fig_fund_aff = px.bar(
        top_funding,
        x="_funding_list",
        y="funded_papers",
        title="Funded Papers by Funding Agencies (Top 20)",
        color_discrete_sequence=["#FF9874"],
    )
    fig_fund_aff.update_layout(
        xaxis_tickangle=-45,
        yaxis_title="Number of funded papers",
        showlegend=False,
    )
    st.plotly_chart(fig_fund_aff, use_container_width=True)

st.markdown("---")

# ------------------------------------------------------------
# Funded Subject Areas by Chulalongkorn University
# ------------------------------------------------------------
st.subheader("Funded Subject Areas by Chulalongkorn University")

mask_chula = dff["_funding_list"].apply(
    lambda lst: any("chulalongkorn" in str(x).lower() for x in lst)
)
chula_df = dff[mask_chula].copy()

if chula_df.empty:
    st.info("No funded papers with Chulalongkorn University in funding agencies for current filters.")
else:
    chula_sub = chula_df[["_subjects_list"]].explode("_subjects_list")
    chula_sub = chula_sub.rename(columns={"_subjects_list": "subject"})
    chula_sub = chula_sub[chula_sub["subject"].notna() & (chula_sub["subject"] != "")]
    chula_sub["paper"] = 1

    chula_agg = (
        chula_sub.groupby("subject")["paper"]
        .sum()
        .reset_index(name="funded_papers")
        .sort_values("funded_papers", ascending=False)
        .head(20)
    )

    fig_chula = px.bar(
        chula_agg,
        x="subject",
        y="funded_papers",
        title="Funded Subject Areas (Chulalongkorn University)",
        color_discrete_sequence=["#FF9874"],
    )
    fig_chula.update_layout(
        xaxis_tickangle=-45,
        yaxis_title="Number of funded papers",
        showlegend=False,
    )
    st.plotly_chart(fig_chula, use_container_width=True)

st.markdown("---")

# ------------------------------------------------------------
# Cited-by count of each paper (top 30) – horizontal bars
# ------------------------------------------------------------
st.subheader("Top Cited Papers")

top_n_cited = st.slider("Number of top papers", 5, 50, 30, key="top_cited_n")

top_cited = (
    dff.sort_values("citedby_num", ascending=False)
      .head(top_n_cited)
      .copy()
)

top_cited["title_short"] = top_cited["citation_title"].fillna("").str.slice(0, 60)
top_cited["title_short"] = np.where(
    top_cited["title_short"].str.len() == 60,
    top_cited["title_short"] + "…",
    top_cited["title_short"],
)

fig_cited = px.bar(
    top_cited,
    x="citedby_num",
    y="title_short",
    orientation="h",
    title=f"Top {top_n_cited} Papers by Cited-by Count",
    color_discrete_sequence=["#FF9874"],
)
fig_cited.update_layout(
    xaxis_title="Cited-by count",
    yaxis_title="Paper title (truncated)",
    showlegend=False,
    yaxis=dict(categoryorder="total ascending"),
)
st.plotly_chart(fig_cited, use_container_width=True)

st.markdown("---")

# ------------------------------------------------------------
# Line chart: most popular keywords for recent years
# ------------------------------------------------------------
if dff["year"].notna().sum() == 0:
    st.subheader("Most Popular Keywords (by Year Range)")
    st.info("No year data available.")
else:
    years_avail = sorted(set(int(y) for y in dff["year"].dropna()))
    # use up to the latest 6 years that remain after filtering
    recent_years = years_avail[-6:] if len(years_avail) > 6 else years_avail
    n_years = len(recent_years)
    start_y, end_y = recent_years[0], recent_years[-1]

    st.subheader(f"Most Popular Keywords in the Last {n_years} Years ({start_y}–{end_y})")

    recent = dff[dff["year"].isin(recent_years)].copy()

    if recent.empty:
        st.info("No rows in the selected year range for current filters.")
    else:
        kw_recent = recent[["year", "pub_date_dt", "_keywords_list"]].explode("_keywords_list")
        kw_recent = kw_recent.rename(columns={"_keywords_list": "keyword"})
        kw_recent = kw_recent[kw_recent["keyword"].notna() & (kw_recent["keyword"] != "")]

        if kw_recent.empty:
            st.info("No keywords in the selected year range for current filters.")
        else:
            # -------- choose top K keywords (used everywhere below) --------
            num_kw = st.slider(
                "Number of top keywords to plot",
                3,
                20,
                8,
                key="kw_line_n",
            )

            top_keywords = kw_recent["keyword"].value_counts().head(num_kw).index.tolist()
            kw_recent = kw_recent[kw_recent["keyword"].isin(top_keywords)]

            kw_mode = st.radio(
                "Display mode",
                ["Yearly", "Cumulative", "Per month"],
                horizontal=True,
                key="kw_mode",
            )

            # ---------- aggregate counts depending on mode ----------
            if kw_mode == "Per month":
                month_df = kw_recent.dropna(subset=["pub_date_dt"]).copy()
                if month_df.empty:
                    st.info("No publication dates available for monthly view.")
                    st.markdown("---")
                else:
                    month_df["month"] = (
                        month_df["pub_date_dt"].dt.to_period("M").dt.to_timestamp()
                    )
                    counts = (
                        month_df.groupby(["month", "keyword"])
                        .size()
                        .reset_index(name="value")
                        .sort_values(["keyword", "month"])
                    )
                    x_col = "month"
                    title_kw = f"Top Keywords from {start_y} to {end_y} (Per Month)"
            else:
                counts = (
                    kw_recent.groupby(["year", "keyword"])
                    .size()
                    .reset_index(name="count")
                    .sort_values(["keyword", "year"])
                )

                if counts.empty:
                    st.info("No keyword counts for yearly view.")
                    st.markdown("---")
                else:
                    if kw_mode == "Cumulative":
                        counts["value"] = counts.groupby("keyword")["count"].cumsum()
                        title_kw = f"Top Keywords from {start_y} to {end_y} (Cumulative Yearly)"
                    else:
                        counts["value"] = counts["count"]
                        title_kw = f"Top Keywords from {start_y} to {end_y} (Yearly)"

                    x_col = "year"

            if "counts" in locals() and not counts.empty:
                # ---------- static line chart (main one) ----------
                fig_kw_line = px.line(
                    counts,
                    x=x_col,
                    y="value",
                    color="keyword",
                    markers=True,
                    title=title_kw,
                )
                fig_kw_line.update_traces(line=dict(width=3))
                if x_col == "year":
                    fig_kw_line.update_layout(xaxis=dict(dtick=1))

                st.plotly_chart(fig_kw_line, use_container_width=True)

                # ---------- grab color map from static fig ----------
                keyword_colors = {}
                for tr in fig_kw_line.data:
                    name = getattr(tr, "name", None)
                    if not name:
                        continue
                    col = None
                    # prefer line color
                    if hasattr(tr, "line") and getattr(tr.line, "color", None) is not None:
                        col = tr.line.color
                    # fallback to marker color
                    elif hasattr(tr, "marker") and getattr(tr.marker, "color", None) is not None:
                        col = tr.marker.color
                    if col is not None:
                        keyword_colors[name] = col

                # ---------- smooth animated version (same K keywords + colors) ----------
                if kw_mode == "Per month":
                    # make a full grid of all months × top_keywords (fill missing with 0)
                    all_periods = sorted(counts["month"].unique())
                    full_idx = pd.MultiIndex.from_product(
                        [all_periods, top_keywords],
                        names=["month", "keyword"],
                    )
                    full_df = full_idx.to_frame(index=False)
                    full_df = full_df.merge(
                        counts[["month", "keyword", "value"]],
                        on=["month", "keyword"],
                        how="left",
                    )
                    full_df["value"] = full_df["value"].fillna(0)

                    # build frames progressively over months
                    anim_frames = []
                    for frame_idx, m in enumerate(all_periods):
                        partial = full_df[full_df["month"] <= m].copy()
                        partial["frame"] = frame_idx
                        anim_frames.append(partial)

                    anim_df = pd.concat(anim_frames, ignore_index=True)

                    fig_kw_anim = px.line(
                        anim_df,
                        x="month",
                        y="value",
                        color="keyword",
                        animation_frame="frame",
                        animation_group="keyword",
                        range_y=[0, max(1, full_df["value"].max() * 1.05)],
                        title=f"Top Keywords from {start_y} to {end_y} (Per Month, Animated)",
                        color_discrete_map=keyword_colors,
                    )
                    fig_kw_anim.update_traces(line=dict(width=3))

                else:
                    # yearly / cumulative animation
                    all_years = sorted(recent_years)
                    full_idx = pd.MultiIndex.from_product(
                        [all_years, top_keywords],
                        names=["year", "keyword"],
                    )
                    full_df = full_idx.to_frame(index=False)
                    full_df = full_df.merge(
                        counts[["year", "keyword", "value"]],
                        on=["year", "keyword"],
                        how="left",
                    )
                    full_df["value"] = full_df["value"].fillna(0)

                    anim_frames = []
                    for frame_idx, yr in enumerate(all_years):
                        partial = full_df[full_df["year"] <= yr].copy()
                        partial["frame"] = frame_idx
                        anim_frames.append(partial)

                    anim_df = pd.concat(anim_frames, ignore_index=True)

                    fig_kw_anim = px.line(
                        anim_df,
                        x="year",
                        y="value",
                        color="keyword",
                        animation_frame="frame",
                        animation_group="keyword",
                        range_y=[0, max(1, full_df["value"].max() * 1.05)],
                        title=f"Top Keywords from {start_y} to {end_y} ({kw_mode}) (Animated)",
                        color_discrete_map=keyword_colors,
                    )
                    fig_kw_anim.update_traces(line=dict(width=3))
                    fig_kw_anim.update_layout(xaxis=dict(dtick=1))

                # smooth / slow-ish animation
                frame_ms = 500        # time each frame is shown (ms)
                transition_ms = 1000  # movement between frames (ms)

                if fig_kw_anim.layout.updatemenus:
                    for um in fig_kw_anim.layout.updatemenus:
                        for btn in um.buttons:
                            if isinstance(btn.args, (list, tuple)) and len(btn.args) > 1:
                                if isinstance(btn.args[1], dict) and "frame" in btn.args[1]:
                                    btn.args[1]["frame"]["duration"] = frame_ms
                                    btn.args[1]["frame"]["redraw"] = False
                                    btn.args[1]["transition"]["duration"] = transition_ms
                                    btn.args[1]["transition"]["easing"] = "linear"

                st.plotly_chart(fig_kw_anim, use_container_width=True)

st.markdown("---")

# ------------------------------------------------------------
# Keyword network graphs
# ------------------------------------------------------------

freq, pair_counts = build_keyword_network(dff["_keywords_list"])

if not freq:
    st.info("No keyword information available for current filters.")
else:
    # Network of top keywords
    st.markdown("### Network Graph of Top Keywords")
    top_n_kw = st.slider("Number of keywords", 10, 50, 25, key="top_kw_n")
    min_edge = st.slider("Minimum co-occurrence for edge", 1, 10, 2, key="top_kw_edge")
    fig_net_top = plot_top_keyword_network(
        freq, pair_counts, top_n=top_n_kw, min_coocc=min_edge
    )
    st.plotly_chart(fig_net_top, use_container_width=True)

    st.markdown("---")

    # Network for an input keyword
    st.markdown("### Network Graph Keywords Relationships")

    all_kw_sorted = sorted(freq.keys(), key=lambda k: (-freq[k], k.lower()))
    default_kw = all_kw_sorted[0] if all_kw_sorted else ""
    query_kw = st.text_input("Enter keyword…", value=default_kw)
    top_related = st.slider("Set number of related tags", 5, 50, 30, key="rel_kw_n")

    if query_kw:
        fig_query = plot_query_keyword_network(
            freq, pair_counts, query_kw, top_n=top_related
        )
        if fig_query is None:
            st.info("Keyword not found or has no co-occurring terms in current filter.")
        else:
            st.plotly_chart(fig_query, use_container_width=True)

st.markdown("---")

# ------------------------------------------------------------
# Data preview
# ------------------------------------------------------------
st.subheader("Filtered rows (preview)")

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
    "authors",
    "authors_count",
    "categories",
    "keywords",
    "funding",
    "funding_count",
    "ref_list",
]
present_cols = [c for c in show_cols if c in dff.columns]

st.dataframe(dff[present_cols].head(200), use_container_width=True)

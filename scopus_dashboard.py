import sqlite3
import json
from collections import Counter
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

import matplotlib.pyplot as plt
from aimodel2 import run_models_page
from eda import run_eda_page

st.set_page_config(page_title="Scopus Dashboard", layout="wide")

st.sidebar.title("Navigation")
page_choice = st.sidebar.radio("Go to", ("Dashboard", "EDA", "Models"), index=0)

if page_choice == "EDA":
    run_eda_page()
    st.stop()

if page_choice == "Models":
    run_models_page()
    st.stop()

# Helpers
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
    for edge in G.edges():
        a, b = edge[:2]
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
    for edge in G.edges():
        a, b = edge[:2]
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


# Load data
@st.cache_data(show_spinner="Loading Scopus data …")
def load_data(db_path="scopus.db"):
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query("SELECT * FROM research_papers", con)

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    for col in ["refcount", "citedbycount", "funding_count", "authors_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


with st.spinner("Loading Scopus data..."):
    df = load_data()

# Pre-processing
df = df.copy()
df["pub_date_dt"] = pd.to_datetime(
    df["publication_date"], format="%d/%m/%Y", errors="coerce"
)
df["_keywords_list"] = df["keywords"].apply(parse_json_list)
df["_subjects_list"] = df["categories"].apply(
    lambda cat: [
        subject
        for item in parse_json_list(cat)
        if isinstance(item, dict) and item
        for subject in item.keys()
    ]
)
df["_funding_list"] = df["funding"].apply(parse_json_list)
df["_papers"] = 1

# Sidebar filters
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

# KPI cards
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

# Literature count (with in-chart toggle)
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

# Normalized references per year
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

# Subject Area metrics (Keywords / Papers / References)
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

# Funded Papers by Different Affiliations (funding agencies)
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

# Funded Subject Areas by Chulalongkorn University
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

# Cited-by count of each paper (top 30) – horizontal bars
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

# Line chart: most popular keywords for recent years
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

            counts: Optional[pd.DataFrame] = None
            x_col: Optional[str] = None
            title_kw = ""

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
                year_counts = (
                    kw_recent.groupby(["year", "keyword"])
                    .size()
                    .reset_index(name="count")
                    .sort_values(["keyword", "year"])
                )

                if year_counts.empty:
                    st.info("No keyword counts for yearly view.")
                    st.markdown("---")
                else:
                    if kw_mode == "Cumulative":
                        year_counts["value"] = year_counts.groupby("keyword")["count"].cumsum()
                        title_kw = f"Top Keywords from {start_y} to {end_y} (Cumulative Yearly)"
                    else:
                        year_counts["value"] = year_counts["count"]
                        title_kw = f"Top Keywords from {start_y} to {end_y} (Yearly)"

                    counts = year_counts
                    x_col = "year"

            if counts is not None and x_col is not None and not counts.empty:
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

                keyword_colors: Dict[str, Any] = {}
                for tr in fig_kw_line.data:
                    trace: Any = tr
                    name = getattr(trace, "name", None)
                    if not name:
                        continue
                    color_val = None
                    if hasattr(trace, "line") and getattr(trace.line, "color", None) is not None:
                        color_val = trace.line.color
                    elif hasattr(trace, "marker") and getattr(trace.marker, "color", None) is not None:
                        color_val = trace.marker.color
                    if color_val is not None:
                        keyword_colors[name] = color_val

                fig_kw_anim: Optional[go.Figure] = None

                if kw_mode == "Per month":
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

                if fig_kw_anim is not None:
                    updatemenus = getattr(fig_kw_anim.layout, "updatemenus", None)
                    if updatemenus:
                        frame_ms = 500
                        transition_ms = 1000
                        for um in updatemenus:
                            for btn in um.buttons:
                                args = getattr(btn, "args", None)
                                if isinstance(args, (list, tuple)) and len(args) > 1:
                                    config = args[1]
                                    if isinstance(config, dict) and "frame" in config:
                                        config["frame"]["duration"] = frame_ms
                                        config["frame"]["redraw"] = False
                                        config["transition"]["duration"] = transition_ms
                                        config["transition"]["easing"] = "linear"

                    st.plotly_chart(fig_kw_anim, use_container_width=True)

st.markdown("---")

# Keyword network graphs

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

# Data preview
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

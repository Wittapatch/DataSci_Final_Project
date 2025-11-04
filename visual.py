import streamlit as st

import sqlite3, pandas as pd

st.write("hello")

conn = sqlite3.connect("scopus.db")

SQL = """
SELECT
  file_id,
  year,

  -- existing fields you already had
  json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head."citation-title"') AS citation_title,
  json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head.abstracts')        AS abstracts,
  json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head.source.publisher.publishername') AS publishername,
  json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head.source.sourcetitle')             AS sourcetitle,
  json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head.source.publicationdate."date-text"."$"') AS date_text,

  /* ce:doi: direct path -> else first ce:doi found anywhere under AR */
  COALESCE(
    json_extract(raw_json,'$."abstracts-retrieval-response".item."item-info"."itemidlist"."ce:doi"'),
    (SELECT t.value
     FROM json_tree(raw_json, '$."abstracts-retrieval-response"') AS t
     WHERE t.key = 'ce:doi'
     LIMIT 1)
  ) AS "ce:doi",

  json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.tail.bibliography."@refcount"') AS refcount,

  /* -------- NEW: citedbycount (integer; NULL if absent) -------- */
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

  /* ---------- authors_deg_name_json (as before) ---------- */
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

  /* ---------- categories (subject → abbrev) ---------- */
  (
    SELECT json_group_array(json_object(subject, abbrev))
    FROM (
      SELECT DISTINCT
        json_extract(sa.value,'$."$"')        AS subject,
        json_extract(sa.value,'$."@abbrev"')  AS abbrev
      FROM (
        -- primary: top-level subject-areas (matches your samples)
        SELECT * FROM json_each(
          CASE json_type(json_extract(raw_json,'$."abstracts-retrieval-response"."subject-areas"."subject-area"'))
            WHEN 'array'  THEN json_extract(raw_json,'$."abstracts-retrieval-response"."subject-areas"."subject-area"')
            WHEN 'object' THEN json_array(json_extract(raw_json,'$."abstracts-retrieval-response"."subject-areas"."subject-area"'))
            ELSE json_array()
          END
        )
        UNION ALL
        -- fallback A: item.coredata.subject-areas
        SELECT * FROM json_each(
          CASE json_type(json_extract(raw_json,'$."abstracts-retrieval-response".item.coredata."subject-areas"."subject-area"'))
            WHEN 'array'  THEN json_extract(raw_json,'$."abstracts-retrieval-response".item.coredata."subject-areas"."subject-area"')
            WHEN 'object' THEN json_array(json_extract(raw_json,'$."abstracts-retrieval-response".item.coredata."subject-areas"."subject-area"'))
            ELSE json_array()
          END
        )
        UNION ALL
        -- fallback B: coredata.subject-areas at top-level
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
  ) AS categories

FROM papers_raw
ORDER BY year, file_id;
"""

df = pd.read_sql_query(SQL, conn)
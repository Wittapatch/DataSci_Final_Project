import sqlite3, pandas as pd

conn = sqlite3.connect("scopus.db")

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

    /* authors = JSON array of "<given> <surname>" (no degrees included) */
    (
      SELECT json_group_array(name_str)
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
    ) AS authors_deg_name_json,

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

    /* funding (array of names) */
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

    /* ---------------- ref_list: JSON array of ref-titletext ---------------- */
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
  COALESCE(json_array_length(base.funding), 0) AS funding_count
FROM base
ORDER BY year, file_id;
"""

df = pd.read_sql_query(SQL, conn)
df

# Chula-Scopus-data

## Before Doing anything
### First, you have to create a folder name "...." and then download the zip file from the google drive here: https://drive.google.com/drive/folders/14QccbuPKigSKFnuN1WYKQU_53hmB7r4w?usp=sharing

### Second, you put the downloaded zip file and test.ipynb into the folder

### Then follow the steps down below:

## Run this code in dataloading.ipynb
```python 
import json
import shutil
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd

DB_PATH = Path("scopus.db").resolve()
ZIP_PATH = Path("ScopusData2018-2023.zip").resolve()
TABLE_NAME = "research_papers"
EXPECTED_ROWS = 20186
YEAR_RANGES = {
    2018: (201800000, 201802761),
    2019: (201900000, 201903081),
    2020: (202000000, 202003392),
    2021: (202100000, 202103814),
    2022: (202200000, 202204243),
    2023: (202300000, 202302889),
}

SQL_PROJECTION = f"""
WITH base AS (
  SELECT
    file_id,
    year,
    json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head."citation-title"') AS citation_title,
    json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head.abstracts')        AS abstracts,
    json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head.source.publisher.publishername') AS publishername,
    json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head.source.sourcetitle')             AS sourcetitle,
    CASE
      WHEN json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head.source.publicationdate.day') IS NOT NULL
       AND json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head.source.publicationdate.month') IS NOT NULL
       AND json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head.source.publicationdate.year') IS NOT NULL
      THEN printf('%02d/%02d/%04d',
                  CAST(json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head.source.publicationdate.day')   AS INTEGER),
                  CAST(json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head.source.publicationdate.month') AS INTEGER),
                  CAST(json_extract(raw_json,'$."abstracts-retrieval-response".item.bibrecord.head.source.publicationdate.year')  AS INTEGER))
      ELSE NULL
    END AS publication_date,
    COALESCE(
      json_extract(raw_json,'$."abstracts-retrieval-response".item."item-info"."itemidlist"."ce:doi"'),
      (SELECT t.value
         FROM json_tree(raw_json, '$."abstracts-retrieval-response"') AS t
         WHERE t.key = 'ce:doi'
         LIMIT 1)
    ) AS document_classification_codes,
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
    (
      SELECT json_group_array(name_str)
      FROM (
        SELECT
          TRIM(
            TRIM(COALESCE(json_extract(a.value,'$."preferred-name"."ce:given-name"'),
                          json_extract(a.value,'$."ce:given-name"'), '')) || ' ' ||
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
    ) AS allauthors_name,
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
    COALESCE((
      SELECT json_group_array(f_src.name)
      FROM (
        SELECT json_extract(f.value, '$."xocs:funding-agency-matched-string"') AS name
        FROM json_each(
          CASE json_type(json_extract(raw_json,'$."abstracts-retrieval-response".item."xocs:meta"."xocs:funding-list"."xocs:funding"'))
            WHEN 'array'  THEN json_extract(raw_json,'$."abstracts-retrieval-response".item."xocs:meta"."xocs:funding-list"."xocs:funding"')
            WHEN 'object' THEN json_array(json_extract(raw_json,'$."abstracts-retrieval-response".item."xocs:meta"."xocs:funding-list"."xocs:funding"'))
            ELSE json_array()
          END
        ) AS f
        UNION ALL
        SELECT json_extract(f2.value, '$."xocs:funding-agency-matched-string"')
        FROM json_each(
          CASE json_type(json_extract(raw_json,'$."abstracts-retrieval-response"."xocs:meta"."xocs:funding-list"."xocs:funding"'))
            WHEN 'array'  THEN json_extract(raw_json,'$."abstracts-retrieval-response"."xocs:meta"."xocs:funding-list"."xocs:funding"')
            WHEN 'object' THEN json_array(json_extract(raw_json,'$."abstracts-retrieval-response"."xocs:meta"."xocs:funding-list"."xocs:funding"'))
            ELSE json_array()
          END
        ) AS f2
        UNION ALL
        SELECT json_extract(f3.value, '$."xocs:funding-agency-matched-string"')
        FROM json_each(
          CASE json_type(json_extract(raw_json,'$."abstracts-retrieval-response"."xocs:funding-list"."xocs:funding"'))
            WHEN 'array'  THEN json_extract(raw_json,'$."abstracts-retrieval-response"."xocs:funding-list"."xocs:funding"')
            WHEN 'object' THEN json_array(json_extract(raw_json,'$."abstracts-retrieval-response"."xocs:funding-list"."xocs:funding"'))
            ELSE json_array()
          END
        ) AS f3
        UNION ALL
        SELECT json_extract(f4.value, '$."xocs:funding-agency-matched-string"')
        FROM json_each(
          CASE json_type(json_extract(raw_json,'$."abstracts-retrieval-response".coredata."xocs:funding-list"."xocs:funding"'))
            WHEN 'array'  THEN json_extract(raw_json,'$."abstracts-retrieval-response".coredata."xocs:funding-list"."xocs:funding"')
            WHEN 'object' THEN json_array(json_extract(raw_json,'$."abstracts-retrieval-response".coredata."xocs:funding-list"."xocs:funding"'))
            ELSE json_array()
          END
        ) AS f4
      ) AS f_src
      WHERE f_src.name IS NOT NULL AND TRIM(f_src.name) <> ''
    ), json_array()) AS funding
  FROM {TABLE_NAME}
)
SELECT
  base.*,
  base.allauthors_name AS authors_deg_name_json,
  COALESCE(json_array_length(base.allauthors_name), 0) AS allauthors_count,
  COALESCE(json_array_length(base.allauthors_name), 0) AS authors_count,
  COALESCE(json_array_length(base.funding), 0) AS funding_count
FROM base
ORDER BY year, file_id;
"""


def detect_db_state():
    if not DB_PATH.exists():
        return "missing"
    with sqlite3.connect(DB_PATH) as conn:
        def table_exists(name: str) -> bool:
            return conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
                (name,),
            ).fetchone() is not None

        def table_state(name: str) -> str | None:
            cols = {row[1] for row in conn.execute(f"PRAGMA table_info('{name}');")}
            if not cols:
                return None
            return "raw" if "raw_json" in cols else "processed"

        if table_exists(TABLE_NAME):
            state = table_state(TABLE_NAME)
            if state == "processed":
                row_count = conn.execute(
                    f"SELECT COUNT(*) FROM {TABLE_NAME};"
                ).fetchone()[0]
                if row_count == 0:
                    return "processed_empty"
                if row_count != EXPECTED_ROWS:
                    return "processed_incomplete"
            return state or "missing"
        return "missing"


def backup_db(tag: str = "snapshot"):
    if not DB_PATH.exists():
        return None
    backup_path = DB_PATH.with_name(f"{DB_PATH.stem}_{tag}{DB_PATH.suffix}")
    shutil.copy2(DB_PATH, backup_path)
    return backup_path


def load_raw_from_zip():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"{ZIP_PATH} not found. Place the archive next to this notebook.")
    with zipfile.ZipFile(ZIP_PATH, "r") as z, sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME};")
        conn.execute(
            f"""
            CREATE TABLE {TABLE_NAME} (
                file_id INTEGER,
                year INTEGER,
                raw_json TEXT
            )
            """
        )
        cur = conn.cursor()
        for year, (start_id, end_id) in YEAR_RANGES.items():
            print(f"Loading {year} ({start_id}-{end_id}) …")
            for file_id in range(start_id, end_id + 1):
                inner_path = f"ScopusData2018-2023/{year}/{file_id}"
                try:
                    with z.open(inner_path) as handle:
                        try:
                            obj = json.load(handle)
                        except Exception:
                            continue
                except KeyError:
                    continue
                raw_text = json.dumps(obj, ensure_ascii=False)
                cur.execute(
                    f"""INSERT INTO {TABLE_NAME} (file_id, year, raw_json) VALUES (?, ?, ?)""",
                    (file_id, year, raw_text),
                )
            conn.commit()
        print("Raw JSON extraction finished.")


def convert_raw_db_to_processed():
    with sqlite3.connect(DB_PATH) as conn:
        df_proc = pd.read_sql_query(SQL_PROJECTION, conn)
    backup_path = backup_db("_rawbackup")
    if backup_path:
        print(f"Backed up raw-format database to {backup_path}")
    with sqlite3.connect(DB_PATH) as conn:
        df_proc.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    print("Replaced research_papers with processed columns.")
    return df_proc.shape
```
```python
state = detect_db_state()

if state == "processed_empty":
    print("Detected empty processed table. Extracting everything from the archive …")
    load_raw_from_zip()
    print("Your scopus.db is ready to be use!!!!")

elif state == "processed_incomplete":
    with sqlite3.connect(DB_PATH) as conn:
        current_rows = conn.execute(
            f"SELECT COUNT(*) FROM {TABLE_NAME};"
        ).fetchone()[0]
    load_raw_from_zip()
    print("Your scopus.db is ready to be use!!!!")

elif state == "processed":
    print("Your scopus.db is ready to be use!!!!")

elif state == "raw":
    print("Detected raw-format research_papers. Converting to processed dataset")
    print("Your scopus.db is ready to be use!!!!")

elif state == "missing":
    print("scopus.db not found. Extracting everything from the archive")
    load_raw_from_zip()
    print("Your scopus.db is ready to be use!!!!")

else:
    raise RuntimeError(f"Unexpected database state: {state}")

with sqlite3.connect(DB_PATH) as conn:
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;", conn)
    row_count = pd.read_sql_query(
        f"SELECT COUNT(*) AS rows FROM {TABLE_NAME};",
        conn,
    ).loc[0, "rows"]
    preview = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME} LIMIT 5;", conn)

print("Tables currently in scopus.db:")
display(tables)
print(f"\n{TABLE_NAME} rows: {row_count:,}")
if row_count != EXPECTED_ROWS:
    print("Row count mismatch detected. Re-run the cell to trigger a rebuild, or inspect the source data.")
display(preview)
```
```python
```
```python
```


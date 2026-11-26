from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from sqlalchemy import text

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PARENT_DIR = str(BASE_DIR.parent)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from utils import get_snowflake_engine  # noqa: E402

RETENTION_DAYS = 30
FILE_PREFIX = "df_output_for_donut_"
FILE_REGEX = re.compile(r"^df_output_for_donut_(\d{4}-\d{2}-\d{2})\.parquet$")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("refresh_donut_data")


def _numeric_min_max(s: pd.Series) -> tuple[float, float]:
    nums = pd.to_numeric(s, errors="coerce").dropna()
    if nums.empty:
        return np.nan, np.nan
    return float(nums.min()), float(nums.max())


def build_donut_df(agg_df: pd.DataFrame) -> pd.DataFrame:
    df = agg_df.copy()
    df.columns = [c.upper() for c in df.columns]

    df["DATASET_NAME"] = df["DATASET_NAME"].astype(str).str.strip()
    df["CHILD_CATEGORY"] = (
        df["CHILD_CATEGORY"].fillna("unknown").astype(str).str.strip().str.lower()
    )
    df["SOURCE_RECORD_TYPE"] = (
        df["SOURCE_RECORD_TYPE"].fillna("unknown").astype(str).str.strip().str.lower()
    )
    df["SOURCE_COUNTRY"] = df["SOURCE_COUNTRY"].fillna("None").astype(str).str.strip()
    df["PARENT_CATEGORY"] = df["PARENT_CATEGORY"].fillna("Unknown").astype(str).str.strip()
    df["EXCLUSIVITY_STATUS"] = (
        df["EXCLUSIVITY_STATUS"].fillna("Unknown").astype(str).str.strip()
    )

    rows = []
    for ds, g in df.groupby("DATASET_NAME", sort=True):
        total_events = int(g["RECORD_COUNT"].sum())

        cat = g.groupby("CHILD_CATEGORY")["RECORD_COUNT"].sum().idxmax()
        rec_type = g.groupby("SOURCE_RECORD_TYPE")["RECORD_COUNT"].sum().idxmax()

        rmid_sums = (
            g.groupby("RECORD_METADATA_ID")["RECORD_COUNT"]
            .sum()
            .sort_values(ascending=False)
        )
        rmid_contrib = ", ".join(f"{k} ({int(v):,})" for k, v in rmid_sums.items())

        country_totals: dict[str, int] = {}
        for _, r in g.iterrows():
            for c in [
                x.strip() for x in str(r["SOURCE_COUNTRY"]).split(",") if x.strip()
            ]:
                c_key = c.lower()
                country_totals[c_key] = country_totals.get(c_key, 0) + int(
                    r["RECORD_COUNT"]
                )

        sorted_countries = sorted(country_totals.keys())
        source_country_list = ", ".join(sorted_countries)
        country_contrib = ", ".join(
            f"{c} ({country_totals[c]:,})" for c in sorted_countries
        )

        min_yr, _ = _numeric_min_max(g["MIN_YEAR"])
        _, max_yr = _numeric_min_max(g["MAX_YEAR"])

        rows.append({
            "dataset": ds,
            "dataset_orig": ds,
            "category": cat,
            "recordtype": rec_type,
            "events": total_events,
            "source_country_list": source_country_list,
            "country_events_contrib": country_contrib,
            "rmid_events_contrib": rmid_contrib,
            "metadataid_nunique": int(len(rmid_sums)),
            "parent_category": g.groupby("PARENT_CATEGORY")["RECORD_COUNT"].sum().idxmax(),
            "exclusivity_status": g.groupby("EXCLUSIVITY_STATUS")["RECORD_COUNT"]
            .sum()
            .idxmax(),
            "min_year": min_yr,
            "max_year": max_yr,
        })

    return pd.DataFrame(rows)


def cleanup_old_files(data_dir: Path, retention_days: int = RETENTION_DAYS) -> None:
    cutoff_date = (datetime.now(timezone.utc) - timedelta(days=retention_days)).date()
    for path in data_dir.glob(f"{FILE_PREFIX}*.parquet"):
        match = FILE_REGEX.match(path.name)
        if not match:
            continue
        file_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
        if file_date < cutoff_date:
            path.unlink(missing_ok=True)
            logger.info("Removed expired file older than %d days: %s", retention_days, path.name)


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    run_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    target_path = DATA_DIR / f"{FILE_PREFIX}{run_date_str}.parquet"
    tmp_path = DATA_DIR / f".{FILE_PREFIX}{run_date_str}.parquet.tmp"

    agg_sql = """
    SELECT
        DATASET_NAME,
        RECORD_METADATA_ID,
        CHILD_CATEGORY,
        PARENT_CATEGORY,
        SOURCE_RECORD_TYPE,
        SOLR_COLLECTION_SOURCE,
        SOURCE_COUNTRY,
        EXCLUSIVITY_STATUS,
        SUM(RECORD_COUNT) AS RECORD_COUNT,
        MIN(EVENT_YEAR)   AS MIN_YEAR,
        MAX(EVENT_YEAR)   AS MAX_YEAR
    FROM rep_recordset_metadata_by_year_and_county
    WHERE DATASET_NAME IS NOT NULL
    GROUP BY 1, 2, 3, 4, 5, 6, 7, 8
    """

    engine = None
    try:
        logger.info("Initializing Snowflake engine via %s/utils.py", PARENT_DIR)
        engine = get_snowflake_engine()
        with engine.connect() as conn:
            agg_df = pd.read_sql(text(agg_sql), conn)

        if agg_df.empty:
            raise RuntimeError("Snowflake query returned 0 rows; aborting write.")

        donut_df = build_donut_df(agg_df)
        if donut_df.empty or int(donut_df["events"].sum()) <= 0:
            raise RuntimeError("Transformed DataFrame is empty or has non-positive total events.")

        donut_df.to_parquet(tmp_path, index=False)

        verify_df = pd.read_parquet(tmp_path)
        if len(verify_df) != len(donut_df):
            raise RuntimeError("Parquet verification failed: row count mismatch.")

        tmp_path.replace(target_path)
        logger.info(
            "Wrote %s (%s rows, %s total events)",
            target_path.name,
            f"{len(donut_df):,}",
            f"{int(donut_df['events'].sum()):,}",
        )

        cleanup_old_files(DATA_DIR, RETENTION_DAYS)
        return 0

    except Exception as exc:
        logger.exception("Refresh failed; existing Parquet files remain untouched: %s", exc)
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return 1
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
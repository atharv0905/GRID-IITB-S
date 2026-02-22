#!/usr/bin/env python3
# ingest/ingest_results_folders.py
"""
Folder watcher ingest — INCREMENTAL (new revision only) + CLEAN SUMMARY LOGS

Now includes:
- NOWCAST_ROOT
- MEDIUM_ROOT
- INTRA_ROOT   (NEW)

Each root ingests CSVs into the same tables, with model_name = NOWCAST / MEDIUM / INTRA.
"""

import os
import time
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Set
from datetime import datetime, timedelta, date

import pandas as pd
from sqlalchemy import create_engine, text
from dateutil import tz

IST_TZ = tz.gettz("Asia/Kolkata")
UTC_TZ = tz.UTC

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg2://reuser:repass@localhost:5432/redb")

NOWCAST_ROOT = os.environ.get("NOWCAST_ROOT", "")
MEDIUM_ROOT  = os.environ.get("MEDIUM_ROOT", "")
INTRA_ROOT   = os.environ.get("INTRA_ROOT", "")  # NEW

POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "30"))
SINGLE_PASS  = os.environ.get("SINGLE_PASS", "0").strip() in ("1", "true", "True", "yes", "YES")

# Defaults to stop “continuous ingesting”
SKIP_UNCHANGED  = os.environ.get("SKIP_UNCHANGED", "1").strip() in ("1", "true", "True", "yes", "YES")
ONLY_NEW_RUNREV = os.environ.get("ONLY_NEW_RUNREV", "1").strip() in ("1", "true", "True", "yes", "YES")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# path -> (mtime, size, date_set, min_date, max_date)
FILE_DATE_CACHE: Dict[str, Tuple[int, int, Set[date], Optional[date], Optional[date]]] = {}

# (model, region_id, revision, run_t0_utc) -> exists?
RUNREV_EXISTS_CACHE: Dict[Tuple[str, str, str, datetime], bool] = {}


def now_ist_str() -> str:
    return datetime.now(tz=IST_TZ).strftime("%Y-%m-%d %H:%M:%S IST")


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_plant_type(site_name: str) -> str:
    s = (site_name or "").lower()
    return "WIND" if "wind" in s else "SOLAR"


def parse_timestamps_no_warning(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    No-warning parsing:
      1) ISO formats
      2) explicit dd/mm
      3) explicit mm/dd
    """
    ts = series.astype(str).fillna("").str.strip()

    dt = pd.to_datetime(ts, errors="coerce", format="%Y-%m-%d %H:%M:%S")
    if dt.isna().any():
        dt = dt.fillna(pd.to_datetime(ts, errors="coerce", format="%Y-%m-%d %H:%M"))
    if dt.isna().any():
        dt = dt.fillna(pd.to_datetime(ts, errors="coerce", format="%Y-%m-%dT%H:%M:%S"))
    if dt.isna().any():
        dt = dt.fillna(pd.to_datetime(ts, errors="coerce", format="%Y-%m-%dT%H:%M:%SZ"))
    if dt.isna().any():
        dt = dt.fillna(pd.to_datetime(ts, errors="coerce", format="%Y-%m-%dT%H:%M:%S%z"))

    if dt.isna().any():
        dt = dt.fillna(pd.to_datetime(ts, errors="coerce", format="%d/%m/%Y %H:%M:%S"))
    if dt.isna().any():
        dt = dt.fillna(pd.to_datetime(ts, errors="coerce", format="%d/%m/%Y %H:%M"))

    if dt.isna().any():
        dt = dt.fillna(pd.to_datetime(ts, errors="coerce", format="%m/%d/%Y %H:%M:%S"))
    if dt.isna().any():
        dt = dt.fillna(pd.to_datetime(ts, errors="coerce", format="%m/%d/%Y %H:%M"))

    return ts, dt


def to_ist_series(dt_series: pd.Series) -> pd.Series:
    dt_series = pd.to_datetime(dt_series, errors="coerce")
    try:
        if getattr(dt_series.dt, "tz", None) is None:
            return dt_series.dt.tz_localize("Asia/Kolkata")
        return dt_series.dt.tz_convert("Asia/Kolkata")
    except Exception:
        return dt_series


def summarize_missing_dates(missing: List[date], max_ranges: int = 8) -> str:
    if not missing:
        return "[]"
    missing = sorted(missing)
    ranges: List[Tuple[date, date]] = []
    start = prev = missing[0]
    for d in missing[1:]:
        if (d - prev).days == 1:
            prev = d
        else:
            ranges.append((start, prev))
            start = prev = d
    ranges.append((start, prev))

    def fmt(a: date, b: date) -> str:
        return a.isoformat() if a == b else f"{a.isoformat()}..{b.isoformat()}"

    out = [fmt(a, b) for a, b in ranges]
    if len(out) <= max_ranges:
        return "[" + ", ".join(out) + "]"
    return "[" + ", ".join(out[:max_ranges]) + f", ... (+{len(out)-max_ranges} ranges)]"


def missing_dates_between(have_dates: set, dmin: date, dmax: date, max_span_days: int = 120) -> List[date]:
    if not have_dates or dmin is None or dmax is None:
        return []
    span = (dmax - dmin).days
    if span <= 0 or span > max_span_days:
        return []
    full = pd.date_range(dmin, dmax, freq="D").date.tolist()
    return [d for d in full if d not in have_dates]


def ensure_schema(conn) -> None:
    conn.execute(text("""ALTER TABLE forecast_runs ADD COLUMN IF NOT EXISTS run_t0_ist TIMESTAMPTZ;"""))
    conn.execute(text("""ALTER TABLE forecast_runs ADD COLUMN IF NOT EXISTS run_t0_raw TEXT;"""))
    conn.execute(text("""ALTER TABLE mi_predictions ADD COLUMN IF NOT EXISTS valid_time_raw TEXT;"""))


def load_ingested_map(conn, root_prefix: str) -> Dict[str, Tuple[int, int, Optional[str]]]:
    rows = conn.execute(text("""
        SELECT path, mtime, size, sha1
        FROM ingested_files
        WHERE path LIKE :pref
    """), {"pref": root_prefix + "%"}).fetchall()
    out: Dict[str, Tuple[int, int, Optional[str]]] = {}
    for p, mt, sz, h in rows:
        out[str(p)] = (int(mt) if mt is not None else 0, int(sz) if sz is not None else 0, h)
    return out


def mark_ingested(conn, path: Path, model_name: str, mtime: int, size: int, sha1: str):
    conn.execute(text("""
        INSERT INTO ingested_files(path, mtime, size, sha1, model_name)
        VALUES (:p, :mt, :sz, :h, :m)
        ON CONFLICT (path)
        DO UPDATE SET
          mtime=EXCLUDED.mtime,
          size=EXCLUDED.size,
          sha1=EXCLUDED.sha1,
          model_name=EXCLUDED.model_name,
          ingested_at=NOW()
    """), {"p": str(path), "mt": mtime, "sz": size, "h": sha1, "m": model_name})


def ensure_region(conn, source_sheet: str) -> str:
    rid = conn.execute(text("""
        INSERT INTO regions(region_code, region_name)
        VALUES (:code, :name)
        ON CONFLICT (region_code)
        DO UPDATE SET region_name = EXCLUDED.region_name
        RETURNING region_id
    """), {"code": source_sheet, "name": source_sheet}).scalar()

    if rid:
        return str(rid)

    rid2 = conn.execute(text("""
        SELECT region_id FROM regions WHERE region_code=:code
    """), {"code": source_sheet}).scalar()
    return str(rid2)


def ensure_plant(conn, region_id: str, site_name: str, lat: Optional[float], lon: Optional[float]) -> str:
    ptype = infer_plant_type(site_name)
    pid = conn.execute(text("""
        INSERT INTO plants(region_id, plant_name, plant_type, lat, lon)
        VALUES (:region_id, :plant_name, :plant_type, :lat, :lon)
        ON CONFLICT (region_id, plant_name)
        DO UPDATE SET
          plant_type = EXCLUDED.plant_type,
          lat = COALESCE(EXCLUDED.lat, plants.lat),
          lon = COALESCE(EXCLUDED.lon, plants.lon)
        RETURNING plant_id
    """), {
        "region_id": region_id,
        "plant_name": site_name,
        "plant_type": ptype,
        "lat": lat,
        "lon": lon,
    }).scalar()

    if pid:
        return str(pid)

    pid2 = conn.execute(text("""
        SELECT plant_id FROM plants
        WHERE region_id=:rid AND plant_name=:p
    """), {"rid": region_id, "p": site_name}).scalar()
    return str(pid2)


def ensure_run(conn, model_name: str, region_id: str, revision: str,
               t0_raw: str, t0_ist, t0_utc) -> str:
    run_id = conn.execute(text("""
        INSERT INTO forecast_runs(model_name, region_id, revision, run_t0_utc, run_t0_ist, run_t0_raw)
        VALUES (:m, :rid, :rev, :t0utc, :t0ist, :t0raw)
        ON CONFLICT (model_name, region_id, revision, run_t0_utc)
        DO UPDATE SET
          run_t0_ist = EXCLUDED.run_t0_ist,
          run_t0_raw = EXCLUDED.run_t0_raw
        RETURNING run_id
    """), {
        "m": model_name,
        "rid": region_id,
        "rev": revision,
        "t0utc": t0_utc,
        "t0ist": t0_ist,
        "t0raw": t0_raw,
    }).scalar()

    if run_id:
        return str(run_id)

    run_id2 = conn.execute(text("""
        SELECT run_id FROM forecast_runs
        WHERE model_name=:m AND region_id=:rid AND revision=:rev AND run_t0_utc=:t0utc
    """), {"m": model_name, "rid": region_id, "rev": revision, "t0utc": t0_utc}).scalar()

    return str(run_id2) if run_id2 else ""


def runrev_exists(conn, model_name: str, region_id: str, revision: str, run_t0_utc: datetime) -> bool:
    key = (model_name, region_id, revision, run_t0_utc)
    if key in RUNREV_EXISTS_CACHE:
        return RUNREV_EXISTS_CACHE[key]

    exists = conn.execute(text("""
        SELECT 1
        FROM forecast_runs
        WHERE model_name=:m AND region_id=:rid AND revision=:rev AND run_t0_utc=:t0
        LIMIT 1
    """), {"m": model_name, "rid": region_id, "rev": revision, "t0": run_t0_utc}).fetchone() is not None

    RUNREV_EXISTS_CACHE[key] = exists
    return exists


def overwrite_predictions(conn, run_id: str, plant_id: str, df_group: pd.DataFrame, plant_type: str) -> int:
    conn.execute(text("DELETE FROM mi_predictions WHERE run_id=:r AND plant_id=:p"),
                 {"r": run_id, "p": plant_id})

    rows: List[Dict] = []
    for _, r in df_group.iterrows():
        vt_raw = r.get("timestamp_raw", "")
        vt_dt = r.get("timestamp_dt")
        if pd.isna(vt_dt):
            continue

        if getattr(vt_dt, "tzinfo", None) is None:
            vt_ist = vt_dt.replace(tzinfo=IST_TZ)
        else:
            vt_ist = vt_dt.astimezone(IST_TZ)

        vt_utc = vt_ist.astimezone(UTC_TZ)

        pred = r.get("prediction")
        ghi  = r.get("ghi")

        power_pred = float(pred) if pd.notna(pred) else None
        ghi_pred   = float(ghi) if pd.notna(ghi) else None

        rows.append({
            "run_id": run_id,
            "plant_id": plant_id,
            "valid_time_utc": vt_utc,
            "valid_time_ist": vt_ist,
            "valid_time_raw": vt_raw,
            "ghi_pred_wm2": ghi_pred,
            "power_pred_mw": power_pred,
            "solar_power_pred_mw": power_pred if plant_type == "SOLAR" else None,
            "wind_power_pred_mw": power_pred if plant_type == "WIND" else None,
        })

    if not rows:
        return 0

    conn.execute(text("""
        INSERT INTO mi_predictions(
          run_id, plant_id, valid_time_utc, valid_time_ist, valid_time_raw,
          ghi_pred_wm2, power_pred_mw, solar_power_pred_mw, wind_power_pred_mw
        )
        VALUES (
          :run_id, :plant_id, :valid_time_utc, :valid_time_ist, :valid_time_raw,
          :ghi_pred_wm2, :power_pred_mw, :solar_power_pred_mw, :wind_power_pred_mw
        )
        ON CONFLICT (run_id, plant_id, valid_time_utc)
        DO UPDATE SET
          valid_time_ist = EXCLUDED.valid_time_ist,
          valid_time_raw = EXCLUDED.valid_time_raw,
          ghi_pred_wm2 = EXCLUDED.ghi_pred_wm2,
          power_pred_mw = EXCLUDED.power_pred_mw,
          solar_power_pred_mw = EXCLUDED.solar_power_pred_mw,
          wind_power_pred_mw = EXCLUDED.wind_power_pred_mw
    """), rows)

    return len(rows)


def list_csv_files(root: Path) -> List[Path]:
    return sorted(set(list(root.rglob("*.csv")) + list(root.rglob("*.CSV"))), key=str)


def update_file_date_cache(path: Path, mtime: int, size: int) -> Tuple[Set[date], Optional[date], Optional[date]]:
    k = str(path)
    cached = FILE_DATE_CACHE.get(k)
    if cached and cached[0] == mtime and cached[1] == size:
        return cached[2], cached[3], cached[4]

    try:
        df_ts = pd.read_csv(path, usecols=["timestamp"])
        _, dt = parse_timestamps_no_warning(df_ts["timestamp"])
        dt_ist = to_ist_series(dt)
        ds = set(dt_ist.dropna().dt.date.tolist()) if dt_ist is not None else set()
        dmin = min(ds) if ds else None
        dmax = max(ds) if ds else None
    except Exception:
        ds, dmin, dmax = set(), None, None

    FILE_DATE_CACHE[k] = (mtime, size, ds, dmin, dmax)
    return ds, dmin, dmax


def db_check_window(model_name: str, dmin: date, dmax: date) -> Tuple[Optional[date], Optional[date], int, List[date]]:
    if dmin is None or dmax is None:
        return None, None, 0, []

    start_dt = datetime(dmin.year, dmin.month, dmin.day, 0, 0, 0, tzinfo=IST_TZ)
    end_dt = datetime(dmax.year, dmax.month, dmax.day, 0, 0, 0, tzinfo=IST_TZ) + timedelta(days=1)

    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT
              MIN(p.valid_time_ist)::date AS dmin,
              MAX(p.valid_time_ist)::date AS dmax,
              COUNT(*)::bigint           AS n
            FROM mi_predictions p
            JOIN forecast_runs r ON r.run_id = p.run_id
            WHERE r.model_name = :m
              AND p.valid_time_ist >= :start_dt
              AND p.valid_time_ist <  :end_dt
        """), {"m": model_name, "start_dt": start_dt, "end_dt": end_dt}).fetchone()

        db_min = row[0]
        db_max = row[1]
        nrows = int(row[2]) if row and row[2] is not None else 0

        daily = conn.execute(text("""
            SELECT p.valid_time_ist::date AS d, COUNT(*)::bigint AS n
            FROM mi_predictions p
            JOIN forecast_runs r ON r.run_id = p.run_id
            WHERE r.model_name = :m
              AND p.valid_time_ist >= :start_dt
              AND p.valid_time_ist <  :end_dt
            GROUP BY 1
            ORDER BY 1
        """), {"m": model_name, "start_dt": start_dt, "end_dt": end_dt}).fetchall()

    have_db_dates = {r[0] for r in daily if r and r[0] is not None}
    missing_db = missing_dates_between(have_db_dates, dmin, dmax, max_span_days=120)
    return db_min, db_max, nrows, missing_db


def ingest_one_file(conn, path: Path, model_name: str) -> Tuple[bool, int, int]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return False, 0, 0

    required = ["revision", "timestamp", "site_name", "source_sheet"]
    if any(c not in df.columns for c in required):
        return False, 0, 0

    df["revision"] = df["revision"].astype(str).str.strip()
    df["source_sheet"] = df["source_sheet"].astype(str).str.strip()
    df["site_name"] = df["site_name"].astype(str).str.strip()

    ts_raw, ts_dt = parse_timestamps_no_warning(df["timestamp"])
    df["timestamp_raw"] = ts_raw
    df["timestamp_dt"] = ts_dt

    grouped = df.groupby(["source_sheet", "site_name", "revision"], dropna=False)

    points_total = 0
    runs_with_points = 0

    for (source_sheet, site_name, revision), g in grouped:
        source_sheet = str(source_sheet or "").strip()
        site_name = str(site_name or "").strip()
        revision = str(revision or "").strip()
        if not source_sheet or not site_name or not revision:
            continue

        g = g.sort_values("timestamp_dt")
        t0_dt = g["timestamp_dt"].min()
        if pd.isna(t0_dt):
            continue

        if getattr(t0_dt, "tzinfo", None) is None:
            t0_ist = t0_dt.replace(tzinfo=IST_TZ)
        else:
            t0_ist = t0_dt.astimezone(IST_TZ)
        t0_utc = t0_ist.astimezone(UTC_TZ)

        region_id = ensure_region(conn, source_sheet)

        if ONLY_NEW_RUNREV and runrev_exists(conn, model_name, region_id, revision, t0_utc):
            continue

        lat = lon = None
        if "latitude" in g.columns:
            vv = pd.to_numeric(g["latitude"], errors="coerce").dropna().head(1)
            lat = float(vv.iloc[0]) if len(vv) else None
        if "longitude" in g.columns:
            vv = pd.to_numeric(g["longitude"], errors="coerce").dropna().head(1)
            lon = float(vv.iloc[0]) if len(vv) else None

        plant_id = ensure_plant(conn, region_id, site_name, lat, lon)

        try:
            t0_raw = g.loc[g["timestamp_dt"].idxmin(), "timestamp_raw"]
        except Exception:
            t0_raw = ""

        run_id = ensure_run(conn, model_name, region_id, revision, t0_raw, t0_ist, t0_utc)
        if not run_id:
            continue

        plant_type = infer_plant_type(site_name)
        n = overwrite_predictions(conn, run_id, plant_id, g, plant_type)
        points_total += n
        if n > 0:
            runs_with_points += 1

    return True, points_total, runs_with_points


def scan_root(root: str, model_name: str):
    global RUNREV_EXISTS_CACHE
    RUNREV_EXISTS_CACHE = {}

    if not root:
        print(f"[{now_ist_str()}] [WARN] {model_name} root is empty")
        return

    rp = Path(root).expanduser().resolve()
    if not rp.exists():
        print(f"[{now_ist_str()}] [WARN] {model_name} root not found: {rp}")
        return

    files = list_csv_files(rp)

    scan_dates: Set[date] = set()
    scan_min: Optional[date] = None
    scan_max: Optional[date] = None

    with engine.begin() as conn:
        ensure_schema(conn)
        ing_map = load_ingested_map(conn, str(rp))

    ok = bad = 0
    new = changed = unchanged = 0
    points_total = 0
    runs_total = 0

    for p in files:
        try:
            st = p.stat()
            mtime, size = int(st.st_mtime), int(st.st_size)

            ds, dmin_f, dmax_f = update_file_date_cache(p, mtime, size)
            scan_dates.update(ds)
            if dmin_f is not None:
                scan_min = dmin_f if scan_min is None else min(scan_min, dmin_f)
            if dmax_f is not None:
                scan_max = dmax_f if scan_max is None else max(scan_max, dmax_f)

            prev = ing_map.get(str(p))
            if prev is None:
                status = "NEW"
            else:
                prev_mtime, prev_size, _ = prev
                status = "UNCHANGED" if (prev_mtime == mtime and prev_size == size) else "CHANGED"

            if status == "NEW":
                new += 1
            elif status == "CHANGED":
                changed += 1
            else:
                unchanged += 1

            if SKIP_UNCHANGED and status == "UNCHANGED":
                ok += 1
                continue

            prev_sha1 = prev[2] if prev else None
            sha1 = prev_sha1 if (status == "UNCHANGED" and prev_sha1) else sha1_file(p)

            with engine.begin() as conn:
                ensure_schema(conn)
                f_ok, pts, runs = ingest_one_file(conn, p, model_name)
                if not f_ok:
                    bad += 1
                    continue
                mark_ingested(conn, p, model_name, mtime, size, sha1)

            ok += 1
            points_total += int(pts)
            runs_total += int(runs)

        except Exception:
            bad += 1

    if scan_min and scan_max and scan_dates:
        missing_files = missing_dates_between(scan_dates, scan_min, scan_max, max_span_days=120)
        miss_files_str = summarize_missing_dates(missing_files)
        range_str = f"{scan_min.isoformat()}..{scan_max.isoformat()}"
    else:
        miss_files_str = "[]"
        range_str = "NONE"

    print(f"[{now_ist_str()}] [SCAN] {model_name} dir={rp} files={len(files)} IST_dates={range_str} missing_in_files={miss_files_str}")

    db_min, db_max, db_rows, missing_db = db_check_window(model_name, scan_min, scan_max)
    miss_db_str = summarize_missing_dates(missing_db) if missing_db else "[]"
    db_range_str = f"{db_min}..{db_max}" if (db_min and db_max) else "NONE"

    print(
        f"[{now_ist_str()}] [DONE] {model_name} dir={rp} ok={ok} bad={bad} "
        f"new={new} changed={changed} unchanged={unchanged} points={points_total} runs={runs_total} "
        f"DB_dates={db_range_str} missing_in_db={miss_db_str} rows_in_db_window={db_rows}"
    )


def main():
    print(f"[{now_ist_str()}] Folder watcher ingest started.")
    print(f"[{now_ist_str()}] DATABASE_URL={DATABASE_URL}")
    print(f"[{now_ist_str()}] NOWCAST_ROOT={NOWCAST_ROOT}")
    print(f"[{now_ist_str()}] MEDIUM_ROOT={MEDIUM_ROOT}")
    print(f"[{now_ist_str()}] INTRA_ROOT={INTRA_ROOT}")
    print(f"[{now_ist_str()}] POLL_SECONDS={POLL_SECONDS} SINGLE_PASS={SINGLE_PASS}")
    print(f"[{now_ist_str()}] SKIP_UNCHANGED={SKIP_UNCHANGED} ONLY_NEW_RUNREV={ONLY_NEW_RUNREV}")
    print(f"[{now_ist_str()}] MODE=OVERWRITE (snapshot: delete+insert per run+plant)\n")

    while True:
        scan_root(NOWCAST_ROOT, "NOWCAST")
        scan_root(MEDIUM_ROOT, "MEDIUM")
        scan_root(INTRA_ROOT, "INTRA")   # NEW

        if SINGLE_PASS:
            break

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()

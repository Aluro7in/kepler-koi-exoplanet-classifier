"""
data_loader.py
Loads the Kepler Q1-Q17 DR25 TCE table and builds a classification label.

Label strategy (see master prompt):
  Option A (preferred): merge with NASA's cumulative KOI table on
                         (kepid, tce_plnt_num) to get real koi_disposition.
  Option B (fallback):  proxy label from tce_rogue_flag + tce_nkoi, used
                         automatically if the KOI table can't be reached
                         (e.g. no internet access / firewalled sandbox).
"""
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

KOI_TAP_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
    "select+kepid,kepoi_name,koi_tce_plnt_num,koi_disposition+from+cumulative"
    "&format=csv"
)


def load_raw_tce(csv_path: str) -> pd.DataFrame:
    """Load the raw TCE csv, skipping the '#' metadata header block."""
    import os
    import zipfile
    if not os.path.exists(csv_path):
        log.warning(f"{csv_path} not found. Attempting to locate kepler-tce-project.zip to extract it...")
        zip_filename = "kepler-tce-project.zip"
        search_dirs = ["data/raw", ".", "..", "../..", "../../.."]
        zip_path = None
        for d in search_dirs:
            p = os.path.join(d, zip_filename)
            if os.path.exists(p):
                zip_path = p
                break
        if zip_path:
            log.info(f"Found zip file at {os.path.abspath(zip_path)}. Extracting...")
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                csv_in_zip = None
                target_filename = os.path.basename(csv_path)
                for f in zf.namelist():
                    if f.endswith(target_filename) or (target_filename in f and f.endswith(".csv")):
                        csv_in_zip = f
                        break
                if csv_in_zip:
                    log.info(f"Extracting {csv_in_zip} from zip to {csv_path}...")
                    with open(csv_path, 'wb') as f_out:
                        f_out.write(zf.read(csv_in_zip))
                    log.info(f"Extracted CSV successfully to {csv_path}.")
                else:
                    raise FileNotFoundError(f"Could not find any CSV matching '{target_filename}' inside {zip_path}")
        else:
            raise FileNotFoundError(f"CSV file '{csv_path}' not found, and '{zip_filename}' could not be located in search paths: {search_dirs}")
            
    df = pd.read_csv(csv_path, comment="#", low_memory=False)
    log.info(f"Loaded raw TCE table: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def build_labels_from_koi_merge(df: pd.DataFrame, koi_cache_path: str) -> pd.DataFrame:
    """Option A: fetch (or load cached) cumulative KOI table and merge in
    real dispositions. Raises on failure so caller can fall back."""
    import os
    if os.path.exists(koi_cache_path):
        koi = pd.read_csv(koi_cache_path)
        log.info(f"Loaded cached KOI table from {koi_cache_path}")
    else:
        koi = pd.read_csv(KOI_TAP_URL)  # will raise if no network access
        koi.to_csv(koi_cache_path, index=False)
        log.info("Downloaded and cached KOI cumulative table.")

    merged = df.merge(
        koi[["kepid", "koi_tce_plnt_num", "koi_disposition"]],
        left_on=["kepid", "tce_plnt_num"],
        right_on=["kepid", "koi_tce_plnt_num"],
        how="left",
    )
    merged["koi_disposition"] = merged["koi_disposition"].fillna("NTP")

    label_map_2class = {
        "CONFIRMED": "PLANET",
        "CANDIDATE": "PLANET",
        "FALSE POSITIVE": "NOT_PLANET",
        "NTP": "NOT_PLANET",
    }
    merged["label"] = merged["koi_disposition"].map(label_map_2class)
    merged["label_source"] = "koi_merge (Option A)"
    return merged


def build_labels_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Option B fallback: weak proxy label from columns already in the file.

    label = PLANET  if tce_rogue_flag == 0 AND tce_nkoi > 0
            NOT_PLANET otherwise
    """
    log.warning(
        "Using PROXY labels (tce_rogue_flag + tce_nkoi). "
        "This is NOT a validated astrophysical disposition -- "
        "treat results as a demonstration of the pipeline, not a "
        "scientific classifier. Use Option A (KOI merge) with internet "
        "access for a real disposition-based model."
    )
    out = df.copy()
    is_planetlike = (out["tce_rogue_flag"] == 0) & (out["tce_nkoi"] > 0)
    out["label"] = np.where(is_planetlike, "PLANET", "NOT_PLANET")
    out["label_source"] = "proxy (Option B)"
    return out


def load_labeled_tce(
    csv_path: str, koi_cache_path: str = "data/koi_cumulative.csv"
) -> pd.DataFrame:
    """Main entry point: load raw data, try Option A, fall back to Option B."""
    df = load_raw_tce(csv_path)
    try:
        labeled = build_labels_from_koi_merge(df, koi_cache_path)
        log.info("Label source: KOI merge (Option A) - real dispositions used.")
    except Exception as e:
        log.warning(f"KOI merge failed ({type(e).__name__}: {e}). Falling back to proxy labels.")
        labeled = build_labels_proxy(df)

    counts = labeled["label"].value_counts()
    log.info(f"Label distribution:\n{counts}")
    return labeled


if __name__ == "__main__":
    df = load_labeled_tce("data/raw/q1_q17_dr25_tce_2026_07_08_22_53_14.csv")
    print(df[["kepid", "tce_plnt_num", "label", "label_source"]].head(10))

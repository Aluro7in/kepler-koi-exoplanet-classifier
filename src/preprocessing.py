"""
preprocessing.py
Column selection, skew correction, and the ColumnTransformer (impute + scale)
used identically by every model (ML, MLP, Transformer).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer

DROP_COLS = [
    # fully-null label columns in this delivery
    "av_pp_afp", "av_pp_pc", "av_vf_ntp_err", "av_pp_ntp", "av_vf_pc_err",
    "av_vf_afp", "av_vf_pc", "av_vf_ntp", "av_vf_afp_err", "av_pred_class",
    "av_training_set", "tce_ioflag",
    # ids / links / bookkeeping
    "rowid", "kepid", "tce_plnt_num", "tce_delivname", "rowupdate",
    "tce_datalink_dvs", "tce_datalink_dvr",
    # constant columns
    "tce_limbdark_mod", "tce_trans_mod",
    # provenance strings (dropped for v1, see master prompt §2.1)
    "tce_steff_prov", "tce_slogg_prov", "tce_smet_prov", "tce_sradius_prov",
    # label bookkeeping columns added by data_loader (never features)
    "label", "label_source", "koi_disposition", "koi_tce_plnt_num",
]

LOG1P_COLS = [
    "tce_period", "tce_depth", "tce_prad", "tce_sma", "tce_insol",
    "tce_model_snr", "tce_max_mult_ev", "wst_depth",
]

# FIXED shifts (derived once from the full training distribution's min values,
# with margin) so log1p(x + shift) is identical whether applied to the full
# 34k-row training set or a single new row at inference time. Using a
# per-batch-computed min (as an earlier version of this file did) silently
# breaks predictions on small batches -- DO NOT recompute these from the
# input batch.
LOG1P_SHIFTS = {
    "tce_period": 0.0,
    "tce_depth": 0.0,
    "tce_prad": 0.0,
    "tce_sma": 0.0,
    "tce_insol": 0.0,
    "tce_model_snr": 2.0,       # observed min ~ -1.0
    "tce_max_mult_ev": 0.0,
    "wst_depth": 1_400_000.0,   # observed min ~ -1,394,000
}

SENTINEL_MISSING = {"boot_fap": -1.0}


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in DROP_COLS]


def clean_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, sentinel in SENTINEL_MISSING.items():
        if col in out.columns:
            out.loc[out[col] == sentinel, col] = np.nan
    return out


def apply_log1p(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p with a FIXED, precomputed shift per column (see
    LOG1P_SHIFTS) so this transform is identical at train time and at
    inference time, no matter how many rows are being processed."""
    out = df.copy()
    for col in LOG1P_COLS:
        if col in out.columns:
            shift = LOG1P_SHIFTS.get(col, 0.0)
            shifted = out[col] + shift
            # clip at 0 in case an unseen row falls slightly outside the
            # observed training range, so log1p never sees a negative input
            out[col] = np.log1p(shifted.clip(lower=0))
    return out


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Full non-model-fitting feature prep: drop cols, fix sentinels, log1p.
    Returns a numeric-only DataFrame ready for the ColumnTransformer."""
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    X = clean_sentinels(X)
    X = apply_log1p(X)
    # keep numeric columns only (object/string provenance cols already dropped)
    X = X.select_dtypes(include=[np.number])
    return X


def build_preprocessor() -> Pipeline:
    """Median-impute then standard-scale. Shared by every model."""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

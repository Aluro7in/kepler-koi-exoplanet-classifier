"""
split.py
Group-aware train/val/test split so TCEs from the same star (kepid) never
appear in more than one split.
"""
import logging
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

log = logging.getLogger(__name__)


def grouped_train_val_test_split(df, groups, label_col="label", random_state=42):
    """80 / 15 / 5 split by unique kepid."""
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(df, groups=groups))

    df_temp = df.iloc[temp_idx]
    groups_temp = groups.iloc[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)  # 0.25*0.20=0.05
    val_idx_rel, test_idx_rel = next(gss2.split(df_temp, groups=groups_temp))

    val_idx = temp_idx[val_idx_rel]
    test_idx = temp_idx[test_idx_rel]

    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        counts = df.iloc[idx][label_col].value_counts(normalize=True) * 100
        log.info(f"{name}: n={len(idx)}  class balance -> " +
                  ", ".join(f"{k}={v:.1f}%" for k, v in counts.items()))

    overlap = set(groups.iloc[train_idx]) & set(groups.iloc[val_idx]) & set(groups.iloc[test_idx])
    assert len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])) == 0, "kepid leakage train/test!"
    assert len(set(groups.iloc[val_idx]) & set(groups.iloc[test_idx])) == 0, "kepid leakage val/test!"

    return train_idx, val_idx, test_idx

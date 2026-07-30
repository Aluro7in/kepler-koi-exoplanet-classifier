"""
train_ml.py
Trains & saves the classical ML models: Logistic Regression, Random Forest,
XGBoost, and SVM — all on the same preprocessed feature matrix used by the
neural models, so results are directly comparable.
"""
import time
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from data_loader import load_labeled_tce
from preprocessing import build_feature_matrix, build_preprocessor
from split import grouped_train_val_test_split

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False

LABEL_MAP = {"NOT_PLANET": 0, "PLANET": 1}


def get_data():
    df = load_labeled_tce("data/raw/q1_q17_dr25_tce_2026_07_08_22_53_14.csv",
                           koi_cache_path="data/koi_cumulative.csv")
    X = build_feature_matrix(df)
    y = df["label"].map(LABEL_MAP).values
    groups = df["kepid"]
    train_idx, val_idx, test_idx = grouped_train_val_test_split(df, groups, "label")
    return X, y, train_idx, val_idx, test_idx


def build_model_pipeline(estimator):
    return Pipeline(steps=[("pre", build_preprocessor()), ("clf", estimator)])


def train_all():
    X, y, train_idx, val_idx, test_idx = get_data()
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    results = {}

    # ---- Logistic Regression ----
    print("\n[1/4] Logistic Regression")
    t0 = time.time()
    logreg = build_model_pipeline(LogisticRegression(class_weight="balanced", max_iter=2000))
    logreg.fit(X_train, y_train)
    print(f"  val accuracy: {logreg.score(X_val, y_val):.4f}  ({time.time()-t0:.1f}s)")
    joblib.dump(logreg, "models/ml/logistic_regression.joblib")
    results["logistic_regression"] = logreg

    # ---- Random Forest ----
    print("\n[2/4] Random Forest (RandomizedSearchCV)")
    t0 = time.time()
    rf_param_dist = {
        "clf__n_estimators": [200, 300, 400],
        "clf__max_depth": [None, 10, 20, 30],
        "clf__min_samples_leaf": [1, 2, 4],
    }
    rf_base = build_model_pipeline(RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1))
    rf_search = RandomizedSearchCV(rf_base, rf_param_dist, n_iter=6, cv=3, random_state=42, n_jobs=-1)
    rf_search.fit(X_train, y_train)
    rf = rf_search.best_estimator_
    print(f"  best params: {rf_search.best_params_}")
    print(f"  val accuracy: {rf.score(X_val, y_val):.4f}  ({time.time()-t0:.1f}s)")
    joblib.dump(rf, "models/ml/random_forest.joblib")
    results["random_forest"] = rf

    # ---- Gradient Boosting (XGBoost preferred) ----
    print(f"\n[3/4] Gradient Boosting ({'XGBoost' if HAS_XGB else 'sklearn GBM'})")
    t0 = time.time()
    if HAS_XGB:
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        gbm_est = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
    else:
        gbm_est = GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=42)
    gbm = build_model_pipeline(gbm_est)
    gbm.fit(X_train, y_train)
    print(f"  val accuracy: {gbm.score(X_val, y_val):.4f}  ({time.time()-t0:.1f}s)")
    joblib.dump(gbm, "models/ml/gradient_boosting.joblib")
    results["gradient_boosting"] = gbm

    # ---- SVM (subsample for tractable training time) ----
    print("\n[4/4] SVM (RBF kernel, subsampled for speed)")
    t0 = time.time()
    max_svm_n = 6000
    if len(X_train) > max_svm_n:
        rng = np.random.RandomState(42)
        sub_idx = rng.choice(len(X_train), size=max_svm_n, replace=False)
        X_svm, y_svm = X_train.iloc[sub_idx], y_train[sub_idx]
    else:
        X_svm, y_svm = X_train, y_train
    svm = build_model_pipeline(SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))
    svm.fit(X_svm, y_svm)
    print(f"  val accuracy: {svm.score(X_val, y_val):.4f}  ({time.time()-t0:.1f}s)  (trained on {len(X_svm)} rows)")
    joblib.dump(svm, "models/ml/svm.joblib")
    results["svm"] = svm

    return results


if __name__ == "__main__":
    train_all()
    print("\nAll classical ML models trained and saved to models/ml/")

"""
store_and_ensemble.py
Logs model training history, predictions, and query performance metrics to
PostgreSQL/AlloyDB Omni (falling back to a local SQLite database if unreachable).
Then loads the stored predictions to train/fine-tune a Stacking Meta-Classifier
that combines the models' individual predictions to serve the optimal exoplanet output.
"""
import os
import time
import sqlite3
import datetime
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Try to import psycopg2 for PostgreSQL/AlloyDB Omni connection
try:
    import psycopg2
    from psycopg2.extras import execute_values
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

from data_loader import load_labeled_tce
from preprocessing import build_feature_matrix
from split import grouped_train_val_test_split
from models_pytorch import MLPClassifier, TabTransformer

LABEL_MAP = {"NOT_PLANET": 0, "PLANET": 1}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_db_connection():
    """Attempts to connect to PostgreSQL/AlloyDB using environment variables.
    Falls back to a local SQLite database if it cannot connect or if psycopg2 is missing."""
    if HAS_POSTGRES:
        host = os.getenv("ALLOYDB_OMNI_HOST", "localhost")
        port = os.getenv("ALLOYDB_OMNI_PORT", "5432")
        db = os.getenv("ALLOYDB_OMNI_DB", "postgres")
        user = os.getenv("ALLOYDB_OMNI_USER", "postgres")
        pwd = os.getenv("ALLOYDB_OMNI_PASSWORD", "postgres")
        
        try:
            conn = psycopg2.connect(
                host=host, port=port, database=db, user=user, password=pwd, connect_timeout=3
            )
            print(f"[INFO] Connected to AlloyDB Omni/PostgreSQL at {host}:{port}")
            return conn, "postgres"
        except Exception as e:
            print(f"[WARNING] PostgreSQL connection failed ({e}). Falling back to SQLite.")
    
    # SQLite fallback
    db_path = "data/kepler_performance_and_predictions.db"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    print(f"[INFO] Connected to SQLite database at {db_path}")
    return conn, "sqlite"


def create_tables(conn, db_type):
    """Creates the tables for training history, predictions, and query performance metrics."""
    cursor = conn.cursor()
    
    # Define SQL syntax based on DB type (SERIAL vs AUTOINCREMENT)
    id_type = "SERIAL PRIMARY KEY" if db_type == "postgres" else "INTEGER PRIMARY KEY AUTOINCREMENT"
    timestamp_default = "CURRENT_TIMESTAMP"
    
    # 1. Training History Table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS training_history (
            id {id_type},
            run_id VARCHAR(50),
            model_name VARCHAR(50),
            epoch INT,
            train_loss REAL,
            val_loss REAL,
            train_acc REAL,
            val_acc REAL,
            created_at TIMESTAMP DEFAULT {timestamp_default}
        )
    """)
    
    # 2. Predictions Table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS predictions (
            id {id_type},
            run_id VARCHAR(50),
            model_name VARCHAR(50),
            kepid BIGINT,
            tce_plnt_num INT,
            true_label INT,
            predicted_label INT,
            probability REAL,
            split VARCHAR(10),
            created_at TIMESTAMP DEFAULT {timestamp_default}
        )
    """)
    
    # 3. DB/Query Performance Table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS db_performance_metrics (
            id {id_type},
            query_desc TEXT,
            query_type VARCHAR(20),
            execution_count INT,
            total_time_ms REAL,
            mean_time_ms REAL,
            rows_affected INT,
            created_at TIMESTAMP DEFAULT {timestamp_default}
        )
    """)
    
    conn.commit()
    cursor.close()
    print("[INFO] DB Tables created/verified successfully.")


def log_performance(conn, db_type, desc, q_type, count, time_ms, rows):
    """Logs database/query execution metrics to the performance table."""
    cursor = conn.cursor()
    query = """
        INSERT INTO db_performance_metrics (query_desc, query_type, execution_count, total_time_ms, mean_time_ms, rows_affected)
        VALUES (%s, %s, %s, %s, %s, %s)
    """ if db_type == "postgres" else """
        INSERT INTO db_performance_metrics (query_desc, query_type, execution_count, total_time_ms, mean_time_ms, rows_affected)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    cursor.execute(query, (desc, q_type, count, time_ms, time_ms / max(1, count), rows))
    conn.commit()
    cursor.close()


def store_training_history(conn, db_type, run_id, model_name, history):
    """Saves neural network training loss and accuracy history per epoch."""
    t0 = time.time()
    cursor = conn.cursor()
    
    epochs = len(history["train_loss"])
    rows_to_insert = []
    for epoch in range(1, epochs + 1):
        rows_to_insert.append((
            run_id,
            model_name,
            epoch,
            float(history["train_loss"][epoch-1]),
            float(history["val_loss"][epoch-1]),
            float(history["train_acc"][epoch-1]),
            float(history["val_acc"][epoch-1])
        ))
        
    query = """
        INSERT INTO training_history (run_id, model_name, epoch, train_loss, val_loss, train_acc, val_acc)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """ if db_type == "postgres" else """
        INSERT INTO training_history (run_id, model_name, epoch, train_loss, val_loss, train_acc, val_acc)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    
    cursor.executemany(query, rows_to_insert)
    conn.commit()
    rows_count = len(rows_to_insert)
    cursor.close()
    
    elapsed_ms = (time.time() - t0) * 1000.0
    print(f"[INFO] Logged {rows_count} training history epochs for {model_name} in {elapsed_ms:.1f}ms")
    log_performance(conn, db_type, f"Insert training history for {model_name}", "INSERT", 1, elapsed_ms, rows_count)


def store_predictions(conn, db_type, run_id, model_name, kepids, tce_nums, y_true, y_pred, probs, split_name):
    """Saves predictions of a model on a dataset split to the database."""
    t0 = time.time()
    cursor = conn.cursor()
    
    rows_to_insert = []
    for i in range(len(kepids)):
        rows_to_insert.append((
            run_id,
            model_name,
            int(kepids[i]),
            int(tce_nums[i]),
            int(y_true[i]),
            int(y_pred[i]),
            float(probs[i]),
            split_name
        ))
        
    query = """
        INSERT INTO predictions (run_id, model_name, kepid, tce_plnt_num, true_label, predicted_label, probability, split)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """ if db_type == "postgres" else """
        INSERT INTO predictions (run_id, model_name, kepid, tce_plnt_num, true_label, predicted_label, probability, split)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    cursor.executemany(query, rows_to_insert)
    conn.commit()
    rows_count = len(rows_to_insert)
    cursor.close()
    
    elapsed_ms = (time.time() - t0) * 1000.0
    print(f"[INFO] Logged {rows_count} predictions for {model_name} ({split_name}) in {elapsed_ms:.1f}ms")
    log_performance(conn, db_type, f"Insert predictions for {model_name} ({split_name})", "INSERT", 1, elapsed_ms, rows_count)


def get_data_splits():
    """Loads and returns training, validation, and test datasets and splits."""
    df = load_labeled_tce("data/raw/q1_q17_dr25_tce_2026_07_08_22_53_14.csv",
                           koi_cache_path="data/koi_cumulative.csv")
    X = build_feature_matrix(df)
    y = df["label"].map(LABEL_MAP).values
    kepids = df["kepid"].values
    tce_nums = df["tce_plnt_num"].values
    
    train_idx, val_idx, test_idx = grouped_train_val_test_split(df, df["kepid"], "label")
    
    return X, y, kepids, tce_nums, train_idx, val_idx, test_idx


def generate_and_store_all_predictions(conn, db_type, run_id):
    """Loads all models, generates predictions on validation and test sets, and writes them to the DB."""
    X, y, kepids, tce_nums, train_idx, val_idx, test_idx = get_data_splits()
    
    # 1. Classical ML models
    ml_models = ["logistic_regression", "random_forest", "gradient_boosting", "svm"]
    for model_name in ml_models:
        path = f"models/ml/{model_name}.joblib"
        if not os.path.exists(path):
            print(f"[WARNING] Model file {path} not found. Skipping.")
            continue
        pipe = joblib.load(path)
        
        # Validation Set Predictions
        val_X = X.iloc[val_idx]
        val_y = y[val_idx]
        val_kepid = kepids[val_idx]
        val_tce = tce_nums[val_idx]
        
        val_preds = pipe.predict(val_X)
        val_probs = pipe.predict_proba(val_X)[:, 1]
        store_predictions(conn, db_type, run_id, model_name, val_kepid, val_tce, val_y, val_preds, val_probs, "val")
        
        # Test Set Predictions
        test_X = X.iloc[test_idx]
        test_y = y[test_idx]
        test_kepid = kepids[test_idx]
        test_tce = tce_nums[test_idx]
        
        test_preds = pipe.predict(test_X)
        test_probs = pipe.predict_proba(test_X)[:, 1]
        store_predictions(conn, db_type, run_id, model_name, test_kepid, test_tce, test_y, test_preds, test_probs, "test")
        
    # 2. PyTorch neural network models (MLP, TabTransformer)
    pre = joblib.load("models/dl/preprocessor.joblib")
    feature_cols = joblib.load("models/dl/feature_columns.joblib")
    
    nn_models = ["mlp", "transformer"]
    for model_name in nn_models:
        ckpt = f"models/dl/{model_name}_final.pt"
        if not os.path.exists(ckpt):
            print(f"[WARNING] Neural network checkpoint {ckpt} not found. Skipping.")
            continue
        
        config = joblib.load(f"models/dl/{model_name}_config.joblib")
        if model_name == "mlp":
            model = MLPClassifier(n_features=config["n_features"], n_classes=config["n_classes"])
        else:
            model = TabTransformer(n_features=config["n_features"], n_classes=config["n_classes"],
                                    d_model=config["d_model"], n_heads=config["n_heads"], n_layers=config["n_layers"])
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.to(DEVICE).eval()
        
        # Validation Set Predictions
        val_X_proc = pre.transform(X.iloc[val_idx][feature_cols])
        val_y = y[val_idx]
        val_kepid = kepids[val_idx]
        val_tce = tce_nums[val_idx]
        
        with torch.no_grad():
            val_logits = model(torch.tensor(val_X_proc, dtype=torch.float32).to(DEVICE))
            val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
            val_preds = val_logits.argmax(1).cpu().numpy()
        store_predictions(conn, db_type, run_id, model_name, val_kepid, val_tce, val_y, val_preds, val_probs, "val")
        
        # Test Set Predictions
        test_X_proc = pre.transform(X.iloc[test_idx][feature_cols])
        test_y = y[test_idx]
        test_kepid = kepids[test_idx]
        test_tce = tce_nums[test_idx]
        
        with torch.no_grad():
            test_logits = model(torch.tensor(test_X_proc, dtype=torch.float32).to(DEVICE))
            test_probs = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
            test_preds = test_logits.argmax(1).cpu().numpy()
        store_predictions(conn, db_type, run_id, model_name, test_kepid, test_tce, test_y, test_preds, test_probs, "test")


def train_meta_ensemble(conn, db_type, run_id):
    """Loads prediction probabilities from the database, trains a Stacking Meta-Classifier (Logistic Regression),
    evaluates it on the test set, and writes the meta-ensemble's final predictions to the database."""
    t0 = time.time()
    
    # Query database to get validation predictions (features for meta-classifier)
    print("\n--- Training Stacking Meta-Classifier ---")
    query_val = """
        SELECT kepid, tce_plnt_num, model_name, probability, true_label
        FROM predictions
        WHERE run_id = %s AND split = 'val'
    """ if db_type == "postgres" else """
        SELECT kepid, tce_plnt_num, model_name, probability, true_label
        FROM predictions
        WHERE run_id = ? AND split = 'val'
    """
    
    df_val = pd.read_sql_query(query_val, conn, params=(run_id,))
    
    # Query database to get test predictions
    query_test = """
        SELECT kepid, tce_plnt_num, model_name, probability, true_label
        FROM predictions
        WHERE run_id = %s AND split = 'test'
    """ if db_type == "postgres" else """
        SELECT kepid, tce_plnt_num, model_name, probability, true_label
        FROM predictions
        WHERE run_id = ? AND split = 'test'
    """
    
    df_test = pd.read_sql_query(query_test, conn, params=(run_id,))
    
    elapsed_query = (time.time() - t0) * 1000.0
    log_performance(conn, db_type, "Query predictions for stacking", "SELECT", 2, elapsed_query, len(df_val) + len(df_test))
    
    if df_val.empty or df_test.empty:
        print("[ERROR] Stored predictions not found. Cannot train ensemble.")
        return
        
    # Pivot predictions so each model's probability is a feature
    def pivot_predictions(df):
        pivoted = df.pivot(index=["kepid", "tce_plnt_num", "true_label"], columns="model_name", values="probability").reset_index()
        return pivoted
        
    pivoted_val = pivot_predictions(df_val)
    pivoted_test = pivot_predictions(df_test)
    
    model_columns = [col for col in pivoted_val.columns if col not in ["kepid", "tce_plnt_num", "true_label"]]
    print(f"Ensembling models: {model_columns}")
    
    # Train-test arrays
    X_meta_train = pivoted_val[model_columns].values
    y_meta_train = pivoted_val["true_label"].values
    
    X_meta_test = pivoted_test[model_columns].values
    y_meta_test = pivoted_test["true_label"].values
    
    # Train the meta-classifier (Logistic Regression stacker)
    meta_clf = LogisticRegression(class_weight="balanced", random_state=42)
    meta_clf.fit(X_meta_train, y_meta_train)
    
    # Predict on test
    meta_preds = meta_clf.predict(X_meta_test)
    meta_probs = meta_clf.predict_proba(X_meta_test)[:, 1]
    
    # Calculate Stacking metrics
    acc = accuracy_score(y_meta_test, meta_preds)
    prec = precision_score(y_meta_test, meta_preds)
    rec = recall_score(y_meta_test, meta_preds)
    f1 = f1_score(y_meta_test, meta_preds)
    roc_auc = roc_auc_score(y_meta_test, meta_probs)
    
    print("\n=== STACKING META-CLASSIFIER TEST PERFORMANCE ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    # Store Stacking Ensemble predictions in the database
    store_predictions(
        conn, db_type, run_id, "stacking_ensemble",
        pivoted_test["kepid"].values, pivoted_test["tce_plnt_num"].values,
        y_meta_test, meta_preds, meta_probs, "test"
    )
    
    # Plot ensembled diagnostics and save
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cm = confusion_matrix(y_meta_test, meta_preds)
    ConfusionMatrixDisplay(cm, display_labels=["NOT_PLANET", "PLANET"]).plot(ax=axes[0], colorbar=False)
    axes[0].set_title("Stacking Ensemble: confusion matrix (test)")
    
    from sklearn.metrics import RocCurveDisplay
    RocCurveDisplay.from_predictions(y_meta_test, meta_probs, ax=axes[1])
    axes[1].set_title("Stacking Ensemble: ROC curve (test)")
    fig.tight_layout()
    os.makedirs("reports/figures", exist_ok=True)
    fig.savefig("reports/figures/ensemble_diagnostics.png", dpi=130)
    plt.close(fig)
    print("[INFO] Saved ensemble diagnostic plots to reports/figures/ensemble_diagnostics.png")
    
    # Update the summary report
    update_comparison_csv(acc, prec, rec, f1, roc_auc)


def update_comparison_csv(acc, prec, rec, f1, roc_auc):
    """Adds the ensemble results to reports/model_comparison.csv."""
    csv_path = "reports/model_comparison.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Remove old ensemble row if it exists
        df = df[df["model"] != "stacking_ensemble"]
        
        new_row = pd.DataFrame([{
            "model": "stacking_ensemble",
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        df = df.sort_values("f1", ascending=False).reset_index(drop=True)
        df.to_csv(csv_path, index=False)
        print("[INFO] Updated reports/model_comparison.csv with Stacking Ensemble")
        print(df.to_string(index=False))


def display_db_stats(conn, db_type):
    """Retrieves and displays the database query performance stats."""
    cursor = conn.cursor()
    print("\n=== DATABASE QUERY PERFORMANCE STATS ===")
    cursor.execute("""
        SELECT query_desc, query_type, execution_count, total_time_ms, mean_time_ms, rows_affected
        FROM db_performance_metrics
        ORDER BY total_time_ms DESC
    """)
    rows = cursor.fetchall()
    print(f"{'Query Description':<40} | {'Type':<8} | {'Count':<5} | {'Total Time (ms)':<15} | {'Mean Time (ms)':<15} | {'Rows':<8}")
    print("-" * 110)
    for r in rows:
        print(f"{r[0][:40]:<40} | {r[1]:<8} | {r[2]:<5} | {r[3]:<15.2f} | {r[4]:<15.2f} | {r[5]:<8}")
    cursor.close()


def main():
    run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Starting Storage & Ensembling Workflow (Run ID: {run_id})")
    
    conn, db_type = get_db_connection()
    create_tables(conn, db_type)
    
    # Generate and store model predictions
    print("\n--- Generating and Storing Model Predictions ---")
    generate_and_store_all_predictions(conn, db_type, run_id)
    
    # Train stacking ensemble and write ensembled outputs
    train_meta_ensemble(conn, db_type, run_id)
    
    # Display database query performance stats
    display_db_stats(conn, db_type)
    
    conn.close()
    print("\nWorkflow completed successfully!")


if __name__ == "__main__":
    main()

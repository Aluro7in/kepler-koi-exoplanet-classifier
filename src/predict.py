"""
predict.py
Load any saved model (ML .joblib or NN .pt) plus its preprocessor and score
new TCE rows.

Usage:
  python src/predict.py --model random_forest --input new_tces.csv
  python src/predict.py --model mlp --input new_tces.csv
"""
import argparse
import joblib
import pandas as pd
import torch

from preprocessing import build_feature_matrix
from models_pytorch import MLPClassifier, TabTransformer

LABEL_NAMES = {0: "NOT_PLANET", 1: "PLANET"}
ML_MODELS = {"logistic_regression", "random_forest", "gradient_boosting", "svm"}
NN_MODELS = {"mlp", "transformer"}


def predict(model_name: str, input_csv: str) -> pd.DataFrame:
    raw = pd.read_csv(input_csv, comment="#", low_memory=False)
    X = build_feature_matrix(raw)

    if model_name in ML_MODELS:
        pipe = joblib.load(f"models/ml/{model_name}.joblib")
        preds = pipe.predict(X)
        probs = pipe.predict_proba(X)[:, 1]
    elif model_name in NN_MODELS:
        pre = joblib.load("models/dl/preprocessor.joblib")
        feature_cols = joblib.load("models/dl/feature_columns.joblib")
        X_proc = pre.transform(X[feature_cols])
        config = joblib.load(f"models/dl/{model_name}_config.joblib")
        model = (MLPClassifier(n_features=config["n_features"], n_classes=config["n_classes"]) if model_name == "mlp"
                 else TabTransformer(n_features=config["n_features"], n_classes=config["n_classes"],
                                      d_model=config["d_model"], n_heads=config["n_heads"], n_layers=config["n_layers"]))
        model.load_state_dict(torch.load(f"models/dl/{model_name}_final.pt", map_location="cpu"))
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_proc, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            preds = logits.argmax(1).numpy()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    out = raw[["kepid", "tce_plnt_num"]].copy() if "kepid" in raw.columns else pd.DataFrame(index=raw.index)
    out["predicted_label"] = [LABEL_NAMES[p] for p in preds]
    out["planet_probability"] = probs
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(ML_MODELS | NN_MODELS))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    result = predict(args.model, args.input)
    result.to_csv(args.output, index=False)
    print(result.head(20).to_string(index=False))
    print(f"\nSaved -> {args.output}")

# 🔭 Kepler TCE Classifier  
**A Complete ML + Deep Learning + Transformer Pipeline for Exoplanet Candidate Classification**

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch 2.0+">
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="License MIT">
  <img src="https://img.shields.io/badge/NASA-Data-lightgrey?style=for-the-badge&logo=nasa&logoColor=white" alt="NASA Data">
  <img src="https://img.shields.io/badge/code%20style-black-000000?style=for-the-badge&logo=Python&logoColor=white" alt="Code Style: Black">
</p>

<p align="center">
  <i>Classify every threshold crossing event from the Kepler DR25 catalog — from classical ML to tabular Transformers.<br>Watch your neural networks learn with live visualizations and animated embedding evolution.</i>
</p>

---

## 🚨 **Read This First: Label Strategy**

> **The DR25 TCE table does not include disposition labels.**  
> This pipeline resolves that automatically in two ways:

| Mode | Description | Required |
|------|-------------|----------|
| **Option A (preferred)** | Downloads the real `koi_disposition` from the NASA Exoplanet Archive and merges it by `(kepid, tce_plnt_num)`. | **Internet access** to `exoplanetarchive.ipac.caltech.edu` |
| **Option B (fallback)** | Builds a **proxy label** when Option A fails (no internet, firewall, etc.):<br>`PLANET = (tce_rogue_flag == 0) AND (tce_nkoi > 0)` | Nothing – works offline |

> ⚠️ **All benchmark results in this README were generated using Option B (proxy labels) because the build environment had no external network access.**  
> This yields near‑perfect accuracy because the proxy rule is directly derived from features already present in the data.  
> **For a scientifically meaningful exoplanet classification task, simply run this pipeline on a machine with internet access.** The code will automatically switch to Option A — no code changes required.

---

## ✨ **What Makes This Pipeline Special**

<table>
  <tr>
    <td width="50%">
      <h3>🧠 Multi‑Paradigm Modeling</h3>
      <ul>
        <li>Four classical ML models: <b>Logistic Regression, Random Forest, XGBoost, SVM</b></li>
        <li>Custom <b>PyTorch MLP</b> with dropout, batchnorm</li>
        <li>State‑of‑the‑art <b>TabTransformer</b> with multi‑head self‑attention</li>
        <li>All models share the same preprocessing and grouped train/val/test split</li>
      </ul>
    </td>
    <td width="50%">
      <h3>📊 Visual Learning Experience</h3>
      <ul>
        <li>Live <b>training curves</b> (loss & accuracy) per epoch</li>
        <li><b>Animated GIFs</b> of the neural network’s internal representation evolving over training</li>
        <li>Confusion matrix + ROC curve for every model</li>
        <li>Optional <b>LangChain narrative report</b> (LLM or template)</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>⚙️ Production‑Ready Engineering</h3>
      <ul>
        <li>Hardcoded log1p shift constants prevent silent inference‑time bugs</li>
        <li>Grouped split by <code>kepid</code> eliminates data leakage</li>
        <li>Model‑agnostic <code>predict.py</code> for scoring new TCE rows</li>
        <li>Clean separation of data loading, preprocessing, training, and evaluation</li>
      </ul>
    </td>
    <td width="50%">
      <h3>🔭 Real NASA Data</h3>
      <ul>
        <li>Uses the official <b>Kepler DR25 TCE table</b></li>
        <li>Automatic merge with cumulative KOI table for real dispositions</li>
        <li>Handles missing values, negative shifts, and complex column types</li>
      </ul>
    </td>
  </tr>
</table>

---

## 📈 **Results at a Glance**  
*(Generated with proxy labels – see warning above)*

| Model | Accuracy | Precision | Recall | F1 | ROC‑AUC |
|-------|:--------:|:---------:|:------:|:--:|:-------:|
| Logistic Regression | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| XGBoost | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **MLP (PyTorch)** | **0.996** | **0.995** | **0.995** | **0.995** | **1.000** |
| **Transformer** | **0.995** | **0.991** | **0.995** | **0.993** | **0.999** |
| SVM | 0.990 | 0.982 | 0.989 | 0.986 | 1.000 |

> 🔥 *The near‑perfect scores are a direct result of the proxy label rule. Run with real labels to face a far more challenging (and scientifically interesting) classification problem.*

---

## 🎥 **“Watch It Learn” – Visualizations**

Every neural network run produces these diagnostic graphics inside `reports/figures/`:

| Visualization | Description |
|---------------|-------------|
| `<model>_training_curves.png` | Training & validation loss/accuracy over epochs |
| `<model>_embedding_evolution.gif` | **PCA projection of the learned representation, updated every few epochs** – watch red (NOT_PLANET) and green (PLANET) points separate as the model trains |
| `<model>_test_diagnostics.png` | Confusion matrix + ROC curve on held‑out test set |

<p align="center">
  <em>Run with <code>--live</code> on a machine with a display to see the loss curve update in real time!</em>
</p>

---

## 🚀 **Quick Start**

```bash
git clone https://github.com/your-org/kepler-tce-classifier.git
cd kepler-tce-classifier
pip install -r requirements.txt
```

Place the DR25 TCE CSV file(s) into `data/raw/` — the pipeline expects the standard NASA column names.

---

## 🔧 **Complete Pipeline Usage**

### 1. Train Classical ML Models
```bash
python src/train_ml.py
```
Trains Logistic Regression, Random Forest, XGBoost, and SVM with hyperparameter tuning.  
Models saved to `models/ml/*.joblib`.

### 2. Train the Deep Neural Network (MLP)
```bash
python src/train_nn.py --model mlp --epochs 30
```
A fully‑connected network with ReLU, dropout, and batch normalisation.

### 3. Train the Transformer (Tabular Self‑Attention)
```bash
# Default (GPU or multi‑core CPU)
python src/train_nn.py --model transformer --epochs 30

# Lightweight config for a weak CPU (finishes in ~6 minutes)
python src/train_nn.py --model transformer --epochs 10 \
    --batch_size 2048 --d_model 16 --n_heads 2 --n_layers 1 --patience 4
```

Both neural models save checkpoints, config, and preprocessor to `models/dl/`.

### 4. Evaluate All Models on the Held‑Out Test Set
```bash
python src/evaluate.py
```
Produces `reports/model_comparison.csv` and every diagnostic plot.

### 5. Generate a Narrative Report (Optional)
```bash
python src/langchain_report.py
```
If an **Anthropic** or **OpenAI** API key is set, a large language model writes a detailed summary; otherwise a robust template is used.  
Output: `reports/summary_report.txt`.

### 6. Predict on New TCE Data
```bash
python src/predict.py --model random_forest --input new_tces.csv
python src/predict.py --model mlp --input new_tces.csv
python src/predict.py --model transformer --input new_tces.csv
```
The script automatically loads the correct preprocessor and guarantees identical transformations.

---

## 📐 **Extending the Project**

- **Real scientific labels** – run on a machine with internet access; Option A activates automatically.
- **3‑class problem** – the code currently collapses `CONFIRMED` + `CANDIDATE` → `PLANET`; uncomment the 3‑class mapping in `data_loader.py` for a finer‑grained task.
- **Explainability with SHAP** – integrate `shap` to understand feature importance; strongly recommended before trusting any model.
- **Larger Transformer** – scale up `d_model`, `n_layers`, and `n_heads` for more capacity; the architecture in `models_pytorch.py` supports it out of the box.
- **Hyperparameter optimisation** – replace the simple grid search with Optuna or Ray Tune for the neural networks.

---

## 🐛 **A Real‑World Bug That Taught Us a Lesson**

> An earlier version of the `log1p` transformation computed the shift constant *on‑the‑fly* from whatever data it was given at the moment.  
> This worked beautifully on the full training set, but when `predict.py` called the same preprocessing on a tiny batch of new rows, the minimum was different → **a different shift constant → completely wrong predictions**.  
> Random Forest and MLP gave contradictory results on the same 20 samples until this silent bug was uncovered.

**Fix:** All shift constants are now **hardcoded** in `preprocessing.py` (`LOG1P_SHIFTS`), computed once from the full training distribution. This guarantees identical transformations at training and inference time, forever.  
*If you retrain on radically different data, just double‑check that those constants still keep all values ≥ 0 before the log.*

---

## 📦 **Dependencies**

- Python 3.9+
- `numpy`, `pandas`, `scikit‑learn`
- `xgboost`
- `torch` (PyTorch 2.0+)
- `matplotlib`, `seaborn`, `imageio` (for GIF creation)
- `langchain` (optional, for LLM report)

All pinned versions are in `requirements.txt`.

---

## 📄 **License**

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>Built from the master prompt in <code>MASTER_PROMPT.md</code>.</b><br>
  <i>For the full specification, original rationale, and label‑mapping details, see that file.</i>
</p>

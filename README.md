
python src/predict.py --model random_forest --input new_tces.csv
python src/predict.py --model mlp --input new_tces.csv
python src/predict.py --model transformer --input new_tces.csv
```

## A bug that was caught and fixed while building this

An earlier version of `preprocessing.py`'s `log1p` step computed its shift
constant (to handle a couple of columns with small negative values) from
`df[col].min()` **on whatever DataFrame was passed in** — correct on the
full 34k-row training set, but silently wrong when `predict.py` later called
it on a tiny batch of new rows (a different min → a different shift → a
different transformed value → wrong predictions). Random Forest and MLP
predictions on the same 20 sample rows disagreed sharply until this was
found. Fixed by hardcoding the shift constants (`LOG1P_SHIFTS` in
`preprocessing.py`) from the full training distribution, so the transform is
now identical at train time and inference time. If you retrain on very
different data, double check those shift constants still make sense (they
just need to keep every value ≥ 0 before `log1p`).

## Extending this

- **Real labels:** run on a machine with internet access to the NASA
  Exoplanet Archive to get real `koi_disposition` labels via Option A.
- **3-class problem:** `data_loader.py`'s `build_labels_from_koi_merge`
  currently collapses to a 2-class `PLANET`/`NOT_PLANET` target; switch to
  the 3-class `CONFIRMED`/`CANDIDATE`/`FALSE_POSITIVE_OR_NTP` mapping
  (commented in the master prompt) if you want the finer-grained task.
- **Feature importance / SHAP:** the langchain report's auto-generated
  "next step" suggests this — worth doing before trusting any model,
  especially given the proxy-label caveat above.
- **Bigger transformer:** if you have more compute, bump `d_model`/`n_layers`
  back up to the defaults (32 / 2) or beyond — the architecture in
  `models_pytorch.py` scales cleanly.

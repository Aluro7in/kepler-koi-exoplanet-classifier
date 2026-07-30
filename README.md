
- **Feature importance / SHAP:** the langchain report's auto-generated
  "next step" suggests this — worth doing before trusting any model,
  especially given the proxy-label caveat above.
- **Bigger transformer:** if you have more compute, bump `d_model`/`n_layers`
  back up to the defaults (32 / 2) or beyond — the architecture in
  `models_pytorch.py` scales cleanly.

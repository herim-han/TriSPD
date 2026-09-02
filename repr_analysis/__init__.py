"""Representational analysis of the trimodal SPMM embedding space.

Library modules
    config   build_config
    data     build_tokenizer, make_loader
    models   load_model
    extract  pooled and layer-wise [CLS] feature extraction
    metrics  linear_cka, cross_pred_r2
    effrank  effective_rank_metrics

Entry point (run from the repo root)
    python -m repr_analysis.run_bi_vs_tri
"""

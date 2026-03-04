"""train_xgb.py — CLI wrapper for the cWB XGBoost training workflow.

All heavy logic lives in ``pycwb.modules.cwb_xgboost``:

    load_flat_parquet            read BKG + SIM Parquet → combined DataFrame
    build_features_from_parquet  xgb_config + preprocess_events + cuts
    make_train_test_split        stratified 90/10 split
    train_classifier             XGBoost 2.x-compatible fit with early stopping
    evaluate_classifier          ROC-AUC, report, feature importance, optional plot

Model is saved with XGBoost's native format (default: .ubj = Universal Binary JSON).
This is more portable and version-stable than pickle.  Load with::

    import xgboost as xgb
    clf = xgb.XGBClassifier()
    clf.load_model("xgb_model.ubj")

Usage::

    python train_xgb.py [--search blf] [--nifo 2]
                        [--bkg bkg_xgb.parquet] [--sim sim_xgb.parquet]
                        [--model xgb_model.ubj] [--dump] [--verbose]
                        [--nthread N] [--device cuda]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# ── use local pyBurst source ───────────────────────────────────────────────────
sys.path.insert(0, "/Users/yumengxu/Project/Physics/cwb/pyBurst")

from pycwb.modules.cwb_xgboost import (
    load_flat_parquet,
    build_features_from_parquet,
    make_train_test_split,
    train_classifier,
    evaluate_classifier,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="cWB XGBoost training script")
    parser.add_argument("--search", default="blf",
                        choices=["blf", "bhf", "bld", "bbh", "imbhb"])
    parser.add_argument("--nifo",    type=int, default=2)
    parser.add_argument("--bkg",     default="bkg_xgb.parquet")
    parser.add_argument("--sim",     default="sim_xgb.parquet")
    parser.add_argument("--model",   default="xgb_model.ubj",
                        help="Output model path (.ubj = binary JSON, .json = text JSON)")
    parser.add_argument("--dump",    action="store_true",
                        help="Save diagnostic plots alongside the model")
    parser.add_argument("--verbose", action="store_true",
                        help="Print XGBoost per-round metrics")
    parser.add_argument("--nthread", type=int, default=-1,
                        help="CPU threads for XGBoost (-1 = all cores, default)")
    parser.add_argument("--device",  default=None,
                        help="XGBoost device string: 'cpu' (default) or 'cuda' / 'cuda:0'")
    parser.add_argument("--config",  default=None,
                        help="Path to a Python override config file defining update_config()")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  cWB XGBoost training  —  search={args.search}  nifo={args.nifo}")
    print(f"  pycwb source : {sys.path[0]}")
    print(f"{'='*60}\n")

    # 1. load
    df = load_flat_parquet(args.bkg, args.sim)

    # 2. preprocess
    events, xgb_params, ML_list, ML_caps, _ = build_features_from_parquet(
        df, args.nifo, args.search, config_file=args.config
    )

    # 3. split
    seed = xgb_params["seed"]
    X_train, X_test, y_train, y_test = make_train_test_split(
        events, ML_list, ML_caps, seed
    )

    # 4. train
    clf = train_classifier(X_train, y_train, X_test, y_test,
                           xgb_params, args.verbose,
                           nthread=args.nthread, device=args.device)

    # 5. evaluate
    model_stem = os.path.splitext(args.model)[0]
    evaluate_classifier(clf, X_test, y_test, args.dump, model_stem)

    # 6. save  — use XGBoost native format (.ubj or .json) instead of pickle;
    #    portable across Python/XGBoost versions and cross-language readable.
    os.makedirs(os.path.dirname(os.path.abspath(args.model)), exist_ok=True)
    clf.save_model(args.model)
    size_kb = os.path.getsize(args.model) / 1024
    log.info("Model saved → %s  (%.1f KB)", args.model, size_kb)
    log.info("Feature names: %s", clf.get_booster().feature_names)

    print("\nDone.")


if __name__ == "__main__":
    main()

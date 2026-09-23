"""Persist training diagnostics for the postproduction report."""

import json
from pathlib import Path

from pycwb.post_production.action_spec import action_spec


@action_spec(
    inputs=["model_file", "reference_model_file"],
    outputs=["output_dir"],
    description="Plot saved XGBoost learning curves and feature importance",
)
def training_diagnostics(
    work_dir, model_file, output_dir, reference_model_file=None, **kwargs
):
    import matplotlib.pyplot as plt
    import xgboost as xgb

    model = xgb.XGBClassifier()
    model.load_model(Path(work_dir) / model_file)
    history = model.evals_result()
    importance = model.get_booster().get_score(importance_type="gain")
    out = Path(work_dir) / output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "best_iteration": model.best_iteration,
        "best_score": float(model.best_score),
        "history": history,
        "feature_gain": importance,
    }
    if reference_model_file:
        reference = xgb.XGBClassifier()
        reference.load_model(Path(work_dir) / reference_model_file)
        comparisons = [
            ("Training evaluation history", history == reference.evals_result()),
            (
                "Trained tree structure and values",
                model.get_booster().get_dump() == reference.get_booster().get_dump(),
            ),
            ("Selected iteration", model.best_iteration == reference.best_iteration),
        ]
        summary["checks"] = [
            {
                "check": name,
                "status": "PASS" if equal else "FAIL",
                "details": "Exact equality" if equal else "Models differ",
            }
            for name, equal in comparisons
        ]
    (out / "training.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    if any(check["status"] == "FAIL" for check in summary.get("checks", [])):
        raise ValueError(f"Training comparison failed; see {out / 'training.json'}")
    fig, ax = plt.subplots(figsize=(8, 4))
    for dataset, metrics in history.items():
        for metric, values in metrics.items():
            ax.plot(values, label=f"{dataset}: {metric}")
    ax.axvline(model.best_iteration, color="grey", ls="--", label="Selected iteration")
    ax.set_xlabel("Boosting iteration")
    ax.set_ylabel("Metric")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "learning.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    items = sorted(importance.items(), key=lambda x: x[1])[-20:]
    ax.barh([k for k, v in items], [v for k, v in items])
    ax.set_xlabel("Mean split gain")
    fig.tight_layout()
    fig.savefig(out / "importance.png", dpi=130)
    plt.close(fig)
    return {
        "summary_file": str(out / "training.json"),
        "best_iteration": model.best_iteration,
        "learning_plot": str(out / "learning.png"),
        "importance_plot": str(out / "importance.png"),
    }

"""Evaluate configured prediction cuts on cWB-compatible feature columns."""

import re

import numpy as np
import pandas as pd


def prediction_mask(frame, expression):
    """Support pandas expressions and common ROOT scalar/array cut syntax.

    ROOT rho[0] means the stored statistic (rho0_std), whereas bare rho0
    means the configured derived feature. Unsupported expressions fail.
    """
    if not expression:
        return pd.Series(True, index=frame.index)
    text = str(expression)
    text = re.sub(r"\((?:float|double)\)", "", text)
    for root, python in [
        ("TMath::Max", "@maximum"),
        ("TMath::Min", "@minimum"),
        ("TMath::Abs", "abs"),
        ("TMath::Sqrt", "sqrt"),
        ("TMath::Log10", "log10"),
    ]:
        text = text.replace(root, python)

    # Training cuts also accept bare, elementwise max/min functions.
    # Match calls only, preserving column names and qualified functions.
    text = re.sub(r"(?<![\w.@])max\s*\(", "@maximum(", text)
    text = re.sub(r"(?<![\w.@])min\s*\(", "@minimum(", text)

    def column(match):
        base, index = match.groups()
        return "rho0_std" if base == "rho" and index == "0" else base + index

    text = re.sub(r"\b([A-Za-z_]\w*)\[(\d+)\]", column, text)
    text = text.replace("&&", " and ").replace("||", " or ")
    text = re.sub(r"!(?!=)", " not ", text).strip()
    values = frame.copy()
    for name in values.select_dtypes(include=["floating"]).columns:
        values[name] = values[name].astype("float64")
    try:
        mask = values.eval(
            text,
            engine="python",
            local_dict={"maximum": np.maximum, "minimum": np.minimum},
        )
    except Exception as exc:
        raise ValueError(f"Unsupported prediction cut {expression!r}: {exc}") from exc
    if isinstance(mask, (bool, np.bool_)):
        mask = pd.Series(bool(mask), index=frame.index)
    if not isinstance(mask, pd.Series) or mask.dtype != bool:
        raise ValueError("Prediction cut must return a boolean mask")
    return mask.fillna(False)

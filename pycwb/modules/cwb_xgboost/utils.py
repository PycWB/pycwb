import pickle
import xgboost as xgb


def getcapname(feature,cap):
   """
   function purpose:
      return formatted cap name

   input params:
      feature: feature name (eg: rho0, Qa, Qp)
      cap:     cap value (eg: 8)

   return:
      formatted cap name (eg: rho0,8 -> rho0_8d0)
      only the first digit is used: (eg. rho0,8.12 -> rho_8d1)
   """

   return (feature+'_'+str("{:.1f}".format(cap))).replace(".","d")


def load_model(model_file: str) -> xgb.XGBClassifier:
    """Load an XGBoost classifier from a file.

    Supports three formats, selected by file extension:

    * ``.ubj``  — XGBoost native Universal Binary JSON (recommended; compact
      and portable across Python/XGBoost versions).
    * ``.json`` — XGBoost native text JSON (human-readable, equally portable).
      Falls back to pickle for legacy files that happen to have a ``.json``
      extension.
    * ``.pkl`` / ``.pickle`` / any other extension — pickle (legacy; avoid for
      new models because it breaks across Python and XGBoost versions).

    Parameters
    ----------
    model_file : str
        Path to the saved model file.

    Returns
    -------
    xgb.XGBClassifier
        The loaded classifier.
    """
    ext = model_file.rsplit(".", 1)[-1].lower()

    if ext == "ubj":
        # XGBoost native binary JSON — preferred format
        model = xgb.XGBClassifier()
        model.load_model(model_file)
    elif ext == "json":
        # XGBoost native text JSON, but fall back to pickle for legacy files
        try:
            model = xgb.XGBClassifier()
            model.load_model(model_file)
        except Exception:
            with open(model_file, "rb") as fh:
                model = pickle.load(fh)
    else:
        # .pkl / .pickle or unknown extension — assume pickle
        with open(model_file, "rb") as fh:
            model = pickle.load(fh)

    return model


def read_json_to_pandas(catalog_file: str):
    """
    Read the json file into a pandas DataFrame.

    Parameters
    ----------
    catalog_file : str
        The path to the json file.

    Returns
    -------
    df : DataFrame
        The DataFrame read from the json file.

    """
    import pandas as pd
    df = pd.read_json(catalog_file)

    # format conversion
    
    return df
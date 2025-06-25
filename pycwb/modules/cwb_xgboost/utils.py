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


def load_model(model_file: str) -> xgb.Booster:
    """
    Load the model from the file.

    Parameters
    ----------
    model_file : str
        The path to the model file.

    Returns
    -------
    model : xgb.Booster
        The loaded model.

    """
    ext = model_file.split('.')[-1]
    if ext == 'json':
        # handle fake json file which is a pickle file
        try:
            model = pickle.load(open(model_file, 'rb'))
        except Exception as e:
            # try to load as xgboost json model
            model = xgb.XGBClassifier()
            model.load_model(fname=model_file)
    elif ext == 'ujs':
        model = xgb.XGBClassifier()
        model.load_model(fname=model_file)
    
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
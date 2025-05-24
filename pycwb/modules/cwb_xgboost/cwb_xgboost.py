from pandas import DataFrame
from pycwb.utils.module import import_function_from_file
from .config import xgb_config
from .prediction import predict
from .utils import load_model
from .read_data import preprocess_events

def xgb_predict(events: DataFrame, config_file: str, model_file: str, search: str, nifo: int):
    """
    Calculate the ranking statistics for the events with given model and configuration.
    The ranking statistics are added to the DataFrame events as new columns as defined in config['ML_options']['ranking_statistics'].
    
    Parameters
    ----------
    events : DataFrame
        The events to be ranked. It is called xpd in cwb code
    config_file : str
        The path to the configuration file.
    model_file : str
        The path to the model file.
    search : str
        The search method to be used. bbh/imbhb/blf/bhf/bld
    nifo : int
        Number of interferometers
    """

    xgb_params, ML_list, ML_caps, ML_balance, ML_options = xgb_config(search, nifo)

    # load the update_config function and add_ranking_statistics function from the configuration file
    update_config = import_function_from_file(config_file, 'update_config')
    add_ranking_statistics = import_function_from_file(config_file, 'add_ranking_statistics')

    update_config(xgb_params, ML_list, ML_caps, ML_balance, ML_options)

    # load the model
    model = load_model(model_file)

    # preprocess the events DataFrame
    events = preprocess_events(events, nifo, ML_options, ML_caps)
    
    # predict the ranking statistics
    predict(events, model, xgb_params, ML_list, add_ranking_statistics)
    
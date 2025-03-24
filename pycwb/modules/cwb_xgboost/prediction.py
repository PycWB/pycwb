import logging
import xgboost as xgb
from pandas import DataFrame
logger = logging.getLogger(__name__)


def predict(events: DataFrame, model: xgb.Booster, config: dict, verbose: bool = False):
    """
    Calculate the ranking statistics for the events with given model and configuration. 
    The ranking statistics are added to the DataFrame events as new columns as defined in config['ML_options']['ranking_statistics'].

    Parameters
    ----------
    events : DataFrame
        The events to be ranked. It is called xpd in cwb code
    model : xgb.Booster
        The model used for ranking. It is called XGB_clf in cwb code
    config : dict
        The configuration for the ranking.
    verbose : bool
        Whether to print the debug information.
    """
    ML_options = config['ML_options']

    # load the function for calculating the ranking statistics
    # check if the function is defined in the file
    # config_ranking_statistics = ML_options.get('ranking_statistics')
    # if config_ranking_statistics is None:
    #     raise ValueError("The configuration does not contain the key 'ranking_statistics'.")
    # config_ranking_statistics_file = config_ranking_statistics.get('file')
    # config_ranking_statistics_function = config_ranking_statistics.get('function')
    # if config_ranking_statistics_file is None or config_ranking_statistics_function is None:
    #     raise ValueError("The configuration does not contain the key 'file' or 'function' in 'ranking_statistics'.")

    # func_add_ranking_statistics = import_function_from_file(config_ranking_statistics_file, config_ranking_statistics_function)
    add_ranking_statistics = config['add_ranking_statistics']

    # set the number of threads for prediction
    model._Booster.set_param('nthread', ML_options['nthread(prediction)'])
    logger.info(model.get_xgb_params())

    # get the list of cWB parameters used in training
    ML_list = model.get_booster().feature_names
    logger.info("ML xgb parameters:\n   " + str(ML_list))

    # calculate MLstat (p_xgb) ... output from XGBoost

    events['MLstat'] = model.predict_proba(events[ML_list], iteration_range=None)[:,1]

    # calculate ranking statistics
    add_ranking_statistics(events)

    if(verbose): 
        columnsNamesArr = events.columns.values
        listOfColumnNames = list(columnsNamesArr)
        print("List Of Column Names" , listOfColumnNames, sep='\n')
        print( events )


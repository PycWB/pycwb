import numpy as np
import orjson
import pandas as pd
import copy

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


def preprocess_events(events: pd.DataFrame, nifo: int, ML_options: dict, ML_caps: dict):
    """
    Preprocess the events DataFrame by extracting features and applying caps.
    Parameters
    ----------
    events : pd.DataFrame
        The DataFrame containing the events.
    nifo : int
        Number of interferometers.
    ML_options : dict
        Dictionary containing options for reading the events.
    ML_caps : dict
        Dictionary containing the caps for the features.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with additional features and caps applied.
    """
    # with open(catalog_file, 'rb') as f:
    #     catalog = orjson.loads(f.read())

    # catalog['events']

    # Convert catalog to DataFrame
    # events = pd.DataFrame(catalog['events'])
    # nifo = len(catalog['config']['ifo'])

    xvars=copy.deepcopy(ML_options['readfile(vars)']) 
    # add netcc[0]  as netcc0
    for n in range(nifo):
        events['rho'+str(n)] = events['rho'].apply(lambda x: x[n] if isinstance(x, list) and len(x) > n else None)
        events['sSNR'+str(n)] = events['sSNR'].apply(lambda x: x[n] if isinstance(x, list) and len(x) > n else None)
        events['duration'+str(n)] = events['duration'].apply(lambda x: x[n] if isinstance(x, list) and len(x) > n else None)
        events['bandwidth'+str(n)] = events['bandwidth'].apply(lambda x: x[n] if isinstance(x, list) and len(x) > n else None)
        events['netcc'+str(n)] = events['netcc'].apply(lambda x: x[n] if isinstance(x, list) and len(x) > n else None)
        events['noise'+str(n)] = events['noise'].apply(lambda x: x[n] if isinstance(x, list) and len(x) > n else None)

    # add chunk = 1
    events['chunk'] = 0
    if 'ecor' in xvars and 'likelihood' in xvars:
        events['ecor/likelihood'] = events['ecor']/events['likelihood']
    if 'duration' in xvars and 'bandwidth' in xvars:
        events['duration0*bandwidth0'] = events['duration0']*events['bandwidth0']
        events['bandwidth0/duration0'] = events['bandwidth0']/events['duration0']
    if 'sSNR' in xvars and 'likelihood' in xvars:
        events['mSNR'] = np.minimum(events['sSNR0'],events['sSNR1'])
        for n in range(2,nifo): events['mSNR'] = np.minimum(events['mSNR'],events['sSNR'+str(n)])
        events['mSNR/likelihood'] = events['mSNR']/events['likelihood']
        for n in range(0,nifo): events['sSNR'+str(n)+'/likelihood'] = events['sSNR'+str(n)]/events['likelihood']

    if 'noise' in xvars:
        events['noise'] = 1/(events['noise0']*events['noise0'])
        for n in range(1,nifo): events['noise'] = events['noise']+1/(events['noise'+str(n)]*events['noise'+str(n)])
        events['noise'] = np.sqrt(1/events['noise'])

    events['Qa']=np.sqrt(events['qveto'])

    # check ML_options
    for option, value in ML_options.items():
        if(option=='Qp(index)'):
            # add Qp feature to xpd
            if (value!=1):
                print('\nError: ML_options(''Qp(index)'') must be 1, only one Qp feature is supported\n')
                exit(1)
            events['Qp'] = events['qfactor']/(2*np.sqrt(np.log10(np.minimum(200,events['ecor']))))
        if(option=='rho0(define)'):
            # select rho0 definition
            if (value!=0) and (value!=1):
                print('\nError: ML_options(''rho0(define)'') must be 0(std) or 1(new)\n')
                exit(1)
            if (value==0):	# standard -> rho0 = rho[0]
                events['rho0'] = events['rho0_std']
            if (value==1):   # new -> rho0 = sqrt(ecor/(1+penalty*max(1,penalty)-1)
                events['rho0'] = np.sqrt(events['ecor']/(1+events['penalty']*(np.maximum(1,events['penalty'])-1)))

    for feature, cap in ML_caps.items():
        if(cap>0):
            feature_cap = getcapname(feature,cap)
            events[feature_cap] = events[feature]
            events.loc[events[feature_cap]>cap, feature_cap] = cap

    return events

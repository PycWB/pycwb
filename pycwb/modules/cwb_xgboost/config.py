from .utils import getcapname

def xgb_config(search, nifo):

  if(search!='blf') and (search!='bhf') and (search!='bld') and (search!='bbh') and (search!='imbhb'):
    print('\nxgb_config - wrong search type, available options are: bbh/imbhb/blf/bhf/bld\n')
    exit(1)

  # -----------------------------------------------------
  # definitions
  # -----------------------------------------------------

  # new definition of rho0
  xrho0 = 'sqrt(ecor/(1+penalty*(TMath::Max((float)1,(float)penalty)-1)))'

  # -----------------------------------------------------
  # XGBoost hyper-parameters - (tuning/training)
  # -----------------------------------------------------

  """
  from https://xgboost.readthedocs.io/_/downloads/en/release_1.4.0/pdf/

  learning_rate(eta)    range: [0,1]
                        Step size shrinkage used in update to prevents overfitting. After each boosting step,
                        we can directly get the weights of new features, and eta shrinks the feature weights to
                        make the boosting process more conservative.

  max_depth
                        Maximum depth of a tree. Increasing this value will make the model more complex and more likely tooverfit.
                        0 is only accepted inlossguidedgrowing policy when tree_method is set as histi or gpu_hist and it indicates no
                        limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree

  min_child_weight      range: [0,inf]
                        Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a
                        leaf node with the sum of instance weight less than min_child_weight, then the building process will
                        giveup further partitioning. In linear regression task, this simply corresponds to minimum number
                        of instances needed to be in each node. The larger min_child_weight is, the more conservative the algorithm will be.

  colsample_bytree      range: [0,1]
                        Is the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.

  subsample             range: [0,1]
                        Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half
                        of the training data prior to growing trees. and this will prevent overfitting.
                        Subsampling will occur once in every boosting iteration.

  gamma                 range: [0,inf]
                        Minimum loss reduction required to make a further partition on a leaf node of the tree.
                        The larger gamma is, the more conservative the algorithm will be.
  """

  if((search=='blf') or (search=='bhf') or (search=='bld')):
    learning_rate    = 0.03
    max_depth        = 6
    min_child_weight = 5.0
    colsample_bytree = 1.0
    subsample        = 0.6
    gamma            = 2.0
  elif(search=='bbh') or (search=='imbhb'):
    learning_rate    = 0.03
    max_depth        = 13
    min_child_weight = 10.0
    colsample_bytree = 1.0
    subsample        = 0.6
    gamma            = 2.0

  xgb_params = {
      		'objective': 		'binary:logistic',
      		'tree_method': 		'hist',
      		'grow_policy': 		'lossguide',
      		'n_estimators': 	20000,
      		'seed': 		150914,
       		'use_label_encoder': 	False,
      		'scale_pos_weight':  	1.0,
      		'learning_rate':    	float(learning_rate),
      		'max_depth':           	int(max_depth),
      		'min_child_weight':  	float(min_child_weight),
      		'colsample_bytree':  	float(colsample_bytree),
      		'subsample':         	float(subsample),
      		'gamma':             	float(gamma),
      		'nthread': 		1
  }

  # -----------------------------------------------------
  # XGBoost ML_list - (tuning/training/prediction)
  # -----------------------------------------------------

  """
  List of features used by XGBoost.

  Note: this list is used in tuning/training stages while in the predictions stage
        the list of features is obtained from the trained XGB model
  """

  if((search=='blf') or (search=='bhf') or (search=='bld')):
    ML_list=[
             'norm', 'netcc0', 'penalty',
             'Lveto2', 'chunk'
            ]
  elif(search=='bbh') or (search=='imbhb'):
    ML_list=[
             'norm', 'netcc0', 'penalty',
             'frequency0', 'bandwidth0', 'duration0', 
             'Lveto2', 'chirp1', 'chirp3',
             'chunk'
            ]

  # add sSNRX/likelihood features to ML_list
  for n in range(0,nifo-1): ML_list.append('sSNR'+str(n)+'/likelihood')

  # -----------------------------------------------------
  # XGBoost ML_caps - (tuning/training/prediction)
  # -----------------------------------------------------

  """
  List of caps values used to constrain the upper value of the features used in the computation of the XGB model.
  This option is used for values where the statistic is lower, for example in the rho0 tail distribution.
  Eg: ML_caps['rho0']=11  -> for rho0 > 11 then rho0=11
  """

  ML_caps = {}
  if((search=='blf') or (search=='bhf') or (search=='bld')):
    ML_caps['rho0'] =  20
    ML_caps['Qa']   =  0
    ML_caps['Qp']   =  0
  elif(search=='bbh') or (search=='imbhb'):
    ML_caps['rho0'] =  11
    ML_caps['Qa']   =  0
    ML_caps['Qp']   =  0

  # add ML_cups features to ML_list
  for feature, cap in ML_caps.items():
    if(cap>0):
      ML_list.append(getcapname(feature,cap))
    elif(cap==0):
      ML_list.append(feature)

  # -----------------------------------------------------
  # XGBoost ML_balance - (tuning/training)
  # -----------------------------------------------------

  """
  List of options used to balance sim/bkg rho0 feature.
  Eg: ML_caps['rho0']=8  -> (bulk = rho0<=8) and (tail = rho0>8)

  tail(tuning/training)   True/False -> enable/disable balance in the tail (tuning/training)
                          if enabled then the number of sim samples are downsampled to the number of bkg sample

  bulk(tuning/training)   True/False -> enable/disable balance in the bulk (tuning/training)

  eval(tuning)            [0,100] (default=0) (tuning
                          if eval=0 than the balance in the tail for X_eval data set is applied as for X_train
                          if eval>0 than eval is the data set with rho0<rho0_perc where rho0_perc is rho0 for which
                                    the percentile of bkg events with rho0<rho0_perc is equal to eval
                          This option is mandatory to evaluate correctly the aucelf score when the caps_rho8 parameter is tuned 
                          (eval>0) -> every trial is performed with the same number of sim events

  - bulk balance has the following options:

    binning(tuning/training)   is the way the binning is maded:
                               bkg(percentiles)       each bin contains the same number of bkg events
                               sim(percentiles)	      each bin contains the same number of sim events
                               linear                 each bin has the same width = (rho0_capvalue-rho0_min)/nbins

    nbins(tuning/training)     number of bins in the range [rho0_min,rho0_capvalue]

    smoothing(tuning/training) True/False, enable/disable the smoothing of the weigths curve used for balance

    slope/balance(tuning/training) rescale the weigths curve -> exp(log(A)*pow(1-i/(nbins-1),q)) where q=slope, A=balance, i=bin_index

    cuts(tuning/training)      data cuts, same syntax used to cut root data tree. Eg: 'rho[0]>7&&netcc[0]>0.7'
  """

  ML_balance = {}

  # tuning
  ML_balance['tail(tuning)']        =  True
  ML_balance['bulk(tuning)']        =  True
  ML_balance['eval(tuning)']        =  0
  ML_balance['binning(tuning)']     =  'sim(percentiles)'
  ML_balance['smoothing(tuning)']   =  False
  ML_balance['nbins(tuning)']       =  100

  # training
  ML_balance['tail(training)']      =  True
  ML_balance['bulk(training)']      =  True
  ML_balance['binning(training)']   =  'sim(percentiles)'
  ML_balance['smoothing(training)'] =  False
  ML_balance['nbins(training)']     =  100

  if((search=='blf') or (search=='bhf') or (search=='bld')):
    # tuning
    ML_balance['cuts(tuning)']        =  'rho0>6.5'
    ML_balance['slope(tuning)']       =  'q=6'
    ML_balance['balance(tuning)']     =  'A=20'
    # training
    ML_balance['cuts(training)']      =  'rho0>6.5'
    ML_balance['slope(training)']     =  'q=6'
    ML_balance['balance(training)']   =  'A=20'
  elif(search=='bbh') or (search=='imbhb'):
    # tuning
    ML_balance['cuts(tuning)']        =  'rho0>6.5'
    ML_balance['slope(tuning)']       =  'q=1'
    ML_balance['balance(tuning)']     =  'A=14'
    # training
    ML_balance['cuts(training)']      =  'rho0>6.5'
    ML_balance['slope(training)']     =  'q=1'
    ML_balance['balance(training)']   =  'A=14'

  # -----------------------------------------------------
  # XGBoost ML_options
  # -----------------------------------------------------

  """
  List of general purposes options

  Qp(index)    select the index of Qveto array to be used for the Qp feature
               available values are 1/2/3, the indexes 2/3 are availables only with the updated Qveto plugin.
               default -> Qp uses Qveto[1]

  rho0(define) select rho0 definition:
               0 -> rho0 = rho[0]
               1 -> rho0 = sqrt(ecor/(1+penalty*max(1,penalty)-1)

  *(mplot1d)   plot options used by the mplot1d(cwb_xgboost.py) function: used in training
               sim(mplot1d)
                   color         sim hist1d color
               bkg(mplot1d)
                   color         bkg hist1d color

  *(mplot2d)   plot options used by the mplot2d(cwb_xgboost.py) function: used in training
               sim(mplot2d)
                   cmap          sim hist2d colormap: eg = rainbow/viridis/Blues/Reds/YnGn
                                 (https://matplotlib.org/stable/tutorials/colors/colormaps.html)
               bkg(mplot2d)
                   rho_name      rho name (default = 'rho0') used to select the background entries (rho_name>rho_thr)
                   rho_thr       rho threshold (default = 8) used to select the background entries (rho_name>rho_thr)
                   rho_label     rho label (default = 'rho0') used in the legend
                   marker_color  marker color (default = 'black') used for the bkg scatter plot
                   marker_size   marker size (default = 22) used for the bkg scatter plot

  *(mplot1d/2d)plot options used by the mplot1d/2d(cwb_xgboost.py) function: used in training

               feature(mplot1d/2d) or feature(mplot*d)
                   enable        True/False -> enable/disable plot
                   xinf/xsup     sim hist2d xpar inf/sup limits
                   yinf/ysup     sim hist2d ypar inf/sup limits
                   xbinw/ybinw   sim hist2d x/y bin widths

  cuts(prediction) data cuts (prediction), same syntax used to cut root data tree. Eg: 'rho[0]>7&&netcc[0]>0.7'
  save_MLstat(prediction) save MLstat into rho[2] root file (prediction), default is True'
  nthread(prediction) number of threads used in calculating MLstat (prediction), default is 1'
  """

  ML_options = {}
  ML_options['Qp(index)']                    =  1	# (tuning/training/prediction/report)
  ML_options['rho0(define)']                 =  1	# (tuning/training/prediction/report)

  # save MLstat into rho[2] output root file (prediction)
  ML_options['save_MLstat(prediction)'] = True

  # split_loops is used in prediction stage
  # the split_loops>1 case is used to reduce memory requirements
  # Memory usage should be reduced by increasing split_loops value
  ML_options['split_loops(prediction)'] = 1

  # enabled/disabled(def) definition of features used for targeted searches
  ML_options['targeted'] = False

  # number of threads used to compute MLstat (prediction)
  ML_options['nthread(prediction)'] = 1

  # data cuts prediction
  if((search=='blf') or (search=='bhf') or (search=='bld')):
    ML_options['cuts(prediction)']           =  xrho0+'>7.2'
  elif(search=='bbh') or (search=='imbhb'):
    ML_options['cuts(prediction)']           =  xrho0+'>6.5'

  # random seed used in readfile function (cwb_xgboost.py)
  ML_options['readfile(seed)'] = 150914

  # user code definitions
  ML_options['readfile(ucode)'] = None

  # parameters loaded in readfile function (cwb_xgboost.py)
  if((search=='blf') or (search=='bhf') or (search=='bld')):
    ML_options['readfile(vars)']             = [
                                                "rho", "norm", "netcc", "Qveto",
                                                "penalty", "ecor", "likelihood", "sSNR", "noise"
                                               ]
  elif(search=='bbh') or (search=='imbhb'):
    ML_options['readfile(vars)']             = [
                                                "rho", "norm", "netcc", "Qveto",
                                                "chirp", "duration", "bandwidth", "frequency",
                                                "penalty", "ecor", "likelihood", "sSNR", "noise"
                                               ]

  # mplot1d options
  ML_options['bkg(mplot1d)'] = {'color':'blue'}
  ML_options['sim(mplot1d)'] = {'color':'green'}

  # mplot2d options
  ML_options['bkg(mplot2d)'] = {'rho_name':'rho0','rho_thr':ML_caps['rho0'],'rho_label':'rho0','marker_color':'royalblue','marker_size':100}
  ML_options['sim(mplot2d)'] = {'cmap':'Reds'}

  # mplot1d/mplot2d options (in order to setup both 1d/2d use mplot*d)
  ML_options['rho0(mplot*d)']       = {'enable':False,	'inf':0,	'sup':20.0,	'bins':100,	'log':True}
  ML_options['norm(mplot*d)']       = {'enable':False,	'inf':1,	'sup':14.0,	'bins':100,	'log':False}
  ML_options['netcc0(mplot*d)']     = {'enable':False,	'inf':0,	'sup':1.0, 	'bins':100,	'log':False}
  ML_options['penalty(mplot*d)']    = {'enable':False,	'inf':0,	'sup':10.0,	'bins':100,	'log':False}
  ML_options['Qa(mplot*d)']         = {'enable':False,	'inf':0,	'sup':6.0,	'bins':100,	'log':False}
  ML_options['Qp(mplot*d)']         = {'enable':False,	'inf':0,	'sup':10.0 ,	'bins':100,	'log':False}
  ML_options['skyDist01(mplot*d)']  = {'enable':False,	'inf':0,	'sup':180.0 ,	'bins':100,	'log':False}
  ML_options['netSens1(mplot*d)']   = {'enable':False,	'inf':0,	'sup':2.0 ,	'bins':100,	'log':False}
  ML_options['erA0(mplot*d)']       = {'enable':False,	'inf':0,	'sup':180.0 ,	'bins':100,	'log':False}

  if(search=='bbh') or (search=='imbhb'):

    ML_options['frequency0(mplot*d)'] = {'enable':False,	'inf':24,	'sup':512.0,	'bins':100,	'log':False}
    ML_options['bandwidth0(mplot*d)'] = {'enable':False,	'inf':0,	'sup':100.0,	'bins':100,	'log':False}
    ML_options['duration0(mplot*d)']  = {'enable':False,	'inf':0,	'sup':10.0,	'bins':100,	'log':False}
    ML_options['chirp1(mplot*d)']     = {'enable':False,	'inf':0,	'sup':100.0,	'bins':100,	'log':False}
    ML_options['chirp3(mplot*d)']     = {'enable':False,	'inf':0,	'sup':100.0,	'bins':100,	'log':False}
    ML_options['chirp4(mplot*d)']     = {'enable':False,	'inf':0,	'sup':100.0,	'bins':100,	'log':False}
    ML_options['chirp5(mplot*d)']     = {'enable':False,	'inf':0,	'sup':100.0,	'bins':100,	'log':False}

  # -----------------------------------------------------

  return xgb_params,ML_list,ML_caps,ML_balance,ML_options


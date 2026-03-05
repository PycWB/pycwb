import logging
import os
import warnings

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LogNorm
import numpy as np

logger = logging.getLogger(__name__)


custom_rc = {
    "text.usetex": True,
    "font.serif": "Times New Roman",
    "font.family": "Serif",
    "xtick.labelsize": 28,
    "ytick.labelsize": 28,
    "legend.fontsize": 22,
    "axes.labelsize": 28,
}


def plot_balance_weight_hist(ofile,X_train,rho0_capname,bins):
    """
    function purpose:
        plot rho0_capname weighted bkg vs sim histograms using the 'bins' binning

    input params:
        X_train:       trained set containig the classifier 0/1
        rho0_capname:  cap name of rho0 used by the histogram (eg: rho0_0d8)
        bins:          binning used by the histogram
        weight:        array of weights

    output params:
        ofile:         output plot file name
    """

    with plt.rc_context(custom_rc):
        print('\nSave plot -> '+ofile+' ...\n')
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111)
        ax.hist(X_train.loc[X_train.classifier == 0, rho0_capname],bins=bins,weights=X_train.loc[X_train.classifier == 0, 'weight1'],alpha=0.5,label='bkg',log=True,color='blue')
        ax.hist(X_train.loc[X_train.classifier == 1, rho0_capname],bins=bins,alpha=0.5,label='sim',log=True,color='green')
        rho0_capname=rho0_capname.replace('_',"\\_")
        ax.set_xlabel(r'$\textrm{'+rho0_capname+'}$')
        ax.set_ylabel(r'$\textrm{Number of events}$')
        ax.legend(loc='upper right')
        plt.title("Background/Simulation Balanced Distributions", fontsize=30, pad=20)
        plt.tight_layout()
        fig.savefig(ofile)
        plt.close()


def plot_balance_hist(ofile,X_train,rho0_capname,bins):
    """
    function purpose:
        plot rho0_capname bkg vs sim histograms using the 'bins' binning

    input params:
        X_train:       trained set containig the classifier 0/1
        rho0_capname:  cap name of rho0 used by the histogram (eg: rho0_0d8)
        bins:          binning used by the histogram

    output params:
        ofile:         output plot file name
    """

    with plt.rc_context(custom_rc):
        print('\nSave plot -> '+ofile+' ...\n')
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111)
        ax.hist(X_train.loc[X_train.classifier == 0, rho0_capname],bins=bins,alpha=0.5,label='bkg',log=True,color='blue')
        ax.hist(X_train.loc[X_train.classifier == 1, rho0_capname],bins=bins,alpha=0.5,label='sim',log=True,color='green')
        rho0_capname=rho0_capname.replace('_',"\\_")
        ax.set_xlabel(r'$\textrm{'+rho0_capname+'}$')
        ax.set_ylabel(r'$\textrm{Number of events}$')
        ax.legend(loc='upper right')
        plt.title("Background/Simulation Distributions", fontsize=30, pad=20)
        plt.tight_layout()
        fig.savefig(ofile)
        plt.close()

def plot_balance_weight(ofile,bins,weight):
    """
    function purpose:
        plot the weights used fot the bulk balance

    input params:
        bins:          binning used by the histograms
        weight:        array of weights

    output params:
        ofile:         output plot file name
    """

    with plt.rc_context(custom_rc):
        print('\nSave plot -> '+ofile+' ...\n')
        fig, ax = plt.subplots()
        bins_x = []
        bins_y = []
        nbins = len(bins)-1
        for i in np.arange(0,nbins):
            # step function
            bins_x.append(bins[i])
            bins_y.append(weight[i])
            bins_x.append(bins[i+1])
            bins_y.append(weight[i])
        bins_x.append(bins[nbins])
        bins_y.append(weight[i-1])
        ax.plot(bins_x,bins_y,marker='',linestyle='-')
        ax.semilogy(bins_x,bins_y)
        ax.set_xlabel(r'$\rho_0$')
        plt.grid()
        rcParams["text.usetex"] = False
        plt.title("Bulk Balance Weights", fontsize=20, pad=20)
        ax.set_ylabel('balance weight ( sim / bkg )')
        plt.tight_layout()
        fig.savefig(ofile)
        plt.close()
        rcParams["text.usetex"] = True


def evaluate_classifier(
    clf,
    X_test,
    y_test,
    dump: bool = False,
    model_stem: str = "model",
) -> dict:
    """Evaluate a trained XGBoost classifier and optionally save diagnostic plots.

    Parameters
    ----------
    clf : xgb.XGBClassifier
        Trained classifier.
    X_test : array-like
        Test feature matrix (numpy array or DataFrame).
    y_test : array-like
        True binary labels.
    dump : bool
        If *True*, save a 3-panel figure ``{model_stem}_eval.png`` with a
        ROC curve, score distribution, and feature-importance bar chart.
    model_stem : str
        Base path/name used for output files (without extension).

    Returns
    -------
    dict
        ``{'auc': float, 'gain': dict}`` — ROC-AUC score and feature-gain dict.
    """
    from sklearn.metrics import roc_auc_score, roc_curve, classification_report

    X_np = np.asarray(X_test, dtype=np.float32)
    y_np = np.asarray(y_test)

    y_prob = clf.predict_proba(X_np)[:, 1]
    y_pred = clf.predict(X_np)

    auc = roc_auc_score(y_np, y_prob)
    print(f"\nROC-AUC : {auc:.5f}")
    print(classification_report(y_np, y_pred, target_names=["BKG", "SIM"]))

    gain = clf.get_booster().get_score(importance_type="gain")
    top = sorted(gain.items(), key=lambda x: -x[1])[:10]
    print("Top-10 features by gain:")
    for feat, g in top:
        print(f"  {feat:<35s}  {g:.2f}")

    if dump:
        try:
            rcParams["text.usetex"] = False
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))

            fpr, tpr, _ = roc_curve(y_np, y_prob)
            axes[0].plot(fpr, tpr, lw=1.5, label=f"AUC={auc:.4f}")
            axes[0].plot([0, 1], [0, 1], "k--", lw=0.7)
            axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC curve")
            axes[0].legend()

            axes[1].hist(y_prob[y_np == 0], bins=50, alpha=0.6, label="BKG", color="royalblue")
            axes[1].hist(y_prob[y_np == 1], bins=50, alpha=0.6, label="SIM", color="tomato")
            axes[1].set(xlabel="XGBoost score", ylabel="Events", title="Score distribution")
            axes[1].legend()

            feat_names = [f for f, _ in top][::-1]
            feat_vals  = [gain[f] for f in feat_names]
            axes[2].barh(feat_names, feat_vals)
            axes[2].set(xlabel="Gain", title="Feature importance (top-10)")

            plt.tight_layout()
            out = f"{model_stem}_eval.png"
            plt.savefig(out, dpi=120)
            plt.close()
            logger.info("Diagnostic plot saved: %s", out)
        except Exception as exc:
            logger.warning("Could not save diagnostic plot: %s", exc)

    return {"auc": auc, "gain": gain}

# ---------------------------------------------------------------------------
# Pre-training diagnostic histograms
# ---------------------------------------------------------------------------

def plot_hist_rho(ofile, bkg, sim, xrho='0'):
    """Plot rho BKG vs SIM overlaid histograms (log-y).

    Parameters
    ----------
    ofile : str
        Output PNG path.
    bkg, sim : pd.DataFrame
        Must contain column ``'rho' + str(xrho)``.
    xrho : str or int
        Suffix identifying the rho column (default ``'0'`` → ``rho0``).
    """
    with plt.rc_context(custom_rc):
        print(f'\nSave plot -> {ofile} ...\n')
        binhist = np.linspace(0, 40.0, 180)
        fig, ax = plt.subplots(figsize=(10, 7))
        col = 'rho' + str(xrho)
        ax.hist(bkg[col], bins=binhist, alpha=0.5, label='bkg', log=True, color='blue')
        ax.hist(sim[col], bins=binhist, alpha=0.5, label='sim', log=True, color='green')
        ax.set_xlabel(r'$\rho_' + str(xrho) + '$')
        ax.set_ylabel(r'$\textrm{Number of events}$')
        ax.legend(loc='upper right')
        plt.title("Detection Statistic Distributions", fontsize=30, pad=20)
        plt.tight_layout()
        fig.savefig(ofile)
        plt.close()


def plot_hist_mchirp(ofile, bkg, sim):
    """Plot reconstructed chirp-mass (``chirp1``) BKG vs SIM histograms.

    Parameters
    ----------
    ofile : str
        Output PNG path.
    bkg, sim : pd.DataFrame
        Must contain column ``chirp1``.
    """
    with plt.rc_context(custom_rc):
        print(f'\nSave plot -> {ofile} ...\n')
        binhist = np.linspace(0, 100.0, 200)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.hist(bkg['chirp1'], bins=binhist, alpha=0.5, label='bkg', log=True, color='blue')
        ax.hist(sim['chirp1'], bins=binhist, alpha=0.5, label='sim', log=True, color='green')
        ax.set_xlabel(r'$mchirp$')
        ax.set_ylabel(r'$\textrm{Number of events}$')
        ax.legend(loc='upper right')
        plt.title("Chirp Distributions", fontsize=30, pad=20)
        plt.tight_layout()
        fig.savefig(ofile)
        plt.close()


def plot_hist_freq(ofile, bkg, sim):
    """Plot central frequency (``frequency0``) BKG vs SIM histograms.

    Parameters
    ----------
    ofile : str
        Output PNG path.
    bkg, sim : pd.DataFrame
        Must contain column ``frequency0``.
    """
    with plt.rc_context(custom_rc):
        print(f'\nSave plot -> {ofile} ...\n')
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.hist(bkg['frequency0'], bins=250, alpha=0.5, label='bkg', log=True, color='blue')
        ax.hist(sim['frequency0'], bins=250, alpha=0.5, label='sim', log=True, color='green')
        ax.set_xlabel(r'$\textrm{Central frequency (Hz)}$')
        plt.grid()
        ax.set_ylabel(r'$\textrm{Number of events}$')
        ax.legend(loc='upper right')
        plt.title("Central Frequency Distributions", fontsize=35, pad=20)
        plt.tight_layout()
        fig.savefig(ofile)
        plt.close()


def plot_QaQp(ofile, bkg, sim, rho_name='rho0', rho_thr=8, rho_label='rho0',
              qfactor=0.15, qoffset=0.8, bkg_marker_color='black', bkg_marker_size=22,
              qa_sup=6.0, qp_sup=10.0, sim_bin_size=0.1):
    """Plot Qa vs Qp: SIM as 2-D histogram, BKG as scatter above *rho_thr*.

    Parameters
    ----------
    ofile : str
        Output PNG path.
    bkg, sim : pd.DataFrame
        Must contain columns ``Qa``, ``Qp``, and *rho_name*.
    rho_name : str
        Column for BKG threshold (default ``'rho0'``).
    rho_thr : float
        Only BKG events with ``rho_name > rho_thr`` are shown.
    qfactor, qoffset : float
        Parameters of dashed line ``Qa = qfactor / (Qp - qoffset)``.
    """
    warnings.filterwarnings(action='ignore', category=UserWarning)
    with plt.rc_context(custom_rc):
        print(f'\nSave plot -> {ofile} ...\n')
        if sim_bin_size <= 0:
            sim_bin_size = 0.1
        nxbins = max(1, int(np.nanmax(sim['Qp']) / sim_bin_size))
        nybins = max(1, int(np.nanmax(sim['Qa']) / sim_bin_size))
        fig, ax = plt.subplots(figsize=(10, 7))
        h = ax.hist2d(sim['Qp'], sim['Qa'], bins=[nxbins, nybins],
                      cmap=plt.get_cmap('rainbow'), norm=LogNorm(), label='sim')
        fig.colorbar(h[3], ax=ax)
        rho_selection = f'bkg ({rho_label}$>{rho_thr})'
        bkg_sel = bkg[bkg[rho_name] > rho_thr]
        ax.scatter(bkg_sel['Qp'], bkg_sel['Qa'], marker='.', color=bkg_marker_color,
                   label=rho_selection, s=bkg_marker_size)
        ax.set_xlabel(r'$\textrm{Qp}$')
        ax.set_xlim(-0.1, qp_sup)
        ax.set_ylabel(r'$\textrm{Qa}$')
        ax.set_ylim(-0.1, qa_sup)
        x = np.arange(qoffset * 1.01, qp_sup, 0.01)
        ax.plot(x, qfactor / (x - qoffset), linestyle='dashed',
                label=f'Qa={qfactor}/(Qp-{qoffset})', color='red')
        ax.legend(loc='lower right')
        plt.title(r'$\textrm{Qa vs Qp Distributions}$', fontsize=30, pad=20)
        plt.tight_layout()
        fig.savefig(ofile)
        plt.close()


# ---------------------------------------------------------------------------
# Generic 1-D and 2-D feature distribution plots
# ---------------------------------------------------------------------------

def plot1d(par, ofile, bkg, sim, inf=None, sup=None, bins=None,
           bkg_color='blue', sim_color='green', xlog=False, ylog=True):
    """Plot a single feature as overlaid BKG/SIM histograms.

    Parameters
    ----------
    par : str
        Column name.
    ofile : str
        Output PNG path.
    bkg, sim : pd.DataFrame
    inf, sup : float, optional
        Histogram range.
    bins : int, optional
        Bin count (default 200).
    xlog, ylog : bool
        Log-scale axes.
    """
    with plt.rc_context(custom_rc):
        print(f'\nSave plot -> {ofile} ...\n')
        data_min = min(float(np.nanmin(bkg[par])), float(np.nanmin(sim[par])))
        data_max = max(float(np.nanmax(bkg[par])), float(np.nanmax(sim[par])))
        if inf is None:
            inf = data_min
        if sup is None:
            sup = data_max
        if bins is None or bins <= 0:
            bins = 200
        inf = max(inf, data_min)
        sup = min(sup, data_max)
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.hist(bkg[(bkg[par] > inf) & (bkg[par] < sup)][par],
                bins=int(bins), alpha=0.5, label='bkg', log=ylog, color=bkg_color)
        ax.hist(sim[(sim[par] > inf) & (sim[par] < sup)][par],
                bins=int(bins), alpha=0.5, label='sim', log=ylog, color=sim_color)
        xlabel = par.replace('_', r'\_')
        ax.set_xlabel(r'$\textrm{' + xlabel + '}$')
        ax.set_xlim(inf, sup)
        plt.grid()
        if xlog:
            plt.xscale('log')
        ax.set_ylabel(r'$\textrm{Number of events}$')
        ax.legend(loc='upper right')
        plt.title(r'$\textrm{' + xlabel + '}$', fontsize=40, pad=20)
        plt.tight_layout()
        fig.savefig(ofile)
        plt.close()


def mplot1d(bkg, sim, odir=".", utag=None, goptions=None):
    """Generate per-feature 1-D BKG/SIM histograms for all enabled entries in *goptions*.

    Features are enabled by ``feature(mplot1d): {enable: True, ...}`` or
    ``feature(mplot*d)`` in the ``ML_options`` dictionary.

    Parameters
    ----------
    bkg, sim : pd.DataFrame
    odir : str
        Output directory (created if absent).
    utag : str, optional
        Filename prefix.
    goptions : dict
        ``ML_options`` from :func:`~.config.xgb_config`.
    """
    if goptions is None:
        raise ValueError("mplot1d: goptions must not be None")
    _goptions = goptions.copy()
    for feature, options in list(_goptions.items()):
        if '(mplot*d)' in feature:
            _feat1d = feature.replace('*', '1')
            if _goptions.get(_feat1d) is None:
                _goptions[_feat1d] = options
            _goptions.pop(feature)
    os.makedirs(odir, exist_ok=True)
    bkg_color = _goptions.get('bkg(mplot1d)', {}).get('color', 'blue')
    sim_color = _goptions.get('sim(mplot1d)', {}).get('color', 'green')
    for feature, options in _goptions.items():
        if '(mplot1d)' not in feature:
            continue
        _feature = feature.replace('(mplot1d)', '').replace(' ', '')
        if _feature in ('bkg', 'sim'):
            continue
        if not options.get('enable', False):
            continue
        if _feature not in bkg.columns or _feature not in sim.columns:
            logger.debug("mplot1d: skipping %s – column not present", _feature)
            continue
        fname = f"bkg_sim_{_feature}_plot.png".replace("/", "S")
        ofile = os.path.join(odir, f"{utag}_{fname}" if utag else fname)
        if os.path.exists(ofile):
            os.remove(ofile)
        try:
            plot1d(_feature, ofile, bkg, sim,
                   inf=options.get('inf'), sup=options.get('sup'),
                   bins=options.get('bins'), bkg_color=bkg_color, sim_color=sim_color)
        except Exception as exc:
            logger.warning("mplot1d: could not plot %s: %s", _feature, exc)


def plot2d(xpar, ypar, ofile, bkg, sim,
           sim_cmap=None, bkg_rho_name=None, bkg_rho_thr=None, bkg_rho_label=None,
           bkg_marker_color=None, bkg_marker_size=None,
           sim_xinf=None, sim_xsup=None, sim_xbins=None,
           sim_yinf=None, sim_ysup=None, sim_ybins=None):
    """Plot two features: SIM as 2-D histogram, BKG as scatter overlay.

    Parameters
    ----------
    xpar, ypar : str
        Column names for x and y axes.
    ofile : str
        Output PNG path.
    bkg, sim : pd.DataFrame
    sim_cmap : str
        Colormap for SIM (default ``'rainbow'``).
    bkg_rho_name : str
        Column for BKG threshold (default ``'rho0'``).
    bkg_rho_thr : float
        Threshold (default 8).
    """
    warnings.filterwarnings(action='ignore', category=UserWarning)
    with plt.rc_context(custom_rc):
        print(f'\nSave plot -> {ofile} ...\n')
        if sim_cmap is None:
            sim_cmap = 'rainbow'
        if bkg_rho_name is None:
            bkg_rho_name = 'rho0'
        if bkg_rho_thr is None:
            bkg_rho_thr = 8
        if bkg_rho_label is None:
            bkg_rho_label = 'rho0'
        if bkg_marker_color is None:
            bkg_marker_color = 'black'
        if bkg_marker_size is None:
            bkg_marker_size = 22
        if sim_xbins is None or sim_xbins <= 0:
            sim_xbins = 200
        if sim_ybins is None or sim_ybins <= 0:
            sim_ybins = 200
        _xinf = sim_xinf if sim_xinf is not None else float(np.nanmin(sim[xpar]))
        _xsup = sim_xsup if sim_xsup is not None else float(np.nanmax(sim[xpar]))
        _yinf = sim_yinf if sim_yinf is not None else float(np.nanmin(sim[ypar]))
        _ysup = sim_ysup if sim_ysup is not None else float(np.nanmax(sim[ypar]))
        _xinf = max(_xinf, float(np.nanmin(sim[xpar])))
        _xsup = min(_xsup, float(np.nanmax(sim[xpar])))
        _yinf = max(_yinf, float(np.nanmin(sim[ypar])))
        _ysup = min(_ysup, float(np.nanmax(sim[ypar])))
        fig, ax = plt.subplots(figsize=(10, 7))
        sel = ((sim[xpar] > _xinf) & (sim[xpar] < _xsup) &
               (sim[ypar] > _yinf) & (sim[ypar] < _ysup))
        h = ax.hist2d(sim[sel][xpar], sim[sel][ypar],
                      bins=[sim_xbins, sim_ybins],
                      cmap=plt.get_cmap(sim_cmap), norm=LogNorm(), label='sim')
        fig.colorbar(h[3], ax=ax)
        rho_sel = f'bkg ({bkg_rho_label}$>{bkg_rho_thr})'
        bkg_sel = bkg[bkg[bkg_rho_name] > bkg_rho_thr]
        ax.scatter(bkg_sel[xpar], bkg_sel[ypar], marker='.', color=bkg_marker_color,
                   label=rho_sel, s=bkg_marker_size)
        xl = xpar.replace('_', r'\_')
        yl = ypar.replace('_', r'\_')
        ax.set_xlabel(r'$\textrm{' + xl + '}$')
        ax.set_xlim(_xinf, _xsup)
        ax.set_ylabel(r'$\textrm{' + yl + '}$')
        ax.set_ylim(_yinf, _ysup)
        legend = plt.legend(fontsize=15, edgecolor='black', loc='lower right',
                            bbox_to_anchor=(1, 0.95))
        legend.get_frame().set_alpha(None)
        plt.title(r'$\textrm{' + yl + ' vs ' + xl + '}$', fontsize=30, pad=20)
        plt.tight_layout()
        fig.savefig(ofile)
        plt.close()


def mplot2d(bkg, sim, odir=".", utag=None, goptions=None):
    """Generate pairwise 2-D BKG/SIM plots for all enabled feature pairs in *goptions*.

    Parameters
    ----------
    bkg, sim : pd.DataFrame
    odir : str
        Output directory (created if absent).
    utag : str, optional
        Filename prefix.
    goptions : dict
        ``ML_options`` from :func:`~.config.xgb_config`.
    """
    if goptions is None:
        raise ValueError("mplot2d: goptions must not be None")
    _goptions = goptions.copy()
    for feature, options in list(_goptions.items()):
        if '(mplot*d)' in feature:
            _feat2d = feature.replace('*', '2')
            if _goptions.get(_feat2d) is None:
                _goptions[_feat2d] = options
            _goptions.pop(feature)
    os.makedirs(odir, exist_ok=True)
    bkg_opts = _goptions.get('bkg(mplot2d)', {})
    sim_opts = _goptions.get('sim(mplot2d)', {})
    bkg_rho_name     = bkg_opts.get('rho_name', 'rho0')
    bkg_rho_thr      = bkg_opts.get('rho_thr', 8)
    bkg_rho_label    = bkg_opts.get('rho_label', 'rho0')
    bkg_marker_color = bkg_opts.get('marker_color', 'royalblue')
    bkg_marker_size  = bkg_opts.get('marker_size', 100)
    sim_cmap         = sim_opts.get('cmap', 'rainbow')
    features_2d = [
        (f.replace('(mplot2d)', '').replace(' ', ''), v)
        for f, v in _goptions.items()
        if '(mplot2d)' in f
        and f.replace('(mplot2d)', '').replace(' ', '') not in ('bkg', 'sim')
    ]
    for i, (feat1, opts1) in enumerate(features_2d):
        if not opts1.get('enable', False):
            continue
        for feat2, opts2 in features_2d[i + 1:]:
            if not opts2.get('enable', False):
                continue
            if feat1 not in bkg.columns or feat2 not in bkg.columns:
                logger.debug("mplot2d: skipping %s vs %s – column missing", feat1, feat2)
                continue
            fname = f"bkg_sim_{feat1}_{feat2}_plot.png".replace("/", "S")
            ofile = os.path.join(odir, f"{utag}_{fname}" if utag else fname)
            if os.path.exists(ofile):
                os.remove(ofile)
            try:
                plot2d(feat1, feat2, ofile, bkg, sim,
                       sim_cmap=sim_cmap,
                       bkg_rho_name=bkg_rho_name, bkg_rho_thr=bkg_rho_thr,
                       bkg_rho_label=bkg_rho_label,
                       bkg_marker_color=bkg_marker_color, bkg_marker_size=bkg_marker_size,
                       sim_xinf=opts1.get('inf'), sim_xsup=opts1.get('sup'),
                       sim_xbins=opts1.get('bins'),
                       sim_yinf=opts2.get('inf'), sim_ysup=opts2.get('sup'),
                       sim_ybins=opts2.get('bins'))
            except Exception as exc:
                logger.warning("mplot2d: could not plot %s vs %s: %s", feat1, feat2, exc)


# ---------------------------------------------------------------------------
# Post-training evaluation plots (standalone file-producing versions)
# ---------------------------------------------------------------------------

def plot_roc(ofile, model, X_eval, y_eval):
    """Save a ROC curve plot (True Positive Rate vs False Positive Rate).

    Parameters
    ----------
    ofile : str
        Output PNG path.
    model : xgb.XGBClassifier
        Trained classifier.
    X_eval : array-like or pd.DataFrame
        Evaluation features.
    y_eval : array-like or pd.DataFrame
        True binary labels; accepts 1-D arrays or a single-column DataFrame.
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    print(f'\nSave plot -> {ofile} ...\n')
    y_true = np.asarray(y_eval).ravel()
    X_np   = np.asarray(X_eval, dtype=np.float32)
    ns_probs = np.zeros(len(y_true))
    lr_probs = model.predict_proba(X_np)[:, 1]
    ns_auc = roc_auc_score(y_true, ns_probs)
    lr_auc = roc_auc_score(y_true, lr_probs)
    print(f'No Skill: ROC AUC={ns_auc:.3f}')
    print(f'Skill   : ROC AUC={lr_auc:.3f}')
    ns_fpr, ns_tpr, _ = roc_curve(y_true, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_true, lr_probs)
    with plt.rc_context({**custom_rc, "text.usetex": False}):
        fig, ax = plt.subplots()
        ax.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        ax.plot(lr_fpr, lr_tpr, marker='.', label=f'Skill (AUC={lr_auc:.3f})')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        plt.title("ROC", fontsize=20, pad=20)
        plt.tight_layout()
        fig.savefig(ofile)
        plt.close()


def plot_pr(ofile, model, X_eval, y_eval):
    """Save a Precision-Recall curve plot.

    Parameters
    ----------
    ofile : str
        Output PNG path.
    model : xgb.XGBClassifier
        Trained classifier.
    X_eval : array-like or pd.DataFrame
        Evaluation features.
    y_eval : array-like or pd.DataFrame
        True binary labels; accepts 1-D arrays or a single-column DataFrame.
    """
    from sklearn.metrics import precision_recall_curve, f1_score, auc as sk_auc

    print(f'\nSave plot -> {ofile} ...\n')
    y_true = np.asarray(y_eval).ravel()
    X_np   = np.asarray(X_eval, dtype=np.float32)
    lr_probs = model.predict_proba(X_np)[:, 1]
    yhat     = model.predict(X_np)
    lr_precision, lr_recall, _ = precision_recall_curve(y_true, lr_probs)
    lr_f1  = f1_score(y_true, yhat)
    lr_auc = sk_auc(lr_recall, lr_precision)
    print(f'Skill: f1={lr_f1:.3f}  auc={lr_auc:.3f}')
    no_skill = float(np.sum(y_true == 1)) / len(y_true)
    with plt.rc_context({**custom_rc, "text.usetex": False}):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        ax.plot(lr_recall, lr_precision, marker='.', label=f'Skill (AUC={lr_auc:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(loc='lower left')
        plt.title("Precision-Recall Plot", fontsize=20, pad=20)
        plt.tight_layout()
        fig.savefig(ofile)
        plt.close()

import logging

import matplotlib.pyplot as plt
from matplotlib import rcParams
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
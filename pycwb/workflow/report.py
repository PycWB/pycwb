import os
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pycwb.modules.report.report import calculate_detection_efficiency
from pycwb.modules.statistics.sigmoid_fit import fit, estimate_hrss, logNfit

logger = logging.getLogger(__name__)

def generate_report(catalog_file: str, simulation: bool,
                    lag: int = None, slag: int = None,
                    simulation_file: str = None, far_threshold: float = 1e-2, inj_key: str = 'hrss',
                    working_dir: str = '.', report_dir: str = 'report'):
    # TODO: add rho/cc/param cuts?

    working_dir = os.path.abspath(working_dir)
    report_dir = os.path.join(working_dir, report_dir)

    if not os.path.exists(catalog_file):
        raise FileNotFoundError(f"{catalog_file} not found")
    if (simulation_file is not None) and (not os.path.exists(simulation_file)):
        raise FileNotFoundError(f"{simulation_file} not found")
    
    if simulation:
        if simulation_file is None:
            raise ValueError("simulation_file must be provided when simulation=True")
        report_simulation(catalog_file, simulation_file, far_threshold, inj_key, report_dir)
    else:
        if lag and slag:
            report_background_lag(catalog_file)
        else:
            report_background(catalog_file)
    

def report_background(catalog_file: str):
    # open catalog file

    
    # ------------------------------------------------------
    # plot detection statistic distributions
    # ------------------------------------------------------
    # rho vs subnet

    # rho vs log10(chi2)

    # rho vs cc

    # rho distribution

    # rate vs rho

    # ------------------------------------------------------
    # plot rates
    # ------------------------------------------------------
    # detector fraction

    # 

    # generate html report
    print('report_background')

def report_background_lag(catalog_file: str):
    print('report_background_lag')

def report_simulation(catalog_file: str, simulation_file: str, far_threshold: float, inj_key: str = 'hrss',
                    report_dir: str = 'report'):
    # FIXME: Or do you plan to use matched.parquet file as input file?

    report_dir = Path(report_dir)
    # FIXME: follow cWB's report naming convention?
    report_dirname = f'{report_dir.parent.name}.S_ifar{int(1/far_threshold)}.R_rMRA'
    report_dir = report_dir / report_dirname
    data_dir = report_dir / 'data'
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    logger.info(f'Open {catalog_file}')
    cat = pq.read_table(catalog_file)
    logger.info(f'Open {simulation_file}')
    sim = pq.read_table(simulation_file)

    # apply FAR threshold
    # cat = cat.filter(pc.field('ifar') > 1 / far_threshold)

    # apply dq vetoes; fill nulls with False so missing veto columns don't discard all injections
    veto_mask = pc.and_(pc.invert(pc.if_else(pc.is_null(sim['vetoed_cat0']), False, sim['vetoed_cat0'])), pc.and_(pc.invert(pc.if_else(pc.is_null(sim['vetoed_cat1']), False, sim['vetoed_cat1'])), pc.invert(pc.if_else(pc.is_null(sim['vetoed_cat2']), False, sim['vetoed_cat2']))))
    sim = sim.filter(veto_mask)

    # target hrss range
    # FIXME: only for hrss? or can be generalized?
    inj_factors = sorted(sim.column(inj_key).combine_chunks().unique().to_pylist())
    xdata = np.arange(np.log10(min(inj_factors)), np.log10(max(inj_factors)), 0.01)

    # FIXME: Waveform groups?
    # groups = sim.column('group').unique().to_pylist()
    
    det_effs = {}
    fit_parameters = {}
    names = sorted(sim.column('name').unique().to_pylist())
    # FIXME: FOR TESTING - drop waveforms with 0 injections at some factors
    _bad = {'WNB17b_0_15', 'WNB17b_0_2', 'WNB17b_0_4', 'WNB17b_10_16', 'WNB17b_10_17', 'WNB17b_10_22', 'WNB17b_10_8', 'WNB17b_11_11', 'WNB17b_11_12', 'WNB17b_11_23', 'WNB17b_11_26', 'WNB17b_11_4', 'WNB17b_11_8', 'WNB17b_12_12', 'WNB17b_12_27', 'WNB17b_12_28', 'WNB17b_12_5', 'WNB17b_12_9', 'WNB17b_13_1', 'WNB17b_13_18', 'WNB17b_13_2', 'WNB17b_13_27', 'WNB17b_13_3', 'WNB17b_14_10', 'WNB17b_14_13', 'WNB17b_14_7', 'WNB17b_14_9', 'WNB17b_15_11', 'WNB17b_15_13', 'WNB17b_15_14', 'WNB17b_15_19', 'WNB17b_15_21', 'WNB17b_15_22', 'WNB17b_15_23', 'WNB17b_15_28', 'WNB17b_15_29', 'WNB17b_1_29', 'WNB17b_1_3', 'WNB17b_2_14', 'WNB17b_2_8', 'WNB17b_3_1', 'WNB17b_3_16', 'WNB17b_4_10', 'WNB17b_4_12', 'WNB17b_4_17', 'WNB17b_4_29', 'WNB17b_4_3', 'WNB17b_5_0', 'WNB17b_5_17', 'WNB17b_5_18', 'WNB17b_5_23', 'WNB17b_5_24', 'WNB17b_5_25', 'WNB17b_5_27', 'WNB17b_5_3', 'WNB17b_5_5', 'WNB17b_5_6', 'WNB17b_6_16', 'WNB17b_6_17', 'WNB17b_6_19', 'WNB17b_6_21', 'WNB17b_6_28', 'WNB17b_7_11', 'WNB17b_7_5', 'WNB17b_7_6', 'WNB17b_7_8', 'WNB17b_8_0', 'WNB17b_8_12', 'WNB17b_8_15', 'WNB17b_8_17', 'WNB17b_8_27', 'WNB17b_8_6', 'WNB17b_9_19', 'WNB17b_9_23', 'WNB17b_9_24', 'WNB17b_9_4', 'WNB17b_9_7', 'WNB17b_9_8'}
    names = [n for n in names if n not in _bad]
    for idx, name in enumerate(names):
        cat_name = cat.filter(pc.equal(cat['injection'].combine_chunks().field('name'), name))
        sim_name = sim.filter(pc.equal(sim['name'], name))

        # calculate detection efficiency per factor
        det_eff, det_err = calculate_detection_efficiency(cat_name, sim_name, inj_key=inj_key, inj_factors=inj_factors)
        logger.debug(f'{name}: inj_factors={inj_factors}, det_eff={det_eff}, det_err={det_err}')  # FIXME: FOR TESTING
        
        if any(np.isnan(det_eff)):
            nan_factors = np.array(inj_factors)[np.isnan(det_eff)]
            raise ValueError(f'{name}: NAN values for {inj_key}={nan_factors}')

        # fit sigmoid function
        chi2, hrss50, hrssEr, sigma, betam, betap, flag = fit(np.log10(inj_factors), det_eff)
        fit_parameters[name] = {'chi2': chi2, 'hrss50': hrss50, 'hrssEr': hrssEr, 'sigma': sigma, 'betam': betam, 'betap': betap, 'flag': flag}
        logger.info(f'{name}: hrss50 = {hrss50:.2e} +/- {hrssEr:.2e}, chi2 = {chi2:.2f}, sigma = {sigma:.2f}, betam = {betam:.2f}, betap = {betap:.2f}')

        # estimate hrss10, hrss50, hrss90
        xlim = (np.log10(min(inj_factors)), np.log10(max(inj_factors)))
        hrss10 = estimate_hrss((hrss50, sigma, betam, betap, flag), xlim, 0.1)
        # hrss50 = estimate_hrss((hrss50, sigma, betam, betap, flag), xlim, 0.5)  # FIXME: hrss50 is already calculated?
        hrss90 = estimate_hrss((hrss50, sigma, betam, betap, flag), xlim, 0.9)

        # plots
        fig_name, ax_name = plt.subplots()
        ax_name.errorbar(inj_factors, det_eff, yerr=det_err, ls='none', marker='s', capsize=2, color=f'C{idx}')
        ax_name.plot(10**xdata, logNfit(xdata, np.log10(hrss50), sigma, betam, betap, flag), color=f'C{idx}')
        ax_name.set_xscale('log')
        ax_name.set_xlabel(inj_key)
        ax_name.set_ylabel('Detection Efficiency')
        ax_name.set_title(f'{name}, hrss50={hrss50:.2e}')
        ax_name.grid(ls='dotted', which='both')
        ax_name.set_ylim(0-0.05, 1+0.05)
        
        # save results
        det_eff_file = data_dir / f'det_eff_{name}'
        logger.info(f'Save {det_eff_file}')
        # save txt file
        np.savetxt(f'{det_eff_file}.txt', np.column_stack([inj_factors, det_eff, det_err]), header=f'{inj_key}\tDet_Eff\tDet_Err', fmt=['%.4e', '%.4f', '%.4f'], comments='')
        # save jpg file
        fig_name.savefig(f'{det_eff_file}.jpg')
        plt.close(fig_name)
    
    # save fit parameters
    fit_params_file = data_dir / 'fit_parameters_ALL.txt'
    logger.info(f'Save {fit_params_file}')
    with open(fit_params_file, 'w') as f:
        f.write('name\tchi2\thrss50\thrssEr\tsigma\tbetam\tbetap\n')
        for name in names:
            params = fit_parameters[name]
            f.write(f"{name}\t{params['chi2']:.4f}\t{params['hrss50']:.4e}\t{params['hrssEr']:.4e}\t{params['sigma']:.4f}\t{params['betam']:.4f}\t{params['betap']:.4f}\n")

    # generate_html_simulation

# def extract_analysis_config(catalog_file: str):
#     # production
#     - pycwb version
#     - search frequency band
#     - working directory
#     - time-frequency resolutions
#     - skymap segmantation: healpix order = 

#     # post-production
#     # TBA

#     # livetime
#     - zero lag:
#     - non-zero lags:

# def extract_catalog_schema_metadata(catalog_file: str):
#     metadata = pq.read_schema(catalog_file).metadata

#     pycwb_version = metadata.get(b'pycwb_version', b'').decode('utf-8')

#     return pycwb_version

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    sim_file = '/Users/jiyoon.sun/pycwb-dev/simulations.parquet'
    cat_file = '/Users/jiyoon.sun/pycwb-dev/catalog.parquet'

    generate_report(catalog_file=cat_file, simulation_file=sim_file, simulation=True, far_threshold=1e-2)



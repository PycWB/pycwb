import os
import Path
import logging
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pycwb.modules.report import calculate_detection_efficiency
from pycwb.modules.statistics.sigmoid_fit import fit, estimate_hrss, logNfit

logger = logging.getLogger(__name__)

def generate_report(catalog_file: str, simulation: bool,
                    lag: int = None, slag: int = None,
                    simulation_file: str = None, far_threshold: float = 1e-2, inj_key: str = 'hrss',
                    working_dir: str = '.', report_dir: str = 'report'):
    # TODO: add rho/cc/param cuts?

    working_dir = Path(working_dir)
    report_dir = working_dir / report_dir

    if not os.path.exists(catalog_file):
        raise FileNotFoundError(f"{catalog_file} not found")
    if (simulation_file is not None) and (not os.path.exists(simulation_file)):
        raise FileNotFoundError(f"{simulation_file} not found")
    
    if simulation:
        report_simulation()
    else:
        if lag and slag:
            report_background_lag(lag: int, slag: int)
        else:
            report_background()

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
    
def report_background_lag(catalog_file: str):

def report_simulation(catalog_file: str, simulation_file: str, far_threshold: float, inj_key: str = 'hrss',
                    report_dir: str = 'report'):
    # FIXME: Or do you plan to use matched.parquet file as input file?
        
    # FIXME: follow cWB's report naming convention?
    report_dirname = f'{working_dir.name}.S_ifar{int(1/far_threshold)}.R_rMRA'
    report_dir = report_dir/ report_dirname
    data_dir = report_dir / 'data'
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    logger.info(f'Open {catalog_file}')
    cat = pq.read_table(catalog_file)
    logger.info(f'Open {simulation_file}')
    sim = pq.read_table(simulation_file)

    # apply FAR threshold
    cat = cat.filter(pc.field('FAR') < far_threshold)

    # apply dq vetoes
    # FIXME: if cat2 doesn't exist -> null -> all vetoed
    veto_mask = pc.and_(pc.invert(sim['vetoed_cat0']), pc.and_(pc.invert(sim['vetoed_cat1']), pc.invert(sim['vetoed_cat2'])))
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
    for idx, name in enumerate(names):
        cat_name = cat.filter(pc.equal(cat['injection'].combine_chunks().field('name'), name))
        sim_name = sim.filter(pc.equal(sim['name'], name))

        # calculate detection efficiency per factor
        det_eff, det_err = calculate_detection_efficiency(cat_name, sim_name, inj_key=inj_key, inj_factors=inj_factors)
        # np.savetxt - det_eff_{name}.txt

        # fit sigmoid function
        chi2, hrss50, hrssEr, sigma, betam, betap, flag = fit(np.log10(xdata), det_eff)
        fit_parameters[name] = {'chi2': chi2, 'hrss50': hrss50, 'hrssEr': hrssEr, 'sigma': sigma, 'betam': betam, 'betap': betap, 'flag': flag}
        logger.info(f'{name}: hrss50 = {hrss50:.2e} +/- {hrssEr:.2e}, chi2 = {chi2:.2f}, sigma = {sigma:.2f}, betam = {betam:.2f}, betap = {betap:.2f}')

        # estimate hrss10, hrss50, hrss90
        hrss10 = getHrss((hrss50, sigma, betam, betap, flag), 0.1)
        # hrss50 = getHrss((hrss50, sigma, betam, betap, flag), 0.5)  # FIXME: hrss50 is already calcualted?
        hrss90 = getHrss((hrss50, sigma, betam, betap, flag), 0.9)

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
        
        det_eff_file = data_dir / f'det_eff_{name}.jpg'
        logger.info(f'Save {det_eff_file}')
        fig_name.savefig(det_eff_file)
        plt.close(fig_name)
        
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
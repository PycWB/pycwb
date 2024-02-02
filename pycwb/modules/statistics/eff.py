import numpy as np
from scipy.special import erfc
from scipy.optimize import root_scalar


def read_inj_type(file_name):
    """
    Read the user-defined injection types file.

    Parameters:
    file_name (str): Path to the list of MDC types file (ASCII).

    Returns:
    int: Number of injections read.
    """
    try:
        with open(file_name, 'r') as file:
            print(f"inj file: {file_name}")

            ninj = 0
            sets, types, names, fcentrals, fbandwidths = [], [], [], [], []

            injections = []

            for line in file:
                # Skip empty lines and comments
                if line.startswith('#') or line.startswith(' ') or not line.strip():
                    continue

                parts = line.split()
                if len(parts) < 5:
                    print(f"ReadMdcType : Error Reading File : {file_name}")
                    print(f"at line : \"{line.strip()}\"")
                    print("check if new line is defined at the end of the string")
                    exit(1)

                set_, type_, name_, fcentral, fbandwidth = parts[:5]

                sets.append(set_)
                types.append(int(type_))
                names.append(name_)
                fcentrals.append(float(fcentral))
                fbandwidths.append(float(fbandwidth))

                print(f" {set_} {type_} {name_} {fcentral} {fbandwidth}")
                injections.append(
                    {'set': set_, 'type': type_, 'name': name_, 'fcentral': fcentral, 'fbandwidth': fbandwidth})

                ninj += 1

            return injections

    except IOError:
        print(f"ReadMdcType : Error Opening File : {file_name}")
        exit(1)


def read_fit_parameters(filepath):
    ecount, chi2, hrss50, err, par1, par2, par3, ewaveform = [], [], [], [], [], [], [], []

    try:
        with open(filepath, 'r') as file:
            for line in file:
                parts = line.split()
                if len(parts) != 9:  # Ensure each line has the correct number of parts
                    print(f"Unexpected format in line: {line}")
                    continue
                ecount.append(int(parts[0]))
                chi2.append(float(parts[1]))
                hrss50.append(float(parts[2]))
                err.append(float(parts[4]))  # Skip the '+-' symbol at parts[3]
                par1.append(float(parts[5]))
                par2.append(float(parts[6]))
                par3.append(float(parts[7]))
                ewaveform.append(parts[8])

        return np.array(ecount), np.array(chi2), np.array(hrss50), np.array(err), np.array(par1), np.array(
            par2), np.array(par3), ewaveform
    except IOError:
        print(f"Error Opening File: {filepath}")
        exit(1)


def logNfit(x, par0, par1, par2, par3, par4):
    y = (np.log10(x) - par0)
    if par4:
        y = -y
    s = par1 * np.exp(y * par2) if y < 0 else par1 * np.exp(y * par3)

    if y > 0:
        if par3 > 1. / y:
            s = par1 * par3 * np.exp(1.)
            y = 1.
        y = np.abs(y / s) if s > 0 else 100.
        return 1 - erfc(y) / 2

    if y < 0:
        y = np.abs(y / s) if s > 0 else 100.
        return erfc(y) / 2

    return 0.5


def get_hrss_from_percentile(percentile, hrss50, par1, par2, par3, pp_factor2distance):
    inf = -25
    sup = -18.5

    # logNfit(x, np.log10(hrss50[1]), par1[1], par2[1], par3[1], pp_factor2distance)

    # find the corresponding x at given y for logNfit with scipy

    def func_to_solve(x):
        return logNfit(x, np.log10(hrss50), par1, par2, par3, pp_factor2distance) - percentile

    try:
        # Find the root of the function
        result = root_scalar(func_to_solve, bracket=[10 ** inf, 10 ** sup], xtol=1e-25, method='brentq')

        if not result.converged:
            print(f"Failed to converge for {percentile}th percentile")
            return None

        return result.root
    except ValueError:
        print(f"Failed to converge for {percentile}th percentile")
        return None


def read_hrss_for_mdc(run_dir, pp_factor2distance=0.):
    injections = read_inj_type(run_dir + 'injectionList.txt')
    imdc_set_name = list(set([inj['set'] for inj in injections]))

    imdc_set_hrss10 = {}
    imdc_set_hrss50 = {}
    imdc_set_hrss90 = {}
    imdc_set_hrss50_err = {}

    for set_name in imdc_set_name:
        ecount, chi2, hrss50, err, par1, par2, par3, ewaveform = read_fit_parameters(run_dir + f'fit_parameters_{set_name}.txt')

        imdc_set_hrss10[set_name] = {
            wf_name: get_hrss_from_percentile(0.1, hrss50[i], par1[i], par2[i], par3[i], pp_factor2distance)
            for i, wf_name in enumerate(ewaveform)
        }
        imdc_set_hrss50[set_name] = {
            wf_name: get_hrss_from_percentile(0.5, hrss50[i], par1[i], par2[i], par3[i], pp_factor2distance)
            for i, wf_name in enumerate(ewaveform)
        }
        imdc_set_hrss50_err[set_name] = {
            wf_name: get_hrss_from_percentile(0.5, hrss50[i]+err[i], par1[i], par2[i], par3[i], pp_factor2distance)
            for i, wf_name in enumerate(ewaveform)
        }
        imdc_set_hrss90[set_name] = {
            wf_name: get_hrss_from_percentile(0.9, hrss50[i], par1[i], par2[i], par3[i], pp_factor2distance)
            for i, wf_name in enumerate(ewaveform)
        }

    return imdc_set_hrss10, imdc_set_hrss50, imdc_set_hrss90, imdc_set_hrss50_err


def plot_hrss_from_mdc(run_dir):
    import matplotlib.pyplot as plt

    injections = read_inj_type(run_dir + 'injectionList.txt')

    imdc_set_hrss10, imdc_set_hrss50, imdc_set_hrss90, imdc_set_hrss50_err = read_hrss_for_mdc(run_dir)

    for inj_set_name in imdc_set_hrss50.keys():
        wf_names = list(imdc_set_hrss50[inj_set_name].keys())
        central_freqs = [float(inj['fcentral']) for inj in injections if
                         inj['set'] == inj_set_name and inj['name'] in wf_names]
        hrss50s = [imdc_set_hrss50[inj_set_name][wf_name] for wf_name in wf_names]
        hrss50_errs = np.array([imdc_set_hrss50_err[inj_set_name][wf_name] for wf_name in wf_names]) - hrss50s
        line, = plt.loglog(central_freqs, hrss50s, label=inj_set_name, marker='o')
        line_color = line.get_color()
        plt.errorbar(central_freqs, hrss50s, yerr=hrss50_errs, fmt='none', capsize=3, color=line_color)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Hrss (1/Hz^{-1/2})')
    # show more x ticks
    plt.title('Hrss50')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig(run_dir + 'hrss50.png')

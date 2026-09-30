"""Plot the first-search input and saved reconstruction after running the CLI.

Usage: python examples/demo/plot_results.py my_first_search
"""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from pycwb.types.time_series import TimeSeries
from pycwb.utils.module import import_function


STYLE = {
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "axes.axisbelow": True,
    "savefig.dpi": 170,
}


def plot_results(run, output):
    run, output = Path(run), Path(output)
    config_file = run / "config/user_parameters.yaml"
    config = yaml.safe_load(config_file.read_text())
    parameters = dict(config["injection"]["parameters"])
    epoch = float(parameters["gps_time"])
    ifos = config["ifo"]

    events = pd.read_parquet(run / "catalog/catalog.parquet")
    progress = pd.read_parquet(run / "catalog/progress.parquet")
    if len(progress) != 1 or not (progress["status"] == "completed").all():
        raise ValueError("The demo must have one completed job; inspect pycwb progress.")
    near = np.ones(len(events), dtype=bool)
    for ifo in ifos:
        near &= (events[f"time_{ifo}"] - epoch).abs().le(1.0).to_numpy()
    near &= np.isfinite(events["rho"]) & (events["rho"] >= abs(config["netRHO"]))
    candidates = events.loc[near].sort_values("rho", ascending=False)
    if candidates.empty:
        raise ValueError("No recovered candidate near the injection; inspect the catalog and log.")
    event = candidates.iloc[0]
    event_hash = str(event["id"]).rsplit("_", 1)[-1]
    output.mkdir(parents=True, exist_ok=True)

    # Regenerate the source polarizations with the same generator and sampling.
    parameters["delta_t"] = 1.0 / config["inRate"]
    generated = import_function(config["injection"]["generator"])(**parameters)
    if generated["type"] != "polarizations":
        raise ValueError("This example expects a generator returning source polarizations.")
    hp = TimeSeries.from_input(generated["hp"])
    hc = TimeSeries.from_input(generated["hc"])
    source_hrss = float(np.sqrt(np.sum(hp.data**2 + hc.data**2) * hp.dt))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
        for wave, label, color in [(hp, r"$h_+$", "#2563eb"),
                                    (hc, r"$h_\times$", "#d97706")]:
            relative_time = wave.t0 + np.arange(len(wave.data)) * wave.dt
            ax.plot(relative_time * 1000, wave.data / 1e-21,
                    label=label, color=color, linewidth=1.7)
        ax.set(xlim=(-45, 45), xlabel="Time from source epoch [ms]",
               ylabel=r"Source strain [$10^{-21}$]",
               title=f"Injected sine-Gaussian · {parameters['frequency']:g} Hz · Q = {parameters['Q']:g}")
        ax.legend(loc="upper right", frameon=False)
        fig.savefig(output / "injected_signal.png")
        plt.close(fig)

        # Read both saved products with their own epochs and sample rates.
        # Do not align peaks, normalize amplitudes, or shift detector times.
        with h5py.File(run / "output/wave.h5", "r") as handle:
            group = handle[event_hash]
            fig, axes = plt.subplots(len(ifos), 1, figsize=(9, 5.3),
                                     sharex=True, squeeze=False, layout="constrained")
            for ax, ifo in zip(axes[:, 0], ifos):
                for product, label, color, linestyle in [
                    ("INJ", "Injected", "#2563eb", "-"),
                    ("REC", "Reconstructed", "#d97706", "--"),
                ]:
                    wave = group[f"{ifo}_wf_{product}"]
                    time = (float(wave.attrs["start_time"]) - epoch
                            + np.arange(len(wave)) / float(wave.attrs["sample_rate"]))
                    ax.plot(time * 1000, wave[:] / 1e-21, label=label,
                            color=color, linestyle=linestyle, linewidth=1.6)
                ax.set(xlim=(-70, 50), ylabel=ifo + r" strain [$10^{-21}$]")
                ax.legend(loc="upper right", frameon=False)
            axes[-1, 0].set_xlabel(f"Time from GPS {epoch:.0f} [ms]")
            fig.suptitle("Saved detector waveforms · original amplitudes and arrival times")
            fig.savefig(output / "reconstruction.png")
            plt.close(fig)

    columns = ["id", "job_id", "trial_idx", "lag_idx", "rho", "net_cc"]
    columns += [f"{field}_{ifo}" for ifo in ifos for field in ("time", "central_freq")]
    summary = {
        "pycwb_version": version("pycwb"),
        "burst_waveform_version": version("burst-waveform"),
        "wdm_wavelet_version": version("wdm-wavelet"),
        "yaml_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
        "source_hrss": source_hrss,
        "injection_gps": epoch,
        "total_triggers": len(events),
        "triggers_within_one_second": len(candidates),
        "selected_event": json.loads(event[columns].to_json()),
        "progress": json.loads(progress.to_json(orient="records")),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(candidates[columns].to_string(index=False))
    print(f"Wrote injected_signal.png, reconstruction.png and summary.json to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Completed first-search work directory")
    parser.add_argument("--output-dir", type=Path, help="Figure directory (default: RUN/plots)")
    args = parser.parse_args()
    plot_results(args.run, args.output_dir or args.run / "plots")

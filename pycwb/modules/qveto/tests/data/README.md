# cWB Q-veto fixture

`cwb_qveto_fixture.npz` stores five waveforms, the actual cWB upsampled arrays,
and Qveto/Qfactor outputs from the original HF `GetQveto` plugin function.
Inputs include fixed-seed random arrays (64, 256, 1024 samples) and two
high-frequency Gaussian-windowed sinusoids (4096 samples), at 8192 Hz.

The capture copies the original function into a diagnostic ROOT macro and runs
it against the original cWB 6.4.6.9 library. No reference values are calculated
using the Python implementation. Scripts and logs are in the parent workspace:
`runs/end_to_end_regression/{prepare_qveto_oracle.py,qveto_oracle.C,qveto_expected.txt}`.

The fixture tests packed real-FFT resizing (including movement of the Nyquist
component), zero-crossing endpoints, and float32 output/ratio conventions.
The zero-crossing reference includes the crossing sample in the preceding peak,
skips waveform sample zero, and discards an unfinished final half-cycle.

# cWB scaling-energy fixture

`cwb_snr_energy_fixture.npz` contains L1/H1 cases 0, 1, 19 and 41 from the matched LF training control (84 injections, CPU healpix 5). Each `_strain` array is a six-second native-whitened reference raw injection, using the noise RMS exported from the cWB ReadData stage. Each `_meta` holds start GPS, sample rate, source GPS, expected cWB detector::setsim ISNR energy, half-window, lower and upper frequencies. Expected energies were independently exported from the original cWB implementation, not computed with the helper under test.

Generation and full 168-detector/source audit: `runs/lf_matched_injections/check_snr_estimator.py`, `cwb_snr_fixture/`, and `snr_estimator_fixture.csv` in the comparison workspace. The reference noise RMS export uses a copied diagnostic plugin, with no cWB library modification. The waveform buffers are reference raw strain; native whitening uses the exported RMS without the later conditioning bandpass. Fixture inputs are therefore not direct exports of cWB whitened arrays. This regression tests energy conventions independently against cWB values, including low-frequency cases.

## HF waveform polarizations

`cwb_hf_waveform_fixture.npz` contains actual cWB 6.4.6.9 plugin exports for
SGE849Q100, SGE2477Q100, SGE5000Q100 and two training WNB realizations.
The archive includes the source parameters and cWB Gaussian inputs for WNB.
PycWB independently generates the waveform from these inputs; reference final
polarizations are used only for comparison. Relative L2 tolerance is 1e-12.
Capture/provenance: `runs/end_to_end_regression/hf/save_waveform_fixture.py` and
`waveform_fixture_provenance.json` in the parent burst workspace. Rate: 16384 Hz.

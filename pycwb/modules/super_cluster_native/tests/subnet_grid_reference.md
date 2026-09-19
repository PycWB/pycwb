# Subnet delay-grid reference

`subnet_grid_reference.npz` contains two arrays:

- `cwb_subnet_delays`: the two-detector delay table exported from cWB 6.4.6.9
  at the supercluster/subnet stage, with L1 as reference, L1/H1 detectors,
  4096 Hz analysis rate and HEALPix order 4. Shape: `(2, 3072)`.
- `native_likelihood_delays`: the existing native likelihood delay table for
  the same segment, with 16384 Hz TD sampling and HEALPix order 5.
  Shape: `(2, 12288)`. This protects likelihood behavior during the subnet fix.

Reference cWB commit: `e03cf7f02fa4d4666c5619ffcf9730e988328e3c`.
The Chunk 16a segment starts at GPS 1387221730 including its 10-second edge.
The cWB trace ran in `CWB_PLUGIN_OSUPERCLUSTER`, before the switch to
oversampled likelihood filters. Its subsequent zero-lag output reproduced
the original three release events on the checked fields.

Tests address the existing oversampled native TD buffers with scaled indices;
the underlying subnet delay grid must remain at the analysis rate. Both equal
and different subnet/likelihood sky resolutions are covered. This fixture does
not establish general detector-network or end-to-end numerical equivalence.

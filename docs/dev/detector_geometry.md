# Detector geometry registry

Detector constants live in `pycwb/constants/detectors.py`. `DETECTORS` contains
bundled geographic parameters, and `DETECTOR_GEOMETRIES` registers named
sources, including cWB's literal Earth-centered vertices and arm vectors.
The separate `release_detector_geometry.py` table has been removed.

Select geometry independently of channel, frame and data-quality names:

```yaml
ifo: [L1, H1, V1]
refIFO: L1
detector_geometry:
  L1: L1:cwb
  H1: H1:cwb
  V1: V1:lal@pycwb-1
```

`L1:cwb` and `H1:cwb` are canonical IDs, with no release suffix. Config loading
resolves omitted detectors and LAL aliases into explicit IDs; the catalog
records this mapping and batch restoration retains it. Unknown IDs,
selections for inactive detectors and cross-detector selections are rejected.
The former global YAML string `detector_geometry: cwb_6.4.6.9` must be replaced
with the mapping above. Qualified names are also supported by the Python API:
`Detector("H1:cwb")`. Its data identity remains `H1`, and its `geometry_id`
contains the canonical registry ID.

The default for each detector is `DETECTOR:lal@pycwb-1`. Here `pycwb-1` identifies
this repository's existing bundled LAL-derived geographic table and conversion,
not a LALSuite release or the installed LAL library. The cWB entries were verified
against 6.4.6.9 `wat/detector.cc`, commit
e03cf7f; only H1/L1 have validated cWB definitions. Other detectors can use their
bundled definitions in the same run. Selecting an unavailable cWB entry fails.

## Why two definitions exist

The cWB definition was introduced for exact release reproduction. It uses rounded
literal arm vectors rather than deriving them from the bundled geographic angles.
The existing audit (`runs/detector_geometry_audit/comparison.json` in the parent
workspace) found these differences against PycWB's default geometry:

| Detector | Vertex displacement | Maximum absolute antenna difference in sampled sky |
| --- | ---: | ---: |
| H1 | 5.367 mm | 3.006e-6 |
| L1 | 3.524 mm | 2.921e-6 |

The antenna comparison sampled 4,099 directions. These are absolute differences,
not relative errors or bounds over the continuous sky. After matching the literal
vectors, antenna differences from the cWB oracle were below 1.6e-15. These tests
establish reproduction of cWB, not which geometry is closer to a surveyed
instrument. More decimal digits alone do not establish physical accuracy.

For existing PycWB analyses the bundled LAL-derived definition preserves their
geometry; for cWB 6.4.6.9 reproduction select the `:cwb` entries. Both affect
injection projection, sky delays, antenna response and event timing. The cWB
selection also retains the pre-existing release export convention of narrowing
stored event angles to float32 before antenna evaluation, now per detector.
This output convention is distinct from the physical constants.

The current upstream LAL reference is
[LALDetectors.h](https://lscsoft.docs.ligo.org/lalsuite/lal/_l_a_l_detectors_8h.html).
Its geographic parameters and directly tabulated vectors should not be conflated:
PycWB's default constructs vectors from the geographic parameters.

# cWB release geometry oracle

`release_detector_geometry.npz` contains H1/L1 antenna patterns at 4,099 sky positions from cWB 6.4.6.9, commit e03cf7f. Source generator and full numerical comparison: `runs/detector_geometry_audit/oracle.py` and `comparison.json` in the parent workspace.

`release_antenna_exports.json` retains only angle and antenna columns from the 300 Chunk 16a job-1 HF/LF cWB events. It tests Float_t angle narrowing before exported antenna evaluation. These fixtures contain no ROOT runtime dependency. They do not certify likelihood equivalence.

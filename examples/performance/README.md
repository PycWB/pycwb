# CPU execution profile

`bounded_cpu.yaml` is a configuration fragment, not a complete search. Merge
its keys into your own full YAML, preserving your detector, data, waveform and
scientific settings. Then use the ordinary `pycwb run` command. For an executable
small example of bounded scheduling and restart, prepare the tutorial inputs
with `python examples/tutorials/prepare.py` from the repository root; the generated
file is `tutorial-work/bounded.yaml`.

Compare results against the baseline before using a different execution profile
for a scientific campaign. Memory limits are configured separately in the
`execution` section; this fragment selects numerical and worker implementations.

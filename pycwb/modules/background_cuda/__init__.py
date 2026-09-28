"""Experimental, opt-in CUDA background stages; no production hooks are installed.

Select ``pycwb.modules.background_cuda.processor.process_job_segment`` as the
``segment_processer`` and enable stages with the ``gpu`` YAML mapping; see the
package README for the switch table, bounds and measured results.
"""

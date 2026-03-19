"""
Glitch injection — stub.

Future implementations may include:

* **Sine-Gaussian** — parametric sine-Gaussian bursts with configurable
  central frequency, Q-factor, and amplitude.
* **Catalogue-based** — inject glitches sampled from the Gravity Spy or
  similar glitch catalogues.
* **Poisson scheduling** — schedule glitch injections at random times drawn
  from a Poisson process.
"""

__all__ = ["inject_glitches"]


def inject_glitches(**kwargs):
    """Inject transient glitches into a time series.

    .. warning:: Not yet implemented.

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError(
        "Glitch injection is not yet implemented. "
        "Contributions welcome — see the module docstring for planned backends."
    )

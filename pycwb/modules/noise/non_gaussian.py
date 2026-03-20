"""
Non-Gaussian noise generation — stub.

Future implementations may include:

* **Modulated Gaussian** — amplitude-modulated Gaussian noise to simulate
  non-stationary detector behaviour.
* **Heavy-tailed** — noise drawn from Student-t or other heavy-tailed
  distributions.
* **Catalogue-based** — replay recorded noise segments from real detector data.
"""

__all__ = ["non_gaussian_noise"]


def non_gaussian_noise(**kwargs):
    """Generate non-Gaussian noise.

    .. warning:: Not yet implemented.

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError(
        "Non-Gaussian noise generation is not yet implemented. "
        "Contributions welcome — see the module docstring for planned backends."
    )

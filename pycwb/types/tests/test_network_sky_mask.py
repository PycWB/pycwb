from types import SimpleNamespace

import pycwb.types.network as network_module
from pycwb.types.network import Network


class _Backend:
    def __init__(self):
        self.calls = []

    def setSkyMask(self, *args):
        self.calls.append(args)
        return 1

    def setIndexMode(self, value):
        self.calls.append(("index", value))


class _Root:
    @staticmethod
    def skymap(*args):
        return SimpleNamespace(constructor_args=args)


def _network():
    instance = Network.__new__(Network)
    instance.net = _Backend()
    return instance


def test_inline_root_mask_uses_user_options(monkeypatch):
    captured = {}

    def fake_make_sky_mask(skymap, theta, phi, radius, ROOT_module=None):
        captured.update(theta=theta, phi=phi, radius=radius, root=ROOT_module)

    monkeypatch.setattr(network_module, "ROOT", _Root, raising=False)
    monkeypatch.setattr(network_module, "make_sky_mask", fake_make_sky_mask)
    config = SimpleNamespace(
        healpix=2, angle=1.0, Theta1=0.0, Theta2=180.0,
        Phi1=0.0, Phi2=360.0,
    )
    net = _network()
    net.set_sky_mask(
        config, "--theta -30 --phi 120 --radius 5", "c", skyres=None
    )
    assert captured == {"theta": -30.0, "phi": 120.0, "radius": 5.0, "root": _Root}


def test_root_mask_file_accepts_none_skyres(monkeypatch):
    monkeypatch.setattr(network_module, "ROOT", _Root, raising=False)
    net = _network()
    config = SimpleNamespace(healpix=0, angle=1.0)
    assert net.set_sky_mask(config, "mask.txt", "c", skyres=None) == 0
    assert net.net.calls == [("mask.txt", "c")]

import importlib
import sys
import warnings


def test_native_package_exports_canonical_and_compatibility_names():
    import pycwb.modules.data_conditioning as data_conditioning

    assert callable(data_conditioning.whiten_injection_strain)
    assert data_conditioning.whitening_mdc is data_conditioning.whiten_injection_strain


def test_native_legacy_module_warns():
    sys.modules.pop("pycwb.modules.data_conditioning.whitening_mdc", None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy = importlib.import_module(
            "pycwb.modules.data_conditioning.whitening_mdc"
        )

    assert any(issubclass(item.category, DeprecationWarning) for item in caught)
    assert legacy.whitening_mdc is legacy.whiten_injection_strain


def test_root_package_exports_canonical_and_compatibility_names():
    import pycwb.modules.data_conditioning_root as data_conditioning_root

    assert callable(data_conditioning_root.whiten_injection_strain)
    assert (
        data_conditioning_root.whitening_mdc
        is data_conditioning_root.whiten_injection_strain
    )


def test_root_legacy_module_warns():
    sys.modules.pop("pycwb.modules.data_conditioning_root.whitening_mdc", None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy = importlib.import_module(
            "pycwb.modules.data_conditioning_root.whitening_mdc"
        )

    assert any(issubclass(item.category, DeprecationWarning) for item in caught)
    assert legacy.whitening_mdc is legacy.whiten_injection_strain

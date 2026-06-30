"""Tests for pycwb.modules.xtalk — catalog header parsing and metadata."""
import struct
import numpy as np
import pytest
import tempfile
import os
from pycwb.modules.xtalk.monster import (
    _read_catalog_header_and_layers,
    read_catalog_metadata,
)


class TestReadCatalogHeaderAndLayers:
    """Tests for _read_catalog_header_and_layers — binary header parser."""

    def _pack_header(self, nRes, tag=0.0, beta_order=0.0, precision=0.0, kwdm=0.0,
                     layers=None):
        """Build a binary header matching the expected format."""
        if layers is None:
            layers = list(range(1, nRes + 1))

        # nRes is stored as float in the binary, negated if extended header
        if tag != 0.0 or beta_order != 0.0 or precision != 0.0 or kwdm != 0.0:
            stored_nRes = -nRes
            fmt = "f" + "4f" + f"{nRes}f"
            data = struct.pack(fmt, float(stored_nRes), tag, beta_order, precision, kwdm, *layers)
        else:
            fmt = "f" + f"{nRes}f"
            data = struct.pack(fmt, float(nRes), *layers)

        return data

    def test_basic_header_no_extended(self):
        """Simple header: nRes=3, no extended fields."""
        data = self._pack_header(3)
        result = _read_catalog_header_and_layers(data)

        assert result['nRes'] == 3
        assert result['tag'] == 0.0
        assert result['beta_order'] == 0.0
        assert result['precision'] == 0.0
        assert result['kwdm'] == 0.0
        assert result['layers'].tolist() == [1, 2, 3]

    def test_extended_header_with_negative_nRes(self):
        """Extended header: nRes < 0 signals extra metadata fields."""
        data = self._pack_header(2, tag=1.5, beta_order=2.0, precision=3.0, kwdm=4.0)
        result = _read_catalog_header_and_layers(data)

        assert result['nRes'] == 2
        assert result['tag'] == 1.5
        assert result['beta_order'] == 2.0
        assert result['precision'] == 3.0
        assert result['kwdm'] == 4.0
        assert result['layers'].tolist() == [1, 2]

    def test_single_layer(self):
        """Single resolution level (nRes=1)."""
        data = self._pack_header(1)
        result = _read_catalog_header_and_layers(data)
        assert result['nRes'] == 1
        assert len(result['layers']) == 1
        assert result['layers'][0] == 1

    def test_ten_layers(self):
        """Larger nRes with many layers."""
        nRes = 10
        layers = [i * 2 for i in range(nRes)]
        data = self._pack_header(nRes, layers=layers)
        result = _read_catalog_header_and_layers(data)
        assert result['nRes'] == 10
        assert result['layers'].tolist() == layers

    def test_offset_tracks_position(self):
        """offset should point to position after header."""
        data = self._pack_header(3)
        result = _read_catalog_header_and_layers(data)
        expected_offset = struct.calcsize("f" + "3f")
        assert result['offset'] == expected_offset


class TestReadCatalogMetadata:
    """Tests for read_catalog_metadata — file-based metadata reader."""

    def test_bin_file_roundtrip(self):
        """Write a binary catalog header, then read metadata back."""
        nRes = 4
        layers = [2, 4, 8, 16]
        fmt = "f" + f"{nRes}f"
        header = struct.pack(fmt, float(nRes), *layers)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(header)
            bin_path = f.name

        try:
            meta = read_catalog_metadata(bin_path)
            assert meta['nRes'] == 4
            assert meta['tag'] == 0.0
            assert meta['layers'].tolist() == layers
        finally:
            os.unlink(bin_path)

    def test_npz_file_roundtrip(self):
        """Write metadata as .npz, then read back."""
        layers = np.array([3, 6, 9], dtype=np.int32)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            npz_path = f.name
        try:
            np.savez(npz_path, nRes=3, layers=layers)
            meta = read_catalog_metadata(npz_path)
            assert meta['nRes'] == 3
            assert meta['layers'].tolist() == [3, 6, 9]
            # defaults when tag/precision not in npz
            assert meta['tag'] == 0.0
        finally:
            os.unlink(npz_path)

    def test_npz_with_extended_fields(self):
        """NPZ with all extended fields present."""
        layers = np.array([1, 2], dtype=np.int32)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            npz_path = f.name
        try:
            np.savez(npz_path, nRes=2, tag=1.0, beta_order=2.0,
                     precision=0.01, kwdm=4.0, layers=layers)
            meta = read_catalog_metadata(npz_path)
            assert meta['nRes'] == 2
            assert meta['tag'] == 1.0
            assert meta['beta_order'] == 2.0
            assert meta['precision'] == 0.01
            assert meta['kwdm'] == 4.0
        finally:
            os.unlink(npz_path)

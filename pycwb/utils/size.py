"""Parsing of explicit byte quantities, independent of execution policy."""

import re

def byte_size(value: int | str) -> int:
    """Parse integer bytes or explicit SI/IEC units; never interpret bare floats."""
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str):
        match = re.fullmatch(
            r"\s*(\d+(?:\.\d+)?)\s*(B|[KMGT]i?B)\s*", value, re.IGNORECASE
        )
        if match:
            unit = match[2].upper()
            exponent = 0 if unit == "B" else "KMGT".index(unit[0]) + 1
            return int(float(match[1]) * (1024 if "I" in unit else 1000) ** exponent)
    raise ValueError(f"Invalid memory size {value!r}; use integer bytes or e.g. '4GiB'")

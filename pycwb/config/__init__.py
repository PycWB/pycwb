"""Configuration models with a lazy export of the full analysis Config.

Importing lightweight policy/schema helpers must not import scientific modules
or recursively load the complete user-parameter schema through Config.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import Config as Config

__all__ = ["Config"]


def __getattr__(name: str) -> Any:
    if name == "Config":
        from .config import Config

        globals()[name] = Config
        return Config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

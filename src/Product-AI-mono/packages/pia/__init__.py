from importlib.metadata import PackageNotFoundError, version


def read_version() -> str:
    """Return the installed package version.

    When the package metadata is unavailable (e.g., running from a source
    checkout without installation), a default version is returned so that
    importing ``pia`` does not fail."""
    try:
        return version("pia")
    except PackageNotFoundError:  # pragma: no cover - fallback for tests
        return "0.0.0"


__version__ = read_version()

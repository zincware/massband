from massband import __all__


def nodes() -> dict[str, list[str]]:
    """Return all available nodes."""
    return {"massband": __all__}

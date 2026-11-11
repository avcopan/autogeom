"""Package exceptions."""


class ElementNotFoundError(Exception):
    """No element matches the given atomic number or symbol."""


class GeometryConversionError(Exception):
    """A geometry could not be converted to or from another representation."""


class XYZFormatError(Exception):
    """An xyz block could not be parsed."""

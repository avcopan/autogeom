"""autogeom tests."""

import autogeom


def test_stub() -> None:
    """Stub test to ensure the test suite runs."""
    print(autogeom.__version__)  # noqa: T201


def test__greet() -> None:
    """Test the greet function."""
    assert autogeom.greet("World") == "Hello, World!"


def test__greet_jim() -> None:
    """Test the greet_jim function."""
    assert autogeom.greet_jim() == "Hello, Jim!"

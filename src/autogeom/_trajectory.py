"""Multi-frame xyz I/O."""

from collections.abc import Sequence
from pathlib import Path

from ._geometry import Geometry, from_xyz_block, xyz_block
from .exc import XYZFormatError

__all__ = [
    "from_xyz_trajectory_block",
    "from_xyz_trajectory_file",
    "xyz_trajectory_block",
    "xyz_trajectory_comments",
    "xyz_trajectory_file",
]


def xyz_trajectory_block(
    geos: Sequence[Geometry], *, comments: Sequence[str] | None = None
) -> str:
    """Format geometries as a multi-frame xyz block, one frame per geometry.

    Args:
        geos: Geometries.
        comments: Comment line for each frame. Defaults to the charge and spin
            (see `xyz_block`).

    Returns:
        The xyz block.

    Raises:
        ValueError: If `comments` doesn't have one entry per geometry.
    """
    frame_comments: Sequence[str | None] = (
        [None] * len(geos) if comments is None else comments
    )
    if len(frame_comments) != len(geos):
        msg = f"Got {len(frame_comments)} comments for {len(geos)} geometries."
        raise ValueError(msg)
    return "\n".join(
        xyz_block(geo, comment=comment)
        for geo, comment in zip(geos, frame_comments, strict=True)
    )


def from_xyz_trajectory_block(
    xyz_block: str, *, charge: int, spin: int
) -> list[Geometry]:
    """Parse geometries from a multi-frame xyz block, e.g. a list of conformers.

    Each frame is an atom count line, a comment line, and the atom lines (see
    `from_xyz_block`). A single-frame block gives a list of one geometry.

    Args:
        xyz_block: The xyz block.
        charge: Total molecular charge, for every frame.
        spin: Number of unpaired electrons, for every frame.

    Returns:
        The parsed geometries, in order.

    Raises:
        XYZFormatError: If the block is empty, or a frame is malformed.
    """
    return [
        from_xyz_block(frame, charge=charge, spin=spin)
        for frame in _split_xyz_frames(xyz_block)
    ]


def xyz_trajectory_comments(xyz_block: str) -> list[str]:
    """Get the comment line of each frame of a multi-frame xyz block.

    Programs often store a per-frame value there, such as CREST's energies.

    Args:
        xyz_block: The xyz block.

    Returns:
        The comment lines, stripped of surrounding whitespace, in order.

    Raises:
        XYZFormatError: If the block is empty, or a frame is malformed.
    """
    return [frame.splitlines()[1].strip() for frame in _split_xyz_frames(xyz_block)]


def xyz_trajectory_file(
    geos: Sequence[Geometry],
    *,
    path: str | Path,
    comments: Sequence[str] | None = None,
) -> None:
    """Write geometries to a multi-frame xyz file (see `xyz_trajectory_block`).

    Args:
        geos: Geometries.
        path: Output path.
        comments: Comment line for each frame. Defaults to the charge and spin.
    """
    Path(path).write_text(xyz_trajectory_block(geos, comments=comments) + "\n")


def from_xyz_trajectory_file(
    path: str | Path, *, charge: int, spin: int
) -> list[Geometry]:
    """Read geometries from a multi-frame xyz file (see `from_xyz_trajectory_block`).

    Args:
        path: Path to the xyz file.
        charge: Total molecular charge, for every frame.
        spin: Number of unpaired electrons, for every frame.

    Returns:
        The parsed geometries, in order.
    """
    return from_xyz_trajectory_block(Path(path).read_text(), charge=charge, spin=spin)


def _split_xyz_frames(xyz_block: str) -> list[str]:
    """Split a multi-frame xyz block into single-frame blocks."""
    lines = xyz_block.strip().splitlines()
    if not lines:
        msg = "The provided xyz block is empty."
        raise XYZFormatError(msg)

    frames = []
    start = 0
    while start < len(lines):
        try:
            natms = int(lines[start])
        except ValueError as exc:
            msg = f"Expected an atom count on line {start + 1}, got {lines[start]!r}."
            raise XYZFormatError(msg) from exc
        stop = start + natms + 2
        if stop > len(lines):
            msg = (
                f"The frame starting on line {start + 1} has {natms} atoms, but "
                f"only {len(lines) - start - 2} atom lines follow."
            )
            raise XYZFormatError(msg)
        frames.append("\n".join(lines[start:stop]))
        start = stop
    return frames

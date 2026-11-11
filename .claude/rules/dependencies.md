# Dependency version bounds

This file comes from the copier template and is updated by `pixi run copier-update`.

## Run dependencies (`pyproject.toml`)

Lower bounds follow [SPEC 0](https://scientific-python.org/specs/spec-0000/).
- `requires-python`: at least the oldest Python minor version released in the last 3 years.
- Core scientific packages (numpy, scipy, matplotlib, pandas, scikit-image, networkx, scikit-learn, xarray, ipython, zarr): `>=` the oldest feature release (X.Y.0) from the last 2 years.
  The SPEC page has the current drop schedule.
- Other packages: the lowest version the code works with.
- Raise a bound above these minimums only when the code needs a newer feature, and say why in a comment.
- `pixi run add` (`uv add`) writes `>=<latest>`.
  Lower that bound afterwards.
- `pixi run test-lowest` tests against the lower bounds, so run it after changing them.
  Revisit the bounds at each release, or when `test-lowest` fails.

## No upper bounds

Don't add upper bounds (`<X`, `==`, `~=`) unless there is a specific reason, such as a known breaking release.
Put the reason in a comment next to the bound.

## Dev and test tools (`pixi.toml` features)

Tools that aren't run dependencies (ruff, ty, pytest, …) should be the newest versions: `"*"`, or the `>=<latest>` floor that `pixi add` writes.
Don't lower their bounds to SPEC 0; the floors also keep `test-lowest` from picking old tool versions.
Refresh them with `pixi update`.

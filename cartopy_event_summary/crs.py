"""
Utilities for handling Coordinate Reference Systems.

This module provides helpers for converting QuakeMigrate / pyproj coordinate reference
system definitions into Cartopy coordinate reference system objects. The resulting
Cartopy objects are used both as GeoAxes projections and as Matplotlib transform=...
arguments when plotting data on Cartopy-backed figures.

:copyright:
    2026, Conor A. Bacon
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from typing import Any, TypeAlias

import cartopy.crs as ccrs
from pyproj import CRS, Proj


CRSInput: TypeAlias = CRS | Proj | str | Mapping[str, object]


def cartopy_projection_from_proj(proj: CRSInput) -> ccrs.Projection:
    """
    Create a Cartopy GeoAxes projection that retains the source CRS parameters.

    Parameters
    ----------
    proj:
    pyproj-compatible CRS, ``Proj``, dictionary, or authority string.

    Returns
    -------
    cartopy.crs.Projection
        Projection expecting degrees for geographic input or metres for projected input
        (even if the source CRS axes are measured in kilometres).

    Raises
    ------
    NotImplementedError
        Unsupported projection family or unrepresentable projection parameters.

    """

    crs = CRS.from_user_input(proj.crs if isinstance(proj, Proj) else proj)


    params = _proj_params(crs)

    incompatible = {"pm", "towgs84", "nadgrids", "lon_wrap", "over", "axis"}
    unsupported = incompatible.intersection(params)
    if unsupported:
        raise NotImplementedError(
            f"Cannot represent {crs.name} in Cartopy: "
            f"datum/axis options that cannot be preserved: {sorted(unsupported)}",
        )

    globe = _globe_from_crs(crs)
    if crs.is_geographic:
        return ccrs.PlateCarree(globe=globe)

    if not crs.is_projected:
        raise NotImplementedError(
            f"Cannot represent {crs.name} in Cartopy: ",
            "only geographic and projected CRSs are supported"
        )

    name = params.get("proj")
    scale = cartopy_unit_conversion_factor(crs)

    if name == "utm":
        if any(key in params for key in ("x_0", "y_0", "k", "k_0")):
            raise NotImplementedError(
                f"Cannot represent {crs.name} in Cartopy: "
                "custom UTM false offsets or scale factors"
            )
        return ccrs.UTM(
            zone=int(params["zone"]),
            southern_hemisphere="south" in params,
            globe=globe,
        )

    if name == "lcc":
        first_parallel = _float(params, "lat_1", _float(params, "lat_0"))
        second_parallel = _float(params, "lat_2", first_parallel)
        if "k" in params or "k_0" in params:
            raise NotImplementedError(
                f"Cannot represent {crs.name} in Cartopy: "
                "Lambert scale factor cannot be preserved"
            )
        return ccrs.LambertConformal(
            central_longitude=_float(params, "lon_0"),
            central_latitude=_float(params, "lat_0"),
            false_easting=_float(params, "x_0") * scale,
            false_northing=_float(params, "y_0") * scale,
            standard_parallels=(first_parallel, second_parallel),
            globe=globe,
        )

    if name == "tmerc":
        return ccrs.TransverseMercator(
            central_longitude=_float(params, "lon_0"),
            central_latitude=_float(params, "lat_0"),
            false_easting=_float(params, "x_0") * scale,
            false_northing=_float(params, "y_0") * scale,
            scale_factor=_float(params, "k", _float(params, "k_0", 1.0)),
            globe=globe,
        )

    if name == "merc":
        if _float(params, "lat_0") != 0:
            raise NotImplementedError(
                f"Cannot represent {crs.name} in Cartopy: "
                "Mercator latitude of origin other than zero"
            )
        if "lat_ts" in params and ("k" in params or "k_0" in params):
            raise NotImplementedError(
                f"Cannot represent {crs.name} in Cartopy: "
                "Mercator with both lat_ts and scale factor"
            )
        return ccrs.Mercator(
            central_longitude=_float(params, "lon_0"),
            false_easting=_float(params, "x_0") * scale,
            false_northing=_float(params, "y_0") * scale,
            latitude_true_scale=(
                None if "lat_ts" not in params else _float(params, "lat_ts")
            ),
            scale_factor=(
                None
                if "lat_ts" in params
                else _float(params, "k", _float(params, "k_0", 1.0))
            ),
            globe=globe,
        )

    if name == "stere":
        if "k" in params or "k_0" in params:
            raise NotImplementedError(
                f"Cannot represent {crs.name} in Cartopy: "
                "stereographic scale factor cannot be preserved"
            )
        return ccrs.Stereographic(
            central_longitude=_float(params, "lon_0"),
            central_latitude=_float(params, "lat_0"),
            false_easting=_float(params, "x_0") * scale,
            false_northing=_float(params, "y_0") * scale,
            true_scale_latitude=(
                None if "lat_ts" not in params else _float(params, "lat_ts")
            ),
            globe=globe,
        )

    if name == "aeqd":
        if "k" in params or "k_0" in params or "lat_ts" in params:
            raise NotImplementedError(
                f"Cannot represent {crs.name} in Cartopy: "
                "non-standard azimuthal equidistant parameters"
            )
        return ccrs.AzimuthalEquidistant(
            central_longitude=_float(params, "lon_0"),
            central_latitude=_float(params, "lat_0"),
            false_easting=_float(params, "x_0") * scale,
            false_northing=_float(params, "y_0") * scale,
            globe=globe,
        )

    if name == "eqc":
        raise NotImplementedError(
            f"Cannot represent {crs.name} in Cartopy: "
            "projected eqc cannot use degree-based PlateCarree"
        )

    raise NotImplementedError(
        f"Cannot represent {crs.name} in Cartopy: "
        f"projection family {name}"
    )


def data_crs_from_proj(proj: CRSInput) -> ccrs.CRS:
    """
    Convert a pyproj-compatible CRS definition into a Cartopy data CRS.

    The returned object is intended for use as a Matplotlib transform=... argument when
    plotting data values on a Cartopy GeoAxes. For normal QuakeMigrate coordinate
    projections, where event, station, and grid-extent coordinates are longitude and
    latitude, this returns cartopy.crs.PlateCarree.

    Parameters
    ----------
    proj:
        Coordinate reference system definition. If the object has a crs attribute, that
        attribute is passed to pyproj.CRS.from_user_input otherwise proj itself is
        passed.

    Returns
    -------
    cartopy.crs.CRS
        Cartopy CRS to use as the plotting transform for data expressed in the input
        coordinate reference system.

    Raises
    ------
    NotImplementedError
        Raised if the input CRS is projected but its projection type is not currently
        supported by cartopy_projection_from_proj.
    pyproj.exceptions.CRSError
        Raised by pyproj.CRS.from_user_input if the input cannot be interpreted as a
        valid CRS.

    """

    crs = CRS.from_user_input(proj.crs if hasattr(proj, "crs") else proj)

    if crs.is_geographic:
        return ccrs.PlateCarree(globe=_globe_from_crs(crs))

    return cartopy_projection_from_proj(crs)


def _globe_from_crs(crs: CRS) -> ccrs.Globe:
    """
    Build a Cartopy globe from the ellipsoid associated with a pyproj CRS.

    Cartopy projections can be supplied with a Globe object to preserve the semi-major
    and semi-minor axes of the source CRS ellipsoid. This helper extracts those axes
    from the pyproj CRS and disables Cartopy's named ellipsoid shortcut by passing
    ellipse=None.

    Parameters
    ----------
    crs:
        pyproj coordinate reference system from which the ellipsoid should be read.

    Returns
    -------
    cartopy.crs.Globe
        Cartopy globe using the CRS ellipsoid semi-major and semi-minor axes.

    """

    ellipsoid = crs.ellipsoid

    semimajor_axis = ellipsoid.semi_major_metre
    semiminor_axis = ellipsoid.semi_minor_metre

    return ccrs.Globe(
        semimajor_axis=semimajor_axis, semiminor_axis=semiminor_axis, ellipse=None
    )


def _proj_params(crs: CRS) -> dict[str, Any]:
    """
    Read PROJ parameters from a CRS while silencing only the known loss warning.

    Parameters
    ----------
    crs:
        Source reference system.

    Returns
    -------
    proj_params:
        Parameters used only to select and configure a supported projection.

    """

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You will likely lose important projection information.*",
            category=UserWarning,
        )

        return crs.to_dict()


def _float(params: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """
    Return a required numeric PROJ parameter with an explicit default.

    Parameters
    ----------
    params:
        PROJ parameter dictionary.
    key:
        Parameter name.
    default:
        Value when the parameter is absent.

    Returns
    -------
    value:
        Numeric projection parameter.

    Raises
    ------
    ValueError
        Raised if the resolved value cannot be parsed as a floating-point number.

    """

    value = params.get(key, default)

    if value is None:
        raise ValueError(f"PROJ parameter {key} cannot be null")

    return float(value)


def cartopy_unit_conversion_factor(proj: CRSInput) -> float:
    """
    Return the conversion from LUT grid units to Cartopy map-axis units.

    A projected grid is converted to metres. Geographic longitude/latitude grids already
    using degrees are left in degrees; angular radians-per-degree factors must never be
    applied to ``PlateCarree`` x/y coordinates.

    Parameters
    ----------
    proj:
        LUT grid projection.

    Returns
    -------
    float
        Map units per LUT grid unit (metres or degrees, respectively).

    Raises
    ------
    NotImplementedError
        If the geographic axes do not use degrees.

    """

    crs = CRS.from_user_input(proj.crs if isinstance(proj, Proj) else proj)

    if crs.is_geographic:
        if not all(
            math.isclose(axis.unit_conversion_factor, math.pi / 180.0, rel_tol=1e-10)
            for axis in crs.axis_info[:2]
        ):
            raise NotImplementedError(
                f"Cannot represent {crs.name} in Cartopy: "
                "geographic grid coordinates must be degrees"
            )
        return 1.0

    if not crs.is_projected:
        raise NotImplementedError(
            f"Cannot represent {crs.name} in Cartopy: "
            "only geographic and projected CRSs are supported"
        )

    return _metres_per_unit(crs)


def _metres_per_unit(crs: CRS) -> float:
    """
    Return the physical metres corresponding to one horizontal grid unit.

    Parameters
    ----------
    crs:
        Projected CRS with linear horizontal axes.

    Returns
    -------
    float
        Positive metres per input unit, including custom PROJ units.

    Raises
    ------
    NotImplementedError
        If axes lack consistent linear units.

    """

    axes = crs.axis_info

    if len(axes) < 2:
        raise NotImplementedError(
            f"Cannot represent {crs.name} in Cartopy: missing horizontal axis units"
        )
    units = [axis.unit_conversion_factor for axis in axes[:2]]

    if not all(math.isfinite(value) and value > 0 for value in units):
        raise NotImplementedError(
            f"Cannot represent {crs.name} in Cartopy: invalid horizontal axis units"
        )

    if not math.isclose(units[0], units[1], rel_tol=1e-12):
        raise NotImplementedError(
            f"Cannot represent {crs.name} in Cartopy: different horizontal axis units"
        )

    return units[0]

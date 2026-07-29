"""
Utilities for handling Coordinate Reference Systems.

This module provides helpers for converting QuakeMigrate / pyproj coordinate
reference system definitions into Cartopy coordinate reference system objects. The
resulting Cartopy objects are used both as GeoAxes projections and as Matplotlib
transform=... arguments when plotting data on Cartopy-backed figures.

The conversion intentionally supports only a small set of common projections. Unknown
projected coordinate systems fail explicitly rather than silently plotting data in an
incorrect CRS.

:copyright:
    2026, Conor A. Bacon
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import warnings
from typing import Any

import cartopy.crs as ccrs
from pyproj import CRS


def cartopy_projection_from_proj(proj: Any) -> ccrs.Projection:
    """
    Convert a pyproj-compatible CRS definition into a Cartopy projection.

    The returned object is suitable for use as the projection=... argument when
    constructing a Cartopy GeoAxes. The input may be a pyproj.Proj object, a
    pyproj.CRS object, an authority string such as "EPSG:4326", a PROJ string, a PROJ
    dictionary, or another CRS-like object accepted by pyproj.CRS.from_user_input.

    The function handles the common projected CRS types expected in QuakeMigrate lookup
    tables. Geographic longitude/latitude coordinate systems are represented with
    cartopy.crs.PlateCarree. Projected coordinate systems are mapped onto explicit
    Cartopy projection classes where possible.

    Parameters
    ----------
    proj:
        Coordinate reference system definition. If the object has a crs attribute, that
        attribute is passed to pyproj.CRS.from_user_input otherwise proj itself is
        passed.

    Returns
    -------
    cartopy.crs.Projection
        Cartopy projection corresponding to the input CRS.

    Raises
    ------
    NotImplementedError
        Raised if the input CRS is projected but its projection type is not currently
        supported by this helper.
    pyproj.exceptions.CRSError
        Raised by pyproj.CRS.from_user_input if the input cannot be interpreted as a
        valid CRS.

    """

    crs = CRS.from_user_input(proj.crs if hasattr(proj, "crs") else proj)

    if crs.is_geographic:
        return ccrs.PlateCarree(globe=_globe_from_crs(crs))

    params = _proj_params(crs)
    proj_name = params.get("proj")
    globe = _globe_from_crs(crs)

    if proj_name == "lcc":
        lat_1 = _float(params, "lat_1")
        lat_2 = _float(params, "lat_2", lat_1)

        return ccrs.LambertConformal(
            central_longitude=_float(params, "lon_0", 0.0),
            central_latitude=_float(params, "lat_0", 0.0),
            false_easting=_linear_to_metres(_float(params, "x_0", 0.0), params),
            false_northing=_linear_to_metres(_float(params, "y_0", 0.0), params),
            standard_parallels=(lat_1, lat_2),
            globe=globe,
        )

    if proj_name == "utm":
        return ccrs.UTM(
            zone=int(params["zone"]),
            southern_hemisphere=bool(params.get("south", False)),
            globe=globe,
        )

    if proj_name == "tmerc":
        return ccrs.TransverseMercator(
            central_longitude=_float(params, "lon_0", 0.0),
            central_latitude=_float(params, "lat_0", 0.0),
            false_easting=_linear_to_metres(_float(params, "x_0", 0.0), params),
            false_northing=_linear_to_metres(_float(params, "y_0", 0.0), params),
            scale_factor=_float(params, "k", _float(params, "k_0", 1.0)),
            globe=globe,
        )

    if proj_name == "merc":
        return ccrs.Mercator(
            central_longitude=_float(params, "lon_0", 0.0),
            min_latitude=-80.0,
            max_latitude=84.0,
            globe=globe,
        )

    if proj_name == "stere":
        return ccrs.Stereographic(
            central_longitude=_float(params, "lon_0", 0.0),
            central_latitude=_float(params, "lat_0", 0.0),
            false_easting=_linear_to_metres(_float(params, "x_0", 0.0), params),
            false_northing=_linear_to_metres(_float(params, "y_0", 0.0), params),
            true_scale_latitude=_float(params, "lat_ts", None),
            globe=globe,
        )

    if proj_name == "aeqd":
        return ccrs.AzimuthalEquidistant(
            central_longitude=_float(params, "lon_0", 0.0),
            central_latitude=_float(params, "lat_0", 0.0),
            false_easting=_linear_to_metres(_float(params, "x_0", 0.0), params),
            false_northing=_linear_to_metres(_float(params, "y_0", 0.0), params),
            globe=globe,
        )

    if proj_name == "eqc":
        return ccrs.PlateCarree(
            central_longitude=_float(params, "lon_0", 0.0),
            globe=globe,
        )

    raise NotImplementedError(
        "Cannot automatically convert this CRS to a Cartopy GeoAxes projection yet. "
        f"CRS name: {crs.name}; authority: {crs.to_authority()}; "
        f"operation method: {_operation_method_name(crs)}"
    )


def data_crs_from_proj(proj: Any) -> ccrs.CRS:
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

    Cartopy projections can be supplied with a Globe object to preserve the
    semi-major and semi-minor axes of the source CRS ellipsoid. This helper extracts
    those axes from the pyproj CRS and disables Cartopy's named ellipsoid shortcut by
    passing ellipse=None.

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
        semimajor_axis=semimajor_axis,
        semiminor_axis=semiminor_axis,
        ellipse=None,
    )


def _proj_params(crs: CRS) -> dict[str, Any]:
    """
    Return PROJ-style parameters from a pyproj CRS.

    This helper centralises the intentional conversion from a rich pyproj CRS object to
    a PROJ-style dictionary. The conversion may be lossy in general, so pyproj emits a
    warning. In this module the dictionary is only used to select a supported Cartopy
    projection constructor and extract the small number of projection parameters needed
    by that constructor.

    Parameters
    ----------
    crs:
        pyproj coordinate reference system to convert into a PROJ-style parameter
        dictionary.

    Returns
    -------
    dict of str to Any
        PROJ-style CRS parameters, for example {"proj": "lcc", "lat_1": ...}.

    Notes
    -----
    This function suppresses pyproj's lossy-conversion warning locally. It does not
    suppress other warnings elsewhere in the application.

    """

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You will likely lose important projection information.*",
            category=UserWarning,
        )
        return crs.to_dict()


def _operation_method_name(crs: CRS) -> str | None:
    """
    Return the coordinate operation method name for a CRS, if available.

    This is used only for diagnostics when an unsupported projection is encountered.
    Some CRS objects, especially geographic CRSs or incomplete definitions, may not
    expose a coordinate operation.

    Parameters
    ----------
    crs:
        pyproj coordinate reference system to inspect.

    Returns
    -------
    str or None
        Coordinate operation method name if available, otherwise None.

    """

    operation = crs.coordinate_operation

    if operation is None:
        return None

    return operation.method_name


def _float(
    params: dict[str, Any],
    key: str,
    default: Any = None,
) -> float | None:
    """
    Return a parameter value as a float.

    Parameters
    ----------
    params:
        Dictionary of projection parameters.
    key:
        Parameter key to read from params.
    default:
        Value to use when key is not present. If the resolved value is None, None is
        returned.

    Returns
    -------
    float or None
        Parameter value converted to float, or None when the resolved value is None.

    Raises
    ------
    TypeError
        Raised if the resolved value cannot be converted to float because it has an
        incompatible type.
    ValueError
        Raised if the resolved value is a string or object value that cannot be parsed
        as a floating-point number.

    """

    value = params.get(key, default)

    if value is None:
        return None

    return float(value)


def _linear_to_metres(value: float | None, params: dict[str, Any]) -> float:
    """
    Convert a linear projection parameter to metres.

    Cartopy projection constructors expect false eastings and false northings in metres.
    Some QuakeMigrate LUT projection definitions use PROJ parameters with units="km", in
    which case the value must be multiplied by 1000 before being passed to Cartopy.

    Parameters
    ----------
    value:
        Linear projection value to convert. This is usually a false easting or false
        northing value read from a PROJ-style parameter dictionary.
    params:
        Dictionary of projection parameters. The "units" key is inspected to decide
        whether conversion from kilometres to metres is required.

    Returns
    -------
    float
        Linear projection value in metres.

    Raises
    ------
    TypeError
        Raised if value is None. Projection constructors that require linear offsets
        should pass a numeric default before calling this helper.

    """

    if value is None:
        raise TypeError("Linear projection value cannot be None.")

    units = params.get("units")

    if units == "km":
        return value * 1000.0

    return value

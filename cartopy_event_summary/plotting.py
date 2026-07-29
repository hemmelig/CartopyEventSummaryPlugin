"""
Core plotting routines for the Cartopy-backed QuakeMigrate event summary plugin.

This module provides a Cartopy implementation of the 2-D event summary plot. It mirrors
the standard QuakeMigrate 2-D summary plot, but it uses a Cartopy GeoAxes for the map
panel so that coastlines, borders, rivers, lakes, and populated places can be drawn
using Cartopy/Natural Earth datasets.

:copyright:
    2026, Conor A. Bacon
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from cartopy.io import shapereader
from matplotlib.gridspec import GridSpec
from quakemigrate.plot.event import (
    _extract_2d_coalescence_slices,
    _merge_waveform_legend_labels,
    _plot_coalescence_trace,
    _plot_text_summary,
    _plot_waveform_gather,
)
from quakemigrate.plot.maps import plot_map_overlays

from .crs import cartopy_projection_from_proj, data_crs_from_proj


@dataclass
class CartopyMapAxes2D:
    """
    Container for the Cartopy 2-D map axes used by the event summary plot.

    Attributes
    ----------
    xy:
        Cartopy GeoAxes used for the horizontal map panel.
    cax:
        Matplotlib axes used for the coalescence colourbar.
    bounds:
        Lookup-table grid bounds in projected grid coordinates before unit conversion.
        The expected shape is (3, 2), with rows corresponding to x, y, and z, and
        columns corresponding to minimum and maximum values.
    map_crs:
        Cartopy projection used by the horizontal map panel. Projected LUT grid data
        should be plotted with this CRS as the Matplotlib transform.
    data_crs:
        Cartopy CRS used for coordinate-space data. For normal QuakeMigrate coordinate
        projections this is usually ccrs.PlateCarree().
    unit_conversion_factor:
        Multiplicative factor used to convert LUT projected grid units into the units
        expected by the Cartopy projection. For LUTs defined in kilometres and Cartopy
        projections expecting metres, this is normally 1000.0.

    """

    xy: object
    cax: object
    bounds: np.ndarray
    map_crs: ccrs.CRS
    data_crs: ccrs.CRS
    unit_conversion_factor: float = 1.0

    def items(self):
        """
        Yield map-panel metadata using the same pattern as QuakeMigrate map axes.

        Yields
        ------
        tuple
            A tuple containing the panel label, axes object, coordinate indices, and
            coordinate-label strings. Only the horizontal xy panel is yielded by this
            Cartopy 2-D summary implementation.

        """

        yield "xy", self.xy, (0, 1), ("longitude", "latitude")


def build_2d_cartopy_map_axes(
    fig: plt.Figure,
    gs: tuple[int, int],
    lut,
    *,
    natural_earth_scale: str = "10m",
    borders: bool = True,
    rivers: bool = True,
    cities: bool = True,
    city_scale: str = "10m",
) -> CartopyMapAxes2D:
    """
    Build the Cartopy map panel and colourbar axes for the 2-D event summary.

    The horizontal map panel is created using the Cartopy projection derived from the
    LUT grid projection. The plotted coalescence slice should therefore be transformed
    with this map CRS.

    The Natural Earth features and city labels are drawn in geographic coordinates via
    the LUT coordinate CRS, which is normally Plate Carrée.

    Parameters
    ----------
    fig:
        Matplotlib figure on which the axes are created.
    gs:
        Grid shape passed to plt.subplot2grid as (nrows, ncols).
    lut:
        QuakeMigrate lookup-table object. The object must provide grid_proj,
        coord_proj, coord2grid(), and get_grid_extent().
    natural_earth_scale:
        Natural Earth resolution to use for coastlines, borders, rivers, and lakes.
        Typical values are "110m", "50m", and "10m".
    borders:
        If True, draw national borders.
    rivers:
        If True, draw rivers and lakes.
    cities:
        If True, draw Natural Earth populated places within the map extent.
    city_scale:
        Natural Earth resolution to use for populated places.

    Returns
    -------
    CartopyMapAxes2D
        Container holding the map axes, colourbar axes, projected grid bounds, CRSs, and
        unit conversion factor.

    """

    map_crs = cartopy_projection_from_proj(lut.grid_proj)
    data_crs = data_crs_from_proj(lut.coord_proj)
    unit_conversion_factor = _get_lut_unit_conversion_factor(lut)

    xy = plt.subplot2grid(
        gs,
        (2, 0),
        colspan=7,
        rowspan=6,
        fig=fig,
        projection=map_crs,
    )
    cax = plt.subplot2grid(gs, (8, 5), colspan=2, rowspan=1, fig=fig)
    cax.set_axis_off()

    bounds = _projected_grid_bounds_from_lut(lut)
    geographic_extent = _geographic_extent_from_lut(lut)

    _set_projected_map_limits(
        xy,
        bounds,
        unit_conversion_factor=unit_conversion_factor,
    )

    xy.coastlines(resolution=natural_earth_scale, zorder=40)

    if borders:
        xy.add_feature(cfeature.BORDERS.with_scale(natural_earth_scale), zorder=39)

    if rivers:
        xy.add_feature(cfeature.RIVERS.with_scale(natural_earth_scale), alpha=0.3, zorder=35)
        xy.add_feature(cfeature.LAKES.with_scale(natural_earth_scale), alpha=0.3, zorder=35)

    if cities:
        _plot_natural_earth_cities(
            xy,
            extent=geographic_extent,
            data_crs=data_crs,
            scale=city_scale,
        )

    gl = xy.gridlines(
        crs=data_crs,
        draw_labels=True,
        linewidth=0.4,
        alpha=0.5,
        linestyle="--",
        x_inline=False,
        y_inline=False,
        rotate_labels=False,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = False
    gl.ylines = False

    return CartopyMapAxes2D(
        xy=xy,
        cax=cax,
        bounds=bounds,
        map_crs=map_crs,
        data_crs=data_crs,
        unit_conversion_factor=unit_conversion_factor,
    )


def event_summary_2d_cartopy(
    run,
    event,
    marginalised_coa_map,
    lut,
    slice_mode: Literal["surface", "maximum"] = "surface",
    surface_depth: float = 0.0,
    overlay_manifest: str | None = None,
    plot_all_stations: bool = True,
    file_type: str = "pdf",
    borders: bool = True,
    rivers: bool = True,
    cities: bool = True,
    city_scale: str = "10m",
    natural_earth_scale: str = "10m",
) -> None:
    """
    Generate a Cartopy-backed 2-D event summary figure.

    This function mirrors the standard QuakeMigrate 2-D event summary, but the map panel
    is a Cartopy GeoAxes. The coalescence slice is plotted in the LUT grid projection,
    while stations, hypocentre markers, Natural Earth cities, and gridline labels are
    plotted in the coordinate CRS.

    Parameters
    ----------
    run:
        QuakeMigrate run metadata and output configuration.
    event:
        Located event to summarise.
    marginalised_coa_map:
        Marginalised coalescence map for the located event.
    lut:
        Lookup table used for event location.
    slice_mode:
        Type of 2-D slice to plot. "maximum" plots the maximum-amplitude slice, while
        "surface" plots a fixed-depth surface slice.
    surface_depth:
        Depth of the surface slice when slice_mode is "surface".
    overlay_manifest:
        Optional path to an overlay manifest. The current implementation delegates to
        QuakeMigrate's core overlay helper, which may need further Cartopy-specific
        handling if overlays are provided in longitude/latitude.
    plot_all_stations:
        If True, plot all stations associated with the event data. If False, plot only
        stations used for the located event.
    file_type:
        Output file type, for example "pdf", "png", or "svg".
    borders:
        If True, draw national borders.
    rivers:
        If True, draw rivers and lakes.
    cities:
        If True, draw Natural Earth populated places.
    city_scale:
        Natural Earth resolution to use for populated places.
    natural_earth_scale:
        Natural Earth resolution to use for coastlines, borders, rivers, and lakes.

    """

    fig = plt.figure(figsize=(25, 15))
    grid_dimensions = (9, 15)

    gs = GridSpec(*grid_dimensions)
    text_summary = fig.add_subplot(gs[0:2, 0:8])
    waveform_gather = fig.add_subplot(gs[0:7, 8:15])
    coalescence_timeseries = fig.add_subplot(gs[7:9, 8:15])

    coalescence_map = build_2d_cartopy_map_axes(
        fig,
        grid_dimensions,
        lut,
        natural_earth_scale=natural_earth_scale,
        borders=borders,
        rivers=rivers,
        cities=cities,
        city_scale=city_scale,
    )

    _plot_text_summary(text_summary, lut, event)

    idx_max, slices = _extract_2d_coalescence_slices(
        marginalised_coa_map,
        lut,
        slice_mode,
        surface_depth,
    )

    _plot_coalescence_panels_cartopy(coalescence_map, slices)

    if plot_all_stations:
        station_list = event.data.stations
    else:
        station_names = {
            key.split("_")[0]
            for key, available in event.onset_data.availability.items()
            if available == 1
        }
        station_list = [
            station for station in event.data.stations if station.id in station_names
        ]

    _plot_stations_cartopy(coalescence_map, station_list, "white")
    _plot_hypocentre_cartopy(coalescence_map, event.hypocentre)

    if overlay_manifest is not None:
        plot_map_overlays(overlay_manifest, coalescence_map.xy)

    _plot_waveform_gather(
        waveform_gather,
        lut,
        event,
        idx_max,
        station_list,
    )

    _plot_coalescence_trace(coalescence_timeseries, event)

    for ax in [waveform_gather, coalescence_timeseries]:
        ax.axvline(
            event.otime.datetime,
            label="Origin time",
            ls="--",
            lw=2,
            c="#F03B20",
        )

    handles, labels = waveform_gather.get_legend_handles_labels()
    handles, labels = _merge_waveform_legend_labels(
        handles,
        labels,
        event.onset_data.channel_maps,
    )

    waveform_gather.legend(
        handles,
        labels,
        fontsize=14,
        loc=1,
        framealpha=1,
        markerscale=0.5,
    )
    coalescence_timeseries.legend(fontsize=14, loc=1, framealpha=1)

    fig.tight_layout(pad=1, h_pad=0)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    fpath = run.path / "locate" / run.subname / "summaries"
    fpath.mkdir(exist_ok=True, parents=True)

    fstem = f"{run.name}_{event.uid}_EventSummary2DCartopy"
    file = (fpath / fstem).with_suffix(f".{file_type}")

    fig.savefig(file, dpi=400)
    plt.close(fig)


def _plot_coalescence_panels_cartopy(axes: CartopyMapAxes2D, slices) -> None:
    """
    Plot the 2-D coalescence slice on the Cartopy map panel.

    The slice grid is constructed in projected LUT grid coordinates and then converted
    into the coordinate units expected by the Cartopy map CRS using
    axes.unit_conversion_factor. For example, a LUT may use kilometres while the Cartopy
    projection expects metres.

    Parameters
    ----------
    axes:
        Cartopy axes container with projected grid bounds and map CRS.
    slices:
        Dictionary of 2-D coalescence slices returned by _extract_2d_coalescence_slices.

    """

    for ax_label, ax, (i, j), _ in axes.items():
        gminx, gmaxx = axes.bounds[i]
        gminy, gmaxy = axes.bounds[j]

        slice_ = slices[ax_label]
        nx, ny = [dim + 1 for dim in slice_.shape]

        grid1, grid2 = np.mgrid[
            gminx : gmaxx : nx * 1j,
            gminy : gmaxy : ny * 1j,
        ]

        sc = ax.pcolormesh(
            grid1 * axes.unit_conversion_factor,
            grid2 * axes.unit_conversion_factor,
            slice_,
            edgecolors="face",
            transform=axes.map_crs,
            zorder=5,
        )

        cb = axes.cax.figure.colorbar(
            sc,
            ax=axes.cax,
            orientation="horizontal",
            fraction=0.8,
            aspect=8,
        )
        cb.ax.set_xlabel(
            "Normalised coalescence\nvalue",
            rotation=0,
            fontsize=14,
        )


def _plot_stations_cartopy(
    axes: CartopyMapAxes2D,
    stations,
    c: str = "white",
) -> None:
    """
    Plot station locations on the Cartopy map panel.

    Station longitudes and latitudes are interpreted in the coordinate CRS, usually
    Plate Carrée. They are not transformed through the LUT grid projection before
    plotting; Cartopy performs the transformation from axes.data_crs to the map
    projection.

    Parameters
    ----------
    axes:
        Cartopy axes container.
    stations:
        Iterable of station objects with longitude and latitude attributes.
    c:
        Matplotlib colour specification for the station markers.

    """

    axes.xy.scatter(
        [station.longitude for station in stations],
        [station.latitude for station in stations],
        s=15,
        marker="^",
        zorder=20,
        c=c,
        transform=axes.data_crs,
    )


def _plot_hypocentre_cartopy(
    axes: CartopyMapAxes2D,
    hypocentre: list[float],
    c: str = "white",
) -> None:
    """
    Plot hypocentre crosshairs on the Cartopy map panel.

    Parameters
    ----------
    axes:
        Cartopy axes container.
    hypocentre:
        Event hypocentre sequence. The first two elements are longitude and latitude.
    c:
        Matplotlib colour specification for the hypocentre marker.

    """

    lon, lat = hypocentre[0], hypocentre[1]

    axes.xy.scatter(
        lon,
        lat,
        marker="+",
        s=120,
        linewidths=1.5,
        c=c,
        transform=axes.data_crs,
        zorder=50,
    )


def _plot_natural_earth_cities(
    ax: plt.Axes,
    extent: list[float],
    data_crs: ccrs.CRS,
    scale: str = "10m",
) -> None:
    """
    Plot Natural Earth populated places within the given geographic extent.

    Parameters
    ----------
    ax:
        Cartopy GeoAxes on which cities are plotted.
    extent:
        Geographic extent as [lon_min, lon_max, lat_min, lat_max].
    data_crs:
        CRS used for the city longitude and latitude coordinates.
    scale:
        Natural Earth resolution for the populated places dataset.

    """

    lon_min, lon_max, lat_min, lat_max = extent

    shp = shapereader.natural_earth(
        resolution=scale,
        category="cultural",
        name="populated_places",
    )

    for rec in shapereader.Reader(shp).records():
        geom = rec.geometry
        lon, lat = geom.x, geom.y

        if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
            continue

        name = rec.attributes.get("NAME") or rec.attributes.get("NAME_EN")

        ax.scatter(
            lon,
            lat,
            marker=".",
            transform=data_crs,
            zorder=45,
            c="white",
        )
        ax.text(
            lon,
            lat,
            name,
            fontsize=18,
            transform=data_crs,
            zorder=45,
            clip_on=True,
            c="white",
        )


def _projected_grid_bounds_from_lut(lut) -> np.ndarray:
    """
    Return lookup-table bounds in projected grid coordinates.

    lut.get_grid_extent(cells=True) is treated as a coordinate-space extent. The lower
    and upper corners are transformed into the LUT grid projection with
    lut.coord2grid(...). The resulting bounds are suitable for building the coalescence
    slice grid that will be plotted with transform=axes.map_crs.

    Parameters
    ----------
    lut:
        QuakeMigrate lookup-table object.

    Returns
    -------
    numpy.ndarray
        Projected grid bounds with shape (3, 2). Rows correspond to x, y, and z;
        columns correspond to minimum and maximum values.

    """

    coord_extent = np.asarray(lut.get_grid_extent(cells=True), dtype=float)

    lower_coord = coord_extent[0, :]
    upper_coord = coord_extent[1, :]

    lower_grid = _first_point(lut.coord2grid(lower_coord))
    upper_grid = _first_point(lut.coord2grid(upper_coord))

    bounds = np.stack(
        [
            [lower_grid[0], upper_grid[0]],
            [lower_grid[1], upper_grid[1]],
            [lower_grid[2], upper_grid[2]],
        ],
        axis=0,
    )

    return np.sort(bounds, axis=1)


def _geographic_extent_from_lut(lut) -> list[float]:
    """
    Return the geographic longitude/latitude extent of the lookup table.

    The extent is derived from lut.get_grid_extent(cells=True) and is used only for
    filtering Natural Earth cities. It assumes the first two coordinate dimensions are
    longitude and latitude, which is the standard QuakeMigrate coordinate-space setup.

    Parameters
    ----------
    lut:
        QuakeMigrate lookup-table object.

    Returns
    -------
    list of float
        Extent in the form [lon_min, lon_max, lat_min, lat_max].

    """

    coord_extent = np.asarray(lut.get_grid_extent(cells=True), dtype=float)

    lon_values = coord_extent[:, 0]
    lat_values = coord_extent[:, 1]

    return [
        float(np.min(lon_values)),
        float(np.max(lon_values)),
        float(np.min(lat_values)),
        float(np.max(lat_values)),
    ]


def _set_projected_map_limits(
    ax,
    bounds: np.ndarray,
    *,
    unit_conversion_factor: float,
) -> None:
    """
    Set GeoAxes x/y limits using projected LUT grid bounds.

    Parameters
    ----------
    ax:
        Cartopy GeoAxes to update.
    bounds:
        Projected LUT grid bounds with shape (3, 2).
    unit_conversion_factor:
        Factor used to convert projected LUT coordinates into the coordinate units
        expected by the Cartopy projection.

    """

    gminx, gmaxx = bounds[0]
    gminy, gmaxy = bounds[1]

    ax.set_xlim(gminx * unit_conversion_factor, gmaxx * unit_conversion_factor)
    ax.set_ylim(gminy * unit_conversion_factor, gmaxy * unit_conversion_factor)


def _get_lut_unit_conversion_factor(lut) -> float:
    """
    Return the LUT-to-Cartopy projected-coordinate unit conversion factor.

    QuakeMigrate LUTs may store projected grid coordinates in kilometres, while
    Cartopy projection classes usually expect metres. If the LUT exposes
    unit_conversion_factor, that value is used. Otherwise, the function falls back
    to 1.0.

    Parameters
    ----------
    lut:
        QuakeMigrate lookup-table object.

    Returns
    -------
    float
        Unit conversion factor to apply to projected grid coordinates before plotting.

    """

    return float(getattr(lut, "unit_conversion_factor", 1.0))


def _first_point(points) -> np.ndarray:
    """
    Normalise a coordinate-transform result to a single 3-element point.

    Parameters
    ----------
    points:
        Coordinate transform result from a LUT method.

    Returns
    -------
    numpy.ndarray
        One-dimensional point array.

    """

    points = np.asarray(points, dtype=float)

    if points.ndim == 1:
        return points

    return points[0]

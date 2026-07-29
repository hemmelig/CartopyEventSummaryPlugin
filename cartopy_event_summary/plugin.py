"""
Plugins for located event summaries with Cartopy features enabled.

:copyright:
    2026, Conor A. Bacon
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

from .plotting import event_summary_2d_cartopy


if TYPE_CHECKING:
    import numpy as np
    from quakemigrate.io.core import Run
    from quakemigrate.io.event import Event
    from quakemigrate.lut import LUT


@dataclass
class EventSummary2DCartopyPlugin:
    """
    Plugin that generates a 2-D event summary plot for a located event.

    This plugin runs during the locate stage and calls
    :func:`quakemigrate.plot.event.event_summary_2d` using the located event, lookup
    table, run metadata, and marginalised coalescence map supplied in the runtime
    context.

    Attributes
    ----------
    stage:
        Processing stage in which the plugin is executed.
    order:
        Relative execution order within the stage. Lower values run earlier.
    name:
        Plugin name used in configuration and reporting.
    kind:
        Plugin category.
    overlay_manifest:
        Optional path to an overlay manifest used to add overlays to the plot.
    slice_mode:
        Type of 2-D slice to plot. "maximum" plots the maximum-amplitude slice;
        "surface" plots a fixed-depth surface slice.
    surface_depth:
        Depth of the surface slice when slice_mode is "surface".
    plot_all_stations:
        Whether to plot all stations, rather than only stations used for the event.
    file_type:
        Output file type passed to the plotting function.

    """

    stage: str = "locate_event"
    order: int = 452
    name: str = "EventSummary2DCartopy"
    kind: str = "visualisation"

    overlay_manifest: str | None = None
    slice_mode: Literal["surface", "maximum"] = "maximum"
    surface_depth: float = 0.0
    plot_all_stations: bool = True
    file_type: str = "pdf"

    # Cartopy-specific options
    borders: bool = True
    rivers: bool = True
    cities: bool = True
    city_scale: str = "10m"
    natural_earth_scale: str = "10m"

    def run(
        self,
        event: Event,
        lut: LUT,
        run: Run,
        marginalised_coa_map: np.ndarray,
    ) -> None:
        """
        Generate the 2-D event summary plot.

        Parameters
        ----------
        event:
            Located event to summarise.
        lut:
            Lookup table used for event location.
        run:
            QuakeMigrate run metadata and output configuration.
        marginalised_coa_map:
            Marginalised coalescence map for the located event.

        """

        event_summary_2d_cartopy(
            run,
            event,
            marginalised_coa_map,
            lut,
            slice_mode=self.slice_mode,
            surface_depth=self.surface_depth,
            overlay_manifest=self.overlay_manifest,
            plot_all_stations=self.plot_all_stations,
            file_type=self.file_type,
            borders=self.borders,
            rivers=self.rivers,
            cities=self.cities,
            city_scale=self.city_scale,
            natural_earth_scale=self.natural_earth_scale,
        )

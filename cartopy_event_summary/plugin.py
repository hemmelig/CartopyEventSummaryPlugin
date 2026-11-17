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

    Attributes
    ----------
    stage:
        Processing stage where the plugin runs.
    order:
        Execution ordering relative to other locate plugins.
    name:
        Registered plugin identifier.
    kind:
        Category for plugin discovery.
    overlay_manifest:
        Optional CSV manifest of geographic line/scatter overlays.
    slice_mode:
        Select maximum-coalescence or fixed-depth XY slice.
    surface_depth:
        Depth for the fixed-depth surface slice, in LUT grid units.
    plot_all_stations:
        Plot every station or only those with available data.
    file_type:
        Matplotlib output format.
    borders:
        Draw Natural Earth political boundaries.
    rivers:
        Draw Natural Earth rivers and lakes.
    cities:
        Draw Natural Earth populated places.
    city_scale:
        Populated-places resolution.
    natural_earth_scale:
        Coastline and hydrology resolution.

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
            Located event with stations and waveform data.
        lut:
            Grid and coordinate reference information.
        run:
            Run identifier, output path, and subname.
        marginalised_coa_map:
            Coalescence volume of shape (nx, ny, nz) in LUT grid-index order.

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

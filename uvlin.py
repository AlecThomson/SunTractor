#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run UVlin to remove the Sun from a measurement set.
"""
from pathlib import Path
from potato import get_pos, msutils
from astropy.coordinates import SkyCoord, get_sun, AltAz, EarthLocation
from astropy.time import Time
from astropy import units as u
from casacore.tables import table
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass(slots=True)
class SunTimes:
    rise: Optional[Time] = None
    set: Optional[Time] = None

def get_unique_times(
        ms: Path,
) -> Time:
    # Get the time of the observation
    logger.info(f"Reading {ms} for time information...")
    with table(ms.as_posix(), ack=False) as tab:
        time_arr = np.unique(tab.getcol("TIME"))
        times = Time(time_arr * u.s, format="mjd")
    return times    


def find_sunrise_sunset(
        ms: Path,
        sun_coords: SkyCoord,
        times: Time,
) -> SunTimes:
    # Check when the Sun is above the horizon
    # Get the position of the observatory
    with table(str(ms / "ANTENNA"), ack=False) as tab:
        logger.info(f"Reading {ms / 'ANTENNA'} for position information...")
        pos = EarthLocation.from_geocentric(
            *tab.getcol("POSITION")[0] * u.m  # First antenna is fine
        )
    # Convert to AltAz
    sun_altaz = sun_coords.transform_to(AltAz(obstime=times, location=pos))

    sun_times = SunTimes()
    above_horizon = sun_altaz.alt > 0 * u.deg
    if not above_horizon.any():
        return sun_times
    
    zero_crossings = np.where(np.diff(np.sign(sun_altaz.alt)))[0]

    for crossing in zero_crossings:
        if np.sign(sun_altaz.alt[crossing-1]) < 0 and np.sign(sun_altaz.alt[crossing+1]) > 0:
            sun_times.rise = times[crossing]
            logger.info(f"Sunrise was at {sun_times.rise.iso}")
        else:
            sun_times.set = times[crossing]
            logger.info(f"Sunset was at {sun_times.set.iso}")

    return sun_times


def main(
        ms: Path,
        data_column: str = "DATA",
):
    # Procedure:
    # 1. Get the position of the Sun for all times in the measurement set
    # 2. Check when the Sun is above the horizon
    # 3. Phase the measurement set to the Sun's position
    # 4. Run UVlin on the measurement set for all times when the Sun is above
    #    the horizon
    # 5. Phase rotate the measurement set back to the original phase centre
    # 6. ????
    # 7. Profit

    times = get_unique_times(ms)
    sun_coords = get_sun(times)
    sun_mean_coord = get_sun(times.mean())

    # Check when the Sun is above the horizon
    sun_times = find_sunrise_sunset(ms, sun_coords, times)

    if sun_times.rise is None and sun_times.set is None:
        logger.info("The Sun is never above the horizon. Yay!")
        return

    if sun_times.rise is None:
        logger.info("Sunrise was before the observation started. Using the first time in the observation.")
        sun_times.rise = times[0]
    
    if sun_times.set is None:
        logger.info("Sunset was after the observation ended. Using the last time in the observation.")
        sun_times.set = times[-1]

    orginal_phase = SkyCoord(*msutils.get_phase_direction(ms.as_posix()), unit="deg")
    # Phase the measurement set to the Sun's position
    logger.info(
        f"""
Phase rotating the measurement set to the Sun's mean position: 
    {sun_mean_coord.ra:0.1f} {sun_mean_coord.dec:0.1f}
        """
    )
    msutils.do_rotate(
        ms.as_posix(),
        ra=sun_mean_coord.ra.deg,
        dec=sun_mean_coord.dec.deg,
        datacolumn=data_column,
    )

    # Run UVlin on the measurement set for all times when the Sun is above
    # the horizon
    logger.info(
        f"""
Running UVlin on the measurement set for all times when the Sun is above the horizon:
    {sun_times.rise.iso} - {sun_times.set.iso}
        """
    )
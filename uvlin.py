#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run UVlin to remove the Sun from a measurement set.

See https://yandasoft.readthedocs.io/en/develop/calim/ccontsubtract.html
for more information on UVlin.
"""
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time
from casacore.tables import table
from potato import msutils
from spython.main import Client

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(slots=True)
class SunTimes:
    rise: Optional[Time] = None
    set: Optional[Time] = None


def get_unique_times(
    ms: Path,
) -> Time:
    """Get the unique times in a measurement set.

    Args:
        ms (Path): Measurement set to get the times from.

    Returns:
        Time: Unique times in the measurement set.
    """
    # Get the time of the observation
    logger.info(f"Reading {ms} for time information...")
    with table(ms.as_posix(), ack=False, readonly=True) as tab:
        time_arr = np.unique(tab.getcol("TIME"))
        times = Time(time_arr * u.s, format="mjd")
    return times


def find_sunrise_sunset(
    ms: Path,
    sun_coords: SkyCoord,
    times: Time,
) -> SunTimes:
    """Find the sunrise and sunset times for a measurement set.

    Args:
        ms (Path): Measurement set to find the sunrise and sunset times for.
        sun_coords (SkyCoord): Coordinates of the Sun
        times (Time): Times in the measurement set.

    Returns:
        SunTimes: Sunrise and sunset times.
    """
    # Check when the Sun is above the horizon
    # Get the position of the observatory
    with table(str(ms / "ANTENNA"), ack=False, readonly=True) as tab:
        logger.info(f"Reading {ms / 'ANTENNA'} for position information...")
        pos = EarthLocation.from_geocentric(
            *tab.getcol("POSITION")[0] * u.m  # First antenna is fine
        )
    # Convert to AltAz
    sun_altaz = sun_coords.transform_to(AltAz(obstime=times, location=pos))

    # Find sunrise and sunset
    sun_times = SunTimes()
    above_horizon = sun_altaz.alt > 0 * u.deg
    if not above_horizon.any():
        return sun_times

    zero_crossings = np.where(np.diff(np.sign(sun_altaz.alt)))[0]

    for crossing in zero_crossings:
        if (
            np.sign(sun_altaz.alt[crossing - 1]) < 0
            and np.sign(sun_altaz.alt[crossing + 1]) > 0
        ):
            sun_times.rise = times[crossing]
            logger.info(f"Sunrise was at {sun_times.rise.iso}")
        else:
            sun_times.set = times[crossing]
            logger.info(f"Sunset was at {sun_times.set.iso}")

    return sun_times


def get_yanda(yanda: Union[Path, str]) -> Path:
    """Pull YandaSoft image from dockerhub (or wherver) or load it if it's already
    on the system.

    Args:
        yanda (Union[Path, str]): Path to YandaSoft image or dockerhub image.

    Returns:
        Path: Path to YandaSoft image.
    """
    Client.load(str(yanda))
    if isinstance(yanda, str):
        return Path(Client.pull(yanda))
    return yanda


def uvlin(
    ms: Path,
    sun_times: SunTimes,
    data_column: str = "DATA",
    order: int = 2,
    harmonic: int = 0,
    width: int = 0,
    offset: int = 0,
    yanda: Union[Path, str] = "docker://csirocass/yandasoft:release-openmpi4",
) -> None:
    """Run UVlin on a measurement set.

    Args:
        ms (Path): Measurement set to run UVlin on.
        sun_times (SunTimes): Sunrise and sunset times.
        data_column (str, optional): Column to fit. Defaults to "DATA".
        order (int, optional): Order of polynomial. Defaults to 2.
        harmonic (int, optional): Order of sinusoids. Defaults to 0.
        width (int, optional): Number of channels to fit in box. Defaults to 0.
        offset (int, optional): Offset of box. Defaults to 0.
        yanda (Union[Path, str], optional): Singularity image. Defaults to "docker://csirocass/yandasoft:release-openmpi4".
    """
    # Create a parset for UVlin and write it to disk
    parset = f"""# ccontsubtract parameters
# The measurement set name - the data will be overwritten
CContSubtract.dataset                   = {ms.as_posix()}
CContsubtract.datacolumn                = {data_column}
CContsubtract.doUVlin                   = true
CContsubtract.order                     = {order}
CContsubtract.harmonic                  = {harmonic}
CContsubtract.width                     = {width}
CContsubtract.offset                    = {offset}
CContsubtract.timerange                 = [{(sun_times.rise.mjd * u.day).to(u.s).value},{(sun_times.set.mjd * u.day).to(u.s).value}]
# The model definition - we provide a zero Jy source to keep ccontsubtract happy
CContsubtract.sources.names              = [lsm]
CContsubtract.sources.lsm.direction      = [00:00:00.00,00:00:00.00,J2000]
CContsubtract.sources.lsm.components     = [comp]
CContsubtract.sources.comp.flux.i        = 0.0
CContsubtract.sources.comp.direction.ra  = 0.
CContsubtract.sources.comp.direction.dec = 0.
# The gridding parameters - we specify a gridder to avoid complaints
CContSubtract.gridder                     = Box
    """

    parset_path = ms.with_suffix(".uvlin.parset")
    logger.info(f"Writing parset to {parset_path}")
    with open(parset_path, "w") as f:
        f.write(parset)

    # Get the YandaSoft image and run ccontsubtract with singularity
    command = f"ccontsubtract -c {parset_path.as_posix()}"
    simage = get_yanda(yanda)
    root_dir = ms.parent
    output = Client.execute(
        image=simage.resolve(strict=True).as_posix(),
        command=command.split(),
        bind=f"{root_dir.resolve(strict=True).as_posix()}:{root_dir.resolve(strict=True).as_posix()}",
        return_result=True,
        quiet=False,
        stream=True,
    )
    for line in output:
        logger.info(line.rstrip())


def main(
    ms: Path,
    data_column: str = "DATA",
    order: int = 2,
    harmonic: int = 0,
    width: int = 0,
    offset: int = 0,
    yanda: Union[Path, str] = "docker://csirocass/yandasoft:release-openmpi4",
):
    """Main function to run UVlin on a measurement set.

    Args:
        ms (Path): Measurement set to run UVlin on.
        data_column (str, optional): MS column to use. Defaults to "DATA".
        order (int, optional): Order of poly fit. Defaults to 2.
        harmonic (int, optional): Order of sinusoid fit. Defaults to 0.
        width (int, optional): Width of channel box. Defaults to 0.
        offset (int, optional): Offset of channel box. Defaults to 0.
        yanda (Union[Path, str], optional): YandaSoft image. Defaults to "docker://csirocass/yandasoft:release-openmpi4".
    """
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
        logger.info(
            "Sunrise was before the observation started. Using the first time in the observation."
        )
        sun_times.rise = times[0]

    if sun_times.set is None:
        logger.info(
            "Sunset was after the observation ended. Using the last time in the observation."
        )
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
        datacolumn=[data_column],
    )

    # Run UVlin on the measurement set for all times when the Sun is above
    # the horizon
    logger.info(
        f"""
Running UVlin on the measurement set for all times when the Sun is above the horizon:
    {sun_times.rise.iso} - {sun_times.set.iso}
        """
    )
    try:
        uvlin(
            ms=ms,
            sun_times=sun_times,
            data_column=data_column,
            order=order,
            harmonic=harmonic,
            width=width,
            offset=offset,
            yanda=yanda,
        )
    except Exception as e:
        logger.error(f"Something went wrong with UVlin: {e}")

    # Phase rotate the measurement set back to the original phase centre
    logger.info(
        f"""
Phase rotating the measurement set back to the original phase centre:
    {orginal_phase.ra:0.1f} {orginal_phase.dec:0.1f}
        """
    )
    msutils.do_rotate(
        ms.as_posix(),
        ra=orginal_phase.ra.deg,
        dec=orginal_phase.dec.deg,
        datacolumn=data_column,
    )

    logger.info("Done!")


def cli():
    """Command line interface for uvlin."""
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "ms",
        type=str,
        help="Measurement set to remove the Sun from",
    )
    parser.add_argument(
        "--data-column",
        type=str,
        default="DATA",
        help="Data column to use",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=2,
        help="Order of the polynomial to fit",
    )
    parser.add_argument(
        "--harmonic",
        type=int,
        default=0,
        help="Order of harmonic to fit",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Width of the window to fit",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset of the window to fit",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--hosted-yanda",
        type=str,
        default="docker://csirocass/yandasoft:release-openmpi4",
        help="Docker or Singularity image for wsclean",
    )
    group.add_argument(
        "--local-yanda",
        type=str,
        default=None,
        help="Path to local wsclean Singularity image",
    )
    args = parser.parse_args()

    yanda = Path(args.local_yanda) if args.local_yanda else args.hosted_yanda

    main(
        Path(args.ms),
        data_column=args.data_column,
        order=args.order,
        harmonic=args.harmonic,
        width=args.width,
        offset=args.offset,
        yanda=yanda,
    )


if __name__ == "__main__":
    sys.exit(cli())

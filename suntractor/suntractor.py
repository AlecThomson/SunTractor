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
from itertools import combinations
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time
from casacore.tables import makecoldesc, table, taql
from potato import msutils
from shade_ms.main import main as shade_ms
from spython.main import Client
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(slots=True)
class SunTimes:
    rise: Optional[Time] = None
    set: Optional[Time] = None


def uvlin_plot(
    ms: Path,
    antenna_1: int,
    antenna_2: int,
    start_time: u.Quantity,
    end_time: u.Quantity,
    input_column: str = "DATA",
    output_column: str = "CORRECTED_DATA",
    out_path: Optional[Path] = None,
):
    with table(ms.as_posix(), readonly=True, ack=False) as tab:
        # Get DATA for 'test_time'
        data = taql(
            f"select {input_column} from $tab where TIME >= {start_time.to(u.s).value} and TIME <= {end_time.to(u.s).value} and ANTENNA1!=ANTENNA2 and ANTENNA1=={antenna_1} and ANTENNA2=={antenna_2}"
        ).getcol(f"{input_column}")
        corrected_data = taql(
            f"select {output_column} from $tab where TIME >= {start_time.to(u.s).value} and TIME <= {end_time.to(u.s).value} and ANTENNA1!=ANTENNA2 and ANTENNA1=={antenna_1} and ANTENNA2=={antenna_2}"
        ).getcol(f"{output_column}")
        flag = taql(
            f"select FLAG from $tab where TIME >= {start_time.to(u.s).value} and TIME <= {end_time.to(u.s).value} and ANTENNA1!=ANTENNA2 and ANTENNA1=={antenna_1} and ANTENNA2=={antenna_2}"
        ).getcol("FLAG")
        data[flag] = np.nan + 1j * np.nan
        corrected_data[flag] = np.nan + 1j * np.nan

    model_data = data - corrected_data
    _correlations = ["XX", "XY", "YX", "YY"]
    _names = ["Original", "Model", "Residual"]

    xx, xy, yx, yy = data.T
    mxx, mxy, myx, myy = model_data.T
    rxx, rxy, ryx, ryy = corrected_data.T

    for i, corrs in enumerate(
        zip(
            (xx, xy, yx, yy),
            (mxx, mxy, myx, myy),
            (rxx, rxy, ryx, ryy),
        )
    ):
        fig, axs = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(15, 15))
        fig.suptitle(f"Antenna {antenna_1} - {antenna_2} - {_correlations[i]}")
        for f, func in enumerate((np.real, np.imag, np.abs, np.angle)):
            if f == 0 or f == 1:
                vmin, vmax = -10, 10
                cmap = "RdBu_r"
            elif f == 2:
                vmin, vmax = 0, 10
                cmap = "viridis"
            else:
                vmin, vmax = -np.pi, np.pi
                cmap = "twilight_shifted"
            for j, cor in enumerate(corrs):
                ax = axs[f, j]
                im = ax.imshow(
                    func(cor),
                    aspect="auto",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    origin="lower",
                )
                fig.colorbar(im)
                ax.set_xlabel("Time step")
                ax.set_ylabel("Channel")
                ax.set_title(f"{func.__name__}({_correlations[i]}) - {_names[j]}")
        if out_path is not None:
            outf = out_path / f"{antenna_1}-{antenna_2}-{_correlations[i]}.pdf"
            fig.savefig(out_path / f"{antenna_1}-{antenna_2}-{_correlations[i]}.pdf")
            logger.info(f"Saved plot to {outf}")

        plt.close(fig)


def make_plots(
    ms: Path,
    input_column: str,
    output_column: str,
    sun_times: SunTimes,
    sub_dir: Optional[str] = None,
    n_antennas: int = 3,
):
    # Plot ± 1 hour around sunrise and sunset
    time_start = sun_times.rise.value * u.day - 1 * u.hour
    time_end = sun_times.rise.value * u.day + 1 * u.hour

    out_path = ms.parent / "suntractor_plots"
    if sub_dir is not None:
        out_path /= sub_dir
    # Make the output directory if it doesn't exist
    out_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory {out_path}")

    # Plot baselines between n_antennas
    for ant1, ant2 in tqdm(
        combinations(range(n_antennas), 2), desc="Plotting baselines"
    ):
        uvlin_plot(
            ms=ms,
            antenna_1=ant1,
            antenna_2=ant2,
            start_time=time_start,
            end_time=time_end,
            out_path=out_path,
            input_column=input_column,
            output_column=output_column,
        )


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

    # Buffer by the diameter of the Sun
    sun_diameter = 32 * u.arcmin
    sun_speed = 360 * u.deg / u.day  # Due to Earth's rotation
    buffer_time = (
        sun_diameter / sun_speed
    ).decompose()  # Time to add to sunrise and sunset

    for crossing in zero_crossings:
        if (
            np.sign(sun_altaz.alt[crossing - 1]) < 0
            and np.sign(sun_altaz.alt[crossing + 1]) > 0
        ):
            sun_times.rise = times[crossing] - 3 * buffer_time
            logger.info(f"Sunrise was at {sun_times.rise.iso}")
        else:
            sun_times.set = times[crossing] + 3 * buffer_time
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
    threshold: float = 0.0,
    MinUV: Optional[float] = None,
    MaxUV: Optional[float] = None,
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
        threshold (float, optional): Threshold for fitting. Defaults to 0.0.
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
CContsubtract.threshold                 = {threshold}
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

    for option in ["MinUV", "MaxUV"]:
        optional_val = locals()[option]
        if optional_val is not None:
            parset += f"CContsubtract.{option} = {optional_val}\n"

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
    input_column: str = "DATA",
    output_column: str = "DATA",
    order: int = 2,
    harmonic: int = 0,
    width: int = 0,
    offset: int = 0,
    threshold: float = 0.0,
    MinUV: Optional[float] = None,
    MaxUV: Optional[float] = None,
    yanda: Union[Path, str] = "docker://csirocass/yandasoft:release-openmpi4",
    overwrite: bool = False,
    plot: bool = False,
    plot_n_antennas: int = 3,
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

    # Check the columns in the measurement set
    if input_column == output_column:
        logger.warning(
            "Input and output columns are the same. Data will be overwritten."
        )

    else:
        logger.info(f"Input column: {input_column}")
        logger.info(f"Output column: {output_column}")
        # Check if the output column already exists
        with table(ms.as_posix(), ack=False, readonly=False) as tab:
            if output_column in tab.colnames() and not overwrite:
                raise ValueError(f"Output column {output_column} already exists.")
            if output_column in tab.colnames() and overwrite:
                logger.warning(f"Overwriting output column {output_column}")
                tab.removecols(output_column)
            # Copy the input column to the output column
            logger.info(f"Copying {input_column} to {output_column}")
            desc = makecoldesc(output_column, tab.getcoldesc(input_column))
            desc["name"] = output_column
            tab.addcols(desc)
            tab.putcol(output_column, tab.getcol(input_column))
            tab.flush()

    times = get_unique_times(ms)
    sun_coords = get_sun(times)
    sun_mean_coord = get_sun(times.mean())

    # Check when the Sun is above the horizon
    sun_times = find_sunrise_sunset(ms, sun_coords, times)

    # Handle when sunrise or sunset is outside the observation
    if sun_times.rise is None and sun_times.set is None:
        logger.info("The Sun was never above the horizon. Yay!")
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
        datacolumn=[input_column, output_column],
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
            data_column=output_column,
            order=order,
            harmonic=harmonic,
            width=width,
            offset=offset,
            yanda=yanda,
            threshold=threshold,
            MinUV=MinUV,
            MaxUV=MaxUV,
        )
    except Exception as e:
        logger.error(f"Something went wrong with UVlin: {e}")

    # Make plots before phase rotating back
    if plot:
        make_plots(
            ms=ms,
            input_column=input_column,
            output_column=output_column,
            sun_times=sun_times,
            sub_dir="sun_phase",
            n_antennas=plot_n_antennas,
        )

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
        datacolumn=[input_column, output_column],
    )

    # Make plots after phase rotating back
    if plot:
        make_plots(
            ms=ms,
            input_column=input_column,
            output_column=output_column,
            sun_times=sun_times,
            sub_dir="target_phase",
            n_antennas=plot_n_antennas,
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
        "--input-column",
        type=str,
        default="DATA",
        help="Data column to use",
    )
    parser.add_argument(
        "--output-column",
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Threshold for fitting",
    )
    parser.add_argument(
        "--minuv",
        type=float,
        default=None,
        help="Minimum UV distance (in metres) to fit",
    )
    parser.add_argument(
        "--maxuv",
        type=float,
        default=None,
        help="Maximum UV distance (in metres)   to fit",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output column if it already exists",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Make plots of the data before and after UVlin",
    )
    parser.add_argument(
        "--nantennas",
        type=int,
        default=3,
        help="Number of antennas to plot",
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
        input_column=args.input_column,
        output_column=args.output_column,
        order=args.order,
        harmonic=args.harmonic,
        width=args.width,
        offset=args.offset,
        threshold=args.threshold,
        MinUV=args.minuv,
        MaxUV=args.maxuv,
        yanda=yanda,
        overwrite=args.overwrite,
        plot=args.plot,
        plot_n_antennas=args.nantennas,
    )


if __name__ == "__main__":
    sys.exit(cli())

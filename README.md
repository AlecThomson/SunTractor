# SunTractor
Subtract the Sun from radio interferometric visibilities.

See YandaSoft's [ccontsubtract](https://yandasoft.readthedocs.io/en/latest/calim/ccontsubtract.html) for more detail.

## Installation

Obtain and install [Singularity](https://docs.sylabs.io/guides/3.5/user-guide/index.html), and Python (I recommend [Miniforge](https://github.com/conda-forge/miniforge) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)).

Install the Python scripts:
```
pip install git+https://github.com/AlecThomson/SunTractor.git
```

## Usage

```
$ suntractor -h
usage: suntractor [-h] [--input-column INPUT_COLUMN] [--output-column OUTPUT_COLUMN] [--order ORDER] [--harmonic HARMONIC] [--width WIDTH] [--offset OFFSET] [--plot] [--hosted-yanda HOSTED_YANDA | --local-yanda LOCAL_YANDA]
                  ms

Run UVlin to remove the Sun from a measurement set. See https://yandasoft.readthedocs.io/en/develop/calim/ccontsubtract.html for more information on UVlin.

positional arguments:
  ms                    Measurement set to remove the Sun from

options:
  -h, --help            show this help message and exit
  --input-column INPUT_COLUMN
                        Data column to use (default: DATA)
  --output-column OUTPUT_COLUMN
                        Data column to use (default: DATA)
  --order ORDER         Order of the polynomial to fit (default: 2)
  --harmonic HARMONIC   Order of harmonic to fit (default: 0)
  --width WIDTH         Width of the window to fit (default: 0)
  --offset OFFSET       Offset of the window to fit (default: 0)
  --plot                Make plots of the visibilities (default: False)
  --hosted-yanda HOSTED_YANDA
                        Docker or Singularity image for wsclean (default: docker://csirocass/yandasoft:release-openmpi4)
  --local-yanda LOCAL_YANDA
                        Path to local wsclean Singularity image (default: None)
```

## What does this actually do?
Procedure:
1. Get the position of the Sun for all times in the measurement set
2. Check when the Sun is above the horizon
3. Phase the measurement set to the Sun's position
4. Run UVlin on the measurement set for all times when the Sun is above
   the horizon
5. Phase rotate the measurement set back to the original phase centre
6. ????
7. Profit

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run UVlin to remove the Sun from a measurement set.
"""
from pathlib import Path
from potato import get_pos, msutils
from astropy.coordinates import SkyCoord, get_sun

def main():
    # Procedure:
    # 1. Get the position of the Sun for all times in the measurement set
    # 2. Check when the Sun is above the horizon
    # 3. Phase the measurement set to the Sun's position
    # 4. Run UVlin on the measurement set for all times when the Sun is above
    #    the horizon
    # 5. Phase rotate the measurement set back to the original phase centre
    # 6. ????
    # 7. Profit
    pass
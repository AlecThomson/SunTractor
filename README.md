# SunTractor
Subtract the Sun from radio interferometric visibilities.

## Installation

```
pip
```

## Usage

```
suntractor -h
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

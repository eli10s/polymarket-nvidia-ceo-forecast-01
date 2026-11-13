"""Polymarket "Next CEO of NVIDIA?" minute-by-minute quantile forecaster.

Reads every <id>.in case file from the input directory and writes one
<id>.out forecast file per case to the output directory.

Method (per market, from its own 72h price history only):
  * The median forecast is the price at the checkpoint. Prediction-market
    prices are (very nearly) martingales over a 4h horizon: every attempt
    to extrapolate a recent drift, or to mean-revert the last price toward
    a short average, scored worse on the practice cases, so the median
    stays put.
  * The spread comes from the market's own recent history, not from a
    fixed constant. For a grid of horizons h we take the empirical
    distribution of h-step returns over the last 12 hours (overlapping
    windows), read off its interquartile range, and turn it into a
    per-horizon scale sigma(h) = IQR(h) / 1.349. Using h-step returns
    directly (instead of scaling up 1-minute returns) matters: 1-minute
    moves are heavy-tailed and strongly mean-reverting after jumps, so a
    random-walk extrapolation of 1-minute volatility wildly overstates
    how far the price can travel in 4 hours.
  * sigma(h) is floored (a market that has been dead flat can still jump)
    and capped (a market that just had a wild hour should not get an
    absurdly wide band), both scaling with sqrt(h), and interpolated
    between grid horizons linearly in sqrt(h).
  * Quantiles are median +/- z * sigma(h) with deliberately narrower
    inner quantiles (z = 0.50 at 25%/75%) and wider tails (z = 1.80 at
    5%/95%) than the normal's, matching the empirical shape of these
    returns: a tall spike at zero with fat jump tails. All quantiles are
    clamped to [0, 1], rounded to 6 decimals, and forced non-decreasing.

Everything is deterministic: no randomness, no wall-clock, no network,
standard library only. The same input always produces the same bytes.
"""

import bisect
import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Multipliers on sigma(h) for each level: narrower inner quartiles and
# wider tails than a normal's (-1.645, -0.674, 0, 0.674, 1.645), tuned on
# the six practice cases (practice loss 0.3265 vs the house baseline's
# 0.3925 on the five cases whose answers the practice inputs reveal).
ZS = (-1.80, -0.50, 0.0, 0.50, 1.80)

# Horizons (minutes) at which the empirical h-step return distribution is
# measured; other horizons interpolate between these in sqrt(h).
GRID = (1, 2, 4, 8, 16, 32, 64, 128, 240)
# Only the most recent 12 hours of prices feed the spread: the volatility
# regime changes faster than the 72h history, and older hours hurt.
WINDOW = 720
# Floor and cap on sigma, in price units per square-root minute.
SIGMA_FLOOR = 0.0003
SIGMA_CAP = 0.003
# Fallback spread (the house baseline's) for a market too short to measure.
FALLBACK_SIGMA = 0.0025


def clamp01(x):
    """Clamp to [0, 1]."""
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def round6(x):
    """Round to 6 decimals, clamped to [0, 1]; never -0."""
    return round(clamp01(x), 6) + 0.0


def _quantile(sorted_values, tau):
    """Linear-interpolation empirical quantile of a sorted list."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    pos = tau * (n - 1)
    lo = int(pos)
    frac = pos - lo
    hi = lo + 1 if lo + 1 < n else lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def hstep_iqrs(prices):
    """Interquartile range of h-step returns for each grid horizon.

    Uses overlapping windows over the last WINDOW prices: the IQR of
    {prices[i] - prices[i-h]} for every i. Returns {h: iqr} for the grid
    horizons that fit inside the history; empty when too short.
    """
    window = prices[-WINDOW:]
    n = len(window)
    iqrs = {}
    for h in GRID:
        if h >= n:
            continue
        steps = sorted(window[i] - window[i - h] for i in range(h, n))
        iqrs[h] = _quantile(steps, 0.75) - _quantile(steps, 0.25)
    return iqrs


def sigma_at(iqrs, h):
    """Spread at horizon h (minutes) from the grid IQRs.

    Interpolates the IQR linearly in sqrt(h), converts to a normal scale
    with / 1.349, then floors and caps it, both growing with sqrt(h).
    """
    horizons = sorted(iqrs)
    if not horizons:
        return FALLBACK_SIGMA * math.sqrt(h)
    if h <= horizons[0]:
        iqr = iqrs[horizons[0]]
    elif h >= horizons[-1]:
        iqr = iqrs[horizons[-1]]
    else:
        j = bisect.bisect_right(horizons, h) - 1
        h0, h1 = horizons[j], horizons[j + 1]
        w = (math.sqrt(h) - math.sqrt(h0)) / (math.sqrt(h1) - math.sqrt(h0))
        iqr = iqrs[h0] * (1.0 - w) + iqrs[h1] * w
    root_h = math.sqrt(h)
    sigma = iqr / 1.349
    if sigma < SIGMA_FLOOR * root_h:
        sigma = SIGMA_FLOOR * root_h
    elif sigma > SIGMA_CAP * root_h:
        sigma = SIGMA_CAP * root_h
    return sigma


def forecast_market(market, horizon):
    """240 (or horizon) rows of five quantiles for one market."""
    prices = market["prices"]
    if not prices:
        raise ValueError("market %s has no prices" % market["id"])
    last = prices[-1]
    iqrs = hstep_iqrs(prices)
    rows = []
    for h in range(1, horizon + 1):
        sigma = sigma_at(iqrs, h)
        row = [round6(last + z * sigma) for z in ZS]
        # Keep each row non-decreasing (clamping at 0/1 can tie levels).
        for j in range(1, len(row)):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def forecast_case(case):
    """One forecast object for a parsed case dict."""
    horizon = int(case["horizon_minutes"])
    forecasts = [
        {"id": market["id"], "q": forecast_market(market, horizon)}
        for market in case["markets"]
    ]
    return {"forecasts": forecasts}


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in in_dir.iterdir() if p.name.endswith(".in"))
    for in_file in inputs:
        case_id = in_file.name[: -len(".in")]
        with open(in_file, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_file = out_dir / (case_id + ".out")
        with open(out_file, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(result, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()

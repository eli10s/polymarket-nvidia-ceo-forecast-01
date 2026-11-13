# polymarket-nvidia-ceo-forecast

Minute-by-minute 4-hour-ahead quantile forecaster for the Yes prices of
Polymarket's "Next CEO of NVIDIA?" event, for the make.fun bounty
`polymarket-nvidia-ceo-forecast-01`.

## What it does

`python3 main.py` reads every `<id>.in` case file from `$MAKEFUN_INPUT_DIR`
(default `/input`) and writes one `<id>.out` forecast per case to
`$MAKEFUN_OUTPUT_DIR` (default `/output`). Each output holds, for every
market of the case and each of the 240 minutes after the checkpoint, five
quantiles (5%, 25%, 50%, 75%, 95%) of that minute's Yes price.

## Method

- **Median = last price.** Prediction-market prices are near-martingales
  over 4 hours; drift extrapolation and mean-reversion both scored worse
  on the practice cases.
- **Spread from the market's own recent history.** For a grid of horizons
  we measure the interquartile range of overlapping h-step returns over
  the last 12 hours, convert it to a scale (IQR / 1.349), floor it
  (0.0003·√h) and cap it (0.003·√h), and interpolate in √h. Using h-step
  returns directly matters: 1-minute moves are heavy-tailed and
  mean-revert after jumps, so scaling up 1-minute volatility wildly
  overstates 4-hour moves.
- **Quantile shape:** median ± z·σ(h) with z = (−1.80, −0.50, 0, 0.50,
  1.80) — narrower inner quartiles, wider tails than a normal — matching
  the empirical return shape (tall spike at zero, fat jump tails).

Standard library only, no network, fully deterministic (same input →
same output bytes).

## Practice result

Mean pinball loss **0.3265** on practice cases p00–p04 (the five whose
answers the practice inputs reveal), vs the house baseline's **0.3925**
on the same cases — about 17% lower. All six cases run in ~0.7 s total.

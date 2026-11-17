# GTA VI Twitch CCU forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ferofast/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ferofast/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Comparables-blend log-ratio forecast (half OLS + quarter local-linear + quarter kNN-12) for GTA VI's launch-day Twitch viewers, hour by hour.

The program reads one case — the launch day's hour-1 Twitch print plus 38–42 past big launches' hourly curves — and writes, for each of hours 2 to 24, five quantiles (5%, 25%, 50%, 75%, 95%) of that hour's concurrent-viewer count. Everything is anchored on the hour-1 print in log space: it forecasts log-ratios `ln(v_h / v_1)`, never absolute levels, because big launches have almost flat curves (peak only ~0.9–1.3x hour 1) while small launches grow 2–5x by evening, so the launch's size decides the curve's shape.

Three estimators of the log-ratio distribution per hour, blended geometrically with weights 1/2 + 1/4 + 1/4:

- a quadratic OLS of the log-ratio on `[ln p1, (ln p1)^2]`, spreads from the empirical quantiles of its residuals;
- a Gaussian-kernel local-linear fit in `ln(p1)` (bandwidth 2.0), spreads from the residuals of nearby launches;
- the empirical quantiles of the log-ratios of the 12 nearest launches by `ln(p1)`.

The blend hedges regime risk: no single estimator wins everywhere, and the final is a single launch day. Channels, pre-release level and platform flags were tested and left out — leave-one-out says they add noise, not signal. All fitting happens at run time on the case file's own comparables; nothing is hardcoded, and the program is deterministic (two runs on the same case write byte-identical output).

## Validation

Leave-one-out over the 37 usable comparables in the kit, scored with the contest's exact pinball loss on the log scale:

| method | LOO loss |
|---|---|
| blend (this entry) | **0.376** |
| quadratic OLS alone | 0.395 |
| kernel local-linear alone | 0.387 |
| kNN alone | 0.450 |
| house persistence baseline | 0.930 |

On the console-only test slice (the closest to GTA VI, a console launch) the blend scores 0.224. The practice cases are scored by the server after submission.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `*.in` in the input directory and writes one `<id>.out` per case: `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}`, exactly 23 rows, hours 2 to 24.

---

<sub>Entered on [make.fun](https://make.fun/?ref=0280650f&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>

# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/CharmLynn/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/CharmLynn/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each PlayStation 5 trophy in week one, as Exophase reports it. `main.py` (Python 3.13, standard library only) reads a case of Exophase reads and writes five quantiles of every trophy's earn rate at the target.

The model: a trophy's rate is earners over tracked players, and both counts only grow, so each is fitted with a saturating exponential in hours since launch and extrapolated separately. Fitting them apart matters — when Exophase refreshes and the tracked pool jumps, the rates fall even though nobody un-earns a trophy. A saturating curve matters too: both counts level off, and the rate at the target turns on which levels off first. Quantiles sit on the logit scale around the median, with a spread that widens with the horizon.

## Run it

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=mpd17n5r&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>

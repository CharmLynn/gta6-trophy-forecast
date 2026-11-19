"""Forecast GTA VI trophy earn rates on Exophase a week after launch.

For every trophy of Grand Theft Auto VI on PlayStation 5, forecast five
quantiles of its earn rate (percent of the players Exophase tracks who have
earned it) at the case's target, from the history of Exophase reads the case
carries.

The model
----------
A trophy's rate is earners over tracked players, r = E / N. Both counts only
grow, and both grow slower over time, so each is fitted with a saturating
exponential in hours since launch and extrapolated separately:

    Q(t) = Q_inf - A * exp(-t / tau)

For a fixed tau this is linear in exp(-t / tau), so the fit is a least
squares over a grid of tau values, taking the tau with the smallest
residual. Fitting the two counts apart matters: when Exophase refreshes and
the tracked pool jumps, the rates fall even though nobody un-earns a trophy,
and a model that only fits the rate itself cannot see that coming. A
saturating curve matters too: a power law would let the tracked pool grow
forever and the earners with it, but both level off, and the rate at the
target turns on which levels off first. Earners never fall below their last
count and never exceed the tracked players.

The spread
-----------
Quantiles sit on the logit scale around the median. The spread combines a
random-walk term that widens with the horizon (0.5 logit units per
square-root day, the house baseline's) with a fixed term for model
uncertainty, so a flat history (no information about growth) still forecasts
an honest, wide band.

How the grader runs it: `python3 -P -s /bot/main.py` in python:3.13-slim,
standard library only, no network. It reads every <id>.in in the input
directory and writes <id>.out to the output directory:
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}, one row per
trophy, each q a finite number from 0 to 100, non-decreasing, the file at
most 65,536 bytes. Outside the grader, MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR stand in for /input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Random-walk spread, logit units per square-root day.
SIGMA_RW = 0.5
# Extra spread for model uncertainty, logit units.
SIGMA_MODEL = 0.4
# A trophy no one has earned yet still counts as half a player, so its
# count stays positive.
EARNER_FLOOR = 0.5
# Timescales (hours) tried for the saturating fit, log-spaced from 6 hours
# to 30 days: a short tau saturates fast, a long one is nearly a straight
# line over the history.
TAUS = [6 * (5 ** (k / 4.0)) for k in range(13)]


def parse_time(text):
    """ISO 8601 UTC timestamp to epoch seconds."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100 - p))


def inv_logit(lg):
    return 100 / (1 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up, as the house baseline does."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def fit_saturating(ts, qs):
    """Fit Q(t) = Q_inf - A * exp(-t / tau) by least squares over a grid of
    tau. Returns (Q_inf, A, tau, rss). With fewer than two distinct times,
    or a flat series, returns the last value with tau 0 (no growth)."""
    n = len(ts)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    q_last = qs[-1]
    if n < 2 or max(ts) - min(ts) <= 0 or max(qs) - min(qs) <= 0:
        return q_last, 0.0, 0.0, 0.0
    best = None
    for tau in TAUS:
        zs = [math.exp(-t / tau) for t in ts]
        # Least squares Q = Q_inf - A * z: regress Q on z.
        mz = sum(zs) / n
        mq = sum(qs) / n
        szz = sum((z - mz) ** 2 for z in zs)
        if szz <= 0:
            continue
        szq = sum((z - mz) * (q - mq) for z, q in zip(zs, qs))
        slope = szq / szz  # this is -A
        q_inf = mq - slope * mz
        rss = sum((q - (q_inf + slope * z)) ** 2 for z, q in zip(zs, qs))
        if best is None or rss < best[3]:
            best = (q_inf, -slope, tau, rss)
    if best is None:
        return q_last, 0.0, 0.0, 0.0
    return best


def predict_saturating(fit, t):
    q_inf, a, tau, _ = fit
    if tau <= 0:
        return q_inf
    return q_inf - a * math.exp(-t / tau)


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")

    release = case["game"]["release_date"]
    launch_epoch = parse_time(release + "T00:00:00Z")
    target_epoch = parse_time(case["target"])
    t_target = max((target_epoch - launch_epoch) / 3600.0, 0.0)
    horizon_days = case["horizon_minutes"] / 1440.0

    # Hours since launch and tracked players per read, oldest first.
    ts = []
    ns = []
    for read in history:
        ts.append(max((parse_time(read["at"]) - launch_epoch) / 3600.0, 0.0))
        ns.append(float(read["players_tracked"]))
    n_last = ns[-1]
    n_fit = fit_saturating(ts, ns)
    n_pred = max(predict_saturating(n_fit, t_target), n_last)

    spread = math.sqrt((SIGMA_RW * math.sqrt(horizon_days)) ** 2 + SIGMA_MODEL ** 2)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        es = []
        use_ts = []
        for j, read in enumerate(history):
            rate = read["rates"][i]
            if rate is None:
                continue
            es.append(max(rate * read["players_tracked"] / 100.0, EARNER_FLOOR))
            use_ts.append(ts[j])
        e_last = es[-1] if es else EARNER_FLOOR
        e_fit = fit_saturating(use_ts, es)
        e_pred = predict_saturating(e_fit, t_target)
        # Earners only grow, and never above the tracked players.
        e_pred = min(max(e_pred, e_last), n_pred)
        median = clamp_rate(100 * e_pred / n_pred)
        lg = logit(median)
        qs = [round4(inv_logit(lg + z * spread)) for z in Z]
        qs = [max(qs[k], qs[k - 1]) if k else qs[k] for k in range(5)]
        qs = [min(100, max(0, q)) for q in qs]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

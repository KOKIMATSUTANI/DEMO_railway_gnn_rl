"""Figure generation (static PNGs via matplotlib, Agg backend).

Colours follow a fixed categorical order (never cycled): GNN = blue, MLP =
orange; reference levels and neutral marks use grey/ink tokens.  Every figure
has a single y-axis and labelled units.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .stats import moving_average  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"gnn": "#2a78d6", "mlp": "#eb6834"}
REF_STYLES = ["--", ":", "-."]


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _curve(rows, key, path, title, ylabel, window, refs=None):
    if not rows:
        raise ValueError("no training rows")
    for r in rows:
        for k in ("kind", "seed", "episode", key):
            if k not in r:
                raise KeyError(k)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for kind in [k for k in SERIES if any(r["kind"] == k for r in rows)]:
        seeds = sorted({r["seed"] for r in rows if r["kind"] == kind})
        per_seed = []
        for s in seeds:
            sr = sorted((r for r in rows if r["kind"] == kind and r["seed"] == s),
                        key=lambda r: r["episode"])
            vals = [float(r[key]) for r in sr]
            ax.scatter([r["episode"] for r in sr], vals, s=3, color=SERIES[kind], alpha=0.12,
                       linewidths=0)
            per_seed.append(moving_average(vals, window))
        n = min(len(p) for p in per_seed)
        arr = np.array([p[:n] for p in per_seed])
        mean, sd = arr.mean(axis=0), arr.std(axis=0)
        x = np.arange(n)
        ax.plot(x, mean, color=SERIES[kind], linewidth=2,
                label=f"{kind.upper()}: mean over {len(seeds)} seeds of {window}-ep. moving avg.")
        ax.fill_between(x, mean - sd, mean + sd, color=SERIES[kind], alpha=0.18, linewidth=0,
                        label=f"{kind.upper()}: ±1 SD across seeds")
    for (name, level), ls in zip((refs or {}).items(), REF_STYLES):
        ax.axhline(level, color=INK2, linestyle=ls, linewidth=1.2, label=f"{name}: {level:.2f}")
    _style(ax, title, "training episode", ylabel)
    ax.legend(fontsize=8, frameon=False)
    _save(fig, path)


def plot_return_curve(rows, path, window=25):
    """Figure A: per-episode training return (dots = raw sampled episodes)."""
    _curve(rows, "episode_return", path,
           "A. Training episode return (higher = less waiting; raw dots, smoothed lines)",
           "episode return (= −total waiting, train-min)", window)


def plot_waiting_curve(rows, path, refs, window=25):
    """Figure B: per-episode training total waiting with evaluation-set reference levels."""
    _curve(rows, "total_waiting", path,
           "B. Training total waiting time (lower is better)",
           "total waiting time per episode (train-min)", window, refs)


def plot_step_trace(step_rewards, path_trace, path_cum, title=""):
    """Figure C: immediate reward per simulation step and, separately, the cumulative return."""
    if len(step_rewards) == 0:
        raise ValueError("empty reward trace")
    t = np.arange(len(step_rewards))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(t, step_rewards, width=0.7, color=SERIES["gnn"])
    ax.axhline(0, color=INK2, linewidth=0.8)
    _style(ax, f"C1. Immediate reward per step - {title}", "simulation time t (min)",
           "reward r_t = −(W_t − W_{t−1}) (train-min)")
    _save(fig, path_trace)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.step(t, np.cumsum(step_rewards), where="post", color=SERIES["gnn"], linewidth=2)
    _style(ax, f"C2. Cumulative return - {title}", "simulation time t (min)",
           "cumulative return Σ r (train-min)")
    _save(fig, path_cum)


def plot_comparison(summary, path, order=None):
    """Figure D: mean total waiting per dispatcher with 95% CI error bars."""
    if not summary:
        raise ValueError("empty summary")
    names = order or list(summary)
    means = [summary[n]["mean"] for n in names]
    err = np.array([[summary[n]["mean"] - summary[n]["ci_lo"] for n in names],
                    [summary[n]["ci_hi"] - summary[n]["mean"] for n in names]])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [SERIES.get(n.split("-")[0], "#8a8984") for n in names]
    ax.bar(names, means, yerr=err, capsize=4, color=colors, width=0.6,
           error_kw={"ecolor": INK, "elinewidth": 1})
    for i, m in enumerate(means):
        ax.text(i, m / 2, f"{m:.2f}", ha="center", color="white", fontsize=9)
    _style(ax, "D. Evaluation set: mean total waiting per scenario (95% CI)", "dispatcher",
           "mean total waiting time (train-min)")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, path)


def plot_paired_differences(diffs, path, label=""):
    """Figure E: distribution of per-scenario waiting differences (policy − FCFS)."""
    if len(diffs) == 0:
        raise ValueError("no differences")
    d = np.asarray(diffs, dtype=float)
    lo, hi = np.floor(d.min()) - 0.5, np.ceil(d.max()) + 0.5
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(d, bins=np.arange(lo, hi + 1), color=SERIES["gnn"], edgecolor=SURFACE, linewidth=2)
    ax.axvline(0, color=INK2, linewidth=1, linestyle="--")
    ax.axvline(d.mean(), color=INK, linewidth=1.5, label=f"mean = {d.mean():.2f}")
    _style(ax, f"E. Per-scenario difference in total waiting: {label}",
           "difference in total waiting (train-min; < 0 means the policy waits less)",
           "number of evaluation scenarios")
    ax.legend(frameon=False)
    _save(fig, path)


def plot_sensitivity(table, path):
    """Figure F: sensitivity study, mean validation waiting per setting with 95% t-CI."""
    if not table:
        raise ValueError("empty table")
    names = [r["setting"] for r in table]
    means = np.array([r["val_mean"] for r in table])
    lo = np.array([r["val_lo"] for r in table])
    hi = np.array([r["val_hi"] for r in table])
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(names))
    ax.errorbar(means, y, xerr=[np.nan_to_num(means - lo), np.nan_to_num(hi - means)], fmt="o",
                color=SERIES["gnn"], ecolor=INK2, capsize=4, markersize=7)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    _style(ax, "F. Sensitivity study: validation-set mean total waiting\n"
               "(dot = mean over training seeds, bar = 95% t-interval over seeds)",
           "mean total waiting time on validation set (train-min)", "setting")
    _save(fig, path)

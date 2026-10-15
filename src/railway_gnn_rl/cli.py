"""Command line entry point: `railway-demo <command>`.

  train        train GNN and MLP policies for the given seeds (parallel processes)
  evaluate     exact optimum, rule baselines and saved checkpoints on the saved
               evaluation / held-out scenarios; summary tables; reproducibility check
  sensitivity  small sensitivity study (depth, hidden size, learning rate, reward)
  figures      figures A-F from the saved CSV/JSON files
  all          train -> evaluate -> sensitivity -> figures

Everything is written below --out (default: outputs/).
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from dataclasses import asdict, replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from . import baselines as B
from .agent import policy_dispatcher, run_episode
from .experiment import (EVAL_COLUMNS, SETS, TRAIN_COLUMNS, TrainConfig, _paired, evaluation_rows,
                         load_scenarios, save_scenarios, scenario_set, summarize, train_run,
                         write_csv)
from .models import make_policy
from .optimum import solve
from .stats import converged, mean_ci_t

KINDS = ("gnn", "mlp")
SENSITIVITY = [  # (name, overrides of the main configuration)
    ("base (L=2, H=32, lr=1e-3, wait)", {}),
    ("layers=1", {"layers": 1}),
    ("layers=3", {"layers": 3}),
    ("hidden=64", {"hidden": 64}),
    ("lr=3e-3", {"lr": 3e-3}),
    ("reward=wait+delay (w=1)", {"delay_weight": 1.0}),
]
OPT_NODE_LIMIT = 1_000_000
CONV_K, CONV_TOL = 3, 0.05


# ------------------------------------------------------------------ helpers
def _config_path(out: Path) -> Path:
    return out / "config.json"


def _update_config(out: Path, **items) -> dict:
    p = _config_path(out)
    cfg = json.loads(p.read_text()) if p.exists() else {}
    cfg.update(items)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2, default=str))
    return cfg


def _versions() -> dict:
    import matplotlib
    import scipy
    import torch_geometric
    return {"python": platform.python_version(), "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
            "platform": platform.platform()}


def _git() -> dict:
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                           check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True).stdout.strip())
        return {"commit": h, "dirty_worktree": dirty}
    except Exception:  # noqa: BLE001 - git is optional
        return {"commit": None, "dirty_worktree": None}


def _ensure_scenarios(out: Path, eval_limit: int | None) -> dict:
    sets = {}
    for name in SETS:
        p = out / "scenarios" / f"{name}.json"
        if not p.exists():
            sc = scenario_set(name)
            if eval_limit is not None and name != "validation":
                sc = sc[:eval_limit]
            save_scenarios(sc, p)
        sets[name] = load_scenarios(p)
    return sets


def _load_policy(path: Path):
    ck = torch.load(path, weights_only=False)
    c = ck["config"]
    pol = make_policy(c["kind"], c["hidden"], c["layers"])
    pol.load_state_dict(ck["state_dict"])
    pol.eval()
    return pol


def _mean_w(recs) -> float:
    return float(np.mean([r.total_waiting for r in recs]))


# ------------------------------------------------------------------ workers
def _train_worker(job):
    cfg_d, seed, out, ckpt_name = job
    torch.set_num_threads(1)
    cfg = TrainConfig(**cfg_d)
    t0 = time.time()
    policy, rows, val = train_run(cfg, seed)
    info = {"kind": cfg.kind, "seed": seed, "wall_s": time.time() - t0,
            "val_curve": [v["val_mean_waiting"] for v in val]}
    info["converged"] = converged(info["val_curve"], CONV_K, CONV_TOL)
    if ckpt_name is not None:
        ck = Path(out) / "checkpoints" / ckpt_name
        ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": policy.state_dict(), "config": asdict(cfg), "seed": seed}, ck)
        info["checkpoint"] = str(ck)
    ev = load_scenarios(Path(out) / "scenarios" / "evaluation.json")
    recs = B.evaluate(policy_dispatcher(policy), ev)
    info["eval_mean_waiting"] = _mean_w(recs)
    info["eval_mean_delay"] = float(np.mean([r.total_delay for r in recs]))
    return rows, val, info


def _opt_worker(job):
    set_name, k, sc = job
    t0 = time.time()
    res = solve(sc, node_limit=OPT_NODE_LIMIT)
    return {"set": set_name, "scenario": k, "status": res.status,
            "total_waiting": res.total_waiting, "episode_return": res.episode_return,
            "states_expanded": res.states_expanded, "actions": json.dumps(res.actions),
            "wall_s": round(time.time() - t0, 3)}


def _pool_map(fn, jobs, workers):
    if workers <= 1:
        return [fn(j) for j in jobs]
    with get_context("spawn").Pool(workers) as pool:
        return pool.map(fn, jobs, chunksize=1)


# ------------------------------------------------------------------ commands
def cmd_train(args) -> None:
    out = Path(args.out)
    _ensure_scenarios(out, args.eval_limit)
    base = TrainConfig(episodes=args.episodes, val_every=args.val_every)
    jobs = [(asdict(replace(base, kind=k)), s, str(out), f"{k}_seed{s}.pt")
            for k in KINDS for s in args.seeds]
    t0 = time.time()
    results = _pool_map(_train_worker, jobs, args.workers)
    rows = [r for res in results for r in res[0]]
    vals = [v for res in results for v in res[1]]
    infos = [res[2] for res in results]
    write_csv(rows, out / "train_metrics.csv", TRAIN_COLUMNS)
    write_csv(vals, out / "validation_curve.csv", ["kind", "seed", "episode", "val_mean_waiting"])
    (out / "train_runs.json").write_text(json.dumps(infos, indent=2))
    _update_config(out, train=asdict(base), train_seeds=list(args.seeds), kinds=list(KINDS),
                   convergence_criterion=f"last {CONV_K} validation means within "
                                         f"{CONV_TOL:.0%} of their mean",
                   train_wall_s=round(time.time() - t0, 1), versions=_versions(), git=_git(),
                   scenario_sets={k: {"seed_base": v[0], "n": v[1], "dep_high": v[2]}
                                  for k, v in SETS.items()},
                   training_scenario_seed="1_000_000 + 100_000*train_seed + episode",
                   torch_threads_per_process=1)


def _summaries(per_set_rows, opt_rows, seeds):
    """Builds the summary dict for each scenario set."""
    out = {}
    for set_name, rows in per_set_rows.items():
        opt = {r["scenario"]: r for r in opt_rows if r["set"] == set_name}
        n = len({r["scenario"] for r in rows})
        ok = [k for k in range(n) if opt.get(k, {}).get("status") == "optimal"]
        w_opt = [opt[k]["total_waiting"] for k in ok]

        def per_scenario(name, seed=None):
            sel = {r["scenario"]: r for r in rows if r["dispatcher"] == name and r["seed"] == seed}
            return [sel[k]["total_waiting"] for k in ok]

        per = {name: per_scenario(name) for name in
               ("fcfs", "fixed_priority", "longest_waiting", "random")}
        seed_level = {}
        for kind in KINDS:
            per_seed = [per_scenario(kind, s) for s in seeds]
            per[kind] = list(np.mean(per_seed, axis=0)) if per_seed else []
            seed_means = [float(np.mean(p)) for p in per_seed]
            m, lo, hi = mean_ci_t(seed_means)
            seed_level[kind] = {"per_seed_mean_waiting": seed_means, "mean": m, "t_ci_lo": lo,
                                "t_ci_hi": hi, "n_seeds": len(seeds),
                                "per_seed_gap": [(x - np.mean(w_opt)) / np.mean(w_opt)
                                                 for x in seed_means]}
            if "fcfs" in per:
                seed_level[kind]["per_seed_mean_diff_vs_fcfs"] = [
                    float(np.mean(np.array(p) - np.array(per["fcfs"]))) for p in per_seed]
        per["optimum"] = w_opt
        s = summarize(per, w_opt)
        extra = {}
        for name in list(per):
            sel = [r for r in rows if r["dispatcher"] == name]
            if name == "optimum":
                sel = []
            extra[name] = {
                "mean_delay": float(np.mean([r["total_delay"] for r in sel])) if sel else None,
                "mean_return": float(np.mean([r["episode_return"] for r in sel])) if sel else None,
                "deadlocks": sum(r["termination"] == "deadlock" for r in sel),
                "truncations": sum(r["termination"] == "truncated" for r in sel),
                "mean_holds": float(np.mean([r["holds"] for r in sel])) if sel else None,
                "mean_decisions": float(np.mean([r["decisions"] for r in sel])) if sel else None,
            }
            s[name].update(extra[name])
        gnn_vs_mlp = None
        if per["gnn"] and per["mlp"]:
            # same seed number = same sequence of training scenarios -> paired by seed
            seed_diff = [a - b for a, b in zip(seed_level["gnn"]["per_seed_mean_waiting"],
                                               seed_level["mlp"]["per_seed_mean_waiting"])]
            gnn_vs_mlp = {**_paired(per["gnn"], per["mlp"]),
                          "per_seed_diff": seed_diff, "seed_level_t_ci": mean_ci_t(seed_diff)}
        out[set_name] = {"n_scenarios": n, "n_optimum_computed": len(ok),
                         "gnn_vs_mlp": gnn_vs_mlp,
                         "optimum_not_computed": [k for k in range(n) if k not in ok],
                         "dispatchers": s, "learned_seed_level": seed_level,
                         "paired_scenarios_gnn_minus_fcfs":
                             list(np.array(per["gnn"]) - np.array(per["fcfs"])) if per["gnn"] else []}
    return out


def _fmt(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _markdown(summary) -> str:
    lines = []
    order = ["fcfs", "fixed_priority", "longest_waiting", "random", "mlp", "gnn", "optimum"]
    for set_name, s in summary.items():
        lines += [f"## {set_name} set ({s['n_scenarios']} scenarios, optimum computed for "
                  f"{s['n_optimum_computed']})", "",
                  "| dispatcher | mean W (train-min) | 95% bootstrap CI | SD | gap vs opt. | "
                  "diff vs FCFS [95% CI] | Wilcoxon p | d_z | mean delay | deadlocks | truncations |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name in order:
            d = s["dispatchers"][name]
            vf = d.get("vs_fcfs")
            vfs = (f"{_fmt(vf['mean_diff'])} [{_fmt(vf['ci_lo'])}, {_fmt(vf['ci_hi'])}]"
                   if vf else "-")
            lines.append(
                f"| {name} | {_fmt(d['mean'])} | [{_fmt(d['ci_lo'])}, {_fmt(d['ci_hi'])}] | "
                f"{_fmt(d['sd'])} | {_fmt(100 * d['gap'], 1)} % | {vfs} | "
                f"{_fmt(vf['wilcoxon_p'], 4) if vf else '-'} | "
                f"{_fmt(vf['cohens_dz']) if vf else '-'} | {_fmt(d['mean_delay'])} | "
                f"{d['deadlocks']} | {d['truncations']} |")
        lines += ["", "Learned policies, seed level (mean over scenarios per seed; 95% t-interval "
                      "over seeds):", ""]
        for kind, sl in s["learned_seed_level"].items():
            per_seed = ", ".join(_fmt(x) for x in sl["per_seed_mean_waiting"])
            lines.append(f"- **{kind}**: per seed [{per_seed}] -> mean {_fmt(sl['mean'])} "
                         f"[{_fmt(sl['t_ci_lo'])}, {_fmt(sl['t_ci_hi'])}] (n = {sl['n_seeds']})")
        a = s.get("gnn_vs_mlp")
        if a:
            m, lo, hi = a["seed_level_t_ci"]
            lines += ["", f"Ablation GNN − MLP: per-scenario paired mean diff {_fmt(a['mean_diff'])} "
                          f"[{_fmt(a['ci_lo'])}, {_fmt(a['ci_hi'])}] (bootstrap), Wilcoxon p = "
                          f"{_fmt(a['wilcoxon_p'], 4)}, d_z = {_fmt(a['cohens_dz'])}; seed level "
                          f"{_fmt(m)} [{_fmt(lo)}, {_fmt(hi)}] (t-interval over paired seeds)"]
        lines.append("")
    return "\n".join(lines)


def cmd_evaluate(args) -> None:
    out = Path(args.out)
    sets = _ensure_scenarios(out, args.eval_limit)
    t0 = time.time()
    jobs = [(name, k, sc) for name in ("evaluation", "heldout") for k, sc in enumerate(sets[name])]
    opt_rows = _pool_map(_opt_worker, jobs, args.workers)
    write_csv(opt_rows, out / "optimum.csv", ["set", "scenario", "status", "total_waiting",
                                              "episode_return", "states_expanded", "actions",
                                              "wall_s"])
    opt_wall = time.time() - t0
    rows = []
    rules = [("fcfs", B.fcfs), ("fixed_priority", B.fixed_priority),
             ("longest_waiting", B.longest_waiting)]
    runs = json.loads((out / "train_runs.json").read_text())
    seeds = sorted({r["seed"] for r in runs})
    repro = []
    for set_name in ("evaluation", "heldout"):
        scen = sets[set_name]
        for name, d in rules:
            rows += evaluation_rows(name, B.evaluate(d, scen), set_name)
        rows += evaluation_rows("random", B.evaluate(B.make_random(0), scen), set_name)
        for r in opt_rows:
            if r["set"] == set_name and r["status"] == "optimal":
                rows.append({"dispatcher": "optimum", "seed": None, "set": set_name,
                             "scenario": r["scenario"], "total_waiting": r["total_waiting"],
                             "total_delay": None, "episode_return": r["episode_return"],
                             "decisions": None, "holds": None, "steps": None,
                             "termination": "success"})
        for info in runs:
            pol = _load_policy(Path(info["checkpoint"]))
            recs = B.evaluate(policy_dispatcher(pol), scen)
            rows += evaluation_rows(info["kind"], recs, set_name, info["seed"])
            if set_name == "evaluation":
                repro.append(abs(_mean_w(recs) - info["eval_mean_waiting"]))
    write_csv(rows, out / "eval_per_scenario.csv", EVAL_COLUMNS)
    per_set = {s: [r for r in rows if r["set"] == s] for s in ("evaluation", "heldout")}
    summary = _summaries(per_set, opt_rows, seeds)
    for set_name in summary:
        for kind in KINDS:
            conv = [r["converged"] for r in runs if r["kind"] == kind]
            summary[set_name]["learned_seed_level"][kind]["converged_seeds"] = sum(conv)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (out / "summary.md").write_text(_markdown(summary))
    # step-level reward trace: GNN seed of the first run on evaluation scenario 0
    first = next(r for r in runs if r["kind"] == "gnn")
    rec, _ = run_episode(sets["evaluation"][0], _load_policy(Path(first["checkpoint"])), "greedy")
    cum = np.cumsum(rec.step_rewards)
    write_csv([{"t": t, "reward": r, "cumulative_return": float(c)}
               for t, (r, c) in enumerate(zip(rec.step_rewards, cum))],
              out / "step_trace.csv", ["t", "reward", "cumulative_return"])
    _update_config(out, evaluate_wall_s=round(time.time() - t0, 1),
                   optimum_wall_s=round(opt_wall, 1), optimum_node_limit=OPT_NODE_LIMIT,
                   step_trace={"policy": f"gnn seed {first['seed']}", "scenario": "evaluation #0"},
                   reproducibility={"description": "re-evaluation of saved checkpoints vs. the "
                                    "evaluation done right after training (mean total waiting)",
                                    "max_abs_diff": max(repro) if repro else None,
                                    "tolerance": 1e-9})


def cmd_sensitivity(args) -> None:
    out = Path(args.out)
    _ensure_scenarios(out, args.eval_limit)
    base = TrainConfig(episodes=args.episodes, val_every=args.val_every)
    jobs, names = [], []
    for name, over in SENSITIVITY:
        for s in args.sens_seeds:
            jobs.append((asdict(replace(base, **over)), s, str(out), None))
            names.append(name)
    t0 = time.time()
    results = _pool_map(_train_worker, jobs, args.workers)
    runs = []
    for name, (_, val, info) in zip(names, results):
        runs.append({"setting": name, "seed": info["seed"],
                     "val_mean_waiting": info["val_curve"][-1] if info["val_curve"] else None,
                     "eval_mean_waiting": info["eval_mean_waiting"],
                     "eval_mean_delay": info["eval_mean_delay"],
                     "converged": info["converged"], "wall_s": round(info["wall_s"], 1)})
    write_csv(runs, out / "sensitivity_runs.csv", list(runs[0]))
    table = []
    for name, _ in SENSITIVITY:
        sel = [r for r in runs if r["setting"] == name]
        vm, vlo, vhi = mean_ci_t([r["val_mean_waiting"] for r in sel])
        em, elo, ehi = mean_ci_t([r["eval_mean_waiting"] for r in sel])
        table.append({"setting": name, "n_seeds": len(sel), "val_mean": vm, "val_lo": vlo,
                      "val_hi": vhi, "eval_mean": em, "eval_lo": elo, "eval_hi": ehi,
                      "eval_mean_delay": float(np.mean([r["eval_mean_delay"] for r in sel])),
                      "converged_seeds": sum(r["converged"] for r in sel)})
    write_csv(table, out / "sensitivity.csv", list(table[0]))
    _update_config(out, sensitivity={"settings": [n for n, _ in SENSITIVITY],
                                     "overrides": {n: o for n, o in SENSITIVITY},
                                     "seeds": list(args.sens_seeds), "selection": "none - the "
                                     "main configuration was fixed before any evaluation"},
                   sensitivity_wall_s=round(time.time() - t0, 1))


def _read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def cmd_figures(args) -> None:
    from . import plots
    out = Path(args.out)
    fig = out / "figures"
    rows = [{**r, "seed": int(r["seed"]), "episode": int(r["episode"]),
             "episode_return": float(r["episode_return"]),
             "total_waiting": float(r["total_waiting"])}
            for r in _read_csv(out / "train_metrics.csv")]
    summary = json.loads((out / "summary.json").read_text())
    ev = summary["evaluation"]["dispatchers"]
    window = max(1, min(25, args.episodes // 5))
    plots.plot_return_curve(rows, fig / "A_return_curve.png", window)
    plots.plot_waiting_curve(rows, fig / "B_waiting_curve.png",
                             {"FCFS (evaluation-set mean)": ev["fcfs"]["mean"],
                              "exact optimum (evaluation-set mean)": ev["optimum"]["mean"]},
                             window)
    trace = _read_csv(out / "step_trace.csv")
    cfg = json.loads(_config_path(out).read_text())
    title = f"{cfg['step_trace']['policy']}, {cfg['step_trace']['scenario']}"
    plots.plot_step_trace([float(r["reward"]) for r in trace], fig / "C_step_reward.png",
                          fig / "C_cumulative_return.png", title)
    order = ["fcfs", "fixed_priority", "longest_waiting", "random", "mlp", "gnn", "optimum"]
    plots.plot_comparison(ev, fig / "D_comparison.png", order)
    plots.plot_paired_differences(summary["evaluation"]["paired_scenarios_gnn_minus_fcfs"],
                                  fig / "E_paired_difference.png",
                                  "GNN (mean over seeds) − FCFS")
    sens = [{"setting": r["setting"], "val_mean": float(r["val_mean"]),
             "val_lo": float(r["val_lo"]) if r["val_lo"] not in ("", "nan") else float("nan"),
             "val_hi": float(r["val_hi"]) if r["val_hi"] not in ("", "nan") else float("nan")}
            for r in _read_csv(out / "sensitivity.csv")]
    plots.plot_sensitivity(sens, fig / "F_sensitivity.png")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="railway-demo", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["train", "evaluate", "sensitivity", "figures", "all"])
    p.add_argument("--out", default="outputs")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--val-every", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--sens-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--eval-limit", type=int, default=None,
                   help="truncate evaluation/held-out sets (tests only)")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)
    torch.set_num_threads(1)
    t0 = time.time()
    steps = {"train": [cmd_train], "evaluate": [cmd_evaluate], "sensitivity": [cmd_sensitivity],
             "figures": [cmd_figures],
             "all": [cmd_train, cmd_evaluate, cmd_sensitivity, cmd_figures]}[args.command]
    for fn in steps:
        print(f"[railway-demo] {fn.__name__} ...", flush=True)
        fn(args)
    _update_config(Path(args.out), **{f"last_{args.command}_wall_s": round(time.time() - t0, 1),
                                      "cli_args": vars(args)})
    print(f"[railway-demo] done in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

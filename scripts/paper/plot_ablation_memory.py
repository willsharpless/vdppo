import fcntl
import multiprocessing as mp
import importlib.util
import shutil
import subprocess
import tempfile
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import cyclopts
import ipdb
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy import stats

from rraa_rl.paper_plot_utils import set_ax_style
from rraa_rl.path_utils import get_paper_plot_dir, get_runs_dir
from rraa_rl.trainer import TrainerCfg

app = cyclopts.App()
GPU_METRIC_NAME = "peak_gpu_mem_gib"


@contextmanager
def _file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def _load_train_fns():
    train_path = Path(__file__).resolve().parents[1] / "train.py"
    spec = importlib.util.spec_from_file_location("ablation_memory_train", train_path)
    assert spec is not None and spec.loader is not None, f"Failed to load training module from {train_path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.vd, module.lcrl


def _parse_n_specs(n_specs: str | list[int]) -> list[int]:
    if isinstance(n_specs, list):
        return sorted(set(int(v) for v in n_specs))

    vals = []
    for chunk in n_specs.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo_s, hi_s = chunk.split("-", maxsplit=1)
            lo, hi = int(lo_s), int(hi_s)
            step = 1 if hi >= lo else -1
            vals.extend(range(lo, hi + step, step))
        else:
            vals.append(int(chunk))
    return sorted(set(vals))


def _parse_algs(algs: str | list[str]) -> list[str]:
    valid_algs = ["vd", "lcrl"]
    if isinstance(algs, list):
        alg_vals = [str(v).lower() for v in algs]
    else:
        alg_vals = [chunk.strip().lower() for chunk in algs.split(",") if chunk.strip()]

    if not alg_vals or alg_vals == ["all"]:
        return valid_algs

    out = []
    for alg in alg_vals:
        assert alg in valid_algs, f"Unsupported alg: {alg}"
        if alg not in out:
            out.append(alg)
    return out


def _get_descendants(root_pid: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        try:
            status_lines = (proc_dir / "status").read_text().splitlines()
        except OSError:
            continue
        pid = int(proc_dir.name)
        ppid = None
        for line in status_lines:
            if line.startswith("PPid:"):
                ppid = int(line.split()[1])
                break
        if ppid is None:
            continue
        children.setdefault(ppid, []).append(pid)

    out = [root_pid]
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        for child_pid in children.get(pid, []):
            out.append(child_pid)
            stack.append(child_pid)
    return out


def _get_gpu_mem_gib(pid: int) -> float:
    proc_pids = {str(proc_pid) for proc_pid in _get_descendants(pid)}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError("nvidia-smi is not installed or not on PATH.") from e
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip() if e.stderr else "unknown nvidia-smi error"
        raise RuntimeError(f"nvidia-smi query failed: {stderr}") from e

    total_mib = 0.0
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        sample_pid, used_mib = parts
        if sample_pid in proc_pids:
            total_mib += float(used_mib)
    return total_mib / 1024.0


def _run_train_child(
    result_q: mp.Queue,
    alg: str,
    env_name: str,
    n_spec: int,
    n_agent: int,
    seed: int,
    n_train_steps: int,
    run_callbacks: bool,
):
    alg_dir = get_runs_dir(debug=True) / env_name / alg
    before = {p.name for p in alg_dir.iterdir()} if alg_dir.exists() else set()
    run_name = f"memory_{env_name}_{alg}_spc{n_spec}_ag{n_agent}_seed{seed}"

    try:
        vd, lcrl = _load_train_fns()
        train_fn = {"vd": vd, "lcrl": lcrl}[alg]
        trainer_cfg = TrainerCfg(
            n_train_steps=n_train_steps,
            eval_every=n_train_steps + 1,
            save_every=n_train_steps + 1,
            log_every=max(n_train_steps, 1),
        )
        train_fn(
            name=run_name,
            debug=True,
            env_name=env_name,
            seed=seed,
            trainer_cfg=trainer_cfg,
            n_agent=n_agent,
            n_spec=n_spec,
            run_callbacks=run_callbacks,
        )
        success = True
        error = ""
    except Exception:
        success = False
        error = traceback.format_exc()

    after = {p.name for p in alg_dir.iterdir()} if alg_dir.exists() else set()
    created_dirs = [str(alg_dir / name) for name in sorted(after - before) if run_name in name]
    result_q.put(
        {
            "success": success,
            "error": error,
            "created_dirs": created_dirs,
            "run_name": run_name,
        }
    )


def _measure_run(
    alg: str,
    env_name: str,
    n_spec: int,
    n_agent: int,
    seed: int,
    n_train_steps: int,
    sample_every_s: float,
    run_callbacks: bool,
) -> dict:
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    proc = ctx.Process(
        target=_run_train_child,
        kwargs=dict(
            result_q=result_q,
            alg=alg,
            env_name=env_name,
            n_spec=n_spec,
            n_agent=n_agent,
            seed=seed,
            n_train_steps=n_train_steps,
            run_callbacks=run_callbacks,
        ),
    )
    proc.start()

    t0 = time.monotonic()
    samples: list[tuple[float, float]] = []
    while proc.is_alive():
        elapsed_s = time.monotonic() - t0
        samples.append((elapsed_s, _get_gpu_mem_gib(proc.pid)))
        time.sleep(sample_every_s)

    proc.join()
    elapsed_s = time.monotonic() - t0
    samples.append((elapsed_s, _get_gpu_mem_gib(proc.pid)))

    if result_q.empty():
        result = {
            "success": False,
            "error": f"Training subprocess exited with code {proc.exitcode} before reporting results.",
            "created_dirs": [],
            "run_name": "",
        }
    else:
        result = result_q.get()
    if proc.exitcode not in (0, None):
        result["success"] = False
        if not result["error"]:
            result["error"] = f"Training subprocess exited with code {proc.exitcode}."

    for run_dir_s in result["created_dirs"]:
        run_dir = Path(run_dir_s)
        if run_dir.exists():
            shutil.rmtree(run_dir)

    peak_gpu_mem_gib = max((gpu_mem_gib for _, gpu_mem_gib in samples), default=0.0)
    return {
        "alg": alg,
        "env_name": env_name,
        "n_spec": n_spec,
        "n_agent": n_agent,
        "seed": seed,
        "n_train_steps": n_train_steps,
        "elapsed_s": elapsed_s,
        "peak_gpu_mem_gib": peak_gpu_mem_gib,
        "samples": samples,
        "success": result["success"],
        "error": result["error"],
    }


def _row_key(row: dict) -> tuple[str, str, int, int, int]:
    return row["alg"], row["env_name"], row["n_spec"], row["n_agent"], row["seed"]


def _load_report_rows(report_path: Path) -> tuple[list[dict], float | None, str | None]:
    if not report_path.exists():
        return [], None, None

    rows_by_key: dict[tuple[str, str, int, int, int], dict] = {}
    sample_every_s = None
    metric = None
    section = None
    summary_header = None
    sample_header = None

    with open(report_path, "r") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if line.startswith("# sample_every_s="):
                sample_every_s = float(line.split("=", maxsplit=1)[1])
                continue
            if line.startswith("# metric="):
                metric = line.split("=", maxsplit=1)[1]
                continue
            if line == "# summary":
                section = "summary"
                continue
            if line == "# samples":
                section = "samples"
                continue
            if line.startswith("#"):
                section = None
                continue

            if section == "summary":
                if line.startswith("alg\t"):
                    summary_header = line.split("\t")
                    if metric is None and len(summary_header) >= 7:
                        metric = summary_header[6]
                    continue
                alg, env_name, n_spec, n_agent, seed, n_train_steps, peak_metric_gib, elapsed_s, success = line.split("\t")
                row = {
                    "alg": alg,
                    "env_name": env_name,
                    "n_spec": int(n_spec),
                    "n_agent": int(n_agent),
                    "seed": int(seed),
                    "n_train_steps": int(n_train_steps),
                    "peak_gpu_mem_gib": float(peak_metric_gib),
                    "elapsed_s": float(elapsed_s),
                    "success": bool(int(success)),
                    "samples": [],
                    "error": "",
                }
                rows_by_key[_row_key(row)] = row
                continue

            if section == "samples":
                if line.startswith("alg\t"):
                    sample_header = line.split("\t")
                    continue
                parts = line.split("\t")
                if len(parts) != 8:
                    continue
                alg, env_name, n_spec, n_agent, seed, sample_idx, elapsed_s, gpu_mem_gib = parts
                key = (alg, env_name, int(n_spec), int(n_agent), int(seed))
                if key not in rows_by_key:
                    continue
                row = rows_by_key[key]
                row["samples"].append((float(elapsed_s), float(gpu_mem_gib)))

    return list(rows_by_key.values()), sample_every_s, metric


def _merge_rows(existing_rows: list[dict], new_rows: list[dict]) -> list[dict]:
    rows_by_key = {_row_key(row): row for row in existing_rows}
    for row in new_rows:
        rows_by_key[_row_key(row)] = row
    return sorted(rows_by_key.values(), key=lambda row: (_row_key(row), row["n_train_steps"]))


def _write_report(report_path: Path, rows: list[dict], sample_every_s: float):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=report_path.parent, prefix=report_path.name + ".", suffix=".tmp", delete=False) as f:
        tmp_path = Path(f.name)
        f.write("# ablation memory report\n")
        f.write(f"# sample_every_s={sample_every_s}\n")
        f.write(f"# metric={GPU_METRIC_NAME}\n")
        f.write("# summary\n")
        f.write(f"alg\tenv_name\tn_spec\tn_agent\tseed\tn_train_steps\t{GPU_METRIC_NAME}\telapsed_s\tsuccess\n")
        for row in rows:
            f.write(
                f"{row['alg']}\t{row['env_name']}\t{row['n_spec']}\t{row['n_agent']}\t{row['seed']}\t"
                f"{row['n_train_steps']}\t{row['peak_gpu_mem_gib']:.6f}\t{row['elapsed_s']:.3f}\t{int(row['success'])}\n"
            )

        f.write("\n# samples\n")
        f.write("alg\tenv_name\tn_spec\tn_agent\tseed\tsample_idx\telapsed_s\tgpu_mem_gib\n")
        for row in rows:
            for sample_idx, (elapsed_s, gpu_mem_gib) in enumerate(row["samples"]):
                f.write(
                    f"{row['alg']}\t{row['env_name']}\t{row['n_spec']}\t{row['n_agent']}\t{row['seed']}\t"
                    f"{sample_idx}\t{elapsed_s:.3f}\t{gpu_mem_gib:.6f}\n"
                )

        failures = [row for row in rows if not row["success"]]
        if failures:
            f.write("\n# failures\n")
            for row in failures:
                f.write(f"[{row['alg']} spc={row['n_spec']} seed={row['seed']}]\n{row['error']}\n")
    tmp_path.replace(report_path)


def _get_mean_ci(vals: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(vals, dtype=float)
    mean = float(arr.mean())
    if len(arr) == 1:
        return mean, mean, mean

    se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
    tcrit = float(stats.t.ppf(0.975, df=len(arr) - 1))
    delta = tcrit * se
    return mean, mean - delta, mean + delta


def _plot_rows(rows: list[dict], fig_path: Path):
    figsize = 0.8 * np.array([4.0, 3.0])
    fig, ax = plt.subplots(figsize=figsize)
    set_ax_style(ax)

    label_map = {"vd": "VDPPO", "lcrl": "LCRL"}
    for alg in ["vd", "lcrl"]:
        alg_rows = [row for row in rows if row["alg"] == alg and row["success"]]
        if not alg_rows:
            continue
        by_spec: dict[int, list[float]] = {}
        for row in alg_rows:
            by_spec.setdefault(row["n_spec"], []).append(row["peak_gpu_mem_gib"])

        plot_specs = sorted(by_spec)
        means = []
        ci_los = []
        ci_his = []
        for n_spec in plot_specs:
            mean, ci_lo, ci_hi = _get_mean_ci(by_spec[n_spec])
            means.append(mean)
            ci_los.append(mean - ci_lo)
            ci_his.append(ci_hi - mean)

        ax.errorbar(
            plot_specs,
            means,
            yerr=[ci_los, ci_his],
            marker="o",
            linestyle="-",
            capsize=5,
            label=label_map.get(alg, alg.upper()),
        )

    ax.set_xlabel("Number of Specifications")
    ax.set_ylabel("Peak GPU Memory [GiB]")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), borderaxespad=0, frameon=False)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig_suffix = fig_path.suffix if fig_path.suffix else ".png"
    with tempfile.NamedTemporaryFile("wb", dir=fig_path.parent, prefix=fig_path.stem + ".", suffix=fig_suffix, delete=False) as tmp_f:
        tmp_path = Path(tmp_f.name)
    fig.savefig(tmp_path, format=fig_suffix.lstrip("."), bbox_inches="tight", pad_inches=1e-3, dpi=400)
    tmp_path.replace(fig_path)
    plt.close(fig)


@app.default()
def main(
    env_name: str = "ablation_depth",
    algs: str = "all",
    n_specs: str = "1-10",
    seed: int = 0,
    n_seeds: int = 1,
    n_train_steps: int = 10,
    sample_every_s: float = 0.1,
    run_callbacks: bool = False,
):
    env_name = env_name.lower()
    assert env_name in {"ablation", "ablation_depth"}, f"Unsupported env_name: {env_name}"

    n_agent = 1
    alg_vals = _parse_algs(algs)
    n_spec_vals = _parse_n_specs(n_specs)
    assert n_spec_vals, "No N_spec values provided."
    assert n_seeds >= 1, "n_seeds must be at least 1."

    _get_gpu_mem_gib(pid=mp.current_process().pid)

    new_rows = []
    for alg in alg_vals:
        for n_spec in n_spec_vals:
            for seed_i in range(seed, seed + n_seeds):
                logger.info(f"Running {alg} memory sweep for {env_name} with N_spec={n_spec}, seed={seed_i}")
                row = _measure_run(
                    alg=alg,
                    env_name=env_name,
                    n_spec=n_spec,
                    n_agent=n_agent,
                    seed=seed_i,
                    n_train_steps=n_train_steps,
                    sample_every_s=sample_every_s,
                    run_callbacks=run_callbacks,
                )
                new_rows.append(row)
                if not row["success"]:
                    logger.error(f"{alg} failed for {env_name} N_spec={n_spec}, seed={seed_i}")

    plot_dir = get_paper_plot_dir()
    report_path = plot_dir / f"{env_name}_memory_load.txt"
    fig_path = plot_dir / f"{env_name}_memory_load.png"
    lock_path = report_path.with_suffix(report_path.suffix + ".lock")
    with _file_lock(lock_path):
        existing_rows, existing_sample_every_s, existing_metric = _load_report_rows(report_path)
        if existing_metric not in (None, GPU_METRIC_NAME):
            logger.warning(
                f"Existing report metric is {existing_metric}; ignoring old rows because this script now records {GPU_METRIC_NAME}."
            )
            existing_rows = []
        rows = _merge_rows(existing_rows, new_rows)
        if existing_sample_every_s is not None and existing_sample_every_s != sample_every_s:
            logger.warning(
                f"Merging with existing report sampled at {existing_sample_every_s}s; rewritten report will use {sample_every_s}s."
            )

        _write_report(report_path, rows, sample_every_s)
        _plot_rows(rows, fig_path)

    logger.success(f"Saved memory report to {report_path}")
    logger.success(f"Saved memory plot to {fig_path}")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()

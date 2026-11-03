#!/usr/bin/env python
"""Run a list of dfd.train runs sequentially and safely (overnight, flaky laptop GPU).

Usage: .venv/bin/python scripts/run_queue.py queues/<name>.yaml [--dry_run]

Queue YAML:
  runs:
    - config: configs/p2/p2_ref.yaml
      folds: "0,1,2"        # or "0", "all"
      run_name: p2_ref      # optional, default = config stem
      timeout_min: 30       # optional, whole-run wall clock, default 20 * n_folds
      args: ["--notes", "x"]  # optional extra CLI args

Features: skip finished runs (eval.json per fold), GPU health check before each run,
per-run timeout with process-group kill, queue log, gpu_monitor.csv sampler.
Exit codes: 0 ok, 1 usage/config error, 2 CUDA unavailable, 130 interrupted.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv" / "bin" / "python"
RUNS_DIR = REPO / "outputs" / "runs"
N_FOLDS = 5
DEFAULT_MIN_PER_FOLD = 20
KILL_GRACE_S = 30

_stop = threading.Event()          # set by SIGINT/SIGTERM
_current_run = {"name": ""}        # read by the GPU monitor
_log_lock = threading.Lock()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def fmt_dur(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"


class QueueLog:
    def __init__(self, path: Path | None):
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, kind: str, run: str = "", **kv) -> None:
        parts = [now(), f"{kind:<7}"]
        if run:
            parts.append(run)
        parts += [f"{k}={v}" for k, v in kv.items() if v not in (None, "")]
        line = " ".join(parts)
        with _log_lock:
            print(line, flush=True)
            if self.path:
                with open(self.path, "a") as f:
                    f.write(line + "\n")

    def raw(self, text: str) -> None:
        with _log_lock:
            print(text, flush=True)
            if self.path:
                with open(self.path, "a") as f:
                    f.write(text + "\n")


# ----------------------------------------------------------------------------- queue parsing
def parse_folds(spec) -> list[int]:
    s = str(spec).strip().lower()
    if s == "all":
        return list(range(N_FOLDS))
    try:
        folds = [int(x) for x in s.split(",") if x.strip() != ""]
    except ValueError:
        raise SystemExit(f"bad folds spec {spec!r}")
    if not folds:
        raise SystemExit(f"empty folds spec {spec!r}")
    return folds


def load_queue(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text()) or {}
    runs = []
    for i, r in enumerate(data.get("runs") or []):
        if "config" not in r:
            raise SystemExit(f"run #{i} has no 'config'")
        cfg = Path(r["config"])
        folds_spec = str(r.get("folds", "0"))
        folds = parse_folds(folds_spec)
        name = r.get("run_name") or cfg.stem
        runs.append(dict(
            config=str(cfg), folds=folds, folds_spec=folds_spec, name=name,
            timeout_min=float(r.get("timeout_min", DEFAULT_MIN_PER_FOLD * len(folds))),
            args=[str(a) for a in (r.get("args") or [])],
        ))
    return runs


def eval_path(run: str, fold: int) -> Path:
    return RUNS_DIR / run / f"fold{fold}" / "eval.json"


def fold_summary(run: str, fold: int) -> str:
    try:
        d = json.loads(eval_path(run, fold).read_text())
        s = d["summary"]
        return f"fold{fold}: sel={s['selection_score']:.4f} clean={s['clean_auc']:.4f}"
    except Exception:
        return f"fold{fold}: n/a"


def pending_folds(r: dict) -> list[int]:
    return [k for k in r["folds"] if not eval_path(r["name"], k).is_file()]


def build_cmd(r: dict) -> list[str]:
    cmd = [str(PY), "-m", "dfd.train", "--config", r["config"], "--fold", r["folds_spec"].replace(" ", ""),
           "--skip_done", "--run_name", r["name"]]
    return cmd + r["args"]


def build_env() -> dict:
    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # dfd is an editable install in .venv; PYTHONPATH=src is a harmless belt-and-braces.
    src = str(REPO / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


# ----------------------------------------------------------------------------- GPU health
def gpu_ok(timeout: float = 60.0) -> tuple[bool, str]:
    try:
        p = subprocess.run(["nvidia-smi"], capture_output=True, timeout=timeout)
        if p.returncode != 0:
            return False, f"nvidia-smi rc={p.returncode}: {p.stderr.decode(errors='replace').strip()[:200]}"
    except Exception as e:
        return False, f"nvidia-smi failed: {e!r}"
    try:
        p = subprocess.run([str(PY), "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
                           capture_output=True, timeout=timeout, cwd=REPO)
        if p.returncode != 0:
            return False, "torch.cuda.is_available() is False"
    except Exception as e:
        return False, f"torch cuda check failed: {e!r}"
    return True, "ok"


# ----------------------------------------------------------------------------- GPU monitor
MON_FIELDS = ("timestamp,temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used,"
              "pstate,clocks_throttle_reasons.active")
MON_HEADER = "timestamp,run,temp_c,sm_clock_mhz,power_w,util_pct,mem_used_mib,pstate,throttle_reasons"


class GpuMonitor(threading.Thread):
    def __init__(self, path: Path, interval: float = 60.0, run_getter=lambda: ""):
        super().__init__(daemon=True)
        self.path, self.interval, self.run_getter = path, interval, run_getter
        self.halt = threading.Event()

    def sample(self) -> str:
        run = self.run_getter()
        try:
            p = subprocess.run(["nvidia-smi", f"--query-gpu={MON_FIELDS}", "--format=csv,noheader,nounits"],
                               capture_output=True, timeout=30)
            if p.returncode != 0:
                raise RuntimeError(f"rc={p.returncode} {p.stderr.decode(errors='replace').strip()[:150]}")
            cells = [c.strip() for c in p.stdout.decode().strip().splitlines()[0].split(",")]
            cells = cells[:1] + [run] + cells[1:]
            return ",".join(cells)
        except Exception as e:
            err = str(e).replace(",", ";").replace("\n", " ")
            return f"{now()},{run},ERROR {err},,,,,,"

    def write_row(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.path.exists() or self.path.stat().st_size == 0
            with open(self.path, "a") as f:
                if new:
                    f.write(MON_HEADER + "\n")
                f.write(self.sample() + "\n")
        except Exception:
            pass  # never crash the queue

    def run(self) -> None:
        while not self.halt.is_set():
            self.write_row()
            self.halt.wait(self.interval)


# ----------------------------------------------------------------------------- launcher
def kill_group(proc: subprocess.Popen, log: QueueLog, run: str) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=KILL_GRACE_S)
    except subprocess.TimeoutExpired:
        log.event("KILL", run, note=f"SIGKILL after {KILL_GRACE_S}s grace")
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def launch(cmd: list[str], logfile: Path, timeout_s: float, env: dict, log: QueueLog, run: str) -> str:
    """Run cmd in its own process group. Returns 'ok', 'fail:<rc>', 'timeout' or 'interrupted'."""
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, "ab") as lf:
        lf.write(f"\n===== {now()} queue launch: {shlex.join(cmd)} =====\n".encode())
        lf.flush()
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True)
        t0 = time.monotonic()
        while True:
            try:
                rc = proc.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                pass
            if _stop.is_set():
                kill_group(proc, log, run)
                return "interrupted"
            if time.monotonic() - t0 > timeout_s:
                kill_group(proc, log, run)
                return "timeout"
    # reap any stragglers left in the group (e.g. dataloader workers)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    return "ok" if rc == 0 else f"fail:{rc}"


def tail(path: Path, n: int = 20) -> list[str]:
    try:
        return path.read_text(errors="replace").splitlines()[-n:]
    except Exception:
        return []


# ----------------------------------------------------------------------------- main loop
def run_queue(runs: list[dict], log: QueueLog, test_cmd: str | None = None) -> tuple[int, list[tuple]]:
    summary: list[tuple] = []
    env = build_env()
    code = 0
    for r in runs:
        name = r["name"]
        folds_txt = r["folds_spec"]
        if _stop.is_set():
            break
        todo = pending_folds(r)
        if not todo and not test_cmd:
            log.event("SKIP", name, folds=folds_txt, note="all eval.json present",
                      **{"summary": "; ".join(fold_summary(name, k) for k in r["folds"])})
            summary.append((name, "SKIP", 0.0))
            continue
        if not test_cmd:
            ok, why = gpu_ok()
            if not ok:
                log.event("STOP", name, note=f"CUDA unavailable ({why})")
                summary.append((name, "STOP", 0.0))
                code = 2
                break
        cmd = shlex.split(test_cmd) if test_cmd else build_cmd(r)
        logfile = RUNS_DIR / f"{name}.log"
        _current_run["name"] = name
        log.event("START", name, folds=folds_txt, pending=",".join(map(str, todo)),
                  timeout_min=r["timeout_min"], log=logfile.relative_to(REPO) if logfile.is_relative_to(REPO) else logfile)
        t0 = time.monotonic()
        status = launch(cmd, logfile, r["timeout_min"] * 60, env, log, name)
        dur = time.monotonic() - t0
        _current_run["name"] = ""
        sums = "; ".join(fold_summary(name, k) for k in r["folds"])
        if status == "ok":
            log.event("END", name, folds=folds_txt, duration=fmt_dur(dur), summary=sums)
            summary.append((name, "OK", dur))
            continue
        if status == "interrupted":
            log.event("STOP", name, note="interrupted by signal; run killed", duration=fmt_dur(dur), summary=sums)
            summary.append((name, "INTERRUPTED", dur))
            code = 130
            break
        if status == "timeout":
            log.event("TIMEOUT", name, folds=folds_txt, duration=fmt_dur(dur),
                      note=f"exceeded {r['timeout_min']} min", summary=sums)
            summary.append((name, "TIMEOUT", dur))
        else:
            log.event("FAIL", name, folds=folds_txt, exit_code=status.split(":")[1],
                      duration=fmt_dur(dur), summary=sums)
            summary.append((name, "FAIL", dur))
        log.raw(f"--- last 20 lines of {logfile.name} ---")
        for ln in tail(logfile):
            log.raw("    " + ln)
        log.raw("--- end ---")
        # after FAIL/TIMEOUT the next iteration's GPU check re-validates CUDA
    return code, summary


def print_summary(summary: list[tuple], log: QueueLog) -> None:
    log.raw("\nQueue summary")
    log.raw(f"{'run':<28}{'status':<14}{'duration':>10}")
    for name, status, dur in summary:
        log.raw(f"{name:<28}{status:<14}{fmt_dur(dur):>10}")


def dry_run(runs: list[dict]) -> None:
    for r in runs:
        st = ", ".join(f"fold{k}:{'done' if eval_path(r['name'], k).is_file() else 'pending'}" for k in r["folds"])
        verdict = "SKIP (all done)" if not pending_folds(r) else "RUN"
        print(f"[{verdict}] run={r['name']} config={r['config']} folds={r['folds_spec']}")
        print(f"    status : {st}")
        print(f"    timeout: {r['timeout_min']} min")
        print(f"    cmd    : {shlex.join(build_cmd(r))}")
        print(f"    env    : PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH={REPO / 'src'}")
        print(f"    log    : outputs/runs/{r['name']}.log")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queue")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--_test_cmd", help=argparse.SUPPRESS)           # replace the train command (tests)
    ap.add_argument("--_monitor_interval", type=float, default=60.0, help=argparse.SUPPRESS)
    a = ap.parse_args()

    qpath = Path(a.queue)
    if not qpath.is_absolute():
        qpath = (Path.cwd() / qpath) if (Path.cwd() / qpath).exists() else REPO / qpath
    runs = load_queue(qpath)
    if a.dry_run:
        dry_run(runs)
        return 0

    if not os.environ.get("DF_DATA_ROOT") and not a._test_cmd:
        print("ERROR: DF_DATA_ROOT is not set; export it to the dataset root (dir with train/ and "
              "train_metadata.csv) before running the queue.", file=sys.stderr)
        return 1
    if not PY.exists():
        print(f"ERROR: {PY} not found", file=sys.stderr)
        return 1

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log = QueueLog(RUNS_DIR / f"queue_{qpath.stem}.log")
    signal.signal(signal.SIGINT, lambda *_: _stop.set())
    signal.signal(signal.SIGTERM, lambda *_: _stop.set())

    mon = GpuMonitor(RUNS_DIR / "gpu_monitor.csv", a._monitor_interval, lambda: _current_run["name"])
    mon.start()
    log.event("QUEUE", qpath.name, runs=len(runs), dry="no")
    try:
        code, summary = run_queue(runs, log, a._test_cmd)
    finally:
        mon.halt.set()
    if _stop.is_set() and code == 0:
        log.event("STOP", note="interrupted between runs")
        code = 130
    print_summary(summary, log)
    return code


if __name__ == "__main__":
    sys.exit(main())

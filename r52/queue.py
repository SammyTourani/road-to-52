# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""A durable job-queue supervisor so v1.0 runs without a human or the planner in the loop.

    python -m r52.queue run queue/v1.yaml              # supervisor loop (foreground)
    python -m r52.queue run queue/v1.yaml --dry-run     # print the plan, run nothing
    python -m r52.queue status queue/v1.yaml            # one status table, then exit
    python -m r52.queue wait-pid runs/<name>/train.pid  # internal helper, see below

``scripts/queue.sh`` is the ``nohup``/PID-file wrapper that runs ``run`` detached, the same
way ``scripts/train.sh`` wraps ``r52.train``.

Job schema (``queue/v1.yaml``, an ordered top-level YAML list)
----------------------------------------------------------------------------------------
    - id: nano-pretrain
      cmd: "shell command"
      needs_gpu: true                # default false
      done_when: <predicate>         # see below; checked before launching AND after running
      resume_cmd: "shell command"    # optional; used for the one automatic retry
      timeout_h: 24                  # optional; SIGTERM (then SIGKILL) the job's process
                                      # group if it runs longer than this
      after: [rung0-export]          # optional; job ids that must be `done` first
      gpu_wait_only: true            # optional, default false -- see "GPU exclusivity" below

``done_when`` is either a plain string (a path or glob, done when it matches >= 1 file) or a
dict whose keys are ANDed together:

    file: <path>                     path must exist
    glob: <pattern>                  glob must match >= 1 path
    files: [<path>, ...]             every path must exist
    pid_dead: <pidfile>              the pidfile is missing, unreadable, or names a dead pid

``done_when`` is the single source of truth for "did this job succeed": it is checked BEFORE
a job is launched (so work the planner already did by hand, or a previous queue run already
finished, is skipped -- this is what makes `run` idempotent across restarts) and AFTER the
job's shell command exits, REGARDLESS of that command's exit code. The latter is what lets a
job's ``cmd`` swallow a "launcher" script's own exit code (see "self-backgrounding launchers"
below) without the queue mistaking a successful re-attach for a failure, and it is also just
a stronger check than trusting exit codes alone: a command that exits 0 without producing its
declared output is a bug worth surfacing, not a silent success.

Self-backgrounding launchers (``scripts/{train,sft,rl}.sh``)
----------------------------------------------------------------------------------------
Those scripts ``nohup ... &`` the real trainer and return immediately once the PID file is
written, so a job whose ``cmd`` is just ``scripts/train.sh ...`` would look "done" to a naive
supervisor within a second. Every job wrapping one of them instead reads, as shell:

    (scripts/train.sh configs/nano_30m.yaml nano-a || true) && \\
        python -m r52.queue wait-pid runs/nano-a/train.pid && \\
        test -e runs/nano-a/ckpt/best

``|| true`` matters for restart-safety: if the queue (or its host) restarted while this job
was mid-flight, ``train.sh`` refuses to start a second copy ("already running") and exits 1;
swallowing that lets the command fall through to ``wait-pid``, which re-attaches to the
already-running trainer by PID instead of erroring out. ``wait-pid`` always exits 0 -- it is
purely a wait, not a verdict -- so the trailing ``test -e ...`` (mirroring the job's own
``done_when``) is what actually decides success once the process is gone. ``rung0-wait`` is
the degenerate case of this pattern with no launcher at all: it only ever waits and checks.

GPU exclusivity
----------------------------------------------------------------------------------------
At most one ``needs_gpu`` job runs at a time (non-``needs_gpu`` jobs: up to two). Before
launching a NEW ``needs_gpu`` job -- i.e. only at the moment the queue's own bookkeeping
already shows the GPU slot free -- the scheduler also runs ``pgrep -f 'r52\\.train'``; a
match at that moment can only be a process the queue did not launch (anything it did launch
would still show as ``running`` in its own state), so it is treated as "GPU busy" and the
job is left ``pending`` for the next poll rather than started. This is what lets the queue
sit behind an already-running headline job (Rung 0) started before the queue existed.

``rung0-wait`` is exempt (``gpu_wait_only: true``): its entire purpose is to wait on exactly
that external process, so the same check would otherwise block it forever.

Everything else: retries, failure, restarts
----------------------------------------------------------------------------------------
A job that finishes (by any means) with ``done_when`` still false gets exactly one more
attempt using ``resume_cmd`` if one is given (``attempts`` counts launches; a job never
exceeds 2). With no ``resume_cmd`` a single failed attempt is final. A ``failed`` job's own
dependents become ``blocked`` (and propagate); jobs that do not depend on it keep going.

The queue survives being killed and restarted: state lives in ``runs/queue/state.json``
(saved after every scheduling pass), and a job recorded as ``running`` whose PID is still
alive is re-attached rather than relaunched (Popen children use ``start_new_session=True``,
so they keep running even if the supervisor itself dies -- exactly what you want for a
multi-hour job under a process whose only job is to watch it). A re-attached job cannot be
reaped for a real exit code (it is no longer this process's child), so its outcome, like
everything else here, is decided by ``done_when`` once its PID is gone.
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "Job",
    "cmd_wait_pid",
    "is_done",
    "load_jobs",
    "main",
    "render_plan",
    "render_status",
    "run_queue",
]

STATE_FILENAME = "state.json"
QUEUE_SUBDIR = "queue"          # runs/queue/{state.json,<id>.log}
DEFAULT_POLL_S = 30.0
MAX_GPU_CONCURRENCY = 1
MAX_NONGPU_CONCURRENCY = 2
TERMINATE_GRACE_S = 30.0
PENDING, RUNNING, DONE, FAILED, BLOCKED = "pending", "running", "done", "failed", "blocked"
TERMINAL_STATUSES = {DONE, FAILED, BLOCKED}


# ------------------------------------------------------------------------------------------
# Job definition
# ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    id: str
    cmd: str
    done_when: str | dict[str, Any]
    needs_gpu: bool = False
    resume_cmd: str | None = None
    timeout_h: float | None = None
    after: tuple[str, ...] = ()
    gpu_wait_only: bool = False


def load_jobs(path: str | Path) -> list[Job]:
    """Parse and validate ``queue/v1.yaml`` (or any file with the same shape)."""
    raw = yaml.safe_load(Path(path).read_text())
    items = raw.get("jobs", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError(f"{path}: expected a top-level list of jobs (or {{jobs: [...]}})")
    jobs = [_job_from_dict(item) for item in items]
    _validate_jobs(jobs)
    return jobs


def _job_from_dict(d: dict[str, Any]) -> Job:
    missing = {"id", "cmd", "done_when"} - d.keys()
    if missing:
        raise ValueError(f"job {d.get('id', '?')!r} missing required key(s): {sorted(missing)}")
    timeout_h = d.get("timeout_h")
    return Job(
        id=str(d["id"]),
        cmd=str(d["cmd"]),
        done_when=d["done_when"],
        needs_gpu=bool(d.get("needs_gpu", False)),
        resume_cmd=(str(d["resume_cmd"]) if d.get("resume_cmd") else None),
        timeout_h=(float(timeout_h) if timeout_h is not None else None),
        after=tuple(d.get("after") or ()),
        gpu_wait_only=bool(d.get("gpu_wait_only", False)),
    )


def _validate_jobs(jobs: list[Job]) -> None:
    seen: set[str] = set()
    for j in jobs:
        if j.id in seen:
            raise ValueError(f"duplicate job id: {j.id!r}")
        seen.add(j.id)
    for j in jobs:
        for dep in j.after:
            if dep not in seen:
                raise ValueError(f"job {j.id!r}: unknown dependency {dep!r}")
    _check_no_cycles(jobs)


def _check_no_cycles(jobs: list[Job]) -> None:
    by_id = {j.id: j for j in jobs}
    white, gray, black = 0, 1, 2
    color = dict.fromkeys(by_id, white)

    def visit(jid: str, stack: list[str]) -> None:
        color[jid] = gray
        for dep in by_id[jid].after:
            if color[dep] == gray:
                raise ValueError(f"cyclic `after` dependency: {' -> '.join([*stack, jid, dep])}")
            if color[dep] == white:
                visit(dep, [*stack, jid])
        color[jid] = black

    for jid in by_id:
        if color[jid] == white:
            visit(jid, [])


# ------------------------------------------------------------------------------------------
# done_when predicates
# ------------------------------------------------------------------------------------------


def _resolve(root: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else root / p


def _file_exists(root: Path, rel: str) -> bool:
    return _resolve(root, rel).exists()


def _glob_matches(root: Path, pattern: str) -> bool:
    p = Path(pattern)
    pat = pattern if p.is_absolute() else str(root / pattern)
    return len(globmod.glob(pat)) > 0


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else -- treat as alive
    return True


def _pid_dead(root: Path, pidfile: str) -> bool:
    p = _resolve(root, pidfile)
    if not p.exists():
        return True
    try:
        pid = int(p.read_text().strip())
    except (ValueError, OSError):
        return True
    return not pid_alive(pid)


def is_done(job: Job, root: Path) -> bool:
    return _check_predicate(job.done_when, root)


def _check_predicate(done_when: str | dict[str, Any], root: Path) -> bool:
    if isinstance(done_when, str):
        return _glob_matches(root, done_when)
    if isinstance(done_when, dict):
        checks: list[bool] = []
        if "file" in done_when:
            checks.append(_file_exists(root, done_when["file"]))
        if "glob" in done_when:
            checks.append(_glob_matches(root, done_when["glob"]))
        if "files" in done_when:
            checks.append(all(_file_exists(root, f) for f in done_when["files"]))
        if "pid_dead" in done_when:
            checks.append(_pid_dead(root, done_when["pid_dead"]))
        if not checks:
            raise ValueError(f"done_when dict has none of file/glob/files/pid_dead: {done_when!r}")
        return all(checks)
    raise ValueError(f"done_when must be a string or a dict, got {done_when!r}")


# ------------------------------------------------------------------------------------------
# wait-pid: the small helper `cmd`/`resume_cmd` strings shell out to
# ------------------------------------------------------------------------------------------


def cmd_wait_pid(pidfile: str, poll_s: float = DEFAULT_POLL_S) -> int:
    """Block until ``pidfile`` is gone or names a dead process. Always exits 0.

    Success/failure is for the caller's ``done_when`` (or the trailing ``test -e ...`` in a
    job's own ``cmd``) to decide -- this only waits.
    """
    p = Path(pidfile)
    while True:
        if not p.exists():
            return 0
        try:
            pid = int(p.read_text().strip())
        except (ValueError, OSError):
            return 0
        if not pid_alive(pid):
            return 0
        time.sleep(poll_s)


# ------------------------------------------------------------------------------------------
# state.json
# ------------------------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp()


def _default_job_state() -> dict[str, Any]:
    return {
        "status": PENDING,
        "attempts": 0,
        "start": None,
        "end": None,
        "exit_code": None,
        "pid": None,
        "note": None,
    }


def state_path_for(root: Path) -> Path:
    return root / "runs" / QUEUE_SUBDIR / STATE_FILENAME


def log_path_for(root: Path, job_id: str) -> Path:
    return root / "runs" / QUEUE_SUBDIR / f"{job_id}.log"


def load_state(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_state(path: Path, state: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(path)


# ------------------------------------------------------------------------------------------
# Process handles: a real Popen (jobs this process launched) or a Ghost (jobs it re-attached
# to by PID after a restart -- no longer this process's child, so it cannot be reaped for a
# real exit status; done_when is what decides its outcome either way).
# ------------------------------------------------------------------------------------------


class Ghost:
    REATTACHED_EXIT = -1  # sentinel exit_code meaning "unknown; re-attached after a restart"

    def __init__(self, pid: int) -> None:
        self.pid = pid

    def poll(self) -> int | None:
        return None if pid_alive(self.pid) else self.REATTACHED_EXIT


ProcHandle = subprocess.Popen | Ghost


def _terminate(proc: ProcHandle, grace_s: float = TERMINATE_GRACE_S) -> int | None:
    """SIGTERM then SIGKILL a job's whole process group (see module docstring).

    Returns the real exit status when ``proc`` is a :class:`subprocess.Popen` we can reap
    (``Popen.wait`` is used for this deliberately -- polling liveness with ``os.kill(pid,
    0)`` alone would leave a real child as an unreaped zombie, which still answers "alive").
    A re-attached :class:`Ghost` cannot be reaped (it is not this process's child), so it is
    only polled until it disappears; this returns ``None`` for it either way.
    """
    pid = proc.pid
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return None
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return None
        budget = grace_s if sig is signal.SIGTERM else 5.0
        if isinstance(proc, subprocess.Popen):
            try:
                return proc.wait(timeout=budget)
            except subprocess.TimeoutExpired:
                continue
        else:
            deadline = time.time() + budget
            while time.time() < deadline:
                if not pid_alive(pid):
                    return None
                time.sleep(0.2)
    return None


# ------------------------------------------------------------------------------------------
# Scheduler
# ------------------------------------------------------------------------------------------


def _launch(job: Job, use_resume: bool, root: Path) -> subprocess.Popen:
    cmd = job.resume_cmd if use_resume else job.cmd
    if cmd is None:  # pragma: no cover - guarded by the caller (retry only offered with one)
        raise ValueError(f"job {job.id!r}: no resume_cmd to retry with")
    log_path = log_path_for(root, job.id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as fh:
        fh.write(f"\n=== {_now_iso()} launching ({'resume' if use_resume else 'fresh'}) ===\n$ {cmd}\n")
    log_fh = log_path.open("a", buffering=1)
    try:
        return subprocess.Popen(
            cmd, shell=True, cwd=root, stdout=log_fh, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_fh.close()  # the child holds its own dup'd fd; safe to close our copy


def reconcile(jobs_by_id: dict[str, Job], state: dict[str, dict[str, Any]],
              root: Path) -> dict[str, ProcHandle]:
    """After a restart: re-attach `running` jobs whose PID is alive; finalize the rest."""
    running: dict[str, ProcHandle] = {}
    for job_id, st in state.items():
        if st.get("status") != RUNNING:
            continue
        job = jobs_by_id.get(job_id)
        if job is None:
            continue  # stale id from a since-edited queue file; leave the record alone
        pid = st.get("pid")
        if pid and pid_alive(int(pid)):
            running[job_id] = Ghost(int(pid))
        else:
            _finalize(job, state, root, exit_code=None)
    return running


def _poll_running(running: dict[str, ProcHandle], jobs_by_id: dict[str, Job],
                   state: dict[str, dict[str, Any]], root: Path) -> None:
    now = time.time()
    for job_id in list(running):
        job = jobs_by_id[job_id]
        proc = running[job_id]
        rc = proc.poll()
        if rc is None:
            if job.timeout_h:
                started = _parse_iso(state[job_id]["start"])
                if now - started > job.timeout_h * 3600.0:
                    rc = _terminate(proc)
                    if rc is None:
                        rc = -signal.SIGTERM
                    state[job_id]["note"] = f"timed out after {job.timeout_h} h; terminated"
                else:
                    continue
            else:
                continue
        _finalize(job, state, root, exit_code=rc)
        del running[job_id]


def _finalize(job: Job, state: dict[str, dict[str, Any]], root: Path, exit_code: int | None) -> None:
    st = state[job.id]
    st["exit_code"] = exit_code
    st["end"] = _now_iso()
    st["pid"] = None
    if is_done(job, root):
        st["status"] = DONE
        st["note"] = None
        return
    if st["attempts"] < 2 and job.resume_cmd:
        st["status"] = PENDING
        st["note"] = f"attempt {st['attempts']} failed (exit {exit_code}); retrying with resume_cmd"
    else:
        st["status"] = FAILED
        if not st.get("note"):
            st["note"] = f"exit {exit_code}, done_when not satisfied"


def _external_gpu_busy(pattern: str = r"r52\.train") -> bool:
    """True iff some ``r52.train`` process is alive that this queue did not launch.

    Only meaningful (and only called) when the queue's own bookkeeping already shows the
    GPU slot free -- see the module docstring's "GPU exclusivity" section.
    """
    try:
        r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return False  # pgrep missing/hung: fail open rather than wedge the queue
    return r.returncode == 0 and bool(r.stdout.strip())


def _deps_done(job: Job, state: dict[str, dict[str, Any]]) -> bool:
    return all(state[d]["status"] == DONE for d in job.after)


def _deps_blocked(job: Job, state: dict[str, dict[str, Any]]) -> bool:
    return any(state[d]["status"] in (FAILED, BLOCKED) for d in job.after)


def schedule_once(jobs: list[Job], state: dict[str, dict[str, Any]],
                   running: dict[str, ProcHandle], root: Path) -> None:
    """One scheduling pass: reap finished jobs, propagate `blocked`, launch what is eligible."""
    jobs_by_id = {j.id: j for j in jobs}
    _poll_running(running, jobs_by_id, state, root)

    for job in jobs:
        st = state[job.id]
        if st["status"] == PENDING and _deps_blocked(job, state):
            st["status"] = BLOCKED
            st["note"] = "a dependency failed"

    gpu_used = sum(1 for j in jobs if state[j.id]["status"] == RUNNING and j.needs_gpu)
    nongpu_used = sum(1 for j in jobs if state[j.id]["status"] == RUNNING and not j.needs_gpu)

    for job in jobs:
        st = state[job.id]
        if st["status"] != PENDING or not _deps_done(job, state):
            continue

        if is_done(job, root):
            st["status"] = DONE
            st["note"] = "already satisfied at scheduling time"
            continue

        if job.needs_gpu:
            if gpu_used >= MAX_GPU_CONCURRENCY:
                continue
            if not job.gpu_wait_only and _external_gpu_busy():
                st["note"] = "waiting: an r52.train process outside the queue is using the GPU"
                continue
        elif nongpu_used >= MAX_NONGPU_CONCURRENCY:
            continue

        use_resume = st["attempts"] >= 1
        proc = _launch(job, use_resume, root)
        st["attempts"] += 1
        st["status"] = RUNNING
        st["start"] = _now_iso()
        st["end"] = None
        st["note"] = None
        st["pid"] = proc.pid
        running[job.id] = proc
        if job.needs_gpu:
            gpu_used += 1
        else:
            nongpu_used += 1


def run_queue(queue_file: str | Path, root: Path, *, dry_run: bool = False,
              poll_s: float = DEFAULT_POLL_S, max_passes: int | None = None) -> int:
    jobs = load_jobs(queue_file)
    if dry_run:
        print(render_plan(jobs, root))
        return 0

    spath = state_path_for(root)
    state = load_state(spath)
    for job in jobs:
        state.setdefault(job.id, _default_job_state())
    jobs_by_id = {j.id: j for j in jobs}

    running = reconcile(jobs_by_id, state, root)
    save_state(spath, state)

    passes = 0
    while True:
        schedule_once(jobs, state, running, root)
        save_state(spath, state)
        passes += 1
        if all(state[j.id]["status"] in TERMINAL_STATUSES for j in jobs):
            break
        if max_passes is not None and passes >= max_passes:
            break
        time.sleep(poll_s)

    return 0 if all(state[j.id]["status"] == DONE for j in jobs) else 1


# ------------------------------------------------------------------------------------------
# Rendering: --dry-run and `status`
# ------------------------------------------------------------------------------------------


def _done_when_str(done_when: str | dict[str, Any]) -> str:
    if isinstance(done_when, str):
        return done_when
    parts = []
    if "pid_dead" in done_when:
        parts.append(f"pid_dead={done_when['pid_dead']}")
    if "file" in done_when:
        parts.append(f"file={done_when['file']}")
    if "glob" in done_when:
        parts.append(f"glob={done_when['glob']}")
    if "files" in done_when:
        parts.append(f"files[{len(done_when['files'])}]={done_when['files'][0]},...")
    return " & ".join(parts)


def render_plan(jobs: list[Job], root: Path) -> str:
    lines = [f"r52 queue -- dry run ({len(jobs)} jobs, root={root})", ""]
    for i, job in enumerate(jobs, 1):
        state = "already done" if is_done(job, root) else "pending"
        kind = "gpu" if job.needs_gpu else "cpu"
        deps = ", ".join(job.after) if job.after else "-"
        lines.append(f"{i:2d}. {job.id:<20} [{kind}] after=[{deps}]  ({state})")
        lines.append(f"      done_when : {_done_when_str(job.done_when)}")
        lines.append(f"      cmd       : {job.cmd}")
        if job.resume_cmd:
            lines.append(f"      resume_cmd: {job.resume_cmd}")
        if job.timeout_h:
            lines.append(f"      timeout_h : {job.timeout_h}")
    return "\n".join(lines)


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _elapsed_str(st: dict[str, Any]) -> str:
    start = st.get("start")
    if not start:
        return "-"
    end = st.get("end")
    t0 = _parse_iso(start)
    t1 = _parse_iso(end) if end else time.time()
    return _fmt_duration(t1 - t0)


def _last_log_line(root: Path, job_id: str, chunk: int = 8192) -> str:
    p = log_path_for(root, job_id)
    if not p.exists():
        return "-"
    try:
        with p.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - chunk))
            data = fh.read()
    except OSError:
        return "-"
    lines = [ln for ln in data.decode("utf-8", "replace").splitlines() if ln.strip()]
    return lines[-1][:100] if lines else "-"


def _pending_reason(job: Job, state: dict[str, dict[str, Any]], root: Path) -> str:
    """Why a `pending` job has not launched -- this is the whole point of `status` existing
    rather than a bare dump of state.json (and what lets rung0-wait visibly show that it is
    waiting on the live run, even before the queue has ever actually been started)."""
    if is_done(job, root):
        return "pending (done_when already satisfied; will be marked done on the next run)"
    unmet = [d for d in job.after if state.get(d, _default_job_state())["status"] != DONE]
    if unmet:
        return f"pending (needs {','.join(unmet)})"
    if job.gpu_wait_only:
        return "pending (waiting on the live run)"
    if job.needs_gpu and _external_gpu_busy():
        return "pending (GPU busy)"
    return "pending (ready)"


def _truncate(s: str, limit: int) -> str:
    return s if len(s) <= limit else s[: limit - 1] + "…"  # "…"


def render_status(jobs: list[Job], state: dict[str, dict[str, Any]], root: Path) -> str:
    headers = ("id", "status", "elapsed", "done_when", "last log line")
    caps = (None, None, None, 60, 70)  # id/status/elapsed: never truncated; the rest: capped

    rows: list[tuple[str, ...]] = []
    for job in jobs:
        st = state.get(job.id, _default_job_state())
        status_text = st["status"] if st["status"] != PENDING else _pending_reason(job, state, root)
        row = (
            job.id, status_text, _elapsed_str(st),
            _done_when_str(job.done_when), _last_log_line(root, job.id),
        )
        rows.append(tuple(c if cap is None else _truncate(c, cap) for c, cap in zip(row, caps, strict=True)))

    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers).rstrip(), fmt.format(*("-" * w for w in widths))]
    lines += [fmt.format(*row).rstrip() for row in rows]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="r52.queue", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="run the queue to completion (long-lived supervisor)")
    pr.add_argument("queue_file")
    pr.add_argument("--dry-run", action="store_true", help="print the plan and run nothing")
    pr.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_S, dest="poll_s")
    pr.add_argument("--root", default=None, help="repo root (default: this package's repo)")

    ps = sub.add_parser("status", help="print one status table and exit")
    ps.add_argument("queue_file")
    ps.add_argument("--root", default=None)

    pw = sub.add_parser("wait-pid", help="block until a pidfile's process is gone (internal)")
    pw.add_argument("pidfile")
    pw.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_S, dest="poll_s")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "wait-pid":
        return cmd_wait_pid(args.pidfile, args.poll_s)

    root = Path(args.root).resolve() if args.root else _repo_root()

    if args.command == "run":
        return run_queue(args.queue_file, root, dry_run=args.dry_run, poll_s=args.poll_s)

    if args.command == "status":
        jobs = load_jobs(args.queue_file)
        state = load_state(state_path_for(root))
        for job in jobs:
            state.setdefault(job.id, _default_job_state())
        print(render_status(jobs, state, root))
        return 0

    return 2  # pragma: no cover - argparse `required=True` already rejects this


if __name__ == "__main__":
    sys.exit(main())

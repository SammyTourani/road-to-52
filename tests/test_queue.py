# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""r52.queue: dependency ordering, done_when skipping, GPU exclusivity, retry-once, restart
re-attachment, timeouts, status/dry-run rendering, and the shipped `queue/v1.yaml`.

Every job `cmd` here is a filesystem-only shell one-liner (`touch`, `sleep`, `mkdir`) -- no
GPU, no network, no MLX import -- driven either through `schedule_once` directly (fast,
precise unit tests) or through `run_queue` against a real temp YAML file (a few end-to-end
checks). `poll_s` is kept at a few milliseconds throughout so the whole module runs in well
under the repo's usual per-file budget.

The real `gpt2-124m-mac` pretraining run may be alive on this machine for days (it is, as of
this writing), which would make the "external GPU busy" guard fire for real here -- every
test neutralizes it via the autouse fixture below, and the two tests that care about that
guard's behaviour opt back in explicitly.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import pytest

from r52 import queue as rq

REPO = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.timeout(20)


@pytest.fixture(autouse=True)
def _no_real_gpu_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep this machine's own live processes out of every test but the ones about that."""
    monkeypatch.setattr(rq, "_external_gpu_busy", lambda pattern=r"r52\.train": False)


def _drive(jobs: list[rq.Job], root: Path, *, max_passes: int = 400,
           poll_s: float = 0.02) -> dict[str, dict]:
    """The same loop `run_queue` runs, without a YAML file or a state.json on disk."""
    state = {j.id: rq._default_job_state() for j in jobs}
    running: dict[str, object] = {}
    for _ in range(max_passes):
        rq.schedule_once(jobs, state, running, root)
        if all(state[j.id]["status"] in rq.TERMINAL_STATUSES for j in jobs):
            break
        time.sleep(poll_s)
    return state


# --------------------------------------------------------------------------------------
# Loading / validating queue/v1.yaml's schema
# --------------------------------------------------------------------------------------


def test_load_jobs_parses_every_documented_field(tmp_path: Path) -> None:
    p = tmp_path / "q.yaml"
    p.write_text(
        "- id: a\n"
        "  cmd: touch a.done\n"
        "  needs_gpu: true\n"
        "  done_when: a.done\n"
        "  resume_cmd: touch a.done\n"
        "  timeout_h: 2.5\n"
        "  after: [z]\n"
        "  gpu_wait_only: true\n"
        "- id: z\n"
        "  cmd: touch z.done\n"
        "  done_when: {file: z.done}\n"
    )
    jobs = rq.load_jobs(p)
    a = next(j for j in jobs if j.id == "a")
    assert a.cmd == "touch a.done"
    assert a.needs_gpu is True
    assert a.done_when == "a.done"
    assert a.resume_cmd == "touch a.done"
    assert a.timeout_h == 2.5
    assert a.after == ("z",)
    assert a.gpu_wait_only is True


def test_needs_gpu_and_after_and_resume_cmd_default_safely(tmp_path: Path) -> None:
    p = tmp_path / "q.yaml"
    p.write_text("- id: a\n  cmd: 'true'\n  done_when: a.done\n")
    (job,) = rq.load_jobs(p)
    assert job.needs_gpu is False
    assert job.resume_cmd is None
    assert job.timeout_h is None
    assert job.after == ()
    assert job.gpu_wait_only is False


def test_missing_required_key_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "q.yaml"
    p.write_text("- id: a\n  cmd: 'true'\n")  # no done_when
    with pytest.raises(ValueError, match="done_when"):
        rq.load_jobs(p)


def test_duplicate_id_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "q.yaml"
    p.write_text(
        "- id: a\n  cmd: 'true'\n  done_when: x\n"
        "- id: a\n  cmd: 'true'\n  done_when: y\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        rq.load_jobs(p)


def test_unknown_dependency_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "q.yaml"
    p.write_text("- id: a\n  cmd: 'true'\n  done_when: x\n  after: [nope]\n")
    with pytest.raises(ValueError, match="nope"):
        rq.load_jobs(p)


def test_cyclic_dependency_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "q.yaml"
    p.write_text(
        "- id: a\n  cmd: 'true'\n  done_when: x\n  after: [b]\n"
        "- id: b\n  cmd: 'true'\n  done_when: y\n  after: [a]\n"
    )
    with pytest.raises(ValueError, match="cyclic"):
        rq.load_jobs(p)


def test_real_queue_v1_yaml_is_well_formed_and_matches_the_design() -> None:
    jobs = rq.load_jobs(REPO / "queue" / "v1.yaml")
    ids = [j.id for j in jobs]
    assert len(ids) == len(set(ids))
    assert len(jobs) >= 17

    for expected in (
        "rung0-wait", "rung0-eval-full", "rung0-export",
        "nano-pretrain", "nano-eval", "nano-export",
        "nano-midtrain", "nano-sft", "nano-sft-eval", "nano-sft-export",
        "nano-passk", "nano-rl", "nano-rl-eval", "nano-rl-export",
        "ablate-tokenizer", "ablate-corpus",
        "prep-sft-data", "prep-midtrain-data", "prep-tokenizer", "prep-corpora",
    ):
        assert expected in ids, expected

    by_id = {j.id: j for j in jobs}
    # Rung 0 is never touched -- rung0-wait only ever waits, never launches or resumes it.
    rung0 = by_id["rung0-wait"]
    assert "scripts/train.sh" not in rung0.cmd
    assert rung0.resume_cmd is None
    assert rung0.timeout_h is None  # a multi-day run must never be timed out

    # Exactly one job is exempt from the external-GPU-busy guard, and it is rung0-wait.
    exempt = [j.id for j in jobs if j.gpu_wait_only]
    assert exempt == ["rung0-wait"]
    assert rung0.needs_gpu is True

    # Every `after` id resolves (load_jobs already enforces this; belt and suspenders).
    for j in jobs:
        for dep in j.after:
            assert dep in by_id

    # Non-GPU prerequisite jobs really are needs_gpu: false, per the design.
    for jid in ("prep-sft-data", "prep-midtrain-data", "prep-tokenizer", "prep-corpora"):
        assert by_id[jid].needs_gpu is False


# --------------------------------------------------------------------------------------
# done_when predicates
# --------------------------------------------------------------------------------------


def test_done_when_string_is_a_glob(tmp_path: Path) -> None:
    job = rq.Job(id="x", cmd="true", done_when="out/*.json")
    assert not rq.is_done(job, tmp_path)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "a.json").write_text("{}")
    assert rq.is_done(job, tmp_path)


def test_done_when_file(tmp_path: Path) -> None:
    job = rq.Job(id="x", cmd="true", done_when={"file": "a.txt"})
    assert not rq.is_done(job, tmp_path)
    (tmp_path / "a.txt").write_text("1")
    assert rq.is_done(job, tmp_path)


def test_done_when_files_requires_every_path(tmp_path: Path) -> None:
    job = rq.Job(id="x", cmd="true", done_when={"files": ["a.txt", "b.txt"]})
    assert not rq.is_done(job, tmp_path)
    (tmp_path / "a.txt").write_text("1")
    assert not rq.is_done(job, tmp_path)
    (tmp_path / "b.txt").write_text("1")
    assert rq.is_done(job, tmp_path)


def test_done_when_pid_dead(tmp_path: Path) -> None:
    job = rq.Job(id="x", cmd="true", done_when={"pid_dead": "p.pid"})
    assert rq.is_done(job, tmp_path)  # no pidfile at all counts as dead
    proc = subprocess.Popen(["sleep", "5"])
    (tmp_path / "p.pid").write_text(str(proc.pid))
    try:
        assert not rq.is_done(job, tmp_path)
    finally:
        proc.terminate()
        proc.wait()
    assert rq.is_done(job, tmp_path)


def test_done_when_compound_pid_dead_and_file_is_anded(tmp_path: Path) -> None:
    job = rq.Job(id="x", cmd="true", done_when={"pid_dead": "p.pid", "file": "ckpt/best"})
    proc = subprocess.Popen(["sleep", "5"])
    (tmp_path / "p.pid").write_text(str(proc.pid))
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "best").write_text("x")
    try:
        # file exists, but the process is still alive -> not done (this is the exact
        # rung0-wait scenario: `ckpt/best` appears well before the run finishes).
        assert not rq.is_done(job, tmp_path)
    finally:
        proc.terminate()
        proc.wait()
    assert rq.is_done(job, tmp_path)


def test_done_when_bad_shape_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="string or a dict"):
        rq.is_done(rq.Job(id="x", cmd="true", done_when=123), tmp_path)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="none of"):
        rq.is_done(rq.Job(id="x", cmd="true", done_when={}), tmp_path)


# --------------------------------------------------------------------------------------
# Dependency ordering + the done_when pre-check (skip work already done)
# --------------------------------------------------------------------------------------


def test_a_job_waits_for_its_dependency_then_runs(tmp_path: Path) -> None:
    jobs = [
        rq.Job(id="a", cmd="touch a.done", done_when="a.done"),
        rq.Job(id="b", cmd="touch b.done", done_when="b.done", after=("a",)),
    ]
    state = _drive(jobs, tmp_path)
    assert state["a"]["status"] == "done"
    assert state["b"]["status"] == "done"


def test_done_when_pre_check_skips_the_command_entirely(tmp_path: Path) -> None:
    (tmp_path / "already.done").write_text("x")
    job = rq.Job(id="x", cmd="touch ran.marker", done_when="already.done")
    state = {"x": rq._default_job_state()}
    rq.schedule_once([job], state, {}, tmp_path)
    assert state["x"]["status"] == "done"
    assert state["x"]["attempts"] == 0
    assert not (tmp_path / "ran.marker").exists()


def test_planner_ran_a_prep_job_by_hand_before_the_queue_started(tmp_path: Path) -> None:
    """The exact scenario the design calls out: `done_when` must detect and skip it."""
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "tokenizers").mkdir()
    (tmp_path / "data" / "tokenizers" / "fineweb32k").mkdir()
    job = rq.Job(id="prep-tokenizer", cmd="touch should_not_run",
                 done_when="data/tokenizers/fineweb32k")
    state = _drive([job], tmp_path)
    assert state["prep-tokenizer"]["status"] == "done"
    assert not (tmp_path / "should_not_run").exists()


# --------------------------------------------------------------------------------------
# GPU exclusivity: at most 1 needs_gpu job, up to 2 non-GPU jobs, concurrently
# --------------------------------------------------------------------------------------


def _concurrency_probe(marker_dir: Path, obs_prefix: Path, jid: str, sleep_s: float = 0.2) -> str:
    """A `cmd` that records, the instant it starts, how many siblings are also mid-flight."""
    return (
        f"mkdir -p {marker_dir} && touch {marker_dir}/{jid} && "
        f"ls {marker_dir} | wc -l > {obs_prefix}_{jid}.txt && "
        f"sleep {sleep_s} && rm -f {marker_dir}/{jid} && touch {jid}.done"
    )


def _read_obs(tmp_path: Path, jid: str) -> int:
    return int((tmp_path / f"obs_{jid}.txt").read_text().strip())


def test_needs_gpu_jobs_never_overlap(tmp_path: Path) -> None:
    marker_dir = tmp_path / "gpu_running"
    jobs = [
        rq.Job(id=f"g{i}", cmd=_concurrency_probe(marker_dir, tmp_path / "obs", f"g{i}"),
               done_when=f"g{i}.done", needs_gpu=True)
        for i in range(3)
    ]
    state = _drive(jobs, tmp_path)
    assert all(state[j.id]["status"] == "done" for j in jobs)
    observed = [_read_obs(tmp_path, j.id) for j in jobs]
    assert max(observed) == 1, f"a GPU job saw a sibling GPU job mid-flight: {observed}"


def test_nongpu_jobs_run_up_to_two_concurrently(tmp_path: Path) -> None:
    marker_dir = tmp_path / "cpu_running"
    jobs = [
        rq.Job(id=f"n{i}", cmd=_concurrency_probe(marker_dir, tmp_path / "obs", f"n{i}"),
               done_when=f"n{i}.done", needs_gpu=False)
        for i in range(3)
    ]
    state = _drive(jobs, tmp_path)
    assert all(state[j.id]["status"] == "done" for j in jobs)
    observed = [_read_obs(tmp_path, j.id) for j in jobs]
    assert max(observed) == 2, f"expected 2-way concurrency at some point: {observed}"
    assert all(o <= 2 for o in observed), f"exceeded the cap of 2: {observed}"


def test_external_r52_train_blocks_a_needs_gpu_job(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rq, "_external_gpu_busy", lambda pattern=r"r52\.train": True)
    job = rq.Job(id="g", cmd="touch g.done", done_when="g.done", needs_gpu=True)
    state = {"g": rq._default_job_state()}
    rq.schedule_once([job], state, {}, tmp_path)
    assert state["g"]["status"] == "pending"
    assert "GPU" in (state["g"]["note"] or "")
    assert not (tmp_path / "g.done").exists()


def test_gpu_wait_only_job_launches_despite_external_busy(tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """rung0-wait's whole point: it must run *because* r52.train is alive, not despite it."""
    monkeypatch.setattr(rq, "_external_gpu_busy", lambda pattern=r"r52\.train": True)
    job = rq.Job(id="rung0-wait", cmd="touch w.done", done_when="w.done",
                 needs_gpu=True, gpu_wait_only=True)
    state = _drive([job], tmp_path)
    assert state["rung0-wait"]["status"] == "done"


# --------------------------------------------------------------------------------------
# Retry-once with resume_cmd; failure propagation to dependents
# --------------------------------------------------------------------------------------


def test_one_retry_with_resume_cmd_then_succeeds(tmp_path: Path) -> None:
    job = rq.Job(id="x", cmd="true", done_when="x.done", resume_cmd="touch x.done")
    state = _drive([job], tmp_path)
    assert state["x"]["status"] == "done"
    assert state["x"]["attempts"] == 2


def test_no_resume_cmd_fails_after_exactly_one_attempt(tmp_path: Path) -> None:
    job = rq.Job(id="x", cmd="true", done_when="never_created")
    state = _drive([job], tmp_path)
    assert state["x"]["status"] == "failed"
    assert state["x"]["attempts"] == 1


def test_resume_cmd_that_also_fails_stops_at_two_attempts(tmp_path: Path) -> None:
    job = rq.Job(id="x", cmd="true", done_when="never_created", resume_cmd="true")
    state = _drive([job], tmp_path)
    assert state["x"]["status"] == "failed"
    assert state["x"]["attempts"] == 2


def test_a_failed_job_blocks_its_dependents_but_not_unrelated_jobs(tmp_path: Path) -> None:
    jobs = [
        rq.Job(id="fails", cmd="true", done_when="nope"),
        rq.Job(id="dependent", cmd="touch dependent.done", done_when="dependent.done",
               after=("fails",)),
        rq.Job(id="unrelated", cmd="touch unrelated.done", done_when="unrelated.done"),
    ]
    state = _drive(jobs, tmp_path)
    assert state["fails"]["status"] == "failed"
    assert state["dependent"]["status"] == "blocked"
    assert "fails" in (state["dependent"]["note"] or "") or state["dependent"]["note"] == \
        "a dependency failed"
    assert state["unrelated"]["status"] == "done"


# --------------------------------------------------------------------------------------
# Restart-safety: a `running` job with a live PID is re-attached, never relaunched
# --------------------------------------------------------------------------------------


def test_restart_reattaches_instead_of_relaunching(tmp_path: Path) -> None:
    marker = tmp_path / "launch_count.txt"
    job = rq.Job(
        id="j",
        cmd=f"echo x >> {marker} && sleep 0.5 && touch out.done",
        done_when="out.done",
    )
    state = {"j": rq._default_job_state()}
    running: dict[str, object] = {}

    rq.schedule_once([job], state, running, tmp_path)
    assert state["j"]["status"] == "running"
    assert state["j"]["attempts"] == 1

    # Simulate the supervisor process dying and a brand new one starting: a fresh
    # in-memory `running` dict, but the same on-disk state (as `run_queue` would reload).
    # The original Popen handle is abandoned here on purpose (that *is* the scenario), but
    # unlike a real restart -- where the reparented orphan is reaped by launchd/init -- this
    # test is still one process, so nothing will ever reap it unless something calls
    # `.wait()` on that original handle; do that in the background so the child cannot
    # become a permanent zombie that `pid_alive()` (and thus the Ghost below) sees as alive
    # forever. This has no bearing on `reconcile`/`Ghost` themselves, which is what is tested.
    threading.Thread(target=running["j"].wait, daemon=True).start()
    running2 = rq.reconcile({"j": job}, state, tmp_path)
    assert isinstance(running2.get("j"), rq.Ghost)

    for _ in range(400):
        rq.schedule_once([job], state, running2, tmp_path)
        if state["j"]["status"] in rq.TERMINAL_STATUSES:
            break
        time.sleep(0.02)

    assert state["j"]["status"] == "done"
    assert state["j"]["attempts"] == 1  # never relaunched
    assert marker.read_text().count("x") == 1  # the command body really ran exactly once


def test_reconcile_finalizes_a_running_job_whose_pid_already_died(tmp_path: Path) -> None:
    (tmp_path / "out.done").write_text("x")  # it actually finished before the "restart"
    job = rq.Job(id="j", cmd="true", done_when="out.done")
    state = {"j": rq._default_job_state()}
    state["j"].update(status=rq.RUNNING, attempts=1, start=rq._now_iso(), pid=999999999)
    running = rq.reconcile({"j": job}, state, tmp_path)
    assert "j" not in running
    assert state["j"]["status"] == "done"


# --------------------------------------------------------------------------------------
# Timeout enforcement
# --------------------------------------------------------------------------------------


def test_timeout_terminates_a_stuck_job_and_marks_it_failed(tmp_path: Path) -> None:
    job = rq.Job(id="j", cmd="sleep 30 && touch out.done", done_when="out.done",
                 timeout_h=1.0 / 3600 / 2)  # 0.5 s
    state = {"j": rq._default_job_state()}
    running: dict[str, object] = {}
    rq.schedule_once([job], state, running, tmp_path)
    assert state["j"]["status"] == "running"
    pid = state["j"]["pid"]
    assert rq.pid_alive(pid)

    time.sleep(0.8)
    rq.schedule_once([job], state, running, tmp_path)

    assert state["j"]["status"] == "failed"
    assert "timed out" in (state["j"]["note"] or "")
    assert not (tmp_path / "out.done").exists()
    assert not rq.pid_alive(pid), "the process group must actually be killed, not just marked failed"


# --------------------------------------------------------------------------------------
# wait-pid (the helper `cmd`/`resume_cmd` strings shell out to)
# --------------------------------------------------------------------------------------


def test_wait_pid_returns_immediately_with_no_pidfile(tmp_path: Path) -> None:
    t0 = time.time()
    rc = rq.cmd_wait_pid(str(tmp_path / "nope.pid"), poll_s=5.0)
    assert rc == 0
    assert time.time() - t0 < 1.0


def test_wait_pid_returns_immediately_for_a_dead_pid(tmp_path: Path) -> None:
    proc = subprocess.Popen(["true"])
    proc.wait()
    pidfile = tmp_path / "p.pid"
    pidfile.write_text(str(proc.pid))
    t0 = time.time()
    rc = rq.cmd_wait_pid(str(pidfile), poll_s=5.0)
    assert rc == 0
    assert time.time() - t0 < 1.0


def test_wait_pid_blocks_until_the_process_exits(tmp_path: Path) -> None:
    proc = subprocess.Popen(["sleep", "0.4"])
    # `wait-pid` targets processes it did not spawn (an unrelated pidfile), which the kernel
    # (or, after a real restart, launchd/init) reaps the moment they exit. Reap this direct
    # child the same way in the background, so it is never a zombie that `pid_alive()` (via
    # `os.kill(pid, 0)`) would see as "still alive" forever.
    threading.Thread(target=proc.wait, daemon=True).start()
    pidfile = tmp_path / "p.pid"
    pidfile.write_text(str(proc.pid))
    t0 = time.time()
    rc = rq.cmd_wait_pid(str(pidfile), poll_s=0.05)
    elapsed = time.time() - t0
    assert rc == 0
    assert elapsed >= 0.3, "returned before the process actually exited"


# --------------------------------------------------------------------------------------
# Rendering: --dry-run and `status`
# --------------------------------------------------------------------------------------


def test_dry_run_prints_the_plan_and_runs_nothing(tmp_path: Path) -> None:
    p = tmp_path / "q.yaml"
    p.write_text(
        "- id: a\n  cmd: touch ran.marker\n  needs_gpu: true\n  done_when: a.done\n"
        "  timeout_h: 3\n  resume_cmd: touch a.done\n"
        "- id: b\n  cmd: touch b.done\n  done_when: b.done\n  after: [a]\n"
    )
    rc = rq.run_queue(p, tmp_path, dry_run=True)
    assert rc == 0
    assert not (tmp_path / "ran.marker").exists()
    assert not rq.state_path_for(tmp_path).exists()

    out = rq.render_plan(rq.load_jobs(p), tmp_path)
    assert "a" in out and "b" in out
    assert "[gpu]" in out
    assert "after=[a]" in out
    assert "touch ran.marker" in out
    assert "resume_cmd" in out
    assert "timeout_h" in out


def test_dry_run_reports_work_already_done(tmp_path: Path) -> None:
    (tmp_path / "a.done").write_text("x")
    p = tmp_path / "q.yaml"
    p.write_text("- id: a\n  cmd: touch a.done\n  done_when: a.done\n")
    out = rq.render_plan(rq.load_jobs(p), tmp_path)
    assert "already done" in out


def test_status_flags_work_already_done_before_the_queue_ever_ran(tmp_path: Path) -> None:
    """The exact situation the shipped queue/v1.yaml is in right now: the planner ran
    prep-tokenizer's equivalent by hand, `state.json` does not exist yet (the real queue has
    never been started), and `status` should say so rather than call it merely "ready"."""
    (tmp_path / "already.done").write_text("x")
    job = rq.Job(id="prep-tokenizer", cmd="touch should_not_run", done_when="already.done")
    state = {"prep-tokenizer": rq._default_job_state()}  # untouched: status is read-only
    out = rq.render_status([job], state, tmp_path)
    assert "already satisfied" in out
    assert not (tmp_path / "should_not_run").exists()


def test_status_table_has_the_documented_columns(tmp_path: Path) -> None:
    jobs = [
        rq.Job(id="a", cmd="touch a.done", done_when="a.done"),
        rq.Job(id="b", cmd="touch b.done", done_when="b.done", after=("a",)),
    ]
    state = _drive(jobs, tmp_path)
    out = rq.render_status(jobs, state, tmp_path)
    header = out.splitlines()[0]
    for col in ("id", "status", "elapsed", "done_when", "last log line"):
        assert col in header
    assert "a" in out and "done" in out
    assert "touch a.done" in out  # the last log line


def test_status_explains_an_unlaunched_dependency(tmp_path: Path) -> None:
    jobs = [
        rq.Job(id="a", cmd="sleep 5 && touch a.done", done_when="a.done"),
        rq.Job(id="b", cmd="touch b.done", done_when="b.done", after=("a",)),
    ]
    state = {j.id: rq._default_job_state() for j in jobs}
    rq.schedule_once(jobs, state, {}, tmp_path)  # launches `a`, `b` stays pending
    out = rq.render_status(jobs, state, tmp_path)
    b_line = next(ln for ln in out.splitlines() if ln.startswith("b"))
    assert "needs a" in b_line
    # cleanup: don't leave a `sleep 5` orphan behind after the test process exits
    pid = state["a"].get("pid")
    if pid and rq.pid_alive(pid):
        subprocess.run(["kill", "-9", str(pid)], check=False)


# --------------------------------------------------------------------------------------
# run_queue end to end (through a real YAML file + real state.json on disk)
# --------------------------------------------------------------------------------------


def test_run_queue_end_to_end_writes_state_and_logs(tmp_path: Path) -> None:
    p = tmp_path / "q.yaml"
    p.write_text(
        "- id: a\n  cmd: touch a.done\n  done_when: a.done\n"
        "- id: b\n  cmd: touch b.done\n  done_when: b.done\n  after: [a]\n  needs_gpu: true\n"
    )
    rc = rq.run_queue(p, tmp_path, poll_s=0.02, max_passes=400)
    assert rc == 0

    state = rq.load_state(rq.state_path_for(tmp_path))
    assert state["a"]["status"] == "done"
    assert state["b"]["status"] == "done"
    assert rq.log_path_for(tmp_path, "a").exists()
    assert "touch a.done" in rq.log_path_for(tmp_path, "a").read_text()


def test_run_queue_is_idempotent_across_a_second_call(tmp_path: Path) -> None:
    """A second `run` over the same root (state.json already says done) does no new work."""
    p = tmp_path / "q.yaml"
    p.write_text("- id: a\n  cmd: touch ran_again.marker\n  done_when: a.done\n")
    (tmp_path / "a.done").write_text("x")  # already satisfied before the queue ever runs

    rc = rq.run_queue(p, tmp_path, poll_s=0.02, max_passes=10)
    assert rc == 0
    assert not (tmp_path / "ran_again.marker").exists()
    state = rq.load_state(rq.state_path_for(tmp_path))
    assert state["a"]["attempts"] == 0

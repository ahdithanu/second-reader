"""Pipeline orchestration: the execution order, status lines, fail-loud STATUS.md."""

from __future__ import annotations

import datetime
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import annotators as ann_mod
from . import data, defects, evals, router, submissions
from .agent import run_grader
from .config import (
    DATA_DIR,
    MAX_ITEMS,
    PROJECT_ROOT,
    RESULTS_DIR,
    load_api_key,
    load_rubric,
    load_thresholds,
)
from .db import connect
from .defects import swapped_view
from .grader import CallStats, GraderError, graded_ids, log_call, save_grade, save_trace
from .schemas import Item, Submission

STATUS_FILE = PROJECT_ROOT / "STATUS.md"
GRADE_WORKERS = 4


def status(msg: str) -> None:
    print(f"STATUS: {msg}", flush=True)


def fail(step: str, error: str) -> None:
    """Write the error to STATUS.md and stop. Never work around failures silently."""
    stamp = datetime.datetime.now(datetime.UTC).isoformat()
    STATUS_FILE.write_text(
        f"# second-reader run FAILED\n\n"
        f"- when: {stamp}\n- step: {step}\n\n## Error\n\n```\n{error}\n```\n\n"
        f"## Resume\n\nThe pipeline is idempotent. Fix the cause and re-run:\n\n"
        f"```\npython -m second_reader run --items {MAX_ITEMS}\n```\n"
        f"Completed work (items, submissions, grades, traces) is kept in "
        f"data/second_reader.duckdb and will not be redone.\n"
    )
    print(f"FAILED at {step}: {error}\nDetails written to {STATUS_FILE}", file=sys.stderr)
    raise SystemExit(1)


def make_client(mock: bool):
    if mock:
        from .mock import MockClient
        return MockClient()
    key = load_api_key()
    if not key:
        fail(
            "client setup",
            "ANTHROPIC_API_KEY is not set (checked environment and ./.env). "
            "The grader agent and synthetic annotator cannot run without it. "
            "Set it, or use --mock for a plumbing-only run.",
        )
    import anthropic
    return anthropic.Anthropic(api_key=key, max_retries=5)


def _grade_one(client, con_ro, item: Item, sub: Submission, rubric, thresholds,
               order_label: str, mode: str):
    """Worker: run the grader for one (item, order, mode). Read-only DB use."""
    if order_label == "swapped":
        item, sub = swapped_view(item, sub)
    return run_grader(
        client, con_ro, item, sub, rubric, thresholds,
        swapped=(order_label == "swapped"), tools_on=(mode == "agent"),
    )


def grade_all(con, client, items: list[Item], rubric, thresholds, mock: bool,
              mode: str, orders: tuple[str, ...] = ("original", "swapped"),
              only_items: set[str] | None = None) -> int:
    """Grade every pending (item, order) pair for a mode. Concurrent API calls,
    DB writes serialized on the main thread; workers read via own cursors."""
    subs = {s.item_id: s for s in submissions.get_submissions(con)}
    tasks = []
    for order_label in orders:
        done = graded_ids(con, order_label, mode, rubric.version)
        for it in items:
            if only_items is not None and it.item_id not in only_items:
                continue
            if it.item_id in done:
                continue
            if it.item_id not in subs:
                fail("grade", f"item_id={it.item_id} has no submission; generation must run first")
            tasks.append((it, subs[it.item_id], order_label))
    if not tasks:
        return 0

    lock = threading.Lock()

    def worker(task):
        it, sub, order_label = task
        # DuckDB: one cursor per thread for reads inside the tool executor.
        with lock:
            cur = con.cursor()
        try:
            return _grade_one(client, cur, it, sub, rubric, thresholds, order_label, mode)
        finally:
            cur.close()

    workers = 1 if mock else GRADE_WORKERS
    n_done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(worker, t): t for t in tasks}
        for fut in as_completed(futures):
            it, sub, order_label = futures[fut]
            try:
                res = fut.result()
            except GraderError as e:
                ex.shutdown(cancel_futures=True)
                fail("grade", str(e))
            except Exception as e:  # noqa: BLE001
                ex.shutdown(cancel_futures=True)
                fail("grade", f"item_id={it.item_id} order={order_label} mode={mode}: "
                              f"{type(e).__name__}: {e}")
            for i, rnd in enumerate(res.rounds, start=1):
                log_call(con, it.item_id, "grade", mode, order_label, rubric.version,
                         thresholds.model, mock,
                         CallStats(tokens_in=rnd["tokens_in"], tokens_out=rnd["tokens_out"],
                                   latency_ms=rnd["latency_ms"], attempts=i),
                         thresholds)
            save_grade(con, it.item_id, order_label, mode, rubric.version,
                       res.output, res.escalated, res.escalation_reason,
                       res.budget_exhausted)
            save_trace(con, it.item_id, order_label, mode, rubric.version,
                       res.trace, res.n_tool_calls, res.budget_exhausted,
                       res.escalated, res.tokens_in, res.tokens_out, res.latency_ms)
            n_done += 1
            if n_done % 25 == 0:
                status(f"[{mode}] graded {n_done}/{len(tasks)} pending runs")
    return n_done


def run(items_n: int = MAX_ITEMS, mock: bool = False, smoke_n: int = 10,
        tools: str = "both") -> None:
    modes = {"on": ["agent"], "off": ["single"], "both": ["single", "agent"]}[tools]
    # Mock runs live in their own DB and results dir so they can never
    # contaminate real grades or published metrics.
    db_path = DATA_DIR / ("second_reader_mock.duckdb" if mock else "second_reader.duckdb")
    out_dir = RESULTS_DIR / "mock" if mock else RESULTS_DIR
    con = connect(db_path)
    rubric = load_rubric()
    thresholds = load_thresholds()
    client = make_client(mock)
    tag = " [MOCK — numbers not publishable]" if mock else ""

    # Step 1: data load (items + peer votes)
    n = data.load_items(con, items_n)
    if n < items_n:
        fail("data load", f"only {n} items loaded, expected {items_n}")
    n_peers = con.execute("SELECT count(*) FROM peer_votes").fetchone()[0]
    status(f"step 1 OK — {n} items, {n_peers} peer votes in DuckDB{tag}")
    items = data.get_items(con)

    # Step 2: annotator model
    per_ann = ann_mod.setup(con)
    n_bad = con.execute("SELECT count(*) FROM annotators WHERE is_defective").fetchone()[0]
    status(f"step 2 OK — {len(per_ann)} annotators, items/annotator "
           f"{min(per_ann.values())}-{max(per_ann.values())}, {n_bad} defective{tag}")

    # Step 3: submissions (annotator-defect-aware) + item-level injection
    def log_annotate(item_id, purpose, stats):
        log_call(con, item_id, purpose, None, None, None, thresholds.model, mock, stats, thresholds)

    n_new = submissions.generate_missing(con, client, items, thresholds, log_annotate)
    item_counts = defects.inject_item_defects(con)
    truth = dict(con.execute(
        "SELECT defect_type, count(*) FROM defects GROUP BY 1 ORDER BY 1"
    ).fetchall())
    status(f"step 3 OK — {n_new} new submissions; defective items by type: {truth}{tag}")

    # Step 4: grader agent smoke test on smoke_n items, original order only
    smoke_mode = modes[-1]  # 'agent' unless --tools off
    smoke_ids = {it.item_id for it in items[:smoke_n]}
    grade_all(con, client, items, rubric, thresholds, mock, smoke_mode,
              orders=("original",), only_items=smoke_ids)
    ok = len(graded_ids(con, "original", smoke_mode, rubric.version) & smoke_ids)
    if ok < len(smoke_ids):
        fail("grader smoke test", f"only {ok}/{len(smoke_ids)} smoke items produced valid output")
    status(f"step 4 OK — [{smoke_mode}] valid structured output on {ok}/{len(smoke_ids)} smoke items{tag}")

    # Step 5: full grading, both orders, each requested mode
    for mode in modes:
        n_graded = grade_all(con, client, items, rubric, thresholds, mock, mode)
        total = con.execute(
            "SELECT count(*) FROM grades WHERE rubric_version=? AND mode=?",
            [rubric.version, mode],
        ).fetchone()[0]
        if total < 2 * len(items):
            fail("grade", f"[{mode}] expected {2*len(items)} grades, have {total}")
        status(f"step 5 OK — [{mode}] {n_graded} new runs; {total} grades total{tag}")

    # Step 6: routing + calibration sweep + tool metrics
    for mode in modes:
        n_routed = router.route_all(con, thresholds, rubric.version, mode)
        status(f"step 6 — [{mode}] routed {n_routed} items{tag}")
    df = evals.sweep(con, thresholds, rubric.version)
    csv_path, md_path = evals.write_results(con, df, thresholds, rubric.version, out_dir=out_dir)
    status(f"step 6 OK — sweep ({df.height} rows) -> {csv_path.name}, {md_path.name}{tag}")

    for mode in modes:
        records = evals.scored_records(con, rubric.version, thresholds.flip_penalty, mode)
        m = evals.flag_metrics(records, thresholds.high, thresholds.low)
        status(
            f"operating point [{mode}] HIGH={thresholds.high} LOW={thresholds.low}: "
            f"precision={m['precision']:.1%} recall={m['recall']:.1%} "
            f"catch={m['catch_rate']:.1%} review_load={m['review_load']:.1%} "
            f"escalated={m['n_escalated']}{tag}"
        )
    if STATUS_FILE.exists():
        STATUS_FILE.unlink()

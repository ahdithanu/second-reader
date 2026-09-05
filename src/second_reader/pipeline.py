"""Pipeline orchestration: the execution order, status lines, fail-loud STATUS.md."""

from __future__ import annotations

import datetime
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import annotator, data, defects, evals, router
from .config import (
    DATA_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    load_api_key,
    load_rubric,
    load_thresholds,
)
from .db import connect
from .grader import GraderError, grade_call, graded_ids, log_call, save_grade
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
        f"```\npython -m second_reader run --items 300\n```\n"
        f"Completed work (loaded items, submissions, grades) is kept in "
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
            "The grader and synthetic annotator cannot run without it. "
            "Set it, or use --mock for a plumbing-only run.",
        )
    import anthropic
    return anthropic.Anthropic(api_key=key, max_retries=5)


def annotate_missing(con, client, items: list[Item], thresholds, mock: bool) -> int:
    done = annotator.submitted_ids(con)
    todo = [it for it in items if it.item_id not in done]
    for it in todo:
        try:
            sub, stats = annotator.annotate_call(client, it, thresholds)
        except Exception as e:  # noqa: BLE001 — fail loud with context
            fail("annotate", f"item_id={it.item_id}: {type(e).__name__}: {e}")
        annotator.save_submission(con, sub)
        log_call(con, it.item_id, "annotate", None, None, thresholds.model, mock, stats, thresholds)
    return len(todo)


def _grade_task(client, item: Item, sub: Submission, rubric, thresholds, order_label: str):
    if order_label == "swapped":
        item, sub = defects.swapped_view(item, sub)
    out, stats = grade_call(client, item, sub, rubric, thresholds)
    return item.item_id if order_label == "original" else sub.item_id, order_label, out, stats


def grade_items(con, client, items: list[Item], rubric, thresholds, mock: bool,
                orders: tuple[str, ...] = ("original", "swapped")) -> int:
    """Grade every (item, order) pair not already graded. Concurrent API calls,
    single-threaded DB writes."""
    subs = {s.item_id: s for s in annotator.get_submissions(con)}
    tasks = []
    for order_label in orders:
        done = graded_ids(con, order_label, rubric.version)
        for it in items:
            if it.item_id in done:
                continue
            if it.item_id not in subs:
                fail("grade", f"item_id={it.item_id} has no submission; run annotate first")
            tasks.append((it, subs[it.item_id], order_label))

    if not tasks:
        return 0

    workers = 1 if mock else GRADE_WORKERS
    n_done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_grade_task, client, it, sub, rubric, thresholds, order_label): (it, order_label)
            for it, sub, order_label in tasks
        }
        for fut in as_completed(futures):
            it, order_label = futures[fut]
            try:
                item_id, order_label, out, stats = fut.result()
            except Exception as e:  # noqa: BLE001
                ex.shutdown(cancel_futures=True)
                fail("grade", f"item_id={it.item_id} order={order_label}: {type(e).__name__}: {e}")
            purpose = "grade" if out is not None else "grade_failed"
            log_call(con, it.item_id, purpose, order_label, rubric.version,
                     thresholds.model, mock, stats, thresholds)
            if out is None:
                ex.shutdown(cancel_futures=True)
                fail("grade", str(GraderError(it.item_id, order_label, stats.last_error)))
            save_grade(con, it.item_id, order_label, rubric.version, out)
            n_done += 1
            if n_done % 25 == 0:
                status(f"graded {n_done}/{len(tasks)} pending calls")
    return n_done


def run(items_n: int = 300, mock: bool = False, smoke_n: int = 10) -> None:
    # Mock runs live in their own DB and results dir so they can never
    # contaminate real grades or published metrics.
    db_path = DATA_DIR / ("second_reader_mock.duckdb" if mock else "second_reader.duckdb")
    out_dir = RESULTS_DIR / "mock" if mock else RESULTS_DIR
    con = connect(db_path)
    rubric = load_rubric()
    thresholds = load_thresholds()
    client = make_client(mock)
    tag = " [MOCK — numbers not publishable]" if mock else ""

    # Step 1: data load
    n = data.load_items(con, items_n)
    if n < items_n:
        fail("data load", f"only {n} items loaded, expected {items_n}")
    status(f"step 1 OK — {n} items in DuckDB{tag}")
    items = data.get_items(con)

    # Step 2: grader smoke test on N clean items (before any defect injection)
    smoke_items = items[:smoke_n]
    annotate_missing(con, client, smoke_items, thresholds, mock)
    grade_items(con, client, smoke_items, rubric, thresholds, mock, orders=("original",))
    ok = len(graded_ids(con, "original", rubric.version) & {i.item_id for i in smoke_items})
    if ok < len(smoke_items):
        fail("grader smoke test", f"only {ok}/{len(smoke_items)} smoke items produced valid output")
    status(f"step 2 OK — grader returned valid structured output on {ok}/{len(smoke_items)} smoke items{tag}")

    # Step 3: annotate everything, then inject defects
    n_annot = annotate_missing(con, client, items, thresholds, mock)
    fresh_injection = con.execute("SELECT count(*) FROM defects").fetchone()[0] == 0
    counts = defects.inject(con)
    if fresh_injection:
        # Smoke grades of items that just became defective are stale — drop them.
        defective_ids = [r[0] for r in con.execute("SELECT item_id FROM defects").fetchall()]
        con.execute(
            "DELETE FROM grades WHERE item_id IN (SELECT item_id FROM defects)"
        )
        status(f"invalidated stale smoke grades for {sum(1 for i in smoke_items if i.item_id in set(defective_ids))} newly defective items")
    n_def = con.execute("SELECT count(*) FROM defects").fetchone()[0]
    status(f"step 3 OK — {n_annot} new submissions; defects injected: {n_def} total {counts}")

    # Step 4: full grading with order swap
    n_graded = grade_items(con, client, items, rubric, thresholds, mock)
    total = con.execute(
        "SELECT count(*) FROM grades WHERE rubric_version=?", [rubric.version]
    ).fetchone()[0]
    if total < 2 * len(items):
        fail("grade", f"expected {2*len(items)} grades, have {total}")
    status(f"step 4 OK — {n_graded} new grader calls this run; {total} grades total{tag}")

    # Step 5: routing + calibration sweep
    n_routed = router.route_all(con, thresholds, rubric.version)
    df = evals.sweep(con, thresholds, rubric.version)
    csv_path, md_path = evals.write_results(df, thresholds, out_dir=out_dir)
    status(f"step 5 OK — routed {n_routed} items; sweep ({df.height} configs) -> {csv_path.name}, {md_path.name}{tag}")

    op = df.filter((df["high"] == thresholds.high) & (df["low"] == thresholds.low))
    if op.height:
        r = op.row(0, named=True)
        status(
            f"operating point HIGH={thresholds.high} LOW={thresholds.low}: "
            f"precision={r['precision']:.1%} recall={r['recall']:.1%} "
            f"review_load={r['review_load']:.1%}{tag}"
        )
    if STATUS_FILE.exists():
        STATUS_FILE.unlink()

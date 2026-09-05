"""CLI entrypoint: python -m second_reader <command>"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    p = argparse.ArgumentParser(prog="second_reader")
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="full pipeline: load -> annotators -> submissions -> smoke -> grade -> route -> sweep")
    run_p.add_argument("--items", type=int, default=150)
    run_p.add_argument("--tools", choices=["on", "off", "both"], default="both",
                       help="agent loop (on), single-call ablation (off), or both for the comparison")
    run_p.add_argument("--mock", action="store_true",
                       help="use the deterministic mock client (plumbing verification only)")
    run_p.add_argument("--smoke", type=int, default=10, help="items for the grader smoke test")

    show_p = sub.add_parser("show", help="dump one item: question, responses, submission, grades, route")
    show_p.add_argument("--item-id", required=True)
    show_p.add_argument("--mock", action="store_true", help="read the mock-run DB")

    trace_p = sub.add_parser("trace", help="print the verbatim agent trace for one item")
    trace_p.add_argument("--item-id", required=True)
    trace_p.add_argument("--order", choices=["original", "swapped"], default="original")
    trace_p.add_argument("--mode", choices=["agent", "single"], default="agent")
    trace_p.add_argument("--mock", action="store_true", help="read the mock-run DB")

    rep_p = sub.add_parser("report", help="metrics at the operating point + missed defects by type")
    rep_p.add_argument("--mode", choices=["agent", "single"], default="agent")
    rep_p.add_argument("--defect", default=None, help="list missed items of this defect type")
    rep_p.add_argument("--mock", action="store_true", help="read the mock-run DB")

    ann_p = sub.add_parser("annotators", help="list annotators with defect truth and item counts")
    ann_p.add_argument("--mock", action="store_true", help="read the mock-run DB")

    args = p.parse_args()

    if args.cmd == "run":
        from .pipeline import run
        run(items_n=args.items, mock=args.mock, smoke_n=args.smoke, tools=args.tools)
        return

    from .config import DATA_DIR, load_rubric, load_thresholds
    from .db import connect

    db_path = DATA_DIR / ("second_reader_mock.duckdb" if args.mock else "second_reader.duckdb")
    con = connect(db_path)
    rubric = load_rubric()
    thresholds = load_thresholds()

    if args.cmd == "show":
        for table in ("items", "submissions", "grades", "routed"):
            cur = con.execute(f"SELECT * FROM {table} WHERE item_id=?", [args.item_id])
            cols = [d[0] for d in cur.description]
            for r in cur.fetchall():
                print(f"\n== {table} ==")
                for c, v in zip(cols, r):
                    v = str(v)
                    print(f"  {c}: {v[:400]}{'…' if len(v) > 400 else ''}")
        return

    if args.cmd == "trace":
        row = con.execute(
            "SELECT trace, n_tool_calls, hit_cap, escalated FROM traces "
            "WHERE item_id=? AND order_label=? AND mode=? AND rubric_version=?",
            [args.item_id, args.order, args.mode, rubric.version],
        ).fetchone()
        if row is None:
            print("no trace found")
            return
        trace, n_calls, hit_cap, escalated = row
        print(json.dumps({
            "item_id": args.item_id, "order": args.order, "mode": args.mode,
            "n_tool_calls": n_calls, "hit_cap": bool(hit_cap),
            "escalated": bool(escalated), "trace": json.loads(trace),
        }, indent=2))
        return

    if args.cmd == "report":
        from .evals import flag_metrics, scored_records
        records = scored_records(con, rubric.version, thresholds.flip_penalty, args.mode)
        m = flag_metrics(records, thresholds.high, thresholds.low)
        print(json.dumps(m, indent=2))
        if args.defect:
            missed = [r for r in records
                      if r["defect_type"] == args.defect
                      and not r["escalated"] and r["score"] > thresholds.low]
            print(f"\nMISSED {args.defect} items in mode={args.mode} (score above LOW={thresholds.low}):")
            for r in missed:
                print(f"  {r['item_id']}  score={r['score']:.2f}")
        return

    if args.cmd == "annotators":
        rows = con.execute("""
            SELECT a.annotator_id, a.is_defective, a.defect_type, count(s.item_id)
            FROM annotators a LEFT JOIN assignments s USING (annotator_id)
            GROUP BY 1,2,3 ORDER BY 1
        """).fetchall()
        for aid, bad, dt, n in rows:
            print(f"  {aid}  items={n:2d}  {'DEFECTIVE: ' + dt if bad else 'clean'}")
        return


if __name__ == "__main__":
    main()

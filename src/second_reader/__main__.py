"""CLI entrypoint: python -m second_reader <command>"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    p = argparse.ArgumentParser(prog="second_reader")
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="full pipeline: load -> smoke -> inject -> grade -> route -> sweep")
    run_p.add_argument("--items", type=int, default=300)
    run_p.add_argument("--mock", action="store_true",
                       help="use the deterministic mock client (plumbing verification only)")
    run_p.add_argument("--smoke", type=int, default=10, help="items for the step-2 grader smoke test")

    show_p = sub.add_parser("show", help="dump one item: question, responses, submission, grades, route")
    show_p.add_argument("--item-id", required=True)
    show_p.add_argument("--mock", action="store_true", help="read the mock-run DB")

    rep_p = sub.add_parser("report", help="metrics at the operating point + missed defects by type")
    rep_p.add_argument("--defect", default=None, help="list missed items of this defect type")
    rep_p.add_argument("--mock", action="store_true", help="read the mock-run DB")

    args = p.parse_args()

    if args.cmd == "run":
        from .pipeline import run
        run(items_n=args.items, mock=args.mock, smoke_n=args.smoke)
        return

    from .config import DATA_DIR, load_rubric, load_thresholds
    from .db import connect

    db_path = DATA_DIR / ("second_reader_mock.duckdb" if args.mock else "second_reader.duckdb")
    con = connect(db_path)
    rubric = load_rubric()
    thresholds = load_thresholds()

    if args.cmd == "show":
        for table in ("items", "submissions", "grades", "routed"):
            rows = con.execute(f"SELECT * FROM {table} WHERE item_id=?", [args.item_id]).fetchall()
            cols = [d[0] for d in con.description]
            print(f"\n== {table} ==")
            for r in rows:
                for c, v in zip(cols, r):
                    v = str(v)
                    print(f"  {c}: {v[:400]}{'…' if len(v) > 400 else ''}")
        return

    if args.cmd == "report":
        from .evals import flag_metrics, scored_records
        records = scored_records(con, rubric.version, thresholds.flip_penalty)
        m = flag_metrics(records, thresholds.high, thresholds.low)
        print(json.dumps(m, indent=2))
        if args.defect:
            missed = [
                r["item_id"] for r in records
                if r["defect_type"] == args.defect and r["score"] > thresholds.low
            ]
            print(f"\nMISSED {args.defect} items (score above LOW={thresholds.low}):")
            for item_id in missed:
                score = next(r["score"] for r in records if r["item_id"] == item_id)
                print(f"  {item_id}  score={score:.2f}")
        return


if __name__ == "__main__":
    main()

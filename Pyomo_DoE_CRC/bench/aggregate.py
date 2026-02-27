import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _flatten(record: Dict[str, Any]) -> Dict[str, Any]:
    flat = {}
    for k, v in record.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}.{kk}"] = vv
        else:
            flat[k] = v
    return flat


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="results/raw")
    parser.add_argument("--out", default="results/agg/results.csv")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for path in sorted(in_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        rows.append(_flatten(data))

    if not rows:
        out_path.write_text("")
        return 0

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

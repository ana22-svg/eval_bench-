import argparse
import sqlite3
from html import escape
from pathlib import Path


def write_leaderboard(db_path: str, output_path: str) -> None:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT id, suite, dataset_version, model, prompt_version, mean_score, cost_usd, created_at
        FROM runs
        ORDER BY mean_score DESC, cost_usd ASC, created_at DESC
        """
    ).fetchall()
    conn.close()

    table_rows = []
    for row in rows:
        run_id, suite, dataset, model, prompt_version, mean_score, cost, created_at = row
        table_rows.append(
            "<tr>"
            f"<td>{escape(run_id)}</td>"
            f"<td>{escape(suite)}</td>"
            f"<td>{escape(model)}</td>"
            f"<td>{escape(prompt_version)}</td>"
            f"<td>{escape(dataset)}</td>"
            f"<td>{mean_score:.3f}</td>"
            f"<td>${cost:.6f}</td>"
            f"<td>{escape(created_at)}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Eval leaderboard</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 32px; color: #172026; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d7dde3; padding: 8px; text-align: left; }}
    th {{ background: #f4f6f8; }}
  </style>
</head>
<body>
  <h1>Eval leaderboard</h1>
  <table>
    <thead>
      <tr><th>Run</th><th>Suite</th><th>Model</th><th>Prompt</th><th>Dataset</th><th>Score</th><th>Cost</th><th>Created</th></tr>
    </thead>
    <tbody>{''.join(table_rows)}</tbody>
  </table>
</body>
</html>
"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="runs/eval_bench.sqlite3")
    parser.add_argument("--output", default="reports/leaderboard.html")
    args = parser.parse_args()
    write_leaderboard(args.db, args.output)
    print(f"Leaderboard: {args.output}")


if __name__ == "__main__":
    main()

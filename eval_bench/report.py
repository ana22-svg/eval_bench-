from html import escape
from pathlib import Path


def write_html_report(summary: dict, results: list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        rows.append(
            "<tr>"
            f"<td>{escape(result.example_id)}</td>"
            f"<td>{escape(result.input)}</td>"
            f"<td>{escape(result.prediction)}</td>"
            f"<td>{escape(result.gold or '')}</td>"
            f"<td>{result.score:.3f}</td>"
            f"<td>{escape(result.reasoning or '')}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Eval report - {escape(summary['run_id'])}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 32px; color: #172026; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d7dde3; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f6f8; }}
    .metric {{ display: inline-block; margin: 0 24px 20px 0; }}
  </style>
</head>
<body>
  <h1>Eval report</h1>
  <div class="metric"><strong>Suite:</strong> {escape(summary['suite'])}</div>
  <div class="metric"><strong>Model:</strong> {escape(summary['model'])}</div>
  <div class="metric"><strong>Dataset:</strong> {escape(summary['dataset_version'])}</div>
  <div class="metric"><strong>Mean score:</strong> {summary['mean_score']:.3f}</div>
  <div class="metric"><strong>Cost:</strong> ${summary['cost_usd']:.6f}</div>
  <table>
    <thead>
      <tr><th>ID</th><th>Input</th><th>Prediction</th><th>Gold</th><th>Score</th><th>Reasoning</th></tr>
    </thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    Path(path).write_text(html, encoding="utf-8")

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path


def init_db(path: str = "runs/eval_bench.sqlite3") -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            suite TEXT NOT NULL,
            dataset_version TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            mean_score REAL NOT NULL,
            cost_usd REAL NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS example_results (
            run_id TEXT NOT NULL,
            example_id TEXT NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY (run_id, example_id)
        )
        """
    )
    return conn


def save_run_record(run_id: str, suite: str, dataset_version: str, summary: dict, results: list, path: str = "runs/eval_bench.sqlite3") -> None:
    conn = init_db(path)
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs
            (id, suite, dataset_version, model, prompt_version, mean_score, cost_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                suite,
                dataset_version,
                summary["model"],
                summary["prompt_version"],
                summary["mean_score"],
                summary["cost_usd"],
            ),
        )
        for result in results:
            conn.execute(
                """
                INSERT OR REPLACE INTO example_results (run_id, example_id, payload)
                VALUES (?, ?, ?)
                """,
                (run_id, result.example_id, json.dumps(asdict(result))),
            )
    conn.close()

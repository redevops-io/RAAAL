"""Reporting helpers for console + artifacts."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
from tabulate import tabulate

from .config import UNIVERSE
from .pipeline import AllocationResult

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def allocation_table(result: AllocationResult) -> str:
    rows = []
    for asset in UNIVERSE:
        weight = result.weights.get(asset.ticker, 0.0)
        trend = result.expected_returns.get(asset.ticker, 0.0) * 252
        rationale = result.rationales.get(asset.ticker, "")
        rows.append(
            {
                "ETF": asset.ticker,
                "Weight": f"{weight:.1%}",
                "Trend(ann)": f"{trend:.1%}",
                "Class": asset.asset_class,
                "Note": rationale,
            }
        )
    return tabulate(rows, headers="keys", tablefmt="github")


def save_reports(result: AllocationResult) -> Dict[str, str]:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = REPORTS_DIR / f"allocation_{timestamp}.csv"
    json_path = REPORTS_DIR / f"allocation_{timestamp}.json"

    df = pd.DataFrame(
        {
            "ticker": list(result.weights.keys()),
            "weight": list(result.weights.values()),
        }
    )
    df.to_csv(csv_path, index=False)

    payload = {
        "regime": result.regime.name,
        "diagnostics": result.regime.diagnostics,
        "metrics": result.metrics,
        "weights": result.weights,
        "rf_per_day": result.rf_rate,
    }
    json_path.write_text(json.dumps(payload, indent=2))

    return {"csv": str(csv_path), "json": str(json_path)}

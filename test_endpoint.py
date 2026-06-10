"""Lab 5 — Validate the deployed MLflow inference endpoint.

Usage:
    python test_endpoint.py                          # defaults to localhost:5000
    python test_endpoint.py http://<lightning-url>    # deployed URL
"""

from __future__ import annotations

import sys

import requests

# ---------------------------------------------------------------------------
# Target endpoint
# ---------------------------------------------------------------------------
BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
INVOCATIONS_URL = f"{BASE_URL.rstrip('/')}/invocations"

# ---------------------------------------------------------------------------
# Sample payload — raw Titanic passenger rows
# The sklearn Pipeline inside the model handles feature engineering
# (TitanicFeatureEngineer), so we send the original CSV-level columns.
# ---------------------------------------------------------------------------
payload = {
    "dataframe_split": {
        "columns": [
            "Pclass", "Name", "Sex", "Age", "SibSp",
            "Parch", "Ticket", "Fare", "Cabin", "Embarked",
        ],
        "data": [
            # Expected: unlikely to survive (3rd class, male, alone)
            [3, "Braund, Mr. Owen Harris", "male", 22.0, 1, 0, "A/5 21171", 7.25, None, "S"],
            # Expected: likely to survive (1st class, female, child)
            [1, "Cumings, Mrs. John Bradley", "female", 38.0, 1, 0, "PC 17599", 71.2833, "C85", "C"],
        ],
    }
}


def main() -> None:
    print(f"→ POST {INVOCATIONS_URL}")
    try:
        response = requests.post(
            INVOCATIONS_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
    except requests.ConnectionError:
        print(f"✗ Could not connect to {BASE_URL}")
        print("  Make sure the MLflow serving container is running.")
        sys.exit(1)

    print(f"← Status: {response.status_code}")

    if response.ok:
        predictions = response.json()
        print(f"✓ Predictions: {predictions}")
        print()
        print("  Passenger 1 (3rd class male)  → predicted:", predictions["predictions"][0])
        print("  Passenger 2 (1st class female) → predicted:", predictions["predictions"][1])
    else:
        print(f"✗ Error: {response.text}")
        sys.exit(1)


if __name__ == "__main__":
    main()

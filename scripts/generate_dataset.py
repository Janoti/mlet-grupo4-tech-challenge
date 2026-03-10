from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main(n_rows: int = 50000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{idx:06d}" for idx in range(1, n_rows + 1)],
            "tenure_months": rng.integers(0, 73, size=n_rows),
            "monthly_charges": rng.uniform(20, 150, size=n_rows).round(2),
            "total_charges": (rng.uniform(0, 9000, size=n_rows)).round(2),
            "support_calls_30d": rng.integers(0, 6, size=n_rows),
            "contract": rng.choice(["month-to-month", "one-year", "two-year"], size=n_rows, p=[0.55, 0.25, 0.20]),
            "payment_method": rng.choice(["credit_card", "bank_transfer", "boleto", "pix"], size=n_rows),
            "internet_service": rng.choice(["dsl", "fiber", "none"], size=n_rows, p=[0.25, 0.60, 0.15]),
            # churn simples, sem “modelo”: só uma taxa base + variação leve
            "churn": rng.choice([0, 1], size=n_rows, p=[0.73, 0.27]),
        }
    )

    out = Path("data/raw/telecom_churn_base.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote: {out} rows={len(df)}")


if __name__ == "__main__":
    main()
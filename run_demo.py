"""Command-line entry point: builds the disclosure-change panel for the
default (or a custom) firm universe, prints the run diagnostics, saves the
panel to CSV, and writes a summary chart to results/.

This exists so the pipeline can produce a reviewable artifact (CSV + PNG)
without requiring Streamlit — useful for a README results section or for
anyone who wants to verify the pipeline actually runs against live SEC
data before reading the code.

Usage:
    python run_demo.py "Your Name your_email@school.edu"
    python run_demo.py "Your Name your_email@school.edu" --tickers AAPL MSFT JPM
    python run_demo.py "Your Name your_email@school.edu" --validate
"""
import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import DEFAULT_FIRM_UNIVERSE, EdgarClient, build_panel, validate_against_forward_returns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("user_agent", help="SEC contact string, e.g. 'Jane Doe jdoe@school.edu'")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_FIRM_UNIVERSE)
    parser.add_argument("--n-filings", type=int, default=6)
    parser.add_argument("--validate", action="store_true", help="also run forward-return validation")
    parser.add_argument("--window-days", type=int, default=21)
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    client = EdgarClient(user_agent=args.user_agent)
    panel, stats = build_panel(client, args.tickers, n_filings=args.n_filings)

    print("\n=== Run diagnostics ===")
    for k, v in stats.as_dict().items():
        if k != "ticker_diagnostics":
            print(f"{k}: {v}")
    print("\nPer-ticker funnel:")
    for ticker, diag in stats.ticker_diagnostics.items():
        print(f"  {ticker}: {diag}")

    if panel.empty:
        print("\nNo panel rows were produced. Check the diagnostics above.", file=sys.stderr)
        sys.exit(1)

    panel_path = f"{args.out_dir}/panel.csv"
    panel.to_csv(panel_path, index=False)
    print(f"\nSaved panel ({len(panel)} rows) to {panel_path}")

    fig, ax = plt.subplots(figsize=(9, 5))
    for ticker, grp in panel.groupby("ticker"):
        grp = grp.sort_values("fiscal_year")
        ax.plot(grp["fiscal_year"], grp["change_score"], marker="o", label=ticker)
    ax.set_xlabel("Fiscal year")
    ax.set_ylabel("Change score (1 - cosine similarity)")
    ax.set_title("Year-over-year Item 1A textual change by firm")
    ax.legend()
    fig.tight_layout()
    chart_path = f"{args.out_dir}/change_score_by_firm.png"
    fig.savefig(chart_path, dpi=150)
    print(f"Saved chart to {chart_path}")

    if args.validate:
        validated = validate_against_forward_returns(
            panel, window_trading_days=args.window_days, benchmark_ticker=args.benchmark
        )
        if validated.empty:
            print("\nValidation returned no matched rows.")
        else:
            metric_col = "excess_return" if validated["excess_return"].notna().any() else "forward_return"
            corr = validated[["change_score", metric_col]].corr().iloc[0, 1]
            validated_path = f"{args.out_dir}/validated.csv"
            validated.to_csv(validated_path, index=False)
            print(f"\nSaved validation ({len(validated)} rows) to {validated_path}")
            print(f"Correlation(change_score, {metric_col}) = {corr:.3f}  (n={len(validated)})")


if __name__ == "__main__":
    main()

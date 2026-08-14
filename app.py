import pandas as pd
import streamlit as st

from pipeline import (
    DEFAULT_FIRM_UNIVERSE,
    EdgarClient,
    build_panel,
    validate_against_forward_returns,
)

st.set_page_config(page_title="EDGAR 10-K Disclosure Change Panel", layout="wide")

st.title("SEC EDGAR 10-K Disclosure Change & Risk Panel")
st.markdown(
    "Tracks year-over-year textual change in Item 1A (Risk Factors) across a "
    "firm universe, following the disclosure-change literature (Cohen, Malloy "
    "& Nguyen 2020). This builds a cross-sectional firm-year panel, not a "
    "single-company time series."
)

st.sidebar.header("Configuration")
user_agent = st.sidebar.text_input(
    "SEC contact string (required)",
    value="",
    placeholder="Your Name your_email@school.edu",
    help="SEC requires a real, identifiable User-Agent on every request. "
         "This is not optional and the app will not run without it.",
)
tickers_input = st.sidebar.text_input(
    "Tickers (comma-separated)", value=", ".join(DEFAULT_FIRM_UNIVERSE)
)
n_filings = st.sidebar.slider("Filings per firm", min_value=2, max_value=10, value=6)
use_cache = st.sidebar.checkbox(
    "Cache SEC responses on disk", value=True,
    help="Filings don't change once filed, so cached responses speed up "
         "reruns and reduce load on SEC's servers.",
)
run_validation = st.sidebar.checkbox(
    "Validate against forward returns (requires yfinance, slower)", value=False
)
if run_validation:
    window_trading_days = st.sidebar.slider(
        "Return window (trading days after filing)", min_value=5, max_value=63, value=21,
        help="Holding period starts the trading day after the 10-K's actual filing date, "
             "not a fixed calendar window — so this is aligned to when the market could "
             "actually have reacted to the filing.",
    )
    benchmark_ticker = st.sidebar.text_input(
        "Benchmark ticker (for excess return)", value="SPY",
        help="Stock return minus benchmark return over the same window, so the "
             "correlation isn't just picking up broad market moves.",
    )

if st.sidebar.button("Build Panel"):
    if not user_agent or "@" not in user_agent:
        st.error("Enter a real SEC contact string before running — see the sidebar help text.")
        st.stop()

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    with st.spinner(f"Building panel for {len(tickers)} firms — this hits SEC's live API "
                    f"and can take a few minutes for larger universes..."):
        try:
            client = EdgarClient(user_agent=user_agent, use_cache=use_cache)
            panel, stats = build_panel(client, tickers, n_filings=n_filings)
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

    with st.expander("Run details (what was skipped and why)", expanded=True):
        st.write(f"Tickers requested: {stats.tickers_requested}")
        st.write(f"Firm-years successfully extracted: {stats.firm_years_extracted}")
        if stats.firm_years_low_confidence:
            st.write(f"Firm-years extracted but flagged low confidence: {len(stats.firm_years_low_confidence)}")
        st.write(f"Panel rows built: {stats.panel_rows_built}")
        if stats.panel_rows_flagged:
            st.warning(
                f"{stats.panel_rows_flagged} of {stats.panel_rows_built} panel rows are flagged "
                f"for low-confidence extraction or an extreme year-over-year swing — see the "
                f"flagged rows table below before treating those numbers as reliable."
            )

        if stats.ticker_diagnostics:
            st.write("Per-ticker funnel (where each ticker succeeded or dropped out):")
            diag_df = pd.DataFrame.from_dict(stats.ticker_diagnostics, orient="index")
            diag_df.index.name = "ticker"
            st.dataframe(diag_df, use_container_width=True)

        if stats.tickers_skipped_no_cik:
            st.write(f"Tickers with no CIK match: {stats.tickers_skipped_no_cik}")
        if stats.tickers_skipped_fetch_error:
            st.write(f"Tickers that failed to fetch: {stats.tickers_skipped_fetch_error}")
        if stats.filings_download_failed:
            st.write(f"Filings that failed to download: {stats.filings_download_failed}")
        if stats.filings_extraction_failed:
            st.write(f"Filings where Item 1A extraction failed outright: {stats.filings_extraction_failed}")
        if stats.firm_years_low_confidence:
            st.write(f"Filings extracted but flagged low confidence (reason shown): {stats.firm_years_low_confidence}")

    if panel.empty:
        st.error("No panel rows were produced. Check tickers and try again — "
                 "see the run details expander above for per-firm failures.")
        st.stop()

    st.success(f"Built a panel of {len(panel)} firm-year observations across "
               f"{panel['ticker'].nunique()} firms.")

    st.subheader("Panel Dataset")
    display_panel = panel.copy()
    display_panel["flagged"] = display_panel["flagged"].map({True: "⚠️ flagged", False: ""})
    st.dataframe(display_panel, use_container_width=True)

    flagged_rows = panel[panel["flagged"]]
    if not flagged_rows.empty:
        with st.expander(f"⚠️ {len(flagged_rows)} flagged row(s) — verify before citing", expanded=False):
            st.caption(
                "These rows involve a low-confidence text extraction or an extreme year-over-year "
                "swing that doesn't match typical disclosure patterns. They're kept in the panel "
                "rather than silently dropped, but should be spot-checked against the source filing "
                "before being used as evidence of anything."
            )
            st.dataframe(
                flagged_rows[["ticker", "fiscal_year", "change_score", "flag_reason"]],
                use_container_width=True,
            )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Change Score by Firm Over Time")
        pivot = panel.pivot_table(index="fiscal_year", columns="ticker", values="change_score")
        st.line_chart(pivot)
    with col2:
        st.subheader("Cross-Sectional Distribution of Change Score")
        st.bar_chart(panel.groupby("ticker")["change_score"].mean())

    csv_data = panel.to_csv(index=False).encode("utf-8")
    st.download_button("Download Panel (CSV)", data=csv_data,
                       file_name="edgar_diff_panel.csv", mime="text/csv")

    if run_validation:
        with st.spinner(f"Fetching {window_trading_days}-trading-day forward returns "
                        f"(vs. {benchmark_ticker}) via yfinance..."):
            try:
                validated = validate_against_forward_returns(
                    panel, window_trading_days=window_trading_days, benchmark_ticker=benchmark_ticker
                )
            except Exception as e:
                st.error(f"Return validation failed: {e}")
                validated = pd.DataFrame()

        if validated.empty:
            st.warning("Validation returned no matched rows — this can happen if filing dates "
                      "are too recent to have a full forward window yet, or if the ticker/"
                      "benchmark data wasn't available for the needed range.")
        else:
            st.subheader("Validation: Change Score vs. Forward Return")
            st.caption(
                f"Return window: the {window_trading_days} trading days starting the day "
                f"after each firm's actual 10-K filing date. Excess return subtracts "
                f"{benchmark_ticker}'s return over the identical window, so it isn't just "
                f"picking up market-wide moves that happen to coincide with filing season."
            )

            has_excess = "excess_return" in validated.columns and validated["excess_return"].notna().any()
            metric_col = "excess_return" if has_excess else "forward_return"
            metric_label = "excess return" if has_excess else "raw return (no benchmark data)"

            corr_all = validated[["change_score", metric_col]].corr().iloc[0, 1]
            clean = validated[~validated["flagged"]] if "flagged" in validated.columns else validated
            vcol1, vcol2 = st.columns(2)
            with vcol1:
                st.metric(f"Correlation w/ {metric_label}, all rows", f"{corr_all:.3f}", help=f"n={len(validated)}")
            with vcol2:
                if len(clean) >= 3:
                    corr_clean = clean[["change_score", metric_col]].corr().iloc[0, 1]
                    st.metric(f"Correlation w/ {metric_label}, flagged rows excluded",
                             f"{corr_clean:.3f}", help=f"n={len(clean)}")
                else:
                    st.metric(f"Correlation w/ {metric_label}, flagged rows excluded",
                             "n/a", help="too few clean rows")
            st.scatter_chart(validated, x="change_score", y=metric_col)

            display_cols = ["ticker", "fiscal_year", "filing_date", "window_start", "window_end",
                            "change_score", "forward_return", "benchmark_return", "excess_return"]
            display_cols = [c for c in display_cols if c in validated.columns]
            st.dataframe(validated[display_cols], use_container_width=True)

            st.caption(
                "Not a risk-adjusted asset-pricing test — no size/value/momentum controls "
                "beyond the single benchmark subtraction, and no handling of delisted or "
                "acquired firms. Sample size at this firm count is far too small to treat "
                "either correlation as statistically meaningful on its own — this is a "
                "directional sanity check, not a publication-grade estimate."
            )

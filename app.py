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
run_validation = st.sidebar.checkbox(
    "Validate against forward returns (requires yfinance, slower)", value=False
)

if st.sidebar.button("Build Panel"):
    if not user_agent or "@" not in user_agent:
        st.error("Enter a real SEC contact string before running — see the sidebar help text.")
        st.stop()

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    with st.spinner(f"Building panel for {len(tickers)} firms — this hits SEC's live API "
                    f"and can take a few minutes for larger universes..."):
        try:
            client = EdgarClient(user_agent=user_agent)
            panel = build_panel(client, tickers, n_filings=n_filings)
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

    if panel.empty:
        st.error("No panel rows were produced. Check tickers and try again — "
                 "see terminal/log output for per-firm extraction failures.")
        st.stop()

    st.success(f"Built a panel of {len(panel)} firm-year observations across "
               f"{panel['ticker'].nunique()} firms.")

    st.subheader("Panel Dataset")
    st.dataframe(panel, use_container_width=True)

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
        with st.spinner("Fetching forward returns via yfinance and computing correlation..."):
            validated = validate_against_forward_returns(panel)
        if validated.empty:
            st.warning("Validation returned no matched rows.")
        else:
            corr = validated[["change_score", "forward_return"]].corr().iloc[0, 1]
            st.subheader("Validation: Change Score vs. Forward Return")
            st.metric("Correlation (change_score, forward_return)", f"{corr:.3f}")
            st.scatter_chart(validated, x="change_score", y="forward_return")
            st.caption(
                "Filing dates in this validation are approximated — treat this as a "
                "directional check, not a publication-grade estimate."
            )
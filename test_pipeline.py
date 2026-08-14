"""Offline unit tests for pipeline.py. No network access is used anywhere
in this file — EdgarClient.get_text is monkeypatched wherever SEC data is
needed, and yfinance is never imported.

Run with:  pytest tests/ -v
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import (
    EdgarClient,
    ExtractionResult,
    _trading_day_return,
    extract_item_1a_detailed,
    fail_fast_user_agent,
    parse_company_tickers,
)


# ---------------------------------------------------------------------------
# fail_fast_user_agent
# ---------------------------------------------------------------------------

def test_user_agent_rejects_empty():
    with pytest.raises(ValueError):
        fail_fast_user_agent("")


def test_user_agent_rejects_no_email():
    with pytest.raises(ValueError):
        fail_fast_user_agent("Jane Doe")


def test_user_agent_rejects_no_space():
    with pytest.raises(ValueError):
        fail_fast_user_agent("jdoe@university.edu")


def test_user_agent_accepts_valid():
    fail_fast_user_agent("Jane Doe jdoe@university.edu")  # should not raise


# ---------------------------------------------------------------------------
# parse_company_tickers
# ---------------------------------------------------------------------------

def test_parse_company_tickers_basic():
    raw = {
        "0": {"cik_str": 320193, "ticker": "aapl", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
    }
    mapping = parse_company_tickers(raw)
    assert mapping["AAPL"] == "320193"
    assert mapping["MSFT"] == "789019"


def test_parse_company_tickers_skips_incomplete_entries():
    raw = {"0": {"cik_str": "", "ticker": "AAPL"}, "1": {"cik_str": 1, "ticker": ""}}
    mapping = parse_company_tickers(raw)
    assert mapping == {}


# ---------------------------------------------------------------------------
# extract_item_1a_detailed
# ---------------------------------------------------------------------------

def _wrap_html(body: str) -> str:
    return f"<html><body>{body}</body></html>"


def test_extract_high_confidence_real_section():
    risk_prose = " ".join(["Our business faces substantial risk from competition."] * 200)
    html_doc = _wrap_html(
        f"<h1>PART I</h1><p>Item 1A. Risk Factors</p><p>{risk_prose}</p>"
        f"<h1>Item 1B. Unresolved Staff Comments</h1><p>None.</p>"
        f"<h1>Item 2. Properties</h1>"
    )
    result = extract_item_1a_detailed(html_doc)
    assert result.confidence == "high"
    assert result.method == "matched"
    assert "competition" in result.text
    assert "Unresolved Staff Comments" not in result.text


def test_extract_rejects_toc_style_false_positive():
    # A dense table-of-contents block that mentions several "Item N"
    # headings in a tight run should NOT be mistaken for the actual
    # risk-factors prose, even though it technically satisfies the
    # start/end anchor pattern.
    toc = (
        "Item 1. Business Item 1A. Risk Factors Item 1B. Unresolved Staff Comments "
        "Item 2. Properties Item 3. Legal Proceedings "
    )
    real_prose = " ".join(["Actual discussion of risk factors follows here."] * 300)
    html_doc = _wrap_html(
        f"<p>{toc}</p><p>Item 1A. Risk Factors</p><p>{real_prose}</p>"
        f"<h1>Item 1B. Unresolved Staff Comments</h1>"
    )
    result = extract_item_1a_detailed(html_doc)
    # Should either land on the real prose (high) or fall back to a
    # low-confidence tier — but must never silently return the bare TOC.
    assert result.text is not None
    assert "Actual discussion" in result.text or result.confidence != "high"


def test_extract_no_item_1a_returns_none():
    html_doc = _wrap_html("<p>This document has no risk factors section at all.</p>")
    result = extract_item_1a_detailed(html_doc)
    assert result.text is None
    assert result.confidence == "none"


def test_extract_short_section_is_low_confidence_not_high():
    html_doc = _wrap_html(
        "<p>Item 1A. Risk Factors</p><p>Brief risk statement, only a little text.</p>"
        "<h1>Item 2. Properties</h1>"
    )
    result = extract_item_1a_detailed(html_doc)
    assert result.confidence in ("low", "none")
    assert result.confidence != "high"


# ---------------------------------------------------------------------------
# fetch_filing_index pagination / array-alignment fix
# ---------------------------------------------------------------------------

class _FakeEdgarClient(EdgarClient):
    """EdgarClient subclass that serves canned JSON instead of hitting the
    network, so the pagination-alignment fix can be tested offline."""

    def __init__(self, pages_by_url: dict):
        # Skip the real __init__ (which validates user agent, opens a
        # session, and may create a cache dir) — we only need get_text().
        self._pages_by_url = pages_by_url
        self.use_cache = False

    def get_text(self, url: str) -> str:
        return json.dumps(self._pages_by_url[url])


def test_fetch_filing_index_handles_short_reportdate_on_older_page():
    """Regression test for the bug where an older paginated submissions
    page with a short reportDate array (relative to its own form array)
    would misalign every filing after it. Each page must be padded to
    its own length before pages are concatenated."""
    recent = {
        "form": ["10-K"],
        "accessionNumber": ["0001-24-000001"],
        "filingDate": ["2024-02-15"],
        "primaryDocument": ["aapl-20231230.htm"],
        "reportDate": ["2023-12-30"],
    }
    # Older page: form/accession/filingDate/primaryDocument all have 2
    # entries, but reportDate is short by one (a real pattern seen in
    # SEC's older paginated submissions files).
    older_page = {
        "form": ["10-K", "10-K"],
        "accessionNumber": ["0001-22-000002", "0001-21-000003"],
        "filingDate": ["2022-02-10", "2021-02-08"],
        "primaryDocument": ["aapl-20211230.htm", "aapl-20201230.htm"],
        "reportDate": ["2021-12-30"],  # only 1 of 2 — should NOT shift the second filing
    }
    main_doc = {
        "filings": {
            "recent": recent,
            "files": [{"name": "CIK0000320193-submissions-001.json"}],
        }
    }
    pages = {
        "https://data.sec.gov/submissions/CIK0000320193.json": main_doc,
        "https://data.sec.gov/submissions/CIK0000320193-submissions-001.json": older_page,
    }
    client = _FakeEdgarClient(pages)
    filings = client.fetch_filing_index("AAPL", "320193")

    by_year = {f.fiscal_year: f for f in filings}
    # The 2021-02-08 filing has no reportDate on its page, so it should
    # fall back to filingDate_year - 1 = 2020, NOT silently borrow the
    # neighboring filing's 2021 reportDate due to an alignment shift.
    assert 2020 in by_year
    assert by_year[2020].accession_number == "0001-21-000003"
    assert 2021 in by_year
    assert by_year[2021].accession_number == "0001-22-000002"
    assert 2023 in by_year


# ---------------------------------------------------------------------------
# _trading_day_return (return-window alignment for validation)
# ---------------------------------------------------------------------------

def _make_price_history(dates, closes):
    idx = pd.to_datetime(dates)
    return pd.DataFrame({"Close": closes}, index=idx)


def test_trading_day_return_basic():
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    closes = [100.0 + i for i in range(30)]
    hist = _make_price_history(dates, closes)

    event_date = dates[5]  # filing date
    result = _trading_day_return(hist, event_date, window_trading_days=10)
    assert result is not None
    ret, start, end = result
    # start should be the first trading day strictly AFTER the event date
    assert pd.Timestamp(start) > event_date
    expected_ret = (closes[16] - closes[6]) / closes[6]
    assert ret == pytest.approx(expected_ret)


def test_trading_day_return_none_when_window_exceeds_history():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    hist = _make_price_history(dates, [100, 101, 102, 103, 104])
    result = _trading_day_return(hist, dates[0], window_trading_days=21)
    assert result is None


def test_trading_day_return_none_on_empty_history():
    hist = pd.DataFrame({"Close": []})
    result = _trading_day_return(hist, pd.Timestamp("2024-01-01"), window_trading_days=5)
    assert result is None

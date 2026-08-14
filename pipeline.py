"""EDGAR 10-K disclosure-change panel builder.

Builds a firm-year panel of year-over-year textual change in Item 1A
(Risk Factors), following the general approach in the disclosure-change
literature (e.g. Cohen, Malloy & Nguyen, "Lazy Prices," 2020): pull
consecutive 10-Ks for a firm, extract the risk-factors section from each,
and measure how much that section changed from one year to the next.

Pipeline:
    1. EdgarClient resolves ticker -> CIK and pulls each firm's 10-K
       filing index from SEC's submissions API (paginating through older
       filings for high-volume filers).
    2. extract_item_1a_detailed() pulls the Item 1A section out of each
       filing's raw HTML via anchor-pattern matching, and tags the result
       with a confidence level rather than a silent pass/fail, since
       regex-based section extraction on inconsistently formatted HTML
       filings is not going to be 100% reliable.
    3. build_panel() assembles firm-year pairs, computes TF-IDF cosine
       similarity and Jaccard token overlap between consecutive years, and
       flags rows with low-confidence extractions or structurally
       implausible swings so they can be reviewed rather than trusted
       blindly.
    4. validate_against_forward_returns() is an optional, secondary sanity
       check: it measures each firm's stock return (raw and
       benchmark-adjusted) over a fixed trading-day window starting the
       day after the actual filing date, for an informal look at whether
       disclosure change correlates with subsequent stock performance.

Known limitations (see README.md for the full list): section extraction
is regex/anchor-based and not a substitute for manually verifying flagged
rows; the return validation is a directional sanity check, not a
risk-adjusted asset-pricing test, and the firm counts used here are far
too small to support a statistically meaningful correlation on their own.
"""

import hashlib
import html
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_FIRM_UNIVERSE = ["AAPL", "MSFT", "BA", "WMT", "JPM", "XOM"]

logger = logging.getLogger("edgar_diff")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class FilingMetadata:
    ticker: str
    fiscal_year: int
    filing_date: str
    accession_number: str
    document_url: str


@dataclass
class RunStats:
    """Tracks what happened during a build_panel run, so failures are visible
    instead of silently swallowed."""
    tickers_requested: int = 0
    tickers_skipped_no_cik: List[str] = field(default_factory=list)
    tickers_skipped_fetch_error: List[Tuple[str, str]] = field(default_factory=list)
    filings_extraction_failed: List[Tuple[str, int, str]] = field(default_factory=list)
    filings_download_failed: List[Tuple[str, int, str]] = field(default_factory=list)
    firm_years_extracted: int = 0
    firm_years_low_confidence: List[Tuple[str, int, str]] = field(default_factory=list)
    panel_rows_built: int = 0
    panel_rows_flagged: int = 0
    # Per-ticker funnel: cik found -> filings found -> downloaded ->
    # extracted (high/low confidence) -> failed. Makes it possible to see
    # exactly which stage a ticker dropped out at, instead of it just
    # being absent from the final panel with no trace of why.
    ticker_diagnostics: Dict[str, dict] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "tickers_requested": self.tickers_requested,
            "tickers_skipped_no_cik": self.tickers_skipped_no_cik,
            "tickers_skipped_fetch_error": self.tickers_skipped_fetch_error,
            "filings_extraction_failed": self.filings_extraction_failed,
            "filings_download_failed": self.filings_download_failed,
            "firm_years_extracted": self.firm_years_extracted,
            "firm_years_low_confidence": self.firm_years_low_confidence,
            "panel_rows_built": self.panel_rows_built,
            "panel_rows_flagged": self.panel_rows_flagged,
            "ticker_diagnostics": self.ticker_diagnostics,
        }


def fail_fast_user_agent(user_agent: str) -> None:
    """Validate that the user agent is a genuine contact string."""
    if not user_agent or "@" not in user_agent or " " not in user_agent.strip():
        raise ValueError(
            "SEC requires a valid contact string (Name + email). "
            f"Received: '{user_agent}'. Example: 'Jane Doe jdoe@university.edu'"
        )


class EdgarClient:
    def __init__(
        self,
        user_agent: str,
        rate_limit_sleep: float = 0.15,
        use_cache: bool = True,
        cache_dir: str = ".edgar_cache",
    ):
        fail_fast_user_agent(user_agent)
        self.headers = {"User-Agent": user_agent}
        self.rate_limit_sleep = rate_limit_sleep
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, url: str) -> str:
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{key}.cache")

    def get(self, url: str) -> requests.Response:
        time.sleep(self.rate_limit_sleep)
        response = self.session.get(url)
        if response.status_code == 403:
            raise PermissionError(
                f"SEC returned 403 Forbidden for {url}. Check your User-Agent string."
            )
        response.raise_for_status()
        return response

    def get_text(self, url: str) -> str:
        """Get response body as text, transparently reading from and writing
        to an on-disk cache. SEC filings and submission indexes do not change
        once filed, so caching avoids re-hitting SEC's servers on every rerun
        and makes local testing and demos much faster and more reliable."""
        if self.use_cache:
            cache_path = self._cache_path(url)
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    return f.read()

        text = self.get(url).text

        if self.use_cache:
            cache_path = self._cache_path(url)
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(text)

        return text

    def fetch_company_tickers(self) -> Dict[str, str]:
        url = "https://www.sec.gov/files/company_tickers.json"
        data = json.loads(self.get_text(url))
        return parse_company_tickers(data)

    def fetch_filing_index(self, ticker: str, cik: str) -> List[FilingMetadata]:
        cik_padded = str(cik).zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        data = json.loads(self.get_text(url))

        # SEC's submissions JSON is a set of parallel arrays (form[i],
        # accessionNumber[i], filingDate[i], ... all describe filing i).
        # Older paginated pages sometimes omit a field (usually reportDate)
        # for just that page, so each page's arrays must be padded to that
        # page's own length *before* concatenating pages together. Padding
        # only once at the very end (on the concatenated arrays) does not
        # fix this: if page 2 is short by one reportDate entry, every
        # filing from page 2 onward silently shifts by one position and
        # gets the wrong report date, and therefore the wrong fiscal year.
        def _extract_page(page: dict) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
            forms = list(page.get("form", []) or [])
            n = len(forms)

            def _padded(key: str) -> List[str]:
                vals = list(page.get(key, []) or [])
                if len(vals) < n:
                    vals = vals + [""] * (n - len(vals))
                return vals[:n]

            return (
                forms,
                _padded("accessionNumber"),
                _padded("filingDate"),
                _padded("primaryDocument"),
                _padded("reportDate"),
            )

        recent = data.get("filings", {}).get("recent", {})
        all_forms, all_accessions, all_filing_dates, all_primary_docs, all_report_dates = _extract_page(recent)

        # SEC submissions API pagination for high-volume filers (JPM, XOM, etc.)
        older_files = data.get("filings", {}).get("files", [])
        for file_info in older_files:
            filename = file_info.get("name")
            if not filename:
                continue
            older_url = f"https://data.sec.gov/submissions/{filename}"
            try:
                older_data = json.loads(self.get_text(older_url))
                p_forms, p_acc, p_fdate, p_pdoc, p_rdate = _extract_page(older_data)
                all_forms.extend(p_forms)
                all_accessions.extend(p_acc)
                all_filing_dates.extend(p_fdate)
                all_primary_docs.extend(p_pdoc)
                all_report_dates.extend(p_rdate)
            except Exception as e:
                logger.warning(
                    "Failed to fetch older submissions page %s for %s: %s",
                    older_url, ticker, e,
                )

        filings = []
        for form, acc, fdate, pdoc, rdate in zip(
            all_forms, all_accessions, all_filing_dates, all_primary_docs, all_report_dates
        ):
            # Exact match only. "10-K".startswith would also match 10-K/A and
            # 10-K405 style amendments/variants, which often only restate
            # Part III and don't carry a full Item 1A. If you later want to
            # accept NT 10-K or 10-K405 explicitly, add them here rather than
            # loosening back to startswith.
            form_clean = (form or "").strip()
            if form_clean != "10-K":
                continue

            fy = None
            if rdate:
                try:
                    fy = int(rdate.split("-")[0])
                except Exception:
                    fy = None
            if fy is None:
                try:
                    fy = int(fdate.split("-")[0]) - 1
                except Exception:
                    continue

            if not pdoc:
                logger.warning(
                    "Skipping %s filing dated %s: no primaryDocument in submissions JSON.",
                    ticker, fdate,
                )
                continue

            acc_no_hyphen = acc.replace("-", "")
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no_hyphen}/{pdoc}"
            filings.append(FilingMetadata(
                ticker=ticker,
                fiscal_year=fy,
                filing_date=fdate,
                accession_number=acc,
                document_url=doc_url
            ))

        seen_years = {}
        for f in sorted(filings, key=lambda x: x.filing_date):
            seen_years[f.fiscal_year] = f

        logger.info("Fetched %d valid 10-K filings for ticker %s", len(seen_years), ticker)
        return list(seen_years.values())

    def download_filing_text(self, url: str) -> str:
        return self.get_text(url)


def parse_company_tickers(data: dict) -> Dict[str, str]:
    mapping = {}
    for entry in data.values():
        ticker = str(entry.get("ticker", "")).upper()
        cik = str(entry.get("cik_str", ""))
        if ticker and cik:
            mapping[ticker] = cik
    return mapping


@dataclass
class ExtractionResult:
    text: Optional[str]
    confidence: str   # "high", "low", "none"
    method: str        # "matched", "matched_short", "fallback", "none"
    char_len: int


# Real Item 1A sections run many pages of prose. A "match" shorter than this
# is treated as unconfirmed rather than accepted outright — this is the bar
# that catches the MSFT-style bug where a compact TOC/index row produced a
# small, tight, technically-valid gap.
MIN_CONFIDENT_LEN = 3000
MIN_ACCEPTABLE_LEN = 500
MAX_SECTION_LEN = 400000
_TOC_DENSITY_WINDOW = 1000
_TOC_DENSITY_MAX_HITS = 2


def _looks_like_toc(window_text: str) -> bool:
    """A real Item 1A section is prose about risks — it rarely mentions
    'Item N' more than once or twice. A compact TOC or index listing (e.g.
    'Item 1A ... Item 1B ... Item 2 ...' packed into a few hundred
    characters) mentions several in quick succession. Check only the head
    of the window since TOC-style clustering, if present, shows up
    immediately at the start of a false-positive match."""
    head = window_text[:2000]
    hits = len(re.findall(r'item\s*\d+[ab]?\b', head, re.IGNORECASE))
    density = hits / max(len(head) / _TOC_DENSITY_WINDOW, 1e-6)
    return density > _TOC_DENSITY_MAX_HITS


def _find_best_gap(
    clean_text: str, start_positions: List[int], doc_len: int, min_len: int, require_toc_guard: bool
) -> Tuple[Optional[int], Optional[int]]:
    end_patterns = [
        r'item\s*1b\.?\s*unresolved',
        r'item\s*2\.?\s*properties'
    ]
    best_start, best_end, best_gap = None, None, None

    for pos in start_positions:
        if len(start_positions) > 3 and pos < doc_len * 0.20:
            continue

        end_pos = None
        for pattern in end_patterns:
            matches = list(re.finditer(pattern, clean_text[pos:], re.IGNORECASE))
            if matches:
                candidate_end = pos + matches[0].start()
                if end_pos is None or candidate_end < end_pos:
                    end_pos = candidate_end

        if end_pos is None:
            continue

        gap = end_pos - pos
        if not (min_len < gap < MAX_SECTION_LEN):
            continue

        if require_toc_guard and _looks_like_toc(clean_text[pos:end_pos]):
            continue

        if best_gap is None or gap < best_gap:
            best_start, best_end, best_gap = pos, end_pos, gap

    return best_start, best_end


def extract_item_1a_detailed(html_content: str) -> ExtractionResult:
    """Extract the Item 1A (Risk Factors) section from a 10-K, tagged with
    a confidence level instead of a binary hit/miss.

    Three tiers, tried in order:
    1. "high" — passes the TOC-density guard and clears MIN_CONFIDENT_LEN
       (thousands of chars). This is what a real Item 1A section looks like.
    2. "low"/"matched_short" — same anchors, but the density guard is
       dropped or the length only clears MIN_ACCEPTABLE_LEN. Structurally
       plausible but short enough to warrant a second look before trusting
       it in an analysis.
    3. "low"/"fallback" — no confirmed end anchor at all; slice forward
       from the last start match. Least reliable path in the function.
    """
    clean_text = html.unescape(html_content)
    clean_text = re.sub(r'<[^>]+>', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    doc_len = len(clean_text)

    start_patterns = [
        r'item\s*1a\.?\s*risk\s*factors',
        r'item\s*1a\b'
    ]
    start_positions = []
    for pattern in start_patterns:
        for m in re.finditer(pattern, clean_text, re.IGNORECASE):
            start_positions.append(m.start())
    start_positions = sorted(set(start_positions))

    if not start_positions:
        return ExtractionResult(None, "none", "none", 0)

    # Tier 1: confident match — long enough, and not TOC-shaped.
    start, end = _find_best_gap(clean_text, start_positions, doc_len, MIN_CONFIDENT_LEN, require_toc_guard=True)
    if start is not None:
        extracted = clean_text[start:end].strip()
        return ExtractionResult(extracted, "high", "matched", len(extracted))

    # Tier 2: relax the length bar, keep the TOC guard. Catches legitimately
    # shorter risk-factor sections without accepting index-row noise.
    start, end = _find_best_gap(clean_text, start_positions, doc_len, MIN_ACCEPTABLE_LEN, require_toc_guard=True)
    if start is not None:
        extracted = clean_text[start:end].strip()
        return ExtractionResult(extracted, "low", "matched_short", len(extracted))

    # Tier 3: no confirmed end anchor. Slice forward from the last start
    # match — this is a guess, and is always tagged low confidence.
    pos = start_positions[-1]
    end = min(pos + MAX_SECTION_LEN, doc_len)
    extracted = clean_text[pos:end].strip()
    if len(extracted) < MIN_ACCEPTABLE_LEN:
        return ExtractionResult(None, "none", "none", 0)
    return ExtractionResult(extracted, "low", "fallback", len(extracted))


def extract_item_1a(html_content: str) -> Optional[str]:
    """Backward-compatible wrapper — text only, no confidence info. Prefer
    extract_item_1a_detailed for anything that consumes the confidence tag."""
    return extract_item_1a_detailed(html_content).text


_WORD_RE = re.compile(r"[a-zA-Z]{2,}")


def _tokenize_for_jaccard(text: str, vectorizer: TfidfVectorizer) -> set:
    """Same lowercasing + punctuation stripping + stopword removal as the
    TfidfVectorizer, so cosine and Jaccard are computed over the same
    vocabulary. Previously Jaccard used raw .split() (no stopword removal,
    "risk," != "risk"), so the two metrics disagreed about what counted as
    a token change."""
    stop_words = vectorizer.get_stop_words() or set()
    tokens = _WORD_RE.findall(text.lower())
    return {t for t in tokens if t not in stop_words}


def year_over_year_similarity(
    prev_vec, curr_vec, prev_text: str, curr_text: str, vectorizer: TfidfVectorizer
) -> dict:
    """Compute change metrics between two firm-year documents."""
    sim = cosine_similarity(prev_vec, curr_vec)[0][0]

    prev_tokens = _tokenize_for_jaccard(prev_text, vectorizer)
    curr_tokens = _tokenize_for_jaccard(curr_text, vectorizer)
    jaccard = len(prev_tokens.intersection(curr_tokens)) / max(len(prev_tokens.union(curr_tokens)), 1)

    return {
        "cosine_similarity": float(sim),
        "jaccard_similarity": float(jaccard),
        "change_score": float(1.0 - sim)
    }


def build_panel(
    client: EdgarClient, tickers: List[str], n_filings: int = 6
) -> Tuple[pd.DataFrame, RunStats]:
    stats = RunStats(tickers_requested=len(tickers))

    ticker_to_cik = client.fetch_company_tickers()

    extracted_texts: Dict[Tuple[str, int], str] = {}
    extraction_confidence: Dict[Tuple[str, int], str] = {}
    filing_dates: Dict[Tuple[str, int], str] = {}

    for ticker in tickers:
        diag = {
            "cik_found": False,
            "filings_found": 0,
            "filings_attempted": 0,
            "filings_downloaded": 0,
            "filings_extracted_high": 0,
            "filings_extracted_low": 0,
            "filings_failed": 0,
        }
        stats.ticker_diagnostics[ticker] = diag

        cik = ticker_to_cik.get(ticker)
        if not cik:
            stats.tickers_skipped_no_cik.append(ticker)
            logger.warning("No CIK found for ticker %s, skipping.", ticker)
            continue
        diag["cik_found"] = True

        try:
            filings = client.fetch_filing_index(ticker, cik)
        except Exception as e:
            stats.tickers_skipped_fetch_error.append((ticker, str(e)))
            logger.warning("Failed to fetch filing index for %s: %s", ticker, e)
            continue
        diag["filings_found"] = len(filings)

        filings = sorted(filings, key=lambda x: x.fiscal_year)[-n_filings:]
        diag["filings_attempted"] = len(filings)

        for f in filings:
            try:
                html_data = client.download_filing_text(f.document_url)
            except Exception as e:
                stats.filings_download_failed.append((ticker, f.fiscal_year, str(e)))
                diag["filings_failed"] += 1
                logger.warning(
                    "Failed to download filing for %s FY%s: %s", ticker, f.fiscal_year, e
                )
                continue
            diag["filings_downloaded"] += 1

            result = extract_item_1a_detailed(html_data)
            if result.text:
                extracted_texts[(ticker, f.fiscal_year)] = result.text
                extraction_confidence[(ticker, f.fiscal_year)] = result.confidence
                filing_dates[(ticker, f.fiscal_year)] = f.filing_date
                stats.firm_years_extracted += 1
                if result.confidence == "high":
                    diag["filings_extracted_high"] += 1
                else:
                    diag["filings_extracted_low"] += 1
                    stats.firm_years_low_confidence.append(
                        (ticker, f.fiscal_year, f"method={result.method}, len={result.char_len}")
                    )
                    logger.warning(
                        "Low-confidence Item 1A extraction for %s FY%s (method=%s, len=%d).",
                        ticker, f.fiscal_year, result.method, result.char_len,
                    )
            else:
                diag["filings_failed"] += 1
                stats.filings_extraction_failed.append(
                    (ticker, f.fiscal_year, "Item 1A not found or too short")
                )
                logger.warning(
                    "Could not extract Item 1A for %s FY%s.", ticker, f.fiscal_year
                )

    if not extracted_texts:
        return pd.DataFrame(), stats

    # Second-pass sanity check: even a "high" confidence extraction can be
    # an outlier if it's much shorter than that same firm's other filings —
    # a firm's Item 1A section doesn't swing from 40 pages to 2 pages year
    # over year in practice, so a big length drop is itself a signal worth
    # flagging even when the extractor was structurally confident.
    keys_by_ticker: Dict[str, List[Tuple[str, int]]] = {}
    for key in extracted_texts.keys():
        keys_by_ticker.setdefault(key[0], []).append(key)

    for t, ticker_keys in keys_by_ticker.items():
        lens = [len(extracted_texts[k]) for k in ticker_keys]
        if len(lens) < 2:
            continue
        median_len = sorted(lens)[len(lens) // 2]
        if median_len <= 0:
            continue
        for key in ticker_keys:
            text_len = len(extracted_texts[key])
            if text_len < 0.25 * median_len and extraction_confidence.get(key) == "high":
                extraction_confidence[key] = "low"
                stats.firm_years_low_confidence.append(
                    (key[0], key[1], f"length_outlier: {text_len} chars vs firm median {median_len}")
                )
                logger.warning(
                    "Flagging %s FY%s as low confidence: extracted length %d is far below "
                    "this firm's median of %d.", key[0], key[1], text_len, median_len,
                )

    keys = list(extracted_texts.keys())
    corpus = [extracted_texts[k] for k in keys]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    vectors = {k: tfidf_matrix[i] for i, k in enumerate(keys)}

    panel_rows = []
    for ticker in tickers:
        firm_years = sorted(y for (t, y) in extracted_texts.keys() if t == ticker)
        for i in range(1, len(firm_years)):
            prev_yr, curr_yr = firm_years[i - 1], firm_years[i]
            if curr_yr - prev_yr != 1:
                continue

            metrics = year_over_year_similarity(
                vectors[(ticker, prev_yr)],
                vectors[(ticker, curr_yr)],
                extracted_texts[(ticker, prev_yr)],
                extracted_texts[(ticker, curr_yr)],
                vectorizer,
            )

            prev_conf = extraction_confidence.get((ticker, prev_yr), "high")
            curr_conf = extraction_confidence.get((ticker, curr_yr), "high")
            # Belt-and-suspenders: even if both extractions were confident,
            # an extreme, structurally implausible swing (near-total token
            # turnover year over year) is worth a second look before it's
            # cited as a finding rather than discovered by trial and error.
            extreme_swing = metrics["change_score"] > 0.5 and metrics["jaccard_similarity"] < 0.2
            flagged = (prev_conf != "high") or (curr_conf != "high") or extreme_swing

            flag_reasons = []
            if prev_conf != "high":
                flag_reasons.append(f"FY{prev_yr} extraction: {prev_conf} confidence")
            if curr_conf != "high":
                flag_reasons.append(f"FY{curr_yr} extraction: {curr_conf} confidence")
            if extreme_swing:
                flag_reasons.append("extreme year-over-year swing — verify extracted text manually")

            panel_rows.append({
                "ticker": ticker,
                "fiscal_year": curr_yr,
                "filing_date": filing_dates.get((ticker, curr_yr)),
                "cosine_similarity": metrics["cosine_similarity"],
                "jaccard_similarity": metrics["jaccard_similarity"],
                "change_score": metrics["change_score"],
                "flagged": flagged,
                "flag_reason": "; ".join(flag_reasons),
            })

    stats.panel_rows_built = len(panel_rows)
    stats.panel_rows_flagged = sum(1 for r in panel_rows if r["flagged"])
    return pd.DataFrame(panel_rows), stats


def _trading_day_return(
    history: pd.DataFrame, event_date: pd.Timestamp, window_trading_days: int
) -> Optional[Tuple[float, str, str]]:
    """Return (pct_return, window_start_date, window_end_date) for holding
    from the first trading day strictly after `event_date` through
    `window_trading_days` trading days later, using `history` (a
    DatetimeIndex-ed frame with a 'Close' column).

    Returns None if there isn't a full window available (e.g. the filing
    is too recent, or the ticker has no data in range) rather than
    silently truncating the window, since a truncated window is not
    comparable across rows.
    """
    if history.empty or pd.isna(event_date):
        return None

    idx = history.index
    start_pos = idx.searchsorted(event_date, side="right")
    if start_pos >= len(idx):
        return None
    end_pos = start_pos + window_trading_days
    if end_pos >= len(idx):
        return None

    start_price = history["Close"].iloc[start_pos]
    end_price = history["Close"].iloc[end_pos]
    if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
        return None

    ret = (end_price - start_price) / start_price
    return float(ret), str(idx[start_pos].date()), str(idx[end_pos].date())


def validate_against_forward_returns(
    panel_df: pd.DataFrame,
    window_trading_days: int = 21,
    benchmark_ticker: str = "SPY",
) -> pd.DataFrame:
    """Event-study-style validation of the disclosure-change panel.

    For each panel row, measures the stock's raw return and its return in
    excess of a market benchmark over a fixed trading-day window starting
    the day after the 10-K was actually filed (`filing_date`), then leaves
    it to the caller to correlate that against `change_score`.

    This intentionally does NOT use a calendar window like "March through
    June of the fiscal year" — that has no necessary relationship to when
    the filing actually happened, so a correlation computed that way could
    just reflect market-wide moves that coincide with filing season rather
    than any reaction to the filing itself. Subtracting the benchmark
    return over the identical window also means the correlation isn't
    purely picking up market beta.

    Still not a rigorous asset-pricing test: no size/value/momentum risk
    adjustment beyond the single benchmark subtraction, no handling of
    delisted or acquired firms, and the firm counts used in this app are
    far too small for the resulting correlation to be statistically
    meaningful on their own — treat it as a directional sanity check, not
    a result to report on its own.

    Parameters
    ----------
    panel_df : output of build_panel(); must contain 'filing_date'.
    window_trading_days : holding period length, in trading days, starting
        the day after filing_date.
    benchmark_ticker : ticker used as the market benchmark for excess
        returns (default SPY).
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance is not installed; skipping forward-return validation.")
        return pd.DataFrame()

    if panel_df.empty:
        return pd.DataFrame()
    if "filing_date" not in panel_df.columns:
        raise ValueError(
            "panel_df has no 'filing_date' column. Rebuild the panel with "
            "the current build_panel(), which carries filing_date through "
            "for each row so returns can be aligned to the real event date."
        )

    df = panel_df.copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df = df.dropna(subset=["filing_date"])
    if df.empty:
        return pd.DataFrame()

    # Trading days skip weekends/holidays, so a `window_trading_days`-day
    # hold needs noticeably more than that many calendar days of buffer.
    calendar_buffer = pd.Timedelta(days=window_trading_days * 2 + 15)
    global_start = df["filing_date"].min() - pd.Timedelta(days=5)
    global_end = df["filing_date"].max() + calendar_buffer

    try:
        bench_hist = yf.Ticker(benchmark_ticker).history(start=global_start, end=global_end)
        if not bench_hist.empty:
            bench_hist.index = bench_hist.index.tz_localize(None)
    except Exception as e:
        logger.warning("Failed to fetch benchmark %s: %s", benchmark_ticker, e)
        bench_hist = pd.DataFrame()

    validated_rows = []
    # Fetch each ticker's price history once (covering every filing_date it
    # needs), not once per row — far fewer yfinance calls and less exposure
    # to rate limiting than the previous per-row fetch.
    for ticker, group in df.groupby("ticker"):
        t_start = group["filing_date"].min() - pd.Timedelta(days=5)
        t_end = group["filing_date"].max() + calendar_buffer
        try:
            stock_hist = yf.Ticker(ticker).history(start=t_start, end=t_end)
        except Exception as e:
            logger.warning("Forward-return fetch failed for %s: %s", ticker, e)
            continue
        if stock_hist.empty:
            continue
        stock_hist.index = stock_hist.index.tz_localize(None)

        for _, row in group.iterrows():
            event_date = row["filing_date"]
            stock_result = _trading_day_return(stock_hist, event_date, window_trading_days)
            if stock_result is None:
                continue
            stock_ret, window_start, window_end = stock_result

            bench_ret = None
            if not bench_hist.empty:
                bench_result = _trading_day_return(bench_hist, event_date, window_trading_days)
                if bench_result is not None:
                    bench_ret = bench_result[0]
            excess_ret = (stock_ret - bench_ret) if bench_ret is not None else None

            row_dict = row.to_dict()
            row_dict["filing_date"] = str(event_date.date())
            row_dict["forward_return"] = stock_ret
            row_dict["benchmark_return"] = bench_ret
            row_dict["excess_return"] = excess_ret
            row_dict["window_start"] = window_start
            row_dict["window_end"] = window_end
            validated_rows.append(row_dict)

    return pd.DataFrame(validated_rows)

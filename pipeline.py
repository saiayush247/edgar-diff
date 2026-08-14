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
    panel_rows_built: int = 0

    def as_dict(self) -> dict:
        return {
            "tickers_requested": self.tickers_requested,
            "tickers_skipped_no_cik": self.tickers_skipped_no_cik,
            "tickers_skipped_fetch_error": self.tickers_skipped_fetch_error,
            "filings_extraction_failed": self.filings_extraction_failed,
            "filings_download_failed": self.filings_download_failed,
            "firm_years_extracted": self.firm_years_extracted,
            "panel_rows_built": self.panel_rows_built,
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

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])

        all_forms = list(forms)
        all_accessions = list(accessions)
        all_filing_dates = list(filing_dates)
        all_primary_docs = list(primary_docs)

        # SEC submissions API pagination for high-volume filers (JPM, XOM, etc.)
        older_files = data.get("filings", {}).get("files", [])
        for file_info in older_files:
            filename = file_info.get("name")
            if not filename:
                continue
            older_url = f"https://data.sec.gov/submissions/{filename}"
            try:
                older_data = json.loads(self.get_text(older_url))
                all_forms.extend(older_data.get("form", []))
                all_accessions.extend(older_data.get("accessionNumber", []))
                all_filing_dates.extend(older_data.get("filingDate", []))
                all_primary_docs.extend(older_data.get("primaryDocument", []))
            except Exception as e:
                logger.warning(
                    "Failed to fetch older submissions page %s for %s: %s",
                    older_url, ticker, e,
                )

        filings = []
        for form, acc, fdate, pdoc in zip(all_forms, all_accessions, all_filing_dates, all_primary_docs):
            if not form.startswith("10-K"):
                continue
            try:
                fy = int(fdate.split("-")[0]) - 1
            except Exception:
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


def extract_item_1a(html_content: str) -> Optional[str]:
    """Extract the Item 1A (Risk Factors) section from a 10-K.

    Handles HTML entities, table of contents filtering for large filers like MSFT,
    and robust gap heuristics to prevent false positives.
    """
    clean_text = html.unescape(html_content)
    clean_text = re.sub(r'<[^>]+>', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)

    start_patterns = [
        r'item\s*1a\.?\s*risk\s*factors',
        r'item\s*1a\b'
    ]
    end_patterns = [
        r'item\s*1b\.?\s*unresolved',
        r'item\s*2\.?\s*properties'
    ]

    start_positions = []
    for pattern in start_patterns:
        for m in re.finditer(pattern, clean_text, re.IGNORECASE):
            start_positions.append(m.start())

    if not start_positions:
        return None

    MAX_SECTION_LEN = 150000
    best_start, best_end, best_gap = None, None, -1
    doc_len = len(clean_text)

    for pos in sorted(set(start_positions)):
        # Skip early occurrences if they are clustered in the Table of Contents preamble
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
        if 500 < gap < MAX_SECTION_LEN and gap > best_gap:
            best_start, best_end, best_gap = pos, end_pos, gap

    # Fallback if strict filtering dropped everything
    if best_start is None and start_positions:
        pos = start_positions[-1]
        best_start = pos
        best_end = min(pos + MAX_SECTION_LEN, doc_len)

    if best_start is None or best_end is None:
        return None

    extracted = clean_text[best_start:best_end].strip()
    if len(extracted) < 500:
        return None

    return extracted


def year_over_year_similarity(prev_vec, curr_vec, prev_text: str, curr_text: str) -> dict:
    """Compute change metrics between two firm-year documents."""
    sim = cosine_similarity(prev_vec, curr_vec)[0][0]

    prev_tokens = set(prev_text.lower().split())
    curr_tokens = set(curr_text.lower().split())
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

    for ticker in tickers:
        cik = ticker_to_cik.get(ticker)
        if not cik:
            stats.tickers_skipped_no_cik.append(ticker)
            logger.warning("No CIK found for ticker %s, skipping.", ticker)
            continue

        try:
            filings = client.fetch_filing_index(ticker, cik)
        except Exception as e:
            stats.tickers_skipped_fetch_error.append((ticker, str(e)))
            logger.warning("Failed to fetch filing index for %s: %s", ticker, e)
            continue

        filings = sorted(filings, key=lambda x: x.fiscal_year)[-n_filings:]

        for f in filings:
            try:
                html_data = client.download_filing_text(f.document_url)
            except Exception as e:
                stats.filings_download_failed.append((ticker, f.fiscal_year, str(e)))
                logger.warning(
                    "Failed to download filing for %s FY%s: %s", ticker, f.fiscal_year, e
                )
                continue

            text = extract_item_1a(html_data)
            if text:
                extracted_texts[(ticker, f.fiscal_year)] = text
                stats.firm_years_extracted += 1
            else:
                stats.filings_extraction_failed.append(
                    (ticker, f.fiscal_year, "Item 1A not found or too short")
                )
                logger.warning(
                    "Could not extract Item 1A for %s FY%s.", ticker, f.fiscal_year
                )

    if not extracted_texts:
        return pd.DataFrame(), stats

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
            )
            panel_rows.append({
                "ticker": ticker,
                "fiscal_year": curr_yr,
                "cosine_similarity": metrics["cosine_similarity"],
                "jaccard_similarity": metrics["jaccard_similarity"],
                "change_score": metrics["change_score"]
            })

    stats.panel_rows_built = len(panel_rows)
    return pd.DataFrame(panel_rows), stats


def validate_against_forward_returns(panel_df: pd.DataFrame) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame()

    validated_rows = []
    for _, row in panel_df.iterrows():
        ticker = row["ticker"]
        fy = int(row["fiscal_year"])

        try:
            stock = yf.Ticker(ticker)
            start_date = f"{fy}-03-01"
            end_date = f"{fy}-06-01"
            hist = stock.history(start=start_date, end=end_date)
            if not hist.empty:
                ret = (hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]
                row_dict = row.to_dict()
                row_dict["forward_return"] = float(ret)
                validated_rows.append(row_dict)
        except Exception as e:
            logger.warning("Forward-return fetch failed for %s FY%s: %s", ticker, fy, e)
            continue

    return pd.DataFrame(validated_rows)

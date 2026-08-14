import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_FIRM_UNIVERSE = ["AAPL", "MSFT", "BA", "WMT", "JPM", "XOM"]


@dataclass
class FilingMetadata:
    ticker: str
    fiscal_year: int
    filing_date: str
    accession_number: str
    document_url: str


def fail_fast_user_agent(user_agent: str) -> None:
    """Validate that the user agent is a genuine contact string."""
    if not user_agent or "@" not in user_agent or " " not in user_agent.strip():
        raise ValueError(
            "SEC requires a valid contact string (Name + email). "
            f"Received: '{user_agent}'. Example: 'Jane Doe jdoe@university.edu'"
        )


class EdgarClient:
    def __init__(self, user_agent: str, rate_limit_sleep: float = 0.1):
        fail_fast_user_agent(user_agent)
        self.headers = {"User-Agent": user_agent}
        self.rate_limit_sleep = rate_limit_sleep
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get(self, url: str) -> requests.Response:
        time.sleep(self.rate_limit_sleep)
        response = self.session.get(url)
        if response.status_code == 403:
            raise PermissionError(
                f"SEC returned 403 Forbidden for {url}. Check your User-Agent string."
            )
        response.raise_for_status()
        return response

    def fetch_company_tickers(self) -> Dict[str, str]:
        url = "https://www.sec.gov/files/company_tickers.json"
        data = self.get(url).json()
        return parse_company_tickers(data)

    def fetch_filing_index(self, ticker: str, cik: str) -> List[FilingMetadata]:
        cik_padded = str(cik).zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        data = self.get(url).json()
        
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
            if filename:
                older_url = f"https://data.sec.gov/submissions/{filename}"
                try:
                    older_data = self.get(older_url).json()
                    all_forms.extend(older_data.get("form", []))
                    all_accessions.extend(older_data.get("accessionNumber", []))
                    all_filing_dates.extend(older_data.get("filingDate", []))
                    all_primary_docs.extend(older_data.get("primaryDocument", []))
                except Exception:
                    pass

        filings = []
        for form, acc, fdate, pdoc in zip(all_forms, all_accessions, all_filing_dates, all_primary_docs):
            if form == "10-K":
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
            
        return list(seen_years.values())

    def download_filing_text(self, url: str) -> str:
        res = self.get(url)
        return res.text


def parse_company_tickers(data: dict) -> Dict[str, str]:
    mapping = {}
    for entry in data.values():
        ticker = str(entry.get("ticker", "")).upper()
        cik = str(entry.get("cik_str", ""))
        if ticker and cik:
            mapping[ticker] = cik
    return mapping


def extract_item_1a(html_content: str) -> Optional[str]:
    clean_text = re.sub(r'<[^>]+>', ' ', html_content)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    start_patterns = [
        r'item\s*1a\.?\s*risk\s*factors',
        r'item\s*1a\b'
    ]
    
    start_pos = -1
    for pattern in start_patterns:
        matches = [m.start() for m in re.finditer(pattern, clean_text, re.IGNORECASE)]
        for pos in matches:
            snippet = clean_text[pos:pos+100]
            if re.search(r'\b(business|operations|conditions|factors|table|contents)\b', snippet, re.IGNORECASE):
                start_pos = pos
                break
        if start_pos != -1:
            break
            
    if start_pos == -1:
        return None
        
    end_patterns = [
        r'item\s*1b\.?\s*unresolved',
        r'item\s*2\.?\s*properties'
    ]
    
    end_pos = -1
    for pattern in end_patterns:
        matches = [m.start() for m in re.finditer(pattern, clean_text[start_pos:], re.IGNORECASE)]
        if matches:
            end_pos = start_pos + matches[0]
            break
            
    if end_pos == -1 or end_pos <= start_pos + 500:
        end_pos = start_pos + 150000
        
    extracted = clean_text[start_pos:end_pos].strip()
    if len(extracted) < 500:
        return None
        
    return extracted


def year_over_year_similarity(prev_text: str, curr_text: str) -> dict:
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    tfidf = vectorizer.fit_transform([prev_text, curr_text])
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    
    prev_tokens = set(prev_text.lower().split())
    curr_tokens = set(curr_text.lower().split())
    jaccard = len(prev_tokens.intersection(curr_tokens)) / max(len(prev_tokens.union(curr_tokens)), 1)
    
    return {
        "cosine_similarity": float(sim),
        "jaccard_similarity": float(jaccard),
        "change_score": float(1.0 - sim)
    }


def build_panel(client: EdgarClient, tickers: List[str], n_filings: int = 6) -> pd.DataFrame:
    ticker_to_cik = client.fetch_company_tickers()
    panel_rows = []
    
    for ticker in tickers:
        cik = ticker_to_cik.get(ticker)
        if not cik:
            continue
            
        try:
            filings = client.fetch_filing_index(ticker, cik)
        except Exception:
            continue
            
        filings = sorted(filings, key=lambda x: x.fiscal_year)[-n_filings:]
        
        extracted_texts = {}
        for f in filings:
            try:
                html = client.download_filing_text(f.document_url)
                text = extract_item_1a(html)
                if text:
                    extracted_texts[f.fiscal_year] = text
            except Exception:
                continue
                
        sorted_years = sorted(extracted_texts.keys())
        for i in range(1, len(sorted_years)):
            prev_yr = sorted_years[i-1]
            curr_yr = sorted_years[i]
            
            if curr_yr - prev_yr == 1:
                metrics = year_over_year_similarity(extracted_texts[prev_yr], extracted_texts[curr_yr])
                panel_rows.append({
                    "ticker": ticker,
                    "fiscal_year": curr_yr,
                    "cosine_similarity": metrics["cosine_similarity"],
                    "jaccard_similarity": metrics["jaccard_similarity"],
                    "change_score": metrics["change_score"]
                })
                
    return pd.DataFrame(panel_rows)


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
        except Exception:
            continue
            
    return pd.DataFrame(validated_rows)
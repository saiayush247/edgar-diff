# edgar-diff

Builds a cross-sectional firm-year panel of year-over-year textual change in
the Item 1A (Risk Factors) section of 10-K filings, pulled directly from
SEC EDGAR.

## Methodology

For each ticker, the pipeline:

1. Resolves the ticker to a CIK via SEC's `company_tickers.json`.
2. Pulls the firm's filing history from `data.sec.gov/submissions`, including
   paginated older-filings indexes for high-volume filers.
3. Downloads each 10-K's primary document and extracts the Item 1A section.
4. Fits a single TF-IDF vectorizer across the full corpus of extracted
   sections for the run, then computes, for each firm's consecutive filing
   years, cosine similarity and Jaccard similarity between the prior and
   current year's Item 1A text.
5. Reports `change_score = 1 - cosine_similarity` as the primary measure of
   year-over-year disclosure change, following the general approach in the
   disclosure-change literature (e.g. Cohen, Malloy & Nguyen 2020, who show
   textual change in 10-K filings predicts returns and future firm outcomes).

An optional validation step correlates `change_score` against a simple
forward stock return window, pulled via `yfinance`.

## Known limitations

- **Extraction is regex-based, not a structured parse.** Item 1A is found by
  scoring candidate "Item 1A" occurrences by the length of text before the
  next Item 1B/2 marker, which reliably separates the real section from its
  Table of Contents listing on standard 10-K formats. HTML entities (e.g.
  `&#160;` used for spacing in headings, common in large filers' HTML) are
  decoded before matching, since undecoded entities silently broke matching
  on Apple, Boeing, and Walmart filings in earlier testing, causing every
  fiscal year for those tickers to fail extraction. This has not been
  validated against every possible filing layout (multi-file iXBRL viewers,
  unusual heading styles), so spot-check a few extracted sections by eye
  after any run against a new ticker universe.
- **Corpus-level TF-IDF is still corpus-of-one-run.** IDF weights are fit
  fresh on whatever tickers and years are requested, not on a stable
  reference corpus. Panels built from different ticker sets are not directly
  comparable to each other.
- **Forward-return validation is a rough directional check, not an event
  study.** The return window is a fixed calendar range (March to June of the
  filing year), not aligned to actual filing dates, is not risk-adjusted,
  and is not benchmarked against the market or a factor model. Treat any
  correlation from this step as suggestive only.
- **No industry or size controls.** The panel reports raw change scores with
  no firm characteristics attached, so it is not yet set up for regression
  analysis of the kind the underlying literature does.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

SEC requires a real contact string as the User-Agent on every request
(e.g. `"Jane Doe jdoe@university.edu"`), entered in the app sidebar.
Responses are cached on disk by default under `.edgar_cache/`, so repeated
runs against the same tickers are fast and do not re-hit SEC's servers.

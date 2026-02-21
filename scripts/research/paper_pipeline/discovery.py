"""Phase 1: Paper discovery from arXiv and SSRN.

Uses arXiv Atom API and SSRN HTML scraping with robust retry logic.
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from typing import Generator
from urllib.parse import quote_plus, urlencode

import requests

from .models import PaperMeta, StrategyType

# ---------------------------------------------------------------------------
# arXiv discovery via Atom API
# ---------------------------------------------------------------------------
ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
TARGET_CATEGORIES = {"q-fin.TR", "q-fin.PM", "q-fin.ST", "q-fin.CP", "q-fin.MF"}

ARXIV_QUERIES = [
    "all:trading+strategy",
    "all:alpha+generation",
    "all:return+predictability",
    "all:momentum+trading",
    "all:mean+reversion+trading",
    "all:factor+investing",
    "all:market+anomaly",
    "all:cryptocurrency+trading",
]


def _arxiv_search(
    query: str,
    category: str = "q-fin.TR",
    max_results: int = 25,
    start: int = 0,
) -> list[PaperMeta]:
    """Query arXiv Atom API and return parsed PaperMeta list."""
    # Build URL manually - arXiv API needs raw +/: characters, not percent-encoded
    search_query = f"cat:{category}+AND+{query}"
    url = (
        f"{ARXIV_API}?search_query={search_query}"
        f"&start={start}&max_results={max_results}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )

    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == 2:
                print(f"  [arxiv] FAILED after 3 attempts: {e}")
                return []
            time.sleep(2 ** attempt)

    return _parse_arxiv_feed(resp.text)


def _parse_arxiv_feed(xml_text: str) -> list[PaperMeta]:
    """Parse arXiv Atom XML into PaperMeta objects."""
    root = ET.fromstring(xml_text)
    papers = []

    for entry in root.findall("atom:entry", ARXIV_NS):
        arxiv_id_raw = entry.findtext("atom:id", "", ARXIV_NS)
        arxiv_id = arxiv_id_raw.split("/abs/")[-1] if "/abs/" in arxiv_id_raw else arxiv_id_raw

        title = entry.findtext("atom:title", "", ARXIV_NS).strip().replace("\n", " ")
        title = re.sub(r"\s+", " ", title)

        abstract = entry.findtext("atom:summary", "", ARXIV_NS).strip().replace("\n", " ")
        abstract = re.sub(r"\s+", " ", abstract)

        pub_date = entry.findtext("atom:published", "", ARXIV_NS)[:10]

        authors = [
            a.findtext("atom:name", "", ARXIV_NS)
            for a in entry.findall("atom:author", ARXIV_NS)
        ]

        categories = [
            c.get("term", "")
            for c in entry.findall("atom:category", ARXIV_NS)
        ]

        pdf_url = None
        for link in entry.findall("atom:link", ARXIV_NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")

        papers.append(PaperMeta(
            paper_id=f"arxiv:{arxiv_id}",
            title=title,
            authors=authors,
            publication_date=pub_date,
            abstract=abstract,
            source="arxiv",
            url=f"https://arxiv.org/abs/{arxiv_id}",
            pdf_url=pdf_url,
            categories=categories,
        ))

    return papers


def discover_arxiv(
    max_results_per_query: int = 25,
    categories: list[str] | None = None,
) -> list[PaperMeta]:
    """Run all arXiv discovery queries across target categories.

    Returns deduplicated list of PaperMeta sorted by date (newest first).
    """
    cats = categories or ["q-fin.TR", "q-fin.PM", "q-fin.ST"]
    seen: set[str] = set()
    all_papers: list[PaperMeta] = []

    for cat in cats:
        for query in ARXIV_QUERIES:
            print(f"  [arxiv] Searching {cat} :: {query}")
            papers = _arxiv_search(query, category=cat, max_results=max_results_per_query)
            for p in papers:
                if p.paper_id not in seen:
                    seen.add(p.paper_id)
                    all_papers.append(p)
            time.sleep(3.5)  # arXiv rate limit: 1 request per 3 seconds

    all_papers.sort(key=lambda p: p.publication_date, reverse=True)
    print(f"  [arxiv] Total unique papers: {len(all_papers)}")
    return all_papers


# ---------------------------------------------------------------------------
# SSRN discovery via HTML scraping
# ---------------------------------------------------------------------------
SSRN_SEARCH_URL = "https://papers.ssrn.com/sol3/results.cfm"
SSRN_QUERIES = [
    "trading strategy alpha",
    "return predictability anomaly",
    "momentum trading strategy",
    "mean reversion trading",
    "factor investing alpha",
    "market anomaly excess return",
    "cryptocurrency trading signal",
]


def _ssrn_search(query: str, npage: int = 1, timeout: int = 45) -> list[PaperMeta]:
    """Scrape SSRN search results page for paper metadata.

    SSRN doesn't provide a clean API, so we parse their search results HTML.
    Falls back gracefully on timeout/error.
    """
    params = {
        "txtKey_Words": query,
        "txtKey_Words_Options": "1",
        "npage": npage,
        "RequestTimeout": "50000000",
    }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
    }

    for attempt in range(3):
        try:
            resp = requests.get(
                SSRN_SEARCH_URL, params=params, headers=headers, timeout=timeout
            )
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == 2:
                print(f"  [ssrn] FAILED after 3 attempts for '{query}': {e}")
                return []
            time.sleep(5 * (attempt + 1))

    return _parse_ssrn_html(resp.text)


def _parse_ssrn_html(html: str) -> list[PaperMeta]:
    """Extract paper metadata from SSRN search results HTML.

    SSRN's HTML structure uses specific class names we can target with regex
    when BeautifulSoup isn't available.
    """
    papers: list[PaperMeta] = []

    # Each result is in a div with class "result-item"
    # Title is in <a class="title optClickTitle" href="...">
    title_pattern = re.compile(
        r'<a[^>]*class="[^"]*title[^"]*"[^>]*href="([^"]*)"[^>]*>\s*(.*?)\s*</a>',
        re.DOTALL,
    )
    # Authors in <span class="authors-list">
    author_pattern = re.compile(
        r'<span[^>]*class="[^"]*authors-list[^"]*"[^>]*>(.*?)</span>',
        re.DOTALL,
    )
    # Date pattern
    date_pattern = re.compile(
        r'(?:Posted|Last revised):\s*(\d{1,2}\s+\w+\s+\d{4})',
    )
    # Abstract in <span class="abstract-text">
    abstract_pattern = re.compile(
        r'<span[^>]*class="[^"]*abstract-text[^"]*"[^>]*>(.*?)</span>',
        re.DOTALL,
    )

    # Split into result blocks
    blocks = re.split(r'class="result-item"', html)

    for block in blocks[1:]:  # skip header
        title_match = title_pattern.search(block)
        if not title_match:
            continue

        url = title_match.group(1).strip()
        title = re.sub(r"<[^>]+>", "", title_match.group(2)).strip()
        title = re.sub(r"\s+", " ", title)

        ssrn_id = ""
        id_match = re.search(r"abstract_id=(\d+)", url)
        if id_match:
            ssrn_id = id_match.group(1)
        elif re.search(r"/(\d{6,})", url):
            ssrn_id = re.search(r"/(\d{6,})", url).group(1)  # type: ignore[union-attr]

        authors: list[str] = []
        auth_match = author_pattern.search(block)
        if auth_match:
            auth_text = re.sub(r"<[^>]+>", "", auth_match.group(1))
            authors = [a.strip() for a in auth_text.split(",") if a.strip()]

        pub_date = ""
        date_match = date_pattern.search(block)
        if date_match:
            pub_date = date_match.group(1)

        abstract = ""
        abs_match = abstract_pattern.search(block)
        if abs_match:
            abstract = re.sub(r"<[^>]+>", "", abs_match.group(1)).strip()
            abstract = re.sub(r"\s+", " ", abstract)

        if not url.startswith("http"):
            url = f"https://papers.ssrn.com{url}"

        papers.append(PaperMeta(
            paper_id=f"ssrn:{ssrn_id}" if ssrn_id else f"ssrn:{title[:40]}",
            title=title,
            authors=authors,
            publication_date=pub_date,
            abstract=abstract,
            source="ssrn",
            url=url,
            pdf_url=None,
            categories=[],
        ))

    return papers


def discover_ssrn(max_pages: int = 1) -> list[PaperMeta]:
    """Run all SSRN search queries.

    Returns deduplicated list of PaperMeta.
    """
    seen: set[str] = set()
    all_papers: list[PaperMeta] = []

    for query in SSRN_QUERIES:
        for page in range(1, max_pages + 1):
            print(f"  [ssrn] Searching: '{query}' (page {page})")
            papers = _ssrn_search(query, npage=page)
            for p in papers:
                if p.paper_id not in seen:
                    seen.add(p.paper_id)
                    all_papers.append(p)
            time.sleep(5)

    print(f"  [ssrn] Total unique papers: {len(all_papers)}")
    return all_papers


# ---------------------------------------------------------------------------
# Heuristic classification
# ---------------------------------------------------------------------------
STRATEGY_KEYWORDS: dict[StrategyType, list[str]] = {
    StrategyType.MOMENTUM: [
        "momentum", "trend following", "trend-following", "moving average",
        "breakout", "time-series momentum", "cross-sectional momentum",
    ],
    StrategyType.MEAN_REVERSION: [
        "mean reversion", "mean-reversion", "contrarian", "reversal",
        "pairs trading", "cointegration", "ornstein-uhlenbeck",
    ],
    StrategyType.STAT_ARB: [
        "statistical arbitrage", "stat arb", "relative value",
        "pairs", "spread trading",
    ],
    StrategyType.VOLATILITY: [
        "volatility", "vol trading", "variance swap", "vix",
        "straddle", "options strategy", "implied volatility",
    ],
    StrategyType.FACTOR: [
        "factor", "value factor", "size factor", "quality",
        "carry", "low volatility", "factor investing",
    ],
    StrategyType.SEASONAL: [
        "seasonal", "calendar effect", "day-of-week", "turn-of-month",
        "january effect", "halloween",
    ],
    StrategyType.ML_SIGNAL: [
        "machine learning", "deep learning", "neural network",
        "reinforcement learning", "random forest", "gradient boosting",
        "transformer", "lstm", "autoencoder",
    ],
    StrategyType.EXECUTION: [
        "execution", "market making", "order flow", "optimal execution",
        "transaction cost", "slippage", "liquidation",
    ],
}

ASSET_KEYWORDS: dict[str, list[str]] = {
    "equities": ["equit", "stock", "s&p", "nasdaq", "russell", "shares"],
    "crypto": ["crypto", "bitcoin", "btc", "ethereum", "defi", "blockchain", "token"],
    "fx": ["forex", "currency", "fx", "exchange rate"],
    "commodities": ["commodity", "commodities", "oil", "gold", "futures"],
    "fixed_income": ["bond", "fixed income", "yield", "interest rate", "treasury"],
    "options": ["option", "derivative", "vol surface"],
}


def classify_paper(paper: PaperMeta) -> PaperMeta:
    """Heuristically classify strategy type and asset class from abstract."""
    text = f"{paper.title} {paper.abstract}".lower()

    best_type = StrategyType.OTHER
    best_score = 0
    for stype, keywords in STRATEGY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_score = score
            best_type = stype
    paper.strategy_type = best_type

    for asset, keywords in ASSET_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            paper.asset_classes.append(asset)
    if not paper.asset_classes:
        paper.asset_classes = ["unspecified"]

    # Extract data period heuristic
    year_matches = re.findall(r"\b(19\d{2}|20\d{2})\b", paper.abstract)
    if len(year_matches) >= 2:
        years = sorted(set(year_matches))
        paper.data_period = f"{years[0]}-{years[-1]}"

    # Extract claimed results from abstract
    sharpe_match = re.search(
        r"sharpe\s*(?:ratio)?\s*(?:of|=|:)?\s*([\d.]+)", text
    )
    alpha_match = re.search(
        r"alpha\s*(?:of|=|:)?\s*([\d.]+)\s*%", text
    )
    return_match = re.search(
        r"(?:annual|annualized)\s*(?:return|excess return)\s*(?:of|=|:)?\s*([\d.]+)\s*%", text
    )
    claims = []
    if sharpe_match:
        claims.append(f"Sharpe={sharpe_match.group(1)}")
    if alpha_match:
        claims.append(f"alpha={alpha_match.group(1)}%")
    if return_match:
        claims.append(f"ann.return={return_match.group(1)}%")
    paper.claimed_result = "; ".join(claims) if claims else "see abstract"

    return paper


def run_discovery(
    arxiv_max_per_query: int = 25,
    ssrn_max_pages: int = 1,
) -> list[PaperMeta]:
    """Run full discovery pipeline across arXiv and SSRN.

    Returns deduplicated, classified list of papers.
    """
    print("=" * 60)
    print("PHASE 1: PAPER DISCOVERY")
    print("=" * 60)

    papers: list[PaperMeta] = []

    print("\n--- arXiv ---")
    arxiv_papers = discover_arxiv(max_results_per_query=arxiv_max_per_query)
    papers.extend(arxiv_papers)

    print("\n--- SSRN ---")
    ssrn_papers = discover_ssrn(max_pages=ssrn_max_pages)
    papers.extend(ssrn_papers)

    print(f"\nClassifying {len(papers)} papers ...")
    papers = [classify_paper(p) for p in papers]

    print(f"Discovery complete: {len(papers)} unique papers found.")
    return papers

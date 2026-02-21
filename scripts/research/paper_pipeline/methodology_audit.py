"""Phase 2b: Methodology audit — plausibility gate.

Runs AFTER the quality filter stack (Phase 2a) and BEFORE strategy extraction
(Phase 3). Detects methodology red flags that inflate reported performance:
implausible Sharpe ratios, circular OOS, window selection, survivorship bias,
look-ahead bias, and overfitting indicators.

Calibration reference: legitimate published factors from top-tier quant shops
(AQR, Two Sigma, Man, etc.) with decades of data rarely exceed Sharpe 2.0
after costs. Anything above 3.0 is suspicious; above 5.0 requires an
explicit, verified explanation before proceeding.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import FilterResult, FilterVerdict, PaperMeta

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
SHARPE_SUSPICIOUS = 3.0
SHARPE_IMPLAUSIBLE = 5.0

CAGR_SUSPICIOUS = 50.0   # annual %, for non-crypto equity strategies
CAGR_IMPLAUSIBLE = 100.0

# Crypto gets a higher CAGR ceiling (the asset class genuinely trends harder)
CAGR_SUSPICIOUS_CRYPTO = 100.0
CAGR_IMPLAUSIBLE_CRYPTO = 300.0


@dataclass
class AuditFlag:
    """A single methodology concern."""
    code: str
    severity: str  # "info", "warning", "reject"
    detail: str


@dataclass
class AuditResult:
    """Aggregated methodology audit outcome."""
    paper_id: str
    passed: bool
    flags: list[AuditFlag] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "passed": self.passed,
            "flags": [
                {"code": f.code, "severity": f.severity, "detail": f.detail}
                for f in self.flags
            ],
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Individual audit checks
# ---------------------------------------------------------------------------

def _extract_sharpe(text: str) -> list[float]:
    """Pull all Sharpe-ratio-like numbers from text."""
    patterns = [
        r"sharpe\s*(?:ratio)?\s*(?:of|=|:|above|exceeding|over)?\s*([\d]+(?:\.[\d]+)?)",
        r"([\d]+(?:\.[\d]+)?)\s*(?:-|\s)?sharpe",
        r"risk[- ]adjusted\s*(?:return)?\s*(?:of|=|:)?\s*([\d]+(?:\.[\d]+)?)\s*times",
    ]
    values: list[float] = []
    for pat in patterns:
        for m in re.finditer(pat, text):
            try:
                v = float(m.group(1))
                if 0.1 < v < 100:  # plausible range for a Sharpe-like number
                    values.append(v)
            except (ValueError, IndexError):
                pass
    return values


def _extract_cagr(text: str) -> list[float]:
    """Pull annualized return percentages from text."""
    patterns = [
        r"annualized\s*(?:return|excess return)s?\s*(?:of|=|:)?\s*([\d]+(?:\.[\d]+)?)\s*(?:percent|%)",
        r"([\d]+(?:\.[\d]+)?)\s*(?:percent|%)\s*(?:annualized|annual)\s*(?:return)?",
        r"return\s*(?:of|=|:)?\s*([\d]+(?:\.[\d]+)?)\s*(?:percent|%)",
    ]
    values: list[float] = []
    for pat in patterns:
        for m in re.finditer(pat, text):
            try:
                v = float(m.group(1))
                if 1.0 < v < 10000:
                    values.append(v)
            except (ValueError, IndexError):
                pass
    return values


def _check_implausible_performance(paper: PaperMeta) -> list[AuditFlag]:
    """IMPLAUSIBLE_SHARPE / IMPLAUSIBLE_CAGR: headline numbers too good."""
    text = f"{paper.title} {paper.abstract}".lower()
    flags: list[AuditFlag] = []

    is_crypto = any(ac in ("crypto",) for ac in paper.asset_classes)

    # --- Sharpe ---
    sharpes = _extract_sharpe(text)
    if sharpes:
        peak = max(sharpes)
        if peak >= SHARPE_IMPLAUSIBLE:
            flags.append(AuditFlag(
                code="IMPLAUSIBLE_SHARPE",
                severity="reject",
                detail=(
                    f"Claimed Sharpe {peak:.1f} is implausible. "
                    f"Top-tier published factors rarely exceed 2.0 after costs. "
                    f"Threshold for auto-reject: >{SHARPE_IMPLAUSIBLE:.1f}."
                ),
            ))
        elif peak >= SHARPE_SUSPICIOUS:
            flags.append(AuditFlag(
                code="SUSPICIOUS_SHARPE",
                severity="warning",
                detail=(
                    f"Claimed Sharpe {peak:.1f} exceeds {SHARPE_SUSPICIOUS:.1f}. "
                    f"Requires explicit verification of methodology before proceeding."
                ),
            ))

    # --- CAGR ---
    cagrs = _extract_cagr(text)
    if cagrs:
        peak = max(cagrs)
        thresh_sus = CAGR_SUSPICIOUS_CRYPTO if is_crypto else CAGR_SUSPICIOUS
        thresh_imp = CAGR_IMPLAUSIBLE_CRYPTO if is_crypto else CAGR_IMPLAUSIBLE

        if peak >= thresh_imp:
            flags.append(AuditFlag(
                code="IMPLAUSIBLE_CAGR",
                severity="reject",
                detail=f"Claimed CAGR {peak:.1f}% is implausible for asset class.",
            ))
        elif peak >= thresh_sus:
            flags.append(AuditFlag(
                code="SUSPICIOUS_CAGR",
                severity="warning",
                detail=f"Claimed CAGR {peak:.1f}% is unusually high; verify methodology.",
            ))

    # --- Sharpe x CAGR cross-check ---
    if sharpes and cagrs:
        peak_sharpe = max(sharpes)
        peak_cagr = max(cagrs)
        if peak_sharpe > 3.0 and peak_cagr > 80.0:
            flags.append(AuditFlag(
                code="SHARPE_CAGR_MISMATCH",
                severity="reject",
                detail=(
                    f"Both Sharpe ({peak_sharpe:.1f}) and CAGR ({peak_cagr:.1f}%) are extreme. "
                    f"Combination is almost certainly a methodology artifact."
                ),
            ))

    return flags


def _check_circular_oos(paper: PaperMeta) -> list[AuditFlag]:
    """CIRCULAR_OOS: walk-forward that isn't truly out-of-sample.

    Detects: "frozen parameters" chosen by in-sample optimization,
    walk-forward with parameter re-selection, lookback chosen to maximize OOS.
    """
    text = f"{paper.title} {paper.abstract}".lower()
    flags: list[AuditFlag] = []

    circular_kw = [
        "frozen parameter",
        "optimal parameter",
        "best parameter",
        "parameter selection",
        "parameter tuning",
        "hyperparameter search",
        "grid search",
    ]
    oos_kw = ["out-of-sample", "out of sample", "walk-forward", "walk forward", "oos"]

    has_oos = any(kw in text for kw in oos_kw)
    has_param_selection = any(kw in text for kw in circular_kw)

    if has_oos and has_param_selection:
        # "frozen parameters" + "grid search" in the same paper = the OOS window
        # was likely selected AFTER seeing results on the full dataset
        truly_blind = any(kw in text for kw in [
            "pre-registered", "pre-specified", "published before",
            "no look-ahead", "truly blind",
        ])
        if not truly_blind:
            flags.append(AuditFlag(
                code="CIRCULAR_OOS",
                severity="warning",
                detail=(
                    "Paper claims OOS but also describes parameter optimization. "
                    "Unless parameters were pre-registered or specified ex-ante, "
                    "the OOS window may be circular (parameters chosen to look good OOS)."
                ),
            ))

    return flags


def _check_window_selection(paper: PaperMeta) -> list[AuditFlag]:
    """SELECTED_WINDOWS: evidence the evaluation period was cherry-picked."""
    text = f"{paper.title} {paper.abstract}".lower()
    flags: list[AuditFlag] = []

    # Single-index, no multi-market, but extraordinary claims
    single_index_kw = ["s&p 500", "s&p500", "russell", "nasdaq", "dow jones"]
    has_single_index = any(kw in text for kw in single_index_kw)

    # Does the paper test on multiple non-overlapping periods?
    multi_period_kw = [
        "sub-period", "subsample", "different time period",
        "pre-crisis", "post-crisis", "multiple period",
        "decade", "halves",
    ]
    has_multi_period = any(kw in text for kw in multi_period_kw)

    # Extract year range
    year_matches = re.findall(r"\b(19\d{2}|20\d{2})\b", paper.abstract)
    data_years = 0
    if len(year_matches) >= 2:
        years = sorted(int(y) for y in set(year_matches))
        data_years = years[-1] - years[0]

    sharpes = _extract_sharpe(f"{paper.title} {paper.abstract}".lower())
    peak_sharpe = max(sharpes) if sharpes else 0

    # Short window + high Sharpe + single index = selected window
    if has_single_index and data_years <= 5 and peak_sharpe > 3.0:
        flags.append(AuditFlag(
            code="SELECTED_WINDOWS",
            severity="reject",
            detail=(
                f"Single index, {data_years}y window, Sharpe {peak_sharpe:.1f}. "
                f"Short, single-market sample with extreme results = likely window selection."
            ),
        ))
    elif has_single_index and not has_multi_period and peak_sharpe > 2.5:
        flags.append(AuditFlag(
            code="SELECTED_WINDOWS",
            severity="warning",
            detail=(
                f"Single index with no multi-period robustness checks and "
                f"Sharpe {peak_sharpe:.1f}. Evaluate window sensitivity."
            ),
        ))

    return flags


def _check_survivorship_bias(paper: PaperMeta) -> list[AuditFlag]:
    """SURVIVORSHIP_BIAS: using only currently listed or surviving constituents."""
    text = f"{paper.title} {paper.abstract}".lower()
    flags: list[AuditFlag] = []

    survives_kw = ["delisted", "survivorship", "survivor bias", "dead stock"]
    addresses_bias = any(kw in text for kw in survives_kw)

    equity_kw = ["equity", "stock", "s&p", "russell", "cross-sectional"]
    is_equity_cs = any(kw in text for kw in equity_kw)

    if is_equity_cs and not addresses_bias:
        flags.append(AuditFlag(
            code="SURVIVORSHIP_BIAS",
            severity="warning",
            detail=(
                "Cross-sectional equity strategy with no mention of survivorship "
                "bias handling. Results may be inflated by excluding delisted losers."
            ),
        ))

    return flags


def _check_lookahead_bias(paper: PaperMeta) -> list[AuditFlag]:
    """LOOKAHEAD_BIAS: signals that may use future information."""
    text = f"{paper.title} {paper.abstract}".lower()
    flags: list[AuditFlag] = []

    lookahead_kw = [
        "closing price",
        "same-day", "same day",
        "contemporaneous",
        "point-in-time", "point in time",
    ]
    # Positive indicators that they handle it properly
    causal_kw = [
        "lagged", "t-1", "previous day", "prior day",
        "strictly causal", "no look-ahead", "no lookahead",
        "point-in-time database",
    ]

    has_concern = any(kw in text for kw in lookahead_kw)
    has_causal = any(kw in text for kw in causal_kw)

    if has_concern and not has_causal:
        flags.append(AuditFlag(
            code="LOOKAHEAD_BIAS",
            severity="warning",
            detail=(
                "Abstract references same-day or contemporaneous data without "
                "explicitly addressing look-ahead bias. Verify signal timing."
            ),
        ))

    return flags


def _check_overfitting(paper: PaperMeta) -> list[AuditFlag]:
    """OVERFITTING: too many degrees of freedom relative to data."""
    text = f"{paper.title} {paper.abstract}".lower()
    flags: list[AuditFlag] = []

    # Many-parameter models with extreme results
    complex_model_kw = [
        "deep learning", "neural network", "transformer",
        "attention mechanism", "lstm", "gru",
        "random forest", "gradient boosting", "xgboost",
        "ensemble of", "stacking", "bagging",
    ]
    has_complex = any(kw in text for kw in complex_model_kw)

    # Regularization / overfitting awareness
    regularization_kw = [
        "regularization", "dropout", "early stopping",
        "cross-validation", "information criterion",
        "aic", "bic", "lasso", "ridge",
        "overfitting", "generalization",
    ]
    has_regularization = any(kw in text for kw in regularization_kw)

    sharpes = _extract_sharpe(text)
    peak_sharpe = max(sharpes) if sharpes else 0

    if has_complex and peak_sharpe > 3.0 and not has_regularization:
        flags.append(AuditFlag(
            code="OVERFIT_LIKELY",
            severity="reject",
            detail=(
                f"Complex ML model with Sharpe {peak_sharpe:.1f} and no mention of "
                f"regularization or overfitting controls. High probability of in-sample fit."
            ),
        ))
    elif has_complex and peak_sharpe > 2.0 and not has_regularization:
        flags.append(AuditFlag(
            code="OVERFIT_RISK",
            severity="warning",
            detail=(
                f"Complex ML model with Sharpe {peak_sharpe:.1f}. "
                f"No regularization mentioned — verify generalization."
            ),
        ))

    return flags


def _check_bias_accumulation(paper: PaperMeta) -> list[AuditFlag]:
    """BIAS_LIKELY: multiple minor biases that compound into implausibility.

    Even if no single flag reaches "reject" severity, stacking 3+ warnings
    on a paper with Sharpe > 3.0 is enough to reject.
    """
    # This is a meta-check — called after all individual checks.
    # Implemented in audit_paper() below.
    return []


# ---------------------------------------------------------------------------
# Composite audit
# ---------------------------------------------------------------------------
ALL_CHECKS = [
    _check_implausible_performance,
    _check_circular_oos,
    _check_window_selection,
    _check_survivorship_bias,
    _check_lookahead_bias,
    _check_overfitting,
]


def audit_paper(paper: PaperMeta) -> AuditResult:
    """Run full methodology audit on a single paper.

    Returns AuditResult with pass/fail and detailed flags.
    """
    all_flags: list[AuditFlag] = []

    for check_fn in ALL_CHECKS:
        all_flags.extend(check_fn(paper))

    # --- Meta-check: bias accumulation ---
    n_rejects = sum(1 for f in all_flags if f.severity == "reject")
    n_warnings = sum(1 for f in all_flags if f.severity == "warning")

    sharpes = _extract_sharpe(f"{paper.title} {paper.abstract}".lower())
    peak_sharpe = max(sharpes) if sharpes else 0

    # Accumulation rule: 3+ warnings on a paper with Sharpe > 3.0 → reject
    if n_warnings >= 3 and peak_sharpe > SHARPE_SUSPICIOUS:
        all_flags.append(AuditFlag(
            code="BIAS_LIKELY",
            severity="reject",
            detail=(
                f"{n_warnings} methodology warnings accumulated on a paper claiming "
                f"Sharpe {peak_sharpe:.1f}. Compound bias makes results unreliable."
            ),
        ))
        n_rejects += 1

    # 2+ warnings with no rejects still gets flagged but passes
    if n_warnings >= 2 and n_rejects == 0:
        all_flags.append(AuditFlag(
            code="ELEVATED_SCRUTINY",
            severity="warning",
            detail=(
                f"{n_warnings} methodology concerns flagged. "
                f"Paper should receive manual review before strategy extraction."
            ),
        ))

    passed = n_rejects == 0

    flag_codes = [f.code for f in all_flags]
    if not passed:
        reject_codes = [f.code for f in all_flags if f.severity == "reject"]
        summary = f"REJECTED: {', '.join(reject_codes)}"
    elif n_warnings > 0:
        summary = f"PASSED with {n_warnings} warning(s): {', '.join(flag_codes)}"
    else:
        summary = "PASSED: no methodology concerns"

    return AuditResult(
        paper_id=paper.paper_id,
        passed=passed,
        flags=all_flags,
        summary=summary,
    )


def run_methodology_audit(
    passed_papers: list[tuple[PaperMeta, FilterResult]],
) -> tuple[
    list[tuple[PaperMeta, FilterResult]],
    list[tuple[PaperMeta, FilterResult]],
]:
    """Run methodology audit on all papers that passed the filter stack.

    Papers that fail the audit are moved to the rejected pile with
    FAIL_METHODOLOGY verdict.

    Returns (audit_passed, audit_rejected).
    """
    print("\n" + "=" * 60)
    print("PHASE 2b: METHODOLOGY AUDIT")
    print("=" * 60)

    audit_passed: list[tuple[PaperMeta, FilterResult]] = []
    audit_rejected: list[tuple[PaperMeta, FilterResult]] = []

    for paper, filt in passed_papers:
        audit = audit_paper(paper)

        # Graft audit results onto the FilterResult
        filt.filter_scores["methodology_audit"] = audit.to_dict()

        if audit.passed:
            flag_str = ""
            if any(f.severity == "warning" for f in audit.flags):
                codes = [f.code for f in audit.flags if f.severity == "warning"]
                flag_str = f" [AUDIT WARN: {', '.join(codes)}]"
            print(f"  AUDIT PASS{flag_str}: {paper.title[:65]}")
            audit_passed.append((paper, filt))
        else:
            reject_codes = [f.code for f in audit.flags if f.severity == "reject"]
            reject_detail = "; ".join(
                f.detail[:80] for f in audit.flags if f.severity == "reject"
            )
            filt.verdict = FilterVerdict.FAIL_METHODOLOGY
            filt.passed = False
            filt.rejection_reason = f"Methodology: {', '.join(reject_codes)} — {reject_detail}"
            print(f"  AUDIT REJECT ({', '.join(reject_codes)}): {paper.title[:55]}")
            for f in audit.flags:
                if f.severity == "reject":
                    print(f"    -> {f.code}: {f.detail[:90]}")
            audit_rejected.append((paper, filt))

    print(
        f"\nAudit results: {len(audit_passed)} passed, "
        f"{len(audit_rejected)} rejected"
    )
    return audit_passed, audit_rejected

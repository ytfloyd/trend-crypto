# Equity-Index Derivatives Universe — Instrument Scope

**Purpose:** the data-collection scope for the equity-index-derivatives sleeve. Hand
this to the data-collection agent. It is a standalone reference — not part of the
K2 Trade Atlas rulebook.

**Seeded by:** Barclays Index Derivatives Weekly — *Enhanced hedging with VSTOXX*
(26 Apr 2010), whose tracked-index "volatility snapshot" defined the initial
complex; extended here to the full set of liquid, exchange-traded equity-index
derivative complexes the fund trades, plus their associated volatility-index /
vol-derivative families (the **VD branch**).

**Execution constraint:** exchange-traded / cleared only. OTC variance & vol swaps
are listed for completeness but are **out-of-scope for execution** (substitute
with vol-futures or option strips); collect their reference levels where available.

**Symbol caveat:** symbols below are common Bloomberg/exchange roots. The agent
must verify the exact contract codes, multipliers, and trading calendars at each
venue — treat specs here as a starting reference, not ground truth.

**Maintained alongside K2 Trade Atlas.** Update tiers as venues/liquidity change;
the vol-index families and constant-maturity series are the priority for the VD
branch.

---

## Current repo coverage (IBKR minute lake, as of Jun 2026)

Partial overlap with Tier 1 US — dated **index futures** only, via
`scripts/volbook/walk_dated_futures_minute.py` and `src/volbook/contracts.py`:

| Complex | Lake symbol | Status |
|---------|-------------|--------|
| S&P 500 (ES) | `ES` | Dated 1m bars; front-month continuous artifact |
| Nasdaq-100 (NQ) | `NQ` | Dated 1m bars |
| Russell 2000 (RTY) | `RTY` | Dated 1m bars |
| VIX (VX) | `VX` / `VIX` | Dated 1m bars; front-month continuous artifact |

**Cboe index levels** — daily `bars_1d` in `data/indices_market.duckdb` via
`python -m scripts.research.cboe_indices.ingest` (Cboe CDN EOD CSVs). Default
universe (`--universe vol_correlation`, ~40 symbols):

| Family | Symbols | Status |
|--------|---------|--------|
| Equity vol / skew | `VIX`, `VVIX`, `VIX9D`, `VIX1D`, `VIX3M`, `VIX6M`, `VIX1Y`, `VXO`, `VXN`, `RVX`, `SKEW`, `SMILE` | Daily on CDN |
| Dispersion | `VIXEQ`, `DSPX` | Daily on CDN |
| Implied correlation | `COR1M`, `COR3M`, `COR6M`, `COR9M`, `COR1Y`, `COR10D`, `COR30D`, `COR70D`, `COR90D` | Daily on CDN |
| Commodity / rates / thematic | `OVX`, `GVZ`, `TYVIX`, `VXTLT`, `VXTH`, `VXEFA`, `VXEEM`, `VXHYG`, `VXSLV`, `VXGDX` | Daily on CDN |
| Single-name vol | `VXAPL`, `VXAZN`, `VXGOG`, `VXGS`, `VXIBM`, `VXFXI`, `VXBAC`, `VXWMT` | Daily on CDN |

Not on public CDN (403): `DSPBX`, `COR3MD`, `VXNFLX`, `VXTSLA`, most other
single-name `VX*` tickers. Non-Cboe indices (e.g. `MOVE`, `VSTOXX`) are out of
scope for this ingest path.

**Not yet in lake:** cash index levels (SPX/NDX/RTY), index options, VIX options,
micros (MES/MNQ/M2K), non-US venues, variance futures, constant-maturity
investable indices, or derived surfaces (ATM/skew/VRP).

---

## How to read each entry — what "data to collect" means

For every underlying index in scope, collect the following families (**daily**,
plus **intraday close** where available), each as a **full history**:

1. **Underlying index** — level (O/H/L/C), total-return variant where relevant,
   dividend points / index dividend futures.
2. **Index futures** — the **full listed curve**: every listed expiry, settlement,
   open interest, volume, and the front/2nd/3rd roll spreads.
3. **Index options** — the surface: strikes × expiries, implied vol per strike,
   greeks, open interest, volume. Derived: ATM term structure
   (1w/1m/2m/3m/6m/1y/2y), skew (25Δ and 10Δ risk reversal + butterfly),
   put/call skew.
4. **Volatility index** — the vol-index **level** (the index's "VIX").
5. **Vol-index futures** — the vol-futures **curve** (every expiry) → term
   structure, contango/backwardation, roll cost (feeds VD-02), and forward vols
   (kinks).
6. **Vol-index options** — surface (vol-of-vol; feeds VD-03).
7. **Variance / vol products** — variance-futures or variance-swap reference
   levels, and any constant-maturity investable vol-futures index (e.g. VST1MT,
   SPVXSTR/SPVXMTR) used by VD-04.

---

## Tier 1 — Core, fund-tradable (highest priority)

| Region | Index | Underlying root | Index futures | Index options | Vol index | Vol futures / options | Variance / investable |
|--------|-------|-----------------|---------------|---------------|-----------|----------------------|------------------------|
| US | S&P 500 | SPX | ES (CME, $50×); micro MES | SPX / SPXW (CBOE, $100×); SPY ETF opts | VIX | VX VIX futures (CFE, $1000×) + mini; VIX options | SPX variance (OTC); SPVXSTR/SPVXMTR investable; VXX/VXZ/UVXY/SVXY ETPs |
| US | Nasdaq-100 | NDX | NQ (CME, $20×); micro MNQ | NDX / NDXP; QQQ ETF opts | VXN | VXN futures (CFE, thin) | QQQ var (OTC) |
| US | Russell 2000 | RTY | RTY (CME, $50×); micro M2K | RUT (CBOE); IWM ETF opts | RVX | RVX futures (CFE, thin) | — |
| Europe | EURO STOXX 50 | SX5E | FESX (Eurex, €10×) | OESX (Eurex) | VSTOXX (V2TX) | FVS VSTOXX futures (Eurex, €100×) + mini FVM; VSTOXX options | EURO STOXX 50 Variance Futures (Eurex); VST1MT investable 1m index |
| Europe | DAX 40 | DAX | FDAX (Eurex, €25×); mini FDXM | ODAX (Eurex) | VDAX-NEW (V1X) | VDAX futures (Eurex, thin) | DAX Variance Futures (Eurex) |
| Asia | Nikkei 225 | NKY | OSE NK225 (¥1000×); CME NKD ($5×) / NIY (¥500×) | OSE Nikkei 225 opts | Nikkei VI (VXJ) | Nikkei VI futures (OSE) | — |
| Asia | Hang Seng | HSI | HSI (HKEX, HK$50×); mini MHI | HSI options (HKEX) | VHSI | VHSI futures (HKEX) | — |

---

## Tier 2 — Liquid, secondary priority

| Region | Index | Underlying root | Index futures | Index options | Vol index | Notes |
|--------|-------|-----------------|---------------|---------------|-----------|-------|
| Europe | CAC 40 | CAC / PX1 | FCE (Euronext Paris, €10×) | PXA options | VCAC | — |
| Europe | FTSE 100 | UKX | Z (ICE Europe, £10×) | options on Z | VFTSE (IVUK) | — |
| Europe | SMI (Swiss) | SMI | FSMI (Eurex, CHF 10×) | OSMI | VSMI (V3X) | least-volatile in the source universe |
| Europe | AEX (NL) | AEX | FTI (Euronext, €200×) | AEX options | VAEX | — |
| Europe | IBEX 35 (ES) | IBEX | (MEFF) | IBEX options | VIBEX | — |
| Europe | FTSE MIB (IT) | FTSEMIB | (IDEM) | MIB options | — | — |
| Asia | KOSPI 200 | KOSPI2 | (KRX) | KRX KOSPI200 opts (very liquid) | VKOSPI | deep options market |
| Asia | NIFTY 50 (India) | NIFTY | NIFTY futures (NSE) | NIFTY options (NSE, very liquid) | India VIX | among world's most-traded index options |
| Asia | ASX 200 (AU) | AS51 | SPI / AP (ASX, A$25×) | XJO options | A-VIX | — |
| Asia | TOPIX (JP) | TPX | OSE TOPIX (¥10000×) | TOPIX options | — | breadth complement to NKY |
| Asia | HSCEI (China H) | HSCEI | HSCEI futures (HKEX) | HSCEI options | VHSCEI | China-exposure |

---

## Tier 3 — Extended / context-only

The Barclays weekly also tracked these — keep for cross-market vol context, but
several are EM / less-liquid / currently restricted and are **not core data
targets**:

| Index | Root | Status |
|-------|------|--------|
| FTSE/JSE Top 40 (South Africa) | TOP40 | Vol index SAVI; EM, lower priority |
| CECE (Central Europe, EUR) | CECEEUR | EM composite; context only |
| RDX (Russian Depositary, USD) | RDXUSD | **Restricted/sanctioned — exclude from collection** |
| (regional, source-specific) | SBR | Verify identity; context only |

---

## Volatility-index families — collect the curve, not just the level

The VD branch (VD-01..04) trades the vol complex, so the vol-**index**
derivatives are first-class data targets, not afterthoughts:

| Vol index | Underlying | Futures venue | Why it matters |
|-----------|------------|---------------|----------------|
| VIX | SPX | CFE (VX) | Deepest vol-futures curve; roll cost (VD-02), vol-of-vol (VD-03), VST-style investable indices + ETPs |
| VSTOXX (V2TX) | SX5E | Eurex (FVS) | EU equivalent; VST1MT investable index (VD-04); variance futures |
| VDAX-NEW (V1X) | DAX | Eurex | German vol; DAX variance futures |
| VXN / RVX | NDX / RTY | CFE (thin) | US cross-index vol RV |
| Nikkei VI, VHSI, VKOSPI, A-VIX, India VIX, VCAC, VFTSE, VSMI, VAEX | resp. index | resp. venue | cross-asset vol-of-vol RV (VD-03); regional dispersion |
| MOVE | US rates | (index only) | rates-vol cross-asset (CV-26 equity-rates hybrid) |
| OVX, GVZ | oil, gold | CBOE (index) | commodity vol cross-reference |

For each vol index with listed futures: collect **every expiry** → build the
term-structure (contango/backwardation, monthly roll cost, forward-vol kinks) and
the constant-maturity 1m/2m/3m series (the VST1MT / SPVXSTR analogue).

---

## Cross-series / derived datasets to compute and store

So the K2 tools can run directly off the collected data:

- ATM vol term structure per index (for OV-17, VD-02, vol-curve screeners).
- Skew term structure — 25Δ/10Δ RR & BF at 1m/3m/6m/12m (OV-10 nuance, OV-32).
- Realized vol (close-to-close and Parkinson/Garman-Klass) at multiple windows
  (for OV-35 GARCH/EGARCH, VRP, OV-34 screen).
- Implied − realized (VRP) per index/tenor.
- Vol-futures roll cost & forward vols per vol index (VD-02, `vol_derivatives.py`).
- Cross-asset driver panel for the fair-value model (OV-36): each index's implied
  vol + EUR/USD & USD/JPY vol, VIX, credit spreads (iTraxx Main / CDX IG), term
  slope, risk reversal, MSCI/global-equity beta.
- Index dividend futures / dividend points (affect option pricing & carry).

---

## Collection checklist (per index, per day)

- [ ] Underlying index O/H/L/C (+ total-return variant, dividend points)
- [ ] Full index-futures curve: expiries, settle, OI, volume, roll spreads
- [ ] Index option surface: strike × expiry IV, greeks, OI, volume
- [ ] Derived: ATM term structure + 25Δ/10Δ skew per tenor
- [ ] Vol-index level
- [ ] Vol-index futures curve (all expiries) + constant-maturity 1m/2m/3m
- [ ] Vol-index options surface (where listed)
- [ ] Variance futures / swap reference levels (where available)
- [ ] Realized vol (multiple windows + estimators)

---

## Agent implementation notes

1. **Venue verification required** before any contract is added to
   `src/volbook/contracts.py` or a new lake schema — multiplier, tick, roll
   calendar, and IBKR qualification must be probed per product.
2. **OTC variance / vol swaps:** reference levels only; no execution data path.
3. **RDXUSD:** hard exclude (sanctions).
4. **Priority order for this repo's existing IBKR stack:** Tier 1 US index futures
   (ES/NQ/RTY) → VIX futures (VX) → VIX options IV → SPX index level → Tier 1
   vol-index curves → Tier 1 non-US → Tier 2.
5. **Frozen backtest set:** continuous ES/VX parquets live under
   `artifacts/research/{es,vx}_continuous/` and in
   `frozen_snapshots/2026-06-03/futures/`; extend freeze manifest when new series
   are production-ready.

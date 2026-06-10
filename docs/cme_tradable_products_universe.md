# CME Tradable Products Universe

Data-driven reference universe for CME Group products, built from the desk's
CME Product Slate export.

## Source

- Source file: `/Users/russellfloyd/Dropbox_OLD/Mac (4)/Downloads/Product Slate Export.xlsx`
- Source format: `xlsx`
- Sheet name: `Product Slate Apr 25 2026`
- Trade date: `Apr 25 2026`
- Parsed product rows: `3,032`
- Columns: Product Name, Clearing, Globex, Floor, ClearPort, Exchange, Asset Class, Product Group, Category, Sub-Category, Cleared As, Volume, Open Interest.

The source table contains product-level volume and open interest. Official
contract specifications such as multiplier, tick size, trading hours, and
last-trade rules still need to be verified from CME contract-spec pages and
rulebooks before execution use.

## Tradability Score

```text
volume_component        = 100 * log1p(volume) / log1p(max_volume)
open_interest_component = 100 * log1p(open_interest) / log1p(max_open_interest)
futures_bonus           = 5 if Cleared As == Futures else 0
tradability_score       = min(100, 0.48*volume_component + 0.47*open_interest_component + futures_bonus)
```

Tiers: `S >= 90`, `A >= 75`, `B >= 55`, `C >= 35`, `D < 35`.

## $20M Account Tradability Screen

Products marked tradable for a `$20,000,000` portfolio pass a source-only
liquidity-capacity screen:

```text
tradability_score >= 55
volume >= 1,000
open_interest >= 10,000
capacity_contracts = min(1% of volume, 0.10% of open interest)
capacity_contracts >= 10
```

This is a liquidity proxy, not a full execution model. The source file does not
include contract notional, margin, tick value, bid/ask spread, live depth,
session liquidity, or broker limits. Use the screen to decide what belongs
in the research/trading universe, then validate execution details separately.

Pass count: `158` products (`92` futures, `66` options).

## $20M Account Tradable Products

| Rank | Tier | Score | Root | Product | Type | Exch | Asset | Group | Volume | Open Interest | Capacity Contracts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | S | 97.69 | SR3 | Three-Month SOFR Futures | Futures | CME | Interest Rate | Stirs | 3,921,027 | 12,006,141 | 12,006 |
| 2 | S | 93.64 | ZN | 10-Year T-Note Futures | Futures | CBOT | Interest Rate | US Treasury | 2,231,457 | 5,258,795 | 5,258 |
| 3 | S | 93.02 | ZF | 5-Year T-Note Futures | Futures | CBOT | Interest Rate | US Treasury | 1,526,162 | 6,500,896 | 6,500 |
| 4 | S | 90.38 | ES | E-mini S&P 500 Futures | Futures | CME | Equities | S&P | 1,865,765 | 1,964,936 | 1,964 |
| 5 | S | 90.14 | ZT | 2-Year T-Note Futures | Futures | CBOT | Interest Rate | US Treasury | 806,924 | 4,728,016 | 4,728 |
| 6 | A | 88.70 | CL | Crude Oil Futures | Futures | NYMEX | Energy | Crude Oil | 1,083,155 | 1,990,645 | 1,990 |
| 7 | A | 88.24 | SR3 | Options on Three-Month SOFR Futures | Options | CME | Interest Rate | Stirs | 462,174 | 27,885,199 | 4,621 |
| 8 | A | 87.57 | TN | Ultra 10-Year U.S. Treasury Note Futures | Futures | CBOT | Interest Rate | US Treasury | 646,850 | 2,391,001 | 2,391 |
| 9 | A | 86.46 | OZN | 10-Year T-Note Options | Options | CBOT | Interest Rate | US Treasury | 1,200,040 | 4,851,280 | 4,851 |
| 10 | A | 86.10 | ZB | U.S. Treasury Bond Futures | Futures | CBOT | Interest Rate | US Treasury | 515,320 | 1,819,011 | 1,819 |
| 11 | A | 85.99 | UB | Ultra U.S. Treasury Bond Futures | Futures | CBOT | Interest Rate | US Treasury | 406,998 | 2,292,581 | 2,292 |
| 12 | A | 85.43 | NG | Henry Hub Natural Gas Futures | Futures | NYMEX | Energy | Natural Gas | 470,798 | 1,580,834 | 1,580 |
| 13 | A | 85.30 | ZC | Corn Futures | Futures | CBOT | Agriculture | Grains | 396,077 | 1,839,147 | 1,839 |
| 14 | A | 85.23 | MNQ | Micro E-mini Nasdaq-100 Index Futures | Futures | CME | Equities | Nasdaq | 2,305,807 | 235,194 | 235 |
| 15 | A | 84.30 | MES | Micro E-mini S&P 500 Index Futures | Futures | CME | Equities | S&P | 1,813,283 | 221,146 | 221 |
| 16 | A | 84.08 | ZQ | 30 Day Federal Funds Futures | Futures | CBOT | Interest Rate | Stirs | 262,418 | 1,896,327 | 1,896 |
| 17 | A | 83.28 | SR1 | One-Month SOFR Futures | Futures | CME | Interest Rate | Stirs | 273,559 | 1,348,113 | 1,348 |
| 18 | A | 82.78 | ZS | Soybean Futures | Futures | CBOT | Agriculture | Oilseeds | 313,577 | 961,272 | 961 |
| 19 | A | 82.47 | S0 | One-Year Mid-Curve Options on Three-Month SOFR Futures | Options | CME | Interest Rate | Stirs | 276,147 | 6,147,308 | 2,761 |
| 20 | A | 81.58 | NQ | E-mini Nasdaq-100 Futures | Futures | CME | Equities | Nasdaq | 644,932 | 269,869 | 269 |
| 21 | A | 81.18 | ZL | Soybean Oil Futures | Futures | CBOT | Agriculture | Oilseeds | 236,475 | 741,901 | 741 |
| 22 | A | 80.68 | OZF | 5-Year T-Note Options | Options | CBOT | Interest Rate | US Treasury | 371,655 | 2,277,077 | 2,277 |
| 23 | A | 80.66 | 6E | Euro FX Futures | Futures | CME | FX | G10 | 189,879 | 790,730 | 790 |
| 24 | A | 80.33 | LNE | Natural Gas Option (European) | Options | NYMEX | Energy | Natural Gas | 196,281 | 4,188,134 | 1,962 |
| 25 | A | 79.54 | ZM | Soybean Meal Futures | Futures | CBOT | Agriculture | Oilseeds | 168,094 | 605,579 | 605 |
| 26 | A | 79.39 | RTY | E-mini Russell 2000 Index Futures | Futures | CME | Equities | Russell | 226,354 | 405,726 | 405 |
| 27 | A | 79.12 | LO | Crude Oil Option | Options | NYMEX | Energy | Crude Oil | 189,263 | 2,800,266 | 1,892 |
| 28 | A | 78.54 | ZW | Chicago SRW Wheat Futures | Futures | CBOT | Agriculture | Grains | 163,099 | 435,414 | 435 |
| 29 | A | 78.50 | RB | RBOB Gasoline Futures | Futures | NYMEX | Energy | Refined Products | 211,094 | 318,398 | 318 |
| 30 | A | 78.39 | BZ | Brent Last Day Financial Futures | Futures | NYMEX | Energy | Crude Oil | 241,535 | 261,580 | 261 |
| 31 | A | 78.19 | GC | Gold Futures | Futures | COMEX | Metals | Precious | 169,939 | 364,471 | 364 |
| 32 | A | 77.50 | 6J | Japanese Yen Futures | Futures | CME | FX | G10 | 139,845 | 355,319 | 355 |
| 33 | A | 76.75 | KE | KC HRW Wheat Futures | Futures | CBOT | Agriculture | Grains | 128,536 | 297,704 | 297 |
| 34 | A | 76.73 | 6A | Australian Dollar Futures | Futures | CME | FX | G10 | 135,231 | 278,860 | 278 |
| 35 | A | 76.55 | HO | NY Harbor ULSD Futures | Futures | NYMEX | Energy | Refined Products | 142,984 | 245,334 | 245 |
| 36 | A | 75.90 | MGC | Micro Gold Futures | Futures | COMEX | Metals | Precious | 416,255 | 56,263 | 56 |
| 37 | A | 75.84 | OZB | U.S. Treasury Bond Options | Options | CBOT | Interest Rate | US Treasury | 148,627 | 1,120,678 | 1,120 |
| 38 | A | 75.55 | 6B | British Pound Futures | Futures | CME | FX | G10 | 98,056 | 262,853 | 262 |
| 39 | A | 75.46 | EW4 | E-mini S&P 500 Friday Weekly Options - Week 4 | Options | CME | Equities | S&P | 219,589 | 620,979 | 620 |
| 40 | A | 75.35 | ASR | Adjusted Interest Rate S&P 500 Total Return (EFFR) Futures | Futures | CME | Equities | S&P | 32,495 | 874,441 | 324 |
| 41 | A | 75.26 | EW | E-mini S&P 500 EOM Options | Options | CME | Equities | S&P | 148,368 | 909,742 | 909 |
| 42 | A | 75.06 | HG | Copper Futures | Futures | COMEX | Metals | Base | 86,799 | 253,321 | 253 |
| 43 | B | 74.64 | HH | Natural Gas (Henry Hub) Last-day Financial Futures | Futures | NYMEX | Energy | Natural Gas | 44,525 | 468,805 | 445 |
| 44 | B | 74.61 | OZC | Corn Options | Options | CBOT | Agriculture | Grains | 67,334 | 1,780,978 | 673 |
| 45 | B | 74.29 | LE | Live Cattle Futures | Futures | CME | Agriculture | Livestock | 52,871 | 338,325 | 338 |
| 46 | B | 74.23 | 6L | Brazilian Real Futures | Futures | CME | FX | Emerging Market | 107,382 | 146,076 | 146 |
| 47 | B | 74.14 | MCL | Micro WTI Crude Oil Futures | Futures | NYMEX | Energy | Crude Oil | 340,134 | 37,500 | 37 |
| 48 | B | 74.08 | HE | Lean Hog Futures | Futures | CME | Agriculture | Livestock | 52,892 | 313,181 | 313 |
| 49 | B | 73.96 | 6C | Canadian Dollar Futures | Futures | CME | FX | G10 | 61,891 | 249,849 | 249 |
| 50 | B | 73.39 | EW3 | E-mini S&P 500 Friday Weekly Options - Week 3 | Options | CME | Equities | S&P | 47,198 | 1,719,467 | 471 |
| 51 | B | 73.35 | SI | Silver Futures | Futures | COMEX | Metals | Precious | 102,063 | 112,566 | 112 |
| 52 | B | 72.88 | 6M | Mexican Peso Futures | Futures | CME | FX | Emerging Market | 54,543 | 195,475 | 195 |
| 53 | B | 72.69 | ZN1 | 10-Year T-Note Weekly Options - Week 1 | Options | CBOT | Interest Rate | US Treasury | 169,588 | 305,213 | 305 |
| 54 | B | 72.48 | YM | E-mini Dow Jones Industrial Average Index Futures | Futures | CBOT | Equities | Dow Jones | 112,383 | 73,405 | 73 |
| 55 | B | 72.43 | ES | E-mini S&P 500 Options | Options | CME | Equities | S&P | 50,000 | 1,134,247 | 500 |
| 56 | B | 71.93 | MYM | Micro E-mini Dow Jones Industrial Average Index Futures | Futures | CBOT | Equities | Dow Jones | 192,481 | 32,227 | 32 |
| 57 | B | 71.93 | OZT | 2-Year T-Note Options | Options | CBOT | Interest Rate | US Treasury | 47,690 | 996,559 | 476 |
| 58 | B | 71.84 | OG | Gold Option | Options | COMEX | Metals | Precious | 55,116 | 817,816 | 551 |
| 59 | B | 71.67 | S2 | Two-Year Mid-Curve Options on Three-Month SOFR Futures | Options | CME | Interest Rate | Stirs | 28,920 | 1,614,130 | 289 |
| 60 | B | 71.49 | EW1 | E-mini S&P 500 Friday Weekly Options - Week 1 | Options | CME | Equities | S&P | 114,029 | 310,851 | 310 |
| 61 | B | 71.39 | HTT | WTI Houston (Argus) vs. WTI Trade Month Futures | Futures | NYMEX | Energy | Crude Oil | 16,603 | 446,384 | 166 |
| 62 | B | 71.20 | OZS | Soybean Options | Options | CBOT | Agriculture | Oilseeds | 42,529 | 872,633 | 425 |
| 63 | B | 70.80 | M2K | Micro E-mini Russell 2000 Index Futures | Futures | CME | Equities | Russell | 134,216 | 32,397 | 32 |
| 64 | B | 70.46 | MET | Micro Ether Futures | Futures | CME | Cryptocurrencies | Ether | 61,402 | 70,352 | 70 |
| 65 | B | 69.68 | 6N | New Zealand Dollar Futures | Futures | CME | FX | G10 | 41,376 | 83,423 | 83 |
| 66 | B | 69.58 | HP | Natural Gas (Henry Hub) Penultimate Financial Futures | Futures | NYMEX | Energy | Natural Gas | 15,255 | 254,290 | 152 |
| 67 | B | 69.38 | ZN2 | 10-Year T-Note Weekly Options - Week 2 | Options | CBOT | Interest Rate | US Treasury | 76,821 | 227,531 | 227 |
| 68 | B | 69.22 | OZL | Soybean Oil Options | Options | CBOT | Agriculture | Oilseeds | 33,147 | 565,040 | 331 |
| 69 | B | 68.65 | MBT | Micro Bitcoin Futures | Futures | CME | Cryptocurrencies | Bitcoin | 75,580 | 28,677 | 28 |
| 70 | B | 68.55 | 6S | Swiss Franc Futures | Futures | CME | FX | G10 | 26,346 | 93,075 | 93 |
| 71 | B | 68.54 | SIL | Micro Silver Futures | Futures | COMEX | Metals | Precious | 97,432 | 20,540 | 20 |
| 72 | B | 68.25 | OZW | Chicago SRW Wheat Options | Options | CBOT | Agriculture | Grains | 32,305 | 408,619 | 323 |
| 73 | B | 67.93 | E4A | E-mini S&P 500 Monday Weekly Options - Week 4 | Options | CME | Equities | S&P | 65,674 | 160,385 | 160 |
| 74 | B | 67.91 | EUU | EUR/USD Monthly Options | Options | CME | FX | G10 | 27,734 | 430,704 | 277 |
| 75 | B | 67.74 | LE | Live Cattle Options | Options | CME | Agriculture | Livestock | 27,171 | 414,845 | 271 |
| 76 | B | 67.65 | E4B | E-mini S&P 500 Tuesday Weekly Options - Week 4 | Options | CME | Equities | S&P | 68,803 | 137,385 | 137 |
| 77 | B | 66.97 | 1OZ | 1-Ounce Gold Futures | Futures | COMEX | Metals | Precious | 62,203 | 19,414 | 19 |
| 78 | B | 66.88 | OKE | KC HRW Wheat Options | Options | CBOT | Agriculture | Grains | 45,805 | 165,554 | 165 |
| 79 | B | 66.75 | GF | Feeder Cattle Futures | Futures | CME | Agriculture | Livestock | 19,631 | 67,788 | 67 |
| 80 | B | 66.39 | NIY | Nikkei (JPY) Futures | Futures | CME | Equities | International Indices | 30,142 | 36,281 | 36 |
| 81 | B | 65.90 | VY4 | 10-Year Treasury Note Monday Weekly Options - Week 4 | Options | CBOT | Interest Rate | US Treasury | 53,035 | 97,938 | 97 |
| 82 | B | 65.60 | AW | Bloomberg Commodity Index Futures | Futures | CBOT | Agriculture | Commodity Indices | 4,499 | 243,364 | 44 |
| 83 | B | 65.53 | ETH | Ether Futures | Futures | CME | Cryptocurrencies | Ether | 27,355 | 29,616 | 29 |
| 84 | B | 65.45 | HE | Lean Hog Options | Options | CME | Agriculture | Livestock | 14,968 | 357,353 | 149 |
| 85 | B | 65.25 | YIW | 5-Year Eris SOFR Swap Futures | Futures | CBOT | Interest Rate | Swap Futures | 4,698 | 204,131 | 46 |
| 86 | B | 65.15 | PL | Platinum Futures | Futures | NYMEX | Metals | Precious | 12,841 | 61,594 | 61 |
| 87 | B | 65.14 | SDA | S&P 500 Annual Dividend Index Futures | Futures | CME | Equities | S&P | 2,761 | 361,510 | 27 |
| 88 | B | 64.92 | S3 | Three-Year Mid-Curve Options on Three-Month SOFR Futures | Options | CME | Interest Rate | Stirs | 8,583 | 558,910 | 85 |
| 89 | B | 64.74 | EW2 | E-mini S&P 500 Friday Weekly Options - Week 2 | Options | CME | Equities | S&P | 22,771 | 169,947 | 169 |
| 90 | B | 64.56 | WTT | WTI Midland (Argus) vs. WTI Trade Month Futures | Futures | NYMEX | Energy | Crude Oil | 2,712 | 298,562 | 27 |
| 91 | B | 64.50 | BK | WTI-Brent Financial Futures | Futures | NYMEX | Energy | Crude Oil | 3,645 | 207,733 | 36 |
| 92 | B | 64.11 | ZF2 | 5-Year T-Note Weekly Options - Week 2 | Options | CBOT | Interest Rate | US Treasury | 38,900 | 72,786 | 72 |
| 93 | B | 64.07 | B0 | Mont Belvieu TET Propane (OPIS) Futures | Futures | NYMEX | Energy | Petrochemicals | 5,287 | 115,771 | 52 |
| 94 | B | 64.00 | M6E | Micro EUR/USD Futures | Futures | CME | FX | G10 | 23,147 | 20,582 | 20 |
| 95 | B | 63.73 | BTC | Bitcoin Futures | Futures | CME | Cryptocurrencies | Bitcoin | 18,324 | 24,356 | 24 |
| 96 | B | 63.27 | OZM | Soybean Meal Options | Options | CBOT | Agriculture | Oilseeds | 9,554 | 270,949 | 95 |
| 97 | B | 63.16 | EMD | E-mini S&P MidCap 400 Futures | Futures | CME | Equities | S&P | 10,439 | 37,842 | 37 |
| 98 | B | 63.12 | YIY | 10-Year Eris SOFR Swap Futures | Futures | CBOT | Interest Rate | Swap Futures | 2,130 | 233,137 | 21 |
| 99 | B | 63.11 | CSX | WTI Financial Futures | Futures | NYMEX | Energy | Crude Oil | 2,117 | 234,429 | 21 |
| 100 | B | 62.78 | E3D | E-mini S&P 500 Thursday Weekly Options - Week 3 | Options | CME | Equities | S&P | 3,539 | 712,057 | 35 |
| 101 | B | 62.44 | RX | Dow Jones Real Estate Futures | Futures | CBOT | Equities | Dow Jones | 4,867 | 70,209 | 48 |
| 102 | B | 62.31 | SDA | S&P 500 Annual Dividend Options | Options | CME | Equities | S&P | 4,180 | 495,002 | 41 |
| 103 | B | 61.94 | CU | Chicago Ethanol (Platts) Futures | Futures | NYMEX | Energy | Biofuels | 5,572 | 50,044 | 50 |
| 104 | B | 61.93 | OCD | Short-Dated New Crop Corn Options | Options | CBOT | Agriculture | Grains | 6,565 | 255,702 | 65 |
| 105 | B | 61.92 | B7A | Crude Oil Financial Calendar Spread Option 1 Month | Options | NYMEX | Energy | Crude Oil | 3,000 | 628,996 | 30 |
| 106 | B | 61.86 | WY5 | 10-Year Treasury Note Wednesday Weekly Options - Week 5 | Options | CBOT | Interest Rate | US Treasury | 19,421 | 71,446 | 71 |
| 107 | B | 61.77 | G4X | Natural Gas (Henry Hub) Last-day Financial 1 Month Spread Option | Options | NYMEX | Energy | Natural Gas | 9,250 | 162,377 | 92 |
| 108 | B | 61.68 | MHG | Micro Copper Futures | Futures | COMEX | Metals | Base | 18,319 | 11,551 | 11 |
| 109 | B | 61.57 | BZO | Brent Crude Oil Futures-Style Margin Option | Options | NYMEX | Energy | Crude Oil | 3,675 | 438,527 | 36 |
| 110 | B | 61.53 | AAO | WTI Average Price Option | Options | NYMEX | Energy | Crude Oil | 3,151 | 515,227 | 31 |
| 111 | B | 61.43 | 6Z | South African Rand Futures | Futures | CME | FX | Emerging Market | 9,323 | 22,941 | 22 |
| 112 | B | 60.88 | HXE | Copper Option | Options | COMEX | Metals | Base | 9,294 | 117,043 | 92 |
| 113 | B | 60.72 | SIC | 100-Ounce Silver Futures | Futures | COMEX | Metals | Precious | 13,249 | 11,839 | 11 |
| 114 | B | 60.63 | OTN | Ultra 10-Year U.S. Treasury Note Options | Options | CBOT | Interest Rate | US Treasury | 15,353 | 59,848 | 59 |
| 115 | B | 60.30 | ON | Natural Gas Option (American) | Options | NYMEX | Energy | Natural Gas | 9,634 | 90,867 | 90 |
| 116 | B | 60.13 | AC0 | Mont Belvieu Non-TET Ethane (OPIS) Futures | Futures | NYMEX | Energy | Petrochemicals | 3,503 | 44,214 | 35 |
| 117 | B | 59.94 | GF | Feeder Cattle Options | Options | CME | Agriculture | Livestock | 6,335 | 129,052 | 63 |
| 118 | B | 59.92 | E5C | E-mini S&P 500 Wednesday Weekly Options - Week 5 | Options | CME | Equities | S&P | 14,870 | 47,917 | 47 |
| 119 | B | 59.87 | QN4 | E-mini Nasdaq-100 Friday Weekly Options - Week 4 | Options | CME | Equities | Nasdaq | 17,512 | 38,879 | 38 |
| 120 | B | 59.83 | WAY | WTI Crude Oil 1 Month Calendar Spread Option | Options | NYMEX | Energy | Crude Oil | 3,000 | 293,500 | 30 |
| 121 | B | 59.73 | SOL | SOL Futures | Futures | CME | Cryptocurrencies | Solana | 7,030 | 17,096 | 17 |
| 122 | B | 59.61 | XAF | E-mini Financial Select Sector Futures | Futures | CME | Equities | Select Sectors | 2,005 | 69,624 | 20 |
| 123 | B | 59.42 | PA | Palladium Futures | Futures | NYMEX | Metals | Precious | 6,740 | 16,037 | 16 |
| 124 | B | 59.06 | ADU | AUD/USD Monthly Options | Options | CME | FX | G10 | 5,389 | 112,869 | 53 |
| 125 | B | 58.97 | AD0 | Mont Belvieu Non-TET Normal Butane (OPIS) Futures | Futures | NYMEX | Energy | Petrochemicals | 2,259 | 48,010 | 22 |
| 126 | B | 58.87 | SO | Silver Option | Options | COMEX | Metals | Precious | 4,794 | 120,400 | 47 |
| 127 | B | 58.65 | RP | Euro/British Pound Futures | Futures | CME | FX | Cross Rates | 2,613 | 36,188 | 26 |
| 128 | B | 58.46 | NQ | E-mini Nasdaq-100 Options | Options | CME | Equities | Nasdaq | 7,574 | 61,116 | 61 |
| 129 | B | 58.44 | ZN3 | 10-Year T-Note Weekly Options - Week 3 | Options | CBOT | Interest Rate | US Treasury | 20,204 | 19,623 | 19 |
| 130 | B | 58.41 | HRC | U.S. Midwest Domestic Hot-Rolled Coil Steel (CRU) Index Futures | Futures | COMEX | Metals | Ferrous | 2,138 | 41,747 | 21 |
| 131 | B | 58.36 | ZF1 | 5-Year T-Note Weekly Options - Week 1 | Options | CBOT | Interest Rate | US Treasury | 11,768 | 35,576 | 35 |
| 132 | B | 58.36 | XAE | E-mini Energy Select Sector Futures | Futures | CME | Equities | Select Sectors | 3,413 | 23,847 | 23 |
| 133 | B | 58.26 | ZB1 | U.S. Treasury Bond Weekly Options - Week 1 | Options | CBOT | Interest Rate | US Treasury | 8,669 | 48,675 | 48 |
| 134 | B | 57.98 | ZB2 | U.S. Treasury Bond Weekly Options - Week 2 | Options | CBOT | Interest Rate | US Treasury | 10,951 | 33,616 | 33 |
| 135 | B | 57.65 | JPU | JPY/USD Monthly Options | Options | CME | FX | G10 | 3,276 | 119,724 | 32 |
| 136 | B | 57.59 | LO4 | Crude Oil Friday Weekly Option - Week 4 | Options | NYMEX | Energy | Crude Oil | 14,394 | 21,277 | 21 |
| 137 | B | 57.26 | DC | Class III Milk Options | Options | CME | Agriculture | Dairy | 2,940 | 117,814 | 29 |
| 138 | B | 57.23 | OSD | Short-Dated New Crop Soybean Options | Options | CBOT | Agriculture | Oilseeds | 3,880 | 84,644 | 38 |
| 139 | B | 57.14 | CPO | USD Malaysian Crude Palm Oil Calendar Futures | Futures | CME | Agriculture | Oilseeds | 1,240 | 49,170 | 12 |
| 140 | B | 57.05 | VB4 | U.S. Treasury Bond Monday Weekly Options - Week 4 | Options | CBOT | Interest Rate | US Treasury | 10,302 | 25,631 | 25 |
| 141 | B | 57.01 | XK | Mini Soybean Futures | Futures | CBOT | Agriculture | Oilseeds | 1,837 | 29,761 | 18 |
| 142 | B | 56.59 | ZT1 | 2-Year T-Note Weekly Options - Week 1 | Options | CBOT | Interest Rate | US Treasury | 10,631 | 20,952 | 20 |
| 143 | B | 56.59 | A1R | Mont Belvieu Non-TET Propane (OPIS) Futures | Futures | NYMEX | Energy | Petrochemicals | 1,707 | 27,827 | 17 |
| 144 | B | 56.58 | ABV | WTI-Brent Crude Oil Spread Option | Options | NYMEX | Energy | Crude Oil | 7,500 | 31,165 | 31 |
| 145 | B | 56.50 | YIB | 7-Year Eris SOFR Swap Futures | Futures | CBOT | Interest Rate | Swap Futures | 1,265 | 38,050 | 12 |
| 146 | B | 56.42 | QNE | E-mini Nasdaq-100 EOM Options | Options | CME | Equities | Nasdaq | 4,870 | 48,340 | 48 |
| 147 | B | 56.40 | ZR | Rough Rice Futures | Futures | CBOT | Agriculture | Grains | 3,211 | 12,524 | 12 |
| 148 | B | 56.19 | GY4 | 10-Year Treasury Note Tuesday Weekly Options - Week 4 | Options | CBOT | Interest Rate | US Treasury | 10,692 | 18,005 | 18 |
| 149 | B | 55.96 | TFO | Dutch TTF Natural Gas Futures-Style Margined Calendar Month Option | Options | NYMEX | Energy | Natural Gas | 1,200 | 205,490 | 12 |
| 150 | B | 55.92 | DC | Class III Milk Futures | Futures | CME | Agriculture | Dairy | 1,377 | 27,944 | 13 |
| 151 | B | 55.86 | XAU | E-mini Utilities Select Sector Futures | Futures | CME | Equities | Select Sectors | 1,591 | 23,146 | 15 |
| 152 | B | 55.84 | Q4A | E-mini Nasdaq-100 Monday Weekly Options - Week 4 | Options | CME | Equities | Nasdaq | 12,282 | 13,484 | 13 |
| 153 | B | 55.63 | CSC | Cash-Settled Cheese Futures | Futures | CME | Agriculture | Dairy | 1,187 | 29,826 | 11 |
| 154 | B | 55.56 | E1A | E-mini S&P 500 Monday Weekly Options - Week 1 | Options | CME | Equities | S&P | 5,577 | 30,243 | 30 |
| 155 | B | 55.51 | A8K | Conway Propane (OPIS) Futures | Futures | NYMEX | Energy | Petrochemicals | 1,590 | 20,355 | 15 |
| 156 | B | 55.34 | CRB | Gulf Coast CBOB Gasoline A2 (Platts) vs. RBOB Gasoline Futures | Futures | NYMEX | Energy | Refined Products | 1,360 | 22,921 | 13 |
| 157 | B | 55.23 | XAZ | E-mini Communication Services Select Sector Futures | Futures | CME | Equities | Select Sectors | 1,592 | 18,398 | 15 |
| 158 | B | 55.00 | LBR | Lumber Futures | Futures | CME | Agriculture | Lumber and Softs | 2,490 | 10,101 | 10 |

## Top 50 Products By Tradability

| Rank | Tier | Score | Root | Product | Type | Exch | Asset | Group | Volume | Open Interest |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | S | 97.69 | SR3 | Three-Month SOFR Futures | Futures | CME | Interest Rate | Stirs | 3,921,027 | 12,006,141 |
| 2 | S | 93.64 | ZN | 10-Year T-Note Futures | Futures | CBOT | Interest Rate | US Treasury | 2,231,457 | 5,258,795 |
| 3 | S | 93.02 | ZF | 5-Year T-Note Futures | Futures | CBOT | Interest Rate | US Treasury | 1,526,162 | 6,500,896 |
| 4 | S | 90.38 | ES | E-mini S&P 500 Futures | Futures | CME | Equities | S&P | 1,865,765 | 1,964,936 |
| 5 | S | 90.14 | ZT | 2-Year T-Note Futures | Futures | CBOT | Interest Rate | US Treasury | 806,924 | 4,728,016 |
| 6 | A | 88.70 | CL | Crude Oil Futures | Futures | NYMEX | Energy | Crude Oil | 1,083,155 | 1,990,645 |
| 7 | A | 88.24 | SR3 | Options on Three-Month SOFR Futures | Options | CME | Interest Rate | Stirs | 462,174 | 27,885,199 |
| 8 | A | 87.57 | TN | Ultra 10-Year U.S. Treasury Note Futures | Futures | CBOT | Interest Rate | US Treasury | 646,850 | 2,391,001 |
| 9 | A | 86.46 | OZN | 10-Year T-Note Options | Options | CBOT | Interest Rate | US Treasury | 1,200,040 | 4,851,280 |
| 10 | A | 86.10 | ZB | U.S. Treasury Bond Futures | Futures | CBOT | Interest Rate | US Treasury | 515,320 | 1,819,011 |
| 11 | A | 85.99 | UB | Ultra U.S. Treasury Bond Futures | Futures | CBOT | Interest Rate | US Treasury | 406,998 | 2,292,581 |
| 12 | A | 85.43 | NG | Henry Hub Natural Gas Futures | Futures | NYMEX | Energy | Natural Gas | 470,798 | 1,580,834 |
| 13 | A | 85.30 | ZC | Corn Futures | Futures | CBOT | Agriculture | Grains | 396,077 | 1,839,147 |
| 14 | A | 85.23 | MNQ | Micro E-mini Nasdaq-100 Index Futures | Futures | CME | Equities | Nasdaq | 2,305,807 | 235,194 |
| 15 | A | 84.30 | MES | Micro E-mini S&P 500 Index Futures | Futures | CME | Equities | S&P | 1,813,283 | 221,146 |
| 16 | A | 84.08 | ZQ | 30 Day Federal Funds Futures | Futures | CBOT | Interest Rate | Stirs | 262,418 | 1,896,327 |
| 17 | A | 83.28 | SR1 | One-Month SOFR Futures | Futures | CME | Interest Rate | Stirs | 273,559 | 1,348,113 |
| 18 | A | 82.78 | ZS | Soybean Futures | Futures | CBOT | Agriculture | Oilseeds | 313,577 | 961,272 |
| 19 | A | 82.47 | S0 | One-Year Mid-Curve Options on Three-Month SOFR Futures | Options | CME | Interest Rate | Stirs | 276,147 | 6,147,308 |
| 20 | A | 81.58 | NQ | E-mini Nasdaq-100 Futures | Futures | CME | Equities | Nasdaq | 644,932 | 269,869 |
| 21 | A | 81.18 | ZL | Soybean Oil Futures | Futures | CBOT | Agriculture | Oilseeds | 236,475 | 741,901 |
| 22 | A | 80.68 | OZF | 5-Year T-Note Options | Options | CBOT | Interest Rate | US Treasury | 371,655 | 2,277,077 |
| 23 | A | 80.66 | 6E | Euro FX Futures | Futures | CME | FX | G10 | 189,879 | 790,730 |
| 24 | A | 80.33 | LNE | Natural Gas Option (European) | Options | NYMEX | Energy | Natural Gas | 196,281 | 4,188,134 |
| 25 | A | 79.54 | ZM | Soybean Meal Futures | Futures | CBOT | Agriculture | Oilseeds | 168,094 | 605,579 |
| 26 | A | 79.39 | RTY | E-mini Russell 2000 Index Futures | Futures | CME | Equities | Russell | 226,354 | 405,726 |
| 27 | A | 79.12 | LO | Crude Oil Option | Options | NYMEX | Energy | Crude Oil | 189,263 | 2,800,266 |
| 28 | A | 78.54 | ZW | Chicago SRW Wheat Futures | Futures | CBOT | Agriculture | Grains | 163,099 | 435,414 |
| 29 | A | 78.50 | RB | RBOB Gasoline Futures | Futures | NYMEX | Energy | Refined Products | 211,094 | 318,398 |
| 30 | A | 78.39 | BZ | Brent Last Day Financial Futures | Futures | NYMEX | Energy | Crude Oil | 241,535 | 261,580 |
| 31 | A | 78.19 | GC | Gold Futures | Futures | COMEX | Metals | Precious | 169,939 | 364,471 |
| 32 | A | 77.50 | 6J | Japanese Yen Futures | Futures | CME | FX | G10 | 139,845 | 355,319 |
| 33 | A | 76.75 | KE | KC HRW Wheat Futures | Futures | CBOT | Agriculture | Grains | 128,536 | 297,704 |
| 34 | A | 76.73 | 6A | Australian Dollar Futures | Futures | CME | FX | G10 | 135,231 | 278,860 |
| 35 | A | 76.55 | HO | NY Harbor ULSD Futures | Futures | NYMEX | Energy | Refined Products | 142,984 | 245,334 |
| 36 | A | 75.90 | MGC | Micro Gold Futures | Futures | COMEX | Metals | Precious | 416,255 | 56,263 |
| 37 | A | 75.84 | OZB | U.S. Treasury Bond Options | Options | CBOT | Interest Rate | US Treasury | 148,627 | 1,120,678 |
| 38 | A | 75.55 | 6B | British Pound Futures | Futures | CME | FX | G10 | 98,056 | 262,853 |
| 39 | A | 75.46 | EW4 | E-mini S&P 500 Friday Weekly Options - Week 4 | Options | CME | Equities | S&P | 219,589 | 620,979 |
| 40 | A | 75.35 | ASR | Adjusted Interest Rate S&P 500 Total Return (EFFR) Futures | Futures | CME | Equities | S&P | 32,495 | 874,441 |
| 41 | A | 75.26 | EW | E-mini S&P 500 EOM Options | Options | CME | Equities | S&P | 148,368 | 909,742 |
| 42 | A | 75.06 | HG | Copper Futures | Futures | COMEX | Metals | Base | 86,799 | 253,321 |
| 43 | B | 74.64 | HH | Natural Gas (Henry Hub) Last-day Financial Futures | Futures | NYMEX | Energy | Natural Gas | 44,525 | 468,805 |
| 44 | B | 74.61 | OZC | Corn Options | Options | CBOT | Agriculture | Grains | 67,334 | 1,780,978 |
| 45 | B | 74.29 | LE | Live Cattle Futures | Futures | CME | Agriculture | Livestock | 52,871 | 338,325 |
| 46 | B | 74.23 | 6L | Brazilian Real Futures | Futures | CME | FX | Emerging Market | 107,382 | 146,076 |
| 47 | B | 74.14 | MCL | Micro WTI Crude Oil Futures | Futures | NYMEX | Energy | Crude Oil | 340,134 | 37,500 |
| 48 | B | 74.08 | HE | Lean Hog Futures | Futures | CME | Agriculture | Livestock | 52,892 | 313,181 |
| 49 | B | 73.96 | 6C | Canadian Dollar Futures | Futures | CME | FX | G10 | 61,891 | 249,849 |
| 50 | B | 73.39 | EW3 | E-mini S&P 500 Friday Weekly Options - Week 3 | Options | CME | Equities | S&P | 47,198 | 1,719,467 |

## Top Futures Products

| Rank | Tier | Score | Root | Product | Type | Exch | Asset | Group | Volume | Open Interest |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | S | 97.69 | SR3 | Three-Month SOFR Futures | Futures | CME | Interest Rate | Stirs | 3,921,027 | 12,006,141 |
| 2 | S | 93.64 | ZN | 10-Year T-Note Futures | Futures | CBOT | Interest Rate | US Treasury | 2,231,457 | 5,258,795 |
| 3 | S | 93.02 | ZF | 5-Year T-Note Futures | Futures | CBOT | Interest Rate | US Treasury | 1,526,162 | 6,500,896 |
| 4 | S | 90.38 | ES | E-mini S&P 500 Futures | Futures | CME | Equities | S&P | 1,865,765 | 1,964,936 |
| 5 | S | 90.14 | ZT | 2-Year T-Note Futures | Futures | CBOT | Interest Rate | US Treasury | 806,924 | 4,728,016 |
| 6 | A | 88.70 | CL | Crude Oil Futures | Futures | NYMEX | Energy | Crude Oil | 1,083,155 | 1,990,645 |
| 7 | A | 87.57 | TN | Ultra 10-Year U.S. Treasury Note Futures | Futures | CBOT | Interest Rate | US Treasury | 646,850 | 2,391,001 |
| 8 | A | 86.10 | ZB | U.S. Treasury Bond Futures | Futures | CBOT | Interest Rate | US Treasury | 515,320 | 1,819,011 |
| 9 | A | 85.99 | UB | Ultra U.S. Treasury Bond Futures | Futures | CBOT | Interest Rate | US Treasury | 406,998 | 2,292,581 |
| 10 | A | 85.43 | NG | Henry Hub Natural Gas Futures | Futures | NYMEX | Energy | Natural Gas | 470,798 | 1,580,834 |
| 11 | A | 85.30 | ZC | Corn Futures | Futures | CBOT | Agriculture | Grains | 396,077 | 1,839,147 |
| 12 | A | 85.23 | MNQ | Micro E-mini Nasdaq-100 Index Futures | Futures | CME | Equities | Nasdaq | 2,305,807 | 235,194 |
| 13 | A | 84.30 | MES | Micro E-mini S&P 500 Index Futures | Futures | CME | Equities | S&P | 1,813,283 | 221,146 |
| 14 | A | 84.08 | ZQ | 30 Day Federal Funds Futures | Futures | CBOT | Interest Rate | Stirs | 262,418 | 1,896,327 |
| 15 | A | 83.28 | SR1 | One-Month SOFR Futures | Futures | CME | Interest Rate | Stirs | 273,559 | 1,348,113 |
| 16 | A | 82.78 | ZS | Soybean Futures | Futures | CBOT | Agriculture | Oilseeds | 313,577 | 961,272 |
| 17 | A | 81.58 | NQ | E-mini Nasdaq-100 Futures | Futures | CME | Equities | Nasdaq | 644,932 | 269,869 |
| 18 | A | 81.18 | ZL | Soybean Oil Futures | Futures | CBOT | Agriculture | Oilseeds | 236,475 | 741,901 |
| 19 | A | 80.66 | 6E | Euro FX Futures | Futures | CME | FX | G10 | 189,879 | 790,730 |
| 20 | A | 79.54 | ZM | Soybean Meal Futures | Futures | CBOT | Agriculture | Oilseeds | 168,094 | 605,579 |
| 21 | A | 79.39 | RTY | E-mini Russell 2000 Index Futures | Futures | CME | Equities | Russell | 226,354 | 405,726 |
| 22 | A | 78.54 | ZW | Chicago SRW Wheat Futures | Futures | CBOT | Agriculture | Grains | 163,099 | 435,414 |
| 23 | A | 78.50 | RB | RBOB Gasoline Futures | Futures | NYMEX | Energy | Refined Products | 211,094 | 318,398 |
| 24 | A | 78.39 | BZ | Brent Last Day Financial Futures | Futures | NYMEX | Energy | Crude Oil | 241,535 | 261,580 |
| 25 | A | 78.19 | GC | Gold Futures | Futures | COMEX | Metals | Precious | 169,939 | 364,471 |
| 26 | A | 77.50 | 6J | Japanese Yen Futures | Futures | CME | FX | G10 | 139,845 | 355,319 |
| 27 | A | 76.75 | KE | KC HRW Wheat Futures | Futures | CBOT | Agriculture | Grains | 128,536 | 297,704 |
| 28 | A | 76.73 | 6A | Australian Dollar Futures | Futures | CME | FX | G10 | 135,231 | 278,860 |
| 29 | A | 76.55 | HO | NY Harbor ULSD Futures | Futures | NYMEX | Energy | Refined Products | 142,984 | 245,334 |
| 30 | A | 75.90 | MGC | Micro Gold Futures | Futures | COMEX | Metals | Precious | 416,255 | 56,263 |
| 31 | A | 75.55 | 6B | British Pound Futures | Futures | CME | FX | G10 | 98,056 | 262,853 |
| 32 | A | 75.35 | ASR | Adjusted Interest Rate S&P 500 Total Return (EFFR) Futures | Futures | CME | Equities | S&P | 32,495 | 874,441 |
| 33 | A | 75.06 | HG | Copper Futures | Futures | COMEX | Metals | Base | 86,799 | 253,321 |
| 34 | B | 74.64 | HH | Natural Gas (Henry Hub) Last-day Financial Futures | Futures | NYMEX | Energy | Natural Gas | 44,525 | 468,805 |
| 35 | B | 74.29 | LE | Live Cattle Futures | Futures | CME | Agriculture | Livestock | 52,871 | 338,325 |
| 36 | B | 74.23 | 6L | Brazilian Real Futures | Futures | CME | FX | Emerging Market | 107,382 | 146,076 |
| 37 | B | 74.14 | MCL | Micro WTI Crude Oil Futures | Futures | NYMEX | Energy | Crude Oil | 340,134 | 37,500 |
| 38 | B | 74.08 | HE | Lean Hog Futures | Futures | CME | Agriculture | Livestock | 52,892 | 313,181 |
| 39 | B | 73.96 | 6C | Canadian Dollar Futures | Futures | CME | FX | G10 | 61,891 | 249,849 |
| 40 | B | 73.35 | SI | Silver Futures | Futures | COMEX | Metals | Precious | 102,063 | 112,566 |

## Top Options Products

| Rank | Tier | Score | Root | Product | Type | Exch | Asset | Group | Volume | Open Interest |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | 88.24 | SR3 | Options on Three-Month SOFR Futures | Options | CME | Interest Rate | Stirs | 462,174 | 27,885,199 |
| 2 | A | 86.46 | OZN | 10-Year T-Note Options | Options | CBOT | Interest Rate | US Treasury | 1,200,040 | 4,851,280 |
| 3 | A | 82.47 | S0 | One-Year Mid-Curve Options on Three-Month SOFR Futures | Options | CME | Interest Rate | Stirs | 276,147 | 6,147,308 |
| 4 | A | 80.68 | OZF | 5-Year T-Note Options | Options | CBOT | Interest Rate | US Treasury | 371,655 | 2,277,077 |
| 5 | A | 80.33 | LNE | Natural Gas Option (European) | Options | NYMEX | Energy | Natural Gas | 196,281 | 4,188,134 |
| 6 | A | 79.12 | LO | Crude Oil Option | Options | NYMEX | Energy | Crude Oil | 189,263 | 2,800,266 |
| 7 | A | 75.84 | OZB | U.S. Treasury Bond Options | Options | CBOT | Interest Rate | US Treasury | 148,627 | 1,120,678 |
| 8 | A | 75.46 | EW4 | E-mini S&P 500 Friday Weekly Options - Week 4 | Options | CME | Equities | S&P | 219,589 | 620,979 |
| 9 | A | 75.26 | EW | E-mini S&P 500 EOM Options | Options | CME | Equities | S&P | 148,368 | 909,742 |
| 10 | B | 74.61 | OZC | Corn Options | Options | CBOT | Agriculture | Grains | 67,334 | 1,780,978 |
| 11 | B | 73.39 | EW3 | E-mini S&P 500 Friday Weekly Options - Week 3 | Options | CME | Equities | S&P | 47,198 | 1,719,467 |
| 12 | B | 72.69 | ZN1 | 10-Year T-Note Weekly Options - Week 1 | Options | CBOT | Interest Rate | US Treasury | 169,588 | 305,213 |
| 13 | B | 72.43 | ES | E-mini S&P 500 Options | Options | CME | Equities | S&P | 50,000 | 1,134,247 |
| 14 | B | 71.93 | OZT | 2-Year T-Note Options | Options | CBOT | Interest Rate | US Treasury | 47,690 | 996,559 |
| 15 | B | 71.84 | OG | Gold Option | Options | COMEX | Metals | Precious | 55,116 | 817,816 |
| 16 | B | 71.67 | S2 | Two-Year Mid-Curve Options on Three-Month SOFR Futures | Options | CME | Interest Rate | Stirs | 28,920 | 1,614,130 |
| 17 | B | 71.49 | EW1 | E-mini S&P 500 Friday Weekly Options - Week 1 | Options | CME | Equities | S&P | 114,029 | 310,851 |
| 18 | B | 71.20 | OZS | Soybean Options | Options | CBOT | Agriculture | Oilseeds | 42,529 | 872,633 |
| 19 | B | 69.38 | ZN2 | 10-Year T-Note Weekly Options - Week 2 | Options | CBOT | Interest Rate | US Treasury | 76,821 | 227,531 |
| 20 | B | 69.22 | OZL | Soybean Oil Options | Options | CBOT | Agriculture | Oilseeds | 33,147 | 565,040 |
| 21 | B | 68.25 | OZW | Chicago SRW Wheat Options | Options | CBOT | Agriculture | Grains | 32,305 | 408,619 |
| 22 | B | 67.93 | E4A | E-mini S&P 500 Monday Weekly Options - Week 4 | Options | CME | Equities | S&P | 65,674 | 160,385 |
| 23 | B | 67.91 | EUU | EUR/USD Monthly Options | Options | CME | FX | G10 | 27,734 | 430,704 |
| 24 | B | 67.74 | LE | Live Cattle Options | Options | CME | Agriculture | Livestock | 27,171 | 414,845 |
| 25 | B | 67.65 | E4B | E-mini S&P 500 Tuesday Weekly Options - Week 4 | Options | CME | Equities | S&P | 68,803 | 137,385 |
| 26 | B | 66.88 | OKE | KC HRW Wheat Options | Options | CBOT | Agriculture | Grains | 45,805 | 165,554 |
| 27 | B | 65.90 | VY4 | 10-Year Treasury Note Monday Weekly Options - Week 4 | Options | CBOT | Interest Rate | US Treasury | 53,035 | 97,938 |
| 28 | B | 65.45 | HE | Lean Hog Options | Options | CME | Agriculture | Livestock | 14,968 | 357,353 |
| 29 | B | 64.92 | S3 | Three-Year Mid-Curve Options on Three-Month SOFR Futures | Options | CME | Interest Rate | Stirs | 8,583 | 558,910 |
| 30 | B | 64.74 | EW2 | E-mini S&P 500 Friday Weekly Options - Week 2 | Options | CME | Equities | S&P | 22,771 | 169,947 |
| 31 | B | 64.11 | ZF2 | 5-Year T-Note Weekly Options - Week 2 | Options | CBOT | Interest Rate | US Treasury | 38,900 | 72,786 |
| 32 | B | 63.27 | OZM | Soybean Meal Options | Options | CBOT | Agriculture | Oilseeds | 9,554 | 270,949 |
| 33 | B | 62.78 | E3D | E-mini S&P 500 Thursday Weekly Options - Week 3 | Options | CME | Equities | S&P | 3,539 | 712,057 |
| 34 | B | 62.31 | SDA | S&P 500 Annual Dividend Options | Options | CME | Equities | S&P | 4,180 | 495,002 |
| 35 | B | 61.93 | OCD | Short-Dated New Crop Corn Options | Options | CBOT | Agriculture | Grains | 6,565 | 255,702 |
| 36 | B | 61.92 | B7A | Crude Oil Financial Calendar Spread Option 1 Month | Options | NYMEX | Energy | Crude Oil | 3,000 | 628,996 |
| 37 | B | 61.86 | WY5 | 10-Year Treasury Note Wednesday Weekly Options - Week 5 | Options | CBOT | Interest Rate | US Treasury | 19,421 | 71,446 |
| 38 | B | 61.77 | G4X | Natural Gas (Henry Hub) Last-day Financial 1 Month Spread Option | Options | NYMEX | Energy | Natural Gas | 9,250 | 162,377 |
| 39 | B | 61.57 | BZO | Brent Crude Oil Futures-Style Margin Option | Options | NYMEX | Energy | Crude Oil | 3,675 | 438,527 |
| 40 | B | 61.53 | AAO | WTI Average Price Option | Options | NYMEX | Energy | Crude Oil | 3,151 | 515,227 |

## Asset Class Summary

| Asset Class | Products | Futures Products | Total Volume | Total Open Interest | Top Futures Roots |
| --- | --- | --- | --- | --- | --- |
| Interest Rate | 264 | 75 | 13,712,746 | 85,610,860 | SR3, ZN, ZF, ZT, TN, ZB, UB, ZQ |
| Equities | 287 | 139 | 8,830,082 | 12,134,711 | ES, MNQ, MES, NQ, RTY, ASR, YM, MYM |
| Energy | 1041 | 773 | 3,108,199 | 17,639,638 | CL, NG, RB, BZ, HO, HH, MCL, HTT |
| Agriculture | 339 | 100 | 1,869,667 | 11,886,890 | ZC, ZS, ZL, ZM, ZW, KE, LE, HE |
| Metals | 178 | 60 | 1,085,032 | 2,217,265 | GC, MGC, HG, SI, SIL, 1OZ, PL, MHG |
| FX | 288 | 63 | 970,945 | 3,572,793 | 6E, 6J, 6A, 6B, 6L, 6C, 6M, 6N |
| Cryptocurrencies | 266 | 65 | 293,750 | 251,772 | MET, MBT, ETH, BTC, SOL, QXF, QTF, XRP |
| Weather | 358 | 179 | 0 | 62,150 | K6, K3K, K4, G2, K6K, D2, G2K, G1K |
| Real Estate | 11 | 11 | 0 | 37 | CUS, WDC, MIA, LAX, SFR, BOS, NYM, LAV |

## Recommended Futures Universe

The default tradable futures universe should start from the highest-scoring
futures roots in each asset class, then apply desk-specific exclusions for
calendar spreads, financial swaps, or products without reliable screen depth.

| Asset Class | Candidate Futures Roots |
| --- | --- |
| Agriculture | ZC, ZS, ZL, ZM, ZW, KE, LE, HE, GF, AW, CPO, XK |
| Cryptocurrencies | MET, MBT, ETH, BTC, SOL, QXF, QTF, XRP, MSL, MXP, BFF, QEF |
| Energy | CL, NG, RB, BZ, HO, HH, MCL, HTT, HP, WTT, BK, B0 |
| Equities | ES, MNQ, MES, NQ, RTY, ASR, YM, MYM, M2K, NIY, SDA, EMD |
| FX | 6E, 6J, 6A, 6B, 6L, 6C, 6M, 6N, 6S, M6E, 6Z, RP |
| Interest Rate | SR3, ZN, ZF, ZT, TN, ZB, UB, ZQ, SR1, YIW, YIY, YIT |
| Metals | GC, MGC, HG, SI, SIL, 1OZ, PL, MHG, SIC, PA, HRC, QO |
| Real Estate | CUS, WDC, MIA, LAX, SFR, BOS, NYM, LAV, DEN, CHI, SDG |
| Weather | K6, K3K, K4, G2, K6K, D2, G2K, G1K, 3G6, G0, G0K, G2N |

## Maintenance Notes

- Rebuild this file whenever a fresh CME product slate export is available.
- Keep the JSON artifact under `data/reference/` as the machine-readable source for downstream tooling.
- Do not scrape CME's live product slate page. Use an exported file, licensed CME APIs, or official reports.
- Treat options rows as evidence of product-family liquidity; model options as tenors/chains under the futures root rather than standalone directional roots.

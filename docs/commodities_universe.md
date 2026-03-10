# Commodities Volatility — Core Universe

Reference contracts for the commodities volatility strategy.
To be used when building the database and live data feeds.

## Contracts

| Commodity       | Exchange  | Ticker | Sector          | Weight    |
|-----------------|-----------|--------|-----------------|-----------|
| WTI Crude Oil   | NYMEX     | CL     | Petroleum       | 0.0000%   |
| Brent Crude Oil | ICE – EU  | CO     | Petroleum       | 6.7582%   |
| Heating Oil     | NYMEX     | HO     | Petroleum       | 5.9942%   |
| GasOil          | ICE – EU  | GO     | Petroleum       | 7.8035%   |
| RBOB Gasoline   | NYMEX     | RB     | Petroleum       | 8.3314%   |
| Natural Gas     | NYMEX     | NG     | Natural Gas     | 0.0000%   |
| Corn            | CBT       | CN     | Grains          | 2.0619%   |
| Wheat           | CBT       | WH     | Grains          | 2.0000%   |
| KC Wheat        | KBT       | KW     | Grains          | 2.0000%   |
| Soybeans        | CBT       | SY     | Grains          | 2.6403%   |
| Soymeal         | CBT       | SM     | Grains          | 2.0000%   |
| Soybean Oil     | CBT       | BO     | Grains          | 2.0000%   |
| Cocoa           | ICE – US  | CC     | Softs           | 2.8330%   |
| Cotton          | ICE – US  | CT     | Softs           | 2.0000%   |
| Coffee          | ICE – US  | KC     | Softs           | 2.0000%   |
| Sugar           | ICE – US  | SB     | Softs           | 2.1929%   |
| Live Cattle     | CME       | LC     | Livestock       | 0.0000%   |
| Lean Hogs       | CME       | LH     | Livestock       | 2.0000%   |
| Copper          | COMEX     | HG     | Base Metals     | 2.0000%   |
| Aluminum        | LME       | AL     | Base Metals     | 2.0000%   |
| Nickel          | LME       | NI     | Base Metals     | 2.0000%   |
| Zinc            | LME       | ZS     | Base Metals     | 2.0000%   |
| Gold            | COMEX     | GC     | Precious Metals | 39.3846%  |
| Silver          | COMEX     | SI     | Precious Metals | 0.0000%   |

## Sector Summary

| Sector          | Contracts | Total Weight |
|-----------------|-----------|--------------|
| Petroleum       | 5         | 28.89%       |
| Natural Gas     | 1         | 0.00%        |
| Grains          | 6         | 12.70%       |
| Softs           | 4         | 9.03%        |
| Livestock       | 2         | 2.00%        |
| Base Metals     | 4         | 8.00%        |
| Precious Metals | 2         | 39.38%       |

## Notes

- 24 contracts across 7 sectors and 6 exchanges (NYMEX, ICE-EU, ICE-US, CBT/KBT, CME/COMEX, LME)
- Weights are from the source allocation; Gold dominates at ~39%
- WTI (CL), Natural Gas (NG), Live Cattle (LC), and Silver (SI) carry 0% weight but are included for tracking/hedging
- This universe will be referenced when building: (1) the historical database, (2) live market data feeds, (3) options/vol surface infrastructure

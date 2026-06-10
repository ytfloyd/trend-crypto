#!/usr/bin/env python3
"""Write a single-sheet screen of underlyings with tradable CME options."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.reference.write_cme_tradable_universe_excel import (
    _content_types,
    _doc_props,
    _root_rels,
    _sheet_xml,
    _styles_xml,
    _workbook_rels,
    _workbook_xml,
)

DEFAULT_JSON = Path("data/reference/cme_products_20260425.json")
DEFAULT_XLSX = Path("artifacts/reference/cme_options_underlying_screen_capacity_1_20260425.xlsx")

CONTRACT_DETAILS = {
    "SR3": ("Three-month SOFR futures/options referencing compounded SOFR over an IMM quarterly period.", "$2,500 x IMM Index", "100 minus annualized SOFR rate", "0.0025 index points = $6.25", "Cash settled; verify serial/quarterly tick rules on CME specs.", "CME SOFR futures/options contract specs"),
    "ES": ("E-mini S&P 500 index futures/options for large-cap U.S. equity beta and index convexity.", "$50 x S&P 500 Index", "Index points", "0.25 index points = $12.50", "Cash settled to S&P 500 index futures.", "CME equity index contract specs"),
    "NQ": ("E-mini Nasdaq-100 futures/options for large-cap growth and technology index exposure.", "$20 x Nasdaq-100 Index", "Index points", "0.25 index points = $5.00", "Cash settled to Nasdaq-100 index futures.", "CME equity index contract specs"),
    "RTY": ("E-mini Russell 2000 futures/options for U.S. small-cap equity exposure.", "$50 x Russell 2000 Index", "Index points", "0.10 index points = $5.00", "Cash settled to Russell 2000 index futures.", "CME equity index contract specs"),
    "SDA": ("S&P 500 Annual Dividend options for convexity on realized annual S&P 500 dividends.", "Dividend index exposure", "Index points", "Varies by product spec", "Cash-settled dividend index product; verify multiplier before trading.", "CME S&P dividend contract specs"),
    "SME": ("S&P 500 month-end options for event/expiry convexity around month-end equity index exposure.", "S&P 500 index option exposure", "Index points", "Varies by product spec", "Cash settled; use as equity-index convexity sleeve after contract validation.", "CME S&P options contract specs"),
    "CL": ("WTI Light Sweet Crude Oil futures/options, the main U.S. crude oil benchmark.", "1,000 barrels", "U.S. dollars per barrel", "$0.01/bbl = $10.00", "Physically deliverable futures; options exercise into futures.", "CME WTI crude oil contract specs"),
    "NG": ("Henry Hub Natural Gas futures/options for U.S. natural gas benchmark exposure.", "10,000 MMBtu", "U.S. dollars per MMBtu", "$0.001/MMBtu = $10.00", "Physically deliverable futures; options exercise into futures.", "CME Henry Hub natural gas contract specs"),
    "RB": ("RBOB Gasoline futures/options for U.S. gasoline/refined-products exposure.", "42,000 gallons", "U.S. dollars per gallon", "$0.0001/gal = $4.20", "Physically deliverable futures; options exercise into futures.", "CME RBOB gasoline contract specs"),
    "HO": ("NY Harbor ULSD futures/options for diesel/heating-oil refined-products exposure.", "42,000 gallons", "U.S. dollars per gallon", "$0.0001/gal = $4.20", "Physically deliverable futures; options exercise into futures.", "CME NY Harbor ULSD contract specs"),
    "GC": ("Gold futures/options for benchmark precious-metals exposure.", "100 troy ounces", "U.S. dollars per troy ounce", "$0.10/oz = $10.00", "Physically deliverable futures; options exercise into futures.", "CME Gold contract specs"),
    "SI": ("Silver futures/options for benchmark silver exposure.", "5,000 troy ounces", "U.S. dollars per troy ounce", "$0.005/oz = $25.00", "Physically deliverable futures; options exercise into futures.", "CME Silver contract specs"),
    "HG": ("Copper futures/options for benchmark COMEX copper exposure.", "25,000 pounds", "U.S. dollars per pound", "$0.0005/lb = $12.50", "Physically deliverable futures; options exercise into futures.", "CME Copper contract specs"),
    "PO": ("Platinum options on NYMEX platinum futures.", "50 troy ounces", "U.S. dollars per troy ounce", "$0.10/oz = $5.00", "Physically deliverable futures; options exercise into futures.", "CME Platinum contract specs"),
    "PAO": ("Palladium options on NYMEX palladium futures.", "100 troy ounces", "U.S. dollars per troy ounce", "$0.10/oz = $10.00", "Physically deliverable futures; options exercise into futures.", "CME Palladium contract specs"),
    "ZN": ("10-Year Treasury Note futures/options for intermediate U.S. duration exposure.", "$100,000 face value", "Points and fractions of points", "1/2 of 1/32 = $15.625", "Deliverable Treasury-note basket; options exercise into futures.", "CME U.S. Treasury contract specs"),
    "ZF": ("5-Year Treasury Note futures/options for belly-of-curve U.S. duration exposure.", "$100,000 face value", "Points and fractions of points", "1/4 of 1/32 = $7.8125", "Deliverable Treasury-note basket; options exercise into futures.", "CME U.S. Treasury contract specs"),
    "ZT": ("2-Year Treasury Note futures/options for front-end U.S. rates exposure.", "$200,000 face value", "Points and fractions of points", "1/8 of 1/32 = $7.8125", "Deliverable Treasury-note basket; options exercise into futures.", "CME U.S. Treasury contract specs"),
    "ZB": ("U.S. Treasury Bond futures/options for long-duration U.S. rates exposure.", "$100,000 face value", "Points and fractions of points", "1/32 = $31.25", "Deliverable Treasury-bond basket; options exercise into futures.", "CME U.S. Treasury bond contract specs"),
    "TN": ("Ultra 10-Year Treasury Note futures/options for longer-duration 10-year sector exposure.", "$100,000 face value", "Points and fractions of points", "1/32 = $31.25", "Deliverable Treasury-note basket; options exercise into futures.", "CME Ultra 10-Year Treasury contract specs"),
    "UB": ("Ultra Treasury Bond futures/options for ultra-long U.S. duration exposure.", "$100,000 face value", "Points and fractions of points", "1/32 = $31.25", "Deliverable Treasury-bond basket; options exercise into futures.", "CME Ultra Treasury bond contract specs"),
    "6E": ("Euro FX futures/options for EUR/USD exposure.", "125,000 euro", "U.S. dollars per euro", "$0.00005/EUR = $6.25", "Physically deliverable FX futures; options exercise into futures.", "CME FX contract specs"),
    "6J": ("Japanese Yen futures/options for JPY/USD exposure.", "12,500,000 yen", "U.S. dollars per yen", "$0.0000005/JPY = $6.25", "Physically deliverable FX futures; options exercise into futures.", "CME FX contract specs"),
    "6A": ("Australian Dollar futures/options for AUD/USD exposure.", "100,000 Australian dollars", "U.S. dollars per AUD", "$0.00005/AUD = $5.00", "Physically deliverable FX futures; options exercise into futures.", "CME FX contract specs"),
    "6B": ("British Pound futures/options for GBP/USD exposure.", "62,500 British pounds", "U.S. dollars per GBP", "$0.0001/GBP = $6.25", "Physically deliverable FX futures; options exercise into futures.", "CME FX contract specs"),
    "6C": ("Canadian Dollar futures/options for CAD/USD exposure.", "100,000 Canadian dollars", "U.S. dollars per CAD", "$0.00005/CAD = $5.00", "Physically deliverable FX futures; options exercise into futures.", "CME FX contract specs"),
    "6S": ("Swiss Franc futures/options for CHF/USD exposure.", "125,000 Swiss francs", "U.S. dollars per CHF", "$0.0001/CHF = $12.50", "Physically deliverable FX futures; options exercise into futures.", "CME FX contract specs"),
    "ZC": ("Corn futures/options for U.S. corn price exposure.", "5,000 bushels", "Cents per bushel", "1/4 cent/bu = $12.50", "Physically deliverable futures; options exercise into futures.", "CME Corn contract specs"),
    "ZS": ("Soybean futures/options for U.S. soybean price exposure.", "5,000 bushels", "Cents per bushel", "1/4 cent/bu = $12.50", "Physically deliverable futures; options exercise into futures.", "CME Soybean contract specs"),
    "ZW": ("Chicago SRW Wheat futures/options for soft red winter wheat exposure.", "5,000 bushels", "Cents per bushel", "1/4 cent/bu = $12.50", "Physically deliverable futures; options exercise into futures.", "CME Wheat contract specs"),
    "KE": ("KC HRW Wheat futures/options for hard red winter wheat exposure.", "5,000 bushels", "Cents per bushel", "1/4 cent/bu = $12.50", "Physically deliverable futures; options exercise into futures.", "CME KC HRW Wheat contract specs"),
    "ZL": ("Soybean Oil futures/options for vegetable-oil exposure.", "60,000 pounds", "Cents per pound", "0.01 cent/lb = $6.00", "Physically deliverable futures; options exercise into futures.", "CME Soybean Oil contract specs"),
    "ZM": ("Soybean Meal futures/options for soybean crush/meal exposure.", "100 short tons", "U.S. dollars per short ton", "$0.10/ton = $10.00", "Physically deliverable futures; options exercise into futures.", "CME Soybean Meal contract specs"),
    "LE": ("Live Cattle futures/options for fed-cattle price exposure.", "40,000 pounds", "Cents per pound", "0.025 cent/lb = $10.00", "Physically deliverable futures; options exercise into futures.", "CME Live Cattle contract specs"),
    "GF": ("Feeder Cattle futures/options for feeder-cattle price exposure.", "50,000 pounds", "Cents per pound", "0.025 cent/lb = $12.50", "Cash settled to CME Feeder Cattle Index.", "CME Feeder Cattle contract specs"),
    "HE": ("Lean Hog futures/options for hog price exposure.", "40,000 pounds", "Cents per pound", "0.025 cent/lb = $10.00", "Cash settled to CME Lean Hog Index.", "CME Lean Hog contract specs"),
    "DC": ("Class III Milk futures/options for cheese-milk price exposure.", "200,000 pounds of milk", "Dollars per hundredweight", "$0.01/cwt = $20.00", "Cash settled against USDA monthly Class III milk price.", "CME Dairy futures/options fact card"),
    "GDK": ("Class IV Milk options for butter/powder milk price exposure.", "200,000 pounds of milk", "Dollars per hundredweight", "$0.01/cwt = $20.00", "Cash settled against USDA monthly Class IV milk price.", "CME Dairy futures/options fact card"),
    "CSC": ("Cash-settled Cheese options for CME cheese price exposure.", "20,000 pounds", "Dollars per pound", "$0.00025/lb = $5.00", "Cash settled against USDA/CME cheese reference pricing.", "CME Dairy futures/options fact card"),
    "GNF": ("Nonfat Dry Milk options for dairy powder price exposure.", "44,000 pounds", "Dollars per pound", "$0.00025/lb = $11.00", "Cash settled against USDA nonfat dry milk price.", "CME Dairy futures/options fact card"),
    "CB": ("Cash-settled Butter options for butter price exposure.", "20,000 pounds", "Dollars per pound", "$0.00025/lb = $5.00", "Cash settled against USDA/CME butter reference pricing.", "CME Dairy futures/options fact card"),
    "DY": ("Dry Whey options for whey price exposure.", "44,000 pounds", "Dollars per pound", "$0.00025/lb = $11.00", "Cash settled against USDA dry whey price.", "CME Dairy futures/options fact card"),
    "VM": ("Options on Micro Ether futures for listed ETH convexity.", "0.1 ether futures exposure", "U.S. dollars per ether", "Varies by option premium increment", "Cash settled to CME CF Ether-Dollar Reference Rate exposure via Micro Ether futures.", "CME Micro Ether futures/options specs"),
    "WM": ("Options on Micro Bitcoin futures for listed BTC convexity.", "0.1 bitcoin futures exposure", "U.S. dollars per bitcoin", "Varies by option premium increment", "Cash settled to CME CF Bitcoin Reference Rate exposure via Micro Bitcoin futures.", "CME Micro Bitcoin futures/options specs"),
}

RULEBOOK_VERIFICATION = {
    "SR3": ("CME", "https://www.cmegroup.com/markets/interest-rates/stirs/three-month-sofr.contractSpecs.html", "$2,500 x IMM Index", "0.0025 index points = $6.25", "Cash settled to compounded daily SOFR over the reference quarter.", "CME contract-spec/rulebook reference reviewed; verify live chapter before trading."),
    "ES": ("CME", "https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.contractSpecs.html", "$50 x S&P 500 Index", "0.25 index points = $12.50", "Cash settled equity-index futures; options exercise into futures unless otherwise specified.", "CME equity-index spec/rulebook reference reviewed."),
    "NQ": ("CME", "https://www.cmegroup.com/markets/equities/nasdaq/e-mini-nasdaq-100.contractSpecs.html", "$20 x Nasdaq-100 Index", "0.25 index points = $5.00", "Cash settled equity-index futures; options exercise into futures unless otherwise specified.", "CME equity-index spec/rulebook reference reviewed."),
    "RTY": ("CME", "https://www.cmegroup.com/markets/equities/russell/e-mini-russell-2000.contractSpecs.html", "$50 x Russell 2000 Index", "0.10 index points = $5.00", "Cash settled equity-index futures; options exercise into futures unless otherwise specified.", "CME Russell spec/rulebook reference reviewed."),
    "SDA": ("CME", "https://www.cmegroup.com/markets/equities/sp/sandp-500-annual-dividend-index.contractSpecs.html", "S&P 500 annual dividend index exposure", "See CME product specs", "Cash-settled dividend-index product.", "Official CME product specs required for final multiplier/tick verification."),
    "SME": ("CME", "https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.contractSpecs.options.html", "S&P 500 month-end option exposure", "See CME option specs", "Cash-settled/equity-index option family; verify exercise and expiry treatment.", "Official CME option specs required for final multiplier/tick verification."),
    "CL": ("NYMEX", "https://www.cmegroup.com/markets/energy/crude-oil/light-sweet-crude.contractSpecs.html", "1,000 barrels", "$0.01/bbl = $10.00", "Physically deliverable WTI crude oil futures; options exercise into futures.", "NYMEX WTI spec/rulebook reference reviewed."),
    "NG": ("NYMEX", "https://www.cmegroup.com/markets/energy/natural-gas/natural-gas.contractSpecs.html", "10,000 MMBtu", "$0.001/MMBtu = $10.00", "Physically deliverable Henry Hub natural gas futures; options exercise into futures.", "NYMEX natural gas spec/rulebook reference reviewed."),
    "RB": ("NYMEX", "https://www.cmegroup.com/markets/energy/refined-products/rbob-gasoline.contractSpecs.html", "42,000 gallons", "$0.0001/gal = $4.20", "Physically deliverable RBOB gasoline futures; options exercise into futures.", "NYMEX RBOB spec/rulebook reference reviewed."),
    "HO": ("NYMEX", "https://www.cmegroup.com/markets/energy/refined-products/heating-oil.contractSpecs.html", "42,000 gallons", "$0.0001/gal = $4.20", "Physically deliverable NY Harbor ULSD futures; options exercise into futures.", "NYMEX ULSD spec/rulebook reference reviewed."),
    "GC": ("COMEX", "https://www.cmegroup.com/markets/metals/precious/gold.contractSpecs.html", "100 troy ounces", "$0.10/oz = $10.00", "Physically deliverable gold futures; options exercise into futures.", "COMEX gold spec/rulebook reference reviewed."),
    "SI": ("COMEX", "https://www.cmegroup.com/markets/metals/precious/silver.contractSpecs.html", "5,000 troy ounces", "$0.005/oz = $25.00", "Physically deliverable silver futures; options exercise into futures.", "COMEX silver spec/rulebook reference reviewed."),
    "HG": ("COMEX", "https://www.cmegroup.com/markets/metals/base/copper.contractSpecs.html", "25,000 pounds", "$0.0005/lb = $12.50", "Physically deliverable copper futures; options exercise into futures.", "COMEX copper spec/rulebook reference reviewed."),
    "PO": ("NYMEX", "https://www.cmegroup.com/markets/metals/precious/platinum.contractSpecs.html", "50 troy ounces", "$0.10/oz = $5.00", "Physically deliverable platinum futures; options exercise into futures.", "NYMEX platinum spec/rulebook reference reviewed."),
    "PAO": ("NYMEX", "https://www.cmegroup.com/markets/metals/precious/palladium.contractSpecs.html", "100 troy ounces", "$0.10/oz = $10.00", "Physically deliverable palladium futures; options exercise into futures.", "NYMEX palladium spec/rulebook reference reviewed."),
    "ZN": ("CBOT", "https://www.cmegroup.com/markets/interest-rates/us-treasury/10-year-us-treasury-note.contractSpecs.html", "$100,000 face value", "1/2 of 1/32 = $15.625", "Deliverable Treasury-note basket; options exercise into futures.", "CBOT Treasury spec/rulebook reference reviewed."),
    "ZF": ("CBOT", "https://www.cmegroup.com/markets/interest-rates/us-treasury/5-year-us-treasury-note.contractSpecs.html", "$100,000 face value", "1/4 of 1/32 = $7.8125", "Deliverable Treasury-note basket; options exercise into futures.", "CBOT Treasury spec/rulebook reference reviewed."),
    "ZT": ("CBOT", "https://www.cmegroup.com/markets/interest-rates/us-treasury/2-year-us-treasury-note.contractSpecs.html", "$200,000 face value", "1/8 of 1/32 = $7.8125", "Deliverable Treasury-note basket; options exercise into futures.", "CBOT Treasury spec/rulebook reference reviewed."),
    "ZB": ("CBOT", "https://www.cmegroup.com/markets/interest-rates/us-treasury/30-year-us-treasury-bond.contractSpecs.html", "$100,000 face value", "1/32 = $31.25", "Deliverable Treasury-bond basket; options exercise into futures.", "CBOT Treasury bond spec/rulebook reference reviewed."),
    "TN": ("CBOT", "https://www.cmegroup.com/markets/interest-rates/us-treasury/ultra-10-year-us-treasury-note.contractSpecs.html", "$100,000 face value", "1/32 = $31.25", "Deliverable Treasury-note basket; options exercise into futures.", "CBOT Ultra 10-Year spec/rulebook reference reviewed."),
    "UB": ("CBOT", "https://www.cmegroup.com/markets/interest-rates/us-treasury/ultra-t-bond.contractSpecs.html", "$100,000 face value", "1/32 = $31.25", "Deliverable Treasury-bond basket; options exercise into futures.", "CBOT Ultra Bond spec/rulebook reference reviewed."),
    "6E": ("CME", "https://www.cmegroup.com/markets/fx/g10/euro-fx.contractSpecs.html", "125,000 euro", "$0.00005/EUR = $6.25", "Physically deliverable FX futures; options exercise into futures.", "CME FX spec/rulebook reference reviewed."),
    "6J": ("CME", "https://www.cmegroup.com/markets/fx/g10/japanese-yen.contractSpecs.html", "12,500,000 yen", "$0.0000005/JPY = $6.25", "Physically deliverable FX futures; options exercise into futures.", "CME FX spec/rulebook reference reviewed."),
    "6A": ("CME", "https://www.cmegroup.com/markets/fx/g10/australian-dollar.contractSpecs.html", "100,000 Australian dollars", "$0.00005/AUD = $5.00", "Physically deliverable FX futures; options exercise into futures.", "CME FX spec/rulebook reference reviewed."),
    "6B": ("CME", "https://www.cmegroup.com/markets/fx/g10/british-pound.contractSpecs.html", "62,500 British pounds", "$0.0001/GBP = $6.25", "Physically deliverable FX futures; options exercise into futures.", "CME FX spec/rulebook reference reviewed."),
    "6C": ("CME", "https://www.cmegroup.com/markets/fx/g10/canadian-dollar.contractSpecs.html", "100,000 Canadian dollars", "$0.00005/CAD = $5.00", "Physically deliverable FX futures; options exercise into futures.", "CME FX spec/rulebook reference reviewed."),
    "6S": ("CME", "https://www.cmegroup.com/markets/fx/g10/swiss-franc.contractSpecs.html", "125,000 Swiss francs", "$0.0001/CHF = $12.50", "Physically deliverable FX futures; options exercise into futures.", "CME FX spec/rulebook reference reviewed."),
    "ZC": ("CBOT", "https://www.cmegroup.com/markets/agriculture/grains/corn.contractSpecs.html", "5,000 bushels", "1/4 cent/bu = $12.50", "Physically deliverable grain futures; options exercise into futures.", "CBOT Corn spec/rulebook reference reviewed."),
    "ZS": ("CBOT", "https://www.cmegroup.com/markets/agriculture/oilseeds/soybean.contractSpecs.html", "5,000 bushels", "1/4 cent/bu = $12.50", "Physically deliverable oilseed futures; options exercise into futures.", "CBOT Soybean spec/rulebook reference reviewed."),
    "ZW": ("CBOT", "https://www.cmegroup.com/markets/agriculture/grains/wheat.contractSpecs.html", "5,000 bushels", "1/4 cent/bu = $12.50", "Physically deliverable wheat futures; options exercise into futures.", "CBOT Wheat spec/rulebook reference reviewed."),
    "KE": ("CBOT", "https://www.cmegroup.com/markets/agriculture/grains/kc-wheat.contractSpecs.html", "5,000 bushels", "1/4 cent/bu = $12.50", "Physically deliverable wheat futures; options exercise into futures.", "CBOT KC HRW Wheat spec/rulebook reference reviewed."),
    "ZL": ("CBOT", "https://www.cmegroup.com/markets/agriculture/oilseeds/soybean-oil.contractSpecs.html", "60,000 pounds", "0.01 cent/lb = $6.00", "Physically deliverable soybean-oil futures; options exercise into futures.", "CBOT Soybean Oil spec/rulebook reference reviewed."),
    "ZM": ("CBOT", "https://www.cmegroup.com/markets/agriculture/oilseeds/soybean-meal.contractSpecs.html", "100 short tons", "$0.10/ton = $10.00", "Physically deliverable soybean-meal futures; options exercise into futures.", "CBOT Soybean Meal spec/rulebook reference reviewed."),
    "LE": ("CME", "https://www.cmegroup.com/markets/agriculture/livestock/live-cattle.contractSpecs.html", "40,000 pounds", "0.025 cent/lb = $10.00", "Physically deliverable live-cattle futures; options exercise into futures.", "CME Live Cattle spec/rulebook reference reviewed."),
    "GF": ("CME", "https://www.cmegroup.com/markets/agriculture/livestock/feeder-cattle.contractSpecs.html", "50,000 pounds", "0.025 cent/lb = $12.50", "Cash settled to CME Feeder Cattle Index; options exercise into futures.", "CME Feeder Cattle spec/rulebook reference reviewed."),
    "HE": ("CME", "https://www.cmegroup.com/markets/agriculture/livestock/lean-hogs.contractSpecs.html", "40,000 pounds", "0.025 cent/lb = $10.00", "Cash settled to CME Lean Hog Index; options exercise into futures.", "CME Lean Hog spec/rulebook reference reviewed."),
    "DC": ("CME", "https://www.cmegroup.com/markets/agriculture/dairy/class-iii-milk.contractSpecs.html", "200,000 pounds of milk", "$0.01/cwt = $20.00", "Cash settled against USDA monthly Class III milk price.", "CME dairy spec/rulebook reference reviewed."),
    "GDK": ("CME", "https://www.cmegroup.com/markets/agriculture/dairy/class-iv-milk.contractSpecs.html", "200,000 pounds of milk", "$0.01/cwt = $20.00", "Cash settled against USDA monthly Class IV milk price.", "CME dairy spec/rulebook reference reviewed."),
    "CSC": ("CME", "https://www.cmegroup.com/markets/agriculture/dairy/cheese.contractSpecs.html", "20,000 pounds", "$0.00025/lb = $5.00", "Cash settled against USDA/CME cheese reference pricing.", "CME dairy spec/rulebook reference reviewed."),
    "GNF": ("CME", "https://www.cmegroup.com/markets/agriculture/dairy/nonfat-dry-milk.contractSpecs.html", "44,000 pounds", "$0.00025/lb = $11.00", "Cash settled against USDA nonfat dry milk price.", "CME dairy spec/rulebook reference reviewed."),
    "CB": ("CME", "https://www.cmegroup.com/markets/agriculture/dairy/butter.contractSpecs.html", "20,000 pounds", "$0.00025/lb = $5.00", "Cash settled/physically settled treatment depends on listed dairy product; verify specific chapter.", "CME dairy spec/rulebook reference reviewed."),
    "DY": ("CME", "https://www.cmegroup.com/markets/agriculture/dairy/dry-whey.contractSpecs.html", "44,000 pounds", "$0.00025/lb = $11.00", "Cash settled against USDA dry whey price.", "CME dairy spec/rulebook reference reviewed."),
    "VM": ("CME", "https://www.cmegroup.com/markets/cryptocurrencies/ether/micro-ether.contractSpecs.html", "0.1 ether futures exposure", "See CME option premium tick table", "Cash settled to CME CF Ether-Dollar Reference Rate exposure via Micro Ether futures.", "CME Micro Ether spec/rulebook reference reviewed."),
    "WM": ("CME", "https://www.cmegroup.com/markets/cryptocurrencies/bitcoin/micro-bitcoin.contractSpecs.html", "0.1 bitcoin futures exposure", "See CME option premium tick table", "Cash settled to CME CF Bitcoin Reference Rate exposure via Micro Bitcoin futures.", "CME Micro Bitcoin spec/rulebook reference reviewed."),
}


def _root_symbol(globex: str, clearing: str) -> str:
    preferred = globex or clearing
    return preferred.split("-")[0] if preferred and preferred != "-" else clearing


def _underlying_hint(option: dict) -> tuple[str, str]:
    name = option["product_name"]
    asset = option["asset_class"]
    group = option["product_group"]

    treasury = {
        "2-Year": ("ZT", "2-Year T-Note Futures"),
        "5-Year": ("ZF", "5-Year T-Note Futures"),
        "10-Year": ("ZN", "10-Year T-Note Futures"),
        "Ultra 10-Year": ("TN", "Ultra 10-Year U.S. Treasury Note Futures"),
        "Ultra U.S. Treasury Bond": ("UB", "Ultra U.S. Treasury Bond Futures"),
        "U.S. Treasury Bond": ("ZB", "U.S. Treasury Bond Futures"),
    }
    for token, result in treasury.items():
        if token in name:
            return result

    rules = [
        ("SOFR", "SR3", "Three-Month SOFR Futures"),
        ("E-mini S&P 500", "ES", "E-mini S&P 500 Futures"),
        ("Micro E-mini S&P 500", "MES", "Micro E-mini S&P 500 Index Futures"),
        ("E-mini Nasdaq-100", "NQ", "E-mini Nasdaq-100 Futures"),
        ("Micro E-mini Nasdaq", "MNQ", "Micro E-mini Nasdaq-100 Index Futures"),
        ("Russell 2000", "RTY", "E-mini Russell 2000 Index Futures"),
        ("Crude Oil", "CL", "Crude Oil Futures"),
        ("WTI", "CL", "Crude Oil Futures"),
        ("Brent", "BZ", "Brent Last Day Financial Futures"),
        ("Natural Gas", "NG", "Henry Hub Natural Gas Futures"),
        ("RBOB", "RB", "RBOB Gasoline Futures"),
        ("ULSD", "HO", "NY Harbor ULSD Futures"),
        ("Gold", "GC", "Gold Futures"),
        ("Silver", "SI", "Silver Futures"),
        ("Copper", "HG", "Copper Futures"),
        ("Corn", "ZC", "Corn Futures"),
        ("Soybean Oil", "ZL", "Soybean Oil Futures"),
        ("Soybean Meal", "ZM", "Soybean Meal Futures"),
        ("Soybean", "ZS", "Soybean Futures"),
        ("Chicago SRW Wheat", "ZW", "Chicago SRW Wheat Futures"),
        ("KC HRW Wheat", "KE", "KC HRW Wheat Futures"),
        ("Live Cattle", "LE", "Live Cattle Futures"),
        ("Feeder Cattle", "GF", "Feeder Cattle Futures"),
        ("Lean Hog", "HE", "Lean Hog Futures"),
        ("Class III Milk", "DC", "Class III Milk Futures"),
        ("EUR/USD", "6E", "Euro FX Futures"),
        ("JPY/USD", "6J", "Japanese Yen Futures"),
        ("AUD/USD", "6A", "Australian Dollar Futures"),
        ("CAD/USD", "6C", "Canadian Dollar Futures"),
        ("GBP/USD", "6B", "British Pound Futures"),
        ("CHF/USD", "6S", "Swiss Franc Futures"),
    ]
    for token, root, underlying in rules:
        if token in name:
            return root, underlying

    return _root_symbol(option["globex"], option["clearing"]), f"{asset} {group}".strip()


def _best_future(products: list[dict], root: str, underlying_name: str, asset: str, group: str) -> dict | None:
    futures = [p for p in products if p["cleared_as"] == "Futures"]
    exact = [p for p in futures if p["root_symbol"] == root or p["product_name"] == underlying_name]
    if exact:
        return sorted(exact, key=lambda p: (p["tradability_score"], p["volume"], p["open_interest"]), reverse=True)[0]
    same_group = [p for p in futures if p["asset_class"] == asset and p["product_group"] == group]
    if same_group:
        return sorted(same_group, key=lambda p: (p["tradability_score"], p["volume"], p["open_interest"]), reverse=True)[0]
    return None


def _strategy_fit(asset: str, option_volume: int, option_oi: int, weekly_count: int) -> str:
    fit = []
    if option_volume >= 100_000 or option_oi >= 1_000_000:
        fit.append("Core convexity")
    elif option_volume >= 10_000 or option_oi >= 100_000:
        fit.append("Liquid convexity")
    else:
        fit.append("Specialist convexity")
    if asset in {"Equities", "Energy", "Metals", "FX", "Interest Rate"}:
        fit.append("trend overlay")
    if weekly_count:
        fit.append("weekly tenor")
    return " + ".join(fit)


def _contract_details(root: str) -> tuple[str, str, str, str, str, str]:
    return CONTRACT_DETAILS.get(
        root,
        (
            "Tradable options market from the CME product slate; contract details require manual validation.",
            "See CME contract specs",
            "See CME contract specs",
            "See CME contract specs",
            "Validate multiplier, tick, settlement, and exercise style before trading.",
            "CME product page / rulebook",
        ),
    )


def _rulebook_verification(root: str) -> tuple[str, str, str, str, str, str]:
    return RULEBOOK_VERIFICATION.get(
        root,
        (
            "",
            "https://www.cmegroup.com/rulebook/",
            "Manual verification required",
            "Manual verification required",
            "Manual verification required",
            "No product-specific rulebook/spec mapping available in the local reference table.",
        ),
    )


def build_rows(json_path: Path) -> list[list[object]]:
    payload = json.loads(json_path.read_text())
    products = payload["products"]
    options = [
        p
        for p in products
        if p["cleared_as"] == "Options" and p["account_20m_capacity_contracts"] >= 1
    ]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    labels: dict[tuple[str, str], tuple[str, str]] = {}
    for option in options:
        root, underlying = _underlying_hint(option)
        key = (root, underlying)
        labels[key] = (root, underlying)
        grouped[key].append(option)

    rows = []
    for key, option_rows in grouped.items():
        root, underlying = labels[key]
        first = option_rows[0]
        future = _best_future(products, root, underlying, first["asset_class"], first["product_group"])
        option_volume = sum(p["volume"] for p in option_rows)
        option_oi = sum(p["open_interest"] for p in option_rows)
        option_capacity = sum(p["account_20m_capacity_contracts"] for p in option_rows)
        weekly_count = sum("Weekly" in p["product_name"] or "Week " in p["product_name"] for p in option_rows)
        top_options = sorted(option_rows, key=lambda p: (p["volume"], p["open_interest"]), reverse=True)[:5]
        top_option_names = "; ".join(f"{p['root_symbol']} {p['product_name']}" for p in top_options)
        description, unit, quotation, tick, notes, source = _contract_details(root)
        rb_exchange, rb_url, rb_unit, rb_tick, rb_settlement, rb_notes = _rulebook_verification(root)
        rows.append(
            [
                0,
                root,
                underlying,
                description,
                unit,
                quotation,
                tick,
                notes,
                source,
                rb_exchange,
                rb_url,
                rb_unit,
                rb_tick,
                rb_settlement,
                rb_notes,
                first["asset_class"],
                first["product_group"],
                future["root_symbol"] if future else root,
                future["product_name"] if future else "",
                future["tradability_score"] if future else "",
                future["volume"] if future else "",
                future["open_interest"] if future else "",
                len(option_rows),
                weekly_count,
                option_volume,
                option_oi,
                option_capacity,
                _strategy_fit(first["asset_class"], option_volume, option_oi, weekly_count),
                top_option_names,
            ]
        )

    rows.sort(key=lambda r: (r[19], r[18], r[16]), reverse=True)
    for i, row in enumerate(rows, 1):
        row[0] = i
    return rows


def write_workbook(json_path: Path, output_path: Path) -> Path:
    header = [
        "Rank",
        "Underlying Root",
        "Underlying",
        "Contract Description",
        "Contract Unit",
        "Price Quotation",
        "Minimum Tick",
        "Settlement / Notes",
        "Research Source",
        "Rulebook Exchange",
        "Rulebook / Spec URL",
        "Rulebook Unit",
        "Rulebook Tick",
        "Rulebook Settlement",
        "Rulebook Verification Notes",
        "Asset Class",
        "Product Group",
        "Matched Future Root",
        "Matched Future",
        "Future Score",
        "Future Volume",
        "Future Open Interest",
        "Tradable Option Products",
        "Weekly/EOM Products",
        "Option Volume",
        "Option Open Interest",
        "Option Capacity Contracts",
        "Strategy Fit",
        "Top Tradable Option Markets",
    ]
    rows = [header] + build_rows(json_path)
    widths = [
        8, 18, 42, 70, 30, 30, 28, 58, 34, 16, 70, 30, 28, 60, 60,
        18, 22, 18, 48, 14, 16, 20, 24, 20, 16, 20, 24, 28, 110,
    ]
    sheet = _sheet_xml(
        rows,
        widths=widths,
        frozen_row=1,
        filter_range=f"A1:AC{len(rows)}",
        numeric_cols={0, 20, 21, 22, 23, 24, 25, 26},
        score_cols={19},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    core, app = _doc_props()
    with ZipFile(output_path, "w", ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _content_types(1))
        z.writestr("_rels/.rels", _root_rels())
        z.writestr("docProps/core.xml", core)
        z.writestr("docProps/app.xml", app)
        z.writestr("xl/workbook.xml", _workbook_xml(["Options Underlyings"]))
        z.writestr("xl/_rels/workbook.xml.rels", _workbook_rels(1))
        z.writestr("xl/styles.xml", _styles_xml())
        z.writestr("xl/worksheets/sheet1.xml", sheet)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=str(DEFAULT_JSON))
    parser.add_argument("--out", default=str(DEFAULT_XLSX))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = write_workbook(Path(args.json), Path(args.out))
    print(f"Wrote {path}")
    print(f"Underlyings: {len(build_rows(Path(args.json)))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
import polars as pl


def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def parse_run_id(stdout: str) -> str | None:
    for line in stdout.splitlines():
        if line.startswith("Run ID:"):
            return line.split("Run ID:")[1].strip()
    return None


def load_summary_json(path: Path) -> Dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="V2.4 target-vol sweep for BTC/ETH daily 5/40 system.")
    parser.add_argument("--base_config_btc", required=True, help="Path to BTC base YAML")
    parser.add_argument("--base_config_eth", required=True, help="Path to ETH base YAML")
    parser.add_argument("--target_vol_grid", default="0.40,0.50,0.60", help="Comma list of target vol annuals")
    parser.add_argument("--out_dir", default="artifacts/compare/v24_sweep", help="Output directory")
    parser.add_argument("--initial_nav", type=float, default=100000.0)
    parser.add_argument("--keep_temp_configs", action="store_true")
    parser.add_argument("--python_bin", default="python")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tvs = [float(x.strip()) for x in args.target_vol_grid.split(",") if x.strip()]
    cfg_btc_base = load_yaml(Path(args.base_config_btc))
    cfg_eth_base = load_yaml(Path(args.base_config_eth))

    results: List[Dict] = []

    for tv in tvs:
        tv_tag = int(tv * 100)
        # BTC config
        cfg_btc = yaml.safe_load(yaml.safe_dump(cfg_btc_base))
        cfg_btc["run_name"] = f"v24_btc_tv{tv_tag:02d}"
        cfg_btc.setdefault("strategy", {})
        cfg_btc["strategy"]["target_vol_annual"] = tv

        # ETH config
        cfg_eth = yaml.safe_load(yaml.safe_dump(cfg_eth_base))
        cfg_eth["run_name"] = f"v24_eth_tv{tv_tag:02d}"
        cfg_eth.setdefault("strategy", {})
        cfg_eth["strategy"]["target_vol_annual"] = tv

        if args.keep_temp_configs:
            cfg_dir = out_dir / "configs"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            btc_cfg_path = cfg_dir / f"btc_tv{tv_tag:02d}.yaml"
            eth_cfg_path = cfg_dir / f"eth_tv{tv_tag:02d}.yaml"
            write_yaml(cfg_btc, btc_cfg_path)
            write_yaml(cfg_eth, eth_cfg_path)
        else:
            btc_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
            btc_cfg_path = Path(btc_tmp.name)
            write_yaml(cfg_btc, btc_cfg_path)
            eth_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
            eth_cfg_path = Path(eth_tmp.name)
            write_yaml(cfg_eth, eth_cfg_path)

        # Run BTC
        code, out, err = run_cmd([args.python_bin, "scripts/run_backtest.py", "--config", str(btc_cfg_path)])
        if code != 0:
            print(f"[tv={tv}] BTC run failed:\nSTDOUT:\n{out}\nSTDERR:\n{err}", file=sys.stderr)
            continue
        btc_run_id = parse_run_id(out)
        if not btc_run_id:
            print(f"[tv={tv}] BTC run_id parse failed", file=sys.stderr)
            continue

        # Run ETH
        code, out, err = run_cmd([args.python_bin, "scripts/run_backtest.py", "--config", str(eth_cfg_path)])
        if code != 0:
            print(f"[tv={tv}] ETH run failed:\nSTDOUT:\n{out}\nSTDERR:\n{err}", file=sys.stderr)
            continue
        eth_run_id = parse_run_id(out)
        if not eth_run_id:
            print(f"[tv={tv}] ETH run_id parse failed", file=sys.stderr)
            continue

        btc_run_dir = Path("artifacts") / "runs" / btc_run_id
        eth_run_dir = Path("artifacts") / "runs" / eth_run_id

        combined_out = out_dir / f"combined_tv{tv_tag:02d}"
        code, out, err = run_cmd(
            [
                args.python_bin,
                "scripts/build_combined_portfolio_50_50.py",
                "--run_a",
                str(btc_run_dir),
                "--run_b",
                str(eth_run_dir),
                "--out_dir",
                str(combined_out),
                "--initial_nav",
                str(args.initial_nav),
            ]
        )
        if code != 0:
            print(f"[tv={tv}] combine failed:\nSTDOUT:\n{out}\nSTDERR:\n{err}", file=sys.stderr)
            continue

        summary_path = combined_out / "summary.json"
        summary = load_summary_json(summary_path)
        if not summary:
            # fallback parse stdout for key metrics
            summary = {}
            for line in out.splitlines():
                if "Total Return" in line and ":" in line:
                    summary["total_return"] = float(line.split(":")[1])
                if "CAGR" in line and ":" in line:
                    summary["cagr"] = float(line.split(":")[1])
                if "Sharpe" in line and ":" in line:
                    summary["sharpe"] = float(line.split(":")[1])
                if "Vol Ann" in line and ":" in line:
                    summary["vol_annual"] = float(line.split(":")[1])
                if "Max Drawdown" in line and ":" in line:
                    summary["max_drawdown"] = float(line.split(":")[1])

        results.append(
            {
                "tv": tv,
                "btc_run_id": btc_run_id,
                "eth_run_id": eth_run_id,
                "combined_total_return": summary.get("total_return"),
                "combined_cagr": summary.get("cagr"),
                "combined_sharpe": summary.get("sharpe"),
                "combined_vol_ann": summary.get("vol_annual"),
                "combined_max_drawdown": summary.get("max_drawdown"),
                "combined_max_dd_ts": summary.get("max_drawdown_ts"),
                "combined_out_dir": str(combined_out),
            }
        )

        if not args.keep_temp_configs:
            if btc_cfg_path.exists():
                btc_cfg_path.unlink()
            if eth_cfg_path.exists():
                eth_cfg_path.unlink()

    if not results:
        print("No successful runs.", file=sys.stderr)
        sys.exit(1)

    # Output CSV and JSON
    df = pl.DataFrame(results)
    df.sort(by=["combined_sharpe"], descending=True).write_csv(out_dir / "v24_target_vol_sweep.csv")
    with open(out_dir / "v24_target_vol_sweep.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print summary
    sorted_rows = sorted(
        results,
        key=lambda r: (
            -(r.get("combined_sharpe") or -1e9),
            r.get("combined_max_drawdown") or 0.0,
            -(r.get("combined_cagr") or -1e9),
        ),
    )
    print("tv,sharpe,max_dd,cagr,total_ret,run_btc,run_eth,combined_out")
    for r in sorted_rows:
        print(
            f"{r['tv']:.2f},"
            f"{(r.get('combined_sharpe') or 0):.4f},"
            f"{(r.get('combined_max_drawdown') or 0):.4f},"
            f"{(r.get('combined_cagr') or 0):.4f},"
            f"{(r.get('combined_total_return') or 0):.4f},"
            f"{r['btc_run_id']},"
            f"{r['eth_run_id']},"
            f"{r['combined_out_dir']}"
        )


if __name__ == "__main__":
    main()


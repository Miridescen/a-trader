"""
回测主入口
用法:
    python -m backtest.run                    # 默认：全部策略对比
    python -m backtest.run --strategy momentum
    python -m backtest.run --strategy institution
    python -m backtest.run --start 20190101 --end 20241231
    python -m backtest.run --freq M           # 月度调仓
"""
import argparse
import pandas as pd
from datetime import date

from data.storage.db import init_db
from backtest.data_loader import load_price_matrix, load_benchmark
from backtest.engine import BacktestEngine, TradeConfig
from backtest.strategies import (
    momentum_strategy, institution_equal_strategy
)
from backtest.report import print_report, save_nav_csv, compare_strategies


# 当前选股结果（机构共同持仓 Top20，来自选股模块输出）
INSTITUTION_POOL = [
    "603993.SH", "601600.SH", "300750.SZ", "600989.SH", "300502.SZ",
    "601601.SH", "300308.SZ", "002463.SZ", "601318.SH", "301606.SZ",
    "002916.SZ", "300394.SZ", "601138.SH", "603799.SH", "600066.SH",
    "300476.SZ", "605117.SH", "002001.SZ", "002371.SZ", "600938.SH",
]


def run_backtest(
    strategy_name: str = "all",
    start_date:    str = "20190101",
    end_date:      str = "20261231",
    freq:          str = "Q",
    top_n:         int = 25,
    initial_capital: float = 1_000_000,
):
    init_db()

    print(f"\n[配置] 策略={strategy_name}  区间={start_date}~{end_date}  调仓={freq}  持仓={top_n}只")

    # ── 加载数据 ──────────────────────────────────────────
    price_df  = load_price_matrix(start_date=start_date, end_date=end_date, min_days=200)
    benchmark = load_benchmark("000300.SH", start_date, end_date)

    if price_df.empty:
        print("价格数据为空，请先运行数据采集")
        return

    # 基准净值归一化
    bm_nav = benchmark / benchmark.iloc[0]

    config  = TradeConfig()
    results = {}

    # ── 策略1：动量 ───────────────────────────────────────
    if strategy_name in ("all", "momentum"):
        print("\n▶ 运行动量策略...")
        engine = BacktestEngine(price_df, benchmark, initial_capital, config, freq)
        strat  = momentum_strategy(lookback=120, skip_recent=20, top_n=top_n)
        port   = engine.run(strat)
        print_report(port, benchmark, f"动量策略(Top{top_n})")
        save_nav_csv(port, benchmark, "data/nav_momentum.csv")
        results["动量策略"] = port

    # ── 策略2：机构持仓等权（前瞻，仅供参考） ────────────────
    if strategy_name in ("all", "institution"):
        pool = [ts for ts in INSTITUTION_POOL if ts in price_df.columns]
        if pool:
            print(f"\n▶ 运行机构持仓策略（{len(pool)} 只）...")
            engine = BacktestEngine(price_df, benchmark, initial_capital, config, freq)
            strat  = institution_equal_strategy(pool, top_n=min(top_n, len(pool)))
            port   = engine.run(strat)
            print_report(port, benchmark, f"机构持仓策略(Top{len(pool)})")
            save_nav_csv(port, benchmark, "data/nav_institution.csv")
            results["机构持仓策略"] = port

    # ── 多策略对比 ────────────────────────────────────────
    if len(results) > 1:
        print("\n" + "="*60)
        print("  多策略对比汇总")
        print("="*60)
        cmp = compare_strategies(results, benchmark)
        print(cmp.to_string())

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="策略回测")
    parser.add_argument("--strategy", default="all",
                        choices=["all", "momentum", "institution"],
                        help="运行的策略")
    parser.add_argument("--start", default="20190101", help="回测开始日期")
    parser.add_argument("--end",   default="20261231", help="回测结束日期")
    parser.add_argument("--freq",  default="Q", choices=["M", "Q"], help="调仓频率")
    parser.add_argument("--top",   type=int, default=25, help="持仓数量")
    parser.add_argument("--capital", type=float, default=1_000_000, help="初始资金")
    args = parser.parse_args()

    run_backtest(
        strategy_name=args.strategy,
        start_date=args.start,
        end_date=args.end,
        freq=args.freq,
        top_n=args.top,
        initial_capital=args.capital,
    )

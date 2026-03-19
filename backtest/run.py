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
from backtest.data_loader import load_price_matrix, load_benchmark, load_mv_matrix
from backtest.engine import BacktestEngine, TradeConfig
from backtest.strategies import (
    momentum_strategy, institution_equal_strategy, multi_factor_strategy,
    small_cap_strategy,
)
from backtest.factor_builder import build_factor_scores
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
    mv_min: float = 200_000,   # 微市值策略下限（万元），默认 20亿
    mv_max: float = 300_000,   # 微市值策略上限（万元），默认 30亿
):
    init_db()

    print(f"\n[配置] 策略={strategy_name}  区间={start_date}~{end_date}  调仓={freq}  持仓={top_n}只")

    # ── 加载数据 ──────────────────────────────────────────
    price_df  = load_price_matrix(start_date=start_date, end_date=end_date, min_days=200)
    benchmark = load_benchmark("000300.SH", start_date, end_date)

    # 微市值策略需要市值矩阵（其他策略跳过以节省内存）
    mv_df = pd.DataFrame()
    if strategy_name in ("all", "smallcap"):
        mv_df = load_mv_matrix(start_date=start_date, end_date=end_date)

    if price_df.empty:
        print("价格数据为空，请先运行数据采集")
        return

    # 基准净值归一化
    bm_nav = benchmark / benchmark.iloc[0]

    config  = TradeConfig()
    results = {}

    # ── 策略：微市值轮动（20~30亿，最小5只，每5交易日换仓）────
    if strategy_name in ("all", "smallcap"):
        if mv_df.empty:
            print("\n⚠ 微市值策略跳过：daily_basic 无市值数据")
        else:
            print(f"\n▶ 运行微市值轮动策略（{mv_min/10000:.0f}~{mv_max/10000:.0f}亿，Top{top_n}，每5交易日调仓）...")
            engine = BacktestEngine(price_df, benchmark, initial_capital, config, rebalance_freq=5)
            strat  = small_cap_strategy(mv_df, top_n=top_n, mv_min=mv_min, mv_max=mv_max)
            port   = engine.run(strat)
            print_report(port, benchmark, f"微市值策略({mv_min/10000:.0f}~{mv_max/10000:.0f}亿 Top{top_n})")
            save_nav_csv(port, benchmark, "data/nav_smallcap.csv")
            results["微市值策略"] = port

    # ── 预构建多因子矩阵（供多因子策略使用）──────────────────
    factor_scores = None
    if strategy_name in ("all", "multifactor"):
        engine_tmp = BacktestEngine(price_df, benchmark, initial_capital, config, freq)
        factor_scores = build_factor_scores(price_df, engine_tmp._rebal_dates)

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

    # ── 策略3：多因子（财务+动量，无前瞻偏差）───────────────
    if strategy_name in ("all", "multifactor"):
        if factor_scores is not None and not factor_scores.empty:
            print("\n▶ 运行多因子策略（财务+动量，无前瞻偏差）...")
            engine = BacktestEngine(price_df, benchmark, initial_capital, config, freq)
            strat  = multi_factor_strategy(factor_scores, top_n=top_n, weight_scheme="equal")
            port   = engine.run(strat)
            print_report(port, benchmark, f"多因子策略(Top{top_n})")
            save_nav_csv(port, benchmark, "data/nav_multifactor.csv")
            results["多因子策略"] = port
        else:
            print("\n⚠ 多因子策略跳过：因子矩阵为空（财务数据不足）")

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
                        choices=["all", "momentum", "institution", "multifactor", "smallcap"],
                        help="运行的策略")
    parser.add_argument("--start", default="20190101", help="回测开始日期")
    parser.add_argument("--end",   default="20261231", help="回测结束日期")
    parser.add_argument("--freq",  default="Q", choices=["M", "Q"], help="调仓频率")
    parser.add_argument("--top",   type=int, default=25, help="持仓数量")
    parser.add_argument("--capital", type=float, default=1_000_000, help="初始资金")
    parser.add_argument("--mv-min", type=float, default=200_000, help="微市值策略市值下限（万元），默认20亿")
    parser.add_argument("--mv-max", type=float, default=300_000, help="微市值策略市值上限（万元），默认30亿")
    args = parser.parse_args()

    run_backtest(
        strategy_name=args.strategy,
        start_date=args.start,
        end_date=args.end,
        freq=args.freq,
        top_n=args.top,
        initial_capital=args.capital,
        mv_min=args.mv_min,
        mv_max=args.mv_max,
    )

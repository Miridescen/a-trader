"""
回测报告生成器
打印绩效表格，保存净值曲线 CSV
"""
import pandas as pd
import numpy as np
from typing import Optional
from backtest.engine import Portfolio
from backtest.metrics import (
    full_report, drawdown_series, underwater_periods,
    annual_return, max_drawdown, sharpe_ratio
)


def print_report(
    portfolio: Portfolio,
    benchmark: Optional[pd.Series] = None,
    strategy_name: str = "策略",
):
    nav = portfolio.nav_series()
    if nav.empty:
        print("无净值数据")
        return

    # 对齐基准
    bm_nav = None
    if benchmark is not None:
        common = nav.index.intersection(benchmark.index)
        bm_aligned = benchmark.loc[common]
        bm_nav = bm_aligned / bm_aligned.iloc[0]

    print("\n" + "="*60)
    print(f"  回测报告 — {strategy_name}")
    print("="*60)
    report = full_report(nav, bm_nav, strategy_name)
    print(report.to_string())

    # 分年度收益
    print("\n─── 分年度收益 ───")
    yearly = nav.resample("Y").last().pct_change().dropna() * 100
    if bm_nav is not None:
        bm_yearly = bm_nav.resample("Y").last().pct_change().dropna() * 100
        yr_df = pd.DataFrame({"策略": yearly, "基准(沪深300)": bm_yearly}).round(2)
        yr_df["超额"] = (yr_df["策略"] - yr_df["基准(沪深300)"]).round(2)
    else:
        yr_df = pd.DataFrame({"策略": yearly}).round(2)
    yr_df.index = yr_df.index.year
    print(yr_df.to_string())

    # 最大回撤统计
    periods = underwater_periods(nav)
    if not periods.empty:
        print("\n─── Top5 最大回撤阶段 ───")
        top5 = periods.nsmallest(5, "drawdown")[
            ["peak", "trough", "recover", "drawdown", "duration_days"]
        ]
        top5.columns = ["峰值日", "谷底日", "恢复日", "回撤%", "持续天数"]
        print(top5.to_string(index=False))

    # 换手率统计
    if portfolio.records:
        turnovers = [r.turnover for r in portfolio.records]
        total_to = sum(turnovers)
        capital  = portfolio.records[0].portfolio_value
        years    = (nav.index[-1] - nav.index[0]).days / 365
        ann_to   = total_to / capital / years * 100 if years > 0 else 0
        print(f"\n─── 交易统计 ───")
        print(f"  总换手金额 : {total_to/1e4:.1f}万元")
        print(f"  年化换手率 : {ann_to:.1f}%")
        print(f"  总交易次数 : {len(portfolio.trade_log)}")


def save_nav_csv(
    portfolio: Portfolio,
    benchmark: Optional[pd.Series] = None,
    filepath: str = "data/backtest_nav.csv",
):
    nav = portfolio.nav_series()
    df  = pd.DataFrame({"portfolio_nav": nav})

    if benchmark is not None:
        common = nav.index.intersection(benchmark.index)
        bm     = benchmark.loc[common] / benchmark.loc[common].iloc[0]
        df["benchmark_nav"] = bm

    df["drawdown"] = drawdown_series(nav) * 100

    df.index.name = "date"
    df.to_csv(filepath)
    print(f"净值曲线已保存: {filepath}")
    return df


def compare_strategies(
    portfolios: dict,      # {name: Portfolio}
    benchmark: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    多策略对比表
    portfolios: {'策略A': portfolio_a, '策略B': portfolio_b}
    """
    rows = []
    for name, port in portfolios.items():
        nav = port.nav_series()
        if nav.empty:
            continue
        bm_nav = None
        if benchmark is not None:
            common = nav.index.intersection(benchmark.index)
            bm_aligned = benchmark.loc[common] / benchmark.loc[common].iloc[0]
            bm_nav = bm_aligned

        row = {"策略": name}
        row["年化收益"]  = f"{annual_return(nav)*100:.2f}%"
        row["最大回撤"]  = f"{max_drawdown(nav)*100:.2f}%"
        row["夏普比率"]  = f"{sharpe_ratio(nav):.3f}"
        if bm_nav is not None:
            from backtest.metrics import information_ratio, alpha
            row["超额收益"] = f"{(annual_return(nav)-annual_return(bm_nav))*100:.2f}%"
            row["信息比率"] = f"{information_ratio(nav, bm_nav):.3f}"
        rows.append(row)

    return pd.DataFrame(rows).set_index("策略")

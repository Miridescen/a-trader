"""
绩效指标计算
输入：净值序列（pd.Series，index=日期，值=净值，初始=1）
"""
import pandas as pd
import numpy as np
from typing import Optional


TRADING_DAYS = 252
RISK_FREE_RATE = 0.025   # 无风险利率（年化）


def annual_return(nav: pd.Series) -> float:
    """年化收益率"""
    if len(nav) < 2:
        return 0.0
    days = (nav.index[-1] - nav.index[0]).days
    if days <= 0:
        return 0.0
    total = nav.iloc[-1] / nav.iloc[0] - 1
    return (1 + total) ** (365 / days) - 1


def max_drawdown(nav: pd.Series) -> float:
    """最大回撤（负值）"""
    if nav.empty:
        return 0.0
    cummax = nav.cummax()
    dd = (nav - cummax) / cummax
    return float(dd.min())


def drawdown_series(nav: pd.Series) -> pd.Series:
    """回撤序列"""
    cummax = nav.cummax()
    return (nav - cummax) / cummax


def sharpe_ratio(nav: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """年化夏普比率"""
    daily_ret = nav.pct_change().dropna()
    if daily_ret.std() == 0:
        return 0.0
    excess = daily_ret - rf / TRADING_DAYS
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))


def sortino_ratio(nav: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """索提诺比率（只惩罚下行波动）"""
    daily_ret = nav.pct_change().dropna()
    downside = daily_ret[daily_ret < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    excess_ann = annual_return(nav) - rf
    return float(excess_ann / (downside.std() * np.sqrt(TRADING_DAYS)))


def calmar_ratio(nav: pd.Series) -> float:
    """卡玛比率 = 年化收益 / |最大回撤|"""
    mdd = abs(max_drawdown(nav))
    if mdd == 0:
        return 0.0
    return annual_return(nav) / mdd


def volatility(nav: pd.Series) -> float:
    """年化波动率"""
    return float(nav.pct_change().dropna().std() * np.sqrt(TRADING_DAYS))


def win_rate(nav: pd.Series, freq: str = "M") -> float:
    """
    正收益期间占比
    freq: 'D'=日胜率  'M'=月胜率  'Q'=季度胜率
    """
    resampled = nav.resample(freq).last().pct_change().dropna()
    if len(resampled) == 0:
        return 0.0
    return float((resampled > 0).mean())


def information_ratio(nav: pd.Series, benchmark: pd.Series) -> float:
    """信息比率 = 超额收益均值 / 超额收益标准差（年化）"""
    strat_ret = nav.pct_change().dropna()
    bench_ret = benchmark.pct_change().dropna()
    common_idx = strat_ret.index.intersection(bench_ret.index)
    if len(common_idx) < 20:
        return 0.0
    excess = strat_ret.loc[common_idx] - bench_ret.loc[common_idx]
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))


def beta(nav: pd.Series, benchmark: pd.Series) -> float:
    """Beta 系数"""
    strat_ret = nav.pct_change().dropna()
    bench_ret = benchmark.pct_change().dropna()
    common_idx = strat_ret.index.intersection(bench_ret.index)
    if len(common_idx) < 20:
        return 1.0
    s = strat_ret.loc[common_idx]
    b = bench_ret.loc[common_idx]
    cov = np.cov(s, b)
    if cov[1, 1] == 0:
        return 1.0
    return float(cov[0, 1] / cov[1, 1])


def alpha(nav: pd.Series, benchmark: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """Jensen's Alpha（年化）"""
    b   = beta(nav, benchmark)
    ann = annual_return(nav)
    bm_ann = annual_return(benchmark)
    return ann - (rf + b * (bm_ann - rf))


def underwater_periods(nav: pd.Series) -> pd.DataFrame:
    """识别所有回撤阶段（峰值→谷底→恢复）"""
    dd = drawdown_series(nav)
    in_dd = False
    peak_date = None
    trough_date = None
    trough_val = 0.0
    periods = []

    for d, v in dd.items():
        if not in_dd and v < -0.001:
            in_dd = True
            peak_date = d
            trough_date = d
            trough_val = v
        elif in_dd:
            if v < trough_val:
                trough_val = v
                trough_date = d
            elif v >= -0.001:
                periods.append({
                    "peak":     peak_date,
                    "trough":   trough_date,
                    "recover":  d,
                    "drawdown": round(trough_val * 100, 2),
                    "duration_days": (d - peak_date).days,
                })
                in_dd = False

    return pd.DataFrame(periods)


def full_report(
    nav: pd.Series,
    benchmark: Optional[pd.Series] = None,
    strategy_name: str = "策略",
) -> pd.DataFrame:
    """
    生成完整绩效报告
    返回 DataFrame，便于打印或保存
    """
    metrics = {
        "策略": strategy_name,
        "起始日期":    str(nav.index[0].date()),
        "结束日期":    str(nav.index[-1].date()),
        "年化收益率":  f"{annual_return(nav)*100:.2f}%",
        "累计收益率":  f"{(nav.iloc[-1]/nav.iloc[0]-1)*100:.2f}%",
        "年化波动率":  f"{volatility(nav)*100:.2f}%",
        "最大回撤":    f"{max_drawdown(nav)*100:.2f}%",
        "夏普比率":    f"{sharpe_ratio(nav):.3f}",
        "索提诺比率":  f"{sortino_ratio(nav):.3f}",
        "卡玛比率":    f"{calmar_ratio(nav):.3f}",
        "月度胜率":    f"{win_rate(nav,'M')*100:.1f}%",
    }

    if benchmark is not None:
        # 对齐索引
        common = nav.index.intersection(benchmark.index)
        nav_a = nav.loc[common]
        bm_a  = benchmark.loc[common] / benchmark.loc[common].iloc[0]   # 归一化

        metrics["基准年化收益"] = f"{annual_return(bm_a)*100:.2f}%"
        metrics["超额年化收益"] = f"{(annual_return(nav_a)-annual_return(bm_a))*100:.2f}%"
        metrics["信息比率"]    = f"{information_ratio(nav_a, bm_a):.3f}"
        metrics["Beta"]       = f"{beta(nav_a, bm_a):.3f}"
        metrics["Alpha(年化)"] = f"{alpha(nav_a, bm_a)*100:.2f}%"

    return pd.DataFrame([metrics]).T.rename(columns={0: "值"})

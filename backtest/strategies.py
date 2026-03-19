"""
策略库
每个策略是一个函数：(date, price_history, context) -> {ts_code: weight}

已实现：
  1. momentum_strategy        — 纯动量（基准验证用）
  2. multi_factor_strategy    — 多因子（方案四，需预先计算因子）
  3. equal_weight_strategy    — 等权买入机构持仓股（简化版方案一）
"""
import pandas as pd
import numpy as np
from datetime import date
from typing import Dict


# ══════════════════════════════════════════════════════════
# 策略1：动量策略（纯价格，基准对比用）
# ══════════════════════════════════════════════════════════

def momentum_strategy(
    lookback:    int = 120,   # 回望期（交易日）
    skip_recent: int = 20,    # 跳过最近N日（避反转）
    top_n:       int = 30,    # 持仓数量
    universe_size: int = 300, # 每次从前N活跃股中选
):
    """
    工厂函数，返回动量策略函数
    选股逻辑：取过去 lookback 天（去掉最近 skip_recent 天）涨幅最大的 top_n 只
    """
    def _strategy(current_date: date, price_hist: pd.DataFrame, context: dict) -> Dict[str, float]:
        if len(price_hist) < lookback + skip_recent:
            return {}

        # 取回望区间
        hist = price_hist.iloc[-(lookback + skip_recent):]
        p_start = hist.iloc[0]
        p_end   = hist.iloc[-(skip_recent + 1)]   # 去掉最近 skip_recent 天

        # 动量：去掉最近1月的涨幅
        momentum = (p_end / p_start - 1).dropna()

        # 过滤：只选最近有价格的股票（避免停牌）
        latest_prices = price_hist.iloc[-1].dropna()
        valid = momentum.index.intersection(latest_prices.index)
        momentum = momentum.loc[valid]

        if momentum.empty:
            return {}

        # 取 Top N
        top = momentum.nlargest(top_n)

        # 等权
        return {ts: 1.0 / len(top) for ts in top.index}

    return _strategy


# ══════════════════════════════════════════════════════════
# 策略2：多因子策略（方案四）
# 需要预先算好每个调仓期的因子得分
# ══════════════════════════════════════════════════════════

def multi_factor_strategy(
    factor_scores: pd.DataFrame,
    top_n: int = 25,
    weight_scheme: str = "score",  # 'equal' 或 'score'
):
    """
    工厂函数，返回多因子策略函数
    factor_scores: DataFrame，index=日期（调仓日），columns=ts_code，values=得分
    weight_scheme: 'equal'=等权  'score'=按得分加权
    """
    def _strategy(current_date: date, price_hist: pd.DataFrame, context: dict) -> Dict[str, float]:
        # 找最近一期因子得分
        score_dates = factor_scores.index
        past_dates  = score_dates[score_dates <= pd.Timestamp(current_date)]
        if past_dates.empty:
            return {}

        latest_scores = factor_scores.loc[past_dates[-1]].dropna()
        latest_prices = price_hist.iloc[-1].dropna()

        # 只选当前有价格的股票
        valid = latest_scores.index.intersection(latest_prices.index)
        scores = latest_scores.loc[valid].nlargest(top_n)

        if scores.empty:
            return {}

        if weight_scheme == "equal":
            return {ts: 1.0 / len(scores) for ts in scores.index}
        else:
            # 按得分加权（归一化）
            total = scores.sum()
            return {ts: s / total for ts, s in scores.items()}

    return _strategy


# ══════════════════════════════════════════════════════════
# 策略3：机构持仓等权（方案一的简化回测）
# ══════════════════════════════════════════════════════════

def institution_equal_strategy(
    holdings_pool: list,   # 固定持仓池（机构共同持仓股票列表）
    top_n: int = 20,
):
    """
    固定持仓池等权策略
    注意：这是"前瞻偏差"版本，仅用于了解该组合历史表现
    不能直接用于真实交易
    """
    pool = holdings_pool[:top_n]

    def _strategy(current_date: date, price_hist: pd.DataFrame, context: dict) -> Dict[str, float]:
        latest = price_hist.iloc[-1].dropna()
        valid  = [ts for ts in pool if ts in latest.index]
        if not valid:
            return {}
        return {ts: 1.0 / len(valid) for ts in valid}

    return _strategy


# ══════════════════════════════════════════════════════════
# 策略4：微市值策略（20~30亿区间，选最小5只）
# ══════════════════════════════════════════════════════════

def small_cap_strategy(
    mv_matrix: pd.DataFrame,   # index=日期, columns=ts_code, values=总市值(万元)
    top_n:  int   = 5,
    mv_min: float = 200_000,   # 20亿 = 200,000万
    mv_max: float = 300_000,   # 30亿 = 300,000万
):
    """
    微市值轮动策略
    ─────────────────────────────────────────────────────
    选股规则：总市值在 [mv_min, mv_max] 万元之间，选最小的 top_n 只
    持仓权重：等权
    调仓频率：由 BacktestEngine 的 freq 参数决定（传 5 即每5个交易日）
    """
    def _strategy(current_date: date, price_hist: pd.DataFrame, context: dict) -> Dict[str, float]:
        ts_d = pd.Timestamp(current_date)

        # 取当日或最近一期市值（严格 <= 当日，防前瞻）
        past_dates = mv_matrix.index[mv_matrix.index <= ts_d]
        if past_dates.empty:
            return {}
        mv_row = mv_matrix.loc[past_dates[-1]].dropna()

        # 筛选市值区间
        filtered = mv_row[(mv_row >= mv_min) & (mv_row <= mv_max)]
        if filtered.empty:
            return {}

        # 排除停牌（当日无收盘价）
        latest_prices = price_hist.iloc[-1].dropna()
        valid = filtered.index.intersection(latest_prices.index)
        filtered = filtered.loc[valid]
        if filtered.empty:
            return {}

        # 选市值最小的 top_n 只
        selected = filtered.nsmallest(top_n)
        n = len(selected)
        return {ts: 1.0 / n for ts in selected.index}

    return _strategy


# ══════════════════════════════════════════════════════════
# 策略5：价值+质量（财务因子，需历史财务数据）
# ══════════════════════════════════════════════════════════

def value_quality_strategy(
    fin_scores: pd.DataFrame,   # 财务得分矩阵，同 multi_factor_strategy
    top_n: int = 25,
):
    """ROE 高 + 估值合理的价值质量策略"""
    def _strategy(current_date: date, price_hist: pd.DataFrame, context: dict) -> Dict[str, float]:
        score_dates = fin_scores.index
        past = score_dates[score_dates <= pd.Timestamp(current_date)]
        if past.empty:
            return {}

        scores = fin_scores.loc[past[-1]].dropna()
        latest = price_hist.iloc[-1].dropna()
        valid  = scores.index.intersection(latest.index)
        top    = scores.loc[valid].nlargest(top_n)

        if top.empty:
            return {}
        return {ts: 1.0 / len(top) for ts in top.index}

    return _strategy

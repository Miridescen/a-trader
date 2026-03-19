"""
历史因子矩阵构建器
在每个调仓日，用"当时可用"的数据计算各因子得分，避免前瞻偏差。

可用因子：
  - 财务因子（DB）：用最新已披露年报（通常有 3-6 个月滞后）
  - 动量因子（DB）：用截止调仓日的价格历史

不可回测因子（无历史快照）：
  - 机构持仓：仅有当前持仓，无法还原历史 → 本模块跳过
  - 实时估值(PE/PB)：无历史快照 → 改用价格/财务推算的 P/B 代理

用法：
    from backtest.factor_builder import build_factor_scores
    factor_scores = build_factor_scores(price_df, rebal_dates)
"""
import pandas as pd
import numpy as np
from datetime import date
from typing import List
from sqlalchemy import text
from data.storage.db import get_engine


def _percentile_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    return series.rank(pct=True, ascending=ascending, na_option="bottom") * 100


def _load_financial_db() -> pd.DataFrame:
    """
    从 financial_indicator 表加载所有财务数据
    返回列: ts_code, end_date, roe, netprofit_yoy, grossprofit_margin, debt_to_assets
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT ts_code, end_date, roe, netprofit_yoy, grossprofit_margin, debt_to_assets
            FROM financial_indicator
            ORDER BY ts_code, end_date
        """)).fetchall()
    df = pd.DataFrame(rows, columns=["ts_code", "end_date", "roe", "netprofit_yoy",
                                      "grossprofit_margin", "debt_to_assets"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    # 年报披露通常在次年 4 月底，保守延迟 5 个月（12-31 → 5-31）
    df["available_date"] = df["end_date"] + pd.DateOffset(months=5)
    return df


def _financial_snapshot(fin_df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """
    截止 as_of 日期，取每只股票最新可用年报数据
    """
    avail = fin_df[fin_df["available_date"] <= as_of].copy()
    if avail.empty:
        return pd.DataFrame(columns=["ts_code", "roe", "netprofit_yoy",
                                      "grossprofit_margin", "debt_to_assets"])
    # 每只股票取最新年报
    latest = avail.sort_values("end_date").groupby("ts_code").last().reset_index()
    return latest[["ts_code", "roe", "netprofit_yoy", "grossprofit_margin", "debt_to_assets"]]


def _momentum_snapshot(price_df: pd.DataFrame, as_of: pd.Timestamp,
                        lookback: int = 120, skip: int = 20) -> pd.Series:
    """
    截止 as_of 日期，计算动量因子
    返回: Series，index=ts_code, values=动量%
    """
    hist = price_df.loc[:as_of]
    if len(hist) < lookback + skip:
        return pd.Series(dtype=float)
    window = hist.iloc[-(lookback + skip):]
    p_start = window.iloc[0]
    p_end = window.iloc[-(skip + 1)]
    momentum = (p_end / p_start - 1) * 100
    # 只取当日有价格的股票
    latest = hist.iloc[-1].dropna()
    return momentum.loc[momentum.index.intersection(latest.index)].dropna()


def _compute_score(fin_snap: pd.DataFrame, mom_snap: pd.Series,
                   universe: pd.Index) -> pd.Series:
    """
    合并财务 + 动量因子，计算综合得分
    财务权重 60%，动量权重 40%（去掉无法回测的机构/估值因子后重新归一化）
    """
    df = pd.DataFrame({"ts_code": universe})

    # 财务得分
    if not fin_snap.empty:
        df = df.merge(fin_snap[["ts_code", "roe", "netprofit_yoy", "grossprofit_margin"]],
                      on="ts_code", how="left")
        roe_score    = _percentile_rank(df["roe"])
        growth_score = _percentile_rank(df["netprofit_yoy"])
        margin_score = _percentile_rank(df["grossprofit_margin"])
        df["fin_score"] = (roe_score * 0.5 + growth_score * 0.3 + margin_score * 0.2).fillna(50)
    else:
        df["fin_score"] = 50.0

    # 动量得分
    mom_df = mom_snap.rename("momentum").reset_index()
    mom_df.columns = ["ts_code", "momentum"]
    df = df.merge(mom_df, on="ts_code", how="left")
    df["mom_score"] = _percentile_rank(df["momentum"]).fillna(50)

    # 综合得分：财务 60% + 动量 40%
    df["score"] = df["fin_score"] * 0.6 + df["mom_score"] * 0.4

    # 硬过滤：ROE < 0 直接设为 0 分
    if "roe" in df.columns:
        df.loc[df["roe"] < 0, "score"] = 0

    return df.set_index("ts_code")["score"]


def build_factor_scores(
    price_df: pd.DataFrame,
    rebal_dates: List[date],
    min_stocks: int = 50,
) -> pd.DataFrame:
    """
    构建历史因子得分矩阵
    返回 DataFrame: index=调仓日(Timestamp), columns=ts_code, values=综合得分

    每个调仓日独立计算，只使用当时可用数据（无前瞻偏差）
    """
    print("[因子] 加载财务数据...")
    fin_df = _load_financial_db()
    print(f"[因子] 财务记录: {len(fin_df)} 行，覆盖 {fin_df['ts_code'].nunique()} 只股票")

    universe = price_df.columns  # 全量股票池
    results = {}

    for i, d in enumerate(rebal_dates):
        ts_d = pd.Timestamp(d)

        # 1. 财务快照（截止调仓日，5个月披露延迟）
        fin_snap = _financial_snapshot(fin_df, ts_d)

        # 2. 动量快照（截止调仓日）
        mom_snap = _momentum_snapshot(price_df, ts_d)

        # 可用股票：动量 + 财务都有的
        avail = universe.intersection(mom_snap.index)
        if len(avail) < min_stocks:
            continue

        # 3. 合并打分
        scores = _compute_score(fin_snap, mom_snap, avail)
        results[ts_d] = scores

        if (i + 1) % 5 == 0 or i == 0:
            print(f"[因子] 调仓日 {d}：{len(scores)} 只股票打分完成 "
                  f"（财务覆盖 {len(fin_snap)} 只）")

    if not results:
        return pd.DataFrame()

    factor_scores = pd.DataFrame(results).T
    factor_scores.index.name = "date"
    print(f"\n[因子] 历史因子矩阵: {factor_scores.shape[0]} 个调仓日 × {factor_scores.shape[1]} 只股票")
    return factor_scores

"""
模块1：量化基金筛选器
从公募基金中筛选量化/指数增强/量化对冲基金，按绩效指标排名
"""
import time
import akshare as ak
import pandas as pd
import numpy as np
from typing import Optional

# 量化基金关键词（基金名称含这些词认定为量化）
QUANT_KEYWORDS = [
    "量化", "指数增强", "阿尔法", "alpha", "Alpha",
    "多因子", "Smart Beta", "smartbeta", "因子",
    "量化选股", "量化对冲", "中性",
]

# 主流宽基指数增强基金（手工维护，确保不漏掉核心品种）
KNOWN_QUANT_FUNDS = {
    # 沪深300增强
    "110003": "易方达沪深300量化增强",
    "000311": "富国沪深300增强",
    "070023": "嘉实沪深300量化增强",
    "160706": "嘉实沪深300增强",
    "002510": "工银沪深300增强",
    # 中证500增强
    "004997": "天弘中证500指数增强",
    "001643": "汇添富中证500量化增强",
    "007474": "富国中证500增强",
    "006560": "华夏中证500量化增强",
    # 中证1000增强
    "013309": "富国中证1000增强",
    "016630": "天弘中证1000指数增强",
    # 全市场量化选股
    "000689": "前海开源量化核心",
    "001917": "招商量化精选",
    "002680": "万家量化睿选",
}


def screen_quant_funds(
    min_1y_return: float = 10.0,    # 近1年收益 >= 10%
    min_3y_return: float = 30.0,    # 近3年收益 >= 30%
    max_funds: int = 100,
) -> pd.DataFrame:
    """
    从天天基金业绩排行筛选量化基金
    返回按近1年收益降序排列的量化基金列表
    """
    print("[筛选] 拉取公募股票型基金业绩排行...")
    results = []

    for fund_type in ["股票型", "混合型"]:
        try:
            df = ak.fund_open_fund_rank_em(symbol=fund_type)
            df["fund_type_cat"] = fund_type
            results.append(df)
            time.sleep(1.0)
        except Exception as e:
            print(f"  {fund_type} 失败: {e}")

    if not results:
        return pd.DataFrame()

    df = pd.concat(results, ignore_index=True)

    # ── 数值清洗 ──────────────────────────────────────────
    for col in ["近1年", "近3年", "近6月", "近3月", "成立来"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 识别量化基金 ──────────────────────────────────────
    def is_quant(name: str) -> bool:
        if not isinstance(name, str):
            return False
        return any(kw.lower() in name.lower() for kw in QUANT_KEYWORDS)

    df["is_quant"] = df["基金简称"].apply(is_quant)

    # 加入手工维护的量化基金列表
    known_codes = set(KNOWN_QUANT_FUNDS.keys())
    df.loc[df["基金代码"].isin(known_codes), "is_quant"] = True

    quant_df = df[df["is_quant"]].copy()

    # ── 绩效过滤 ──────────────────────────────────────────
    if "近1年" in quant_df.columns:
        quant_df = quant_df[quant_df["近1年"] >= min_1y_return]
    if "近3年" in quant_df.columns:
        # 近3年可能为空（成立不足3年），保留空值
        mask = (quant_df["近3年"].isna()) | (quant_df["近3年"] >= min_3y_return)
        quant_df = quant_df[mask]

    # ── 整理输出 ──────────────────────────────────────────
    keep_cols = [c for c in [
        "基金代码", "基金简称", "基金类型", "日期",
        "单位净值", "累计净值", "近1周", "近1月",
        "近3月", "近6月", "近1年", "近2年", "近3年", "成立来"
    ] if c in quant_df.columns]

    quant_df = (quant_df[keep_cols]
                .sort_values("近1年", ascending=False, na_position="last")
                .head(max_funds)
                .reset_index(drop=True))

    print(f"[筛选] 找到量化基金: {len(quant_df)} 只")
    return quant_df


def get_fund_performance_detail(fund_code: str, fund_name: str = "") -> dict:
    """
    获取单只基金的详细绩效指标（夏普、最大回撤等）
    通过净值历史计算
    """
    try:
        df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
        df = df.rename(columns={"净值日期": "date", "单位净值": "nav"})
        df["date"] = pd.to_datetime(df["date"])
        df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
        df = df.dropna(subset=["nav"]).sort_values("date")

        if len(df) < 60:
            return {"fund_code": fund_code, "fund_name": fund_name, "error": "数据不足"}

        # 计算日收益率
        df["ret"] = df["nav"].pct_change()
        ret = df["ret"].dropna()

        # 年化收益
        days = (df["date"].max() - df["date"].min()).days
        total_ret = df["nav"].iloc[-1] / df["nav"].iloc[0] - 1
        ann_ret = (1 + total_ret) ** (365 / days) - 1 if days > 0 else 0

        # 最大回撤
        nav_series = df["nav"].values
        cummax = np.maximum.accumulate(nav_series)
        drawdown = (nav_series - cummax) / cummax
        max_drawdown = drawdown.min()

        # 夏普（假设无风险利率2.5%）
        rf_daily = 0.025 / 252
        excess_ret = ret - rf_daily
        sharpe = (excess_ret.mean() / excess_ret.std() * np.sqrt(252)
                  if excess_ret.std() > 0 else 0)

        # 卡玛比率
        calmar = ann_ret / abs(max_drawdown) if max_drawdown != 0 else 0

        # 近1年
        one_year_ago = df["date"].max() - pd.Timedelta(days=365)
        df_1y = df[df["date"] >= one_year_ago]
        ret_1y = (df_1y["nav"].iloc[-1] / df_1y["nav"].iloc[0] - 1
                  if len(df_1y) > 1 else None)

        return {
            "fund_code":    fund_code,
            "fund_name":    fund_name,
            "ann_return":   round(ann_ret * 100, 2),
            "max_drawdown": round(max_drawdown * 100, 2),
            "sharpe":       round(sharpe, 3),
            "calmar":       round(calmar, 3),
            "return_1y":    round(ret_1y * 100, 2) if ret_1y else None,
            "start_date":   str(df["date"].min().date()),
            "nav_latest":   round(df["nav"].iloc[-1], 4),
            "total_days":   days,
        }
    except Exception as e:
        return {"fund_code": fund_code, "fund_name": fund_name, "error": str(e)}

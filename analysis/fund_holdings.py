"""
模块2：基金持仓爬取与聚合
从天天基金获取季报持仓，分析机构集中持仓的股票
"""
import time
import akshare as ak
import pandas as pd
from typing import Optional


# 天天基金持仓接口支持的年份（按年取4个季度）
AVAILABLE_YEARS = ["2024", "2023", "2022", "2021", "2020"]


def fetch_fund_holdings(
    fund_code: str,
    years: list[str] = None,
) -> pd.DataFrame:
    """
    获取单只基金的历史持仓（按季度）
    返回字段: fund_code, quarter, stock_code, stock_name, weight, shares, market_value
    """
    if years is None:
        years = AVAILABLE_YEARS[:2]  # 默认拉最近2年

    all_dfs = []
    for year in years:
        try:
            df = ak.fund_portfolio_hold_em(symbol=fund_code, date=year)
            if df.empty:
                continue
            df = df.rename(columns={
                "序号":   "rank",
                "股票代码": "stock_code",
                "股票名称": "stock_name",
                "占净值比例": "weight",
                "持股数":  "shares",
                "持仓市值": "market_value",
                "季度":   "quarter",
            })
            df["fund_code"] = fund_code
            all_dfs.append(df[["fund_code", "quarter", "stock_code",
                               "stock_name", "weight", "shares", "market_value"]])
            time.sleep(0.5)
        except Exception as e:
            pass

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce")
    return df


def fetch_all_funds_holdings(
    fund_codes: list[str],
    years: list[str] = None,
    delay: float = 1.0,
) -> pd.DataFrame:
    """
    批量获取多只基金持仓并合并
    """
    all_dfs = []
    total = len(fund_codes)
    for i, code in enumerate(fund_codes):
        df = fetch_fund_holdings(code, years=years)
        if not df.empty:
            all_dfs.append(df)
        if (i + 1) % 10 == 0:
            print(f"  持仓进度: {i+1}/{total}")
        time.sleep(delay)

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def aggregate_holdings(holdings_df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """
    聚合分析：统计各股票被多少量化基金持有，以及平均权重
    核心逻辑：被更多基金持有 = 更受机构认可

    返回列:
        stock_code, stock_name,
        fund_count      - 持有该股的基金数量
        avg_weight      - 平均持仓权重（%）
        total_weight    - 总持仓权重之和
        max_weight      - 最大单只基金持仓权重
        quarters        - 出现的季度列表
    """
    if holdings_df.empty:
        return pd.DataFrame()

    # 取最新季度（每只基金）
    latest = (holdings_df
              .sort_values("quarter", ascending=False)
              .groupby(["fund_code", "stock_code"])
              .first()
              .reset_index())

    agg = (latest
           .groupby(["stock_code", "stock_name"])
           .agg(
               fund_count=("fund_code",  "nunique"),
               avg_weight=("weight",     "mean"),
               total_weight=("weight",   "sum"),
               max_weight=("weight",     "max"),
               total_market_value=("market_value", "sum"),
           )
           .reset_index()
           .sort_values("fund_count", ascending=False)
           .head(top_n))

    agg = agg.round({"avg_weight": 2, "total_weight": 2, "max_weight": 2})
    return agg.reset_index(drop=True)


def holdings_overlap_matrix(holdings_df: pd.DataFrame) -> pd.DataFrame:
    """
    基金间持仓重叠度矩阵
    overlap(A,B) = |A∩B| / |A∪B|（Jaccard相似度）
    相似度高 = 策略相似，可用于基金聚类
    """
    if holdings_df.empty:
        return pd.DataFrame()

    # 每只基金取其最新季度的持仓
    fund_latest_q = (holdings_df.groupby("fund_code")["quarter"].max().rename("latest_q"))
    latest = (holdings_df
              .join(fund_latest_q, on="fund_code")
              .query("quarter == latest_q")
              .drop(columns=["latest_q"]))

    # 每只基金取最新季度的前20大持仓
    def top20(grp):
        return set(grp.sort_values("weight", ascending=False).head(20)["stock_code"])

    fund_stocks = (latest
                   .groupby("fund_code")
                   .apply(top20)
                   .to_dict())

    funds = list(fund_stocks.keys())
    n = len(funds)
    matrix = pd.DataFrame(index=funds, columns=funds, dtype=float)

    for i in range(n):
        for j in range(n):
            a, b = fund_stocks[funds[i]], fund_stocks[funds[j]]
            union = a | b
            inter = a & b
            matrix.iloc[i, j] = len(inter) / len(union) if union else 0.0

    return matrix.round(3)


def holding_consistency(
    fund_code: str,
    holdings_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    分析单只基金持仓稳定性：
    某只股票在几个季度都出现 = 基金长期看好
    """
    fund_holdings = holdings_df[holdings_df["fund_code"] == fund_code].copy()
    if fund_holdings.empty:
        return pd.DataFrame()

    quarters = sorted(fund_holdings["quarter"].unique())
    n_quarters = len(quarters)

    consistency = (fund_holdings
                   .groupby(["stock_code", "stock_name"])
                   .agg(
                       appear_count=("quarter", "nunique"),
                       avg_weight=("weight", "mean"),
                       latest_weight=("weight", "last"),
                   )
                   .reset_index())

    consistency["consistency_rate"] = (
        consistency["appear_count"] / n_quarters * 100
    ).round(1)

    return (consistency
            .sort_values(["appear_count", "avg_weight"], ascending=False)
            .head(top_n)
            .reset_index(drop=True))

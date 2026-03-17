"""
模块3：策略因子归因
通过基金持仓，分析其隐含的选股因子偏好
（简化版 Barra 归因：无需购买商业数据）
"""
import akshare as ak
import pandas as pd
import numpy as np
from typing import Optional
from sqlalchemy import select, text
from data.storage.db import get_engine, stock_basic, financial_indicator, daily_basic


# ══════════════════════════════════════════════════════════
# 1. 个股因子计算
# ══════════════════════════════════════════════════════════

def get_stock_factors(stock_codes: list[str]) -> pd.DataFrame:
    """
    为一批股票计算多维度因子值
    数据来源：AKShare 实时/近期数据
    """
    records = []
    for code in stock_codes:
        symbol = code.replace(".SZ", "").replace(".SH", "")
        record = {"stock_code": code, "symbol": symbol}

        # ── 估值因子 ──────────────────────────────────────
        try:
            info = ak.stock_individual_info_em(symbol=symbol)
            info_dict = dict(zip(info["item"], info["value"]))
            record["pe_ttm"]  = _safe_float(info_dict.get("市盈率(TTM)"))
            record["pb"]      = _safe_float(info_dict.get("市净率"))
            record["total_mv"] = _safe_float(info_dict.get("总市值"))   # 元
        except Exception:
            pass

        # ── 财务因子（最新年报） ──────────────────────────────
        try:
            fin = ak.stock_financial_abstract_ths(symbol=symbol, indicator="按年度")
            if not fin.empty:
                latest = fin.iloc[0]
                record["roe"] = _safe_float(
                    str(latest.get("净资产收益率", "")).replace("%", ""))
                record["netprofit_yoy"] = _safe_float(
                    str(latest.get("净利润增长率", "")).replace("%", ""))
                record["revenue_yoy"] = _safe_float(
                    str(latest.get("营业总收入增长率", "")).replace("%", ""))
                record["gross_margin"] = _safe_float(
                    str(latest.get("毛利率", "")).replace("%", ""))
        except Exception:
            pass

        records.append(record)

    df = pd.DataFrame(records)
    return df


def get_stock_momentum(
    stock_codes: list[str],
    period_days: int = 120,
) -> pd.DataFrame:
    """
    计算动量因子（过去N个交易日的收益率）
    使用数据库中的历史行情
    """
    engine = get_engine()
    records = []

    for code in stock_codes:
        try:
            with engine.connect() as conn:
                rows = conn.execute(text(
                    """SELECT trade_date, close FROM stock_daily
                       WHERE ts_code = :code
                       ORDER BY trade_date DESC LIMIT :n"""
                ), {"code": code, "n": period_days + 5}).fetchall()

            if len(rows) < period_days:
                records.append({"stock_code": code, "momentum": None})
                continue

            prices = [r[1] for r in reversed(rows)]
            momentum = (prices[-1] / prices[0] - 1) * 100
            # 去掉最近1个月（避免反转效应）
            prices_skip1m = prices[:-20] if len(prices) > 25 else prices
            momentum_adj = (prices[-1] / prices_skip1m[0] - 1) * 100 if prices_skip1m else None

            records.append({
                "stock_code":    code,
                "momentum_120d": round(momentum, 2),
                "momentum_adj":  round(momentum_adj, 2) if momentum_adj else None,
            })
        except Exception:
            records.append({"stock_code": code, "momentum_120d": None})

    return pd.DataFrame(records)


def _safe_float(val) -> Optional[float]:
    try:
        return float(str(val).replace(",", "").strip())
    except Exception:
        return None


# ══════════════════════════════════════════════════════════
# 2. 基金持仓归因分析
# ══════════════════════════════════════════════════════════

def attribute_fund_style(
    holdings_df: pd.DataFrame,
    fund_code: str,
    factors_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    对单只基金的持仓做风格归因
    返回：各因子的平均值 vs 全市场中位数 → 判断基金偏好
    """
    fund_h = holdings_df[holdings_df["fund_code"] == fund_code].copy()
    # 取最新季度
    if fund_h.empty:
        return {}
    latest_q = fund_h["quarter"].max()
    fund_h = fund_h[fund_h["quarter"] == latest_q]

    if factors_df is None or factors_df.empty:
        return {"fund_code": fund_code, "error": "需要先运行因子计算"}

    # 将持仓股票 codes 转换为统一格式
    fund_h["stock_code_clean"] = fund_h["stock_code"].apply(
        lambda x: f"{x}.SH" if x.startswith("6") else f"{x}.SZ"
    )
    merged = fund_h.merge(
        factors_df, left_on="stock_code_clean", right_on="stock_code", how="left"
    )

    # 加权平均（按持仓权重）
    result = {"fund_code": fund_code, "quarter": latest_q}
    numeric_factors = ["pe_ttm", "pb", "roe", "netprofit_yoy",
                       "gross_margin", "momentum_120d", "total_mv"]

    for fac in numeric_factors:
        if fac not in merged.columns:
            continue
        valid = merged[["weight", fac]].dropna()
        if valid.empty:
            continue
        w_sum = valid["weight"].sum()
        if w_sum > 0:
            result[f"wt_{fac}"] = round(
                (valid["weight"] * valid[fac]).sum() / w_sum, 3
            )

    # 市值风格判断
    if "wt_total_mv" in result:
        mv = result["wt_total_mv"]
        if mv > 500e8:
            result["style_size"] = "大盘"
        elif mv > 100e8:
            result["style_size"] = "中盘"
        else:
            result["style_size"] = "小盘"

    # 估值风格判断
    if "wt_pe_ttm" in result:
        pe = result["wt_pe_ttm"]
        if pe < 20:
            result["style_value"] = "价值"
        elif pe < 40:
            result["style_value"] = "均衡"
        else:
            result["style_value"] = "成长"

    return result


# ══════════════════════════════════════════════════════════
# 3. 行业分布分析
# ══════════════════════════════════════════════════════════

def analyze_sector_distribution(holdings_df: pd.DataFrame) -> pd.DataFrame:
    """
    分析基金持仓的行业分布
    通过查询 AKShare 个股所属行业
    """
    if holdings_df.empty:
        return pd.DataFrame()

    # 去重取所有持仓股票
    fund_latest_q = holdings_df.groupby("fund_code")["quarter"].max().rename("latest_q")
    latest = (holdings_df
              .join(fund_latest_q, on="fund_code")
              .query("quarter == latest_q")
              .drop(columns=["latest_q"])
              .copy())
    all_stocks = latest["stock_code"].unique()

    print(f"  查询 {len(all_stocks)} 只持仓股票的行业信息...")

    # 从 AKShare 获取行业分类（申万一级）
    try:
        sw_df = ak.stock_board_industry_name_em()
        # 这个接口返回行业板块，不是个股映射，用于参考
    except Exception:
        sw_df = pd.DataFrame()

    # 简化：通过个股基本面接口获取行业
    sector_map = {}
    for i, code in enumerate(all_stocks[:100]):  # 限制100只，避免太慢
        symbol = code.replace(".SZ", "").replace(".SH", "")
        try:
            info = ak.stock_individual_info_em(symbol=symbol)
            info_dict = dict(zip(info["item"], info["value"]))
            sector_map[code] = info_dict.get("行业", "未知")
            if (i + 1) % 20 == 0:
                print(f"    行业查询进度: {i+1}/{min(len(all_stocks), 100)}")
        except Exception:
            sector_map[code] = "未知"

    latest["sector"] = latest["stock_code"].map(sector_map).fillna("未知")

    sector_agg = (latest
                  .groupby("sector")
                  .agg(
                      stock_count=("stock_code", "nunique"),
                      fund_count=("fund_code", "nunique"),
                      avg_weight=("weight", "mean"),
                      total_weight=("weight", "sum"),
                  )
                  .sort_values("fund_count", ascending=False)
                  .reset_index())

    return sector_agg


# ══════════════════════════════════════════════════════════
# 4. 买卖信号（季报对比）
# ══════════════════════════════════════════════════════════

def detect_holding_changes(
    holdings_df: pd.DataFrame,
    fund_code: str,
) -> dict:
    """
    对比相邻季度持仓变化，识别机构加仓/减仓信号
    """
    fund_h = holdings_df[holdings_df["fund_code"] == fund_code].copy()
    quarters = sorted(fund_h["quarter"].unique())

    if len(quarters) < 2:
        return {"fund_code": fund_code, "error": "季度数据不足"}

    q_latest = quarters[-1]
    q_prev   = quarters[-2]

    latest_stocks = set(fund_h[fund_h["quarter"] == q_latest]["stock_code"])
    prev_stocks   = set(fund_h[fund_h["quarter"] == q_prev]["stock_code"])

    new_buys  = latest_stocks - prev_stocks
    full_sell = prev_stocks   - latest_stocks
    hold_both = latest_stocks & prev_stocks

    # 加仓/减仓
    def get_weight(stocks, quarter):
        sub = fund_h[(fund_h["quarter"] == quarter) &
                     (fund_h["stock_code"].isin(stocks))][["stock_code", "stock_name", "weight"]]
        return sub.set_index("stock_code")

    latest_w = get_weight(hold_both, q_latest)
    prev_w   = get_weight(hold_both, q_prev)
    weight_change = (latest_w["weight"] - prev_w["weight"]).dropna()
    increased = weight_change[weight_change > 0.3].sort_values(ascending=False)
    decreased = weight_change[weight_change < -0.3].sort_values()

    return {
        "fund_code":    fund_code,
        "quarter_from": q_prev,
        "quarter_to":   q_latest,
        "new_buys":     sorted(new_buys),
        "full_sells":   sorted(full_sell),
        "increased":    increased.to_dict(),
        "decreased":    decreased.to_dict(),
    }

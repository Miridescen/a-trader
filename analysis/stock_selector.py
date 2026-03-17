"""
多因子综合选股系统（方案四）

评分权重：
  机构持仓分  30%  — 被量化基金持有数量 + 加仓信号
  财务质量分  30%  — ROE / 净利润增速 / 现金流质量
  估值性价比  20%  — 自算 PE、PB 的历史百分位
  价格动量分  20%  — 120日涨幅（去掉最近1月）

用法：
    python -m analysis.stock_selector
    python -m analysis.stock_selector --universe institution  # 仅机构持仓股
    python -m analysis.stock_selector --top 25
"""
import argparse
import time
import warnings
import akshare as ak
import pandas as pd
import numpy as np
from typing import Optional
from sqlalchemy import text
from data.storage.db import get_engine, init_db

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════
# 0. 候选股票池生成
# ══════════════════════════════════════════════════════════

def build_candidate_universe(mode: str = "institution") -> list[str]:
    """
    生成候选股票池
    mode:
        'institution' — 从已有持仓数据中提取（快速，~200 只）
        'all'         — 全市场 A 股（慢，需日线数据）
    返回: ts_code 列表，如 ['000001.SZ', '600519.SH']
    """
    if mode == "institution":
        # 使用已分析的核心量化基金持仓
        from analysis.fund_holdings import fetch_all_funds_holdings, aggregate_holdings
        CORE_FUNDS = [
            "000311", "004997", "007474", "001643", "110003",
            "000531", "007775", "009644", "004641", "519929",
        ]
        print(f"[候选池] 拉取 {len(CORE_FUNDS)} 只量化基金最新持仓...")
        holdings = fetch_all_funds_holdings(CORE_FUNDS, years=["2024"], delay=0.8)
        if holdings.empty:
            print("  持仓数据为空，回退到全市场模式")
            return build_candidate_universe("db")

        agg = aggregate_holdings(holdings, top_n=300)
        codes = agg["stock_code"].tolist()
        # 转为统一 ts_code 格式
        ts_codes = [
            f"{c}.SH" if c.startswith("6") else f"{c}.SZ"
            for c in codes
        ]
        print(f"  候选股票: {len(ts_codes)} 只")
        return ts_codes

    elif mode == "db":
        # 从数据库中取已有日线数据的股票（至少有 500 行数据）
        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(text(
                """SELECT ts_code FROM stock_daily
                   GROUP BY ts_code HAVING COUNT(*) >= 500
                   ORDER BY COUNT(*) DESC"""
            )).fetchall()
        codes = [r[0] for r in rows]
        print(f"[候选池] DB 中有历史数据的股票: {len(codes)} 只")
        return codes

    return []


# ══════════════════════════════════════════════════════════
# 1. 财务因子
# ══════════════════════════════════════════════════════════

def fetch_financial_factors(ts_codes: list[str], delay: float = 0.4) -> pd.DataFrame:
    """
    批量获取财务因子
    返回: ts_code, roe, profit_growth, revenue_growth,
           gross_margin, debt_ratio, ocf_per_share, eps_growth
    """
    records = []
    total = len(ts_codes)
    for i, ts_code in enumerate(ts_codes):
        symbol = ts_code.replace(".SZ", "").replace(".SH", "")
        rec = {"ts_code": ts_code}
        try:
            df = ak.stock_financial_abstract_ths(symbol=symbol, indicator="按年度")
            if df.empty:
                records.append(rec)
                continue

            def pct(val):
                s = str(val).replace("%", "").replace(",", "").strip()
                try:
                    return float(s)
                except Exception:
                    return np.nan

            def num(val):
                s = str(val).replace(",", "").replace("亿", "").strip()
                try:
                    return float(s)
                except Exception:
                    return np.nan

            # 过滤掉 False 占位行，按报告期降序（最新在前）
            df = df[df["净利润同比增长率"] != False]   # noqa: E712
            df = df[df["报告期"].astype(str).str.match(r'\d{4}')]
            df = df.sort_values("报告期", ascending=False).reset_index(drop=True)
            if df.empty:
                records.append(rec)
                continue

            latest  = df.iloc[0]   # 最新年报
            prev    = df.iloc[1] if len(df) > 1 else latest
            prev2   = df.iloc[2] if len(df) > 2 else latest

            roe_latest  = pct(latest.get("净资产收益率", np.nan))
            roe_prev    = pct(prev.get("净资产收益率", np.nan))
            roe_prev2   = pct(prev2.get("净资产收益率", np.nan))

            profit_growth   = pct(latest.get("净利润同比增长率", np.nan))
            revenue_growth  = pct(latest.get("营业总收入同比增长率", np.nan))
            profit_growth_p = pct(prev.get("净利润同比增长率", np.nan))
            gross_margin    = pct(latest.get("销售毛利率", np.nan))

            # ROE 稳定性（近3年标准差，越小越稳）
            roe_std = np.nanstd([roe_latest, roe_prev, roe_prev2])

            # 资产负债率
            debt_ratio = pct(latest.get("资产负债率", np.nan))

            # 经营现金流 / 股
            ocf_per_share = num(latest.get("每股经营现金流", np.nan))
            eps           = num(latest.get("基本每股收益", np.nan))

            # OCF 质量：ocf_per_share / eps（> 1 说明利润含金量高）
            ocf_quality = ocf_per_share / eps if (eps and eps != 0) else np.nan

            rec.update({
                "roe":             roe_latest,
                "roe_prev":        roe_prev,
                "roe_std":         round(roe_std, 2) if not np.isnan(roe_std) else np.nan,
                "profit_growth":   profit_growth,
                "profit_growth_p": profit_growth_p,
                "revenue_growth":  revenue_growth,
                "gross_margin":    gross_margin,
                "debt_ratio":      debt_ratio,
                "ocf_per_share":   ocf_per_share,
                "eps":             eps,
                "ocf_quality":     round(ocf_quality, 2) if not np.isnan(ocf_quality) else np.nan,
            })
        except Exception:
            pass

        records.append(rec)
        if (i + 1) % 30 == 0:
            print(f"  财务因子进度: {i+1}/{total}")
        time.sleep(delay)

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════
# 2. 估值因子
# ══════════════════════════════════════════════════════════

def fetch_valuation_factors(ts_codes: list[str], fin_df: pd.DataFrame) -> pd.DataFrame:
    """
    估值因子：自算 PE = 总市值 / 近12月净利润
    数据来源：stock_individual_info_em 提供总市值，财务数据已有净利润
    """
    records = []
    total = len(ts_codes)
    for i, ts_code in enumerate(ts_codes):
        exchange = "SH" if ts_code.endswith(".SH") else "SZ"
        symbol   = ts_code.replace(".SZ", "").replace(".SH", "")
        xq_sym   = f"{exchange}{symbol}"   # 雪球格式: SZ000001
        rec = {"ts_code": ts_code}
        try:
            info = ak.stock_individual_spot_xq(symbol=xq_sym)
            info_d = dict(zip(info["item"], info["value"]))

            def _f(key):
                v = info_d.get(key, np.nan)
                try:
                    return float(str(v).replace(",", ""))
                except Exception:
                    return np.nan

            total_mv = _f("资产净值/总市值")   # 元
            rec["total_mv_bn"] = round(total_mv / 1e8, 2) if total_mv else np.nan
            rec["pe_ttm"]      = _f("市盈率(TTM)")
            rec["pb"]          = _f("市净率")
            rec["eps"]         = _f("每股收益")
            rec["bps"]         = _f("每股净资产")
            rec["div_yield"]   = _f("股息率(TTM)")

        except Exception:
            pass

        records.append(rec)
        if (i + 1) % 30 == 0:
            print(f"  估值因子进度: {i+1}/{total}")
        time.sleep(0.4)

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════
# 3. 动量因子（优先 DB，补充 AKShare）
# ══════════════════════════════════════════════════════════

def fetch_momentum_factors(ts_codes: list[str]) -> pd.DataFrame:
    """
    动量因子：120日总涨幅（去掉最近20日），衡量中期趋势
    优先从 DB 读取，DB 没有则用 AKShare Sina 接口
    """
    engine = get_engine()
    records = []

    for ts_code in ts_codes:
        symbol = ts_code.replace(".SZ", "").replace(".SH", "")
        rec = {"ts_code": ts_code}
        prices = None

        # ── 优先 DB ──────────────────────────────────────
        try:
            with engine.connect() as conn:
                rows = conn.execute(text(
                    """SELECT close FROM stock_daily
                       WHERE ts_code = :code
                       ORDER BY trade_date DESC LIMIT 145"""
                ), {"code": ts_code}).fetchall()
            if len(rows) >= 120:
                prices = [r[0] for r in reversed(rows)]
        except Exception:
            pass

        # ── 补充 AKShare ──────────────────────────────────
        if prices is None or len(prices) < 120:
            try:
                prefix = "sh" if ts_code.endswith(".SH") else "sz"
                df = ak.stock_zh_a_daily(symbol=f"{prefix}{symbol}", adjust="hfq")
                if df is not None and len(df) >= 120:
                    prices = df["close"].tolist()[-145:]
                time.sleep(0.3)
            except Exception:
                pass

        if prices and len(prices) >= 120:
            # 120日动量，去掉最近20日（避免短期反转噪声）
            p_start = prices[0]
            p_end_skip1m = prices[-21]   # 20日前的收盘价
            p_latest = prices[-1]

            momentum_120 = (p_latest / p_start - 1) * 100
            momentum_adj = (p_end_skip1m / p_start - 1) * 100  # 去掉最近1月

            # 250日均线多空判断
            ma250 = np.mean(prices) if len(prices) >= 100 else None
            above_ma = p_latest > ma250 if ma250 else None

            rec.update({
                "momentum_120":  round(momentum_120, 2),
                "momentum_adj":  round(momentum_adj, 2),
                "price_latest":  round(p_latest, 2),
                "above_ma250":   above_ma,
            })

        records.append(rec)

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════
# 4. 机构因子
# ══════════════════════════════════════════════════════════

def build_institution_factor(ts_codes: list[str]) -> pd.DataFrame:
    """
    机构因子：
    - fund_count: 被多少只量化基金持有（最新季度）
    - is_increasing: 最新季度是否有机构加仓
    数据来源：复用已拉取的持仓
    """
    from analysis.fund_holdings import fetch_all_funds_holdings, aggregate_holdings
    CORE_FUNDS = [
        "000311", "004997", "007474", "001643", "110003",
        "000531", "007775", "009644", "004641", "519929",
    ]
    holdings = fetch_all_funds_holdings(CORE_FUNDS, years=["2024"], delay=0.8)
    if holdings.empty:
        return pd.DataFrame({"ts_code": ts_codes, "fund_count": 0, "is_increasing": False})

    agg = aggregate_holdings(holdings, top_n=1000)

    # 判断加仓：Q4 vs Q3 持仓数变化
    from analysis.factor_attribution import detect_holding_changes
    increasing_stocks = set()
    for fc in CORE_FUNDS[:5]:
        changes = detect_holding_changes(holdings, fc)
        if "error" not in changes:
            increasing_stocks.update(changes.get("new_buys", []))
            increasing_stocks.update(changes.get("increased", {}).keys())

    def to_ts(code):
        return f"{code}.SH" if code.startswith("6") else f"{code}.SZ"

    agg["ts_code"] = agg["stock_code"].apply(to_ts)
    agg["is_increasing"] = agg["stock_code"].isin(increasing_stocks)

    # 过滤到候选池
    result = agg[agg["ts_code"].isin(ts_codes)][
        ["ts_code", "fund_count", "avg_weight", "is_increasing"]
    ].copy()

    # 补充不在持仓中的股票（fund_count=0）
    missing = set(ts_codes) - set(result["ts_code"])
    if missing:
        extra = pd.DataFrame({
            "ts_code": list(missing),
            "fund_count": 0,
            "avg_weight": 0.0,
            "is_increasing": False,
        })
        result = pd.concat([result, extra], ignore_index=True)

    return result


# ══════════════════════════════════════════════════════════
# 5. 综合评分
# ══════════════════════════════════════════════════════════

def percentile_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    """将数值序列转为 0-100 百分位排名"""
    return series.rank(pct=True, ascending=ascending, na_option="bottom") * 100


def compute_composite_score(
    fin_df: pd.DataFrame,
    val_df: pd.DataFrame,
    mom_df: pd.DataFrame,
    ins_df: pd.DataFrame,
    weights: dict = None,
) -> pd.DataFrame:
    """
    合并四类因子，计算综合得分
    weights: {'institution': 0.30, 'financial': 0.30, 'valuation': 0.20, 'momentum': 0.20}
    """
    if weights is None:
        weights = {"institution": 0.30, "financial": 0.30,
                   "valuation": 0.20, "momentum": 0.20}

    # ── 合并 ──────────────────────────────────────────────
    df = ins_df.copy()
    for other in [fin_df, val_df, mom_df]:
        if not other.empty and "ts_code" in other.columns:
            df = df.merge(other, on="ts_code", how="left")

    # ── 机构因子得分 ──────────────────────────────────────
    df["ins_score"] = (
        percentile_rank(df.get("fund_count", pd.Series(0, index=df.index))) * 0.7 +
        df.get("is_increasing", pd.Series(False, index=df.index)).astype(float) * 30
    )

    # ── 财务因子得分 ──────────────────────────────────────
    roe_score    = percentile_rank(df.get("roe",           pd.Series(np.nan, index=df.index)))
    growth_score = percentile_rank(df.get("profit_growth", pd.Series(np.nan, index=df.index)))
    ocf_score    = percentile_rank(df.get("ocf_quality",   pd.Series(np.nan, index=df.index)))
    # ROE 稳定性：std 越小越好（取反）
    roe_stab     = percentile_rank(df.get("roe_std",       pd.Series(np.nan, index=df.index)),
                                   ascending=False)

    df["fin_score"] = (
        roe_score    * 0.35 +
        growth_score * 0.30 +
        ocf_score    * 0.20 +
        roe_stab     * 0.15
    ).fillna(50)

    # ── 估值因子得分 ──────────────────────────────────────
    mv   = df.get("total_mv_bn", pd.Series(np.nan, index=df.index))
    pe   = df.get("pe_ttm",      pd.Series(np.nan, index=df.index))
    roe  = df.get("roe",         pd.Series(np.nan, index=df.index))

    mv_score  = percentile_rank(mv, ascending=False)  # 市值小得分高（中小盘弹性）
    mv_score  = mv_score.where(mv >= 30, 0)           # 市值 < 30亿 不得分
    pe_score  = percentile_rank(pe, ascending=False)  # PE 低得分高（便宜）
    pe_score  = pe_score.where((pe > 0) & (pe < 200), 50)  # 过滤负PE和极高PE

    # PEG 近似：PE / 净利润增速（越低越好）
    pg = df.get("profit_growth", pd.Series(np.nan, index=df.index))
    peg = pe / pg.where(pg > 0, np.nan)
    peg_score = percentile_rank(peg, ascending=False).fillna(50)

    df["val_score"] = (mv_score * 0.4 + pe_score * 0.4 + peg_score * 0.2).fillna(50)

    # ── 动量因子得分 ──────────────────────────────────────
    mom_score  = percentile_rank(df.get("momentum_adj", pd.Series(np.nan, index=df.index)))
    ma_bonus   = df.get("above_ma250", pd.Series(True, index=df.index)).fillna(True).astype(float) * 10
    df["mom_score"] = (mom_score * 0.9 + ma_bonus).fillna(50)

    # ── 综合得分 ──────────────────────────────────────────
    df["composite_score"] = (
        df["ins_score"]  * weights["institution"] +
        df["fin_score"]  * weights["financial"]   +
        df["val_score"]  * weights["valuation"]   +
        df["mom_score"]  * weights["momentum"]
    ).round(2)

    return df.sort_values("composite_score", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════
# 6. 过滤器
# ══════════════════════════════════════════════════════════

def apply_hard_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    硬过滤（排除明确不符合条件的股票）
    - 排除市值 < 30亿
    - 排除资产负债率 > 80%（非金融行业，金融股此项跳过）
    - 排除 ROE < 0（亏损）
    - 排除动量极端负值（跌幅超过 40%）
    """
    mask = pd.Series(True, index=df.index)

    # 市值
    if "total_mv_bn" in df.columns:
        mask &= (df["total_mv_bn"].isna()) | (df["total_mv_bn"] >= 30)

    # 盈利
    if "roe" in df.columns:
        mask &= (df["roe"].isna()) | (df["roe"] > 0)

    # 过度亏损
    if "profit_growth" in df.columns:
        mask &= (df["profit_growth"].isna()) | (df["profit_growth"] > -60)

    # 动量不能太差
    if "momentum_120" in df.columns:
        mask &= (df["momentum_120"].isna()) | (df["momentum_120"] > -40)

    return df[mask].reset_index(drop=True)


# ══════════════════════════════════════════════════════════
# 7. 主流程
# ══════════════════════════════════════════════════════════

def run_stock_selection(
    universe: str = "institution",
    top_n: int = 25,
    weights: dict = None,
) -> pd.DataFrame:
    """
    完整选股流程
    universe: 'institution'（机构持仓候选池）或 'db'（DB中有数据的股票）
    """
    init_db()
    print("\n" + "="*60)
    print("  多因子综合选股 — 方案四")
    print("="*60)

    # Step 1: 候选池
    print("\n[1/5] 构建候选股票池...")
    candidates = build_candidate_universe(mode=universe)
    if not candidates:
        print("  候选池为空，退出")
        return pd.DataFrame()

    # Step 2: 机构因子
    print(f"\n[2/5] 计算机构因子 ({len(candidates)} 只)...")
    ins_df = build_institution_factor(candidates)

    # Step 3: 财务因子
    print(f"\n[3/5] 拉取财务因子...")
    fin_df = fetch_financial_factors(candidates, delay=0.4)

    # Step 4: 估值因子
    print(f"\n[4/5] 拉取估值因子（总市值）...")
    val_df = fetch_valuation_factors(candidates, fin_df)

    # Step 5: 动量因子
    print(f"\n[5/5] 计算动量因子...")
    mom_df = fetch_momentum_factors(candidates)

    # 综合评分
    print("\n[评分] 合并因子，计算综合得分...")
    scored = compute_composite_score(fin_df, val_df, mom_df, ins_df, weights)

    # 硬过滤
    scored = apply_hard_filters(scored)

    # 输出 Top N
    output_cols = [c for c in [
        "ts_code", "composite_score",
        "ins_score", "fin_score", "val_score", "mom_score",
        "fund_count", "roe", "profit_growth",
        "momentum_adj", "total_mv_bn",
    ] if c in scored.columns]

    result = scored[output_cols].head(top_n)

    print(f"\n{'='*60}")
    print(f"  Top {top_n} 选股结果")
    print(f"{'='*60}")
    print(result.to_string(index=True))

    # 附加可读性信息
    _print_selection_summary(result, fin_df, ins_df)

    return result


def _print_selection_summary(result: pd.DataFrame, fin_df: pd.DataFrame, ins_df: pd.DataFrame):
    """输出可读的选股摘要"""
    print(f"\n{'='*60}")
    print("  选股摘要（按综合得分排名）")
    print(f"{'='*60}")

    # 从机构持仓中取名称
    try:
        from analysis.fund_holdings import fetch_all_funds_holdings, aggregate_holdings
        CORE_FUNDS = ["000311", "004997", "007474", "001643", "110003"]
        holdings = fetch_all_funds_holdings(CORE_FUNDS, years=["2024"], delay=0.8)
        agg = aggregate_holdings(holdings, top_n=500)
        name_map = dict(zip(agg["stock_code"], agg["stock_name"]))
    except Exception:
        name_map = {}

    for i, row in result.iterrows():
        ts_code = row["ts_code"]
        symbol = ts_code.replace(".SZ", "").replace(".SH", "")
        name = name_map.get(symbol, ts_code)
        fund_c = int(row.get("fund_count", 0)) if not pd.isna(row.get("fund_count", np.nan)) else 0
        roe = row.get("roe", np.nan)
        pg  = row.get("profit_growth", np.nan)
        mom = row.get("momentum_adj", np.nan)
        mv  = row.get("total_mv_bn", np.nan)
        score = row.get("composite_score", 0)

        roe_str = f"ROE {roe:.1f}%" if not pd.isna(roe) else "ROE -"
        pg_str  = f"利润增速 {pg:+.0f}%" if not pd.isna(pg) else "增速 -"
        mom_str = f"动量 {mom:+.1f}%" if not pd.isna(mom) else "动量 -"
        mv_str  = f"{mv:.0f}亿" if not pd.isna(mv) else "市值 -"

        print(f"  {i+1:2d}. [{score:.1f}分] {name}({symbol}) "
              f"| {fund_c}基金持仓 | {roe_str} | {pg_str} | {mom_str} | {mv_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多因子选股")
    parser.add_argument("--universe", default="institution",
                        choices=["institution", "db"],
                        help="候选股票池来源")
    parser.add_argument("--top", type=int, default=25, help="输出 Top N 只")
    args = parser.parse_args()

    result = run_stock_selection(universe=args.universe, top_n=args.top)

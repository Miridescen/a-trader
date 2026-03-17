"""
基金分析报告生成器
整合筛选、持仓、归因，输出完整分析报告
用法:
    python -m analysis.report
    python -m analysis.report --fund 000311
"""
import argparse
import time
import pandas as pd
import numpy as np
from analysis.fund_screener import screen_quant_funds, get_fund_performance_detail, KNOWN_QUANT_FUNDS
from analysis.fund_holdings import (
    fetch_fund_holdings, fetch_all_funds_holdings,
    aggregate_holdings, holding_consistency, holdings_overlap_matrix
)
from analysis.factor_attribution import detect_holding_changes


def _hr(title: str, width: int = 60):
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def run_full_report(max_funds: int = 20):
    """运行完整基金分析报告"""

    # ── Step 1: 筛选量化基金 ──────────────────────────────
    _hr("Step 1: 量化基金筛选")
    quant_funds = screen_quant_funds(min_1y_return=10.0, min_3y_return=20.0, max_funds=max_funds)

    if quant_funds.empty:
        print("  未找到符合条件的量化基金，使用手工维护列表")
        quant_funds = pd.DataFrame([
            {"基金代码": k, "基金简称": v} for k, v in KNOWN_QUANT_FUNDS.items()
        ])

    print(f"\n  入选基金 ({len(quant_funds)} 只):")
    display_cols = [c for c in ["基金代码", "基金简称", "近1年", "近3年"] if c in quant_funds.columns]
    print(quant_funds[display_cols].head(20).to_string(index=False))

    # ── Step 2: 计算绩效详情（夏普、最大回撤） ─────────────
    _hr("Step 2: 绩效详情计算")
    fund_codes = quant_funds["基金代码"].tolist()
    perf_records = []
    for i, (_, row) in enumerate(quant_funds.head(10).iterrows()):
        print(f"  [{i+1}/10] {row['基金简称']} ({row['基金代码']})...", end="", flush=True)
        perf = get_fund_performance_detail(row["基金代码"], row.get("基金简称", ""))
        perf_records.append(perf)
        status = "✓" if "error" not in perf else f"✗ {perf.get('error','')}"
        print(f" {status}")
        time.sleep(0.5)

    perf_df = pd.DataFrame(perf_records)
    if "error" in perf_df.columns:
        perf_df = perf_df[perf_df["error"].isna()].drop(columns=["error"], errors="ignore")

    if not perf_df.empty:
        keep = [c for c in ["fund_name", "ann_return", "max_drawdown",
                             "sharpe", "calmar", "return_1y", "start_date"] if c in perf_df.columns]
        print(f"\n  绩效汇总:")
        print(perf_df[keep].sort_values("sharpe", ascending=False).to_string(index=False))

    # ── Step 3: 批量拉取持仓 ──────────────────────────────
    _hr("Step 3: 基金持仓爬取")
    # 取绩效最好的前10只（或手工列表前10）
    top10_codes = fund_codes[:10]
    print(f"  拉取 {len(top10_codes)} 只基金的持仓...")
    holdings_df = fetch_all_funds_holdings(
        top10_codes, years=["2024", "2023"], delay=1.0
    )
    if holdings_df.empty:
        print("  持仓数据为空，跳过后续分析")
        return quant_funds, pd.DataFrame(), pd.DataFrame()

    print(f"  获取持仓记录: {len(holdings_df)} 条")

    # ── Step 4: 机构共同持仓聚合 ─────────────────────────
    _hr("Step 4: 机构共同持仓 Top30")
    agg = aggregate_holdings(holdings_df, top_n=30)
    if not agg.empty:
        print(agg[["stock_code", "stock_name", "fund_count",
                    "avg_weight", "total_weight"]].to_string(index=False))

    # ── Step 5: 持仓重叠度 ───────────────────────────────
    _hr("Step 5: 基金策略相似度矩阵（Jaccard）")
    overlap = holdings_overlap_matrix(holdings_df)
    if not overlap.empty:
        print("  （数值越高代表持仓越相似，可能策略相同）")
        print(overlap.to_string())

    # ── Step 6: 持仓变化信号 ────────────────────────────
    _hr("Step 6: 持仓变化信号（最新两季对比）")
    for code in top10_codes[:3]:
        changes = detect_holding_changes(holdings_df, code)
        if "error" in changes:
            continue
        fund_name = quant_funds[quant_funds["基金代码"] == code]["基金简称"].values
        name = fund_name[0] if len(fund_name) > 0 else code
        print(f"\n  【{name}】 {changes['quarter_from']} → {changes['quarter_to']}")
        if changes["new_buys"]:
            print(f"    新进:  {', '.join(list(changes['new_buys'])[:5])}")
        if changes["full_sells"]:
            print(f"    清仓:  {', '.join(list(changes['full_sells'])[:5])}")
        if changes["increased"]:
            inc_str = ", ".join([f"{k}(+{v:.1f}%)" for k, v in list(changes["increased"].items())[:3]])
            print(f"    加仓:  {inc_str}")
        if changes["decreased"]:
            dec_str = ", ".join([f"{k}({v:.1f}%)" for k, v in list(changes["decreased"].items())[:3]])
            print(f"    减仓:  {dec_str}")

    print("\n✓ 报告生成完毕\n")
    return quant_funds, holdings_df, agg


def run_single_fund_report(fund_code: str):
    """单只基金深度分析"""
    _hr(f"基金深度分析: {fund_code}")

    # 绩效
    print("\n[1] 绩效指标:")
    perf = get_fund_performance_detail(fund_code)
    for k, v in perf.items():
        if k not in ("fund_code", "error"):
            print(f"    {k:20s}: {v}")

    # 持仓
    print("\n[2] 历史持仓:")
    holdings = fetch_fund_holdings(fund_code, years=["2024", "2023", "2022"])
    if holdings.empty:
        print("    无持仓数据")
        return

    # 持仓稳定性
    print("\n[3] 持仓稳定性（多季度持续出现）:")
    consistency = holding_consistency(fund_code, holdings, top_n=15)
    if not consistency.empty:
        print(consistency.to_string(index=False))

    # 持仓变化
    print("\n[4] 最新季度变化:")
    changes = detect_holding_changes(holdings, fund_code)
    if "error" not in changes:
        print(f"    对比区间: {changes['quarter_from']} → {changes['quarter_to']}")
        print(f"    新进建仓: {changes['new_buys'][:5]}")
        print(f"    完全清仓: {changes['full_sells'][:5]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基金分析报告")
    parser.add_argument("--fund", default="", help="单只基金代码（如 000311）")
    parser.add_argument("--max", type=int, default=20, help="筛选基金数量")
    args = parser.parse_args()

    if args.fund:
        run_single_fund_report(args.fund)
    else:
        run_full_report(max_funds=args.max)

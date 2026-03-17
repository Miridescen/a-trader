"""
查看数据拉取进度
运行: python scripts/progress.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from data.storage.db import get_engine, init_db

def main():
    init_db()
    engine = get_engine()
    with engine.connect() as conn:
        # 股票列表
        total = conn.execute(text("SELECT COUNT(*) FROM stock_basic")).fetchone()[0]

        # 日线完成情况
        done = conn.execute(text(
            "SELECT COUNT(DISTINCT ts_code) FROM fetch_log WHERE task='stock_daily' AND status='ok'"
        )).fetchone()[0]
        errors = conn.execute(text(
            "SELECT COUNT(*) FROM fetch_log WHERE task='stock_daily' AND status='error'"
        )).fetchone()[0]
        rows = conn.execute(text("SELECT COUNT(*) FROM stock_daily")).fetchone()[0]

        # 最新日期
        latest = conn.execute(text(
            "SELECT MAX(last_date) FROM fetch_log WHERE task='stock_daily' AND status='ok'"
        )).fetchone()[0]

        # 指数数据
        idx_rows = conn.execute(text("SELECT COUNT(*) FROM index_daily")).fetchone()[0]
        idx_codes = conn.execute(text("SELECT COUNT(DISTINCT ts_code) FROM index_daily")).fetchone()[0]

        # 财务数据
        fin_rows = conn.execute(text("SELECT COUNT(*) FROM financial_indicator")).fetchone()[0]

    pct = done / total * 100 if total else 0
    bar_len = 30
    filled = int(bar_len * done / total) if total else 0
    bar = "█" * filled + "░" * (bar_len - filled)

    print(f"\n{'='*50}")
    print(f"  股票列表   : {total:,} 只")
    print(f"  指数日线   : {idx_rows:,} 行 ({idx_codes} 个指数)")
    print(f"\n  股票日线进度:")
    print(f"  [{bar}] {done}/{total} ({pct:.1f}%)")
    print(f"  已入库行数 : {rows:,}")
    print(f"  失败数     : {errors}")
    print(f"  最新日期   : {latest}")
    print(f"\n  财务指标   : {fin_rows:,} 行")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()

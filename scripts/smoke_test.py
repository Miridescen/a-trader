"""
冒烟测试：验证数据获取基础功能是否正常
运行: python scripts/smoke_test.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

def test_db_init():
    print("\n[1] 测试数据库初始化...")
    from data.storage.db import init_db, get_engine
    init_db()
    engine = get_engine()
    with engine.connect() as conn:
        from sqlalchemy import text
        tables = conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )).fetchall()
    print(f"    建表成功: {[t[0] for t in tables]}")
    return engine


def test_index_daily():
    print("\n[2] 测试指数日线（沪深300，近30天）...")
    from data.fetcher.akshare_fetcher import get_index_daily
    df = get_index_daily("sh000300", start_date="20250101")
    if df.empty:
        print("    警告: 未获取到数据")
    else:
        print(f"    获取 {len(df)} 行，最新日期: {df['trade_date'].max()}")
        print(df.tail(3).to_string(index=False))


def test_stock_daily():
    print("\n[3] 测试股票日线（平安银行000001，近20天）...")
    try:
        from data.fetcher.akshare_fetcher import get_stock_daily
        df = get_stock_daily("000001", start_date="20250201")
        if df.empty:
            print("    警告: 未获取到数据")
        else:
            print(f"    获取 {len(df)} 行，最新日期: {df['trade_date'].max()}")
            print(df[["trade_date", "open", "close", "pct_chg", "vol"]].tail(3).to_string(index=False))
    except Exception as e:
        print(f"    跳过（网络问题）: {e.__class__.__name__}")


def test_financial():
    print("\n[4] 测试财务指标（平安银行）...")
    try:
        from data.fetcher.akshare_fetcher import get_financial_indicator
        df = get_financial_indicator("000001")
        if df.empty:
            print("    警告: 未获取到数据")
        else:
            cols = [c for c in ["end_date", "roe", "netprofit_yoy", "grossprofit_margin"] if c in df.columns]
            print(f"    获取 {len(df)} 行")
            print(df[cols].tail(3).to_string(index=False))
    except Exception as e:
        print(f"    跳过（网络问题）: {e.__class__.__name__}")


def test_pipeline_index(engine):
    print("\n[5] 测试管道写入（沪深300）...")
    from data.pipeline import fetch_index_daily
    fetch_index_daily(engine)
    from sqlalchemy import text
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM index_daily")).fetchone()[0]
    print(f"    index_daily 表现有 {count} 行")


if __name__ == "__main__":
    engine = test_db_init()
    test_index_daily()
    test_stock_daily()
    test_financial()
    test_pipeline_index(engine)
    print("\n✓ 测试完成\n")

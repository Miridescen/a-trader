"""
回测数据加载器
从数据库中读取价格矩阵和基准数据
"""
import pandas as pd
import numpy as np
from sqlalchemy import text
from data.storage.db import get_engine
from data.config import INDEX_CODES
from typing import Optional


def load_price_matrix(
    ts_codes: Optional[list] = None,
    start_date: str = "20150101",
    end_date:   str = "20261231",
    min_days:   int = 500,
    price_col:  str = "close",
) -> pd.DataFrame:
    """
    从 stock_daily 加载价格矩阵
    index=日期, columns=ts_code, values=收盘价（后复权）
    只返回有效交易日数 >= min_days 的股票
    """
    engine = get_engine()

    # 先筛选有足够数据的股票
    with engine.connect() as conn:
        if ts_codes:
            placeholders = ",".join([f"'{c}'" for c in ts_codes])
            valid_codes = conn.execute(text(f"""
                SELECT ts_code FROM stock_daily
                WHERE ts_code IN ({placeholders})
                  AND trade_date BETWEEN :s AND :e
                GROUP BY ts_code HAVING COUNT(*) >= :m
            """), {"s": start_date[:4]+"-"+start_date[4:6]+"-"+start_date[6:],
                   "e": end_date[:4]+"-"+end_date[4:6]+"-"+end_date[6:],
                   "m": min_days}).fetchall()
        else:
            valid_codes = conn.execute(text("""
                SELECT ts_code FROM stock_daily
                WHERE trade_date BETWEEN :s AND :e
                GROUP BY ts_code HAVING COUNT(*) >= :m
            """), {"s": start_date[:4]+"-"+start_date[4:6]+"-"+start_date[6:],
                   "e": end_date[:4]+"-"+end_date[4:6]+"-"+end_date[6:],
                   "m": min_days}).fetchall()

        codes = [r[0] for r in valid_codes]
        if not codes:
            return pd.DataFrame()

        print(f"[数据] 加载 {len(codes)} 只股票价格数据...")

        # 批量读取（SQLite 有变量上限，分批）
        all_dfs = []
        batch = 200
        for i in range(0, len(codes), batch):
            chunk = codes[i:i+batch]
            ph = ",".join([f"'{c}'" for c in chunk])
            rows = conn.execute(text(f"""
                SELECT ts_code, trade_date, {price_col}
                FROM stock_daily
                WHERE ts_code IN ({ph})
                  AND trade_date BETWEEN :s AND :e
                ORDER BY trade_date
            """), {"s": start_date[:4]+"-"+start_date[4:6]+"-"+start_date[6:],
                   "e": end_date[:4]+"-"+end_date[4:6]+"-"+end_date[6:]}).fetchall()
            all_dfs.append(pd.DataFrame(rows, columns=["ts_code", "date", price_col]))

    df = pd.concat(all_dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot(index="date", columns="ts_code", values=price_col)
    print(f"[数据] 价格矩阵: {pivot.shape[0]} 天 × {pivot.shape[1]} 只")
    return pivot


def load_benchmark(
    ts_code: str = "000300.SH",
    start_date: str = "20150101",
    end_date:   str = "20261231",
) -> pd.Series:
    """加载基准指数收盘价"""
    engine = get_engine()
    s = start_date[:4]+"-"+start_date[4:6]+"-"+start_date[6:]
    e = end_date[:4]+"-"+end_date[4:6]+"-"+end_date[6:]
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT trade_date, close FROM index_daily
            WHERE ts_code = :code AND trade_date BETWEEN :s AND :e
            ORDER BY trade_date
        """), {"code": ts_code, "s": s, "e": e}).fetchall()

    if not rows:
        return pd.Series()
    df = pd.DataFrame(rows, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"]


def load_return_matrix(price_df: pd.DataFrame) -> pd.DataFrame:
    """从价格矩阵计算日收益率矩阵"""
    return price_df.pct_change()


def load_mv_matrix(
    start_date: str = "20150101",
    end_date:   str = "20261231",
) -> pd.DataFrame:
    """
    从 daily_basic 加载总市值矩阵（万元）
    index=日期, columns=ts_code, values=total_mv
    """
    engine = get_engine()
    s = start_date[:4]+"-"+start_date[4:6]+"-"+start_date[6:]
    e = end_date[:4]+"-"+end_date[4:6]+"-"+end_date[6:]
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT ts_code, trade_date, total_mv
            FROM daily_basic
            WHERE trade_date BETWEEN :s AND :e
              AND total_mv IS NOT NULL
            ORDER BY trade_date
        """), {"s": s, "e": e}).fetchall()

    if not rows:
        print("[数据] daily_basic 无市值数据，请先运行: python -m data.pipeline --task daily_basic")
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["ts_code", "date", "total_mv"])
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot(index="date", columns="ts_code", values="total_mv")
    print(f"[数据] 市值矩阵: {pivot.shape[0]} 天 × {pivot.shape[1]} 只")
    return pivot


def get_trading_dates(
    start_date: str = "20150101",
    end_date:   str = "20261231",
) -> list:
    """获取所有交易日列表"""
    engine = get_engine()
    s = start_date[:4]+"-"+start_date[4:6]+"-"+start_date[6:]
    e = end_date[:4]+"-"+end_date[4:6]+"-"+end_date[6:]
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT trade_date FROM index_daily
            WHERE ts_code='000300.SH' AND trade_date BETWEEN :s AND :e
            ORDER BY trade_date
        """), {"s": s, "e": e}).fetchall()
    return [pd.Timestamp(r[0]) for r in rows]

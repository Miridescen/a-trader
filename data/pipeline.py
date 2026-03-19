"""
数据管道：统一调度数据拉取并写入数据库
支持增量更新（断点续传）
用法:
    python -m data.pipeline --task all
    python -m data.pipeline --task index          # 只拉指数
    python -m data.pipeline --task stock_basic    # 只拉股票列表
    python -m data.pipeline --task stock_daily --code 000001.SZ
"""
import argparse
import logging
from datetime import datetime, date
import pandas as pd
from sqlalchemy import select, func, text
from data.storage.db import (
    init_db, get_engine,
    stock_basic, stock_daily, index_daily,
    daily_basic, financial_indicator,
    fund_basic, fund_nav, fund_portfolio, fetch_log
)
from data.config import INDEX_CODES, DEFAULT_START_DATE, TUSHARE_TOKEN
from data.fetcher.akshare_fetcher import (
    INDEX_CODE_MAP,
    get_index_daily as ak_index_daily,
    get_stock_list as ak_stock_list,
    get_stock_daily as ak_stock_daily,
    get_financial_indicator as ak_fin_indicator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _upsert(engine, table, df: pd.DataFrame, unique_cols: list[str]):
    """简单 upsert：先删后插（SQLite 兼容方式）"""
    if df.empty:
        return 0
    with engine.begin() as conn:
        for _, row in df.iterrows():
            row_dict = {k: v for k, v in row.items() if k in [c.name for c in table.columns]}
            # 删除已有记录
            condition = " AND ".join(f"{c} = :{c}" for c in unique_cols)
            conn.execute(text(f"DELETE FROM {table.name} WHERE {condition}"),
                         {c: row_dict[c] for c in unique_cols})
            conn.execute(table.insert(), row_dict)
    return len(df)


def _get_last_date(engine, task: str, ts_code: str = "") -> str:
    """从 fetch_log 获取上次同步的最新日期，用于增量更新"""
    with engine.connect() as conn:
        stmt = select(fetch_log.c.last_date).where(
            (fetch_log.c.task == task) &
            (fetch_log.c.ts_code == ts_code)
        )
        row = conn.execute(stmt).fetchone()
        if row and row[0]:
            # 在上次日期基础上 +1 天
            return (pd.to_datetime(str(row[0])) + pd.Timedelta(days=1)).strftime("%Y%m%d")
    return DEFAULT_START_DATE


def _save_log(engine, task: str, ts_code: str, last_date, rows: int, status="ok", message=""):
    with engine.begin() as conn:
        conn.execute(text(
            f"DELETE FROM fetch_log WHERE task=:t AND ts_code=:c"
        ), {"t": task, "c": ts_code})
        conn.execute(fetch_log.insert(), {
            "task": task,
            "ts_code": ts_code,
            "last_date": last_date,
            "status": status,
            "rows": rows,
            "message": message,
            "updated_at": datetime.now(),
        })


# ══════════════════════════════════════════════════════════
# Task 1: 指数日线
# ══════════════════════════════════════════════════════════
def fetch_index_daily(engine):
    log.info("=== 拉取指数日线行情 ===")
    for name, ts_code in INDEX_CODES.items():
        start = _get_last_date(engine, "index_daily", ts_code)
        ak_code = INDEX_CODE_MAP.get(ts_code, "")
        if not ak_code:
            continue
        log.info(f"  {name} ({ts_code}) 起始日期: {start}")
        try:
            df = ak_index_daily(ak_code, start_date=start)
            rows = _upsert(engine, index_daily, df,
                           ["ts_code", "trade_date"])
            last = df["trade_date"].max() if not df.empty else None
            _save_log(engine, "index_daily", ts_code, last, rows)
            log.info(f"  {name}: 写入 {rows} 行")
        except Exception as e:
            log.error(f"  {name} 失败: {e}")
            _save_log(engine, "index_daily", ts_code, None, 0, "error", str(e))


# ══════════════════════════════════════════════════════════
# Task 2: 股票基本信息
# ══════════════════════════════════════════════════════════
def fetch_stock_basic(engine):
    log.info("=== 拉取股票基本信息 ===")
    try:
        if TUSHARE_TOKEN:
            from data.fetcher.tushare_fetcher import get_stock_basic
            df = get_stock_basic()
        else:
            df = ak_stock_list()

        rows = _upsert(engine, stock_basic, df, ["ts_code"])
        _save_log(engine, "stock_basic", "", date.today(), rows)
        log.info(f"  股票基本信息: 写入 {rows} 行")
    except Exception as e:
        log.error(f"  股票基本信息失败: {e}")


# ══════════════════════════════════════════════════════════
# Task 3: 股票日线（按股票逐个拉，支持断点续传）
# ══════════════════════════════════════════════════════════
def fetch_stock_daily(engine, ts_code: str = "", limit: int = 0):
    """
    limit=0 表示拉全量；limit=N 只拉前N只（测试用）
    """
    log.info("=== 拉取股票日线行情 ===")

    # 获取股票列表
    with engine.connect() as conn:
        result = conn.execute(select(stock_basic.c.ts_code, stock_basic.c.symbol))
        stocks = result.fetchall()

    if ts_code:
        stocks = [s for s in stocks if s[0] == ts_code]
    if limit:
        stocks = stocks[:limit]

    import time as _time
    from datetime import timedelta

    # 今日和昨日
    today = date.today()
    yesterday = today - timedelta(days=1)

    log.info(f"  待处理股票数: {len(stocks)}")
    for i, (code, symbol) in enumerate(stocks):
        start_str = _get_last_date(engine, "stock_daily", code)
        start_d   = pd.to_datetime(start_str).date()

        # ── 跳过已是最新的股票（上次拉取日期 >= 昨日），避免无效 API 调用 ──
        if start_d >= yesterday:
            continue

        try:
            df = ak_stock_daily(symbol=symbol, start_date=start_str)
            if df.empty:
                _save_log(engine, "stock_daily", code, None, 0, "ok", "empty")
            else:
                rows = _upsert(engine, stock_daily, df, ["ts_code", "trade_date"])
                last = df["trade_date"].max()
                _save_log(engine, "stock_daily", code, last, rows)
            if (i + 1) % 200 == 0:
                log.info(f"  进度: {i + 1}/{len(stocks)} ({(i+1)/len(stocks)*100:.1f}%)")
        except Exception as e:
            log.warning(f"  {code} 失败: {e}")
            _save_log(engine, "stock_daily", code, None, 0, "error", str(e))
            _time.sleep(1.0)   # 出错时额外等待
        finally:
            _time.sleep(0.3)   # 防止 Sina 限速


# ══════════════════════════════════════════════════════════
# Task 4: 财务指标（AKShare 版，Tushare 更全）
# ══════════════════════════════════════════════════════════
def fetch_financial_indicator(engine, ts_code: str = "", limit: int = 0):
    log.info("=== 拉取财务指标 ===")

    with engine.connect() as conn:
        result = conn.execute(select(stock_basic.c.ts_code, stock_basic.c.symbol))
        stocks = result.fetchall()

    if ts_code:
        stocks = [s for s in stocks if s[0] == ts_code]
    if limit:
        stocks = stocks[:limit]

    import time as _time
    # 已完成的股票
    with engine.connect() as conn:
        done = set(r[0] for r in conn.execute(text(
            "SELECT ts_code FROM fetch_log WHERE task='financial_indicator' AND status='ok'"
        )).fetchall())
    log.info(f"  已完成: {len(done)}，待拉取: {len(stocks)-len(done)}")

    for i, (code, symbol) in enumerate(stocks):
        if code in done:
            continue
        try:
            df = ak_fin_indicator(symbol=symbol)
            if df.empty:
                _save_log(engine, "financial_indicator", code, None, 0, "ok", "empty")
            else:
                rows = _upsert(engine, financial_indicator, df, ["ts_code", "end_date"])
                last = df["end_date"].max() if "end_date" in df.columns else None
                _save_log(engine, "financial_indicator", code, last, rows)
            if (i + 1) % 100 == 0:
                log.info(f"  进度: {i + 1}/{len(stocks)} ({(i+1)/len(stocks)*100:.1f}%)")
        except Exception as e:
            log.warning(f"  {code} 财务数据失败: {e}")
            _save_log(engine, "financial_indicator", code, None, 0, "error", str(e))
            _time.sleep(1.0)
        finally:
            _time.sleep(0.3)


# ══════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="A-Trader 数据管道")
    parser.add_argument(
        "--task", default="all",
        choices=["all", "index", "stock_basic", "stock_daily",
                 "financial", "fund"],
        help="指定要运行的任务"
    )
    parser.add_argument("--code", default="", help="指定股票代码（用于 stock_daily）")
    parser.add_argument("--limit", type=int, default=0,
                        help="限制处理数量（测试用）")
    args = parser.parse_args()

    # 初始化数据库
    init_db()
    engine = get_engine()

    tasks = {
        "index":       lambda: fetch_index_daily(engine),
        "stock_basic": lambda: fetch_stock_basic(engine),
        "stock_daily": lambda: fetch_stock_daily(engine, args.code, args.limit),
        "financial":   lambda: fetch_financial_indicator(engine, args.code, args.limit),
    }

    if args.task == "all":
        tasks["index"]()
        tasks["stock_basic"]()
        # 全量日线数据量大，默认不自动跑，需显式指定
        log.info("提示: 股票日线/财务数据量较大，请单独运行:")
        log.info("  python -m data.pipeline --task stock_daily")
        log.info("  python -m data.pipeline --task financial")
    else:
        tasks[args.task]()

    log.info("=== 完成 ===")


if __name__ == "__main__":
    main()

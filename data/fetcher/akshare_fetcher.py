"""
AKShare 数据获取（免费，无需 token）
覆盖：股票列表、日线行情、财务数据、指数行情
"""
import time
import akshare as ak
import pandas as pd
from datetime import datetime, date
from typing import Optional


def _retry(fn, retries: int = 3, delay: float = 2.0):
    """简单重试装饰器（网络抖动）"""
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(delay * (i + 1))


def get_stock_list() -> pd.DataFrame:
    """
    获取A股全量股票列表
    返回字段: ts_code, symbol, name, area, industry, market, list_date
    """
    df = ak.stock_info_a_code_name()
    df = df.rename(columns={"code": "symbol", "name": "name"})
    df["ts_code"] = df["symbol"].apply(
        lambda x: f"{x}.SH" if x.startswith("6") else f"{x}.SZ"
    )
    df["list_date"] = None
    df["area"] = None
    df["industry"] = None
    df["market"] = None
    return df[["ts_code", "symbol", "name", "area", "industry", "market", "list_date"]]


def get_stock_daily(
    symbol: str,
    start_date: str = "20150101",
    end_date: Optional[str] = None,
    adjust: str = "hfq",   # 后复权
) -> pd.DataFrame:
    """
    获取单只股票日线行情（后复权）
    主接口: stock_zh_a_daily（新浪，稳定）
    备用接口: stock_zh_a_hist（东方财富）
    symbol: 6位纯代码，如 '000001'
    """
    ts_code = f"{symbol}.SH" if symbol.startswith("6") else f"{symbol}.SZ"
    prefix  = "sh" if symbol.startswith("6") else "sz"
    ak_symbol = f"{prefix}{symbol}"   # 新浪格式: sz000001

    # ── 主接口：新浪（全量历史，速度快） ──────────────────────
    try:
        df = _retry(lambda: ak.stock_zh_a_daily(
            symbol=ak_symbol,
            adjust=adjust,
        ), retries=3, delay=1.0)

        if df is not None and not df.empty:
            df = df.rename(columns={
                "date":   "trade_date",
                "open":   "open",
                "high":   "high",
                "low":    "low",
                "close":  "close",
                "volume": "vol",
                "amount": "amount",
            })
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
            start_d = pd.to_datetime(start_date).date()
            df = df[df["trade_date"] >= start_d]
            if end_date:
                df = df[df["trade_date"] <= pd.to_datetime(end_date).date()]
            df["ts_code"]    = ts_code
            df["pre_close"]  = df["close"].shift(1)
            df["pct_chg"]    = df["close"].pct_change() * 100
            df["change"]     = df["close"] - df["pre_close"]
            df["adj_factor"] = None
            keep = ["ts_code", "trade_date", "open", "high", "low", "close",
                    "pre_close", "change", "pct_chg", "vol", "amount", "adj_factor"]
            return df[[c for c in keep if c in df.columns]].dropna(subset=["trade_date"])
    except Exception:
        pass  # 主接口失败，走备用

    # ── 备用接口：东方财富 ────────────────────────────────────
    end = end_date or datetime.today().strftime("%Y%m%d")
    df = _retry(lambda: ak.stock_zh_a_hist(
        symbol=symbol, period="daily",
        start_date=start_date, end_date=end, adjust=adjust,
    ), retries=2, delay=3.0)

    if df is None or df.empty:
        return pd.DataFrame()

    col_map = {
        "日期": "trade_date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "vol",
        "成交额": "amount", "涨跌幅": "pct_chg", "涨跌额": "change",
    }
    df = df.rename(columns=col_map)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["ts_code"]    = ts_code
    df["pre_close"]  = df["close"].shift(1)
    df["adj_factor"] = None
    keep = ["ts_code", "trade_date", "open", "high", "low", "close",
            "pre_close", "change", "pct_chg", "vol", "amount", "adj_factor"]
    return df[keep].dropna(subset=["trade_date"])


def get_index_daily(
    index_code: str,
    start_date: str = "20150101",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    获取指数日线行情
    index_code: AKShare 格式，如 'sh000300'（沪深300）
    常用: sh000300 / sh000905 / sh000852 / sh000001 / sz399006
    """
    end = end_date or datetime.today().strftime("%Y%m%d")
    df = _retry(lambda: ak.stock_zh_index_daily(symbol=index_code))
    if df.empty:
        return df

    df = df.rename(columns={
        "date":   "trade_date",
        "open":   "open",
        "high":   "high",
        "low":    "low",
        "close":  "close",
        "volume": "vol",
    })
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df = df[
        (df["trade_date"] >= pd.to_datetime(start_date).date()) &
        (df["trade_date"] <= pd.to_datetime(end).date())
    ]

    # 统一 ts_code 格式（转为 000300.SH 形式）
    code_map = {
        "sh000300": "000300.SH",
        "sh000905": "000905.SH",
        "sh000852": "000852.SH",
        "sh000001": "000001.SH",
        "sz399006": "399006.SZ",
    }
    df["ts_code"] = code_map.get(index_code, index_code)
    df["pre_close"] = df["close"].shift(1)
    df["pct_chg"] = df["close"].pct_change() * 100
    df["amount"] = df.get("amount", None)

    keep = ["ts_code", "trade_date", "open", "high", "low", "close",
            "pre_close", "pct_chg", "vol", "amount"]
    return df[[c for c in keep if c in df.columns]].dropna(subset=["trade_date"])


def get_financial_indicator(symbol: str) -> pd.DataFrame:
    """
    获取个股关键财务指标（ROE、净利润增速、毛利率等）
    symbol: 6位纯代码
    """
    try:
        df = ak.stock_financial_abstract_ths(symbol=symbol, indicator="按年度")
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    col_map = {
        "报告期":           "end_date",
        "净资产收益率":      "roe",
        "净资产收益率-加权": "roe",
        "每股收益":          "eps",
        "基本每股收益":      "eps",
        "净利润增长率":      "netprofit_yoy",
        "净利润同比增长率":  "netprofit_yoy",
        "营业总收入增长率":  "tr_yoy",
        "营业总收入同比增长率": "tr_yoy",
        "毛利率":            "grossprofit_margin",
        "销售净利率":        "grossprofit_margin",
        "资产负债率":        "debt_to_assets",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["ts_code"] = f"{symbol}.SH" if symbol.startswith("6") else f"{symbol}.SZ"

    # end_date 清洗：整数年份 → YYYY-12-31，字符串日期直接转
    def _parse_date(v):
        try:
            v = str(v).strip()
            if len(v) == 4 and v.isdigit():
                return pd.Timestamp(f"{v}-12-31")
            return pd.Timestamp(v)
        except Exception:
            return pd.NaT

    df["end_date"] = df["end_date"].apply(_parse_date)
    df = df.dropna(subset=["end_date"])
    df = df[df["end_date"] >= pd.Timestamp("2010-01-01")]
    df["end_date"] = df["end_date"].dt.date

    # 数值清洗：去掉 %
    for col in ["roe", "netprofit_yoy", "tr_yoy", "grossprofit_margin", "debt_to_assets"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace("%", "").str.strip(),
                errors="coerce"
            )
    return df


def get_fund_list(market: str = "E") -> pd.DataFrame:
    """
    获取公募基金列表
    market: E=场内ETF  O=场外基金
    """
    try:
        df = ak.fund_etf_category_sina(symbol="封闭式基金")
        return df
    except Exception:
        return pd.DataFrame()


def get_fund_nav_history(fund_code: str) -> pd.DataFrame:
    """
    获取公募基金净值历史（场外基金）
    fund_code: 6位基金代码，如 '110022'
    """
    try:
        df = ak.fund_open_fund_info_em(fund=fund_code, indicator="单位净值走势")
        df = df.rename(columns={
            "净值日期": "nav_date",
            "单位净值": "unit_nav",
            "累计净值": "accum_nav",
        })
        df["nav_date"] = pd.to_datetime(df["nav_date"]).dt.date
        df["ts_code"] = fund_code
        return df[["ts_code", "nav_date", "unit_nav", "accum_nav"]]
    except Exception:
        return pd.DataFrame()


# ── 指数代码对照 ────────────────────────────────────────────
INDEX_CODE_MAP = {
    "000300.SH": "sh000300",
    "000905.SH": "sh000905",
    "000852.SH": "sh000852",
    "000001.SH": "sh000001",
    "399006.SZ": "sz399006",
}

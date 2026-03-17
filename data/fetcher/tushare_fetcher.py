"""
Tushare Pro 数据获取（需要 token，数据更全面）
覆盖：股票基本信息、日线行情、复权因子、市值、财务、基金持仓
申请 token: https://tushare.pro/register
"""
import tushare as ts
import pandas as pd
from typing import Optional
from data.config import TUSHARE_TOKEN

_pro = None


def get_pro():
    """懒加载 Tushare Pro 客户端"""
    global _pro
    if _pro is None:
        if not TUSHARE_TOKEN:
            raise ValueError(
                "未配置 TUSHARE_TOKEN，请在 .env 或环境变量中设置。\n"
                "申请地址: https://tushare.pro/register"
            )
        ts.set_token(TUSHARE_TOKEN)
        _pro = ts.pro_api()
    return _pro


def get_stock_basic() -> pd.DataFrame:
    """
    获取A股全量股票基本信息
    返回字段: ts_code, symbol, name, area, industry, market, list_date
    """
    pro = get_pro()
    df = pro.stock_basic(
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,area,industry,market,list_date"
    )
    df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce").dt.date
    return df


def get_stock_daily(
    ts_code: str,
    start_date: str = "20150101",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    获取单只股票日线行情（不含复权）
    ts_code: 如 '000001.SZ'
    """
    pro = get_pro()
    df = pro.daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date or "",
        fields="ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df.sort_values("trade_date")


def get_adj_factor(
    ts_code: str,
    start_date: str = "20150101",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """获取后复权因子"""
    pro = get_pro()
    df = pro.adj_factor(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date or "",
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df.sort_values("trade_date")


def get_daily_basic(
    ts_code: str = "",
    trade_date: str = "",
    start_date: str = "20150101",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    获取每日指标（市值、PE、PB等）
    可按股票代码 或 交易日期 查询
    """
    pro = get_pro()
    fields = (
        "ts_code,trade_date,close,turnover_rate,turnover_rate_f,"
        "volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,total_mv,circ_mv"
    )
    df = pro.daily_basic(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date or "",
        fields=fields
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df.sort_values("trade_date")


def get_financial_indicator(ts_code: str, period: str = "") -> pd.DataFrame:
    """
    获取财务指标（ROE、净利润增速等）
    period: 如 '20231231'（为空则拉全部）
    """
    pro = get_pro()
    fields = (
        "ts_code,ann_date,end_date,eps,bps,roe,roe_yoy,roa,"
        "netprofit_yoy,tr_yoy,or_yoy,grossprofit_margin,"
        "profit_to_op,debt_to_assets,current_ratio,ocf_to_profit"
    )
    df = pro.fina_indicator(
        ts_code=ts_code,
        period=period,
        fields=fields
    )
    for col in ["ann_date", "end_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df.sort_values("end_date")


def get_index_daily(
    ts_code: str,
    start_date: str = "20150101",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    获取指数日线行情
    ts_code: 如 '000300.SH'
    """
    pro = get_pro()
    df = pro.index_daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date or "",
        fields="ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount"
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df.sort_values("trade_date")


def get_fund_basic(market: str = "E") -> pd.DataFrame:
    """
    获取基金基本信息
    market: E=场内  O=场外
    """
    pro = get_pro()
    df = pro.fund_basic(
        market=market,
        fields="ts_code,name,management,fund_type,found_date,due_date,issue_date,market,status"
    )
    for col in ["found_date", "due_date", "issue_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df


def get_fund_nav(
    ts_code: str,
    start_date: str = "20150101",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """获取基金净值历史"""
    pro = get_pro()
    df = pro.fund_nav(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date or "",
        fields="ts_code,nav_date,unit_nav,accum_nav,net_asset"
    )
    df["nav_date"] = pd.to_datetime(df["nav_date"]).dt.date
    return df.sort_values("nav_date")


def get_fund_portfolio(ts_code: str, period: str = "") -> pd.DataFrame:
    """
    获取基金持仓（季报）
    ts_code: 基金代码
    period: 如 '20231231'（为空则拉全部）
    """
    pro = get_pro()
    df = pro.fund_portfolio(
        ts_code=ts_code,
        period=period,
        fields="ts_code,ann_date,end_date,symbol,mkv,amount,stk_mkv_ratio,stk_float_ratio"
    )
    for col in ["ann_date", "end_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df.sort_values("end_date")

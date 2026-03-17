"""
数据库初始化与表结构定义（SQLAlchemy Core）
支持 SQLite（默认）和 PostgreSQL（DB_URL 切换）
"""
from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    String, Float, Integer, BigInteger, Date, DateTime, Text,
    UniqueConstraint, Index, text
)
from sqlalchemy.exc import OperationalError
from datetime import datetime
from data.config import DB_URL

engine   = create_engine(DB_URL, echo=False)
metadata = MetaData()

# ── 1. 股票日线行情 ────────────────────────────────────────
stock_daily = Table("stock_daily", metadata,
    Column("ts_code",       String(12), nullable=False),   # 000001.SZ
    Column("trade_date",    Date,       nullable=False),
    Column("open",          Float),
    Column("high",          Float),
    Column("low",           Float),
    Column("close",         Float),
    Column("pre_close",     Float),
    Column("change",        Float),
    Column("pct_chg",       Float),
    Column("vol",           Float),    # 手
    Column("amount",        Float),    # 千元
    Column("adj_factor",    Float),    # 复权因子
    UniqueConstraint("ts_code", "trade_date", name="uq_stock_daily"),
    Index("ix_stock_daily_date", "trade_date"),
    Index("ix_stock_daily_code", "ts_code"),
)

# ── 2. 指数日线行情 ────────────────────────────────────────
index_daily = Table("index_daily", metadata,
    Column("ts_code",    String(12), nullable=False),
    Column("trade_date", Date,       nullable=False),
    Column("open",       Float),
    Column("high",       Float),
    Column("low",        Float),
    Column("close",      Float),
    Column("pre_close",  Float),
    Column("pct_chg",    Float),
    Column("vol",        Float),
    Column("amount",     Float),
    UniqueConstraint("ts_code", "trade_date", name="uq_index_daily"),
    Index("ix_index_daily_date", "trade_date"),
)

# ── 3. 股票基本信息 ────────────────────────────────────────
stock_basic = Table("stock_basic", metadata,
    Column("ts_code",    String(12), primary_key=True),
    Column("symbol",     String(8)),
    Column("name",       String(32)),
    Column("area",       String(16)),
    Column("industry",   String(32)),
    Column("market",     String(8)),   # 主板/创业板/科创板
    Column("list_date",  Date),
    Column("delist_date",Date),
    Column("is_hs",      String(4)),   # 沪深港通标的
    Column("updated_at", DateTime, default=datetime.now),
)

# ── 4. 财务数据（核心指标） ────────────────────────────────
financial_indicator = Table("financial_indicator", metadata,
    Column("ts_code",         String(12), nullable=False),
    Column("ann_date",        Date),        # 公告日期
    Column("end_date",        Date,         nullable=False),  # 报告期
    Column("eps",             Float),       # 每股收益
    Column("bps",             Float),       # 每股净资产
    Column("roe",             Float),       # ROE
    Column("roe_yoy",         Float),       # ROE同比增长
    Column("roa",             Float),       # ROA
    Column("netprofit_yoy",   Float),       # 净利润同比增长%
    Column("tr_yoy",          Float),       # 营收同比增长%
    Column("or_yoy",          Float),       # 营业利润同比增长%
    Column("grossprofit_margin", Float),    # 毛利率
    Column("profit_to_op",    Float),       # 净利润/营业利润（利润质量）
    Column("debt_to_assets",  Float),       # 资产负债率
    Column("current_ratio",   Float),       # 流动比率
    Column("ocf_to_profit",   Float),       # 经营现金流/净利润
    UniqueConstraint("ts_code", "end_date", name="uq_fin_indicator"),
    Index("ix_fin_ts_code", "ts_code"),
)

# ── 5. 市值数据（每日） ─────────────────────────────────────
daily_basic = Table("daily_basic", metadata,
    Column("ts_code",        String(12), nullable=False),
    Column("trade_date",     Date,       nullable=False),
    Column("close",          Float),
    Column("turnover_rate",  Float),     # 换手率
    Column("turnover_rate_f",Float),     # 自由流通换手率
    Column("volume_ratio",   Float),     # 量比
    Column("pe",             Float),     # 市盈率
    Column("pe_ttm",         Float),     # 市盈率TTM
    Column("pb",             Float),     # 市净率
    Column("ps",             Float),     # 市销率
    Column("ps_ttm",         Float),
    Column("dv_ratio",       Float),     # 股息率
    Column("total_mv",       Float),     # 总市值（万元）
    Column("circ_mv",        Float),     # 流通市值
    UniqueConstraint("ts_code", "trade_date", name="uq_daily_basic"),
    Index("ix_daily_basic_date", "trade_date"),
)

# ── 6. 基金基本信息 ────────────────────────────────────────
fund_basic = Table("fund_basic", metadata,
    Column("ts_code",    String(16), primary_key=True),
    Column("name",       String(64)),
    Column("management", String(64)),  # 管理人
    Column("fund_type",  String(32)),
    Column("found_date", Date),
    Column("due_date",   Date),
    Column("issue_date", Date),
    Column("market",     String(8)),   # E=场内 O=场外
    Column("status",     String(4)),
    Column("updated_at", DateTime, default=datetime.now),
)

# ── 7. 基金净值 ────────────────────────────────────────────
fund_nav = Table("fund_nav", metadata,
    Column("ts_code",      String(16), nullable=False),
    Column("nav_date",     Date,       nullable=False),
    Column("unit_nav",     Float),     # 单位净值
    Column("accum_nav",    Float),     # 累计净值
    Column("net_asset",    Float),     # 资产净值（亿元）
    UniqueConstraint("ts_code", "nav_date", name="uq_fund_nav"),
)

# ── 8. 基金持仓 ────────────────────────────────────────────
fund_portfolio = Table("fund_portfolio", metadata,
    Column("ts_code",      String(16), nullable=False),  # 基金代码
    Column("ann_date",     Date),
    Column("end_date",     Date,       nullable=False),  # 报告期
    Column("symbol",       String(12)),                  # 持仓股票
    Column("mkv",          Float),                       # 持仓市值（万元）
    Column("amount",       Float),                       # 持仓量（万股）
    Column("stk_mkv_ratio",Float),                      # 占基金净值比%
    Column("stk_float_ratio", Float),                   # 占流通股比%
    UniqueConstraint("ts_code", "end_date", "symbol", name="uq_fund_portfolio"),
)

# ── 9. 数据拉取任务记录（增量更新用） ─────────────────────────
fetch_log = Table("fetch_log", metadata,
    Column("id",        Integer, autoincrement=True, primary_key=True),
    Column("task",      String(64), nullable=False),   # 任务名称
    Column("ts_code",   String(16)),
    Column("last_date", Date),                          # 最新数据日期
    Column("status",    String(16), default="ok"),
    Column("rows",      BigInteger, default=0),
    Column("message",   Text),
    Column("updated_at",DateTime, default=datetime.now),
    Index("ix_fetch_log_task", "task", "ts_code"),
)


def init_db():
    """建表（已存在则跳过）"""
    metadata.create_all(engine)
    print(f"[DB] 初始化完成: {DB_URL}")


def get_engine():
    return engine

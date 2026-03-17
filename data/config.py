"""
数据层配置：API token、数据库路径、通用参数
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# ── Tushare ──────────────────────────────────────────────
# 申请地址: https://tushare.pro/register
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")

# ── 数据库 ────────────────────────────────────────────────
# 默认使用 SQLite（本地文件，无需安装），生产可换 PostgreSQL
DB_URL = os.getenv(
    "DB_URL",
    f"sqlite:///{BASE_DIR}/data/market.db"
)

# ── 通用参数 ──────────────────────────────────────────────
DEFAULT_START_DATE = "20150101"   # 回测起点
DEFAULT_END_DATE   = None         # None = 今日

# A股主要指数代码
INDEX_CODES = {
    "沪深300":  "000300.SH",
    "中证500":  "000905.SH",
    "中证1000": "000852.SH",
    "上证综指": "000001.SH",
    "创业板指": "399006.SZ",
}

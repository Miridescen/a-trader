"""
回测引擎核心
职责：模拟持仓变化、处理交易成本、记录每日净值
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import date


@dataclass
class TradeConfig:
    """交易成本配置（A股）"""
    commission:    float = 0.0003   # 佣金（双向）
    stamp_duty:    float = 0.001    # 印花税（仅卖出）
    slippage:      float = 0.001    # 滑点（双向）
    min_trade_lot: int   = 100      # 最小交易单位（手）


@dataclass
class Position:
    ts_code:  str
    shares:   float = 0.0
    cost:     float = 0.0    # 持仓均价


@dataclass
class DailyRecord:
    date:        date
    portfolio_value: float
    cash:        float
    positions:   Dict[str, float]   # ts_code -> market_value
    turnover:    float = 0.0        # 当日换手金额


class Portfolio:
    """
    投资组合状态机
    管理现金、持仓、每日净值记录
    """
    def __init__(self, initial_capital: float, config: TradeConfig):
        self.cash    = initial_capital
        self.capital = initial_capital
        self.config  = config
        self.positions: Dict[str, Position] = {}
        self.records:   List[DailyRecord]   = []
        self.trade_log: List[dict]          = []

    # ── 估值 ──────────────────────────────────────────────
    def market_value(self, prices: Dict[str, float]) -> float:
        pos_value = sum(
            pos.shares * prices.get(ts, pos.cost)
            for ts, pos in self.positions.items()
        )
        return self.cash + pos_value

    def position_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        total = self.market_value(prices)
        if total <= 0:
            return {}
        return {
            ts: pos.shares * prices.get(ts, pos.cost) / total
            for ts, pos in self.positions.items()
        }

    # ── 交易 ──────────────────────────────────────────────
    def buy(self, ts_code: str, price: float, amount: float, trade_date: date):
        """以 amount 元买入，扣除佣金+滑点"""
        if price <= 0 or amount <= 0:
            return 0.0
        eff_price = price * (1 + self.config.commission + self.config.slippage)
        shares    = amount / eff_price
        cost      = shares * eff_price

        if cost > self.cash:
            shares = self.cash / eff_price * 0.999
            cost   = self.cash * 0.999

        if ts_code in self.positions:
            old = self.positions[ts_code]
            total_shares = old.shares + shares
            avg_cost = (old.shares * old.cost + shares * price) / total_shares
            self.positions[ts_code] = Position(ts_code, total_shares, avg_cost)
        else:
            self.positions[ts_code] = Position(ts_code, shares, price)

        self.cash -= cost
        self.trade_log.append({
            "date": trade_date, "ts_code": ts_code,
            "action": "buy", "price": price, "shares": shares, "amount": cost
        })
        return cost

    def sell(self, ts_code: str, price: float, shares: float, trade_date: date):
        """卖出指定数量，扣除印花税+佣金+滑点"""
        if ts_code not in self.positions or price <= 0:
            return 0.0
        pos = self.positions[ts_code]
        shares = min(shares, pos.shares)
        if shares <= 0:
            return 0.0

        eff_price = price * (1 - self.config.stamp_duty
                             - self.config.commission - self.config.slippage)
        proceeds  = shares * eff_price

        pos.shares -= shares
        if pos.shares < 0.01:
            del self.positions[ts_code]

        self.cash += proceeds
        self.trade_log.append({
            "date": trade_date, "ts_code": ts_code,
            "action": "sell", "price": price, "shares": shares, "amount": proceeds
        })
        return proceeds

    def sell_all(self, ts_code: str, price: float, trade_date: date):
        if ts_code in self.positions:
            return self.sell(ts_code, price, self.positions[ts_code].shares, trade_date)
        return 0.0

    # ── 再平衡 ────────────────────────────────────────────
    def rebalance(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        trade_date: date,
    ) -> float:
        """
        调仓到目标权重
        target_weights: {ts_code: weight}，weight 之和应 ≈ 1
        返回：本次调仓的换手金额
        """
        total_value = self.market_value(prices)
        current_w   = self.position_weights(prices)
        turnover    = 0.0

        # 先卖出需要减仓的股票（释放现金）
        for ts_code, target_w in target_weights.items():
            curr_w = current_w.get(ts_code, 0.0)
            if curr_w > target_w + 0.001:
                p = prices.get(ts_code, 0)
                if p <= 0:
                    continue
                target_val  = total_value * target_w
                current_val = total_value * curr_w
                sell_amount = current_val - target_val
                sell_shares = sell_amount / p
                proceeds = self.sell(ts_code, p, sell_shares, trade_date)
                turnover += proceeds

        # 清仓不在目标中的股票
        for ts_code in list(self.positions.keys()):
            if ts_code not in target_weights:
                p = prices.get(ts_code, 0)
                if p > 0:
                    proceeds = self.sell_all(ts_code, p, trade_date)
                    turnover += proceeds

        # 再买入需要加仓的股票
        total_value = self.market_value(prices)   # 卖后重新估值
        for ts_code, target_w in target_weights.items():
            curr_w = current_w.get(ts_code, 0.0)
            if target_w > curr_w + 0.001:
                p = prices.get(ts_code, 0)
                if p <= 0:
                    continue
                target_val  = total_value * target_w
                current_val = self.positions.get(ts_code, Position(ts_code)).shares * p
                buy_amount  = target_val - current_val
                if buy_amount > 100:
                    cost = self.buy(ts_code, p, buy_amount, trade_date)
                    turnover += cost

        return turnover

    # ── 快照 ──────────────────────────────────────────────
    def snapshot(self, trade_date: date, prices: Dict[str, float], turnover: float = 0.0):
        total = self.market_value(prices)
        pos_values = {
            ts: pos.shares * prices.get(ts, pos.cost)
            for ts, pos in self.positions.items()
        }
        self.records.append(DailyRecord(
            date=trade_date,
            portfolio_value=total,
            cash=self.cash,
            positions=pos_values,
            turnover=turnover,
        ))

    def nav_series(self) -> pd.Series:
        """返回净值序列（初始=1）"""
        if not self.records:
            return pd.Series()
        dates  = [r.date for r in self.records]
        values = [r.portfolio_value for r in self.records]
        s = pd.Series(values, index=pd.DatetimeIndex(dates), name="portfolio")
        return s / s.iloc[0]

    def trade_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.trade_log)


class BacktestEngine:
    """
    回测驱动器
    负责：
      1. 加载价格数据
      2. 生成调仓日历
      3. 逐日驱动组合更新
      4. 在调仓日调用策略生成目标权重
    """
    def __init__(
        self,
        price_df: pd.DataFrame,      # index=date, columns=ts_code, values=close
        benchmark_df: pd.Series,     # index=date, values=close
        initial_capital: float = 1_000_000,
        config: TradeConfig = None,
        rebalance_freq: str = "Q",   # 'M'=月度  'Q'=季度
    ):
        self.price_df   = price_df.sort_index()
        self.benchmark  = benchmark_df.sort_index()
        self.capital    = initial_capital
        self.config     = config or TradeConfig()
        self.freq       = rebalance_freq
        self._rebal_dates = self._build_rebal_dates()

    def _build_rebal_dates(self) -> List[date]:
        """生成调仓日历（每月/季最后一个交易日）"""
        dates = pd.DatetimeIndex(self.price_df.index)
        if self.freq == "M":
            grouped = dates.to_period("M")
        else:
            grouped = dates.to_period("Q")

        rebal = []
        df_tmp = pd.DataFrame({"dt": dates, "period": grouped})
        for period, grp in df_tmp.groupby("period"):
            rebal.append(grp["dt"].iloc[-1].date())
        return sorted(rebal)

    def run(
        self,
        strategy_fn: Callable[[date, pd.DataFrame, dict], Dict[str, float]],
        start_date: Optional[date] = None,
        end_date:   Optional[date] = None,
    ) -> Portfolio:
        """
        运行回测
        strategy_fn(current_date, price_history, context) -> {ts_code: weight}
        """
        portfolio = Portfolio(self.capital, self.config)
        all_dates = [d.date() if hasattr(d, 'date') else d
                     for d in self.price_df.index]

        start = start_date or all_dates[0]
        end   = end_date   or all_dates[-1]
        dates = [d for d in all_dates if start <= d <= end]
        rebal_set = set(d for d in self._rebal_dates if start <= d <= end)

        context = {}
        turnover_today = 0.0

        for i, d in enumerate(dates):
            ts_d = pd.Timestamp(d)
            prices = self.price_df.loc[ts_d].dropna().to_dict()
            if not prices:
                continue

            # 调仓日
            if d in rebal_set:
                history = self.price_df.loc[:ts_d]
                target_weights = strategy_fn(d, history, context)
                # 过滤掉当日无价格的股票
                target_weights = {
                    ts: w for ts, w in target_weights.items()
                    if ts in prices and prices[ts] > 0
                }
                # 归一化权重
                total_w = sum(target_weights.values())
                if total_w > 0:
                    target_weights = {ts: w / total_w for ts, w in target_weights.items()}
                turnover_today = portfolio.rebalance(target_weights, prices, d)
            else:
                turnover_today = 0.0

            portfolio.snapshot(d, prices, turnover_today)

        return portfolio

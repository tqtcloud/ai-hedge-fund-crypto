"""
期货交易Mock数据生成器

生成高质量的模拟期货交易数据，包括：
- 市场数据：K线、深度、交易数据、价格统计
- 账户数据：余额、持仓、订单更新
- 实时数据流模拟和历史数据回放
- 支持并行开发和测试环境

提供逼真的市场行为模拟，包括价格波动、成交量变化、深度订单簿等。
"""

import asyncio
import random
import time
import math
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json

from .data_formats import (
    KlineData, DepthUpdate, MarkPriceUpdate, AggTradeData, TickerData,
    AccountUpdate, OrderTradeUpdate, Position, Balance, OrderUpdate,
    EventType, PositionSide, OrderType, OrderStatus, TimeInForce
)


@dataclass
class MockMarketConfig:
    """Mock市场配置"""
    # 基础价格参数
    base_price: Decimal = Decimal("50000")          # 基础价格
    price_volatility: float = 0.02                 # 价格波动率 (2%)
    trend_strength: float = 0.5                    # 趋势强度 (0-1)
    trend_change_prob: float = 0.01               # 趋势变化概率
    
    # 成交量参数
    base_volume: Decimal = Decimal("100")           # 基础成交量
    volume_volatility: float = 0.3                 # 成交量波动率 (30%)
    
    # 深度数据参数
    depth_levels: int = 20                         # 深度档位数量
    spread_percentage: float = 0.001               # 买卖价差百分比 (0.1%)
    depth_refresh_interval: float = 0.5            # 深度刷新间隔(秒)
    
    # K线参数
    kline_intervals: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h"])
    
    # 交易参数
    trade_frequency: float = 0.2                   # 交易频率(秒)
    large_trade_prob: float = 0.05                 # 大额交易概率
    large_trade_multiplier: float = 10.0           # 大额交易倍数


@dataclass
class MockAccountConfig:
    """Mock账户配置"""
    # 初始余额
    initial_usdt_balance: Decimal = Decimal("10000")    # 初始USDT余额
    initial_btc_balance: Decimal = Decimal("0.1")       # 初始BTC余额
    
    # 持仓配置
    initial_positions: List[Dict[str, Any]] = field(default_factory=list)
    
    # 订单配置  
    max_active_orders: int = 5                          # 最大活跃订单数
    order_fill_prob: float = 0.1                       # 订单成交概率
    partial_fill_prob: float = 0.3                     # 部分成交概率


class MockPriceEngine:
    """模拟价格引擎"""
    
    def __init__(self, config: MockMarketConfig):
        self.config = config
        self.current_price = config.base_price
        self.trend_direction = 1  # 1为上涨，-1为下跌
        self.last_update_time = time.time()
        
        # 价格历史记录（用于计算技术指标）
        self.price_history: List[Tuple[float, Decimal]] = []
        self.max_history_size = 1000
    
    def generate_next_price(self) -> Decimal:
        """生成下一个价格"""
        now = time.time()
        time_delta = now - self.last_update_time
        
        # 检查是否改变趋势
        if random.random() < self.config.trend_change_prob:
            self.trend_direction *= -1
        
        # 计算价格变化
        # 基础随机波动
        random_change = random.gauss(0, self.config.price_volatility)
        
        # 趋势影响
        trend_change = self.trend_direction * self.config.trend_strength * 0.0001
        
        # 总变化率
        total_change = random_change + trend_change
        
        # 应用价格变化
        new_price = self.current_price * (1 + Decimal(str(total_change)))
        
        # 确保价格为正数且保持合理范围
        if new_price <= 0:
            new_price = self.current_price * Decimal("0.9999")
        
        self.current_price = new_price.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        self.last_update_time = now
        
        # 记录价格历史
        self.price_history.append((now, self.current_price))
        if len(self.price_history) > self.max_history_size:
            self.price_history.pop(0)
        
        return self.current_price
    
    def get_price_at_time(self, target_time: float) -> Decimal:
        """获取指定时间的价格（用于历史数据）"""
        # 简化实现：基于当前价格和时间差生成
        time_diff = target_time - self.last_update_time
        if time_diff == 0:
            return self.current_price
        
        # 基于时间差生成价格变化
        change_rate = random.gauss(0, self.config.price_volatility) * abs(time_diff) / 3600
        return self.current_price * (1 + Decimal(str(change_rate)))


class MockVolumeEngine:
    """模拟成交量引擎"""
    
    def __init__(self, config: MockMarketConfig):
        self.config = config
        self.base_volume = config.base_volume
    
    def generate_volume(self) -> Decimal:
        """生成成交量"""
        # 使用对数正态分布模拟真实的成交量分布
        log_volume = math.log(float(self.base_volume)) + random.gauss(0, self.config.volume_volatility)
        volume = Decimal(str(math.exp(log_volume)))
        return volume.quantize(Decimal("0.001"), rounding=ROUND_DOWN)


class MockDepthEngine:
    """模拟深度数据引擎"""
    
    def __init__(self, config: MockMarketConfig, price_engine: MockPriceEngine):
        self.config = config
        self.price_engine = price_engine
    
    def generate_depth_data(self) -> Tuple[List[List[str]], List[List[str]]]:
        """生成深度数据 (bids, asks)"""
        current_price = self.price_engine.current_price
        spread = current_price * Decimal(str(self.config.spread_percentage))
        
        # 生成买单
        bids = []
        for i in range(self.config.depth_levels):
            price = current_price - spread * (i + 1)
            quantity = self._generate_depth_quantity(i)
            bids.append([str(price), str(quantity)])
        
        # 生成卖单
        asks = []
        for i in range(self.config.depth_levels):
            price = current_price + spread * (i + 1)
            quantity = self._generate_depth_quantity(i)
            asks.append([str(price), str(quantity)])
        
        return bids, asks
    
    def _generate_depth_quantity(self, level: int) -> Decimal:
        """生成深度档位的数量"""
        # 越接近价格中心，数量越多
        base_qty = self.config.base_volume / Decimal(str(level + 1))
        random_factor = Decimal(str(random.uniform(0.5, 2.0)))
        return (base_qty * random_factor).quantize(Decimal("0.001"), rounding=ROUND_DOWN)


class FuturesMockDataGenerator:
    """期货Mock数据生成器主类"""
    
    def __init__(
        self,
        symbols: List[str] = None,
        market_config: MockMarketConfig = None,
        account_config: MockAccountConfig = None
    ):
        """
        初始化Mock数据生成器
        
        Args:
            symbols: 交易对列表
            market_config: 市场配置
            account_config: 账户配置
        """
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        self.market_config = market_config or MockMarketConfig()
        self.account_config = account_config or MockAccountConfig()
        
        # 价格引擎（每个交易对一个）
        self.price_engines: Dict[str, MockPriceEngine] = {}
        self.volume_engines: Dict[str, MockVolumeEngine] = {}
        self.depth_engines: Dict[str, MockDepthEngine] = {}
        
        # 初始化引擎
        for symbol in self.symbols:
            # 为不同交易对设置不同的基础价格
            config = MockMarketConfig(
                base_price=self._get_base_price_for_symbol(symbol),
                price_volatility=self.market_config.price_volatility,
                trend_strength=self.market_config.trend_strength,
                trend_change_prob=self.market_config.trend_change_prob,
                base_volume=self.market_config.base_volume,
                volume_volatility=self.market_config.volume_volatility,
                depth_levels=self.market_config.depth_levels,
                spread_percentage=self.market_config.spread_percentage
            )
            
            price_engine = MockPriceEngine(config)
            self.price_engines[symbol] = price_engine
            self.volume_engines[symbol] = MockVolumeEngine(config)
            self.depth_engines[symbol] = MockDepthEngine(config, price_engine)
        
        # K线数据缓存
        self.kline_data: Dict[str, Dict[str, Dict]] = {}
        
        # 账户数据
        self.account_balances: Dict[str, Decimal] = {
            "USDT": self.account_config.initial_usdt_balance,
            "BTC": self.account_config.initial_btc_balance
        }
        self.positions: Dict[str, Position] = {}
        self.active_orders: Dict[int, OrderUpdate] = {}
        self.next_order_id = 1000
        self.next_trade_id = 1
        
        # 统计数据
        self.trade_id_counters: Dict[str, int] = {symbol: 1 for symbol in self.symbols}
        self.agg_trade_id_counters: Dict[str, int] = {symbol: 1 for symbol in self.symbols}
    
    def _get_base_price_for_symbol(self, symbol: str) -> Decimal:
        """获取交易对的基础价格"""
        price_map = {
            "BTCUSDT": Decimal("50000"),
            "ETHUSDT": Decimal("3000"),
            "BNBUSDT": Decimal("400"),
            "ADAUSDT": Decimal("1.5"),
            "DOTUSDT": Decimal("25"),
        }
        return price_map.get(symbol, Decimal("100"))
    
    async def generate_kline_data(self, symbol: str, interval: str) -> KlineData:
        """生成K线数据"""
        current_time = int(time.time() * 1000)
        
        # 计算K线时间窗口
        interval_ms = self._interval_to_ms(interval)
        kline_start_time = (current_time // interval_ms) * interval_ms
        kline_end_time = kline_start_time + interval_ms - 1
        
        # 生成价格数据
        price_engine = self.price_engines[symbol]
        volume_engine = self.volume_engines[symbol]
        
        # 获取或创建K线缓存
        if symbol not in self.kline_data:
            self.kline_data[symbol] = {}
        if interval not in self.kline_data[symbol]:
            self.kline_data[symbol][interval] = {}
        
        kline_cache = self.kline_data[symbol][interval]
        
        # 检查是否需要新K线
        if kline_start_time not in kline_cache:
            # 创建新K线
            open_price = price_engine.current_price
            kline_cache[kline_start_time] = {
                "open": open_price,
                "high": open_price,
                "low": open_price,
                "close": open_price,
                "volume": Decimal("0"),
                "quote_volume": Decimal("0"),
                "trade_count": 0,
                "taker_buy_volume": Decimal("0"),
                "taker_buy_quote_volume": Decimal("0")
            }
        
        # 更新当前K线
        current_kline = kline_cache[kline_start_time]
        new_price = price_engine.generate_next_price()
        volume = volume_engine.generate_volume()
        
        # 更新OHLC
        current_kline["close"] = new_price
        current_kline["high"] = max(current_kline["high"], new_price)
        current_kline["low"] = min(current_kline["low"], new_price)
        
        # 更新成交量
        current_kline["volume"] += volume
        quote_volume = volume * new_price
        current_kline["quote_volume"] += quote_volume
        current_kline["trade_count"] += 1
        
        # 模拟主动买入成交量
        taker_buy_ratio = Decimal(str(random.uniform(0.3, 0.7)))
        taker_buy_volume = volume * taker_buy_ratio
        current_kline["taker_buy_volume"] += taker_buy_volume
        current_kline["taker_buy_quote_volume"] += taker_buy_volume * new_price
        
        # 判断K线是否结束
        is_closed = current_time >= kline_end_time
        
        return KlineData(
            event_type="kline",
            event_time=current_time,
            symbol=symbol,
            start_time=kline_start_time,
            close_time=kline_end_time,
            interval=interval,
            first_trade_id=self.trade_id_counters[symbol],
            last_trade_id=self.trade_id_counters[symbol] + current_kline["trade_count"] - 1,
            open_price=current_kline["open"],
            close_price=current_kline["close"],
            high_price=current_kline["high"],
            low_price=current_kline["low"],
            base_asset_volume=current_kline["volume"],
            number_of_trades=current_kline["trade_count"],
            is_kline_closed=is_closed,
            quote_asset_volume=current_kline["quote_volume"],
            taker_buy_base_asset_volume=current_kline["taker_buy_volume"],
            taker_buy_quote_asset_volume=current_kline["taker_buy_quote_volume"]
        )
    
    async def generate_depth_update(self, symbol: str) -> DepthUpdate:
        """生成深度更新数据"""
        depth_engine = self.depth_engines[symbol]
        bids, asks = depth_engine.generate_depth_data()
        
        current_time = int(time.time() * 1000)
        
        return DepthUpdate(
            event_type="depthUpdate",
            event_time=current_time,
            transaction_time=current_time,
            symbol=symbol,
            first_update_id=random.randint(1000000, 9999999),
            final_update_id=random.randint(1000000, 9999999),
            prev_final_update_id=random.randint(1000000, 9999999),
            bids=bids,
            asks=asks
        )
    
    async def generate_mark_price_update(self, symbol: str) -> MarkPriceUpdate:
        """生成标记价格更新"""
        price_engine = self.price_engines[symbol]
        mark_price = price_engine.current_price
        
        # 指数价格稍微偏离标记价格
        index_price = mark_price * Decimal(str(random.uniform(0.999, 1.001)))
        
        # 模拟资金费率
        funding_rate = Decimal(str(random.uniform(-0.001, 0.001)))
        
        # 下次资金费时间（每8小时）
        current_time = int(time.time() * 1000)
        next_funding = ((current_time // (8 * 3600 * 1000)) + 1) * (8 * 3600 * 1000)
        
        return MarkPriceUpdate(
            event_type="markPriceUpdate",
            event_time=current_time,
            symbol=symbol,
            mark_price=mark_price,
            index_price=index_price,
            estimated_settle_price=mark_price,  # 简化处理
            funding_rate=funding_rate,
            next_funding_time=next_funding
        )
    
    async def generate_agg_trade_data(self, symbol: str) -> AggTradeData:
        """生成归集交易数据"""
        price_engine = self.price_engines[symbol]
        volume_engine = self.volume_engines[symbol]
        
        # 生成交易价格（在当前价格附近小幅波动）
        base_price = price_engine.current_price
        price_change = Decimal(str(random.uniform(-0.001, 0.001)))
        trade_price = base_price * (1 + price_change)
        
        # 生成交易数量
        quantity = volume_engine.generate_volume()
        
        # 检查是否为大额交易
        if random.random() < self.market_config.large_trade_prob:
            quantity *= Decimal(str(self.market_config.large_trade_multiplier))
        
        current_time = int(time.time() * 1000)
        trade_id = self.agg_trade_id_counters[symbol]
        self.agg_trade_id_counters[symbol] += 1
        
        return AggTradeData(
            event_type="aggTrade",
            event_time=current_time,
            symbol=symbol,
            agg_trade_id=trade_id,
            price=trade_price,
            quantity=quantity,
            first_trade_id=trade_id,
            last_trade_id=trade_id,
            trade_time=current_time - random.randint(0, 100),
            is_buyer_maker=random.choice([True, False])
        )
    
    async def generate_ticker_data(self, symbol: str) -> TickerData:
        """生成24小时价格统计数据"""
        price_engine = self.price_engines[symbol]
        current_price = price_engine.current_price
        
        # 模拟24小时前的价格
        price_24h_ago = current_price * Decimal(str(random.uniform(0.95, 1.05)))
        price_change = current_price - price_24h_ago
        price_change_percent = (price_change / price_24h_ago) * 100
        
        # 模拟其他统计数据
        high_price = current_price * Decimal(str(random.uniform(1.01, 1.05)))
        low_price = current_price * Decimal(str(random.uniform(0.95, 0.99)))
        
        volume_24h = Decimal(str(random.uniform(10000, 100000)))
        quote_volume_24h = volume_24h * current_price * Decimal(str(random.uniform(0.8, 1.2)))
        
        current_time = int(time.time() * 1000)
        
        return TickerData(
            event_type="24hrTicker",
            event_time=current_time,
            symbol=symbol,
            price_change=price_change,
            price_change_percent=price_change_percent,
            weighted_avg_price=current_price * Decimal(str(random.uniform(0.99, 1.01))),
            last_price=current_price,
            last_quantity=Decimal(str(random.uniform(1, 100))),
            open_price=price_24h_ago,
            high_price=high_price,
            low_price=low_price,
            total_traded_base_asset_volume=volume_24h,
            total_traded_quote_asset_volume=quote_volume_24h,
            statistics_open_time=current_time - (24 * 3600 * 1000),
            statistics_close_time=current_time,
            first_trade_id=1,
            last_trade_id=random.randint(10000, 99999),
            total_number_of_trades=random.randint(1000, 10000)
        )
    
    async def generate_account_update(self) -> AccountUpdate:
        """生成账户更新数据"""
        # 更新余额（模拟盈亏变化）
        for asset in self.account_balances:
            change_rate = Decimal(str(random.uniform(-0.01, 0.01)))
            self.account_balances[asset] *= (1 + change_rate)
        
        # 创建余额对象
        balances = []
        for asset, wallet_balance in self.account_balances.items():
            unrealized_pnl = wallet_balance * Decimal(str(random.uniform(-0.1, 0.1)))
            balance = Balance(
                asset=asset,
                wallet_balance=wallet_balance,
                unrealized_pnl=unrealized_pnl,
                margin_balance=wallet_balance + unrealized_pnl,
                maint_margin=wallet_balance * Decimal("0.05"),
                initial_margin=wallet_balance * Decimal("0.1"),
                position_initial_margin=wallet_balance * Decimal("0.08"),
                open_order_initial_margin=wallet_balance * Decimal("0.02"),
                cross_wallet_balance=wallet_balance,
                cross_unrealized_pnl=unrealized_pnl,
                available_balance=wallet_balance * Decimal("0.9"),
                max_withdraw_amount=wallet_balance * Decimal("0.8")
            )
            balances.append(balance)
        
        # 更新持仓
        positions = list(self.positions.values())
        
        current_time = int(time.time() * 1000)
        
        return AccountUpdate(
            event_type="ACCOUNT_UPDATE",
            event_time=current_time,
            transaction_time=current_time,
            update_data={},
            balances=balances,
            positions=positions
        )
    
    def _interval_to_ms(self, interval: str) -> int:
        """将时间间隔转换为毫秒"""
        unit = interval[-1]
        value = int(interval[:-1])
        
        multipliers = {
            's': 1000,
            'm': 60 * 1000,
            'h': 3600 * 1000,
            'd': 24 * 3600 * 1000,
            'w': 7 * 24 * 3600 * 1000
        }
        
        return value * multipliers.get(unit, 60 * 1000)  # 默认分钟


class MockWebSocketConnection:
    """Mock WebSocket连接实现"""
    
    def __init__(self, stream_name: str, data_generator: FuturesMockDataGenerator):
        self.stream_name = stream_name
        self.data_generator = data_generator
        self.is_active = True
        self._message_queue = asyncio.Queue()
        self._producer_task: Optional[asyncio.Task] = None
        
        # 解析流名称
        self.symbol, self.stream_type = self._parse_stream_name(stream_name)
    
    def _parse_stream_name(self, stream_name: str) -> Tuple[str, str]:
        """解析流名称获取交易对和流类型"""
        if '@' in stream_name:
            symbol, stream_type = stream_name.upper().split('@', 1)
            return symbol, stream_type
        else:
            return stream_name.upper(), "unknown"
    
    async def start_data_production(self):
        """启动数据生成任务"""
        if self._producer_task is not None:
            return
        
        self._producer_task = asyncio.create_task(self._data_producer())
    
    async def _data_producer(self):
        """数据生产者协程"""
        try:
            while self.is_active:
                try:
                    # 根据流类型生成对应数据
                    message = await self._generate_message_for_stream()
                    if message:
                        await self._message_queue.put(message)
                    
                    # 等待下次生成
                    await asyncio.sleep(self._get_generation_interval())
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"Mock数据生成错误: {e}")
                    await asyncio.sleep(1.0)
        
        finally:
            self.is_active = False
    
    async def _generate_message_for_stream(self) -> Optional[Dict[str, Any]]:
        """为特定流类型生成消息"""
        try:
            if "kline" in self.stream_type:
                # K线数据
                interval = self.stream_type.split('_')[1] if '_' in self.stream_type else "1m"
                data = await self.data_generator.generate_kline_data(self.symbol, interval)
                return self._format_kline_message(data)
            
            elif "depth" in self.stream_type:
                # 深度数据
                data = await self.data_generator.generate_depth_update(self.symbol)
                return self._format_depth_message(data)
            
            elif "markPrice" in self.stream_type:
                # 标记价格数据
                data = await self.data_generator.generate_mark_price_update(self.symbol)
                return self._format_mark_price_message(data)
            
            elif "aggTrade" in self.stream_type:
                # 归集交易数据
                data = await self.data_generator.generate_agg_trade_data(self.symbol)
                return self._format_agg_trade_message(data)
            
            elif "ticker" in self.stream_type:
                # 价格统计数据
                data = await self.data_generator.generate_ticker_data(self.symbol)
                return self._format_ticker_message(data)
            
            else:
                return None
        
        except Exception as e:
            print(f"生成消息失败 [{self.stream_name}]: {e}")
            return None
    
    def _format_kline_message(self, data: KlineData) -> Dict[str, Any]:
        """格式化K线消息"""
        return {
            "e": data.event_type,
            "E": data.event_time,
            "s": data.symbol,
            "k": {
                "t": data.start_time,
                "T": data.close_time,
                "s": data.symbol,
                "i": data.interval,
                "f": data.first_trade_id,
                "L": data.last_trade_id,
                "o": str(data.open_price),
                "c": str(data.close_price),
                "h": str(data.high_price),
                "l": str(data.low_price),
                "v": str(data.base_asset_volume),
                "n": data.number_of_trades,
                "x": data.is_kline_closed,
                "q": str(data.quote_asset_volume),
                "V": str(data.taker_buy_base_asset_volume),
                "Q": str(data.taker_buy_quote_asset_volume),
                "B": data.ignore
            }
        }
    
    def _format_depth_message(self, data: DepthUpdate) -> Dict[str, Any]:
        """格式化深度消息"""
        return {
            "e": data.event_type,
            "E": data.event_time,
            "T": data.transaction_time,
            "s": data.symbol,
            "U": data.first_update_id,
            "u": data.final_update_id,
            "pu": data.prev_final_update_id,
            "b": data.bids,
            "a": data.asks
        }
    
    def _format_mark_price_message(self, data: MarkPriceUpdate) -> Dict[str, Any]:
        """格式化标记价格消息"""
        return {
            "e": data.event_type,
            "E": data.event_time,
            "s": data.symbol,
            "p": str(data.mark_price),
            "i": str(data.index_price),
            "P": str(data.estimated_settle_price),
            "r": str(data.funding_rate),
            "T": data.next_funding_time
        }
    
    def _format_agg_trade_message(self, data: AggTradeData) -> Dict[str, Any]:
        """格式化归集交易消息"""
        return {
            "e": data.event_type,
            "E": data.event_time,
            "s": data.symbol,
            "a": data.agg_trade_id,
            "p": str(data.price),
            "q": str(data.quantity),
            "f": data.first_trade_id,
            "l": data.last_trade_id,
            "T": data.trade_time,
            "m": data.is_buyer_maker
        }
    
    def _format_ticker_message(self, data: TickerData) -> Dict[str, Any]:
        """格式化价格统计消息"""
        return {
            "e": data.event_type,
            "E": data.event_time,
            "s": data.symbol,
            "p": str(data.price_change),
            "P": str(data.price_change_percent),
            "w": str(data.weighted_avg_price),
            "c": str(data.last_price),
            "Q": str(data.last_quantity),
            "o": str(data.open_price),
            "h": str(data.high_price),
            "l": str(data.low_price),
            "v": str(data.total_traded_base_asset_volume),
            "q": str(data.total_traded_quote_asset_volume),
            "O": data.statistics_open_time,
            "C": data.statistics_close_time,
            "F": data.first_trade_id,
            "L": data.last_trade_id,
            "n": data.total_number_of_trades
        }
    
    def _get_generation_interval(self) -> float:
        """获取数据生成间隔"""
        if "kline" in self.stream_type:
            return 1.0  # K线每秒更新一次
        elif "depth" in self.stream_type:
            return 0.1  # 深度数据更新频率高
        elif "markPrice" in self.stream_type:
            return 1.0  # 标记价格每秒更新
        elif "aggTrade" in self.stream_type:
            return 0.2  # 交易数据频率较高
        elif "ticker" in self.stream_type:
            return 1.0  # 价格统计每秒更新
        else:
            return 1.0  # 默认间隔
    
    async def recv(self) -> Optional[Dict[str, Any]]:
        """接收消息"""
        if not self.is_active:
            return None
        
        try:
            return await asyncio.wait_for(self._message_queue.get(), timeout=5.0)
        except asyncio.TimeoutError:
            return None
    
    async def close(self):
        """关闭连接"""
        self.is_active = False
        if self._producer_task:
            self._producer_task.cancel()
            try:
                await self._producer_task
            except asyncio.CancelledError:
                pass
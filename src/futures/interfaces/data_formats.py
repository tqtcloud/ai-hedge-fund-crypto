"""
期货交易WebSocket数据格式定义

定义了币安期货WebSocket API的标准数据格式，包括：
- 市场数据格式 (K线、深度、交易数据等)
- 账户数据格式 (余额、持仓、订单更新等)
- 错误和事件格式

所有数据类使用dataclass和类型注解，确保类型安全和代码清晰度。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
from enum import Enum
import time


class EventType(Enum):
    """WebSocket事件类型枚举"""
    # 市场数据事件
    KLINE = "kline"                           # K线/蜡烛图数据
    CONTINUOUS_KLINE = "continuous_kline"     # 连续合约K线
    MINI_TICKER = "24hrMiniTicker"            # 24小时价格统计
    TICKER = "24hrTicker"                     # 24小时价格统计(完整)
    BOOK_TICKER = "bookTicker"                # 实时价格
    AGG_TRADE = "aggTrade"                    # 归集交易
    TRADE = "trade"                           # 逐笔交易
    DEPTH_UPDATE = "depthUpdate"              # 订单簿深度更新
    MARK_PRICE_UPDATE = "markPriceUpdate"     # 标记价格更新
    INDEX_PRICE_UPDATE = "indexPriceUpdate"   # 指数价格更新
    
    # 账户数据事件
    ACCOUNT_UPDATE = "ACCOUNT_UPDATE"         # 账户更新
    ORDER_TRADE_UPDATE = "ORDER_TRADE_UPDATE" # 订单交易更新
    ACCOUNT_CONFIG_UPDATE = "ACCOUNT_CONFIG_UPDATE" # 账户配置更新
    
    # 系统事件
    ERROR = "error"                           # 错误事件
    LISTEN_KEY_EXPIRED = "listenKeyExpired"   # 监听密钥过期


class PositionSide(Enum):
    """持仓方向枚举"""
    BOTH = "BOTH"      # 单向持仓
    LONG = "LONG"      # 多头持仓
    SHORT = "SHORT"    # 空头持仓


class OrderType(Enum):
    """订单类型枚举"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TRAILING_STOP_MARKET = "TRAILING_STOP_MARKET"


class OrderStatus(Enum):
    """订单状态枚举"""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class TimeInForce(Enum):
    """有效时间类型枚举"""
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    GTX = "GTX"  # Good Till Crossing


@dataclass
class WebSocketMessage:
    """WebSocket消息基类"""
    stream: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    event_type: Optional[EventType] = None
    event_time: Optional[int] = None
    
    def __post_init__(self):
        """消息处理后初始化"""
        if self.data and "e" in self.data:
            self.event_type = EventType(self.data["e"])
        if self.data and "E" in self.data:
            self.event_time = self.data["E"]


@dataclass
class KlineData:
    """K线数据"""
    event_type: str                    # 事件类型 "kline"
    event_time: int                    # 事件时间
    symbol: str                        # 交易对
    start_time: int                    # 这根K线的起始时间
    close_time: int                    # 这根K线的结束时间
    interval: str                      # K线间隔
    first_trade_id: int                # 这根K线期间第一笔成交ID
    last_trade_id: int                 # 这根K线期间末一笔成交ID
    open_price: Decimal                # 这根K线期间第一笔成交价
    close_price: Decimal               # 这根K线期间末一笔成交价
    high_price: Decimal                # 这根K线期间最高成交价
    low_price: Decimal                 # 这根K线期间最低成交价
    base_asset_volume: Decimal         # 这根K线期间成交量
    number_of_trades: int              # 这根K线期间成交笔数
    is_kline_closed: bool              # 这根K线是否完结(是否已经开始下一根K线)
    quote_asset_volume: Decimal        # 这根K线期间成交额
    taker_buy_base_asset_volume: Decimal  # 主动买入成交量
    taker_buy_quote_asset_volume: Decimal # 主动买入成交额
    ignore: str = "0"                  # 忽略此参数
    
    @classmethod
    def from_websocket(cls, data: Dict[str, Any]) -> 'KlineData':
        """从WebSocket数据创建K线数据对象"""
        k = data['k']
        return cls(
            event_type=data['e'],
            event_time=data['E'],
            symbol=data['s'],
            start_time=k['t'],
            close_time=k['T'],
            interval=k['i'],
            first_trade_id=k['f'],
            last_trade_id=k['L'],
            open_price=Decimal(k['o']),
            close_price=Decimal(k['c']),
            high_price=Decimal(k['h']),
            low_price=Decimal(k['l']),
            base_asset_volume=Decimal(k['v']),
            number_of_trades=k['n'],
            is_kline_closed=k['x'],
            quote_asset_volume=Decimal(k['q']),
            taker_buy_base_asset_volume=Decimal(k['V']),
            taker_buy_quote_asset_volume=Decimal(k['Q']),
            ignore=k.get('B', '0')
        )


@dataclass
class DepthUpdate:
    """订单簿深度更新数据"""
    event_type: str                    # 事件类型 "depthUpdate"
    event_time: int                    # 事件时间
    transaction_time: int              # 撮合引擎时间
    symbol: str                        # 交易对
    first_update_id: int               # 从上次推送至今新增的第一个 update Id
    final_update_id: int               # 从上次推送至今新增的最后一个 update Id
    prev_final_update_id: int          # 上次推送的最后一个 update Id
    bids: List[List[str]]              # 变动的买单深度
    asks: List[List[str]]              # 变动的卖单深度
    
    @classmethod
    def from_websocket(cls, data: Dict[str, Any]) -> 'DepthUpdate':
        """从WebSocket数据创建深度更新对象"""
        return cls(
            event_type=data['e'],
            event_time=data['E'],
            transaction_time=data['T'],
            symbol=data['s'],
            first_update_id=data['U'],
            final_update_id=data['u'],
            prev_final_update_id=data['pu'],
            bids=data['b'],
            asks=data['a']
        )


@dataclass
class MarkPriceUpdate:
    """标记价格更新数据"""
    event_type: str                    # 事件类型 "markPriceUpdate"
    event_time: int                    # 事件时间
    symbol: str                        # 交易对
    mark_price: Decimal                # 标记价格
    index_price: Decimal               # 指数价格
    estimated_settle_price: Decimal    # 预估结算价,仅在交割开始前最后一小时有意义
    funding_rate: Decimal              # 本期资金费率
    next_funding_time: int             # 下次资金费时间
    
    @classmethod
    def from_websocket(cls, data: Dict[str, Any]) -> 'MarkPriceUpdate':
        """从WebSocket数据创建标记价格更新对象"""
        return cls(
            event_type=data['e'],
            event_time=data['E'],
            symbol=data['s'],
            mark_price=Decimal(data['p']),
            index_price=Decimal(data['i']),
            estimated_settle_price=Decimal(data.get('P', '0')),
            funding_rate=Decimal(data['r']),
            next_funding_time=data['T']
        )


@dataclass
class AggTradeData:
    """归集交易数据"""
    event_type: str                    # 事件类型 "aggTrade"
    event_time: int                    # 事件时间
    symbol: str                        # 交易对
    agg_trade_id: int                  # 归集交易ID
    price: Decimal                     # 成交价格
    quantity: Decimal                  # 成交数量
    first_trade_id: int                # 被归集的首个交易ID
    last_trade_id: int                 # 被归集的末个交易ID
    trade_time: int                    # 成交时间
    is_buyer_maker: bool               # 买方是否为挂单方
    
    @classmethod
    def from_websocket(cls, data: Dict[str, Any]) -> 'AggTradeData':
        """从WebSocket数据创建归集交易数据对象"""
        return cls(
            event_type=data['e'],
            event_time=data['E'],
            symbol=data['s'],
            agg_trade_id=data['a'],
            price=Decimal(data['p']),
            quantity=Decimal(data['q']),
            first_trade_id=data['f'],
            last_trade_id=data['l'],
            trade_time=data['T'],
            is_buyer_maker=data['m']
        )


@dataclass
class TickerData:
    """24小时价格变动统计"""
    event_type: str                    # 事件类型 "24hrTicker"
    event_time: int                    # 事件时间
    symbol: str                        # 交易对
    price_change: Decimal              # 24小时价格变动
    price_change_percent: Decimal      # 24小时价格变动百分比
    weighted_avg_price: Decimal        # 平均价格
    last_price: Decimal                # 最近一次成交价格
    last_quantity: Decimal             # 最近一次成交数量
    open_price: Decimal                # 24小时内第一次成交的价格
    high_price: Decimal                # 24小时内最高成交价
    low_price: Decimal                 # 24小时内最低成交价
    total_traded_base_asset_volume: Decimal  # 24小时内成交量
    total_traded_quote_asset_volume: Decimal # 24小时内成交额
    statistics_open_time: int          # 统计开始时间
    statistics_close_time: int         # 统计结束时间
    first_trade_id: int                # 24小时内第一笔成交交易ID
    last_trade_id: int                 # 24小时内最后一笔成交交易ID
    total_number_of_trades: int        # 24小时内成交笔数
    
    @classmethod
    def from_websocket(cls, data: Dict[str, Any]) -> 'TickerData':
        """从WebSocket数据创建价格统计对象"""
        return cls(
            event_type=data['e'],
            event_time=data['E'],
            symbol=data['s'],
            price_change=Decimal(data['p']),
            price_change_percent=Decimal(data['P']),
            weighted_avg_price=Decimal(data['w']),
            last_price=Decimal(data['c']),
            last_quantity=Decimal(data['Q']),
            open_price=Decimal(data['o']),
            high_price=Decimal(data['h']),
            low_price=Decimal(data['l']),
            total_traded_base_asset_volume=Decimal(data['v']),
            total_traded_quote_asset_volume=Decimal(data['q']),
            statistics_open_time=data['O'],
            statistics_close_time=data['C'],
            first_trade_id=data['F'],
            last_trade_id=data['L'],
            total_number_of_trades=data['n']
        )


@dataclass
class Position:
    """持仓信息"""
    symbol: str                        # 交易对
    position_side: PositionSide        # 持仓方向
    position_amount: Decimal           # 持仓数量
    entry_price: Decimal               # 平均持仓价格
    breakeven_price: Decimal           # 盈亏平衡价
    mark_price: Decimal                # 当前标记价格
    unrealized_pnl: Decimal            # 持仓未实现盈亏
    margin_type: str                   # 逐仓或全仓，ISOLATED(逐仓), CROSSED(全仓)
    isolated_margin: Decimal           # 逐仓保证金
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """从字典数据创建持仓对象"""
        return cls(
            symbol=data['s'],
            position_side=PositionSide(data['ps']),
            position_amount=Decimal(data['pa']),
            entry_price=Decimal(data['ep']),
            breakeven_price=Decimal(data.get('bep', '0')),
            mark_price=Decimal(data['mp']),
            unrealized_pnl=Decimal(data['up']),
            margin_type=data['mt'],
            isolated_margin=Decimal(data['iw'])
        )


@dataclass
class Balance:
    """余额信息"""
    asset: str                         # 资产名称
    wallet_balance: Decimal            # 钱包余额
    unrealized_pnl: Decimal            # 未实现盈亏
    margin_balance: Decimal            # 保证金余额
    maint_margin: Decimal              # 维持保证金
    initial_margin: Decimal            # 当前所需起始保证金
    position_initial_margin: Decimal   # 持仓所需起始保证金
    open_order_initial_margin: Decimal # 当前挂单所需起始保证金
    cross_wallet_balance: Decimal      # 全仓账户余额
    cross_unrealized_pnl: Decimal      # 全仓持仓未实现盈亏
    available_balance: Decimal         # 可用余额
    max_withdraw_amount: Decimal       # 最大可转出余额
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Balance':
        """从字典数据创建余额对象"""
        return cls(
            asset=data['a'],
            wallet_balance=Decimal(data['wb']),
            unrealized_pnl=Decimal(data['up']),
            margin_balance=Decimal(data['bc']),
            maint_margin=Decimal(data['mm']),
            initial_margin=Decimal(data['im']),
            position_initial_margin=Decimal(data['pim']),
            open_order_initial_margin=Decimal(data['oim']),
            cross_wallet_balance=Decimal(data['cw']),
            cross_unrealized_pnl=Decimal(data['cp']),
            available_balance=Decimal(data['ab']),
            max_withdraw_amount=Decimal(data['mwa'])
        )


@dataclass
class OrderUpdate:
    """订单更新数据"""
    symbol: str                        # 交易对
    client_order_id: str               # 用户自定义的订单号
    order_side: str                    # 订单方向
    order_type: OrderType              # 订单类型
    time_in_force: TimeInForce         # 有效方法
    original_quantity: Decimal         # 订单原始数量
    original_price: Decimal            # 订单原始价格
    average_price: Decimal             # 订单平均价格
    stop_price: Decimal                # 条件订单触发价格
    execution_type: str                # 本次事件的具体执行类型
    order_status: OrderStatus          # 订单的当前状态
    order_id: int                      # 订单ID
    last_executed_quantity: Decimal    # 订单末次成交数量
    cumulative_filled_quantity: Decimal # 订单累计成交数量
    last_executed_price: Decimal       # 订单末次成交价格
    commission_amount: Decimal         # 手续费数量
    commission_asset: Optional[str]    # 手续费资产类别
    trade_time: int                    # 成交时间
    trade_id: int                      # 成交ID
    bids_notional: Decimal             # 买单净值
    ask_notional: Decimal              # 卖单净值
    is_maker_side: bool                # 该成交是作为挂单成交吗
    reduce_only: bool                  # 是否为只减仓单
    stop_price_working_type: str       # 条件价格触发类型
    original_order_type: OrderType     # 触发前订单类型
    position_side: PositionSide        # 持仓方向
    close_all: bool                    # 是否条件全平仓单
    activation_price: Decimal          # 跟踪止损激活价格
    callback_rate: Decimal             # 跟踪止损回调比例
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrderUpdate':
        """从字典数据创建订单更新对象"""
        return cls(
            symbol=data['s'],
            client_order_id=data['c'],
            order_side=data['S'],
            order_type=OrderType(data['o']),
            time_in_force=TimeInForce(data['f']),
            original_quantity=Decimal(data['q']),
            original_price=Decimal(data['p']),
            average_price=Decimal(data['ap']),
            stop_price=Decimal(data['sp']),
            execution_type=data['x'],
            order_status=OrderStatus(data['X']),
            order_id=data['i'],
            last_executed_quantity=Decimal(data['l']),
            cumulative_filled_quantity=Decimal(data['z']),
            last_executed_price=Decimal(data['L']),
            commission_amount=Decimal(data['n']),
            commission_asset=data.get('N'),
            trade_time=data['T'],
            trade_id=data['t'],
            bids_notional=Decimal(data['b']),
            ask_notional=Decimal(data['a']),
            is_maker_side=data['m'],
            reduce_only=data['R'],
            stop_price_working_type=data['wt'],
            original_order_type=OrderType(data['ot']),
            position_side=PositionSide(data['ps']),
            close_all=data['cp'],
            activation_price=Decimal(data.get('AP', '0')),
            callback_rate=Decimal(data.get('cr', '0'))
        )


@dataclass
class AccountUpdate:
    """账户更新数据"""
    event_type: str                    # 事件类型 "ACCOUNT_UPDATE"
    event_time: int                    # 事件时间
    transaction_time: int              # 撮合时间
    update_data: Dict[str, Any]        # 账户更新的具体数据
    balances: List[Balance] = field(default_factory=list)        # 余额信息
    positions: List[Position] = field(default_factory=list)      # 持仓信息
    
    @classmethod
    def from_websocket(cls, data: Dict[str, Any]) -> 'AccountUpdate':
        """从WebSocket数据创建账户更新对象"""
        account_data = data['a']
        
        # 解析余额信息
        balances = []
        if 'B' in account_data:
            balances = [Balance.from_dict(b) for b in account_data['B']]
        
        # 解析持仓信息
        positions = []
        if 'P' in account_data:
            positions = [Position.from_dict(p) for p in account_data['P']]
        
        return cls(
            event_type=data['e'],
            event_time=data['E'],
            transaction_time=data['T'],
            update_data=account_data,
            balances=balances,
            positions=positions
        )


@dataclass
class OrderTradeUpdate:
    """订单交易更新数据"""
    event_type: str                    # 事件类型 "ORDER_TRADE_UPDATE"
    event_time: int                    # 事件时间
    transaction_time: int              # 撮合时间
    order: OrderUpdate                 # 订单信息
    
    @classmethod
    def from_websocket(cls, data: Dict[str, Any]) -> 'OrderTradeUpdate':
        """从WebSocket数据创建订单交易更新对象"""
        return cls(
            event_type=data['e'],
            event_time=data['E'],
            transaction_time=data['T'],
            order=OrderUpdate.from_dict(data['o'])
        )


@dataclass
class ErrorMessage:
    """错误消息"""
    event_type: str = "error"          # 事件类型
    error_code: Optional[int] = None   # 错误代码
    error_msg: str = ""                # 错误消息
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    
    @classmethod
    def from_exception(cls, exception: Exception) -> 'ErrorMessage':
        """从异常创建错误消息"""
        return cls(
            error_msg=str(exception),
            timestamp=int(time.time() * 1000)
        )


@dataclass
class ConnectionStatus:
    """连接状态信息"""
    is_connected: bool = False         # 是否已连接
    connection_time: Optional[int] = None  # 连接时间
    last_message_time: Optional[int] = None  # 最后消息时间
    reconnect_count: int = 0           # 重连次数
    error_count: int = 0              # 错误次数
    
    def update_connection(self, connected: bool):
        """更新连接状态"""
        self.is_connected = connected
        if connected:
            self.connection_time = int(time.time() * 1000)
        
    def update_message_time(self):
        """更新最后消息时间"""
        self.last_message_time = int(time.time() * 1000)
        
    def increment_reconnect(self):
        """增加重连计数"""
        self.reconnect_count += 1
        
    def increment_error(self):
        """增加错误计数"""
        self.error_count += 1


# 消息类型映射
MESSAGE_TYPE_MAPPING = {
    EventType.KLINE: KlineData,
    EventType.DEPTH_UPDATE: DepthUpdate,
    EventType.MARK_PRICE_UPDATE: MarkPriceUpdate,
    EventType.AGG_TRADE: AggTradeData,
    EventType.TICKER: TickerData,
    EventType.ACCOUNT_UPDATE: AccountUpdate,
    EventType.ORDER_TRADE_UPDATE: OrderTradeUpdate,
    EventType.ERROR: ErrorMessage,
}


def parse_websocket_message(raw_data: Dict[str, Any]) -> Union[WebSocketMessage, Any]:
    """
    解析WebSocket原始消息为对应的数据类对象
    
    Args:
        raw_data: 原始WebSocket消息数据
        
    Returns:
        解析后的数据对象
    """
    try:
        # 处理multiplexed streams格式: {"stream": "...", "data": {...}}
        if "stream" in raw_data and "data" in raw_data:
            data = raw_data["data"]
            stream = raw_data["stream"]
        else:
            data = raw_data
            stream = None
        
        # 获取事件类型
        event_type_str = data.get("e")
        if not event_type_str:
            return WebSocketMessage(stream=stream, data=data)
        
        try:
            event_type = EventType(event_type_str)
        except ValueError:
            # 未知事件类型，返回原始消息
            return WebSocketMessage(stream=stream, data=data, event_type=None)
        
        # 根据事件类型解析为对应的数据类
        message_class = MESSAGE_TYPE_MAPPING.get(event_type)
        if message_class and hasattr(message_class, 'from_websocket'):
            return message_class.from_websocket(data)
        elif message_class and hasattr(message_class, 'from_dict'):
            return message_class.from_dict(data)
        else:
            # 返回基础消息格式
            return WebSocketMessage(stream=stream, data=data, event_type=event_type)
            
    except Exception as e:
        # 解析失败，返回错误消息
        return ErrorMessage.from_exception(e)
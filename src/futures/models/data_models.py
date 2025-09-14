"""
期货交易核心数据类模块

定义期货交易系统中使用的核心数据结构，包括交易信号、
仓位信息、保证金状态和验证结果等。
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, Literal, TYPE_CHECKING
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
import logging
from enum import Enum

from ...utils.exceptions import (
    ContractTradingError,
    MarginInsufficientError,
    LeverageExceedsLimitError,
    LiquidationRiskError,
    PositionSizeError
)

# 官方SDK集成支持
if TYPE_CHECKING:
    try:
        from binance_sdk_derivatives_trading_usds_futures.websocket_api.models import (
            PositionInformationResponse,
            AccountInformationResponse
        )
        # SymbolInformationResponse在当前SDK版本中不存在，使用占位符
        SymbolInformationResponse = Any
    except ImportError:
        # 如果SDK不可用，定义类型占位符
        PositionInformationResponse = Any
        AccountInformationResponse = Any
        SymbolInformationResponse = Any
else:
    try:
        from binance_sdk_derivatives_trading_usds_futures.websocket_api.models import (
            PositionInformationResponse,
            AccountInformationResponse
        )
        # SymbolInformationResponse在当前SDK版本中不存在，使用None
        SymbolInformationResponse = None
    except ImportError:
        PositionInformationResponse = None
        AccountInformationResponse = None
        SymbolInformationResponse = None

logger = logging.getLogger(__name__)

# SDK兼容性检查
SDK_AVAILABLE = PositionInformationResponse is not None

if not SDK_AVAILABLE:
    logger.warning(
        "官方Binance SDK未找到。某些功能可能不可用。\n"
        "请安装: pip install binance-sdk-derivatives-trading-usds-futures"
    )


class TradingDirection(Enum):
    """交易方向枚举"""
    LONG = "long"      # 做多
    SHORT = "short"    # 做空
    NEUTRAL = "neutral"  # 中性/无方向


class OperationType(Enum):
    """操作类型枚举"""
    OPEN = "open"       # 开仓
    CLOSE = "close"     # 平仓
    ADD = "add"         # 加仓
    REDUCE = "reduce"   # 减仓
    HOLD = "hold"       # 持仓不变


class PositionSide(Enum):
    """仓位方向枚举"""
    BOTH = "BOTH"       # 双向持仓模式
    LONG = "LONG"       # 单向做多
    SHORT = "SHORT"     # 单向做空


class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = "low"         # 低风险
    MEDIUM = "medium"   # 中等风险
    HIGH = "high"       # 高风险
    CRITICAL = "critical"  # 关键风险
    EMERGENCY = "emergency"  # 紧急风险


class ValidationSeverity(Enum):
    """验证问题严重程度枚举"""
    INFO = "info"       # 信息
    WARNING = "warning" # 警告
    ERROR = "error"     # 错误
    CRITICAL = "critical"  # 严重错误


@dataclass
class FuturesSignal:
    """
    期货交易信号类
    
    包含完整的交易决策信息，包括方向、操作类型、
    建议杠杆、仓位大小、入场价格、止盈止损等。
    """
    ticker: str                                    # 交易对符号
    direction: TradingDirection                    # 交易方向
    operation_type: OperationType                  # 操作类型
    confidence: float                              # 信号置信度 (0-100)
    strength: float                                # 信号强度 (0-1)
    
    # 仓位和杠杆信息
    suggested_leverage: float = 1.0                # 建议杠杆倍数
    position_size: Optional[float] = None          # 仓位大小 (USDT)
    position_ratio: Optional[float] = None         # 仓位比例 (0-1)
    
    # 价格信息
    entry_price: Optional[float] = None            # 入场价格
    current_price: Optional[float] = None          # 当前价格
    
    # 止盈止损
    take_profit_price: Optional[float] = None      # 止盈价格
    stop_loss_price: Optional[float] = None        # 止损价格
    take_profit_ratio: Optional[float] = None      # 止盈比例
    stop_loss_ratio: Optional[float] = None        # 止损比例
    
    # 时间信息
    timestamp: datetime = field(default_factory=datetime.now)  # 信号生成时间
    expiry_time: Optional[datetime] = None         # 信号过期时间
    
    # 市场环境
    volatility: Optional[float] = None             # 市场波动率
    market_trend: Optional[str] = None             # 市场趋势
    volume_profile: Optional[str] = None           # 成交量特征
    
    # 技术指标背景
    indicators: Dict[str, Any] = field(default_factory=dict)  # 技术指标值
    timeframes: List[str] = field(default_factory=list)      # 分析时间框架
    
    # 风险评估
    risk_level: RiskLevel = RiskLevel.MEDIUM       # 风险等级
    max_loss_percentage: Optional[float] = None    # 最大损失百分比
    
    # 元数据
    strategy_source: Optional[str] = None          # 策略来源
    signal_id: Optional[str] = None                # 信号ID
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def __post_init__(self):
        """初始化后验证"""
        self._validate_signal()
    
    def _validate_signal(self) -> None:
        """验证信号参数的有效性"""
        # 验证置信度范围
        if not 0 <= self.confidence <= 100:
            raise ValueError(f"置信度必须在0-100范围内，当前值: {self.confidence}")
        
        # 验证强度范围
        if not 0 <= self.strength <= 1:
            raise ValueError(f"信号强度必须在0-1范围内，当前值: {self.strength}")
        
        # 验证杠杆倍数
        if self.suggested_leverage <= 0:
            raise ValueError(f"杠杆倍数必须大于0，当前值: {self.suggested_leverage}")
        
        # 验证仓位比例
        if self.position_ratio is not None and not 0 <= self.position_ratio <= 1:
            raise ValueError(f"仓位比例必须在0-1范围内，当前值: {self.position_ratio}")
        
        # 验证价格为正数
        for price_field in ['entry_price', 'current_price', 'take_profit_price', 'stop_loss_price']:
            price = getattr(self, price_field)
            if price is not None and price <= 0:
                raise ValueError(f"{price_field}必须大于0，当前值: {price}")
    
    def is_valid(self) -> bool:
        """检查信号是否有效（未过期且置信度足够）"""
        try:
            # 检查是否过期
            if self.expiry_time and datetime.now() > self.expiry_time:
                return False
            
            # 检查置信度阈值
            min_confidence = 30.0  # 最小置信度阈值
            if self.confidence < min_confidence:
                return False
            
            return True
        except Exception as e:
            logger.warning(f"信号有效性检查失败: {e}")
            return False
    
    def calculate_risk_reward_ratio(self) -> Optional[float]:
        """计算风险收益比"""
        if not all([self.entry_price, self.take_profit_price, self.stop_loss_price]):
            return None
        
        try:
            if self.direction == TradingDirection.LONG:
                potential_profit = self.take_profit_price - self.entry_price
                potential_loss = self.entry_price - self.stop_loss_price
            else:  # SHORT
                potential_profit = self.entry_price - self.take_profit_price
                potential_loss = self.stop_loss_price - self.entry_price
            
            if potential_loss <= 0:
                return None
            
            return potential_profit / potential_loss
        except Exception as e:
            logger.error(f"计算风险收益比失败: {e}")
            return None
    
    def to_binance_order_request(self, quantity: float = None) -> Dict[str, Any]:
        """
        将信号转换为币安API订单请求格式
        
        Args:
            quantity: 订单数量（可选，默认使用position_size计算）
            
        Returns:
            币安API兼容的订单请求字典
        """
        # 计算订单数量
        if quantity is None and self.position_size and self.current_price:
            quantity = self.position_size / self.current_price
        
        # 确定订单方向
        if self.direction == TradingDirection.LONG:
            side = "BUY"
            position_side = "LONG" if self.operation_type == OperationType.OPEN else "SHORT"
        elif self.direction == TradingDirection.SHORT:
            side = "SELL"
            position_side = "SHORT" if self.operation_type == OperationType.OPEN else "LONG"
        else:
            side = "BUY"  # 中性默认为买入
            position_side = "BOTH"
        
        # 确定订单类型
        order_type = "MARKET"  # 默认市价单
        order_params = {
            "symbol": self.ticker,
            "side": side,
            "type": order_type,
            "positionSide": position_side,
            "timeInForce": "GTC",
        }
        
        if quantity:
            order_params["quantity"] = str(quantity)
        
        # 添加价格信息（如果有）
        if self.entry_price and order_type == "LIMIT":
            order_params["price"] = str(self.entry_price)
        
        # 添加止损止盈（如果有）
        if self.stop_loss_price:
            order_params["stopPrice"] = str(self.stop_loss_price)
        
        # 添加元数据
        order_params["newClientOrderId"] = self.signal_id or f"signal_{int(self.timestamp.timestamp())}"
        
        return order_params
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "ticker": self.ticker,
            "direction": self.direction.value,
            "operation_type": self.operation_type.value,
            "confidence": self.confidence,
            "strength": self.strength,
            "suggested_leverage": self.suggested_leverage,
            "position_size": self.position_size,
            "position_ratio": self.position_ratio,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "take_profit_price": self.take_profit_price,
            "stop_loss_price": self.stop_loss_price,
            "take_profit_ratio": self.take_profit_ratio,
            "stop_loss_ratio": self.stop_loss_ratio,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "expiry_time": self.expiry_time.isoformat() if self.expiry_time else None,
            "volatility": self.volatility,
            "market_trend": self.market_trend,
            "volume_profile": self.volume_profile,
            "indicators": self.indicators,
            "timeframes": self.timeframes,
            "risk_level": self.risk_level.value,
            "max_loss_percentage": self.max_loss_percentage,
            "strategy_source": self.strategy_source,
            "signal_id": self.signal_id,
            "metadata": self.metadata
        }
        
        # 添加币安SDK兼容格式
        if SDK_AVAILABLE:
            try:
                result["binance_order_format"] = self.to_binance_order_request()
            except Exception as e:
                logger.warning(f"生成币安订单格式失败: {e}")
                result["binance_order_format"] = None
        
        return result


@dataclass
class Position:
    """
    仓位信息类
    
    包含完整的持仓信息，包括交易对、方向、大小、
    入场价格、杠杆、保证金等。
    """
    ticker: str                                    # 交易对符号
    side: PositionSide                             # 仓位方向
    size: float                                    # 仓位大小（合约数量）
    entry_price: float                             # 入场价格
    current_price: float                           # 当前价格
    leverage: float                                # 杠杆倍数
    
    # 保证金信息
    initial_margin: float                          # 初始保证金
    maintenance_margin: float                      # 维持保证金
    margin_ratio: Optional[float] = None           # 保证金率
    
    # 盈亏信息
    unrealized_pnl: float = 0.0                    # 未实现盈亏
    realized_pnl: float = 0.0                      # 已实现盈亏
    roe_percentage: float = 0.0                    # 投资回报率百分比
    
    # 风险信息
    liquidation_price: Optional[float] = None      # 强平价格
    bankruptcy_price: Optional[float] = None       # 破产价格
    distance_to_liquidation: Optional[float] = None  # 距离强平的距离（百分比）
    
    # 时间信息
    open_time: datetime = field(default_factory=datetime.now)  # 开仓时间
    update_time: datetime = field(default_factory=datetime.now)  # 更新时间
    
    # 订单信息
    avg_entry_price: Optional[float] = None        # 平均入场价格
    position_value: Optional[float] = None         # 仓位价值
    notional_value: Optional[float] = None         # 名义价值
    
    # 元数据
    position_id: Optional[str] = None              # 仓位ID
    strategy_id: Optional[str] = None              # 策略ID
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后计算"""
        self._calculate_metrics()
        self._validate_position()
    
    def _validate_position(self) -> None:
        """验证仓位参数"""
        if self.size <= 0:
            raise PositionSizeError(
                position_size=self.size,
                issue_type="invalid",
                ticker=self.ticker
            )
        
        if self.entry_price <= 0:
            raise ValueError(f"入场价格必须大于0，当前值: {self.entry_price}")
        
        if self.current_price <= 0:
            raise ValueError(f"当前价格必须大于0，当前值: {self.current_price}")
        
        if self.leverage <= 0:
            raise LeverageExceedsLimitError(
                requested_leverage=self.leverage,
                max_allowed_leverage=1.0,
                ticker=self.ticker,
                reason="invalid_leverage"
            )
    
    def _calculate_metrics(self) -> None:
        """计算仓位指标"""
        try:
            # 计算仓位价值
            self.position_value = self.size * self.current_price
            self.notional_value = self.position_value
            
            # 计算未实现盈亏
            if self.side == PositionSide.LONG:
                self.unrealized_pnl = self.size * (self.current_price - self.entry_price)
            else:  # SHORT
                self.unrealized_pnl = self.size * (self.entry_price - self.current_price)
            
            # 计算投资回报率
            if self.initial_margin > 0:
                self.roe_percentage = (self.unrealized_pnl / self.initial_margin) * 100
            
            # 计算保证金率
            if self.position_value > 0:
                self.margin_ratio = self.maintenance_margin / self.position_value
            
            # 计算强平距离
            if self.liquidation_price:
                if self.side == PositionSide.LONG:
                    self.distance_to_liquidation = ((self.current_price - self.liquidation_price) / self.current_price) * 100
                else:  # SHORT
                    self.distance_to_liquidation = ((self.liquidation_price - self.current_price) / self.current_price) * 100
                    
        except Exception as e:
            logger.error(f"计算仓位指标失败: {e}")
    
    def update_price(self, new_price: float) -> None:
        """更新当前价格并重新计算指标"""
        if new_price <= 0:
            raise ValueError(f"新价格必须大于0，当前值: {new_price}")
        
        self.current_price = new_price
        self.update_time = datetime.now()
        self._calculate_metrics()
    
    def is_at_risk(self, risk_threshold: float = 10.0) -> bool:
        """检查是否存在强平风险"""
        if self.distance_to_liquidation is None:
            return False
        return self.distance_to_liquidation <= risk_threshold
    
    def get_risk_level(self) -> RiskLevel:
        """获取当前风险等级"""
        if self.distance_to_liquidation is None:
            return RiskLevel.LOW
        
        distance = abs(self.distance_to_liquidation)
        if distance <= 2.0:
            return RiskLevel.EMERGENCY
        elif distance <= 5.0:
            return RiskLevel.CRITICAL
        elif distance <= 10.0:
            return RiskLevel.HIGH
        elif distance <= 20.0:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    @classmethod
    def from_binance_position_response(cls, response_data: Any, symbol: str = None) -> 'Position':
        """
        从币安官方SDK仓位响应创建Position实例
        
        Args:
            response_data: 币安SDK的PositionInformationResponse数据
            symbol: 交易对符号（可选，如果response中没有）
            
        Returns:
            Position实例
        """
        if not SDK_AVAILABLE:
            raise ImportError("Binance SDK不可用，无法转换官方SDK响应")
        
        try:
            # 从响应数据中提取字段
            symbol = symbol or getattr(response_data, 'symbol', 'UNKNOWN')
            position_side = getattr(response_data, 'positionSide', 'BOTH')
            position_amt = float(getattr(response_data, 'positionAmt', 0))
            entry_price = float(getattr(response_data, 'entryPrice', 0))
            mark_price = float(getattr(response_data, 'markPrice', entry_price))
            leverage = float(getattr(response_data, 'leverage', 1))
            
            # 转换仓位方向
            if position_side == 'LONG':
                side = PositionSide.LONG
            elif position_side == 'SHORT':
                side = PositionSide.SHORT
            else:
                side = PositionSide.BOTH
            
            # 计算保证金
            notional = abs(position_amt) * mark_price
            initial_margin = notional / leverage if leverage > 0 else 0
            maintenance_margin = float(getattr(response_data, 'isolatedWallet', 0))
            
            # 创建Position实例
            position = cls(
                ticker=symbol,
                side=side,
                size=abs(position_amt),
                entry_price=entry_price,
                current_price=mark_price,
                leverage=leverage,
                initial_margin=initial_margin,
                maintenance_margin=maintenance_margin,
                unrealized_pnl=float(getattr(response_data, 'unRealizedProfit', 0)),
                liquidation_price=float(getattr(response_data, 'liquidationPrice', 0)) or None,
                update_time=datetime.now()
            )
            
            return position
            
        except Exception as e:
            logger.error(f"从币安SDK响应创建Position失败: {e}")
            raise ValueError(f"无效的币安SDK响应数据: {e}")
    
    def to_binance_format(self) -> Dict[str, Any]:
        """
        转换为币安SDK兼容格式
        
        Returns:
            币安SDK兼容的字典格式
        """
        return {
            "symbol": self.ticker,
            "positionSide": self.side.value,
            "positionAmt": str(self.size) if self.side == PositionSide.LONG else str(-self.size),
            "entryPrice": str(self.entry_price),
            "markPrice": str(self.current_price),
            "unRealizedProfit": str(self.unrealized_pnl),
            "leverage": str(self.leverage),
            "isolatedWallet": str(self.initial_margin),
            "liquidationPrice": str(self.liquidation_price) if self.liquidation_price else "0",
            "marginType": "cross",  # 默认全仓模式
            "updateTime": int(self.update_time.timestamp() * 1000) if self.update_time else None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "ticker": self.ticker,
            "side": self.side.value,
            "size": self.size,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "leverage": self.leverage,
            "initial_margin": self.initial_margin,
            "maintenance_margin": self.maintenance_margin,
            "margin_ratio": self.margin_ratio,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "roe_percentage": self.roe_percentage,
            "liquidation_price": self.liquidation_price,
            "bankruptcy_price": self.bankruptcy_price,
            "distance_to_liquidation": self.distance_to_liquidation,
            "open_time": self.open_time.isoformat() if self.open_time else None,
            "update_time": self.update_time.isoformat() if self.update_time else None,
            "avg_entry_price": self.avg_entry_price,
            "position_value": self.position_value,
            "notional_value": self.notional_value,
            "position_id": self.position_id,
            "strategy_id": self.strategy_id,
            "risk_level": self.get_risk_level().value,
            "metadata": self.metadata,
            # 币安SDK兼容字段
            "binance_format": self.to_binance_format() if SDK_AVAILABLE else None
        }


@dataclass
class MarginStatus:
    """
    保证金状态类
    
    包含账户的完整保证金信息，包括总余额、可用保证金、
    已用保证金、保证金率等。
    """
    total_balance: float                           # 总余额 (USDT)
    available_margin: float                        # 可用保证金
    used_margin: float                             # 已用保证金
    margin_ratio: float                            # 保证金率
    
    # 保证金要求
    initial_margin_requirement: float = 0.0       # 初始保证金要求
    maintenance_margin_requirement: float = 0.0   # 维持保证金要求
    
    # 风险指标
    risk_level: RiskLevel = RiskLevel.LOW          # 风险等级
    margin_call_threshold: float = 0.8             # 追加保证金阈值
    liquidation_threshold: float = 0.9             # 强制平仓阈值
    
    # 账户状态
    account_status: str = "normal"                 # 账户状态
    can_trade: bool = True                         # 是否可交易
    can_open_position: bool = True                 # 是否可开仓
    
    # 时间信息
    update_time: datetime = field(default_factory=datetime.now)
    
    # 详细信息
    cross_margin: Optional[float] = None           # 全仓保证金
    isolated_margin: Optional[float] = None        # 逐仓保证金
    position_margin: Optional[float] = None        # 仓位保证金
    order_margin: Optional[float] = None           # 挂单保证金
    
    # 元数据
    currency: str = "USDT"                         # 保证金货币
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后验证和计算"""
        self._validate_margin_status()
        self._calculate_risk_level()
        self._update_trading_permissions()
    
    def _validate_margin_status(self) -> None:
        """验证保证金状态参数"""
        # 验证余额为非负数
        balances = [self.total_balance, self.available_margin, self.used_margin]
        for balance in balances:
            if balance < 0:
                raise ValueError(f"保证金余额不能为负数: {balance}")
        
        # 验证保证金率范围
        if not 0 <= self.margin_ratio <= 1:
            raise ValueError(f"保证金率必须在0-1范围内: {self.margin_ratio}")
        
        # 验证阈值范围
        if not 0 < self.margin_call_threshold < self.liquidation_threshold < 1:
            raise ValueError("保证金阈值设置无效")
    
    def _calculate_risk_level(self) -> None:
        """计算风险等级"""
        if self.margin_ratio >= self.liquidation_threshold:
            self.risk_level = RiskLevel.EMERGENCY
        elif self.margin_ratio >= self.margin_call_threshold:
            self.risk_level = RiskLevel.CRITICAL
        elif self.margin_ratio >= 0.7:
            self.risk_level = RiskLevel.HIGH
        elif self.margin_ratio >= 0.5:
            self.risk_level = RiskLevel.MEDIUM
        else:
            self.risk_level = RiskLevel.LOW
    
    def _update_trading_permissions(self) -> None:
        """更新交易权限"""
        # 基于风险等级设置交易权限
        if self.risk_level in [RiskLevel.EMERGENCY, RiskLevel.CRITICAL]:
            self.can_trade = False
            self.can_open_position = False
            self.account_status = "margin_call"
        elif self.risk_level == RiskLevel.HIGH:
            self.can_trade = True
            self.can_open_position = False
            self.account_status = "restricted"
        else:
            self.can_trade = True
            self.can_open_position = True
            self.account_status = "normal"
    
    def get_max_position_size(self, leverage: float, price: float) -> float:
        """计算给定杠杆和价格下的最大仓位大小"""
        if not self.can_open_position or leverage <= 0 or price <= 0:
            return 0.0
        
        # 保留一定的缓冲保证金
        usable_margin = self.available_margin * 0.95
        max_notional = usable_margin * leverage
        max_size = max_notional / price
        
        return max_size
    
    def check_margin_sufficiency(
        self,
        required_margin: float,
        ticker: str = "UNKNOWN",
        position_size: float = 0.0,
        leverage: float = 1.0
    ) -> bool:
        """检查保证金是否充足"""
        if required_margin <= self.available_margin:
            return True
        
        # 如果保证金不足，抛出异常
        raise MarginInsufficientError(
            required_margin=required_margin,
            available_margin=self.available_margin,
            ticker=ticker,
            position_size=position_size,
            leverage=leverage
        )
    
    @classmethod
    def from_binance_account_response(cls, response_data: Any) -> 'MarginStatus':
        """
        从币安官方SDK账户响应创建MarginStatus实例
        
        Args:
            response_data: 币安SDK的AccountInformationResponse数据
            
        Returns:
            MarginStatus实例
        """
        if not SDK_AVAILABLE:
            raise ImportError("Binance SDK不可用，无法转换官方SDK响应")
        
        try:
            # 提取账户信息
            total_wallet_balance = float(getattr(response_data, 'totalWalletBalance', 0))
            total_unrealized_profit = float(getattr(response_data, 'totalUnrealizedProfit', 0))
            total_margin_balance = float(getattr(response_data, 'totalMarginBalance', 0))
            total_initial_margin = float(getattr(response_data, 'totalInitialMargin', 0))
            total_maint_margin = float(getattr(response_data, 'totalMaintMargin', 0))
            
            # 计算派生值
            total_balance = total_wallet_balance + total_unrealized_profit
            available_margin = total_margin_balance - total_initial_margin
            used_margin = total_initial_margin
            margin_ratio = total_maint_margin / total_margin_balance if total_margin_balance > 0 else 0
            
            return cls(
                total_balance=total_balance,
                available_margin=max(0, available_margin),  # 确保非负
                used_margin=used_margin,
                margin_ratio=margin_ratio,
                initial_margin_requirement=total_initial_margin,
                maintenance_margin_requirement=total_maint_margin,
                cross_margin=total_margin_balance,
                update_time=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"从币安SDK账户响应创建MarginStatus失败: {e}")
            raise ValueError(f"无效的币安SDK账户数据: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "total_balance": self.total_balance,
            "available_margin": self.available_margin,
            "used_margin": self.used_margin,
            "margin_ratio": self.margin_ratio,
            "initial_margin_requirement": self.initial_margin_requirement,
            "maintenance_margin_requirement": self.maintenance_margin_requirement,
            "risk_level": self.risk_level.value,
            "margin_call_threshold": self.margin_call_threshold,
            "liquidation_threshold": self.liquidation_threshold,
            "account_status": self.account_status,
            "can_trade": self.can_trade,
            "can_open_position": self.can_open_position,
            "update_time": self.update_time.isoformat(),
            "cross_margin": self.cross_margin,
            "isolated_margin": self.isolated_margin,
            "position_margin": self.position_margin,
            "order_margin": self.order_margin,
            "currency": self.currency,
            "metadata": self.metadata
        }


@dataclass
class ValidationResult:
    """
    验证结果类
    
    包含验证状态、错误信息、警告信息等完整的验证结果。
    此类与现有的BaseValidator中的ValidationResult保持兼容。
    """
    is_valid: bool                                 # 是否验证通过
    severity: ValidationSeverity                   # 问题严重程度
    field_name: str                                # 字段名称
    message: str                                   # 验证消息
    
    # 详细信息
    current_value: Any = None                      # 当前值
    expected_range: Optional[Dict[str, Any]] = None  # 期望范围
    suggestion: Optional[str] = None               # 修正建议
    corrected_value: Any = None                    # 建议修正值
    context: Optional[Dict[str, Any]] = None       # 上下文信息
    
    # 错误分类
    error_code: Optional[str] = None               # 错误代码
    error_category: Optional[str] = None           # 错误分类
    
    # 时间信息
    validation_time: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"[{self.severity.value.upper()}] {self.field_name}: {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "is_valid": self.is_valid,
            "severity": self.severity.value,
            "field_name": self.field_name,
            "message": self.message,
            "current_value": self.current_value,
            "expected_range": self.expected_range,
            "suggestion": self.suggestion,
            "corrected_value": self.corrected_value,
            "context": self.context,
            "error_code": self.error_code,
            "error_category": self.error_category,
            "validation_time": self.validation_time.isoformat()
        }
    
    def is_blocking(self) -> bool:
        """判断是否为阻塞性错误"""
        return self.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
    
    def is_warning(self) -> bool:
        """判断是否为警告"""
        return self.severity == ValidationSeverity.WARNING
    
    def is_info(self) -> bool:
        """判断是否为信息"""
        return self.severity == ValidationSeverity.INFO


@dataclass 
class FuturesOrderRequest:
    """
    期货订单请求类
    
    包含创建期货订单所需的完整信息
    """
    ticker: str                                    # 交易对符号
    side: Literal["BUY", "SELL"]                  # 订单方向
    order_type: Literal["MARKET", "LIMIT", "STOP", "TAKE_PROFIT"] = "MARKET"  # 订单类型
    quantity: float = 0.0                          # 数量
    price: Optional[float] = None                  # 价格（限价单使用）
    position_side: PositionSide = PositionSide.BOTH  # 仓位方向
    
    # 杠杆和保证金
    leverage: float = 1.0                          # 杠杆倍数
    margin_type: Literal["cross", "isolated"] = "cross"  # 保证金模式
    
    # 止盈止损
    take_profit_price: Optional[float] = None      # 止盈价格
    stop_loss_price: Optional[float] = None        # 止损价格
    
    # 订单选项
    time_in_force: Literal["GTC", "IOC", "FOK"] = "GTC"  # 有效期类型
    reduce_only: bool = False                      # 只减仓
    close_position: bool = False                   # 平仓
    
    # 客户端信息
    client_order_id: Optional[str] = None          # 客户端订单ID
    
    def __post_init__(self):
        """初始化后验证"""
        self._validate_order_request()
    
    def _validate_order_request(self) -> None:
        """验证订单请求参数"""
        if self.quantity <= 0:
            raise ValueError(f"订单数量必须大于0: {self.quantity}")
        
        if self.order_type == "LIMIT" and (self.price is None or self.price <= 0):
            raise ValueError("限价单必须设置有效价格")
        
        if self.leverage <= 0:
            raise ValueError(f"杠杆倍数必须大于0: {self.leverage}")
    
    @classmethod
    def from_futures_signal(cls, signal: 'FuturesSignal', quantity: float = None) -> 'FuturesOrderRequest':
        """
        从FuturesSignal创建FuturesOrderRequest
        
        Args:
            signal: 期货交易信号
            quantity: 订单数量（可选）
            
        Returns:
            FuturesOrderRequest实例
        """
        # 确定订单方向
        if signal.direction == TradingDirection.LONG:
            side = "BUY" if signal.operation_type == OperationType.OPEN else "SELL"
            position_side = PositionSide.LONG
        elif signal.direction == TradingDirection.SHORT:
            side = "SELL" if signal.operation_type == OperationType.OPEN else "BUY"
            position_side = PositionSide.SHORT
        else:
            side = "BUY"
            position_side = PositionSide.BOTH
        
        # 计算订单数量
        if quantity is None and signal.position_size and signal.current_price:
            quantity = signal.position_size / signal.current_price
        elif quantity is None:
            quantity = 0.001  # 默认最小数量
        
        return cls(
            ticker=signal.ticker,
            side=side,
            quantity=quantity,
            price=signal.entry_price,
            position_side=position_side,
            leverage=signal.suggested_leverage,
            take_profit_price=signal.take_profit_price,
            stop_loss_price=signal.stop_loss_price,
            client_order_id=signal.signal_id
        )
    
    def to_binance_api_format(self) -> Dict[str, Any]:
        """
        转换为币安API标准格式
        
        Returns:
            币安API兼容的订单请求
        """
        order_data = {
            "symbol": self.ticker,
            "side": self.side,
            "type": self.order_type,
            "quantity": str(self.quantity),
            "positionSide": self.position_side.value,
            "timeInForce": self.time_in_force,
            "reduceOnly": str(self.reduce_only).lower(),  # 布尔值转字符串
            "closePosition": str(self.close_position).lower()
        }
        
        # 添加可选参数
        if self.price is not None:
            order_data["price"] = str(self.price)
        
        if self.client_order_id:
            order_data["newClientOrderId"] = self.client_order_id
        
        # 添加止损止盈参数
        if self.stop_loss_price:
            order_data["stopPrice"] = str(self.stop_loss_price)
            
        if self.take_profit_price:
            order_data["takeProfitPrice"] = str(self.take_profit_price)
        
        return order_data
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（包含币安API格式）"""
        result = {
            "ticker": self.ticker,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": self.quantity,
            "price": self.price,
            "position_side": self.position_side.value,
            "leverage": self.leverage,
            "margin_type": self.margin_type,
            "take_profit_price": self.take_profit_price,
            "stop_loss_price": self.stop_loss_price,
            "time_in_force": self.time_in_force,
            "reduce_only": self.reduce_only,
            "close_position": self.close_position,
            "client_order_id": self.client_order_id
        }
        
        # 添加币安API兼容格式
        if SDK_AVAILABLE:
            result["binance_api_format"] = self.to_binance_api_format()
        
        return result


# ============================================================================
# Binance SDK兼容性工具类
# ============================================================================

class BinanceSDKCompatibility:
    """
    币安官方SDK兼容性工具类
    
    提供统一的转换接口，用于在现有数据类和官方SDK数据模型之间转换。
    """
    
    @staticmethod
    def is_sdk_available() -> bool:
        """检查SDK是否可用"""
        return SDK_AVAILABLE
    
    @staticmethod
    def create_position_from_response(response_data: Any, symbol: str = None) -> Position:
        """
        从币安SDK响应创建Position实例
        
        Args:
            response_data: SDK响应数据
            symbol: 交易对符号
            
        Returns:
            Position实例
        """
        return Position.from_binance_position_response(response_data, symbol)
    
    @staticmethod
    def create_margin_status_from_response(response_data: Any) -> MarginStatus:
        """
        从币安SDK账户响应创建MarginStatus实例
        
        Args:
            response_data: SDK账户响应数据
            
        Returns:
            MarginStatus实例
        """
        return MarginStatus.from_binance_account_response(response_data)
    
    @staticmethod
    def create_order_from_signal(signal: FuturesSignal, quantity: float = None) -> FuturesOrderRequest:
        """
        从交易信号创建订单请求
        
        Args:
            signal: 期货交易信号
            quantity: 订单数量（可选）
            
        Returns:
            FuturesOrderRequest实例
        """
        return FuturesOrderRequest.from_futures_signal(signal, quantity)
    
    @staticmethod
    def get_sdk_configuration() -> Dict[str, Any]:
        """
        获取SDK推荐配置
        
        Returns:
            SDK配置字典
        """
        if not SDK_AVAILABLE:
            return {}
        
        return {
            "environment": "testnet",  # 默认使用测试网
            "timeout": 5000,
            "retries": 3,
            "backoff": 1000,
            "compression": True,
            "keep_alive": True
        }
    
    @staticmethod
    def validate_sdk_response(response_data: Any, expected_fields: List[str]) -> bool:
        """
        验证SDK响应数据完整性
        
        Args:
            response_data: SDK响应数据
            expected_fields: 期望的字段列表
            
        Returns:
            验证是否通过
        """
        if not response_data:
            return False
        
        try:
            for field in expected_fields:
                if not hasattr(response_data, field):
                    logger.warning(f"SDK响应缺少必需字段: {field}")
                    return False
            return True
        except Exception as e:
            logger.error(f"验证SDK响应时发生错误: {e}")
            return False


# ============================================================================
# 工厂函数和兼容性助手
# ============================================================================

def create_position_from_binance_data(data: Any, symbol: str = None) -> Position:
    """
    工厂函数：从币安数据创建Position实例
    
    Args:
        data: 币安API/SDK数据
        symbol: 交易对符号
        
    Returns:
        Position实例
    """
    if SDK_AVAILABLE:
        return Position.from_binance_position_response(data, symbol)
    else:
        raise ImportError("Binance SDK不可用，无法从SDK数据创建Position")


def create_margin_status_from_binance_data(data: Any) -> MarginStatus:
    """
    工厂函数：从币安数据创建MarginStatus实例
    
    Args:
        data: 币安账户API/SDK数据
        
    Returns:
        MarginStatus实例
    """
    if SDK_AVAILABLE:
        return MarginStatus.from_binance_account_response(data)
    else:
        raise ImportError("Binance SDK不可用，无法从SDK数据创建MarginStatus")


def convert_signal_to_binance_order(signal: FuturesSignal, quantity: float = None) -> Dict[str, Any]:
    """
    将交易信号转换为币安API订单格式
    
    Args:
        signal: 期货交易信号
        quantity: 订单数量（可选）
        
    Returns:
        币安API格式的订单字典
    """
    order_request = FuturesOrderRequest.from_futures_signal(signal, quantity)
    return order_request.to_binance_api_format()


def get_recommended_sdk_settings() -> Dict[str, Any]:
    """
    获取推荐的SDK设置
    
    Returns:
        推荐的SDK配置
    """
    return BinanceSDKCompatibility.get_sdk_configuration()


# 导出新的SDK兼容性相关内容
__all__ = [
    # 现有导出
    'TradingDirection', 'OperationType', 'PositionSide', 'RiskLevel', 'ValidationSeverity',
    'FuturesSignal', 'Position', 'MarginStatus', 'ValidationResult', 'FuturesOrderRequest',
    
    # 新增SDK兼容性导出
    'BinanceSDKCompatibility', 'SDK_AVAILABLE',
    'create_position_from_binance_data', 'create_margin_status_from_binance_data',
    'convert_signal_to_binance_order', 'get_recommended_sdk_settings'
]
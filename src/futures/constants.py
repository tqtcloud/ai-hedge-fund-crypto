"""
期货交易常量定义模块

定义期货交易系统中使用的所有常量，包括杠杆限制、
保证金率、风险控制参数、交易对配置等。
已集成币安官方SDK配置和常量。
"""
from enum import Enum
from typing import Dict, Any, List, TYPE_CHECKING
from decimal import Decimal

# 集成币安官方SDK常量
if TYPE_CHECKING:
    try:
        from binance_common.constants import (
            DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
            DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL,
            DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_PROD_URL,
            DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_TESTNET_URL
        )
    except ImportError:
        # 如果SDK不可用，定义默认值
        DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL = "https://fapi.binance.com"
        DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL = "https://testnet.binancefuture.com"
        DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_PROD_URL = "wss://fstream.binance.com/ws"
        DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_TESTNET_URL = "wss://stream.binancefuture.com/ws"
else:
    try:
        from binance_common.constants import (
            DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
            DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL,
            DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_PROD_URL,
            DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_TESTNET_URL
        )
    except ImportError:
        # 使用默认的币安API端点
        DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL = "https://fapi.binance.com"
        DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL = "https://testnet.binancefuture.com"
        DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_PROD_URL = "wss://fstream.binance.com/ws"
        DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_TESTNET_URL = "wss://stream.binancefuture.com/ws"

# ============================================================================
# 杠杆限制常量
# ============================================================================

class LeverageLimits:
    """杠杆限制常量类"""
    
    # 全局杠杆限制
    MIN_LEVERAGE = 1.0                    # 最小杠杆倍数
    MAX_LEVERAGE_GLOBAL = 125.0           # 全局最大杠杆倍数
    DEFAULT_LEVERAGE = 10.0               # 默认杠杆倍数
    
    # 基于风险的杠杆限制
    MAX_LEVERAGE_LOW_RISK = 75.0          # 低风险最大杠杆
    MAX_LEVERAGE_MEDIUM_RISK = 50.0       # 中等风险最大杠杆
    MAX_LEVERAGE_HIGH_RISK = 20.0         # 高风险最大杠杆
    MAX_LEVERAGE_CRITICAL_RISK = 10.0     # 严重风险最大杠杆
    
    # 新手保护
    MAX_LEVERAGE_NEW_USER = 20.0          # 新用户最大杠杆
    MAX_LEVERAGE_SMALL_ACCOUNT = 50.0     # 小资金账户最大杠杆
    
    # 波动率相关限制
    HIGH_VOLATILITY_MAX_LEVERAGE = 10.0   # 高波动率环境最大杠杆
    EXTREME_VOLATILITY_MAX_LEVERAGE = 5.0 # 极端波动率最大杠杆


# 交易对特定杠杆限制
SYMBOL_LEVERAGE_LIMITS: Dict[str, Dict[str, float]] = {
    "BTCUSDT": {
        "max_leverage": 125.0,
        "recommended_leverage": 20.0,
        "high_risk_leverage": 50.0
    },
    "ETHUSDT": {
        "max_leverage": 100.0,
        "recommended_leverage": 20.0,
        "high_risk_leverage": 50.0
    },
    "ADAUSDT": {
        "max_leverage": 75.0,
        "recommended_leverage": 15.0,
        "high_risk_leverage": 30.0
    },
    "DOTUSDT": {
        "max_leverage": 75.0,
        "recommended_leverage": 15.0,
        "high_risk_leverage": 30.0
    },
    "LINKUSDT": {
        "max_leverage": 75.0,
        "recommended_leverage": 15.0,
        "high_risk_leverage": 30.0
    },
    # 小市值币种限制更严格
    "DEFAULT_ALTCOIN": {
        "max_leverage": 50.0,
        "recommended_leverage": 10.0,
        "high_risk_leverage": 20.0
    }
}


# ============================================================================
# 保证金率常量
# ============================================================================

class MarginRates:
    """保证金率常量类"""
    
    # 初始保证金率 (1/leverage)
    INITIAL_MARGIN_RATE_1X = 1.0          # 1倍杠杆
    INITIAL_MARGIN_RATE_5X = 0.2          # 5倍杠杆
    INITIAL_MARGIN_RATE_10X = 0.1         # 10倍杠杆
    INITIAL_MARGIN_RATE_20X = 0.05        # 20倍杠杆
    INITIAL_MARGIN_RATE_50X = 0.02        # 50倍杠杆
    INITIAL_MARGIN_RATE_100X = 0.01       # 100倍杠杆
    INITIAL_MARGIN_RATE_125X = 0.008      # 125倍杠杆
    
    # 维持保证金率
    MAINTENANCE_MARGIN_RATE_BTC = 0.004   # BTC维持保证金率 (0.4%)
    MAINTENANCE_MARGIN_RATE_ETH = 0.005   # ETH维持保证金率 (0.5%)
    MAINTENANCE_MARGIN_RATE_ALTCOIN = 0.01  # 其他币种维持保证金率 (1.0%)
    
    # 保证金等级阶梯
    MARGIN_BRACKETS: Dict[str, List[Dict[str, Any]]] = {
        "BTCUSDT": [
            {"notional": 0, "mmr": 0.004, "cum": 0},
            {"notional": 50000, "mmr": 0.005, "cum": 50},
            {"notional": 250000, "mmr": 0.01, "cum": 1300},
            {"notional": 1000000, "mmr": 0.025, "cum": 16300},
            {"notional": 5000000, "mmr": 0.05, "cum": 141300},
            {"notional": 20000000, "mmr": 0.1, "cum": 1141300},
        ],
        "ETHUSDT": [
            {"notional": 0, "mmr": 0.005, "cum": 0},
            {"notional": 10000, "mmr": 0.0065, "cum": 15},
            {"notional": 100000, "mmr": 0.01, "cum": 350},
            {"notional": 500000, "mmr": 0.02, "cum": 5350},
            {"notional": 1000000, "mmr": 0.05, "cum": 35350},
            {"notional": 2000000, "mmr": 0.1, "cum": 135350},
        ]
    }


# ============================================================================
# 风险控制参数
# ============================================================================

class RiskParameters:
    """风险控制参数常量类"""

    # 杠杆限制
    MAX_LEVERAGE = 20.0                   # 系统最大杠杆限制（符合交易所要求）
    DEFAULT_LEVERAGE = 5.0                # 默认杠杆
    CONSERVATIVE_LEVERAGE = 3.0           # 保守杠杆

    # 仓位限制
    MAX_POSITION_RATIO = 0.3              # 最大单仓位占总资金比例
    MAX_CORRELATED_POSITIONS_RATIO = 0.5  # 最大相关性仓位占比
    MAX_SECTOR_EXPOSURE = 0.4             # 最大行业敞口
    
    # 止损参数
    DEFAULT_STOP_LOSS_RATIO = 0.02        # 默认止损比例 (2%)
    MAX_STOP_LOSS_RATIO = 0.1             # 最大止损比例 (10%)
    MIN_STOP_LOSS_RATIO = 0.005           # 最小止损比例 (0.5%)
    
    # 止盈参数
    DEFAULT_TAKE_PROFIT_RATIO = 0.06      # 默认止盈比例 (6%)
    MAX_TAKE_PROFIT_RATIO = 0.5           # 最大止盈比例 (50%)
    MIN_TAKE_PROFIT_RATIO = 0.01          # 最小止盈比例 (1%)
    
    # 风险收益比
    MIN_RISK_REWARD_RATIO = 1.5           # 最小风险收益比
    RECOMMENDED_RISK_REWARD_RATIO = 2.0   # 推荐风险收益比
    
    # 强平风险阈值
    LIQUIDATION_WARNING_DISTANCE = 15.0   # 强平警告距离 (15%)
    LIQUIDATION_CRITICAL_DISTANCE = 10.0  # 强平严重警告距离 (10%)
    LIQUIDATION_EMERGENCY_DISTANCE = 5.0  # 强平紧急距离 (5%)
    
    # 保证金使用率阈值
    MARGIN_USAGE_WARNING = 0.7            # 保证金使用率警告阈值
    MARGIN_USAGE_CRITICAL = 0.85          # 保证金使用率严重阈值
    MARGIN_USAGE_MAX = 0.95               # 保证金使用率最大值
    
    # 回撤控制
    MAX_DRAWDOWN_DAILY = 0.05             # 日最大回撤 (5%)
    MAX_DRAWDOWN_WEEKLY = 0.15            # 周最大回撤 (15%)
    MAX_DRAWDOWN_MONTHLY = 0.25           # 月最大回撤 (25%)
    
    # VAR (Value at Risk) 限制
    VAR_95_DAILY_LIMIT = 0.02             # 95%置信度日VAR限制 (2%)
    VAR_99_DAILY_LIMIT = 0.03             # 99%置信度日VAR限制 (3%)
    
    # 相关性限制
    MAX_POSITION_CORRELATION = 0.7        # 最大仓位相关性
    CORRELATION_CHECK_WINDOW = 30         # 相关性检查窗口 (天)


# ============================================================================
# 交易对配置
# ============================================================================

class TradingPairConfig:
    """交易对配置常量类"""
    
    # 主流币种配置
    MAJOR_PAIRS = {
        "BTCUSDT": {
            "min_qty": 0.001,
            "max_qty": 1000.0,
            "qty_precision": 3,
            "price_precision": 2,
            "tick_size": 0.01,
            "min_notional": 5.0,
            "max_leverage": 125,
            "maintenance_margin_rate": 0.004,
            "maker_fee": 0.0002,
            "taker_fee": 0.0004,
            "category": "major"
        },
        "ETHUSDT": {
            "min_qty": 0.001,
            "max_qty": 10000.0,
            "qty_precision": 3,
            "price_precision": 2,
            "tick_size": 0.01,
            "min_notional": 5.0,
            "max_leverage": 100,
            "maintenance_margin_rate": 0.005,
            "maker_fee": 0.0002,
            "taker_fee": 0.0004,
            "category": "major"
        }
    }
    
    # 其他山寨币默认配置
    DEFAULT_ALTCOIN_CONFIG = {
        "min_qty": 1.0,
        "max_qty": 1000000.0,
        "qty_precision": 0,
        "price_precision": 4,
        "tick_size": 0.0001,
        "min_notional": 5.0,
        "max_leverage": 50,
        "maintenance_margin_rate": 0.01,
        "maker_fee": 0.0002,
        "taker_fee": 0.0004,
        "category": "altcoin"
    }
    
    # 支持的交易对列表
    SUPPORTED_PAIRS = [
        "BTCUSDT", "ETHUSDT", "BCHUSDT", "XRPUSDT", "EOSUSDT",
        "LTCUSDT", "TRXUSDT", "ETCUSDT", "LINKUSDT", "XLMUSDT",
        "ADAUSDT", "XMRUSDT", "DASHUSDT", "ZECUSDT", "XTZUSDT",
        "BNBUSDT", "ATOMUSDT", "ONTUSDT", "IOTAUSDT", "BATUSDT",
        "VETUSDT", "NEOUSDT", "QTUMUSDT", "IOSTUSDT", "THETAUSDT",
        "ALGOUSDT", "ZILUSDT", "KNCUSDT", "ZRXUSDT", "COMPUSDT",
        "OMGUSDT", "DOGEUSDT", "SXPUSDT", "KAVAUSDT", "BANDUSDT",
        "RLCUSDT", "WAVESUSDT", "MKRUSDT", "SNXUSDT", "DOTUSDT",
        "DEFIUSDT", "YFIUSDT", "BALUSDT", "CRVUSDT", "SANDUSDT",
        "UNIUSDT", "AVAXUSDT", "1INCHUSDT", "CHZUSDT", "SUSHIUSDT"
    ]


# ============================================================================
# 环境相关配置
# ============================================================================

class EnvironmentConfig:
    """环境相关配置常量类（集成官方SDK配置）"""
    
    # Testnet配置（使用官方SDK常量）
    TESTNET_CONFIG = {
        "base_url": DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL,
        "api_endpoint": "/fapi/v1",
        "ws_endpoint": DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_TESTNET_URL,
        "max_leverage_limit": 20.0,         # Testnet杠杆限制更严格
        "initial_balance": 10000.0,         # 测试网初始余额
        "rate_limit": {
            "requests_per_minute": 1200,
            "orders_per_second": 10,
            "orders_per_day": 100000
        },
        # 官方SDK配置参数
        "sdk_config": {
            "timeout": 5000,
            "retries": 3,
            "backoff": 1000,
            "compression": True,
            "keep_alive": True
        }
    }
    
    # Mainnet配置（使用官方SDK常量）
    MAINNET_CONFIG = {
        "base_url": DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
        "api_endpoint": "/fapi/v1", 
        "ws_endpoint": DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_PROD_URL,
        "max_leverage_limit": 125.0,
        "rate_limit": {
            "requests_per_minute": 1200,
            "orders_per_second": 10,
            "orders_per_day": 200000
        },
        # 官方SDK配置参数
        "sdk_config": {
            "timeout": 10000,
            "retries": 3,
            "backoff": 1500,
            "compression": True,
            "keep_alive": True
        }
    }
    
    # API限制
    RATE_LIMITS = {
        "weight_per_minute": 1200,          # 每分钟权重限制
        "orders_per_second": 10,            # 每秒订单数限制
        "orders_per_day": 200000,           # 每日订单数限制
        "raw_requests_per_5min": 6100       # 每5分钟原始请求数限制
    }
    
    # 官方SDK WebSocket配置
    WEBSOCKET_CONFIG = {
        "testnet": {
            "streams_url": DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_TESTNET_URL,
            "api_url": DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_TESTNET_URL.replace('/ws', '/ws-api/v1'),
            "compression": True,
            "keep_alive_interval": 30,
            "reconnect_delay": 5000,
            "max_reconnect_attempts": 5
        },
        "mainnet": {
            "streams_url": DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_PROD_URL,
            "api_url": DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_PROD_URL.replace('/ws', '/ws-api/v1'),
            "compression": True,
            "keep_alive_interval": 30,
            "reconnect_delay": 5000,
            "max_reconnect_attempts": 10
        }
    }


# ============================================================================
# 交易时间相关常量
# ============================================================================

class TradingTimeConstants:
    """交易时间相关常量类"""
    
    # 信号有效期 (秒)
    SIGNAL_EXPIRY_SHORT_TERM = 300        # 短期信号有效期 (5分钟)
    SIGNAL_EXPIRY_MEDIUM_TERM = 1800      # 中期信号有效期 (30分钟)  
    SIGNAL_EXPIRY_LONG_TERM = 3600        # 长期信号有效期 (1小时)
    
    # 订单超时时间 (秒)
    ORDER_TIMEOUT_MARKET = 30             # 市价单超时
    ORDER_TIMEOUT_LIMIT = 300             # 限价单超时
    ORDER_TIMEOUT_STOP = 600              # 止损单超时
    
    # 数据刷新间隔 (秒)
    PRICE_UPDATE_INTERVAL = 1             # 价格更新间隔
    POSITION_UPDATE_INTERVAL = 5          # 仓位更新间隔
    BALANCE_UPDATE_INTERVAL = 10          # 余额更新间隔
    RISK_CHECK_INTERVAL = 15              # 风险检查间隔
    
    # 市场开放时间 (UTC)
    MARKET_OPEN_HOUR = 0                  # 市场开放小时 (24小时制)
    MARKET_CLOSE_HOUR = 24                # 市场关闭小时 (实际为连续交易)
    
    # 维护时间窗口
    MAINTENANCE_START_HOUR = 8            # 维护开始时间 (UTC)
    MAINTENANCE_DURATION_MINUTES = 30     # 维护持续时间


# ============================================================================
# 费用相关常量
# ============================================================================

class FeeConstants:
    """费用相关常量类"""
    
    # 标准交易费率
    MAKER_FEE_RATE = 0.0002              # 挂单手续费率 (0.02%)
    TAKER_FEE_RATE = 0.0004              # 吃单手续费率 (0.04%)
    
    # VIP等级费率 (基于30天交易量和BNB持有量)
    VIP_FEE_RATES = {
        "VIP0": {"maker": 0.0002, "taker": 0.0004},
        "VIP1": {"maker": 0.0002, "taker": 0.0004},
        "VIP2": {"maker": 0.0002, "taker": 0.0003},
        "VIP3": {"maker": 0.0002, "taker": 0.0003},
        "VIP4": {"maker": 0.0002, "taker": 0.0003},
        "VIP5": {"maker": 0.0001, "taker": 0.0002},
        "VIP6": {"maker": 0.0001, "taker": 0.0002},
        "VIP7": {"maker": 0.0001, "taker": 0.0002},
        "VIP8": {"maker": 0.0001, "taker": 0.0002},
        "VIP9": {"maker": 0.0000, "taker": 0.0002}
    }
    
    # 资金费率相关
    FUNDING_RATE_INTERVAL_HOURS = 8       # 资金费率收取间隔
    MAX_FUNDING_RATE = 0.0075            # 最大资金费率 (0.75%)
    MIN_FUNDING_RATE = -0.0075           # 最小资金费率 (-0.75%)
    
    # 提现费率
    WITHDRAWAL_FEES = {
        "USDT": 1.0,                      # USDT提现手续费
        "BTC": 0.0005,                    # BTC提现手续费
        "ETH": 0.005,                     # ETH提现手续费
        "BNB": 0.0005                     # BNB提现手续费
    }


# ============================================================================
# 精度和数量相关常量
# ============================================================================

class PrecisionConstants:
    """精度和数量相关常量类"""
    
    # 价格精度
    PRICE_PRECISION_HIGH = 8              # 高精度价格小数位
    PRICE_PRECISION_MEDIUM = 4            # 中精度价格小数位
    PRICE_PRECISION_LOW = 2               # 低精度价格小数位
    
    # 数量精度
    QUANTITY_PRECISION_HIGH = 6           # 高精度数量小数位
    QUANTITY_PRECISION_MEDIUM = 3         # 中精度数量小数位
    QUANTITY_PRECISION_LOW = 0            # 低精度数量小数位
    
    # 最小变动单位
    MIN_TICK_SIZE = Decimal('0.0001')     # 最小价格变动
    MIN_STEP_SIZE = Decimal('0.001')      # 最小数量变动
    
    # 数值范围
    MIN_NOTIONAL_VALUE = 5.0              # 最小名义价值 (USDT)
    MAX_NOTIONAL_VALUE = 1000000.0        # 最大名义价值 (USDT)
    
    # 百分比精度
    PERCENTAGE_PRECISION = 4              # 百分比小数位精度
    RATIO_PRECISION = 6                   # 比率小数位精度


# ============================================================================
# 错误代码和消息
# ============================================================================

class ErrorCodes:
    """错误代码常量类"""
    
    # 系统错误
    SYSTEM_ERROR = "SYSTEM_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    API_ERROR = "API_ERROR"
    
    # 验证错误
    INVALID_SYMBOL = "INVALID_SYMBOL"
    INVALID_QUANTITY = "INVALID_QUANTITY"
    INVALID_PRICE = "INVALID_PRICE"
    INVALID_LEVERAGE = "INVALID_LEVERAGE"
    
    # 余额相关错误
    INSUFFICIENT_BALANCE = "INSUFFICIENT_BALANCE"
    INSUFFICIENT_MARGIN = "INSUFFICIENT_MARGIN"
    
    # 订单相关错误
    ORDER_REJECTED = "ORDER_REJECTED"
    ORDER_EXPIRED = "ORDER_EXPIRED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    
    # 风险控制错误
    POSITION_SIZE_EXCEEDED = "POSITION_SIZE_EXCEEDED"
    LEVERAGE_EXCEEDED = "LEVERAGE_EXCEEDED"
    RISK_LIMIT_EXCEEDED = "RISK_LIMIT_EXCEEDED"
    LIQUIDATION_RISK = "LIQUIDATION_RISK"
    
    # 市场相关错误
    MARKET_CLOSED = "MARKET_CLOSED"
    SYMBOL_NOT_TRADING = "SYMBOL_NOT_TRADING"
    PRICE_OUT_OF_RANGE = "PRICE_OUT_OF_RANGE"


# ============================================================================
# 默认配置聚合
# ============================================================================

class DefaultConfig:
    """默认配置聚合类"""
    
    # 交易默认设置
    DEFAULT_TRADING_CONFIG = {
        "leverage": LeverageLimits.DEFAULT_LEVERAGE,
        "stop_loss_ratio": RiskParameters.DEFAULT_STOP_LOSS_RATIO,
        "take_profit_ratio": RiskParameters.DEFAULT_TAKE_PROFIT_RATIO,
        "max_position_ratio": RiskParameters.MAX_POSITION_RATIO,
        "margin_usage_limit": RiskParameters.MARGIN_USAGE_MAX,
        "risk_reward_ratio": RiskParameters.RECOMMENDED_RISK_REWARD_RATIO
    }
    
    # 风险管理默认设置  
    DEFAULT_RISK_CONFIG = {
        "max_drawdown": RiskParameters.MAX_DRAWDOWN_DAILY,
        "var_limit": RiskParameters.VAR_95_DAILY_LIMIT,
        "correlation_limit": RiskParameters.MAX_POSITION_CORRELATION,
        "liquidation_warning": RiskParameters.LIQUIDATION_WARNING_DISTANCE,
        "margin_warning": RiskParameters.MARGIN_USAGE_WARNING
    }
    
    # 系统默认设置
    DEFAULT_SYSTEM_CONFIG = {
        "environment": "testnet",
        "price_precision": PrecisionConstants.PRICE_PRECISION_MEDIUM,
        "quantity_precision": PrecisionConstants.QUANTITY_PRECISION_MEDIUM,
        "signal_expiry": TradingTimeConstants.SIGNAL_EXPIRY_MEDIUM_TERM,
        "update_interval": TradingTimeConstants.POSITION_UPDATE_INTERVAL
    }


# ============================================================================
# 辅助函数
# ============================================================================

def get_symbol_config(symbol: str) -> Dict[str, Any]:
    """
    获取指定交易对的配置信息
    
    Args:
        symbol: 交易对符号
        
    Returns:
        交易对配置字典
    """
    if symbol in TradingPairConfig.MAJOR_PAIRS:
        return TradingPairConfig.MAJOR_PAIRS[symbol]
    else:
        # 返回山寨币默认配置
        config = TradingPairConfig.DEFAULT_ALTCOIN_CONFIG.copy()
        config["symbol"] = symbol
        return config


def get_leverage_limit(symbol: str, risk_level: str = "medium") -> float:
    """
    获取指定交易对和风险等级的杠杆限制
    
    Args:
        symbol: 交易对符号
        risk_level: 风险等级 (low, medium, high, critical)
        
    Returns:
        最大杠杆倍数
    """
    # 获取交易对特定限制
    if symbol in SYMBOL_LEVERAGE_LIMITS:
        symbol_limits = SYMBOL_LEVERAGE_LIMITS[symbol]
    else:
        symbol_limits = SYMBOL_LEVERAGE_LIMITS["DEFAULT_ALTCOIN"]
    
    # 根据风险等级调整
    risk_multipliers = {
        "low": 1.0,
        "medium": 0.8,
        "high": 0.5,
        "critical": 0.2
    }
    
    base_limit = symbol_limits["max_leverage"]
    risk_multiplier = risk_multipliers.get(risk_level, 0.8)
    
    return min(base_limit * risk_multiplier, LeverageLimits.MAX_LEVERAGE_GLOBAL)


def get_margin_rate(symbol: str, leverage: float) -> float:
    """
    计算指定交易对和杠杆的初始保证金率
    
    Args:
        symbol: 交易对符号
        leverage: 杠杆倍数
        
    Returns:
        初始保证金率
    """
    if leverage <= 0:
        raise ValueError("杠杆倍数必须大于0")
    
    # 基础保证金率 = 1 / 杠杆
    base_rate = 1.0 / leverage
    
    # 根据交易对调整
    if symbol.startswith(("BTC", "ETH")):
        adjustment = 1.0
    else:
        adjustment = 1.2  # 山寨币需要更高保证金
    
    return base_rate * adjustment


def is_trading_hours() -> bool:
    """
    检查当前是否在交易时间内
    
    Returns:
        是否可以交易
    """
    from datetime import datetime
    
    current_hour = datetime.utcnow().hour
    
    # 检查是否在维护时间
    maintenance_start = TradingTimeConstants.MAINTENANCE_START_HOUR
    maintenance_end = maintenance_start + (TradingTimeConstants.MAINTENANCE_DURATION_MINUTES // 60)
    
    if maintenance_start <= current_hour < maintenance_end:
        return False
    
    # 期货市场24小时交易
    return True


# ============================================================================
# Binance SDK 集成配置
# ============================================================================

class BinanceSDKConfig:
    """币安官方SDK配置常量类"""
    
    # SDK认证类型
    AUTH_TYPE_API_SECRET = "api_secret"
    AUTH_TYPE_RSA_KEY = "rsa_key"
    AUTH_TYPE_ED25519_KEY = "ed25519_key"
    
    # SDK连接模式
    CONNECTION_MODE_SINGLE = "single"
    CONNECTION_MODE_POOL = "pool"
    
    # 默认SDK配置
    DEFAULT_REST_CONFIG = {
        "timeout": 5000,
        "retries": 3,
        "backoff": 1000,
        "compression": True,
        "keep_alive": True
    }
    
    DEFAULT_WEBSOCKET_CONFIG = {
        "compression": True,
        "keep_alive_interval": 30,
        "reconnect_delay": 5000,
        "max_reconnect_attempts": 5,
        "pool_size": 3
    }
    
    # 环境特定配置
    ENVIRONMENT_CONFIGS = {
        "testnet": {
            "rest_api_url": DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL,
            "ws_streams_url": DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_TESTNET_URL,
            "ws_api_url": DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_TESTNET_URL.replace('/ws', '/ws-api/v1'),
            "max_leverage": 20.0,
            "initial_balance": 10000.0,
            "sdk_config": {
                **DEFAULT_REST_CONFIG,
                "timeout": 3000,  # 测试网超时时间更短
                "retries": 2
            }
        },
        "mainnet": {
            "rest_api_url": DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
            "ws_streams_url": DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_PROD_URL,
            "ws_api_url": DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_PROD_URL.replace('/ws', '/ws-api/v1'),
            "max_leverage": 125.0,
            "sdk_config": {
                **DEFAULT_REST_CONFIG,
                "timeout": 10000,  # 主网超时时间更长
                "retries": 5
            }
        }
    }


class SDKErrorCodes:
    """SDK错误代码映射"""
    
    # SDK特有错误代码
    SDK_NOT_AVAILABLE = "SDK_NOT_AVAILABLE"
    SDK_AUTHENTICATION_ERROR = "SDK_AUTH_ERROR"
    SDK_CONFIGURATION_ERROR = "SDK_CONFIG_ERROR"
    SDK_RESPONSE_PARSE_ERROR = "SDK_PARSE_ERROR"
    
    # 与现有错误代码的映射
    ERROR_CODE_MAPPING = {
        "ClientError": ErrorCodes.API_ERROR,
        "UnauthorizedError": "UNAUTHORIZED_ERROR",
        "ForbiddenError": "FORBIDDEN_ERROR",
        "TooManyRequestsError": "RATE_LIMIT_ERROR",
        "ServerError": ErrorCodes.SYSTEM_ERROR,
        "NetworkError": ErrorCodes.NETWORK_ERROR,
        "NotFoundError": "NOT_FOUND_ERROR",
        "BadRequestError": "BAD_REQUEST_ERROR"
    }


# ============================================================================
# SDK配置助手函数
# ============================================================================

def get_sdk_environment_config(environment: str = "testnet") -> Dict[str, Any]:
    """
    获取指定环境的SDK配置
    
    Args:
        environment: 环境名称 ("testnet" 或 "mainnet")
        
    Returns:
        环境配置字典
    """
    if environment not in BinanceSDKConfig.ENVIRONMENT_CONFIGS:
        raise ValueError(f"不支持的环境: {environment}. 支持的环境: {list(BinanceSDKConfig.ENVIRONMENT_CONFIGS.keys())}")
    
    return BinanceSDKConfig.ENVIRONMENT_CONFIGS[environment].copy()


def create_sdk_rest_config(
    api_key: str,
    api_secret: str = None,
    private_key: str = None,
    private_key_passphrase: str = None,
    environment: str = "testnet",
    **kwargs
) -> Dict[str, Any]:
    """
    创建SDK REST API配置
    
    Args:
        api_key: API密钥
        api_secret: API密钥对应的秘密（可选，用于传统认证）
        private_key: 私钥内容（可选，用于RSA/ED25519认证）
        private_key_passphrase: 私钥密码（可选）
        environment: 环境名称
        **kwargs: 其他配置参数
        
    Returns:
        SDK配置字典
    """
    env_config = get_sdk_environment_config(environment)
    
    config = {
        "api_key": api_key,
        "base_path": env_config["rest_api_url"],
        **env_config["sdk_config"],
        **kwargs
    }
    
    # 添加认证信息
    if api_secret:
        config["api_secret"] = api_secret
    elif private_key:
        config["private_key"] = private_key
        if private_key_passphrase:
            config["private_key_passphrase"] = private_key_passphrase
    
    return config


def create_sdk_websocket_config(
    api_key: str = None,
    api_secret: str = None,
    private_key: str = None,
    private_key_passphrase: str = None,
    environment: str = "testnet",
    connection_type: str = "streams",  # "streams" 或 "api"
    **kwargs
) -> Dict[str, Any]:
    """
    创建SDK WebSocket配置
    
    Args:
        api_key: API密钥（WebSocket API需要）
        api_secret: API密钥对应的秘密
        private_key: 私钥内容
        private_key_passphrase: 私钥密码
        environment: 环境名称
        connection_type: 连接类型（"streams"用于数据流，"api"用于WebSocket API）
        **kwargs: 其他配置参数
        
    Returns:
        WebSocket配置字典
    """
    env_config = get_sdk_environment_config(environment)
    
    # 选择WebSocket URL
    if connection_type == "api":
        url_key = "ws_api_url"
    else:
        url_key = "ws_streams_url"
    
    config = {
        "stream_url": env_config[url_key],
        **BinanceSDKConfig.DEFAULT_WEBSOCKET_CONFIG,
        **kwargs
    }
    
    # 添加认证信息（仅WebSocket API需要）
    if connection_type == "api" and api_key:
        config["api_key"] = api_key
        if api_secret:
            config["api_secret"] = api_secret
        elif private_key:
            config["private_key"] = private_key
            if private_key_passphrase:
                config["private_key_passphrase"] = private_key_passphrase
    
    return config


def get_current_environment() -> str:
    """
    获取当前环境设置
    
    Returns:
        当前环境名称
    """
    import os
    return os.getenv("BINANCE_ENVIRONMENT", "testnet")


def is_testnet_environment() -> bool:
    """
    检查是否为测试网环境
    
    Returns:
        是否为测试网
    """
    return get_current_environment() == "testnet"


# ============================================================================
# 导出常量
# ============================================================================

# 更新DefaultConfig以包含SDK设置
class DefaultConfig:
    """默认配置聚合类（集成SDK配置）"""
    
    # 交易默认设置
    DEFAULT_TRADING_CONFIG = {
        "leverage": LeverageLimits.DEFAULT_LEVERAGE,
        "stop_loss_ratio": RiskParameters.DEFAULT_STOP_LOSS_RATIO,
        "take_profit_ratio": RiskParameters.DEFAULT_TAKE_PROFIT_RATIO,
        "max_position_ratio": RiskParameters.MAX_POSITION_RATIO,
        "margin_usage_limit": RiskParameters.MARGIN_USAGE_MAX,
        "risk_reward_ratio": RiskParameters.RECOMMENDED_RISK_REWARD_RATIO
    }
    
    # 风险管理默认设置  
    DEFAULT_RISK_CONFIG = {
        "max_drawdown": RiskParameters.MAX_DRAWDOWN_DAILY,
        "var_limit": RiskParameters.VAR_95_DAILY_LIMIT,
        "correlation_limit": RiskParameters.MAX_POSITION_CORRELATION,
        "liquidation_warning": RiskParameters.LIQUIDATION_WARNING_DISTANCE,
        "margin_warning": RiskParameters.MARGIN_USAGE_WARNING
    }
    
    # 系统默认设置
    DEFAULT_SYSTEM_CONFIG = {
        "environment": "testnet",
        "price_precision": PrecisionConstants.PRICE_PRECISION_MEDIUM,
        "quantity_precision": PrecisionConstants.QUANTITY_PRECISION_MEDIUM,
        "signal_expiry": TradingTimeConstants.SIGNAL_EXPIRY_MEDIUM_TERM,
        "update_interval": TradingTimeConstants.POSITION_UPDATE_INTERVAL
    }
    
    # SDK默认设置
    DEFAULT_SDK_CONFIG = {
        "environment": "testnet",
        "timeout": BinanceSDKConfig.DEFAULT_REST_CONFIG["timeout"],
        "retries": BinanceSDKConfig.DEFAULT_REST_CONFIG["retries"],
        "compression": BinanceSDKConfig.DEFAULT_REST_CONFIG["compression"],
        "keep_alive": BinanceSDKConfig.DEFAULT_REST_CONFIG["keep_alive"],
        "websocket": BinanceSDKConfig.DEFAULT_WEBSOCKET_CONFIG
    }


__all__ = [
    # 主要类
    'LeverageLimits',
    'MarginRates', 
    'RiskParameters',
    'TradingPairConfig',
    'EnvironmentConfig',
    'TradingTimeConstants',
    'FeeConstants',
    'PrecisionConstants',
    'ErrorCodes',
    'DefaultConfig',
    
    # 新增SDK相关类
    'BinanceSDKConfig',
    'SDKErrorCodes',
    
    # 配置字典
    'SYMBOL_LEVERAGE_LIMITS',
    
    # 原有辅助函数
    'get_symbol_config',
    'get_leverage_limit', 
    'get_margin_rate',
    'is_trading_hours',
    
    # 新增SDK辅助函数
    'get_sdk_environment_config',
    'create_sdk_rest_config',
    'create_sdk_websocket_config',
    'get_current_environment',
    'is_testnet_environment',
    
    # SDK常量
    'DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL',
    'DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL',
    'DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_PROD_URL',
    'DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_TESTNET_URL'
]
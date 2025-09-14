"""
期货交易系统模块

提供完整的期货交易功能，包括数据模型、常量定义、
配置管理、接口定义等。
"""

# 从子模块导入主要组件
from . import constants
from . import models
# from . import config  # 暂时注释，避免导入错误
from . import interfaces

# 导入常用的类和函数
from .models import (
    TradingDirection,
    OperationType,
    PositionSide,
    RiskLevel,
    FuturesSignal,
    Position,
    MarginStatus,
    ValidationResult,
    FuturesOrderRequest
)

from .constants import (
    LeverageLimits,
    MarginRates,
    RiskParameters,
    TradingPairConfig,
    EnvironmentConfig,
    get_symbol_config,
    get_leverage_limit,
    get_margin_rate
)

__version__ = "1.0.0"

__all__ = [
    # 子模块
    "constants",
    "models", 
    # "config",  # 暂时注释
    "interfaces",
    
    # 枚举类型
    "TradingDirection",
    "OperationType",
    "PositionSide", 
    "RiskLevel",
    
    # 数据类
    "FuturesSignal",
    "Position",
    "MarginStatus",
    "ValidationResult",
    "FuturesOrderRequest",
    
    # 常量类
    "LeverageLimits",
    "MarginRates",
    "RiskParameters",
    "TradingPairConfig",
    "EnvironmentConfig",
    
    # 辅助函数
    "get_symbol_config",
    "get_leverage_limit",
    "get_margin_rate"
]
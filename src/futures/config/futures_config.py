"""
期货交易配置模块

提供期货交易系统的配置管理功能
"""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """交易模式枚举"""
    TESTNET = "testnet"
    MAINNET = "mainnet"
    PAPER = "paper"


@dataclass
class FuturesConfig:
    """期货交易配置类"""

    # 基础配置
    trading_mode: TradingMode = TradingMode.TESTNET
    api_key: Optional[str] = None
    api_secret: Optional[str] = None

    # 风险配置
    max_leverage: float = 20.0
    max_position_ratio: float = 0.3
    margin_usage_limit: float = 0.8

    # 交易配置
    default_leverage: float = 5.0
    min_position_size: float = 10.0
    max_position_size: float = 10000.0

    # 信号配置
    signal_expiry_minutes: int = 30
    min_confidence: float = 60.0

    def __post_init__(self):
        """配置验证"""
        if self.max_leverage > 125.0:
            logger.warning(f"最大杠杆{self.max_leverage}超过币安限制，调整为125.0")
            self.max_leverage = 125.0

        if self.trading_mode == TradingMode.TESTNET and self.max_leverage > 20.0:
            logger.info(f"测试网环境，杠杆限制为20.0")
            self.max_leverage = 20.0

    def get_config_dict(self) -> Dict[str, Any]:
        """获取配置字典"""
        return {
            "trading_mode": self.trading_mode.value,
            "max_leverage": self.max_leverage,
            "max_position_ratio": self.max_position_ratio,
            "margin_usage_limit": self.margin_usage_limit,
            "default_leverage": self.default_leverage,
            "min_position_size": self.min_position_size,
            "max_position_size": self.max_position_size,
            "signal_expiry_minutes": self.signal_expiry_minutes,
            "min_confidence": self.min_confidence
        }

    @classmethod
    def create_default_config(cls) -> 'FuturesConfig':
        """创建默认配置"""
        return cls()

    @classmethod
    def create_testnet_config(cls) -> 'FuturesConfig':
        """创建测试网配置"""
        return cls(
            trading_mode=TradingMode.TESTNET,
            max_leverage=20.0,
            default_leverage=3.0,
            max_position_ratio=0.2,
            margin_usage_limit=0.7
        )

    @classmethod
    def create_mainnet_config(cls) -> 'FuturesConfig':
        """创建主网配置"""
        return cls(
            trading_mode=TradingMode.MAINNET,
            max_leverage=125.0,
            default_leverage=10.0,
            max_position_ratio=0.3,
            margin_usage_limit=0.8
        )
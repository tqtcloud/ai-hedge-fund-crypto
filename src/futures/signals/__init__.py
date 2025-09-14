"""
期货信号系统模块

提供完整的期货交易信号生成系统，包括：
- 统一的策略接口
- 策略工厂模式
- 策略管理器
- 多时间框架信号聚合
- 杠杆和风险参数整合
- 完整的期货信号系统
"""

from .base_strategy import FuturesBaseStrategy
from .strategy_factory import FuturesStrategyFactory
from .strategy_manager import FuturesStrategyManager
from .futures_signal_system import FuturesSignalSystem, SignalConfidenceMetrics, PositionOperation

__all__ = [
    'FuturesBaseStrategy',
    'FuturesStrategyFactory',
    'FuturesStrategyManager',
    'FuturesSignalSystem',
    'SignalConfidenceMetrics',
    'PositionOperation'
]
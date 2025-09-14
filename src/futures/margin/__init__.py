"""
期货保证金管理模块

该模块提供完整的期货保证金管理功能，包括：
- 保证金计算和验证
- 强平价格计算
- 实时账户数据监控
- 风险评估和预警
- 保证金健康状态监控
"""

from .margin_manager import (
    MarginManager,
    MarginConfig,
    WebSocketTradingInterface
)

__all__ = [
    "MarginManager",
    "MarginConfig",
    "WebSocketTradingInterface"
]
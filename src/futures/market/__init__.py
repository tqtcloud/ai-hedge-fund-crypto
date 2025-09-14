"""
期货市场分析模块

提供全面的市场分析功能，包括波动率分析、技术指标集成、
市场条件分析、相关性分析和风险度量等核心功能。
"""

from .market_analyzer import (
    MarketAnalyzer,
    VolatilityMetrics,
    TechnicalIndicators,
    MarketConditionAnalysis,
    RiskMetrics,
    MarketCondition,
    VolatilityRegime,
    LiquidityLevel
)

__all__ = [
    'MarketAnalyzer',
    'VolatilityMetrics',
    'TechnicalIndicators', 
    'MarketConditionAnalysis',
    'RiskMetrics',
    'MarketCondition',
    'VolatilityRegime',
    'LiquidityLevel'
]

__version__ = '1.0.0'
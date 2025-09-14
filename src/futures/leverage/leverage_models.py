"""
杠杆控制核心数据模型

定义杠杆控制系统中使用的数据结构，包括杠杆配置、
计算结果、调整因子和限制条件等。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Literal
from datetime import datetime
from enum import Enum
from decimal import Decimal

from ..models.data_models import RiskLevel


class LeverageStrategy(Enum):
    """杠杆策略枚举"""
    CONSERVATIVE = "conservative"      # 保守策略
    MODERATE = "moderate"              # 温和策略
    AGGRESSIVE = "aggressive"          # 激进策略
    EXPERT = "expert"                  # 专家策略


class MarketRegime(Enum):
    """市场状态枚举"""
    CALM = "calm"                      # 平静市场
    VOLATILE = "volatile"              # 波动市场
    TRENDING = "trending"              # 趋势市场
    RANGING = "ranging"                # 震荡市场
    HIGH_VOLATILITY = "high_volatility"  # 高波动市场


class LiquidityLevel(Enum):
    """流动性等级枚举"""
    HIGH = "high"                      # 高流动性
    MEDIUM = "medium"                  # 中等流动性
    LOW = "low"                        # 低流动性
    VERY_LOW = "very_low"              # 极低流动性


@dataclass
class MarketCondition:
    """
    市场状况类

    包含影响杠杆计算的市场因子信息
    """
    ticker: str                                    # 交易对
    volatility: float                              # 波动率 (0-1)
    liquidity_score: float                         # 流动性评分 (0-1)
    market_regime: MarketRegime                    # 市场状态
    liquidity_level: LiquidityLevel                # 流动性等级

    # 技术指标
    atr_percentage: Optional[float] = None         # ATR百分比
    volume_ratio: Optional[float] = None           # 成交量比率
    spread_bps: Optional[float] = None             # 点差基点

    # 时间信息
    update_time: datetime = field(default_factory=datetime.now)
    data_age_seconds: float = 0.0                  # 数据年龄（秒）

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_data_fresh(self, max_age_seconds: float = 60.0) -> bool:
        """检查数据是否新鲜"""
        return self.data_age_seconds <= max_age_seconds

    def get_volatility_multiplier(self) -> float:
        """获取基于波动率的杠杆倍数调整因子"""
        if self.volatility <= 0.1:  # 低波动
            return 1.2
        elif self.volatility <= 0.2:  # 中等波动
            return 1.0
        elif self.volatility <= 0.3:  # 高波动
            return 0.8
        else:  # 极高波动
            return 0.6

    def get_liquidity_multiplier(self) -> float:
        """获取基于流动性的杠杆倍数调整因子"""
        if self.liquidity_level == LiquidityLevel.HIGH:
            return 1.1
        elif self.liquidity_level == LiquidityLevel.MEDIUM:
            return 1.0
        elif self.liquidity_level == LiquidityLevel.LOW:
            return 0.8
        else:  # VERY_LOW
            return 0.6

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "ticker": self.ticker,
            "volatility": self.volatility,
            "liquidity_score": self.liquidity_score,
            "market_regime": self.market_regime.value,
            "liquidity_level": self.liquidity_level.value,
            "atr_percentage": self.atr_percentage,
            "volume_ratio": self.volume_ratio,
            "spread_bps": self.spread_bps,
            "update_time": self.update_time.isoformat(),
            "data_age_seconds": self.data_age_seconds,
            "metadata": self.metadata
        }


@dataclass
class LeverageLimits:
    """
    杠杆限制类

    定义不同条件下的杠杆限制
    """
    ticker: str                                    # 交易对
    max_leverage: float                            # 最大杠杆
    min_leverage: float = 1.0                      # 最小杠杆

    # 基于风险等级的限制
    risk_based_limits: Dict[RiskLevel, float] = field(default_factory=dict)

    # 基于策略的限制
    strategy_based_limits: Dict[LeverageStrategy, float] = field(default_factory=dict)

    # 基于市场状态的限制
    market_regime_limits: Dict[MarketRegime, float] = field(default_factory=dict)

    # 动态调整参数
    volatility_threshold: float = 0.3              # 波动率阈值
    liquidity_threshold: float = 0.5               # 流动性阈值
    emergency_reduction_factor: float = 0.5        # 紧急降低因子

    def __post_init__(self):
        """初始化后设置默认限制"""
        self._set_default_limits()

    def _set_default_limits(self):
        """设置默认限制"""
        # 默认风险等级限制
        if not self.risk_based_limits:
            self.risk_based_limits = {
                RiskLevel.LOW: self.max_leverage,
                RiskLevel.MEDIUM: self.max_leverage * 0.8,
                RiskLevel.HIGH: self.max_leverage * 0.6,
                RiskLevel.CRITICAL: self.max_leverage * 0.4,
                RiskLevel.EMERGENCY: self.max_leverage * 0.2
            }

        # 默认策略限制
        if not self.strategy_based_limits:
            self.strategy_based_limits = {
                LeverageStrategy.CONSERVATIVE: min(5.0, self.max_leverage * 0.5),
                LeverageStrategy.MODERATE: min(10.0, self.max_leverage * 0.7),
                LeverageStrategy.AGGRESSIVE: min(20.0, self.max_leverage * 0.9),
                LeverageStrategy.EXPERT: self.max_leverage
            }

        # 默认市场状态限制
        if not self.market_regime_limits:
            self.market_regime_limits = {
                MarketRegime.CALM: self.max_leverage,
                MarketRegime.VOLATILE: self.max_leverage * 0.8,
                MarketRegime.TRENDING: self.max_leverage * 0.9,
                MarketRegime.RANGING: self.max_leverage * 0.7,
                MarketRegime.HIGH_VOLATILITY: self.max_leverage * 0.5
            }

    def get_effective_limit(
        self,
        risk_level: RiskLevel,
        strategy: LeverageStrategy,
        market_regime: MarketRegime
    ) -> float:
        """获取有效杠杆限制（取最严格的限制）"""
        limits = [
            self.risk_based_limits.get(risk_level, self.max_leverage),
            self.strategy_based_limits.get(strategy, self.max_leverage),
            self.market_regime_limits.get(market_regime, self.max_leverage)
        ]
        return min(limits)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "ticker": self.ticker,
            "max_leverage": self.max_leverage,
            "min_leverage": self.min_leverage,
            "risk_based_limits": {k.value: v for k, v in self.risk_based_limits.items()},
            "strategy_based_limits": {k.value: v for k, v in self.strategy_based_limits.items()},
            "market_regime_limits": {k.value: v for k, v in self.market_regime_limits.items()},
            "volatility_threshold": self.volatility_threshold,
            "liquidity_threshold": self.liquidity_threshold,
            "emergency_reduction_factor": self.emergency_reduction_factor
        }


@dataclass
class LeverageAdjustmentFactor:
    """
    杠杆调整因子类

    包含影响杠杆计算的各种调整因子
    """
    volatility_multiplier: float = 1.0             # 波动率调整倍数
    liquidity_multiplier: float = 1.0              # 流动性调整倍数
    market_adjustment: float = 1.0                 # 市场状态调整
    position_adjustment: float = 1.0               # 仓位调整
    risk_adjustment: float = 1.0                   # 风险调整
    time_adjustment: float = 1.0                   # 时间调整

    # 特殊调整因子
    news_impact_factor: float = 1.0                # 新闻影响因子
    correlation_factor: float = 1.0                # 相关性因子
    momentum_factor: float = 1.0                   # 动量因子

    # 计算信息
    calculation_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_combined_multiplier(self) -> float:
        """获取综合调整倍数"""
        return (
            self.volatility_multiplier *
            self.liquidity_multiplier *
            self.market_adjustment *
            self.position_adjustment *
            self.risk_adjustment *
            self.time_adjustment *
            self.news_impact_factor *
            self.correlation_factor *
            self.momentum_factor
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "volatility_multiplier": self.volatility_multiplier,
            "liquidity_multiplier": self.liquidity_multiplier,
            "market_adjustment": self.market_adjustment,
            "position_adjustment": self.position_adjustment,
            "risk_adjustment": self.risk_adjustment,
            "time_adjustment": self.time_adjustment,
            "news_impact_factor": self.news_impact_factor,
            "correlation_factor": self.correlation_factor,
            "momentum_factor": self.momentum_factor,
            "combined_multiplier": self.get_combined_multiplier(),
            "calculation_time": self.calculation_time.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class LeverageConfig:
    """
    杠杆配置类

    定义杠杆控制系统的配置参数
    """
    # 基础杠杆设置
    default_leverage: float = 5.0                  # 默认杠杆
    base_leverage_by_strategy: Dict[LeverageStrategy, float] = field(default_factory=dict)

    # 交易对特定设置
    symbol_specific_limits: Dict[str, LeverageLimits] = field(default_factory=dict)

    # 调整参数
    max_adjustment_percentage: float = 50.0        # 最大调整百分比
    min_adjustment_percentage: float = 10.0        # 最小调整百分比

    # 风险控制参数
    emergency_leverage_cap: float = 2.0            # 紧急情况杠杆上限
    high_volatility_threshold: float = 0.4         # 高波动阈值
    low_liquidity_threshold: float = 0.3           # 低流动性阈值

    # 时间窗口设置
    volatility_window_minutes: int = 60            # 波动率计算窗口（分钟）
    liquidity_window_minutes: int = 30             # 流动性计算窗口（分钟）
    position_check_interval_seconds: int = 30     # 仓位检查间隔（秒）

    # 安全设置
    enable_emergency_reduction: bool = True        # 启用紧急降杠杆
    enable_position_size_limits: bool = True      # 启用仓位大小限制
    enable_correlation_checks: bool = True        # 启用相关性检查

    def __post_init__(self):
        """初始化后设置默认值"""
        self._set_default_base_leverage()

    def _set_default_base_leverage(self):
        """设置默认基础杠杆"""
        if not self.base_leverage_by_strategy:
            self.base_leverage_by_strategy = {
                LeverageStrategy.CONSERVATIVE: 3.0,
                LeverageStrategy.MODERATE: 5.0,
                LeverageStrategy.AGGRESSIVE: 8.0,
                LeverageStrategy.EXPERT: 12.0
            }

    def get_symbol_limits(self, ticker: str) -> Optional[LeverageLimits]:
        """获取交易对特定限制"""
        return self.symbol_specific_limits.get(ticker)

    def add_symbol_limits(self, ticker: str, limits: LeverageLimits):
        """添加交易对特定限制"""
        self.symbol_specific_limits[ticker] = limits

    def get_base_leverage(self, strategy: LeverageStrategy) -> float:
        """获取策略对应的基础杠杆"""
        return self.base_leverage_by_strategy.get(strategy, self.default_leverage)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "default_leverage": self.default_leverage,
            "base_leverage_by_strategy": {k.value: v for k, v in self.base_leverage_by_strategy.items()},
            "symbol_specific_limits": {k: v.to_dict() for k, v in self.symbol_specific_limits.items()},
            "max_adjustment_percentage": self.max_adjustment_percentage,
            "min_adjustment_percentage": self.min_adjustment_percentage,
            "emergency_leverage_cap": self.emergency_leverage_cap,
            "high_volatility_threshold": self.high_volatility_threshold,
            "low_liquidity_threshold": self.low_liquidity_threshold,
            "volatility_window_minutes": self.volatility_window_minutes,
            "liquidity_window_minutes": self.liquidity_window_minutes,
            "position_check_interval_seconds": self.position_check_interval_seconds,
            "enable_emergency_reduction": self.enable_emergency_reduction,
            "enable_position_size_limits": self.enable_position_size_limits,
            "enable_correlation_checks": self.enable_correlation_checks
        }


@dataclass
class LeverageCalculationResult:
    """
    杠杆计算结果类

    包含杠杆计算的完整结果和相关信息
    """
    ticker: str                                    # 交易对
    calculated_leverage: float                     # 计算得出的杠杆
    applied_leverage: float                        # 实际应用的杠杆

    # 输入参数
    base_leverage: float                           # 基础杠杆
    requested_leverage: Optional[float] = None     # 请求杠杆

    # 调整因子
    adjustment_factors: LeverageAdjustmentFactor = field(default_factory=LeverageAdjustmentFactor)

    # 限制信息
    limits: Optional[LeverageLimits] = None        # 应用的限制
    effective_limit: float = 100.0                 # 有效限制

    # 计算过程
    calculation_steps: List[str] = field(default_factory=list)  # 计算步骤
    warnings: List[str] = field(default_factory=list)          # 警告信息
    errors: List[str] = field(default_factory=list)            # 错误信息

    # 市场状况
    market_condition: Optional[MarketCondition] = None

    # 风险评估
    risk_level: RiskLevel = RiskLevel.LOW          # 风险等级
    confidence_score: float = 1.0                  # 计算置信度 (0-1)

    # 建议信息
    recommendation: Optional[str] = None           # 杠杆建议
    suggested_position_size: Optional[float] = None  # 建议仓位大小

    # 时间信息
    calculation_time: datetime = field(default_factory=datetime.now)
    valid_until: Optional[datetime] = None         # 结果有效期

    # 元数据
    calculation_method: str = "dynamic"            # 计算方法
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_adjustment_significant(self, threshold: float = 0.1) -> bool:
        """判断杠杆调整是否显著"""
        if self.requested_leverage is None:
            return abs(self.applied_leverage - self.base_leverage) / self.base_leverage > threshold
        return abs(self.applied_leverage - self.requested_leverage) / self.requested_leverage > threshold

    def get_adjustment_percentage(self) -> float:
        """获取杠杆调整百分比"""
        base = self.requested_leverage or self.base_leverage
        return ((self.applied_leverage - base) / base) * 100

    def has_warnings(self) -> bool:
        """检查是否有警告"""
        return len(self.warnings) > 0

    def has_errors(self) -> bool:
        """检查是否有错误"""
        return len(self.errors) > 0

    def is_valid(self) -> bool:
        """检查结果是否有效（无错误且未过期）"""
        if self.has_errors():
            return False

        if self.valid_until and datetime.now() > self.valid_until:
            return False

        return True

    def add_calculation_step(self, step: str):
        """添加计算步骤"""
        self.calculation_steps.append(f"[{datetime.now().strftime('%H:%M:%S')}] {step}")

    def add_warning(self, warning: str):
        """添加警告"""
        self.warnings.append(f"[{datetime.now().strftime('%H:%M:%S')}] {warning}")

    def add_error(self, error: str):
        """添加错误"""
        self.errors.append(f"[{datetime.now().strftime('%H:%M:%S')}] {error}")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "ticker": self.ticker,
            "calculated_leverage": self.calculated_leverage,
            "applied_leverage": self.applied_leverage,
            "base_leverage": self.base_leverage,
            "requested_leverage": self.requested_leverage,
            "adjustment_factors": self.adjustment_factors.to_dict(),
            "limits": self.limits.to_dict() if self.limits else None,
            "effective_limit": self.effective_limit,
            "calculation_steps": self.calculation_steps,
            "warnings": self.warnings,
            "errors": self.errors,
            "market_condition": self.market_condition.to_dict() if self.market_condition else None,
            "risk_level": self.risk_level.value,
            "confidence_score": self.confidence_score,
            "recommendation": self.recommendation,
            "suggested_position_size": self.suggested_position_size,
            "calculation_time": self.calculation_time.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "calculation_method": self.calculation_method,
            "adjustment_percentage": self.get_adjustment_percentage(),
            "is_significant_adjustment": self.is_adjustment_significant(),
            "is_valid": self.is_valid(),
            "metadata": self.metadata
        }


# 导出所有数据模型
__all__ = [
    'LeverageStrategy',
    'MarketRegime',
    'LiquidityLevel',
    'MarketCondition',
    'LeverageLimits',
    'LeverageAdjustmentFactor',
    'LeverageConfig',
    'LeverageCalculationResult'
]